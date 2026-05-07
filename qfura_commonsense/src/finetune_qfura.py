import sys
import os

# qfura_commonsense/src/ — for `utils.*`
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
# FuRA repo root — for `fura.*`
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)
# qfura_commonsense/ — for `tools.system_metrics`
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

import copy
import time
import torch
import json
import random
import math
import argparse
from tqdm.auto import tqdm

from torch.utils.data import DataLoader
import torch.nn as nn

import bitsandbytes as bnb

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    SchedulerType,
    get_scheduler,
)

from utils.utils import (
    print_rank_0,
    get_all_reduce_mean,
    int_or_float,
)

from accelerate import Accelerator
from accelerate.utils import set_seed

from utils.model_utils import (
    load_hf_tokenizer,
    save_hf_format,
    make_model_gradient_checkpointing_compatible,
)

from utils.data_utils import SupervisedDataset, DataCollatorForSupervisedDataset

from fura.btt_layer import (
    BTTLayer,
    convert_linear_to_btt,
    convert_btt_to_qbtt_,
    configure_blocktt_trainability,
    get_blocktt_target_module_names,
    normalize_trainable_blocktt_cores_,
    resolve_blocktt_decomp_modes,
)

from tools.system_metrics import SysMon

# Calibrated-BTT activations require the `compress` library. The default QFuRA
# recipe runs with --calib_mode none, so stub it out unless the user opts in.
try:
    from compress_integration import (  # type: ignore
        add_calibrated_btt_args,
        validate_calibrated_btt_args,
        apply_calibrated_btt,
        build_calib_loader,
        save_calibrated_btt_checkpoint,
    )
except ImportError:
    def add_calibrated_btt_args(parser, *, hyphen_style: bool = False):
        prefix = "--calib-" if hyphen_style else "--calib_"
        parser.add_argument(prefix + "mode", type=str, default="none",
                            choices=["none", "v2_bp"])
        parser.add_argument(prefix + ("source" if hyphen_style else "source"),
                            type=str, default="c4")
        parser.add_argument(prefix + ("traces-path" if hyphen_style else "traces_path"),
                            type=str, default=None)
        parser.add_argument(prefix + ("num-seqs" if hyphen_style else "num_seqs"),
                            type=int, default=128)
        parser.add_argument(prefix + ("max-length" if hyphen_style else "max_length"),
                            type=int, default=2048)
        parser.add_argument(prefix + ("seed" if hyphen_style else "seed"),
                            type=int, default=3)
        parser.add_argument(prefix + ("batch-size" if hyphen_style else "batch_size"),
                            type=int, default=8)

    def validate_calibrated_btt_args(args, *, argv=None, hyphen_style=False):
        if getattr(args, "calib_mode", "none") not in (None, "none"):
            raise RuntimeError(
                "--calib_mode != 'none' requires the optional `compress` library, "
                "which is not bundled with this open-source release."
            )

    def apply_calibrated_btt(*args, **kwargs):
        raise RuntimeError("Calibrated BTT not available in open-source build.")

    def build_calib_loader(*args, **kwargs):
        raise RuntimeError("Calibrated BTT not available in open-source build.")

    def save_calibrated_btt_checkpoint(*args, **kwargs):
        raise RuntimeError("Calibrated BTT not available in open-source build.")


def resolve_blocktt_rank(rank_arg):
    """Parse --blocktt_rank: 'full' or a positive integer."""
    if rank_arg == "full":
        return "full"
    try:
        rank = int(rank_arg)
    except ValueError as exc:
        raise ValueError("--blocktt_rank must be 'full' or a positive integer") from exc
    if rank <= 0:
        raise ValueError("--blocktt_rank must be > 0")
    return rank


def materialize_btt_to_linear(model, offload_device=None):
    """Replace all BTTLayer modules with nn.Linear containing materialized dense weights.

    This makes the model saveable/loadable as a standard HF checkpoint.
    QBTTLayer inherits from BTTLayer and its materialize_dense_weight() handles
    dequantization internally, so this function works for both BTTLayer and QBTTLayer.

    Args:
      offload_device: if not None (e.g. torch.device("cpu")), each materialized
        nn.Linear is moved here right after construction, AND every other
        nn.Module already on GPU is offloaded once the BTT modules are dealt
        with. This bounds the GPU peak by one bf16 layer rather than the full
        materialised model. Required for >=30B base models on a single H100,
        where the full bf16 model is bigger than GPU memory. The original
        BTT/QBTT modules are also moved off-GPU before being discarded so
        their NF4 buffers don't linger.
    """
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, BTTLayer):
            replacements.append((name, module))

    for name, btt_module in replacements:
        dense_weight = btt_module.materialize_dense_weight()
        target_device = offload_device if offload_device is not None else dense_weight.device
        linear = nn.Linear(
            btt_module.in_features,
            btt_module.out_features,
            bias=btt_module.bias is not None,
            device=target_device,
            dtype=dense_weight.dtype,
        )
        # If offloading: stage the dense bf16 tensor through host RAM so the
        # GPU memory occupied by the dequantised QBTT weight + the new linear
        # is freed in the same loop iteration.
        if offload_device is not None and dense_weight.device.type != offload_device.type:
            linear.weight.data.copy_(dense_weight.to(offload_device))
        else:
            linear.weight.data.copy_(dense_weight)
        if btt_module.bias is not None:
            bias = btt_module.bias.data
            if offload_device is not None and bias.device.type != offload_device.type:
                bias = bias.to(offload_device)
            linear.bias.data.copy_(bias)

        # Navigate to parent and replace the child
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], linear)

        # Drop our reference to the dequantised buffer immediately. The old
        # btt_module's NF4 buffers are still alive on GPU until garbage
        # collection because the `replacements` list holds a strong ref; we
        # explicitly clear it after the loop.
        del dense_weight
        if offload_device is not None and offload_device.type == "cpu":
            torch.cuda.empty_cache()

    # Drop the last batch of strong refs (the BTT modules), then nuke the
    # GPU pool so the freed NF4 + dequant buffers actually become available.
    n_replaced = len(replacements)
    replacements.clear()
    if offload_device is not None:
        # Also offload any non-BTT modules still on GPU (embed_tokens,
        # layernorms, lm_head). This puts the entire materialised model on
        # the host before save_pretrained walks it.
        model.to(offload_device)
        torch.cuda.empty_cache()

    print(f"Materialized {n_replaced} BTTLayer modules to nn.Linear" + (" (offloaded to CPU)" if offload_device is not None else ""))
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="QFura Fine-Tuning (LIFT benchmark) — BlockTT with NF4-quantized frozen core + PagedAdamW8bit")
    parser.add_argument(
        "--data_path",
        nargs="*",
        default=["./LLM-Adapters/ft-training_set/commonsense_170k.json"],
        help="Path to the training dataset (json).",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--per_device_train_batch_size", type=int, default=16,
        help="Batch size (per device) for training.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size", type=int, default=16,
        help="Batch size (per device) for evaluation.",
    )
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--val_set_size", type=int, default=100,
        help="Size of the validation set. If 0, no validation set is used.")
    parser.add_argument("--load_last_model", action="store_true",
        help="Skip best-model tracking, save only the last model.")
    parser.add_argument("--eval_step", type=int, default=80)
    parser.add_argument("--eval_delay", type=int_or_float, default=0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="If > 0, cap total optimizer steps at this value (for short-horizon system-eval runs).",
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument(
        "--lr_scheduler_type", type=SchedulerType, default="linear",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial",
                 "constant", "constant_with_warmup"],
    )
    parser.add_argument("--num_warmup_steps", type=float, default=0.03)
    parser.add_argument(
        "--mixed_precision", type=str, default="bf16",
        choices=["fp16", "bf16", "fp32"],
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument(
        "--instruction_type", type=str, choices=["single", "multi"], default="single",
    )
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument(
        "--use_flash_attn", type=str, default="False",
    )
    parser.add_argument(
        "--prompt_style",
        type=str,
        choices=["lift", "pissa"],
        default="lift",
        help=(
            "Which training prompt + target format to use. "
            "'lift' (default): LIFT BASE_PROMPT (literal '<s>', trailing whitespace, "
            "newline before response, target '<output> <eos>'). "
            "'pissa': PiSSA's clean format (no literal <s>, no trailing whitespace, "
            "target '<output>\\n<eos>'). Use 'pissa' to align with the published "
            "QPiSSA recipe on MetaMathQA."
        ),
    )
    parser.add_argument(
        "--trainable_param_dtype",
        type=str,
        choices=["bf16", "fp32"],
        default="bf16",
        help=(
            "After streaming/direct conversion, upcast trainable BTT params "
            "(btt_l/r/s + bias) to this dtype. fp32 matches the QPiSSA paper "
            "recipe; bf16 is the default qfura recipe (faster, less memory)."
        ),
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["paged_adamw_8bit", "adamw"],
        default="paged_adamw_8bit",
        help=(
            "'paged_adamw_8bit' (default qfura): bnb 8-bit moments, paged for "
            "OOM safety. 'adamw': torch.optim.AdamW with fp32 moments — matches "
            "QPiSSA paper. Only meaningful at fp32 trainable_param_dtype."
        ),
    )
    parser.add_argument(
        "--load_strategy",
        type=str,
        choices=["direct", "layer_stream"],
        default="direct",
        help=(
            "How to materialise the bf16 base into a qfura model on GPU. "
            "'direct': load full bf16 model on GPU, then BTT-decompose, then "
            "NF4-quantise. Fits up to ~13B on a 94 GB H100. "
            "'layer_stream': load full bf16 model on CPU, then walk every "
            "target Linear, stage its weight on GPU, BTT-decompose + NF4-"
            "quantise it, replace, and free. Peak GPU memory is bounded by "
            "the largest single Linear (~470 MB for Llama-3-70B's down_proj). "
            "Required for ≥30B models on a single H100."
        ),
    )

    # BlockTT-specific arguments
    parser.add_argument("--trainable_type", type=str, default="all",
        choices=["all", "mlp", "attn"],
        help="Which modules to convert to BTT: all, mlp, attn")
    parser.add_argument("--decomp_mode", type=str, default="output_one_block",
        help="BTT decomposition mode: input_one_block, output_one_block, or dict literal")
    parser.add_argument("--blocktt_rank", type=str, default="full",
        help="BTT rank: 'full' for lossless or a positive integer")
    parser.add_argument("--train_position", type=str, default="small",
        choices=["small", "large", "both"],
        help="Which TT core to train: small, large, both")
    parser.add_argument(
        "--s_merged_to",
        type=str,
        default="keep_trainable",
        choices=[
            "frozen",
            "trainable",
            "output",
            "input",
            "split",
            "keep_frozen",
            "keep_trainable",
        ],
        help=(
            "Where to merge/keep singular values during SVD init: "
            "frozen, trainable, output, input, split, keep_frozen, keep_trainable"
        ),
    )
    parser.add_argument("--blocktt_normalize_after_update", action="store_true",
        help="Normalize trainable BTT cores after each optimizer step")
    parser.add_argument("--blocktt_factorize_by_head", action="store_true", default=True,
        help="Align attention BTT blocks with head structure")
    parser.add_argument("--no_blocktt_factorize_by_head", action="store_false",
        dest="blocktt_factorize_by_head")
    parser.add_argument(
        "--blocktt_input_factorization",
        type=str,
        default=None,
        help=(
            "Override input-side (n,b) factorization. Accepts: 'head' / 'closest' / "
            "'n,b' as a scalar applied to all modules, or a JSON/Python dict literal "
            "mapping group names (qkv, o, mlp_upgate, mlp_down) or leaf names "
            "(q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj) to "
            "individual specs. When set, this overrides --blocktt_factorize_by_head "
            "for any module covered by the spec."
        ),
    )
    parser.add_argument("--no_train_bias", action="store_true",
        help="Freeze BTT biases")

    # QFura-specific arguments
    parser.add_argument(
        "--quant_block_layout",
        type=str,
        default="flat",
        choices=["flat", "per_core_block"],
        help="Block layout for NF4 quantization of the frozen BTT core: flat or per_core_block",
    )

    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Weights & Biases run name.",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable Weights & Biases logging.",
    )

    add_calibrated_btt_args(parser, hyphen_style=False)

    args = parser.parse_args()
    validate_calibrated_btt_args(args, argv=sys.argv[1:], hyphen_style=False)

    # Hard constraints for qfura
    if args.train_position != "small":
        raise ValueError(
            f"finetune_qfura.py requires --train_position=small (got {args.train_position!r}). "
            "The NF4 quantization step freezes one core and quantizes it; only 'small' is "
            "supported because qfura quantizes the large (frozen) core."
        )
    if not args.gradient_checkpointing:
        raise ValueError(
            "finetune_qfura.py requires --gradient_checkpointing. "
            "The NF4-quantized frozen core increases activation memory; "
            "gradient checkpointing is mandatory to keep peak memory manageable."
        )

    return args


def main():
    args = parse_args()

    use_wandb = not args.no_wandb
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb" if use_wandb else None,
    )
    if not torch.cuda.is_available() or accelerator.device.type != "cuda":
        raise RuntimeError(
            "finetune_qfura.py requires CUDA so BlockTT conversion SVD runs on GPU. "
            f"Current accelerator device: {accelerator.device}."
        )

    set_seed(args.seed)
    args.global_rank = 1

    if use_wandb:
        if args.wandb_project is None:
            args.wandb_project = "qfura"
        tracker_config = vars(args).copy()
        wandb_init_kwargs = {}
        if args.wandb_run_name:
            wandb_init_kwargs["name"] = args.wandb_run_name
        accelerator.init_trackers(
            project_name=args.wandb_project,
            config=tracker_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )

    # Load tokenizer
    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    tokenizer.model_max_length = args.max_seq_len

    # Load model.
    # `load_strategy=direct` (default): load on accelerator.device in bf16. The
    # full bf16 model + intermediate decomposition state must fit; OK up to ~13B
    # on a 94 GB H100.
    # `load_strategy=layer_stream`: load on CPU in bf16 (we have plenty of RAM),
    # then BTT-decompose + NF4-quantise each target Linear via a brief CUDA
    # staging round-trip. Peak GPU memory bounded by the largest single Linear.
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model_kwargs = {"torch_dtype": torch.bfloat16}
    if args.use_flash_attn == "True":
        model_kwargs["use_flash_attention_2"] = True
    if args.load_strategy == "layer_stream":
        # device_map="cpu" keeps the whole bf16 model on host RAM during load.
        model_kwargs["device_map"] = {"": "cpu"}
        model_kwargs["low_cpu_mem_usage"] = True
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        **model_kwargs,
    )
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    target_vocab = int(8 * math.ceil(len(tokenizer) / 8.0))
    if target_vocab != model.config.vocab_size:
        # mean_resizing=False: new rows are zero-initialised instead of drawn
        # from a fitted multivariate normal. The fitted-normal path is
        # exorbitantly slow on 70B with low_cpu_mem_usage (it materialises
        # the whole 128k×8k embed at once). Since we only ever add a single
        # PAD row that won't appear in math labels, the init choice is
        # functionally irrelevant.
        model.resize_token_embeddings(target_vocab, mean_resizing=False)
    if args.load_strategy == "direct":
        model = model.to(accelerator.device)
    # For layer_stream we leave the model on CPU until the streaming converter
    # below pulls each target Linear's weight to GPU one at a time. The
    # post-BTT remainder (embeds, layernorms, lm_head) is moved at the end.

    # --- Dataset ---
    # Hoisted above decomposition so calibration can reuse train_dataset/collator.
    if args.prompt_style == "pissa":
        # Monkey-patch BASE_PROMPT in data_utils to match PiSSA's training prompt
        # (no literal '<s>', no trailing whitespace, and trailing newline only between
        # 'Response:' and the target output). Used to align qfura training prompt
        # with the published QPiSSA recipe on MetaMathQA so eval-time prompts (which
        # already use this format) match the train distribution.
        import utils.data_utils as _data_utils
        _data_utils.BASE_PROMPT = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:"
        )
        print_rank_0("[prompt_style=pissa] using PiSSA-style prompt for training", args.global_rank)
    if len(args.data_path) == 1 and ".json" in args.data_path[0]:
        train_dataset = SupervisedDataset(
            data_path=args.data_path[0],
            tokenizer=tokenizer,
            instruction_type=args.instruction_type,
            args=args,
        )
        if args.val_set_size > 0:
            train_dataset, eval_dataset = torch.utils.data.random_split(
                train_dataset,
                [len(train_dataset) - args.val_set_size, args.val_set_size],
            )
    else:
        raise ValueError("Only json format is supported for now.")

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    if args.val_set_size > 0:
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=data_collator,
        )

    # --- BlockTT conversion ---
    if getattr(args, "calib_mode", "none") != "none":
        calib_loader = build_calib_loader(
            args,
            tokenizer=tokenizer,
            training_dataset=train_dataset,
            training_collate_fn=data_collator,
            hyphen_style=False,
        )
        model, calib_stats = apply_calibrated_btt(
            model, args, calib_loader=calib_loader, hyphen_style=False,
        )
        print(f"[calib-btt] installed {calib_stats['num_btt_layers']} BTT layers")
    else:
        blocktt_rank = resolve_blocktt_rank(args.blocktt_rank)
        target_modules = get_blocktt_target_module_names(args.trainable_type)
        train_bias = not args.no_train_bias

        # Resolve decomp mode (may be scalar or per-module dict)
        decomp_mode, module_decomp_modes = resolve_blocktt_decomp_modes(
            args.decomp_mode,
            include_names=target_modules,
            default_mode="output_one_block",
        )

        if args.load_strategy == "layer_stream":
            # Fused BTT + NF4 quantisation, one Linear at a time, with a
            # transient CUDA round-trip per layer. This is the path required
            # for ≥30B models on a single 94 GB H100.
            from btt_layer import convert_and_quantize_linear_to_qbtt_streaming
            qstream_stats = convert_and_quantize_linear_to_qbtt_streaming(
                model,
                btt_rank=blocktt_rank,
                decomp_mode=module_decomp_modes if module_decomp_modes is not None else decomp_mode,
                train_position=args.train_position,
                s_merged_to=args.s_merged_to,
                quant_layout=args.quant_block_layout,
                target_modules=target_modules,
                cuda_device=accelerator.device,
                skip_names=("lm_head",),
                factorize_by_head=args.blocktt_factorize_by_head,
                convert_mode="svd",
                init_mode="default",
                input_factorization=args.blocktt_input_factorization,
            )
            print(
                f"[qfura streaming] converted+quantised {qstream_stats['num_converted']} "
                f"modules; bytes_saved={qstream_stats['bytes_saved']:,}"
            )
            # The streaming converter only sets trainability on the BTT cores
            # it created. Freeze every non-BTT param explicitly here (embeds,
            # layernorms, lm_head). Cannot use configure_blocktt_trainability
            # because that function reads btt_l/btt_r on every BTTLayer, but
            # those attrs have already been replaced with NF4 blobs.
            from btt_layer import QBTTLayer
            qbtt_param_ids = set()
            for _, mod in model.named_modules():
                if isinstance(mod, QBTTLayer):
                    for p in mod.parameters(recurse=False):
                        qbtt_param_ids.add(id(p))
            for n, p in model.named_parameters():
                if id(p) not in qbtt_param_ids:
                    p.requires_grad = False
            n_btt = sum(1 for _, m in model.named_modules() if isinstance(m, QBTTLayer))
            n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            n_total = sum(p.numel() for p in model.parameters())
            print(f"Converted modules: {qstream_stats['num_converted']} (BTT layers={n_btt})")
            print(
                f"Trainable params: {n_train:,} / {n_total:,} "
                f"({100 * n_train / n_total:.4f}%)"
            )
            # Move the residual (embeds, layernorms, lm_head) to GPU now.
            print("[qfura streaming] moving non-BTT residual modules to GPU...")
            model = model.to(accelerator.device)
            qstats = {
                "num_converted": qstream_stats["num_converted"],
                "bytes_saved": qstream_stats["bytes_saved"],
                "layout": args.quant_block_layout,
            }
        else:
            converted_modules = convert_linear_to_btt(
                model,
                btt_rank=blocktt_rank,
                decomp_mode=module_decomp_modes if module_decomp_modes is not None else decomp_mode,
                init_mode="default",
                include_names=target_modules,
                skip_names=("lm_head",),
                lr_act=False,
                s_merged_to=args.s_merged_to,
                train_position=args.train_position,
                factorize_by_head=args.blocktt_factorize_by_head,
                model_config=model.config,
                input_factorization=args.blocktt_input_factorization,
            )
            stats = configure_blocktt_trainability(
                model,
                train_bias=train_bias,
                train_position=args.train_position,
                train_singular_values=(args.s_merged_to == "keep_trainable"),
            )
            if stats["num_btt_layers"] == 0:
                raise ValueError("No layers were converted to BTT; check --trainable_type.")

            print(f"Converted modules: {len(converted_modules)}")
            print(
                f"Trainable params: {stats['trainable_param_count']:,} / "
                f"{stats['total_param_count']:,} "
                f"({100 * stats['trainable_param_count'] / stats['total_param_count']:.4f}%)"
            )
            print(
                f"Tuned cores: left={stats['tuned_left_cores']}, "
                f"right={stats['tuned_right_cores']}, biases={stats['tuned_biases']}"
            )

            # --- QFura: quantize frozen BTT cores to NF4 ---
            qstats = convert_btt_to_qbtt_(model, layout=args.quant_block_layout)
            print(
                f"[qfura] NF4 conversion: num_converted={qstats['num_converted']}, "
                f"bytes_saved={qstats['bytes_saved']:,}, layout={qstats['layout']!r}"
            )
        if use_wandb:
            accelerator.log(
                {
                    "qfura/num_converted": qstats["num_converted"],
                    "qfura/bytes_saved": qstats["bytes_saved"],
                    "qfura/layout": qstats["layout"],
                },
                step=0,
            )

    # --- Optionally upcast trainable params to fp32 (matches QPiSSA paper) ---
    # Walks every QBTTLayer / BTTLayer and replaces trainable parameters with
    # fp32 copies. The frozen NF4 large core (Params4bit) is untouched. This
    # path is for paper-faithful experiments; default qfura keeps bf16.
    if args.trainable_param_dtype == "fp32":
        from btt_layer import BTTLayer
        n_upcast = 0
        for _, mod in model.named_modules():
            if not isinstance(mod, BTTLayer):
                continue
            for attr in ("btt_l", "btt_r", "btt_s", "bias"):
                p = getattr(mod, attr, None)
                if p is None or not isinstance(p, torch.nn.Parameter) or not p.requires_grad:
                    continue
                if p.dtype == torch.float32:
                    continue
                fp32_param = torch.nn.Parameter(p.data.to(torch.float32), requires_grad=True)
                setattr(mod, attr, fp32_param)
                n_upcast += 1
        print_rank_0(f"[trainable_param_dtype=fp32] upcast {n_upcast} BTT params to fp32",
                     args.global_rank)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"param {name} is trainable (dtype={param.dtype})")

    if args.gradient_checkpointing:
        model = make_model_gradient_checkpointing_compatible(model)
        model.gradient_checkpointing_enable()

    # --- Optimizer ---
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "paged_adamw_8bit":
        optimizer = bnb.optim.PagedAdamW8bit(
            trainable_params,
            lr=args.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError(f"unknown --optimizer: {args.optimizer}")
    print_rank_0(f"[optimizer={args.optimizer}] {len(trainable_params)} trainable tensors",
                 args.global_rank)

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    if args.max_steps > 0:
        max_train_steps = min(max_train_steps, args.max_steps)

    if args.num_warmup_steps < 1:
        args.num_warmup_steps = int(args.num_warmup_steps * max_train_steps)
    else:
        args.num_warmup_steps = int(args.num_warmup_steps)

    print(f"max trainable steps: {max_train_steps}, warmup steps: {args.num_warmup_steps}")
    total_batch_size = (
        args.per_device_train_batch_size * args.gradient_accumulation_steps
    )

    print("***** Running QFura training *****")
    print(f"  Num examples = {len(train_dataloader)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    print(f"  Total train batch size (w. accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {max_train_steps}")

    progress_bar = tqdm(
        range(max_train_steps), disable=not accelerator.is_local_main_process
    )
    args.completed_steps = 0

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    # Prepare with accelerator
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    if args.val_set_size > 0:
        eval_dataloader = accelerator.prepare(eval_dataloader)

    best_model = None

    sysmon = SysMon(
        out_dir=args.output_dir or ".",
        method="qfura",
        rank=(None if args.blocktt_rank == "full" else int(args.blocktt_rank)),
        base_params=sum(p.numel() for p in model.parameters()),
    )
    _base = sysmon.base_params
    for name, p in model.named_parameters():
        if "btt_" in name:
            _base -= p.numel()
    sysmon.base_params = _base

    def train_epoch(epoch):
        nonlocal best_model, best_eval_loss
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                total_loss += loss.detach().float()

            if accelerator.sync_gradients:
                _t0 = time.time()
                optimizer.step()
                if args.blocktt_normalize_after_update:
                    unwrapped = accelerator.unwrap_model(model)
                    normalize_trainable_blocktt_cores_(unwrapped)
                lr_scheduler.step()
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                sysmon.record_step(time.time() - _t0)
                progress_bar.update(1)
                args.completed_steps += 1
                if args.max_steps > 0 and args.completed_steps >= args.max_steps:
                    return

                if (
                    args.logging_steps
                    and args.completed_steps % args.logging_steps == 0
                ):
                    divisor = args.gradient_accumulation_steps * args.logging_steps
                    avg_loss = (
                        accelerator.gather(total_loss).mean().item() / divisor
                    )
                    print(
                        f"  Step: {args.completed_steps}, "
                        f"LR: {lr_scheduler.get_last_lr()[0]:.8f}, "
                        f"Loss: {avg_loss:.6f}"
                    )
                    accelerator.log(
                        {
                            "learning_rate": lr_scheduler.get_last_lr()[0],
                            "train_loss": avg_loss,
                        },
                        step=args.completed_steps,
                    )
                    total_loss = 0

                if (
                    args.completed_steps % args.eval_step == 0
                    and args.val_set_size > 0
                    and not args.load_last_model
                ):
                    perplexity, eval_loss = evaluate(model)
                    accelerator.print(
                        f"Epoch {epoch+1} Step {args.completed_steps}: "
                        f"Eval perplexity = {perplexity:.4f}, Eval loss = {eval_loss:.4f}"
                    )
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        if accelerator.is_main_process and args.output_dir:
                            accelerator.wait_for_everyone()
                            unwrapped_model = accelerator.unwrap_model(model)
                            best_model = copy.deepcopy(unwrapped_model).to("cpu")
                            print("New best model")

        return total_loss / len(train_dataloader)

    def evaluate(model):
        model.eval()
        losses = 0
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            losses += loss.float()
        losses = losses / (step + 1)
        try:
            losses = get_all_reduce_mean(losses)
        except Exception:
            pass
        try:
            perplexity = torch.exp(losses).item()
        except OverflowError:
            perplexity = float("inf")
        model.train()
        return perplexity, losses.item()

    # --- Training loop ---
    best_eval_loss = float("inf")
    for epoch in range(args.num_train_epochs):
        train_loss = train_epoch(epoch)
        if train_loss is not None:
            accelerator.print(f"Epoch {epoch+1}: Average loss = {train_loss:.4f}")
        if args.max_steps > 0 and args.completed_steps >= args.max_steps:
            break

    effective_tokens = (
        args.per_device_train_batch_size
        * args.gradient_accumulation_steps
        * args.max_seq_len
    )
    sysmon.dump(
        model,
        extra={
            "effective_tokens_per_step": effective_tokens,
            "learning_rate": args.learning_rate,
            "train_position": args.train_position,
            "decomp_mode": args.decomp_mode,
            "s_merged_to": args.s_merged_to,
            "quant_block_layout": args.quant_block_layout,
        },
    )

    # --- Save policy: write last/ always, write best/ if best-tracking was active.
    # Layout:
    #   <output_dir>/last/  : final-step weights (always saved)
    #   <output_dir>/best/  : weights at lowest val_loss seen during training
    #                        (saved only if val_set_size > 0)
    # When --load_last_model is set, best-tracking is skipped during training
    # to save memory; only last/ is written.
    # When the model was loaded layer-streaming (>=30B), the materialised bf16
    # checkpoint cannot fit in GPU memory, so we offload each materialised
    # nn.Linear to CPU as it is built. save_hf_format then walks the CPU model
    # and writes safetensors shards from there. CPU has 1.5 TB on this box, so
    # this is fine; for the direct path on smaller models we keep everything on
    # GPU as before to avoid the host roundtrip.
    save_offload_device = (
        torch.device("cpu") if args.load_strategy == "layer_stream" else None
    )

    def _save_one(src_model, sub_folder):
        # Use save_hf_format(args, sub_folder=...) which writes to
        # <args.output_dir>/<sub_folder>. Mutates src_model in place via
        # materialize_btt_to_linear. Caller must not reuse src_model after.
        if getattr(args, "calib_mode", "none") != "none":
            target_dir = os.path.join(args.output_dir, sub_folder)
            os.makedirs(target_dir, exist_ok=True)
            save_calibrated_btt_checkpoint(src_model, target_dir, tokenizer)
        else:
            materialize_btt_to_linear(src_model, offload_device=save_offload_device)
            save_hf_format(src_model, tokenizer, args, sub_folder=sub_folder)

    if args.output_dir is not None and accelerator.is_main_process:
        accelerator.wait_for_everyone()

        # End-of-training eval, used to refresh best_model if better.
        if args.val_set_size > 0 and not args.load_last_model:
            ppl, val_loss = evaluate(model)
            print_rank_0(
                f"Validation perplexity: {ppl}, Validation loss: {val_loss}",
                args.global_rank,
            )
            if val_loss < best_eval_loss:
                best_eval_loss = val_loss
                if args.global_rank == 0:
                    best_model = copy.deepcopy(model.module).to("cpu")

        # Always save the final (last-step) model.
        last_model = accelerator.unwrap_model(model)
        _save_one(last_model, "last")
        print_rank_0(f"Saved last-step checkpoint to {os.path.join(args.output_dir, 'last')}", args.global_rank)

        # Save best if best-tracking ran (val_set_size > 0 and not skip).
        if best_model is not None:
            _save_one(best_model, "best")
            print_rank_0(
                f"Saved best-eval checkpoint to {os.path.join(args.output_dir, 'best')} "
                f"(val_loss={best_eval_loss:.4f})",
                args.global_rank,
            )

    if use_wandb:
        accelerator.end_training()


if __name__ == "__main__":
    main()
