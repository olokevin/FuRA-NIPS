import sys
import os

# Add this directory (commonsense/src/) to path so `utils.*` imports work.
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
# Add the FuRA repo root (parent of `commonsense/`) to path so the shared
# `fura/` package and the local `tools/` helper are importable.
_FURA_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)
)
if _FURA_REPO_ROOT not in sys.path:
    sys.path.insert(0, _FURA_REPO_ROOT)
# Local commonsense/ directory (for `tools.system_metrics`).
_COMMONSENSE_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.path.pardir)
)
if _COMMONSENSE_ROOT not in sys.path:
    sys.path.insert(0, _COMMONSENSE_ROOT)

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
    configure_blocktt_trainability,
    get_blocktt_target_module_names,
    normalize_trainable_blocktt_cores_,
    resolve_blocktt_decomp_modes,
)

from tools.system_metrics import SysMon

# Calibrated-BTT (used only when --calib_mode != 'none') is part of QFuRA and
# requires the optional `compress` library. The default commonsense recipe runs
# with --calib_mode none, so we provide stubs that fail loudly only if the user
# explicitly opts into calibration. For the QFuRA reproduction, see the
# `qfura_commonsense/` track.
try:
    from compress_integration import (  # type: ignore
        add_calibrated_btt_args,
        validate_calibrated_btt_args,
        apply_calibrated_btt,
        build_calib_loader,
        save_calibrated_btt_checkpoint,
    )
except ImportError:
    # Mirror the original `add_calibrated_btt_args` flag set so existing launch
    # scripts (which pass `--calib_mode none` and friends) continue to parse.
    # All branches that consume calib_mode != 'none' raise below.
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
                "which is not bundled with this open-source release. "
                "See the `qfura_commonsense/` track for the QFuRA reproduction."
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


def materialize_btt_to_linear(model):
    """Replace all BTTLayer modules with nn.Linear containing materialized dense weights.

    This makes the model saveable/loadable as a standard HF checkpoint.
    """
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, BTTLayer):
            replacements.append((name, module))

    for name, btt_module in replacements:
        dense_weight = btt_module.materialize_dense_weight()
        linear = nn.Linear(
            btt_module.in_features,
            btt_module.out_features,
            bias=btt_module.bias is not None,
            device=dense_weight.device,
            dtype=dense_weight.dtype,
        )
        linear.weight.data.copy_(dense_weight)
        if btt_module.bias is not None:
            linear.bias.data.copy_(btt_module.bias.data)

        # Navigate to parent and replace the child
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], linear)

    print(f"Materialized {len(replacements)} BTTLayer modules to nn.Linear")
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="BlockTT Fine-Tuning (LIFT benchmark)")
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

    # BlockTT-specific arguments
    parser.add_argument("--trainable_type", type=str, default="all",
        choices=["all", "mlp", "attn"],
        help="Which modules to convert to BTT: all, mlp, attn")
    parser.add_argument("--decomp_mode", type=str, default="input_one_block",
        help="BTT decomposition mode: input_one_block, output_one_block, or dict literal")
    parser.add_argument("--blocktt_rank", type=str, default="full",
        help="BTT rank: 'full' for lossless or a positive integer")
    parser.add_argument("--train_position", type=str, default="small",
        choices=["small", "large", "both"],
        help="Which TT core to train: small, large, both")
    parser.add_argument(
        "--s_merged_to",
        type=str,
        default="frozen",
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
            "individual specs. Example: '{\"qkv\":\"head\",\"o\":\"head\","
            "\"mlp_upgate\":\"512,8\",\"mlp_down\":\"1792,8\"}'. When set, this "
            "overrides --blocktt_factorize_by_head for any module covered by the spec."
        ),
    )
    parser.add_argument("--no_train_bias", action="store_true",
        help="Freeze BTT biases")
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
            "finetune_blocktt.py requires CUDA so BlockTT conversion SVD runs on GPU. "
            f"Current accelerator device: {accelerator.device}."
        )

    set_seed(args.seed)
    args.global_rank = 1

    if use_wandb:
        if args.wandb_project is None:
            args.wandb_project = "lift"
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

    # Load model
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model_kwargs = {"torch_dtype": torch.bfloat16}
    if args.use_flash_attn == "True":
        model_kwargs["use_flash_attention_2"] = True
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        **model_kwargs,
    )
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(int(8 * math.ceil(len(tokenizer) / 8.0)))
    model = model.to(accelerator.device)

    # --- Dataset ---
    # Hoisted above decomposition so calibration can reuse train_dataset/collator.
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
            default_mode="input_one_block",
        )

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

        if args.train_position == "both":
            for p in model.parameters():
                p.requires_grad = True
            print("[train_position=both] all parameters set trainable (incl. lm_head, embeddings)")

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"param {name} is trainable")

    if args.gradient_checkpointing:
        model = make_model_gradient_checkpointing_compatible(model)
        model.gradient_checkpointing_enable()

    # --- Optimizer: standard AdamW on trainable TT cores ---
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

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

    print("***** Running BlockTT training *****")
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
        method="blocktt",
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
        },
    )

    # --- Save policy: write last/ always, best/ if best-tracking ran.
    def _save_one(src_model, sub_folder):
        if getattr(args, "calib_mode", "none") != "none":
            target_dir = os.path.join(args.output_dir, sub_folder)
            os.makedirs(target_dir, exist_ok=True)
            save_calibrated_btt_checkpoint(src_model, target_dir, tokenizer)
        else:
            materialize_btt_to_linear(src_model)
            save_hf_format(src_model, tokenizer, args, sub_folder=sub_folder)

    if args.output_dir is not None and accelerator.is_main_process:
        accelerator.wait_for_everyone()

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

        last_model = accelerator.unwrap_model(model)
        _save_one(last_model, "last")
        print_rank_0(f"Saved last-step checkpoint to {os.path.join(args.output_dir, 'last')}", args.global_rank)

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
