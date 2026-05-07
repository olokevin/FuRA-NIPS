import sys
import os

# qdora_baseline/src/ — for `utils.*`
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
# qdora_baseline/ — for `qdora_fast` and `tools.system_metrics`
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
# FuRA repo root — for `fura.*` (qdora doesn't use it directly, but kept for symmetry)
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir))
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
    BitsAndBytesConfig,
    SchedulerType,
    get_scheduler,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
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

from tools.system_metrics import SysMon


def parse_args():
    parser = argparse.ArgumentParser(
        description="qdora Fine-Tuning (LIFT benchmark) — 4-bit NF4 base model + PEFT DoRA adapters (use_dora=True) + PagedAdamW8bit"
    )
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

    # LoRA-specific arguments
    parser.add_argument(
        "--lora_r",
        type=int,
        default=64,
        help="LoRA rank.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=128,
        help="LoRA alpha scaling factor.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout probability.",
    )
    parser.add_argument(
        "--target_modules",
        nargs="+",
        type=str,
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        help="List of module names to apply LoRA adapters to.",
    )
    parser.add_argument(
        "--qdora_impl",
        type=str,
        default="fast",
        choices=["peft", "fast"],
        help=(
            "DoRA implementation. 'peft' uses LoraConfig(use_dora=True) — slow "
            "but matches the published reference exactly. 'fast' uses the "
            "hand-rolled Qdora4bitLinear with bnb.matmul_4bit fused kernel and "
            "cached column-norm — ~2x faster, default."
        ),
    )
    parser.add_argument(
        "--dora_norm_cache_steps",
        type=int,
        default=16,
        help=(
            "Recompute the DoRA column norm every K training steps. K=1 "
            "matches PEFT's exact behavior (norm refreshed every step). "
            "Higher K trades a little staleness for speed. Ignored when "
            "--qdora_impl=peft."
        ),
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
            "Upcast trainable DoRA params (lora_A/lora_B + magnitude) to this "
            "dtype after model construction. fp32 matches the QPiSSA paper; "
            "bf16 (default) is the standard QLoRA/QDoRA recipe."
        ),
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["paged_adamw_8bit", "adamw"],
        default="paged_adamw_8bit",
        help=(
            "'paged_adamw_8bit' (default): bnb 8-bit moments, paged for OOM "
            "safety. 'adamw': torch.optim.AdamW with fp32 moments — matches "
            "QPiSSA paper. Only meaningful at fp32 trainable_param_dtype."
        ),
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

    args = parser.parse_args()

    # Hard constraint: qdora requires gradient checkpointing
    if not args.gradient_checkpointing:
        raise ValueError(
            "finetune_qdora.py requires --gradient_checkpointing. "
            "The 4-bit NF4-quantized base model increases activation memory; "
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

    set_seed(args.seed)
    args.global_rank = 1

    if use_wandb:
        if args.wandb_project is None:
            args.wandb_project = "qdora"
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

    # Load model with 4-bit NF4 quantization (QLoRA recipe; DoRA adapters land on top)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model_kwargs = {}
    if args.use_flash_attn == "True":
        model_kwargs["use_flash_attention_2"] = True
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map={"": accelerator.device.index if accelerator.device.index is not None else 0},
        **model_kwargs,
    )
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(int(8 * math.ceil(len(tokenizer) / 8.0)))

    # --- Dataset ---
    if args.prompt_style == "pissa":
        # Match the QPiSSA published training prompt: no literal '<s>', no
        # trailing whitespace, single newline between 'Response:' and target.
        # Lets qdora reuse the same train distribution as QPiSSA when comparing
        # MetaMathQA results.
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

    # --- qdora: NF4 base + DoRA adapters.
    # DoRA decomposes each linear's weight into a magnitude vector m
    # (one scalar per output column) and a direction matrix; trainable
    # params are m, A, B with the original W frozen (NF4-quantized here).
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    if args.qdora_impl == "peft":
        # Reference path: PEFT's Linear4bit + use_dora=True. Slow because
        # PEFT dequantizes the full weight on every forward to compute the
        # column norm. Kept as a baseline for ablation.
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=list(args.target_modules),
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            use_dora=True,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print(f"[qdora] using PEFT reference path (use_dora=True).")
    elif args.qdora_impl == "fast":
        # Hand-rolled Qdora4bitLinear: bnb.matmul_4bit fused base matmul +
        # column-norm cached every K steps. Numerically equivalent to PEFT's
        # path at K=1. Default K=16 trades a tiny bit of staleness for speed.
        sys.path.insert(
            0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir))
        )
        from qdora_fast import convert_to_qdora_fast
        # Freeze everything first.
        for p in model.parameters():
            p.requires_grad = False
        stats = convert_to_qdora_fast(
            model,
            target_module_names=list(args.target_modules),
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            norm_cache_steps=args.dora_norm_cache_steps,
        )
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(
            f"[qdora] fast path: replaced {stats['num_converted']} Linear4bit "
            f"layers with Qdora4bitLinear (norm_cache_steps={args.dora_norm_cache_steps})."
        )
        print(
            f"[qdora] trainable params: {n_trainable:,} || all params: {n_total:,} "
            f"|| trainable%: {100*n_trainable/n_total:.4f}"
        )
    else:
        raise ValueError(f"unknown --qdora_impl: {args.qdora_impl}")

    # --- Optionally upcast trainable params to fp32 (matches QPiSSA paper) ---
    # Replaces every trainable Parameter (lora_A.weight, lora_B.weight,
    # magnitude / lora_magnitude_vector) with an fp32 copy. The frozen NF4
    # base weight (Params4bit) is untouched.
    if args.trainable_param_dtype == "fp32":
        n_upcast = 0
        for parent_name, mod in list(model.named_modules()):
            for attr_name, p in list(mod.named_parameters(recurse=False)):
                if not p.requires_grad or p.dtype == torch.float32:
                    continue
                fp32_param = torch.nn.Parameter(p.data.to(torch.float32), requires_grad=True)
                setattr(mod, attr_name, fp32_param)
                n_upcast += 1
        print_rank_0(f"[trainable_param_dtype=fp32] upcast {n_upcast} qdora params to fp32",
                     args.global_rank)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"param {name} is trainable (dtype={param.dtype})")

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

    print("***** Running qdora training *****")
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
        method="qdora",
        rank=int(args.lora_r),
        base_params=sum(p.numel() for p in model.parameters()),
    )
    _total_now = sum(p.numel() for p in model.parameters())
    _adapter_params = sum(
        p.numel() for n, p in model.named_parameters()
        if "lora_" in n
    )
    sysmon.base_params = _total_now - _adapter_params

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
            "method": "qdora",
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
        },
    )

    # --- Save policy: write last/ always, best/ if best-tracking ran.
    # qdora_impl=fast: writes a fully-merged HF model directly to
    #     <output_dir>/<sub_folder>/. Eval can load it as a standard model.
    # qdora_impl=peft: writes only the PEFT adapter to
    #     <output_dir>/<sub_folder>_adapter/. The shell merges into bf16 base
    #     to produce <output_dir>/<sub_folder>/ for eval.
    def _save_one(src_model, sub_folder):
        unwrapped = accelerator.unwrap_model(src_model)
        if args.qdora_impl == "fast":
            from qdora_fast import materialize_qdora_to_linear
            # Spill merged bf16 weights to CPU as we materialize so peak GPU
            # memory stays bounded by a single layer's bf16 size (~470 MB on
            # Llama-3-70B's down_proj). Required for 70B on a single 94 GB H100,
            # cheap (~few seconds) for smaller models.
            n_merged = materialize_qdora_to_linear(unwrapped, target_device="cpu")
            print(f"[qdora] fast path ({sub_folder}): materialized {n_merged} Qdora4bitLinear modules to nn.Linear (CPU) before save")
            # Move the residual (embeds, layernorms, lm_head) to CPU too, so
            # save_pretrained's state_dict aggregation and shard write happen
            # entirely on CPU — required for 70B which does not fit in one GPU
            # as bf16 (141 GB > 94 GB).
            unwrapped = unwrapped.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            target_dir = os.path.join(args.output_dir, sub_folder)
            os.makedirs(target_dir, exist_ok=True)
            unwrapped.save_pretrained(
                target_dir, safe_serialization=True, max_shard_size="5GB"
            )
            tokenizer.save_pretrained(target_dir)
        else:
            target_dir = os.path.join(args.output_dir, f"{sub_folder}_adapter")
            os.makedirs(target_dir, exist_ok=True)
            unwrapped.save_pretrained(target_dir)
            tokenizer.save_pretrained(target_dir)

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

        _save_one(model, "last")
        print_rank_0(
            f"Saved last-step checkpoint to {os.path.join(args.output_dir, 'last' if args.qdora_impl == 'fast' else 'last_adapter')}",
            args.global_rank,
        )

        if best_model is not None:
            _save_one(best_model, "best")
            print_rank_0(
                f"Saved best-eval checkpoint to {os.path.join(args.output_dir, 'best' if args.qdora_impl == 'fast' else 'best_adapter')} "
                f"(val_loss={best_eval_loss:.4f})",
                args.global_rank,
            )

    if use_wandb:
        accelerator.end_training()


if __name__ == "__main__":
    main()
