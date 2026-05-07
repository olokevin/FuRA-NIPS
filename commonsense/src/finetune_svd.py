import sys
import os

# commonsense/src/ — for `utils.*`
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
# FuRA repo root — for `fura.*`
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)
# commonsense/ — for `tools.system_metrics`
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

import copy
import time
import torch
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

from fura.svd_layer import (
    SVDLayer,
    convert_linear_to_svd,
    configure_svd_trainability,
    get_svd_target_module_names,
)

from tools.system_metrics import SysMon


def materialize_svd_to_linear(model):
    """Replace all SVDLayer modules with nn.Linear containing materialized dense weights.

    Mirrors materialize_btt_to_linear in finetune_blocktt.py: needed so the saved
    HF checkpoint contains standard Linear weights instead of svd_a/svd_b factors.
    """
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, SVDLayer):
            replacements.append((name, module))

    for name, svd_module in replacements:
        dense_weight = svd_module.materialize_dense_weight()
        linear = nn.Linear(
            svd_module.in_features,
            svd_module.out_features,
            bias=svd_module.bias is not None,
            device=dense_weight.device,
            dtype=dense_weight.dtype,
        )
        linear.weight.data.copy_(dense_weight)
        if svd_module.bias is not None:
            linear.bias.data.copy_(svd_module.bias.data)

        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], linear)

    print(f"Materialized {len(replacements)} SVDLayer modules to nn.Linear")
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="SVD Fine-Tuning (LIFT benchmark)")
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

    # SVD-specific arguments (mirror run_rl.py --train-mode svd)
    parser.add_argument(
        "--trainable_type", type=str, default="all",
        choices=["all", "mlp", "attn"],
        help="Which modules to convert to SVD: all, mlp, attn",
    )
    parser.add_argument(
        "--train_position", type=str, default="output",
        choices=["output", "input", "both"],
        help="Which SVD factor to train: output (svd_a / U-side), input (svd_b / V-side), or both",
    )
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
    parser.add_argument("--no_train_bias", action="store_true",
        help="Freeze biases on SVD layers")
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

    # Mirror run_rl.py default: when train_position=both with frozen/trainable
    # alias, the downstream resolver maps to "split". We allow it but warn the
    # user that frozen/trainable have no merge target with two trainable sides.
    if args.train_position == "both" and args.s_merged_to in {"frozen", "trainable"}:
        # Promote to canonical "split" so the user gets a deterministic init.
        args.s_merged_to = "split"

    return args


def main():
    args = parse_args()

    use_wandb = not args.no_wandb
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb" if use_wandb else None,
    )
    if not torch.cuda.is_available() or accelerator.device.type != "cuda":
        raise RuntimeError(
            "finetune_svd.py requires CUDA so SVD decomposition runs on GPU. "
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

    # --- SVD conversion (full-rank decomposition of pretrained weights) ---
    target_modules = get_svd_target_module_names(args.trainable_type)
    train_bias = not args.no_train_bias

    converted_modules = convert_linear_to_svd(
        model,
        skip_names=("lm_head",),
        include_names=target_modules,
        s_merged_to=args.s_merged_to,
        train_position=args.train_position,
    )
    stats = configure_svd_trainability(
        model,
        train_position=args.train_position,
        train_bias=train_bias,
        train_embed_lm_head=(args.train_position == "both"),
        train_singular_values=(args.s_merged_to == "keep_trainable"),
    )
    if stats["num_svd_layers"] == 0:
        raise ValueError("No layers were converted to SVD; check --trainable_type.")

    print(f"Converted modules: {len(converted_modules)}")
    print(
        f"Trainable params: {stats['trainable_param_count']:,} / "
        f"{stats['total_param_count']:,} "
        f"({100 * stats['trainable_param_count'] / stats['total_param_count']:.4f}%)"
    )
    print(
        f"Tuned cores: output={stats['tuned_output_cores']}, "
        f"input={stats['tuned_input_cores']}, biases={stats['tuned_biases']}"
    )

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"param {name} is trainable")

    if args.gradient_checkpointing:
        model = make_model_gradient_checkpointing_compatible(model)
        model.gradient_checkpointing_enable()

    # --- Optimizer: standard AdamW on trainable SVD factors ---
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

    print("***** Running SVD training *****")
    print(f"  Num examples = {len(train_dataloader)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    print(f"  Total train batch size (w. accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {max_train_steps}")
    print(f"  Train position = {args.train_position}")
    print(f"  S merged to = {args.s_merged_to}")

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

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    if args.val_set_size > 0:
        eval_dataloader = accelerator.prepare(eval_dataloader)

    best_model = None

    sysmon = SysMon(
        out_dir=args.output_dir or ".",
        method="svd",
        rank=None,  # full-rank SVD
        base_params=sum(p.numel() for p in model.parameters()),
    )
    _base = sysmon.base_params
    for name, p in model.named_parameters():
        if "svd_" in name:
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
            "s_merged_to": args.s_merged_to,
        },
    )

    # --- Save policy: write last/ always, best/ if best-tracking ran.
    def _save_one(src_model, sub_folder):
        materialize_svd_to_linear(src_model)
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
