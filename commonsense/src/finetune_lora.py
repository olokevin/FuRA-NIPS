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
import torch
import json
import random
import math
import time
import argparse
from tqdm.auto import tqdm


from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
    SchedulerType,
    DataCollatorForSeq2Seq,
    get_scheduler,
    GPTNeoXTokenizerFast,
    GPT2Tokenizer,
    OPTForCausalLM,
    BitsAndBytesConfig,
    PreTrainedTokenizerFast,
)

# from peft import (  # noqa: E402
#     LoraConfig,
#     # DoraConfig,
#     BottleneckConfig,
#     PrefixTuningConfig,
#     get_peft_model,
#     get_peft_model_state_dict,
#     # prepare_model_for_int8_training,
#     set_peft_model_state_dict,
# )
from peft import (  # noqa: E402
    LoraConfig,
    PrefixTuningConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    prepare_model_for_kbit_training,
)


from utils.utils import (
    print_rank_0,
    to_device,
    set_random_seed,
    get_all_reduce_mean,
    int_or_float,
)

from accelerate import Accelerator
from accelerate.utils import set_seed

from utils.model_utils import (
    load_hf_tokenizer,
    create_hf_model,
    save_hf_format,
    get_optimizer_grouped_parameters,
    make_model_gradient_checkpointing_compatible,
    print_throughput
)

from utils.data_utils import SupervisedDataset, DataCollatorForSupervisedDataset

from tools.system_metrics import SysMon

@torch.no_grad()
def apply_milora_init_(peft_model, *, rank: int) -> None:
    """In-place MiLoRA initialization (vendored locally to keep LIFT
    independent from the main repo's run_sft.py helper)."""
    from peft.tuners.lora import LoraLayer
    first_check = True
    for _name, module in peft_model.named_modules():
        if not isinstance(module, LoraLayer):
            continue
        base = module.get_base_layer()
        W = base.weight.data.detach().clone().float()
        dtype, device = base.weight.data.dtype, base.weight.data.device
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        r = rank
        U_r, S_r, Vh_r = U[:, -r:], S[-r:], Vh[-r:, :]
        sqrt_S = S_r.sqrt()
        adapter_name = list(module.lora_A.keys())[0]
        alpha = module.lora_alpha[adapter_name]
        scale_correction = (r / alpha) ** 0.5
        lora_A = (sqrt_S.unsqueeze(1) * Vh_r) * scale_correction
        lora_B = (U_r * sqrt_S.unsqueeze(0)) * scale_correction
        residual = W - U_r @ torch.diag(S_r) @ Vh_r
        module.lora_A[adapter_name].weight.data.copy_(lora_A.to(dtype=dtype, device=device))
        module.lora_B[adapter_name].weight.data.copy_(lora_B.to(dtype=dtype, device=device))
        base.weight.data.copy_(residual.to(dtype=dtype, device=device))
        if first_check:
            reconstructed = (alpha / r) * lora_B @ lora_A + residual
            rel_err = (
                torch.linalg.norm(reconstructed - W)
                / torch.linalg.norm(W)
            )
            assert rel_err < 1e-3, (
                f"MiLoRA init reconstruction error too high: {rel_err:.2e}"
            )
            first_check = False

def parse_args():
    parser = argparse.ArgumentParser(description="S2FT Training")
    parser.add_argument(
        "--data_path",
        nargs="*",
        default=["./LLM-Adapters/ft-training_set/commonsense_170k.json"],
        help="Path to the training dataset. Accepted format:"
        "1) a single data path, 2) multiple datasets in the"
        "form: dataset1-path dataset2-path ...",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=2048,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--val_set_size",
        type=int,
        default=100,
        help="Size of the validation set. If 0, no validation set is used.",
    )
    parser.add_argument(
        "--load_last_model", action="store_true", help="only save the last model"
    )
    parser.add_argument(
        "--eval_step",
        type=int,
        default=80,
        help="size of eval_step",
    )
    parser.add_argument(
        "--eval_delay",
        type=int_or_float,
        default=0,
        help="eval after certain steps if it is an integer, or eval after certain ratio of steps if it is a float",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="If > 0, cap total optimizer steps at this value (for short-horizon system-eval runs).",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=float,
        default=0.,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["fp16", "bf16", "fp32"],
        help="Training data type",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the model."
    )
    parser.add_argument(
        "--seed", type=int, default=1234, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable HF gradient checkpointing for model.",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.0, help="dropout rate of the model."
    )
    # deepspeed features
    parser.add_argument(
        "--offload", action="store_true", help="Enable ZeRO Offload techniques."
    )
    parser.add_argument(
        "--zero_stage",
        type=int,
        default=0,
        help="ZeRO optimization stage for Actor model (and clones).",
    )

    # S2FT
    parser.add_argument(
        "--s2", action="store_true", help="use S2FT for efficient training."
    )
    parser.add_argument("--v_ratio", type=float, default=0.0)
    parser.add_argument("--o_ratio", type=float, default=0.0)
    parser.add_argument("--u_ratio", type=float, default=0.0)
    parser.add_argument("--d_ratio", type=float, default=0.0)
    ## Tensorboard logging
    parser.add_argument(
        "--enable_tensorboard", action="store_true", help="Enable tensorboard logging"
    )
    parser.add_argument("--tensorboard_path", type=str, default="tensorboard")
    parser.add_argument(
        "--instruction_type", type=str, choices=["single", "multi"], default="single"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=500,
        help="save deepspeed engine checkpoint for recover the training",
    )

    parser.add_argument(
        "--peft_tuner",
        type=str,
        default='none',
        help="sparse fine-tuning tuner",
    )

    parser.add_argument(
        "--use_flash_attn",
        type=str,
        default='False',
        help="sparse fine-tuning tuner",
    )

    parser.add_argument(
        "--adapter_name",
        type=str,
        default='lora',
        help="sparse fine-tuning tuner",
    )

    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="save deepspeed engine checkpoint for recover the training",
    )

    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="save deepspeed engine checkpoint for recover the training",
    )

    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="save deepspeed engine checkpoint for recover the training",
    )

    parser.add_argument(
        "--target_modules",  # Argument name
        nargs="+",                # Allows multiple inputs
        type=str,                 # Each input should be a string
        default=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"],
        help="List of LoRA target modules separated by spaces."
    )
    parser.add_argument(
        "--Wdecompose_target_modules",  # Argument name
        nargs="+",                # Allows multiple inputs
        type=str,                 # Each input should be a string
        default=None,
        help="List of LoRA target modules separated by spaces."
    )

    parser.add_argument(
        "--dora_simple",
        action='store_true',
    )

    parser.add_argument(
        "--randlora_projection_prng_key",
        type=int,
        default=0,
        help="Seed for RandLoRA's shared random bases (default: 0).",
    )

    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="sparse fine-tuning tuner",
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

    # Set seed for reproducibility
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
    
    # Load tokenizer and model
    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    tokenizer.model_max_length = args.max_seq_len

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(args.model_name_or_path)

    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_config(config)

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id

    model.resize_token_embeddings(int(
        8 *
        math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8
        
    if args.gradient_checkpointing:
        model = make_model_gradient_checkpointing_compatible(model)
        model.gradient_checkpointing_enable()

    # model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    print("Target Modules:")
    print(args.target_modules)

    if args.adapter_name in ["lora", "dora", "pissa"]:
        if args.adapter_name == "lora":
            print(f"LoRA Init")
            config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=args.target_modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
        elif args.adapter_name == 'dora':
            print(f"DoRA Init")
            config = LoraConfig(
                use_dora=True,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=args.target_modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
        elif args.adapter_name == "pissa":
            print("Pissa init")
            config = LoraConfig(
                # init_lora_weights="pissa", # Configure the initialization method to "pissa", which may take several minutes to execute SVD on the pre-trained model.
                init_lora_weights="pissa_niter_4", # Initialize the PiSSA with fast SVD, which completes in just a few seconds.
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=0, # Since the component of the PiSSA adapter are the principal singular values and vectors, dropout should be set to 0 to avoid random discarding.
                target_modules=args.target_modules,
                task_type="CAUSAL_LM",
            )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
    elif args.adapter_name == "milora":
        print("MiLoRA Init")
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0,
            target_modules=args.target_modules,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        apply_milora_init_(model, rank=args.lora_r)
        model.print_trainable_parameters()
    elif args.adapter_name == "randlora":
        print("RandLoRA Init")
        from peft import RandLoraConfig
        config = RandLoraConfig(
            r=args.lora_r,
            randlora_alpha=args.lora_alpha,
            target_modules=args.target_modules,
            projection_prng_key=args.randlora_projection_prng_key,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

    print(model)
    print(model.dtype)

    # config.save_pretrained(args.output_dir)

    # Prepare datasets
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

    for name, param in model.named_parameters():
        if not param.requires_grad:
            pass
        else:
            print(f'param {name} is trainable')

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=DataCollatorForSupervisedDataset(tokenizer=tokenizer),
    )

    if args.val_set_size > 0:
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=DataCollatorForSupervisedDataset(tokenizer=tokenizer),
        )

    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
       args, model, args.weight_decay, args.learning_rate
    )

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    if args.max_steps > 0:
        max_train_steps = min(max_train_steps, args.max_steps)

    if args.num_warmup_steps < 1:
        args.num_warmup_steps = int(args.num_warmup_steps * max_train_steps)
    else:
        args.num_warmup_steps = int(args.num_warmup_steps)

    print(f"max trainable steps: {max_train_steps}, warmup steps: {args.num_warmup_steps}")
    total_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataloader)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)

    args.completed_steps = 0

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    # Prepare everything with accelerator
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    if args.val_set_size > 0:
        eval_dataloader = accelerator.prepare(eval_dataloader)

    best_model = None

    sysmon = SysMon(
        out_dir=args.output_dir or ".",
        method=args.adapter_name,
        rank=int(args.lora_r) if hasattr(args, "lora_r") else None,
        base_params=0,
    )
    _total_now = sum(p.numel() for p in model.parameters())
    _adapter_params = sum(
        p.numel() for n, p in model.named_parameters()
        if "lora_" in n or "randlora" in n.lower() or "magnitude" in n.lower()
    )
    sysmon.base_params = _total_now - _adapter_params

    # Training function
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

                if args.logging_steps and args.completed_steps % args.logging_steps == 0:
                    divisor = args.gradient_accumulation_steps * args.logging_steps
                    avg_loss = accelerator.gather(total_loss).mean().item() / divisor
                    log_vals = [
                        f'LR: {lr_scheduler.get_last_lr()[0]:.8f}',
                        f'Loss: {avg_loss:.6f}',
                    ]
                    if hasattr(model, '_losses'):
                        for k in list(model._losses.keys()):
                            log_vals.append(f'{k}: {model._losses[k] / divisor:.8f}')
                            model._losses[k] = 0
                    log_str = ', '.join(log_vals)
                    print(f"  Step: {args.completed_steps}, {log_str}")
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
                    accelerator.print(f"Epoch {epoch+1}: Eval perplexity = {perplexity:.4f}, Eval loss = {eval_loss:.4f}")
                    
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        if accelerator.is_main_process and args.output_dir:
                            accelerator.wait_for_everyone()
                            unwrapped_model = accelerator.unwrap_model(model)
                            best_model = copy.deepcopy(unwrapped_model).to("cpu")
                            print("New best model")
                            
                            # save_hf_format(unwrapped_model, tokenizer, args)
                
        return total_loss / len(train_dataloader)

    # Evaluation function
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
        except:
            pass
        try:
            perplexity = torch.exp(losses).item()
        except OverflowError:
            perplexity = float("inf")
        model.train()
        return perplexity, losses.item()

    # Training loop
    best_eval_loss = float('inf')
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
            "adapter_name": args.adapter_name,
            "lora_alpha": args.lora_alpha,
        },
    )

    # --- Save policy: write last/ always, best/ if best-tracking ran.
    # For peft adapters (lora/dora/pissa/milora/randlora), merge_and_unload
    # before save so the on-disk checkpoint is a full HF model usable directly
    # by the eval scripts.
    def _save_one(src_model, sub_folder):
        if args.adapter_name in ["lora", "dora", "pissa", "milora", "randlora"]:
            src_model = src_model.merge_and_unload()
        save_hf_format(src_model, tokenizer, args, sub_folder=sub_folder)

    if args.output_dir is not None and accelerator.is_main_process:
        accelerator.wait_for_everyone()

        if args.val_set_size > 0 and not args.load_last_model:
            ppl, val_loss = evaluate(model)
            print_rank_0(
                f"Validation perplexity: {ppl}, Validation loss: {val_loss}",
                args.global_rank,
            )
            print(f"Validation perplexity: {ppl}, Validation loss: {val_loss}")
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

def save_with_accelerate(accelerator, model, tokenizer, output_dir, args):
    unwrapped_model = accelerator.unwrap_model(model)
    # When doing multi-gpu training, we need to use accelerator.get_state_dict(model) to get the state_dict.
    # Otherwise, sometimes the model will be saved with only part of the parameters.
    # Also, accelerator needs to use the wrapped model to get the state_dict.
    state_dict = accelerator.get_state_dict(model)
    if args.peft_tuner:
        # When using lora, the unwrapped model is a PeftModel, which doesn't support the is_main_process 
        # and has its own save_pretrained function for only saving lora modules.
        # We have to manually specify the is_main_process outside the save_pretrained function.
        if accelerator.is_main_process:
            # unwrapped_model.save_pretrained(output_dir, state_dict=state_dict, safe_serialization=False)
            model = model.merge_and_unload()
            # model.save_pretrained(output_dir)
            print(model)
            model.save_pretrained(
                output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save, state_dict=state_dict
            )
    else:
        unwrapped_model.save_pretrained(
            output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save, state_dict=state_dict
        )

if __name__ == "__main__":
    main()
    os._exit(0)
