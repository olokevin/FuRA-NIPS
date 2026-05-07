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
import torch.nn as nn

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
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
    get_optimizer_grouped_parameters,
    make_model_gradient_checkpointing_compatible,
)

from utils.data_utils import SupervisedDataset, DataCollatorForSupervisedDataset

from tools.system_metrics import SysMon


_PROJ_ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
_PROJ_MLP = ("gate_proj", "up_proj", "down_proj")
_PROJ_UPROJ_MODULES = ("gate_proj", "up_proj")  # informative u-proj on Llama-3-8B
_PROJ_VPROJ_MODULES = ("down_proj",)            # informative v-proj on Llama-3-8B


def _proj_module_path(layer_idx: int, module_name: str) -> str:
    sub = "self_attn" if module_name in _PROJ_ATTN else "mlp"
    return f"model.layers.{layer_idx}.{sub}.{module_name}.weight"


class ProjEnergyRecorder:
    """Record U/V projection energy of W', G, and dW = W' - W0 during training.

    For each tracked (layer, module), at each record step we compute up to six
    scalars: u_proj and v_proj of {W' (current weight), G (current grad), dW =
    W' - W0}. Only u_proj is plotted for {gate_proj, up_proj} and only v_proj
    for down_proj (other shapes are trivial-1.0 — see docs/exp_results/motivating.md).

    U0, V0^T, and W0 are SVD-cached at init on CPU. At each record we move the
    needed factor to GPU, compute ||U0^T X||_F / ||X||_F (and the V version),
    and append one JSON line per step under output_dir/proj_energy.jsonl. At
    end-of-training, the main process writes proj_energy.csv and a 2x3 plot
    (rows = U/V; cols = W', G, dW) to output_dir/proj_energy.png.

    IMPORTANT: record() must be called between accelerator.backward(loss) and
    optimizer.zero_grad(), so that param.grad still holds the per-step gradient.
    """

    QUANTITIES = ("W", "G", "dW")  # W' (current weight), G (grad), dW = W' - W0

    def __init__(self, model, layer_indices, module_names, output_dir, device,
                 svd_dtype=torch.float32, is_main_process=True):
        self.layer_indices = list(layer_indices)
        self.module_names = list(module_names)
        self.output_dir = output_dir
        self.device = device
        self.svd_dtype = svd_dtype
        self.is_main_process = is_main_process
        self.records = []  # in-memory mirror of the JSONL

        os.makedirs(output_dir, exist_ok=True)
        self.jsonl_path = os.path.join(output_dir, "proj_energy.jsonl")
        self.csv_path = os.path.join(output_dir, "proj_energy.csv")
        self.png_path = os.path.join(output_dir, "proj_energy.png")

        # Build target list and resolve parameters
        # t = {layer, mod, key, param (live ref), U0_cpu?, Vt0_cpu?, W0_cpu}
        self.targets = []
        sd = dict(model.named_parameters())
        for L in self.layer_indices:
            for mod in self.module_names:
                key = _proj_module_path(L, mod)
                if key not in sd:
                    if is_main_process:
                        print(f"[proj-energy] WARN: {key} not in model.named_parameters(); skipping")
                    continue
                self.targets.append({
                    "layer": L, "mod": mod, "key": key,
                    "param": sd[key],
                    "U0": None, "Vt0": None, "W0": None,
                })

        if is_main_process:
            print(f"[proj-energy] caching W0, U0/V0 for {len(self.targets)} (layer, module) pairs ...")
        # SVD pretrained weights once; cache W0 + relevant factors on CPU
        with torch.no_grad():
            for t in self.targets:
                W0_gpu = t["param"].detach().to(self.device, dtype=self.svd_dtype)
                U0, _, Vt0 = torch.linalg.svd(W0_gpu, full_matrices=False)
                if t["mod"] in _PROJ_UPROJ_MODULES:
                    t["U0"] = U0.detach().cpu().contiguous()
                if t["mod"] in _PROJ_VPROJ_MODULES:
                    t["Vt0"] = Vt0.detach().cpu().contiguous()
                # Cache W0 in svd_dtype on CPU so dW = W' - W0 stays exact across steps
                t["W0"] = W0_gpu.detach().cpu().contiguous()
                del W0_gpu, U0, Vt0
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if is_main_process:
            with open(self.jsonl_path, "w") as f:
                pass
            print(f"[proj-energy] writing live records to {self.jsonl_path}")

    def _proj_pair(self, X: torch.Tensor, U0_gpu, Vt0_gpu) -> tuple[float, float]:
        """Return (u_proj, v_proj) for matrix X. NaN where the factor is None."""
        den = torch.linalg.norm(X)
        if float(den) == 0.0:
            return float("nan"), float("nan")
        u = float(torch.linalg.norm(U0_gpu.T @ X) / den) if U0_gpu is not None else float("nan")
        v = float(torch.linalg.norm(X @ Vt0_gpu.T) / den) if Vt0_gpu is not None else float("nan")
        return u, v

    @torch.no_grad()
    def record(self, step: int):
        """Compute u/v_proj of W', G, dW at the current state and append to JSONL.

        Must be called BEFORE optimizer.zero_grad() so that .grad is still set.
        """
        if not self.is_main_process:
            return
        row = {"step": int(step)}
        for t in self.targets:
            W_cur = t["param"].detach().to(self.device, dtype=self.svd_dtype)
            G_cur = t["param"].grad
            G_cur = G_cur.detach().to(self.device, dtype=self.svd_dtype) if G_cur is not None else None

            U0_gpu = t["U0"].to(self.device, non_blocking=True) if t["U0"] is not None else None
            Vt0_gpu = t["Vt0"].to(self.device, non_blocking=True) if t["Vt0"] is not None else None

            # W' (current weight)
            uW, vW = self._proj_pair(W_cur, U0_gpu, Vt0_gpu)
            # dW = W' - W0  (W0 cached in same dtype on CPU)
            W0_gpu = t["W0"].to(self.device, non_blocking=True)
            dW = W_cur - W0_gpu
            udW, vdW = self._proj_pair(dW, U0_gpu, Vt0_gpu)
            del W0_gpu, dW
            # G (per-step gradient at this optimizer step)
            if G_cur is not None:
                uG, vG = self._proj_pair(G_cur, U0_gpu, Vt0_gpu)
            else:
                uG, vG = float("nan"), float("nan")

            tag = f"L{t['layer']}.{t['mod']}"
            if t["U0"] is not None:
                row[f"{tag}.W.u_proj"] = uW
                row[f"{tag}.G.u_proj"] = uG
                row[f"{tag}.dW.u_proj"] = udW
            if t["Vt0"] is not None:
                row[f"{tag}.W.v_proj"] = vW
                row[f"{tag}.G.v_proj"] = vG
                row[f"{tag}.dW.v_proj"] = vdW

            del W_cur, G_cur, U0_gpu, Vt0_gpu

        self.records.append(row)
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    def finalize(self):
        """Write CSV and PNG plot from accumulated records. Main process only."""
        if not self.is_main_process:
            return
        if not self.records:
            print("[proj-energy] no records to finalize; skipping plot")
            return

        records = []
        with open(self.jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        if not records:
            return

        all_keys = set()
        for r in records:
            all_keys.update(r.keys())
        all_keys.discard("step")
        cols = ["step"] + sorted(all_keys)

        with open(self.csv_path, "w") as f:
            f.write(",".join(cols) + "\n")
            for r in records:
                f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
        print(f"[proj-energy] wrote {self.csv_path}")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"[proj-energy] matplotlib unavailable ({e}); skipping plot")
            return

        steps = [r["step"] for r in records]
        # Group keys by (proj_type, quantity). Key format: L{N}.{mod}.{Q}.{u|v}_proj
        def _filter(proj, quantity):
            suffix = f".{quantity}.{proj}_proj"
            return sorted(k for k in all_keys if k.endswith(suffix))

        fig, axes = plt.subplots(2, 3, figsize=(14, 6.5), constrained_layout=True)
        col_titles = {"W": "W' (current weight)", "G": "G (per-step gradient)", "dW": "ΔW = W' − W₀"}
        row_titles = {"u": "U-projection ‖U₀ᵀX‖_F / ‖X‖_F",
                       "v": "V-projection ‖XV₀‖_F / ‖X‖_F"}

        for r_idx, proj in enumerate(("u", "v")):
            for c_idx, q in enumerate(("W", "G", "dW")):
                ax = axes[r_idx, c_idx]
                keys = _filter(proj, q)
                for k in keys:
                    ys = [r.get(k, float("nan")) for r in records]
                    label = ".".join(k.split(".")[:2])  # e.g. L0.gate_proj
                    ax.plot(steps, ys, marker=".", markersize=3, linewidth=1.0, label=label)
                if r_idx == 0:
                    ax.set_title(col_titles[q], fontsize=10)
                if c_idx == 0:
                    ax.set_ylabel(row_titles[proj], fontsize=9)
                ax.set_xlabel("Step")
                ax.grid(True, alpha=0.2, linewidth=0.4)
                if keys:
                    ax.legend(loc="best", fontsize=6, framealpha=0.9)

        fig.savefig(self.png_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[proj-energy] wrote {self.png_path}")


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
        "--no_grad",
        type=str,
        default='none',
        help="sparse fine-tuning tuner",
    )

    parser.add_argument(
        "--mask_type",
        type=str,
        default='weight_filtered_mag_abs_largest',
        help="sparse fine-tuning tuner",
    )

    parser.add_argument(
        "--update_interval",
        type=int,
        default=200,
        help="sparse fine-tuning tuner",
    )

    parser.add_argument(
        "--lora_rank",
        type=int,
        default=128,
        help="sparse fine-tuning tuner",
    )

    parser.add_argument(
        "--filter_rank",
        type=int,
        default=128,
        help="sparse fine-tuning tuner",
    )

    parser.add_argument(
        "--use_flash_attn",
        type=str,
        default='False',
        help="sparse fine-tuning tuner",
    )

    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="sparse fine-tuning tuner",
    )

    parser.add_argument(
        "--adapter_name",
        type=str,
        default='none',
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

    # --- Projection-energy recorder ---
    parser.add_argument(
        "--record_proj_energy",
        action="store_true",
        help="Record U/V projection energy of the live weight against the pretrained "
             "U0/V0 during training. Writes proj_energy.jsonl (live) and proj_energy.{csv,png} "
             "(at end) under --output_dir.",
    )
    parser.add_argument(
        "--proj_record_layers",
        type=str,
        default="0,15,31",
        help="Comma-separated layer indices to record (default: 0,15,31).",
    )
    parser.add_argument(
        "--proj_record_modules",
        type=str,
        default="gate_proj,up_proj,down_proj",
        help="Comma-separated module names. Only u-proj for gate/up_proj and v-proj "
             "for down_proj are informative on Llama-3-8B; other modules are trivial-1.",
    )
    parser.add_argument(
        "--proj_record_interval",
        type=int,
        default=0,
        help="Optimizer-step interval between projection-energy records "
             "(default 0 = use --logging_steps).",
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
        model_kwargs = {"dtype": torch.bfloat16}
        base_kwargs = dict(
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            **model_kwargs,
        )

        if args.use_flash_attn == "True":
            # Transformers switched from `use_flash_attention_2` to
            # `attn_implementation`. Try modern API first, then legacy fallback.
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    attn_implementation="flash_attention_2",
                    **base_kwargs,
                )
            except TypeError:
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        args.model_name_or_path,
                        use_flash_attention_2=True,
                        **base_kwargs,
                    )
                except TypeError:
                    print(
                        "Warning: flash-attention flags are unsupported in this "
                        "transformers version; loading model without explicit FA2 flag."
                    )
                    model = AutoModelForCausalLM.from_pretrained(
                        args.model_name_or_path,
                        **base_kwargs,
                    )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                **base_kwargs,
            )
    else:
        # model = AutoModelForCausalLM.from_config(config)
        model = AutoModelForCausalLM.from_config(config)

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id

    model.resize_token_embeddings(int(
        8 *
        math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8
        
    if args.gradient_checkpointing:
        model = make_model_gradient_checkpointing_compatible(model)
        model.gradient_checkpointing_enable()

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

    if 'sparse' in args.peft_tuner:
        layer_dict = {
            'q': 'q_proj',
            'k': 'k_proj',
            'v': 'v_proj',
            'o': 'o_proj',
            'gate': 'gate_proj',
            'up': 'up_proj',
            'down': 'down_proj'
        }
        no_grad_params = ["bias", "layernorm", "norm"]
        for char, val in layer_dict.items():
            if char == 'o':
                char = 'o_'
            if char in args.no_grad:
                no_grad_params.append(val)
        if 'head' in args.no_grad:
            no_grad_params += ['lm_head', "embed_tokens"]
        for name, param in model.named_parameters():
            if any([s in name for s in no_grad_params]):
                param.requires_grad = False

    for name, param in model.named_parameters():
        if not param.requires_grad:
            # print(f'param {name} is not trainable')
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

    ## Init optimizer
    if 'sparse' in args.peft_tuner:
        from sparseAdam import SparseAdamW
        optimizer = SparseAdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, betas=(0.9, 0.95), mask_type=args.mask_type, args=args
        )
    else:
        # Prepare optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
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

    _method = "full" if getattr(args, "peft_tuner", None) in (None, "", "none") else args.peft_tuner
    _rank = int(args.lora_rank) if hasattr(args, "lora_rank") and args.lora_rank else None
    sysmon = SysMon(
        out_dir=args.output_dir or ".",
        method=_method,
        rank=_rank,
        base_params=sum(p.numel() for p in model.parameters()),
    )

    proj_recorder = None
    if args.record_proj_energy:
        if args.output_dir is None:
            print("[proj-energy] WARN: --output_dir not set; disabling recorder")
        else:
            try:
                proj_layers = [int(x) for x in args.proj_record_layers.split(",") if x.strip()]
                proj_modules = [m.strip() for m in args.proj_record_modules.split(",") if m.strip()]
                proj_recorder = ProjEnergyRecorder(
                    model=accelerator.unwrap_model(model),
                    layer_indices=proj_layers,
                    module_names=proj_modules,
                    output_dir=args.output_dir,
                    device=accelerator.device,
                    is_main_process=accelerator.is_main_process,
                )
                # Record step 0 (initial state, should give 1.0)
                proj_recorder.record(step=0)
            except Exception as e:
                print(f"[proj-energy] init failed ({e}); recorder disabled")
                proj_recorder = None
    proj_interval = args.proj_record_interval if args.proj_record_interval > 0 else args.logging_steps

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
                _t0 = time.time()
                optimizer.step()
                lr_scheduler.step()
                # Record proj-energy BEFORE zero_grad so .grad still holds G.
                if (
                    proj_recorder is not None
                    and proj_interval > 0
                    and accelerator.sync_gradients
                    and (args.completed_steps + 1) % proj_interval == 0
                ):
                    proj_recorder.record(step=args.completed_steps + 1)
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                sysmon.record_step(time.time() - _t0)

                total_loss += loss.detach().float()

            if accelerator.sync_gradients:
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
                    accelerator.print(f"Epoch {epoch+1} Step {args.completed_steps}: Eval perplexity = {perplexity:.4f}, Eval loss = {eval_loss:.4f}")
                    
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        if accelerator.is_main_process and args.output_dir:
                            accelerator.wait_for_everyone()
                            unwrapped_model = accelerator.unwrap_model(model)
                            best_model = copy.deepcopy(unwrapped_model).to("cpu")
                            print("New best model")

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
        },
    )

    if proj_recorder is not None:
        # Record final step too (if not already)
        if not proj_recorder.records or proj_recorder.records[-1]["step"] != args.completed_steps:
            try:
                proj_recorder.record(step=args.completed_steps)
            except Exception as e:
                print(f"[proj-energy] final record failed ({e})")
        try:
            proj_recorder.finalize()
        except Exception as e:
            print(f"[proj-energy] finalize failed ({e})")

    # --- Save policy: write last/ always, best/ if best-tracking ran.
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
        save_hf_format(last_model, tokenizer, args, sub_folder="last")
        print_rank_0(f"Saved last-step checkpoint to {os.path.join(args.output_dir, 'last')}", args.global_rank)

        if best_model is not None:
            save_hf_format(best_model, tokenizer, args, sub_folder="best")
            print_rank_0(
                f"Saved best-eval checkpoint to {os.path.join(args.output_dir, 'best')} "
                f"(val_loss={best_eval_loss:.4f})",
                args.global_rank,
            )

    if use_wandb:
        accelerator.end_training()


if __name__ == "__main__":
    main()
