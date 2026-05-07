"""
Unified RL training entrypoint.

Examples:
  uv run run_rl.py --train-mode full --lr 1e-5 --no-wandb
  uv run run_rl.py --train-mode full --optimizer muon --lr 1e-5 --no-wandb
  uv run run_rl.py --train-mode lora --lr 1e-4 --lora-rank 1 --trainable-type all --no-wandb
  uv run run_rl.py --train-mode lora_full --lr 1e-4 --lora-rank 1 --trainable-type all --no-wandb
  uv run run_rl.py --train-mode blocktt --lr 1e-4 --trainable-type all --decomp-mode input_one_block --train-position small --no-wandb
  uv run run_rl.py --train-mode svd --lr 1e-4 --trainable-type all --train-position output --no-wandb
"""

import argparse
import json
import math
import os
import random
import sys
import time
from functools import partial
from pathlib import Path

import requests
import torch
import wandb
from safetensors.torch import save_file as save_safetensors_file
# Make the shared `fura/` package (sibling of this `rl/` directory) importable.
import os as _os
import sys as _sys
_REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

from fura.btt_layer import (
    BTTLayer,
    BLOCKTT_DECOMP_GROUP_TO_MODULES,
    configure_blocktt_trainability,
    convert_linear_to_btt,
    get_blocktt_target_module_names,
    normalize_trainable_blocktt_cores_,
    resolve_blocktt_decomp_modes,
)
from datasets import load_dataset
from fura.math_utils import is_equiv, last_boxed_only_string, remove_boxed
from fura.svd_layer import (
    SVDLayer,
    configure_svd_trainability,
    convert_linear_to_svd,
    get_svd_target_module_names,
)

# Optional: Muon optimizer. Not required for the open-source RL recipes
# (default --optimizer adamw). Stub out if unavailable.
try:
    from optim.muon import Muon  # type: ignore
except ImportError:
    class Muon:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "Muon optimizer is not bundled in this open-source release. "
                "Use --optimizer adamw to reproduce the paper's default RL setup."
            )

# Optional: calibrated-BTT (QFuRA on RL) integration. The QFuRA RL recipe is not
# part of the open-source release; only commonsense QFuRA is provided. Stub out
# the calibration helpers so plain BlockTT/SVD/LoRA RL still works.
try:
    from compress_integration import (  # type: ignore
        add_calibrated_btt_args,
        validate_calibrated_btt_args,
        apply_calibrated_btt,
        build_calib_loader,
        materialize_calibrated_btt_weights,
        save_calibrated_btt_hf_pretrained,
    )
    from compress.btt.btt_linear import BTTLinear  # type: ignore
    _HAS_CALIB_BTT = True
except ImportError:
    _HAS_CALIB_BTT = False

    def add_calibrated_btt_args(parser, *, hyphen_style: bool = True):
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

    def validate_calibrated_btt_args(args, *, argv=None, hyphen_style=True):
        if getattr(args, "calib_mode", "none") not in (None, "none"):
            raise RuntimeError(
                "Calibrated BTT (QFuRA-RL) is not included in this open-source release. "
                "See `qfura_commonsense/` for the QFuRA commonsense reproduction."
            )

    def apply_calibrated_btt(*args, **kwargs):
        raise RuntimeError("Calibrated BTT not available in open-source build.")

    def build_calib_loader(*args, **kwargs):
        raise RuntimeError("Calibrated BTT not available in open-source build.")

    def materialize_calibrated_btt_weights(*args, **kwargs):
        raise RuntimeError("Calibrated BTT not available in open-source build.")

    def save_calibrated_btt_hf_pretrained(*args, **kwargs):
        raise RuntimeError("Calibrated BTT not available in open-source build.")

    class BTTLinear:  # type: ignore
        """Placeholder so isinstance checks in mode-export paths are inert."""
        pass
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from transformers.utils import is_flash_attn_2_available

MODE_DEFAULTS = {
    "full": {
        "lr": 1e-5,
        "wandb_project": "math_grpo_full",
        "micro_batch_size": 4,
        "gradient_accumulation_steps": 64,
    },
    "lora": {
        "lr": 9e-5,
        "wandb_project": "math-grpo",
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 128,
    },
    "lora_full": {
        "lr": 9e-5,
        "wandb_project": "math-grpo-lora-full",
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 128,
    },
    "blocktt": {
        "lr": 9e-5,
        "wandb_project": "math-grpo-blocktt",
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 128,
    },
    "svd": {
        "lr": 9e-5,
        "wandb_project": "math-grpo-svd",
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 128,
    },
    "dora":     {"lr": 9e-5, "wandb_project": "math-grpo-dora",
                 "micro_batch_size": 2, "gradient_accumulation_steps": 128},
    "pissa":    {"lr": 9e-5, "wandb_project": "math-grpo-pissa",
                 "micro_batch_size": 2, "gradient_accumulation_steps": 128},
    "milora":   {"lr": 9e-5, "wandb_project": "math-grpo-milora",
                 "micro_batch_size": 2, "gradient_accumulation_steps": 128},
    "randlora": {"lr": 9e-5, "wandb_project": "math-grpo-randlora",
                 "micro_batch_size": 2, "gradient_accumulation_steps": 128},
    "lift":     {"lr": 1e-4, "wandb_project": "math-grpo-lift",
                 "micro_batch_size": 4, "gradient_accumulation_steps": 64},
}


def resolve_attn_implementation():
    if torch.cuda.is_available() and is_flash_attn_2_available():
        return "flash_attention_2"
    return "sdpa"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Unified RL training script")
    parser.add_argument(
        "--train-mode",
        type=str,
        required=True,
        choices=["full", "lora", "lora_full", "dora", "pissa", "milora", "randlora",
                 "lift", "blocktt", "svd"],
        help="Training mode: full, lora, lora_full, dora, pissa, milora, randlora, lift, blocktt, or svd",
    )

    # Shared configuration
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="HuggingFace model ID to use",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default depends on train mode)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "adamw8bit", "muon"],
        help=(
            "Optimizer to use: adamw, adamw8bit (bitsandbytes 8-bit AdamW), or muon "
            "(default: adamw). adamw8bit cuts optimizer memory ~4x — useful for "
            "single-GPU full-FT of 7B+ models."
        ),
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable HF gradient checkpointing on the model. Saves activation memory "
            "(~5x for full-FT) at the cost of ~30%% slower training. Default: disabled."
        ),
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="none",
        choices=["none", "linear", "cosine"],
        help="Learning rate scheduler to use (default: none)",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.0,
        help="Warmup ratio in [0, 1] over total update steps (default: 0.0)",
    )
    parser.add_argument(
        "--cycle-length",
        type=int,
        default=None,
        help="Cycle length for cosine scheduler (default: full training length)",
    )
    parser.add_argument(
        "--min-lr-ratio",
        type=float,
        default=0.1,
        help="Minimum LR ratio for cosine scheduler (default: 0.1)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay for optimizer (default: 0.0)",
    )
    parser.add_argument(
        "--lr-adam",
        type=float,
        default=None,
        help="Muon-only: AdamW fallback learning rate (default: inherit --lr)",
    )
    parser.add_argument(
        "--lr-embedding",
        type=float,
        default=None,
        help="Muon-only: embedding AdamW learning rate (default: inherit AdamW lr)",
    )
    parser.add_argument(
        "--norm-method",
        type=str,
        default=None,
        choices=["row", "col", "row-col", "col-row", "shape"],
        help="Muon-only: optional update normalization method",
    )
    parser.add_argument(
        "--n-grpo-steps",
        type=int,
        default=50,
        help="Number of GRPO training steps",
    )
    parser.add_argument(
        "--n-prompts-per-step",
        type=int,
        default=32,
        help="Number of prompts to sample per step",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=8,
        help="Number of rollouts per prompt",
    )
    parser.add_argument(
        "--epochs-per-step",
        type=int,
        default=1,
        help="Number of epochs to train per step",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=None,
        help="Micro batch size (default depends on train mode)",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="Gradient accumulation steps (default depends on train mode)",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="./runs",
        help="Base directory for saving runs",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="boxed.prompt",
        help="Path to prompt template file",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help="vLLM max_model_len for local rollout backends (default: 2048)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.4,
        help="vLLM GPU memory utilization fraction for local rollout backends (default: 0.4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--enable-save-ckpt",
        action="store_true",
        help="Save checkpoints at step 1, step 10, and final step (default: disabled)",
    )
    parser.add_argument(
        "--save-best-val-ckpt",
        action="store_true",
        help="Save a merged HF checkpoint to <run_dir>/best/ whenever eval/accuracy improves.",
    )
    parser.add_argument(
        "--save-first-step-grads-path",
        type=str,
        default=None,
        help=(
            "Optional path to save raw gradients at the first optimizer step "
            "(safetensors)."
        ),
    )
    parser.add_argument(
        "--save-first-step-grads-prefixes",
        type=str,
        default=None,
        help=(
            "Comma-separated parameter-name prefixes to include when saving first-step "
            "grads. If omitted, all trainable grads are saved."
        ),
    )
    parser.add_argument(
        "--stop-after-first-step",
        action="store_true",
        help="Stop training immediately after the first optimizer step.",
    )
    parser.add_argument(
        "--save-grads-steps",
        type=str,
        default=None,
        help=(
            "Comma-separated optimizer steps to save gradients "
            "(e.g. '0,10,30'). Step 0 = before first update. Default: disabled."
        ),
    )

    # LoRA-family args (lora / lora_full)
    parser.add_argument(
        "--lora-rank", type=int, default=1, help="LoRA rank (default: 1)"
    )
    parser.add_argument(
        "--vllm-url",
        type=str,
        default="http://localhost:8000",
        help="URL for vLLM API server (lora mode only; lora_full uses local in-process rollout)",
    )
    parser.add_argument(
        "--randlora-projection-prng-key",
        type=int,
        default=0,
        help="Seed for RandLoRA's shared random bases (default: 0). "
             "Only valid when --train-mode randlora.",
    )
    parser.add_argument(
        "--lift-lora-rank",
        type=int,
        default=128,
        help="LIFT: rank used for the low-rank approximation that drives "
             "mask selection (default: 128). Only valid when --train-mode lift.",
    )
    parser.add_argument(
        "--lift-filter-rank",
        type=int,
        default=128,
        help="LIFT: filter rank for mask selection (default: 128). "
             "Only valid when --train-mode lift.",
    )
    parser.add_argument(
        "--lift-update-interval",
        type=int,
        default=400,
        help="LIFT: optimizer steps between mask recomputations (default: 400). "
             "Only valid when --train-mode lift.",
    )

    # BlockTT-only args
    parser.add_argument(
        "--decomp-mode",
        type=str,
        default="input_one_block",
        help=(
            "BlockTT decomposition mode: scalar input_one_block|output_one_block "
            "or dict literal with keys qkv,o,mlp_upgate,mlp_down "
            "(values: input|output|input_one_block|output_one_block)"
        ),
    )
    parser.add_argument(
        "--blocktt-rank",
        type=str,
        default="full",
        help="BTT rank; default full for lossless decomposition",
    )
    parser.add_argument(
        "--convert-mode",
        type=str,
        default="svd",
        choices=["svd", "qr"],
        help=(
            "Per-block decomposition for BlockTT init. 'svd' (default) gives "
            "U S V^T factors; 'qr' uses LQ when a>=b and QR when a<b so the "
            "small (k x k) factor is always orthogonal. 'qr' has no singular "
            "values and ignores --s-merged-to."
        ),
    )
    parser.add_argument(
        "--no-train-bias",
        action="store_true",
        help="Freeze BTT biases; by default biases are trainable",
    )
    parser.add_argument(
        "--blocktt-normalize-after-update",
        action="store_true",
        help="Normalize trainable BTT cores after each optimizer update",
    )
    parser.add_argument(
        "--blocktt-factorize-by-head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Align attention layer BTT blocks with head structure (default: enabled)",
    )

    # Shared trainable module selector for LoRA/BlockTT/SVD
    parser.add_argument(
        "--trainable-type",
        type=str,
        default="all",
        choices=["all", "mlp", "attn"],
        help="Trainable target modules: all, mlp, or attn (default: all)",
    )
    parser.add_argument(
        "--train-position",
        type=str,
        default=None,
        choices=["output", "input", "small", "large", "both"],
        help="Trainable side selector: svd uses output|input, blocktt uses small|large|both",
    )
    parser.add_argument(
        "--s-merged-to",
        type=str,
        default=None,
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
            "Where to merge S during SVD init for svd/blocktt: "
            "frozen, trainable, output, input, split, keep_frozen, or keep_trainable"
        ),
    )

    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="W&B project name (default depends on train mode)",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B run name (default: mode-specific auto-generated name)",
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")

    # Merged-checkpoint and post-training math-verify eval flags
    parser.add_argument(
        "--enable-merged-ckpt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Save checkpoints in plain HF format (LoRA merged into base, "
            "BlockTT/SVD materialized to nn.Linear). Default: enabled."
        ),
    )
    parser.add_argument(
        "--enable-math-verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After training, evaluate the final checkpoint on math reasoning "
            "benchmarks via the math-verify library. Default: enabled."
        ),
    )
    parser.add_argument(
        "--math-verify-datasets",
        type=str,
        default="MATH-500,AIME-24,AIME-25,AMC23",
        help=(
            "Comma-separated list of eval dataset names. "
            "Default: MATH-500,AIME-24,AIME-25,AMC23. "
            "Minerva is supported via the registry but excluded from the default "
            "because its 272-problem score has shown high run-to-run variance "
            "that masks method differences; pass it explicitly to opt in."
        ),
    )
    parser.add_argument(
        "--math-verify-n-samples",
        type=int,
        default=None,
        help=(
            "Override per-dataset n_samples. Default: registry per-dataset "
            "(8 for AIME-24/25, 1 for the rest)."
        ),
    )
    parser.add_argument(
        "--math-verify-temperature",
        type=float,
        default=None,
        help=(
            "Override per-dataset sampling temperature. Default: registry "
            "per-dataset (0.6 for AIME-24/25, 0.0 for the rest)."
        ),
    )
    parser.add_argument(
        "--math-verify-max-tokens",
        type=int,
        default=2048,
        help="Max tokens per eval generation. Default: 2048.",
    )

    add_calibrated_btt_args(parser, hyphen_style=True)
    args = parser.parse_args(argv)
    args.math_verify_datasets = [
        s.strip() for s in args.math_verify_datasets.split(",") if s.strip()
    ]
    return args


def apply_mode_defaults(args):
    defaults = MODE_DEFAULTS[args.train_mode]
    if args.lr is None:
        args.lr = defaults["lr"]
    if args.wandb_project is None:
        args.wandb_project = defaults["wandb_project"]
    if args.micro_batch_size is None:
        args.micro_batch_size = defaults["micro_batch_size"]
    if args.gradient_accumulation_steps is None:
        args.gradient_accumulation_steps = defaults["gradient_accumulation_steps"]
    if args.train_mode == "blocktt" and args.train_position is None:
        args.train_position = "small"
    if args.train_mode == "svd" and args.train_position is None:
        args.train_position = "output"
    if args.train_mode in {"blocktt", "svd"} and args.s_merged_to is None:
        if args.train_position == "both":
            args.s_merged_to = "split"
        else:
            args.s_merged_to = "frozen"


def require_cuda_for_structured_conversion(train_mode, entrypoint):
    if train_mode not in {"blocktt", "svd"}:
        return
    if torch.cuda.is_available():
        return
    raise RuntimeError(
        f"{entrypoint}: --train-mode {train_mode} requires CUDA so SVD/BTT conversion "
        "runs on GPU. No CUDA device is available."
    )


def get_local_cuda_device_id():
    return int(os.environ.get("LOCAL_RANK", "0"))


def _flag_was_passed(argv, flag_name):
    return any(
        token == flag_name or token.startswith(f"{flag_name}=") for token in argv
    )


def validate_mode_specific_flags(args, argv):
    LORA_FAMILY = {"lora", "lora_full", "dora", "pissa", "milora", "randlora"}
    LORA_HTTP_OK = {"lora", "dora", "randlora"}  # vllm-url valid for these only

    mode_to_flag_sets = {
        "lora_family_shared":   ["--lora-rank"],
        "lora_http":            ["--vllm-url"],
        "randlora":             ["--randlora-projection-prng-key"],
        "lift":                 ["--lift-lora-rank", "--lift-filter-rank",
                                 "--lift-update-interval"],
        "blocktt": [
            "--decomp-mode", "--blocktt-rank", "--no-train-bias",
            "--blocktt-normalize-after-update",
            "--blocktt-factorize-by-head", "--no-blocktt-factorize-by-head",
            "--convert-mode",
        ],
        "svd": [],
    }

    # --lora-rank is only valid for the lora family
    if args.train_mode not in LORA_FAMILY:
        passed = [f for f in mode_to_flag_sets["lora_family_shared"] if _flag_was_passed(argv, f)]
        if passed:
            raise ValueError(
                f"{', '.join(passed)} is only valid when --train-mode is one of: "
                f"{sorted(LORA_FAMILY)}"
            )

    # --vllm-url is only valid for lora/dora/randlora
    if args.train_mode not in LORA_HTTP_OK:
        passed = [f for f in mode_to_flag_sets["lora_http"] if _flag_was_passed(argv, f)]
        if passed:
            raise ValueError(
                f"{', '.join(passed)} is only valid when --train-mode is one of: "
                f"{sorted(LORA_HTTP_OK)}"
            )

    # --randlora-projection-prng-key is randlora-only
    if args.train_mode != "randlora":
        passed = [f for f in mode_to_flag_sets["randlora"] if _flag_was_passed(argv, f)]
        if passed:
            raise ValueError(f"{', '.join(passed)} is only valid when --train-mode randlora")

    # --lift-* is lift-only
    if args.train_mode != "lift":
        passed = [f for f in mode_to_flag_sets["lift"] if _flag_was_passed(argv, f)]
        if passed:
            raise ValueError(f"{', '.join(passed)} is only valid when --train-mode lift")

    # --optimizer muon is incompatible with lift
    if args.train_mode == "lift" and args.optimizer == "muon":
        raise ValueError(
            "--optimizer muon is incompatible with --train-mode lift. "
            "LIFT supplies its own SparseAdamW optimizer."
        )

    # --- existing blocktt/svd/full guards below ---
    if args.train_mode != "blocktt":
        passed = [f for f in mode_to_flag_sets["blocktt"] if _flag_was_passed(argv, f)]
        if passed:
            raise ValueError(
                f"{', '.join(passed)} is only valid when --train-mode blocktt"
            )
    else:
        blocktt_targets = get_blocktt_target_module_names(args.trainable_type)
        decomp_mode, module_decomp_modes = resolve_blocktt_decomp_modes(
            args.decomp_mode,
            include_names=blocktt_targets,
            default_mode="input_one_block",
        )
        args.decomp_mode = decomp_mode
        args.blocktt_module_decomp_modes = module_decomp_modes
        args.decomp_mode_display = format_blocktt_decomp_mode(decomp_mode)

    if args.train_mode == "full" and _flag_was_passed(argv, "--trainable-type"):
        raise ValueError(
            "--trainable-type is only valid when --train-mode "
            "lora, lora_full, dora, pissa, milora, randlora, blocktt, or svd"
        )

    train_position_passed = _flag_was_passed(argv, "--train-position")
    if args.train_mode in ({"full", "lift"} | LORA_FAMILY) and train_position_passed:
        raise ValueError("--train-position is only valid when --train-mode blocktt or svd")
    if args.train_mode == "blocktt" and train_position_passed:
        if args.train_position not in {"small", "large", "both"}:
            raise ValueError("--train-position for blocktt must be one of: small, large, both")
    if args.train_mode == "svd" and train_position_passed:
        if args.train_position not in {"output", "input", "both"}:
            raise ValueError("--train-position for svd must be one of: output, input, both")

    s_merged_to_passed = _flag_was_passed(argv, "--s-merged-to")
    if args.train_mode in ({"full", "lift"} | LORA_FAMILY) and s_merged_to_passed:
        raise ValueError("--s-merged-to is only valid when --train-mode blocktt or svd")
    if (
        args.train_mode == "blocktt"
        and s_merged_to_passed
        and args.train_position == "both"
        and args.s_merged_to in {"frozen", "trainable"}
    ):
        raise ValueError(
            "--s-merged-to frozen/trainable is invalid when blocktt --train-position is both; "
            "use 'split' or 'keep_trainable'"
        )
    if (
        args.train_mode == "blocktt"
        and getattr(args, "convert_mode", "svd") == "qr"
        and args.s_merged_to in {"keep_frozen", "keep_trainable"}
    ):
        raise ValueError(
            "--convert-mode qr is incompatible with --s-merged-to keep_frozen/keep_trainable: "
            "QR has no singular values to keep."
        )

    # Math-verify validation
    from eval_datasets import REGISTRY as _MV_REGISTRY

    unknown = [d for d in args.math_verify_datasets if d not in _MV_REGISTRY]
    if unknown:
        known = sorted(_MV_REGISTRY.keys())
        raise ValueError(
            f"Unknown --math-verify-datasets entries: {unknown}. "
            f"Known names: {known}"
        )
    if args.math_verify_n_samples is not None and args.math_verify_n_samples <= 0:
        raise ValueError(
            f"--math-verify-n-samples must be > 0, got {args.math_verify_n_samples}"
        )
    if args.math_verify_max_tokens <= 0:
        raise ValueError(
            f"--math-verify-max-tokens must be > 0, got {args.math_verify_max_tokens}"
        )

    if (args.enable_math_verify and not args.enable_merged_ckpt
            and args.train_mode not in {"full", "lift"}):
        import sys as _sys

        print(
            "WARNING: --enable-math-verify with --no-enable-merged-ckpt may fail at "
            "eval time for non-full modes (the saved checkpoint isn't loadable by "
            "vanilla AutoModelForCausalLM.from_pretrained).",
            file=_sys.stderr,
        )

    validate_calibrated_btt_args(args, argv=argv, hyphen_style=True)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def resolve_blocktt_rank(rank_arg: str):
    if rank_arg == "full":
        return "full"
    try:
        rank = int(rank_arg)
    except ValueError as exc:
        raise ValueError("--blocktt-rank must be 'full' or a positive integer") from exc
    if rank <= 0:
        raise ValueError("--blocktt-rank must be > 0")
    return rank


def format_blocktt_decomp_mode(decomp_mode):
    if isinstance(decomp_mode, str):
        return decomp_mode
    if isinstance(decomp_mode, dict):
        order = tuple(BLOCKTT_DECOMP_GROUP_TO_MODULES.keys())
        return ",".join(f"{group}={decomp_mode[group]}" for group in order)
    return str(decomp_mode)


def get_lora_target_modules(trainable_type):
    if trainable_type == "all":
        return [
            "gate_proj",
            "up_proj",
            "down_proj",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ]
    if trainable_type == "mlp":
        return ["gate_proj", "up_proj", "down_proj"]
    return ["q_proj", "k_proj", "v_proj", "o_proj"]


@torch.no_grad()
def materialize_btt_weight(layer: BTTLayer) -> torch.Tensor:
    if getattr(layer, "lr_act", False):
        raise ValueError("BlockTT rollout export requires lr_act=False")
    return layer.materialize_dense_weight()


@torch.no_grad()
def materialize_svd_weight(layer: SVDLayer) -> torch.Tensor:
    return layer.materialize_dense_weight()


@torch.no_grad()
def export_weights_for_vllm(model: torch.nn.Module):
    factorized_module_names = []
    weight_tuples = []

    for module_name, module in model.named_modules():
        if isinstance(module, BTTLayer):
            factorized_module_names.append(module_name)
            dense_weight = materialize_btt_weight(module)
            weight_tuples.append((f"{module_name}.weight", dense_weight))
            if module.bias is not None:
                weight_tuples.append((f"{module_name}.bias", module.bias))
        elif isinstance(module, SVDLayer):
            factorized_module_names.append(module_name)
            dense_weight = materialize_svd_weight(module)
            weight_tuples.append((f"{module_name}.weight", dense_weight))
            if module.bias is not None:
                weight_tuples.append((f"{module_name}.bias", module.bias))
        elif isinstance(module, BTTLinear):
            factorized_module_names.append(module_name)
            dense_weight = module.materialize_dense_weight()
            weight_tuples.append((f"{module_name}.weight", dense_weight))
            if module.bias is not None:
                weight_tuples.append((f"{module_name}.bias", module.bias))

    if factorized_module_names:
        factorized_prefixes = tuple(f"{name}." for name in factorized_module_names)
    else:
        factorized_prefixes = ()

    for name, param in model.named_parameters():
        if factorized_prefixes and name.startswith(factorized_prefixes):
            continue
        weight_tuples.append((name, param))

    return weight_tuples


def load_datasets_and_tokenizer(model_id, prompt_template_path):
    train_dataset = load_dataset("qwedsacf/competition_math", split="train[:7500]")
    val_dataset = load_dataset("qwedsacf/competition_math", split="train[-5000:]")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    with open(prompt_template_path, "r", encoding="utf-8") as f:
        template = f.read().strip()

    has_chat_template = tokenizer.chat_template is not None

    def process_data(example):
        with_template = template.replace("{question}", example["problem"])
        if has_chat_template:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": with_template}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = with_template
        answer = remove_boxed(last_boxed_only_string(example["solution"]))
        return {"prompt": prompt, "answer": answer}

    train_dataset = train_dataset.map(process_data)
    val_dataset = val_dataset.map(process_data)
    return train_dataset, val_dataset, tokenizer


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    prompt_t = [tokenizer.encode(p) for p in prompt_strs]
    output_t = [tokenizer.encode(o) for o in output_strs]
    full = []
    max_len = 0

    for i in range(len(prompt_t)):
        row_len = len(prompt_t[i]) + len(output_t[i])
        max_len = max(max_len, row_len)

    for i in range(len(prompt_t)):
        padding_size = max_len - len(prompt_t[i]) - len(output_t[i])
        padding = [tokenizer.pad_token_id] * padding_size
        row = torch.tensor(prompt_t[i] + output_t[i] + padding, dtype=torch.long)
        full.append(row.unsqueeze(0))

    f2 = torch.cat(full)
    input_ids = f2[:, :-1]
    labels = f2[:, 1:]

    response_mask = torch.zeros(len(prompt_strs), max_len - 1)
    for i in range(len(prompt_t)):
        response_mask[
            i, len(prompt_t[i]) - 1 : len(prompt_t[i]) + len(output_t[i]) - 1
        ] = 1

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask.bool(),
    }


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    logits = model(input_ids).logits
    z = logits - logits.max(dim=-1, keepdim=True).values
    denom = torch.exp(z).sum(dim=-1, keepdim=True)
    logprobs = z - torch.log(denom)
    return torch.gather(logprobs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


def compute_run_name(args, mode_info: dict) -> str:
    """Compute a human-readable run name (used for both directory and W&B)."""
    if args.wandb_run_name is not None:
        return args.wandb_run_name
    if args.train_mode == "full":
        return f"{args.model_id}_{args.lr:.1e}_full"
    if args.train_mode == "lora":
        return f"{args.model_id}_{args.lr:.1e}_r{args.lora_rank}"
    if args.train_mode == "lora_full":
        return f"{args.model_id}_{args.lr:.1e}_r{args.lora_rank}_lora_full"
    if args.train_mode == "dora":
        return f"{args.model_id}_{args.lr:.1e}_r{args.lora_rank}_dora"
    if args.train_mode == "pissa":
        return f"{args.model_id}_{args.lr:.1e}_r{args.lora_rank}_pissa"
    if args.train_mode == "milora":
        return f"{args.model_id}_{args.lr:.1e}_r{args.lora_rank}_milora"
    if args.train_mode == "randlora":
        return f"{args.model_id}_{args.lr:.1e}_r{args.lora_rank}_randlora"
    if args.train_mode == "lift":
        return (
            f"{args.model_id}_{args.lr:.1e}_r{args.lift_lora_rank}"
            f"_int{args.lift_update_interval}_lift"
        )
    if args.train_mode == "blocktt":
        decomp_mode_name = mode_info.get("decomp_mode_display", args.decomp_mode)
        return f"{args.model_id}_{args.lr:.1e}_{decomp_mode_name}_{args.train_position}_{args.trainable_type}"
    return f"{args.model_id}_{args.lr:.1e}_{args.train_position}_{args.trainable_type}"


def create_run_dir(base_dir: str, train_mode: str, run_name: str, model_id: str) -> str:
    """Create run directory at runs/{model_name}/{train_mode}/{run_name}-{timestamp}."""
    timestamp = time.strftime("%m%d-%H%M%S")
    model_name = model_id.split("/")[-1]
    run_dir = os.path.join(base_dir, model_name, train_mode, f"{run_name}-{timestamp}")
    os.makedirs(run_dir)
    return run_dir


def should_save_checkpoint(step_num: int, total_steps: int) -> bool:
    return step_num == 10 or step_num == 30 or step_num == total_steps


def _build_factored_dense_state_dict(model):
    """Return a state_dict where every BTTLayer / SVDLayer is replaced by a
    dense `{prefix}.weight` (and `{prefix}.bias` if present) materialized from
    the factored cores. Other parameters are kept verbatim. The model object
    is never mutated.
    """
    factored_prefixes = []
    extra = {}

    for module_name, module in model.named_modules():
        if isinstance(module, BTTLayer):
            factored_prefixes.append(module_name + ".")
            dense = module.materialize_dense_weight().detach().clone()
            extra[f"{module_name}.weight"] = dense
            if module.bias is not None:
                extra[f"{module_name}.bias"] = module.bias.detach().clone()
        elif isinstance(module, SVDLayer):
            factored_prefixes.append(module_name + ".")
            dense = module.materialize_dense_weight().detach().clone()
            extra[f"{module_name}.weight"] = dense
            if module.bias is not None:
                extra[f"{module_name}.bias"] = module.bias.detach().clone()

    new_sd = {}
    for name, tensor in model.state_dict().items():
        if any(name.startswith(p) for p in factored_prefixes):
            continue
        new_sd[name] = tensor
    new_sd.update(extra)
    return new_sd


def save_merged_checkpoint(model, tokenizer, ckpt_dir: str, train_mode: str, args):
    """Save model in plain HuggingFace format. The on-disk result contains
    only nn.Linear layers (no LoRA adapters, no BTT/SVD factored cores).
    The in-memory model object is never mutated; training can resume.
    """
    os.makedirs(ckpt_dir, exist_ok=True)

    if train_mode in {"full", "lift"}:
        model.save_pretrained(ckpt_dir)
    elif train_mode in {"lora", "lora_full", "dora", "pissa", "milora", "randlora"}:
        model.merge_adapter()
        try:
            base = model.get_base_model()
            # RandLoRA shares the randlora_A projection tensors across layers;
            # safetensors refuses to serialize shared storages, so fall back
            # to the non-safetensors path for randlora. The merged base
            # weights are what downstream eval loads from disk.
            if train_mode == "randlora":
                base.save_pretrained(ckpt_dir, safe_serialization=False)
            else:
                base.save_pretrained(ckpt_dir)
        finally:
            model.unmerge_adapter()
    elif train_mode in {"blocktt", "svd"}:
        if train_mode == "blocktt" and getattr(args, "calib_mode", "none") != "none":
            save_calibrated_btt_hf_pretrained(model, ckpt_dir)
        else:
            state_dict = _build_factored_dense_state_dict(model)
            model.save_pretrained(ckpt_dir, state_dict=state_dict)
    else:
        raise ValueError(f"Unknown train_mode for save_merged_checkpoint: {train_mode}")

    tokenizer.save_pretrained(ckpt_dir)


def save_checkpoint(model, tokenizer, run_dir: str, step_num: int, args=None):
    ckpt_dir = os.path.join(run_dir, f"step={step_num}")
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Saving checkpoint to {ckpt_dir}")
    if args is not None and getattr(args, "enable_merged_ckpt", True):
        save_merged_checkpoint(model, tokenizer, ckpt_dir, args.train_mode, args)
    elif args is not None and getattr(args, "calib_mode", "none") != "none":
        save_calibrated_btt_hf_pretrained(model, ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
    else:
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
    print(f"Checkpoint saved to {ckpt_dir}")


def parse_save_grads_steps(s):
    """Parse comma-separated step numbers, or return empty set if None."""
    if s is None:
        return set()
    return {int(x.strip()) for x in s.split(",")}


def save_gradients(trainable_named_params, base_dir, step, subdir="step"):
    """Save all trainable gradients to {base_dir}/{subdir}={step}/grads.safetensors."""
    grad_payload = {}
    for name, p in trainable_named_params:
        if p.grad is None:
            continue
        grad_payload[name] = p.grad.detach().cpu()
    if not grad_payload:
        print(f"Warning: no gradients to save at step {step}")
        return
    ckpt_dir = os.path.join(base_dir, f"{subdir}={step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    save_safetensors_file(grad_payload, os.path.join(ckpt_dir, "grads.safetensors"))
    print(f"Saved gradients ({len(grad_payload)} tensors) to {ckpt_dir}")


def maybe_init_wandb(args, run_dir, run_name, mode_info, num_training_steps):
    if args.no_wandb:
        return

    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "train_mode": args.train_mode,
            "model_id": args.model_id,
            "learning_rate": args.lr,
            "optimizer": args.optimizer,
            "lr_scheduler": args.lr_scheduler,
            "warmup_ratio": args.warmup_ratio,
            "warmup_steps": resolve_warmup_steps(args.warmup_ratio, num_training_steps),
            "cycle_length": args.cycle_length,
            "min_lr_ratio": args.min_lr_ratio,
            "num_training_steps": num_training_steps,
            "weight_decay": args.weight_decay,
            "lr_adam": args.lr_adam,
            "lr_embedding": args.lr_embedding,
            "norm_method": args.norm_method,
            "n_grpo_steps": args.n_grpo_steps,
            "n_prompts_per_step": args.n_prompts_per_step,
            "group_size": args.group_size,
            "epochs_per_step": args.epochs_per_step,
            "micro_batch_size": args.micro_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "seed": args.seed,
            "prompt_template": args.prompt_template,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enable_save_ckpt": args.enable_save_ckpt,
            "run_dir": run_dir,
            **mode_info,
        },
        dir=run_dir,
    )


def is_vllm_http_available(vllm_url: str, timeout_sec: float = 2.0) -> bool:
    try:
        response = requests.get(f"{vllm_url.rstrip('/')}/v1/models", timeout=timeout_sec)
    except requests.RequestException:
        return False
    return response.ok


def resolve_lora_rollout_backend(train_mode: str, vllm_url: str) -> str | None:
    if train_mode == "lora_full":
        return "local_inproc"
    if train_mode in {"pissa", "milora"}:
        # PiSSA/MiLoRA modify the base weight at init; the running vLLM HTTP
        # server is out of sync. Force local in-process rollout so we
        # merge_adapter() and push the materialized dense weights into vLLM.
        return "local_inproc"
    if train_mode in {"lora", "dora", "randlora"}:
        return "http" if is_vllm_http_available(vllm_url) else "local_inproc"
    return None


def normalize_lora_merged_weight_name(name: str) -> str | None:
    # Skip adapter-only tensors; local vLLM wants dense base-model parameter names.
    lora_markers = (
        ".lora_A.",
        ".lora_B.",
        ".lora_embedding_A.",
        ".lora_embedding_B.",
        ".lora_magnitude_vector",
        ".randlora_lambda",
        ".randlora_gamma",
        ".randlora_m",
    )
    if any(marker in name for marker in lora_markers):
        return None
    return name.replace(".base_layer.", ".")


@torch.no_grad()
def export_lora_merged_weights_for_vllm(model):
    """Return dense base-model weights with the active PEFT adapter merged.

    The tensors must be cloned before unmerge_adapter() runs. Otherwise the
    returned tuples hold references to the base parameters, which PEFT mutates
    back to the unmerged values in-place.
    """
    model.merge_adapter()
    try:
        base_model = model.get_base_model()
        weight_tuples = []
        seen = set()
        for name, param in base_model.named_parameters():
            normalized = normalize_lora_merged_weight_name(name)
            if normalized is None:
                continue
            if normalized in seen:
                raise RuntimeError(f"Duplicate normalized LoRA weight name: {normalized}")
            seen.add(normalized)
            weight_tuples.append((normalized, param.detach().clone()))
        return weight_tuples
    finally:
        model.unmerge_adapter()


def build_lora_http_generators(args, model, run_dir):
    loaded_loras = []

    def generate_http(prompts: list[str], vllm_model_id: str, temperature=0, responses_per_prompt=1):
        api_url = f"{args.vllm_url}/v1/completions"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": vllm_model_id,
            "prompt": prompts,
            "max_tokens": 1024,
            "temperature": temperature,
            "n": responses_per_prompt,
        }
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return [choice["text"] for choice in result["choices"]]

    def load_lora(lora_name):
        if lora_name in loaded_loras:
            return
        api_url = f"{args.vllm_url}/v1/load_lora_adapter"
        headers = {"Content-Type": "application/json"}
        lora_path = str(Path.cwd() / lora_name)
        payload = {"lora_name": lora_name, "lora_path": lora_path}
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        loaded_loras.append(lora_name)

    def save_lora(step):
        lora_name = f"{run_dir}/lora_adapters/step={step}"
        os.makedirs(os.path.dirname(lora_name), exist_ok=True)
        if not os.path.exists(lora_name):
            model.save_pretrained(lora_name)
        return lora_name

    def generate_for_train(prompts: list[str], step: int):
        vllm_model_id = save_lora(step)
        load_lora(vllm_model_id)
        return generate_http(
            prompts,
            vllm_model_id=vllm_model_id,
            temperature=1,
            responses_per_prompt=args.group_size,
        )

    def generate_for_eval(prompts: list[str], step: int):
        vllm_model_id = save_lora(step)
        load_lora(vllm_model_id)
        return generate_http(
            prompts,
            vllm_model_id=vllm_model_id,
            temperature=0,
            responses_per_prompt=1,
        )

    return generate_for_train, generate_for_eval, None


def build_lora_local_generators(args, model):
    os.environ["VLLM_USE_V1"] = "0"
    from vllm import LLM, SamplingParams

    if not all(hasattr(model, attr) for attr in ("merge_adapter", "unmerge_adapter", "get_base_model")):
        raise RuntimeError(
            "Local LoRA fallback requires PEFT model methods: "
            "merge_adapter, unmerge_adapter, get_base_model."
        )

    vllm_model = LLM(
        model=args.model_id,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=4096,
        logprobs_mode="processed_logprobs",
    )

    def generate(prompts: list[str], temperature=0, responses_per_prompt=1):
        weight_tuples = export_lora_merged_weights_for_vllm(model)
        vllm_internal_model = (
            vllm_model.llm_engine.model_executor.driver_worker.model_runner.model
        )
        vllm_internal_model.load_weights(weight_tuples)

        sampling_params = SamplingParams(
            max_tokens=1024,
            temperature=temperature,
            n=responses_per_prompt,
        )
        outputs = vllm_model.generate(prompts, sampling_params)
        return [o.text for output in outputs for o in output.outputs]

    def generate_for_train(prompts: list[str], _step: int):
        return generate(
            prompts,
            temperature=1,
            responses_per_prompt=args.group_size,
        )

    def generate_for_eval(prompts: list[str], _step: int):
        return generate(prompts, temperature=0, responses_per_prompt=1)

    return generate_for_train, generate_for_eval, vllm_model


def build_local_vllm_generators(args, model):
    os.environ["VLLM_USE_V1"] = "0"
    from vllm import LLM, SamplingParams

    vllm_model = LLM(
        model=args.model_id,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=4096,
        logprobs_mode="processed_logprobs",
    )

    def generate(prompts: list[str], temperature=0, responses_per_prompt=1):
        if args.train_mode in {"blocktt", "svd"}:
            weight_tuples = export_weights_for_vllm(model)
        else:
            weight_tuples = [(name, param) for name, param in model.named_parameters()]

        vllm_internal_model = (
            vllm_model.llm_engine.model_executor.driver_worker.model_runner.model
        )
        vllm_internal_model.load_weights(weight_tuples)

        sampling_params = SamplingParams(
            max_tokens=1024,
            temperature=temperature,
            n=responses_per_prompt,
        )
        outputs = vllm_model.generate(prompts, sampling_params)
        return [o.text for output in outputs for o in output.outputs]

    def generate_for_train(prompts: list[str], _step: int):
        return generate(
            prompts,
            temperature=1,
            responses_per_prompt=args.group_size,
        )

    def generate_for_eval(prompts: list[str], _step: int):
        return generate(prompts, temperature=0, responses_per_prompt=1)

    return generate_for_train, generate_for_eval, vllm_model


def validate_trainable_params(trainable_params):
    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters found.")
    if any(not p.requires_grad for p in trainable_params):
        raise ValueError("Found non-trainable parameter in optimizer parameter list.")
    if len({id(p) for p in trainable_params}) != len(trainable_params):
        raise ValueError("Duplicate trainable parameters found.")


def _flatten_param_ids(params_by_group):
    return {id(p) for group in params_by_group for p in group}


def assert_muon_routing(train_mode, trainable_named_params, optimizer):
    if not isinstance(optimizer, Muon):
        return

    matrix_ids = _flatten_param_ids(optimizer._bucket_muon_matrix_by_group)
    cola_ids = _flatten_param_ids(optimizer._bucket_muon_cola_a_by_group)
    cola_ids |= _flatten_param_ids(optimizer._bucket_muon_cola_b_by_group)
    cola_ids |= _flatten_param_ids(optimizer._bucket_muon_cola_g_by_group)
    btt_ids = _flatten_param_ids(optimizer._bucket_btt_by_group)

    def _param_ids_by_suffix(suffixes):
        ids = set()
        for name, p in trainable_named_params:
            if any(s in name for s in suffixes):
                ids.add(id(p))
        return ids

    if train_mode == "svd":
        svd_ids = _param_ids_by_suffix((".svd_a", ".svd_b"))
        if len(svd_ids) == 0:
            raise ValueError(
                "Muon routing check failed: no trainable SVD factors were found."
            )
        missing = sorted(pid for pid in svd_ids if pid not in matrix_ids)
        if missing:
            raise ValueError(
                "Muon routing check failed: some SVD factors are not in matrix-Muon buckets."
            )
        if svd_ids & cola_ids:
            raise ValueError(
                "Muon routing check failed: SVD factors were routed to CoLA buckets."
            )
        return

    if train_mode == "blocktt":
        btt_trainable_ids = _param_ids_by_suffix((".btt_l", ".btt_r"))
        if len(btt_trainable_ids) == 0:
            raise ValueError(
                "Muon routing check failed: no trainable BlockTT factors were found."
            )
        missing = sorted(pid for pid in btt_trainable_ids if pid not in btt_ids)
        if missing:
            raise ValueError(
                "Muon routing check failed: some BlockTT factors are not in BTT-Muon buckets."
            )
        return

    if train_mode in {"lora", "lora_full", "dora", "pissa", "milora", "randlora"}:
        lora_ids = _param_ids_by_suffix(("lora_A", "lora_B"))
        if len(lora_ids) == 0:
            raise ValueError(
                "Muon routing check failed: no trainable LoRA factors were found."
            )
        missing = sorted(pid for pid in lora_ids if pid not in matrix_ids)
        if missing:
            raise ValueError(
                "Muon routing check failed: some LoRA factors are not in matrix-Muon buckets."
            )
        return

    if train_mode == "full":
        if len(matrix_ids) == 0:
            raise ValueError(
                "Muon routing check failed: no matrix-Muon parameters were detected in full mode."
            )


def build_optimizer(args, trainable_params, trainable_named_params):
    if args.train_mode == "lift":
        if args.optimizer == "muon":
            raise ValueError(
                "--train-mode lift is incompatible with --optimizer muon. "
                "LIFT supplies its own SparseAdamW optimizer."
            )
        from optim.sparse_adam import SparseAdamW
        import torch.nn as nn
        model = getattr(args, "_lift_model", None)
        if model is None:
            raise RuntimeError(
                "LIFT optimizer requires args._lift_model to be set by prepare_model."
            )

        weights_with_mask, decay_ids = [], []
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and "lm_head" not in name and mod.weight.requires_grad:
                weights_with_mask.append(mod.weight)
                decay_ids.append(id(mod.weight))
        decay_id_set = set(decay_ids)
        other_decay, other_nodecay = [], []
        no_decay_names: tuple[str, ...] = ()
        for name, p in model.named_parameters():
            if not p.requires_grad or id(p) in decay_id_set:
                continue
            if any(nd in name for nd in no_decay_names):
                other_nodecay.append(p)
            else:
                other_decay.append(p)

        param_groups = [
            {
                "params": weights_with_mask, "weight_decay": 0.0,
                "rank": args.lift_lora_rank, "filter_rank": args.lift_filter_rank,
                "update_proj_gap": args.lift_update_interval,
                "group_name": "weights_with_mask",
            },
            {"params": other_decay,    "weight_decay": 0.0, "group_name": "other_params_w_decay"},
            {"params": other_nodecay,  "weight_decay": 0.0, "group_name": "other_params"},
        ]
        return SparseAdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

    if args.optimizer == "adamw":
        return torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    if args.optimizer == "adamw8bit":
        from bitsandbytes.optim import AdamW8bit
        return AdamW8bit(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    if args.optimizer == "muon":
        return Muon(
            trainable_named_params,
            lr=args.lr,
            lr_adam=args.lr_adam,
            lr_embedding=args.lr_embedding,
            weight_decay=args.weight_decay,
            norm_method=args.norm_method,
        )
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def _get_cyclical_cosine_schedule_with_min_lr_lambda(
    current_step,
    *,
    num_warmup_steps,
    cycle_length,
    min_lr_ratio,
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))

    progress = float((current_step - num_warmup_steps) % cycle_length) / float(
        max(1, cycle_length - num_warmup_steps)
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def get_cyclical_cosine_schedule_with_min_lr(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    cycle_length,
    min_lr_ratio=0.1,
    last_epoch=-1,
):
    assert (
        cycle_length is not None or num_training_steps is not None
    ), "You must specify either cycle_length or num_training_steps"

    if cycle_length is None:
        cycle_length = num_training_steps

    if num_training_steps % cycle_length != 0:
        raise ValueError(
            f"num_training_steps ({num_training_steps}) must be divisible by cycle_length ({cycle_length})"
        )

    lr_lambda = partial(
        _get_cyclical_cosine_schedule_with_min_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        cycle_length=cycle_length,
        min_lr_ratio=min_lr_ratio,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def build_lr_scheduler(args, optimizer, num_training_steps):
    warmup_steps = resolve_warmup_steps(args.warmup_ratio, num_training_steps)

    if args.lr_scheduler == "none":
        return None

    if num_training_steps <= 0:
        raise ValueError(
            f"num_training_steps must be > 0 when using lr scheduler, got {num_training_steps}"
        )

    if args.lr_scheduler == "linear":
        return transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            last_epoch=-1,
        )
    if args.lr_scheduler == "cosine":
        return get_cyclical_cosine_schedule_with_min_lr(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            cycle_length=args.cycle_length,
            min_lr_ratio=args.min_lr_ratio,
            last_epoch=-1,
        )
    raise ValueError(f"Unsupported lr scheduler: {args.lr_scheduler}")


def compute_num_training_steps(args):
    if args.micro_batch_size <= 0:
        raise ValueError("--micro-batch-size must be > 0")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be > 0")

    samples_per_step = args.n_prompts_per_step * args.group_size
    micro_batches_per_epoch = samples_per_step // args.micro_batch_size
    optimizer_steps_per_epoch = (
        micro_batches_per_epoch // args.gradient_accumulation_steps
    )
    return args.n_grpo_steps * args.epochs_per_step * optimizer_steps_per_epoch


def resolve_warmup_steps(warmup_ratio, num_training_steps):
    if not 0.0 <= warmup_ratio <= 1.0:
        raise ValueError(f"--warmup-ratio must be in [0, 1], got {warmup_ratio}")
    if num_training_steps < 0:
        raise ValueError(f"num_training_steps must be >= 0, got {num_training_steps}")
    return int(math.ceil(num_training_steps * warmup_ratio))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    validate_mode_specific_flags(args, argv)
    apply_mode_defaults(args)
    require_cuda_for_structured_conversion(args.train_mode, entrypoint="run_rl.py")
    if args.max_model_len <= 0:
        raise ValueError("--max-model-len must be > 0.")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        raise ValueError("--gpu-memory-utilization must be in (0, 1].")
    grad_prefixes = None
    if args.save_first_step_grads_prefixes is not None:
        grad_prefixes = [
            p.strip()
            for p in args.save_first_step_grads_prefixes.split(",")
            if p.strip()
        ]

    set_seed(args.seed)
    lora_rollout_backend = resolve_lora_rollout_backend(args.train_mode, args.vllm_url)

    mode_info = {}
    if args.train_mode in {"lora", "lora_full", "dora", "pissa", "milora", "randlora"}:
        lora_target_modules = get_lora_target_modules(args.trainable_type)
        mode_info.update(
            {
                "lora_rank": args.lora_rank,
                "lora_full_backbone_trainable": args.train_mode == "lora_full",
                "trainable_type": args.trainable_type,
                "target_modules": lora_target_modules,
                "vllm_url": args.vllm_url,
                "rollout_backend": lora_rollout_backend,
            }
        )
        if args.train_mode == "randlora":
            mode_info["randlora_projection_prng_key"] = args.randlora_projection_prng_key
    elif args.train_mode == "lift":
        mode_info.update({
            "lift_lora_rank": args.lift_lora_rank,
            "lift_filter_rank": args.lift_filter_rank,
            "lift_update_interval": args.lift_update_interval,
        })
    elif args.train_mode == "blocktt":
        blocktt_rank = resolve_blocktt_rank(args.blocktt_rank)
        blocktt_targets = get_blocktt_target_module_names(args.trainable_type)
        train_bias = not args.no_train_bias
        train_position = args.train_position
        decomp_mode = args.decomp_mode
        decomp_mode_display = getattr(
            args, "decomp_mode_display", format_blocktt_decomp_mode(decomp_mode)
        )
        module_decomp_modes = getattr(args, "blocktt_module_decomp_modes", None)
        if not isinstance(decomp_mode, dict):
            module_decomp_modes = None
        mode_info.update(
            {
                "blocktt_rank": blocktt_rank,
                "trainable_type": args.trainable_type,
                "decomp_mode": decomp_mode,
                "decomp_mode_by_module": module_decomp_modes,
                "decomp_mode_display": decomp_mode_display,
                "train_position": train_position,
                "s_merged_to": args.s_merged_to,
                "target_modules": blocktt_targets,
                "train_bias": train_bias,
                "blocktt_normalize_after_update": args.blocktt_normalize_after_update,
                "blocktt_factorize_by_head": args.blocktt_factorize_by_head,
                "convert_mode": getattr(args, "convert_mode", "svd"),
            }
        )
    elif args.train_mode == "svd":
        svd_targets = get_svd_target_module_names(args.trainable_type)
        mode_info.update(
            {
                "trainable_type": args.trainable_type,
                "train_position": args.train_position,
                "s_merged_to": args.s_merged_to,
                "target_modules": svd_targets,
            }
        )

    print("Training configuration:")
    print(f"  Train mode: {args.train_mode}")
    print(f"  Model ID: {args.model_id}")
    print(f"  Optimizer: {args.optimizer}")
    print(f"  Gradient checkpointing: {args.gradient_checkpointing}")
    print(f"  Learning rate: {args.lr}")
    print(f"  vLLM max model len: {args.max_model_len}")
    print(f"  vLLM GPU memory utilization: {args.gpu_memory_utilization}")
    print(f"  LR scheduler: {args.lr_scheduler}")
    print(f"  Warmup ratio: {args.warmup_ratio}")
    if args.lr_scheduler == "cosine":
        print(f"  Cycle length: {args.cycle_length}")
        print(f"  Min LR ratio: {args.min_lr_ratio}")
    print(f"  Weight decay: {args.weight_decay}")
    if args.optimizer == "muon":
        print(f"  Muon lr_adam: {args.lr_adam}")
        print(f"  Muon lr_embedding: {args.lr_embedding}")
        print(f"  Muon norm_method: {args.norm_method}")
    print(f"  GRPO steps: {args.n_grpo_steps}")
    print(f"  Prompts per step: {args.n_prompts_per_step}")
    print(f"  Group size: {args.group_size}")
    print(f"  Epochs per step: {args.epochs_per_step}")
    print(f"  Micro batch size: {args.micro_batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    if args.train_mode in {"lora", "lora_full", "dora", "pissa", "milora", "randlora"}:
        print(f"  LoRA rank: {args.lora_rank}")
        print(f"  Trainable type: {args.trainable_type}")
        print(
            "  Backbone trainable: "
            f"{'yes' if args.train_mode == 'lora_full' else 'no'}"
        )
        print(f"  Target modules: {mode_info['target_modules']}")
        print(f"  Rollout backend: {lora_rollout_backend}")
        print(f"  vLLM URL: {args.vllm_url}")
    if args.train_mode == "lift":
        print(f"  LIFT lora_rank: {args.lift_lora_rank}")
        print(f"  LIFT filter_rank: {args.lift_filter_rank}")
        print(f"  LIFT update_interval: {args.lift_update_interval}")
    if args.train_mode == "blocktt":
        print(f"  BlockTT rank: {mode_info['blocktt_rank']}")
        print(f"  Trainable type: {args.trainable_type}")
        print(f"  Decomp mode: {mode_info['decomp_mode_display']}")
        print(f"  Train position: {mode_info['train_position']}")
        print(f"  S merged to: {mode_info['s_merged_to']}")
        print(f"  Train BTT bias: {mode_info['train_bias']}")
        print(
            "  Normalize BTT after update: "
            f"{mode_info['blocktt_normalize_after_update']}"
        )
        print(f"  Factorize by head: {mode_info['blocktt_factorize_by_head']}")
        print(f"  Target modules: {mode_info['target_modules']}")
    if args.train_mode == "svd":
        print(f"  Trainable type: {args.trainable_type}")
        print(f"  Train position: {args.train_position}")
        print(f"  S merged to: {mode_info['s_merged_to']}")
        print(f"  Target modules: {mode_info['target_modules']}")
    print(f"  Save checkpoint: {'enabled' if args.enable_save_ckpt else 'disabled'}")
    print(f"  W&B logging: {'enabled' if not args.no_wandb else 'disabled'}")
    print()

    num_training_steps = compute_num_training_steps(args)
    warmup_steps = resolve_warmup_steps(args.warmup_ratio, num_training_steps)
    print(f"  Optimizer update steps: {num_training_steps}")
    print(f"  Warmup steps (derived): {warmup_steps}")

    run_name = compute_run_name(args, mode_info)
    run_dir = create_run_dir(args.base_dir, args.train_mode, run_name, args.model_id)
    print(f"Created: {run_dir}")

    maybe_init_wandb(args, run_dir, run_name, mode_info, num_training_steps)

    train_dataset, val_dataset, tokenizer = load_datasets_and_tokenizer(
        args.model_id,
        args.prompt_template,
    )

    device_id = get_local_cuda_device_id()
    device = f"cuda:{device_id}"
    attn_implementation = resolve_attn_implementation()
    print(f"  Attention backend: {attn_implementation}")
    model_kwargs = dict(
        attn_implementation=attn_implementation,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        device_map={"": device_id},
    )
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)

    if args.train_mode in {"lora", "lora_full"}:
        from peft import LoraConfig, get_peft_model

        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=32,
            target_modules=mode_info["target_modules"],
        )
        model = get_peft_model(model, peft_config)
        if args.train_mode == "lora_full":
            for p in model.parameters():
                p.requires_grad = True
        model.print_trainable_parameters()
        trainable_params = [p for p in model.parameters() if p.requires_grad]
    elif args.train_mode == "dora":
        from peft import LoraConfig, get_peft_model
        peft_config = LoraConfig(
            r=args.lora_rank, lora_alpha=32,
            target_modules=mode_info["target_modules"],
            use_dora=True,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        trainable_params = [p for p in model.parameters() if p.requires_grad]
    elif args.train_mode == "pissa":
        from peft import LoraConfig, get_peft_model
        peft_config = LoraConfig(
            r=args.lora_rank, lora_alpha=32,
            target_modules=mode_info["target_modules"],
            init_lora_weights="pissa_niter_4", lora_dropout=0,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        trainable_params = [p for p in model.parameters() if p.requires_grad]
    elif args.train_mode == "milora":
        from peft import LoraConfig, get_peft_model
        from run_sft import apply_milora_init_
        peft_config = LoraConfig(
            r=args.lora_rank, lora_alpha=32,
            target_modules=mode_info["target_modules"],
            lora_dropout=0,
        )
        model = get_peft_model(model, peft_config)
        apply_milora_init_(model, rank=args.lora_rank)
        model.print_trainable_parameters()
        trainable_params = [p for p in model.parameters() if p.requires_grad]
    elif args.train_mode == "randlora":
        from peft import RandLoraConfig, get_peft_model
        peft_config = RandLoraConfig(
            r=args.lora_rank, randlora_alpha=32,
            target_modules=mode_info["target_modules"],
            projection_prng_key=args.randlora_projection_prng_key,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        trainable_params = [p for p in model.parameters() if p.requires_grad]
    elif args.train_mode == "lift":
        # Dense model; no PEFT wrapping. trainable = everything.
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        args._lift_model = model
    elif args.train_mode == "blocktt":
        if getattr(args, "calib_mode", "none") != "none":
            # Calibrated BTT branch: run rollout on dense base model, then decompose
            def _rl_calib_rollout(n):
                from random import Random
                rng = Random(int(args.calib_seed))
                indices = list(range(len(train_dataset)))
                rng.shuffle(indices)
                pairs = []
                model.eval()
                for idx in indices[:n]:
                    prompt = train_dataset[idx]["prompt"]
                    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                    with torch.no_grad():
                        out = model.generate(
                            **inputs,
                            max_new_tokens=1024,
                            temperature=1.0,
                            do_sample=True,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                    full = tokenizer.decode(out[0], skip_special_tokens=True)
                    completion = full[len(prompt):] if full.startswith(prompt) else full
                    pairs.append((prompt, completion))
                model.train()
                return pairs

            calib_loader = build_calib_loader(
                args,
                tokenizer=tokenizer,
                rl_rollout_fn=_rl_calib_rollout,
                hyphen_style=True,
            )
            model, calib_stats = apply_calibrated_btt(
                model, args, calib_loader=calib_loader, hyphen_style=True,
            )
            print(f"[calib-btt] installed {calib_stats.get('num_btt_layers', 0)} BTT layers")
            trainable_params = [p for p in model.parameters() if p.requires_grad]
        else:
            decomp_mode_for_convert = (
                mode_info["decomp_mode_by_module"]
                if mode_info["decomp_mode_by_module"] is not None
                else mode_info["decomp_mode"]
            )
            converted_modules = convert_linear_to_btt(
                model,
                btt_rank=mode_info["blocktt_rank"],
                decomp_mode=decomp_mode_for_convert,
                init_mode="default",
                include_names=mode_info["target_modules"],
                skip_names=("lm_head",),
                lr_act=False,
                s_merged_to=mode_info["s_merged_to"],
                train_position=mode_info["train_position"],
                factorize_by_head=mode_info.get("blocktt_factorize_by_head", False),
                model_config=model.config,
                convert_mode=mode_info.get("convert_mode", "svd"),
            )
            stats = configure_blocktt_trainability(
                model,
                train_bias=mode_info["train_bias"],
                train_position=mode_info["train_position"],
                train_singular_values=(mode_info["s_merged_to"] == "keep_trainable"),
            )
            if stats["num_btt_layers"] == 0:
                raise ValueError(
                    "No layers were converted to BTT; check --trainable-type selection."
                )
            trainable_params = stats["trainable_params"]
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
    elif args.train_mode == "svd":
        converted_modules = convert_linear_to_svd(
            model,
            include_names=mode_info["target_modules"],
            skip_names=("lm_head",),
            s_merged_to=mode_info["s_merged_to"],
            train_position=args.train_position,
        )
        stats = configure_svd_trainability(
            model,
            train_position=args.train_position,
            train_bias=True,
            train_embed_lm_head=(args.train_position == "both"),
            train_singular_values=(mode_info["s_merged_to"] == "keep_trainable"),
        )
        if stats["num_svd_layers"] == 0:
            raise ValueError("No layers were converted to SVD; check --trainable-type selection.")
        trainable_params = stats["trainable_params"]
        print(f"Converted modules: {len(converted_modules)}")
        print(
            f"Trainable params: {stats['trainable_param_count']:,} / "
            f"{stats['total_param_count']:,} "
            f"({100 * stats['trainable_param_count'] / stats['total_param_count']:.4f}%)"
        )
        print(
            f"Tuned factors: output={stats['tuned_output_cores']}, "
            f"input={stats['tuned_input_cores']}, biases={stats['tuned_biases']}"
        )
    else:
        trainable_params = list(model.parameters())

    trainable_named_params = [
        (name, p) for name, p in model.named_parameters() if p.requires_grad
    ]
    trainable_params = [p for _, p in trainable_named_params]
    validate_trainable_params(trainable_params)

    if args.gradient_checkpointing:
        # PEFT wrappers intercept gradient_checkpointing_enable; reach the base model.
        base_for_gc = getattr(model, "base_model", None)
        gc_target = base_for_gc if base_for_gc is not None and hasattr(base_for_gc, "gradient_checkpointing_enable") else model
        gc_target.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        print(f"  Gradient checkpointing: enabled (use_reentrant=False) on {type(gc_target).__name__}")

    if args.train_mode in {"lora", "lora_full", "dora", "pissa", "milora", "randlora"}:
        if lora_rollout_backend == "http":
            generate_for_train, generate_for_eval, in_process_llm = build_lora_http_generators(
                args,
                model,
                run_dir,
            )
        else:
            generate_for_train, generate_for_eval, in_process_llm = build_lora_local_generators(
                args,
                model,
            )
    else:
        generate_for_train, generate_for_eval, in_process_llm = build_local_vllm_generators(
            args, model,
        )

    optimizer = build_optimizer(args, trainable_params, trainable_named_params)
    if args.optimizer == "muon":
        assert_muon_routing(args.train_mode, trainable_named_params, optimizer)
    scheduler = build_lr_scheduler(args, optimizer, num_training_steps)
    normalize_blocktt_after_update = (
        args.train_mode == "blocktt" and args.blocktt_normalize_after_update
    )
    first_step_grads_saved = False
    optimizer_steps = 0
    save_grads_steps = parse_save_grads_steps(args.save_grads_steps)
    stop_requested = False

    best_val_state = {"acc": float("-inf"), "step": -1}

    def eval_model(step):
        val_prompts = val_dataset[:1000]["prompt"]

        eval_start = time.time()
        outputs = generate_for_eval(val_prompts, step)
        eval_time = time.time() - eval_start

        correct = 0
        for i in range(len(outputs)):
            correct_answer = val_dataset[i]["answer"]
            generated_answer = remove_boxed(last_boxed_only_string(outputs[i]))
            if generated_answer is not None and is_equiv(generated_answer, correct_answer):
                correct += 1

        accuracy = correct / len(outputs)
        print(f"step={step}, correct: {correct} / {len(outputs)} ({accuracy:.2%})")

        if not args.no_wandb:
            log_payload = {
                "eval/accuracy": accuracy,
                "eval/correct": correct,
                "eval/total": len(outputs),
                "eval/time_seconds": eval_time,
            }
            if step == 0:
                log_payload["eval/baseline_accuracy"] = accuracy
                wandb.run.summary["eval/baseline_accuracy"] = accuracy
            wandb.log(log_payload, step=step)

        if args.save_best_val_ckpt and accuracy > best_val_state["acc"]:
            best_val_state["acc"] = accuracy
            best_val_state["step"] = step
            best_dir = os.path.join(run_dir, "best")
            os.makedirs(best_dir, exist_ok=True)
            try:
                save_merged_checkpoint(model, tokenizer, best_dir, args.train_mode, args)
                meta = {"step": step, "eval_accuracy": accuracy}
                with open(os.path.join(best_dir, "best_val_info.json"), "w") as fh:
                    json.dump(meta, fh, indent=2)
                print(
                    f"[save-best-val-ckpt] step={step} new best eval/accuracy={accuracy:.4f} -> {best_dir}"
                )
                if not args.no_wandb:
                    wandb.run.summary["best_val/accuracy"] = accuracy
                    wandb.run.summary["best_val/step"] = step
            except Exception as exc:
                print(f"[save-best-val-ckpt] WARNING: save failed at step={step}: {exc!r}")

        return accuracy

    print("Pre-training baseline evaluation (step=0, before any optimizer update)...")
    model = model.to(device)
    baseline_accuracy = eval_model(0)
    print(f"Pre-training baseline accuracy: {baseline_accuracy:.2%}")
    model.train()

    for i in range(args.n_grpo_steps):
        step_start_time = time.time()

        sample_indices = random.sample(
            range(0, len(train_dataset)),
            args.n_prompts_per_step,
        )
        batch = train_dataset[sample_indices]

        generation_start = time.time()
        outputs = generate_for_train(batch["prompt"], i)
        generation_time = time.time() - generation_start

        generated_answers = [remove_boxed(last_boxed_only_string(o)) for o in outputs]
        raw_reward = [
            ans is not None and is_equiv(ans, batch["answer"][j // args.group_size])
            for j, ans in enumerate(generated_answers)
        ]
        raw_reward_tensor = torch.tensor(raw_reward, dtype=torch.float).reshape(
            (args.n_prompts_per_step, args.group_size)
        )
        means = raw_reward_tensor.mean(dim=-1).unsqueeze(1)
        advantages = (raw_reward_tensor - means).reshape(
            (args.n_prompts_per_step * args.group_size,)
        )

        train_accuracy = raw_reward_tensor.mean().item()
        reward_std = raw_reward_tensor.std().item()
        reward_max = raw_reward_tensor.max().item()
        reward_min = raw_reward_tensor.min().item()
        advantage_mean = advantages.mean().item()
        advantage_std = advantages.std().item()
        advantage_max = advantages.max().item()
        advantage_min = advantages.min().item()

        prompts_expanded = [x for x in batch["prompt"] for _ in range(args.group_size)]
        data = tokenize_prompt_and_output(prompts_expanded, outputs, tokenizer)
        input_ids = data["input_ids"].to(device)
        labels = data["labels"].to(device)
        response_mask = data["response_mask"].to(device)

        with torch.inference_mode():
            old_logprobs_all = []
            for b in range(len(input_ids) // args.micro_batch_size):
                idx = b * args.micro_batch_size
                end = idx + args.micro_batch_size
                old_logprobs_all.append(
                    get_response_log_probs(model, input_ids[idx:end], labels[idx:end]).detach()
                )
            old_logprobs_all = torch.cat(old_logprobs_all, dim=0)
            assert old_logprobs_all.shape == labels.shape

        epoch_grad_norms = []
        epoch_policy_ratios = []
        epoch_kl_divs = []

        training_start = time.time()
        for epoch in range(args.epochs_per_step):
            for b in tqdm(
                range(len(input_ids) // args.micro_batch_size),
                desc=(
                    f"Step {i + 1}/{args.n_grpo_steps}, "
                    f"Epoch {epoch + 1}/{args.epochs_per_step}"
                ),
            ):
                idx = b * args.micro_batch_size
                end = idx + args.micro_batch_size
                x = input_ids[idx:end]
                y = labels[idx:end]
                mask = response_mask[idx:end]
                micro_batch_adv = advantages[idx:end].unsqueeze(-1).to(device)

                policy_logprobs = get_response_log_probs(model, x, y)
                old_logprobs = old_logprobs_all[idx:end]

                ratio = torch.exp(policy_logprobs - old_logprobs)
                per_token_loss = -ratio * micro_batch_adv

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)) * mask
                    approx_kl_mean = approx_kl.sum() / mask.sum()
                    epoch_kl_divs.append(approx_kl_mean.item())

                    policy_ratio_mean = (ratio * mask).sum() / mask.sum()
                    epoch_policy_ratios.append(policy_ratio_mean.item())

                masked_loss = per_token_loss * mask
                denom = mask.sum(dim=-1).clamp_min(1)
                loss_per_prompt = masked_loss.sum(dim=-1) / denom
                loss = loss_per_prompt.mean() / args.gradient_accumulation_steps

                loss.backward()

                if (b + 1) % args.gradient_accumulation_steps == 0:
                    if (
                        args.save_first_step_grads_path is not None
                        and not first_step_grads_saved
                    ):
                        grad_payload = {}
                        for name, p in trainable_named_params:
                            if p.grad is None:
                                continue
                            if grad_prefixes is not None and not any(
                                name.startswith(prefix) for prefix in grad_prefixes
                            ):
                                continue
                            grad_payload[name] = p.grad.detach().cpu()
                        if len(grad_payload) == 0:
                            raise ValueError(
                                "No gradients matched --save-first-step-grads-prefixes "
                                f"for path {args.save_first_step_grads_path}"
                            )
                        grad_dir = os.path.dirname(args.save_first_step_grads_path)
                        if grad_dir:
                            os.makedirs(grad_dir, exist_ok=True)
                        save_safetensors_file(
                            grad_payload,
                            args.save_first_step_grads_path,
                        )
                        first_step_grads_saved = True
                        print(
                            "Saved first-step gradients to "
                            f"{args.save_first_step_grads_path} "
                            f"({len(grad_payload)} tensors)"
                        )
                    if optimizer_steps in save_grads_steps:
                        save_gradients(
                            trainable_named_params, run_dir, optimizer_steps,
                            subdir="optim_step",
                        )
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    epoch_grad_norms.append(grad_norm.item())
                    optimizer.step()
                    if normalize_blocktt_after_update:
                        normalize_trainable_blocktt_cores_(model)
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()
                    optimizer_steps += 1
                    if args.stop_after_first_step and optimizer_steps >= 1:
                        stop_requested = True
                        break
            if stop_requested:
                break
        if stop_requested:
            print("Stopping early after first optimizer step (--stop-after-first-step).")

        training_time = time.time() - training_start
        step_time = time.time() - step_start_time

        mean_grad_norm = (
            sum(epoch_grad_norms) / len(epoch_grad_norms) if epoch_grad_norms else 0
        )
        mean_policy_ratio = (
            sum(epoch_policy_ratios) / len(epoch_policy_ratios)
            if epoch_policy_ratios
            else 0
        )
        mean_kl = sum(epoch_kl_divs) / len(epoch_kl_divs) if epoch_kl_divs else 0

        total_tokens = response_mask.sum().item()
        tokens_per_sec = total_tokens / generation_time if generation_time > 0 else 0
        avg_generation_len = int(response_mask.sum(dim=-1).float().mean().item())

        if not args.no_wandb:
            wandb.log(
                {
                    "train/accuracy": train_accuracy,
                    "train/reward_mean": train_accuracy,
                    "train/reward_std": reward_std,
                    "train/reward_max": reward_max,
                    "train/reward_min": reward_min,
                    "train/advantage_mean": advantage_mean,
                    "train/advantage_std": advantage_std,
                    "train/advantage_max": advantage_max,
                    "train/advantage_min": advantage_min,
                    "train/grad_norm": mean_grad_norm,
                    "train/policy_ratio": mean_policy_ratio,
                    "train/approx_kl": mean_kl,
                    "train/avg_gen_length": avg_generation_len,
                    "time/step_time": step_time,
                    "time/generation_time": generation_time,
                    "time/training_time": training_time,
                    "time/tokens_per_sec": tokens_per_sec,
                },
                step=i + 1,
            )

        print(
            f"Step {i + 1}/{args.n_grpo_steps} | "
            f"Train Acc: {train_accuracy:.2%} | KL: {mean_kl:.4f} | "
            f"Time: {step_time:.1f}s"
        )

        if (i + 1) % 5 == 0:
            eval_model(i + 1)

        # Save checkpoints after first update, at step 10, and at final step
        if args.enable_save_ckpt:
            step_num = i + 1
            if should_save_checkpoint(step_num, args.n_grpo_steps):
                save_checkpoint(model, tokenizer, run_dir, step_num, args=args)
        if stop_requested:
            break

    if not args.enable_save_ckpt:
        print("Checkpoint saving disabled (--enable-save-ckpt not set).")

    # ─── Final merged-checkpoint save ──────────────────────────────────────
    final_step = args.n_grpo_steps
    final_ckpt_dir = None
    if args.enable_merged_ckpt or args.enable_math_verify:
        final_ckpt_dir = os.path.join(run_dir, f"step={final_step}")
        os.makedirs(final_ckpt_dir, exist_ok=True)
        if args.enable_merged_ckpt:
            print(f"Saving final merged checkpoint to {final_ckpt_dir}")
            try:
                save_merged_checkpoint(
                    model, tokenizer, final_ckpt_dir, args.train_mode, args
                )
            except Exception as _save_exc:
                # Don't let a checkpoint-save failure take down the
                # post-training math-verify eval — that eval uses the
                # in-memory model, not the on-disk checkpoint.
                print(
                    f"WARNING: save_merged_checkpoint failed: {_save_exc!r}. "
                    "Continuing to math-verify eval via the in-memory model."
                )
                if not args.no_wandb:
                    wandb.log({"save_merged_checkpoint/error": repr(_save_exc)})

    # ─── Post-training math-verify eval ────────────────────────────────────
    if args.enable_math_verify:
        import json as _json
        from math_verify_eval import math_verify_eval as _math_verify_eval

        try:
            if in_process_llm is not None:
                # Reuse the in-memory model + existing vLLM via hot-swap.
                if args.train_mode in {"blocktt", "svd"}:
                    weight_tuples = export_weights_for_vllm(model)
                elif args.train_mode in {"lora", "lora_full", "dora", "pissa", "milora", "randlora"}:
                    weight_tuples = export_lora_merged_weights_for_vllm(model)
                    in_process_llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(
                        weight_tuples
                    )
                    weight_tuples = None  # already loaded
                else:
                    weight_tuples = [(n, p) for n, p in model.named_parameters()]

                if weight_tuples is not None:
                    in_process_llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(
                        weight_tuples
                    )

                results = _math_verify_eval(
                    model=None,
                    tokenizer=tokenizer,
                    datasets=args.math_verify_datasets,
                    n_samples_override=args.math_verify_n_samples,
                    temperature_override=args.math_verify_temperature,
                    max_tokens=args.math_verify_max_tokens,
                    prompt_template_path=args.prompt_template,
                    vllm_kwargs={"llm": in_process_llm},
                )
            else:
                # HTTP-lora training: spin up a fresh in-process LLM from disk.
                if final_ckpt_dir is None:
                    raise RuntimeError(
                        "math-verify eval requires either an in-process LLM "
                        "or --enable-merged-ckpt for a saved checkpoint to load."
                    )
                results = _math_verify_eval(
                    model=None,
                    tokenizer=tokenizer,
                    datasets=args.math_verify_datasets,
                    n_samples_override=args.math_verify_n_samples,
                    temperature_override=args.math_verify_temperature,
                    max_tokens=args.math_verify_max_tokens,
                    prompt_template_path=args.prompt_template,
                    vllm_kwargs={
                        "model": final_ckpt_dir,
                        "tensor_parallel_size": 1,
                        "gpu_memory_utilization": args.gpu_memory_utilization,
                        "max_model_len": args.max_model_len,
                        "max_num_batched_tokens": 4096,
                    },
                )

            results["checkpoint"] = final_ckpt_dir or "<in-memory>"
            results["model_id_at_train_time"] = args.model_id

            if final_ckpt_dir is not None:
                results_path = os.path.join(final_ckpt_dir, "eval_results.json")
                with open(results_path, "w", encoding="utf-8") as f:
                    _json.dump(results, f, indent=2)
                print(f"Wrote {results_path}")

            if not args.no_wandb:
                for ds_name, ds_result in results["datasets"].items():
                    wandb.log({f"eval/{ds_name}/accuracy": ds_result["accuracy"]})
                for ds_name, reason in results.get("errors", {}).items():
                    wandb.log({f"eval/{ds_name}/error": reason})

            print("Math-verify results:")
            for ds_name, ds_result in results["datasets"].items():
                print(
                    f"  {ds_name}: {ds_result['accuracy']:.2%} "
                    f"({ds_result['n_correct']}/{ds_result['n_total']})"
                )
        except Exception as exc:
            print(f"ERROR: math-verify eval failed: {exc!r}")
            if not args.no_wandb:
                wandb.log({"eval/error": repr(exc)})

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
