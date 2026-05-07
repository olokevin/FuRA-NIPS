#!/usr/bin/env bash
#
# Reproduce the commonsense reasoning numbers from the FuRA paper (Section 5.3).
# Trains on commonsense_170k and evaluates on the eight LLM-Adapters tasks
# (BoolQ / PIQA / SIQA / HellaSwag / WinoGrande / ARC-Easy / ARC-Challenge / OBQA).
#
# Usage:
#     bash reproduce.sh fura          # FuRA / BlockTT (paper main result)
#     bash reproduce.sh full          # Full fine-tuning baseline
#     bash reproduce.sh lora          # LoRA baseline (rank 32 by default)
#     bash reproduce.sh svd           # SVD reparameterization baseline
#
# Environment overrides:
#     MODEL=...           HuggingFace model id (default: meta-llama/Meta-Llama-3-8B)
#     lr=...              learning rate (per-method default applies)
#     seed=...            random seed (default: 43)
#     OUTPUT=...          output directory (default: ./output/<run-tag>)
#
# Hardware: Llama-3-8B FuRA fits on 1× H100 (94 GB) with PER_DEVICE_TRAIN_BS=8,
# GRAD_ACC_STEPS=2 (= effective batch 16). Reduce if OOM.
set -euo pipefail

VARIANT="${1:-fura}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

case "$VARIANT" in
  full)
    bash bash_scripts/finetune_commonsense_full.sh
    ;;
  lora)
    bash bash_scripts/finetune_commonsense_lora.sh
    ;;
  svd)
    bash bash_scripts/finetune_commonsense_svd.sh
    ;;
  fura|blocktt)
    bash bash_scripts/finetune_commonsense_blocktt.sh
    ;;
  *)
    echo "Unknown variant: $VARIANT" >&2
    echo "Choose one of: fura | full | lora | svd" >&2
    exit 1
    ;;
esac
