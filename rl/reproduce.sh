#!/usr/bin/env bash
#
# Reproduce the RL (GRPO on math reasoning) numbers reported in the FuRA paper.
#
# Usage:
#     bash reproduce.sh fura          # FuRA / BlockTT  (the paper's main result)
#     bash reproduce.sh full          # Full fine-tuning baseline
#     bash reproduce.sh lora          # LoRA baseline (rank 32)
#     bash reproduce.sh svd           # SVD reparameterization baseline
#
# Hardware: a single H100 (94 GB) per run is sufficient for Qwen3-1.7B.
# DEVICE / LR can be overridden via env vars, e.g. DEVICE=1 LR=5e-5 bash reproduce.sh fura.
set -euo pipefail

VARIANT="${1:-fura}"
DEVICE="${DEVICE:-0}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-1.7B}"
WANDB_PROJECT="${WANDB_PROJECT:-fura-rl}"

case "$VARIANT" in
  full)
    LR="${LR:-1e-5}"
    CUDA_VISIBLE_DEVICES="$DEVICE" python run_rl.py \
      --train-mode full \
      --lr "$LR" \
      --optimizer adamw \
      --model-id "$MODEL_ID" \
      --wandb-project "$WANDB_PROJECT" \
      --wandb-run-name "full-lr_${LR}"
    ;;

  lora)
    LR="${LR:-9e-5}"
    LORA_RANK="${LORA_RANK:-32}"
    CUDA_VISIBLE_DEVICES="$DEVICE" python run_rl.py \
      --train-mode lora \
      --lora-rank "$LORA_RANK" \
      --trainable-type all \
      --lr "$LR" \
      --optimizer adamw \
      --model-id "$MODEL_ID" \
      --wandb-project "$WANDB_PROJECT" \
      --wandb-run-name "lora-r${LORA_RANK}-lr_${LR}"
    ;;

  svd)
    LR="${LR:-9e-5}"
    CUDA_VISIBLE_DEVICES="$DEVICE" python run_rl.py \
      --train-mode svd \
      --train-position output \
      --s-merged-to frozen \
      --trainable-type all \
      --lr "$LR" \
      --optimizer adamw \
      --model-id "$MODEL_ID" \
      --wandb-project "$WANDB_PROJECT" \
      --wandb-run-name "svd-lr_${LR}"
    ;;

  fura|blocktt)
    # Project default (see CLAUDE.md / paper Section 3.1):
    #   blocktt rank=full, decomp_mode=output_one_block,
    #   train_position=small, s_merged_to=keep_trainable.
    LR="${LR:-9e-5}"
    CUDA_VISIBLE_DEVICES="$DEVICE" python run_rl.py \
      --train-mode blocktt \
      --blocktt-rank full \
      --decomp-mode output_one_block \
      --train-position small \
      --s-merged-to keep_trainable \
      --trainable-type all \
      --lr "$LR" \
      --optimizer adamw \
      --model-id "$MODEL_ID" \
      --wandb-project "$WANDB_PROJECT" \
      --wandb-run-name "fura-lr_${LR}"
    ;;

  *)
    echo "Unknown variant: $VARIANT" >&2
    echo "Choose one of: fura | full | lora | svd" >&2
    exit 1
    ;;
esac
