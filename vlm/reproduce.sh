#!/usr/bin/env bash
#
# Reproduce visual-instruction-tuning numbers from the FuRA paper (Section 6.2).
# Trains LLaVA-v1.5-7B on the llava_v1_5_mix665k mixture with FuRA / BlockTT
# applied to the language model's linear layers.
#
# Usage:
#     bash reproduce.sh fura      # FuRA / BlockTT  (paper main result)
#     bash reproduce.sh dora      # DoRA baseline (from the original DoRA repo)
#
# Prereqs (see vlm/README.md):
#   1. Install the vlm-specific environment (uses the bundled `peft/` patch).
#      This env is **separate** from the FuRA root env.
#   2. Download:
#        - LLaVA pretrain projector: ./checkpoints/llava-v1.5-7b-pretrain/mm_projector.bin
#        - Instruction data: ./playground/data/llava_v1_5_mix665k.json
#        - Image folder:     ./playground/data/<datasets>/...
set -euo pipefail

VARIANT="${1:-fura}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

case "$VARIANT" in
  fura|blocktt|btt)
    bash BTT_7b.sh
    ;;
  dora)
    bash Dora_7b.sh
    ;;
  *)
    echo "Unknown variant: $VARIANT" >&2
    echo "Choose one of: fura | dora" >&2
    exit 1
    ;;
esac
