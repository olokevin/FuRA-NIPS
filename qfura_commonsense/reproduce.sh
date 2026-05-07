#!/usr/bin/env bash
#
# Reproduce QFuRA results on commonsense reasoning (paper Section 6.1).
# Trains on commonsense_170k and evaluates on the eight LLM-Adapters tasks.
#
# Usage:
#     bash reproduce.sh qfura          # QFuRA  (paper main result)
#     bash reproduce.sh qlora          # QLoRA  baseline
#
# For QDoRA — which requires a separate Python environment — run:
#     cd qdora_baseline && bash reproduce.sh
set -euo pipefail

VARIANT="${1:-qfura}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

case "$VARIANT" in
  qfura|fura)
    bash bash_scripts/finetune_commonsense_qfura.sh
    ;;
  qlora)
    bash bash_scripts/finetune_commonsense_qlora.sh
    ;;
  *)
    echo "Unknown variant: $VARIANT" >&2
    echo "Choose one of: qfura | qlora    (qdora lives in qdora_baseline/)" >&2
    exit 1
    ;;
esac
