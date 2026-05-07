#!/usr/bin/env bash
#
# Reproduce QDoRA baseline on commonsense reasoning.
#
# QDoRA uses a separate Python environment from the rest of FuRA — see
# qdora_baseline/requirements.txt. Set up a dedicated venv first:
#
#     python -m venv .venv-qdora
#     source .venv-qdora/bin/activate
#     pip install -r requirements.txt
#
# then run this script.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
bash bash_scripts/finetune_commonsense_qdora.sh
