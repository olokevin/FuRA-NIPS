#!/bin/bash

CKPT=""
base_model="meta-llama/Meta-Llama-3-8B"
wandb_project="${wandb_project:-}"
wandb_run_name="${run_name:-}"
wandb_run_id="${wandb_run_id:-${WANDB_RUN_ID:-}}"

usage() {
    echo "Usage: bash bash_scripts/eval_commonsense.sh CKPT=<model_checkpoint_path> [base_model=<hf_model>] [wandb_project=<project>] [wandb_run_name=<name>] [wandb_run_id=<id>]"
}

for arg in "$@"; do
    case "$arg" in
        CKPT=*|ckpt=*)
            CKPT="${arg#*=}"
            ;;
        MODEL=*|model=*)
            CKPT="${arg#*=}"
            ;;
        base_model=*)
            base_model="${arg#*=}"
            ;;
        wandb_project=*)
            wandb_project="${arg#*=}"
            ;;
        wandb_run_name=*|run_name=*)
            wandb_run_name="${arg#*=}"
            ;;
        wandb_run_id=*)
            wandb_run_id="${arg#*=}"
            ;;
        *)
            echo "Unknown argument: $arg"
            usage
            exit 1
            ;;
    esac
done

if [ -z "$CKPT" ]; then
    usage
    exit 1
fi

# Resolve $CKPT against the dual-save layout introduced by finetune_*.py:
#   <output_dir>/last/  : final-step weights (always written by training)
#   <output_dir>/best/  : best-eval weights (only when val_set_size > 0 and
#                        --load_last_model is not set)
# If $CKPT is itself a flat HF checkpoint (config.json present), use as-is.
# Else prefer last/ over best/ (project default = save last model).
# Override with EVAL_PREFER=best to pick best/ when both exist.
if [ -f "${CKPT}/config.json" ]; then
    MODEL="$CKPT"
elif [ "${EVAL_PREFER:-last}" = "best" ] && [ -f "${CKPT}/best/config.json" ]; then
    MODEL="${CKPT}/best"
elif [ -f "${CKPT}/last/config.json" ]; then
    MODEL="${CKPT}/last"
elif [ -f "${CKPT}/best/config.json" ]; then
    MODEL="${CKPT}/best"
else
    echo "Error: no usable checkpoint at ${CKPT}, ${CKPT}/last, or ${CKPT}/best (config.json missing)" >&2
    exit 1
fi
echo "[eval_commonsense] resolved CKPT=${CKPT} -> MODEL=${MODEL}"

OUTPUT_DIR="${MODEL}/commonsense"

# commonsense/ root (parent of bash_scripts/) — contains src/eval/run_commonsense_parallel.py
SRC_DIR="${SRC_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# Path to the LLM-Adapters evaluation datasets. Download from:
#   https://github.com/AGI-Edgerunners/LLM-Adapters/tree/main/dataset
# and set DATA_DIR accordingly (default points at the bundled LLM-Adapters/dataset).
DATA_DIR="${DATA_DIR:-${SRC_DIR}/LLM-Adapters/dataset}"

# datasets=(boolq)
# Allow caller to override the dataset list via env var EVAL_DATASETS
# (space-separated). Default = full 8-task suite.
if [ -n "${EVAL_DATASETS:-}" ]; then
    read -r -a datasets <<< "$EVAL_DATASETS"
else
    datasets=(boolq piqa social_i_qa ARC-Challenge ARC-Easy openbookqa hellaswag winogrande)
fi

cd $SRC_DIR

if [ -n "$wandb_run_id" ]; then
    export WANDB_RUN_ID="$wandb_run_id"
    export WANDB_RESUME="${WANDB_RESUME:-must}"
fi

master_port=$((RANDOM % 5000 + 20000))
for dataset in "${datasets[@]}"; do
    OUTPUT=$OUTPUT_DIR/$dataset
    mkdir -p $OUTPUT
    BATCH_SIZE=32
    cmd=(
        accelerate launch --main_process_port "$master_port" ./src/eval/run_commonsense_parallel.py
        --data_path "${DATA_DIR}/$dataset/test.json"
        --model_name_or_path "$MODEL"
        --base_model "$base_model"
        --per_device_eval_batch_size "$BATCH_SIZE"
        --seed 1234
        --dtype bf16
        --dataset "$dataset"
        --output_dir "$OUTPUT"
    )
    if [ -n "$wandb_project" ]; then
        cmd+=(--wandb_project "$wandb_project")
    fi
    if [ -n "$wandb_run_name" ]; then
        cmd+=(--wandb_run_name "$wandb_run_name")
    fi
    if [ -n "$wandb_run_id" ]; then
        cmd+=(--wandb_run_id "$wandb_run_id")
    fi

    "${cmd[@]}" 2> >(tee "$OUTPUT/eval_err.log" >&2) | tee "$OUTPUT/eval.log"
done
