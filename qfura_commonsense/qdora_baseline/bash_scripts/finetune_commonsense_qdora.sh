#!/bin/bash

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export LIBRARY_PATH="/usr/local/cuda/lib64:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
# export HF_HOME=...

SRC_DIR="${SRC_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PROJECT_DIR="${PROJECT_DIR:-${SRC_DIR}}"

DATA_DIR="${DATA_DIR:-${SRC_DIR}/LLM-Adapters}"
OUTPUT_SRC_DIR="${OUTPUT_SRC_DIR:-${SRC_DIR}/output}"

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B}"
lora_r="${lora_r:-64}"
lora_alpha="${lora_alpha:-128}"
lora_dropout="${lora_dropout:-0.05}"
lr="${lr:-2e-4}"
seed="${seed:-43}"
MAX_STEPS="${MAX_STEPS:-0}"
qdora_impl="${qdora_impl:-fast}"
dora_norm_cache_steps="${dora_norm_cache_steps:-16}"
model_tag="${MODEL##*/}"

wandb_project="${wandb_project:-qdora-commonsense-${model_tag}}"
wandb_run_id="${wandb_run_id:-$(python -c 'import wandb; print(wandb.util.generate_id())' 2>/dev/null)}"
no_wandb_flag=""
if [ "${no_wandb:-0}" = "1" ]; then
    no_wandb_flag="--no_wandb"
fi

# Project default: --load_last_model (save last only). Set save_both=1 to drop
# the flag, which lets finetune_qdora.py track best-eval and write both
# <OUTPUT>/last and <OUTPUT>/best.
load_last_flag="--load_last_model"
if [ "${save_both:-0}" = "1" ]; then
    load_last_flag=""
fi

export WANDB_RUN_ID="${wandb_run_id}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"

echo $MODEL

OUTPUT=${OUTPUT_SRC_DIR}/commonsense/${MODEL}/qdora-r_${lora_r}-alpha_${lora_alpha}-lr_${lr}-seed_${seed}
run_name="${run_name:-$(basename "$OUTPUT")}"

mkdir -p $OUTPUT

cd ${SRC_DIR}

accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    --mixed_precision="bf16" \
    src/finetune_qdora.py \
    --model_name_or_path ${MODEL} \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --max_seq_len 2048 \
    --learning_rate ${lr} \
    --weight_decay 0. \
    --num_train_epochs 3 \
    --mixed_precision bf16 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type linear \
    --num_warmup_steps 0.03 \
    --seed ${seed} \
    --gradient_checkpointing \
    --instruction_type single \
    --lora_r ${lora_r} \
    --lora_alpha ${lora_alpha} \
    --lora_dropout ${lora_dropout} \
    --qdora_impl ${qdora_impl} \
    --dora_norm_cache_steps ${dora_norm_cache_steps} \
    --max_steps ${MAX_STEPS} \
    --val_set_size 120 \
    --eval_step 400 \
    --save_interval 100000 \
    ${load_last_flag} \
    --data_path ${DATA_DIR}/ft-training_set/commonsense_170k.json \
    --wandb_project "${wandb_project}" \
    --wandb_run_name "${run_name}" \
    ${no_wandb_flag} \
    --output_dir $OUTPUT 2> >(tee $OUTPUT/err.log >&2) | tee $OUTPUT/training.log

if [ "${MAX_STEPS}" = "0" ]; then
    # finetune_qdora.py dual-save layout:
    #   fast path: writes full HF models to <OUTPUT>/last/ (always) and
    #              <OUTPUT>/best/ (if best-tracking ran).
    #   peft path: writes adapters to <OUTPUT>/last_adapter/ and
    #              <OUTPUT>/best_adapter/; merge each into a sibling
    #              <OUTPUT>/{last,best}/ for eval.
    if [ "${qdora_impl}" != "fast" ]; then
        for sub in last best; do
            ADAPTER_DIR="${OUTPUT}/${sub}_adapter"
            MERGED_DIR="${OUTPUT}/${sub}"
            if [ -f "${ADAPTER_DIR}/adapter_config.json" ] && [ ! -f "${MERGED_DIR}/config.json" ]; then
                python ${SRC_DIR}/tools/merge_qlora_for_eval.py \
                    --base_model ${MODEL} \
                    --adapter_dir "${ADAPTER_DIR}" \
                    --output_dir "${MERGED_DIR}"
            fi
        done
    fi
    # eval_commonsense.sh prefers last/ over best/.
    bash ./bash_scripts/eval_commonsense.sh \
        CKPT="${OUTPUT}" \
        base_model="${MODEL}" \
        wandb_project="${wandb_project}" \
        wandb_run_name="${run_name}" \
        wandb_run_id="${wandb_run_id}"
fi
