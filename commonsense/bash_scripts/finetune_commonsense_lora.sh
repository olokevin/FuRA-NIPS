#!/bin/bash

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export LIBRARY_PATH="/usr/local/cuda/lib64:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
# export HF_HOME=...  # set if you want a custom HF cache directory

SRC_DIR="${SRC_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_DIR=LLM-Adapters      # MODIFY THIS LINE
OUTPUT_SRC_DIR="${OUTPUT_SRC_DIR:-${SRC_DIR}/output}"

# SLURM_ARRAY_TASK_ID=$1
# cfg=$(sed -n "$SLURM_ARRAY_TASK_ID"p ${SRC_DIR}/bash_scripts/slurm_config_lora_commonsense.txt)
# MODEL=$(echo $cfg | cut -f 1 -d ' ')
# adapter_name=$(echo $cfg | cut -f 2 -d ' ')
# lr=$(echo $cfg | cut -f 3 -d ' ')
# lora_r=$(echo $cfg | cut -f 4 -d ' ')
# lora_alpha=$(echo $cfg | cut -f 5 -d ' ')
# seed=$(echo $cfg | cut -f 6 -d ' ')

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B}"
adapter_name="${adapter_name:-lora}"
lr="${lr:-2e-4}"
lora_r="${lora_r:-64}"
lora_alpha="${lora_alpha:-128}"
seed="${seed:-43}"
MAX_STEPS="${MAX_STEPS:-0}"
model_tag="${MODEL##*/}"

# target_modules: space-separated list. Default = original 5-module set so legacy runs are unchanged.
target_modules="${target_modules:-q_proj k_proj v_proj up_proj down_proj}"
# Tag the run when overridden (count tokens; 5 -> default = no tag, 7 -> "7mod", else "Nmod").
_tm_count=$(echo $target_modules | wc -w)
if [ "$_tm_count" = "5" ] && [ "$target_modules" = "q_proj k_proj v_proj up_proj down_proj" ]; then
    _tm_tag=""
else
    _tm_tag="-tgt_${_tm_count}mod"
fi

wandb_project="${wandb_project:-commonsense-${model_tag}}"
run_name="${run_name:-${adapter_name}-lr_${lr}-rank_${lora_r}${_tm_tag}-seed_${seed}}"
wandb_run_id="${wandb_run_id:-$(python -c 'import wandb; print(wandb.util.generate_id())')}"

export WANDB_RUN_ID="${wandb_run_id}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"

OUTPUT="${OUTPUT:-${OUTPUT_SRC_DIR}/commonsense/${MODEL}/${adapter_name}-lr_${lr}-rank_${lora_r}${_tm_tag}-seed_${seed}}"
mkdir -p $OUTPUT

cd $SRC_DIR

accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    --mixed_precision="bf16" \
    src/finetune_lora.py \
    --model_name_or_path ${MODEL} \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --logging_steps 10 \
    --max_seq_len 2048 \
    --learning_rate ${lr} \
    --weight_decay 0. \
    --num_train_epochs ${num_train_epochs:-3} \
    --mixed_precision bf16 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type linear \
    --num_warmup_steps 0.03 \
    --seed ${seed} \
    --gradient_checkpointing \
    --instruction_type single \
    --load_last_model \
    --adapter_name ${adapter_name} \
    --lora_r ${lora_r} \
    --lora_alpha ${lora_alpha} \
    --target_modules ${target_modules} \
    --data_path ${DATA_DIR}/ft-training_set/commonsense_170k.json \
    --wandb_project "${wandb_project}" \
    --wandb_run_name "${run_name}" \
    --max_steps ${MAX_STEPS} \
    --save_interval 100000 \
    --output_dir $OUTPUT 2> >(tee $OUTPUT/err.log >&2) | tee $OUTPUT/training.log

if [ "${MAX_STEPS}" = "0" ]; then
    # If training saved last/ (and optionally best/) subdirs, eval each separately;
    # otherwise fall back to the old single-checkpoint layout.
    if [ -d "$OUTPUT/last" ]; then
        bash bash_scripts/eval_commonsense_lora.sh \
            CKPT="$OUTPUT/last" \
            adapter_name="${adapter_name}" \
            base_model="${MODEL}" \
            wandb_project="${wandb_project}" \
            wandb_run_name="${run_name}-last" \
            wandb_run_id=""
        if [ -d "$OUTPUT/best" ]; then
            bash bash_scripts/eval_commonsense_lora.sh \
                CKPT="$OUTPUT/best" \
                adapter_name="${adapter_name}" \
                base_model="${MODEL}" \
                wandb_project="${wandb_project}" \
                wandb_run_name="${run_name}-best" \
                wandb_run_id=""
        fi
    else
        bash bash_scripts/eval_commonsense_lora.sh \
            CKPT="$OUTPUT" \
            adapter_name="${adapter_name}" \
            base_model="${MODEL}" \
            wandb_project="${wandb_project}" \
            wandb_run_name="${run_name}" \
            wandb_run_id="${wandb_run_id}"
    fi
fi
