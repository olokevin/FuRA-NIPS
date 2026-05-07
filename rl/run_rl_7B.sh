############ RL — Qwen2.5-7B (single H100 NVL 96GB) ############
#
# Memory recipe (single H100 NVL 93GB usable) — identical for full / lora / dora / blocktt:
#   --micro-batch-size 1
#   --gradient-accumulation-steps 256   (effective batch unchanged vs the 1.7B mbs=2/gacc=128 default)
#   --gpu-memory-utilization 0.25       (vs run_rl.py default 0.4 — frees ~14 GB for trainer activations)
#   --max-model-len 2048
#
# Background: the run_rl.py defaults (mbs=2 / gacc=128 / gpu_util=0.4) OOM at step 1 on 7B for
# blocktt and dora; lora barely fits. See docs/exp_results/rl.md "Qwen2.5-7B (base)" section.
#
# For LORA / DORA:
# - in-process vLLM rollout used by default (no `vllm serve` HTTP server needed)
# - to use HTTP vLLM instead, export VLLM_URL=http://localhost:8000 and start a server on a
#   separate GPU with: VLLM_ALLOW_RUNTIME_LORA_UPDATING=True vllm serve Qwen/Qwen2.5-7B \
#     --enable-lora --max-lora-rank 64

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B}"
WANDB_PROJECT="${WANDB_PROJECT:-qwen2_5-7B-RL}"

# Shared memory-safe flags for the 7B runs.
MEM_FLAGS=(
  --micro-batch-size 1
  --gradient-accumulation-steps 256
  --gpu-memory-utilization 0.25
  --max-model-len 2048
)

# Optional LR-scheduler flags (default: fixed lr, matching run_rl.py default of --lr-scheduler none).
# Set LR_SCHEDULER=linear WARMUP_RATIO=0.03 to mirror the LIFT recipe (linear decay to 0,
# 3% warmup), or LR_SCHEDULER=cosine with WARMUP_RATIO=... MIN_LR_RATIO=... for cosine.
SCHED_FLAGS=()
if [[ -n "${LR_SCHEDULER:-}" ]]; then
  SCHED_FLAGS+=(--lr-scheduler "$LR_SCHEDULER")
fi
if [[ -n "${WARMUP_RATIO:-}" ]]; then
  SCHED_FLAGS+=(--warmup-ratio "$WARMUP_RATIO")
fi
if [[ -n "${MIN_LR_RATIO:-}" ]]; then
  SCHED_FLAGS+=(--min-lr-ratio "$MIN_LR_RATIO")
fi

run_full()
{
  local train_mode="full"
  local lr="${LR:-1e-5}"
  local optimizer="${OPTIMIZER:-adamw}"
  local sched_tag=""
  if [[ -n "${LR_SCHEDULER:-}" ]]; then
    sched_tag="-${LR_SCHEDULER}_w${WARMUP_RATIO:-0}"
  fi
  local run_name="${train_mode}-${optimizer}-lr_${lr}${sched_tag}"
  local device="${DEVICE:-2}"
  local -a cfg_suffix_args=()
  if [[ -n "${CFG_SUFFIX:-}" ]]; then
    # Intended for trusted local overrides, e.g. CFG_SUFFIX="--flag --arg value".
    read -r -a cfg_suffix_args <<< "${CFG_SUFFIX}"
  fi

  CUDA_VISIBLE_DEVICES="$device" uv run run_rl.py \
    --train-mode "$train_mode" \
    --lr "$lr" \
    --optimizer "$optimizer" \
    "${MEM_FLAGS[@]}" \
    "${SCHED_FLAGS[@]}" \
    --model-id "$MODEL_ID" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-run-name "$run_name" \
    "${cfg_suffix_args[@]}"
}

run_lora()
{
  local train_mode="${TRAIN_MODE:-lora}"
  local lr="${LR:-1e-4}"
  local optimizer="${OPTIMIZER:-adamw}"
  local lora_rank="${LORA_RANK:-64}"
  local trainable_type="${TRAINABLE_TYPE:-all}"
  local name_suffix="${NAME_SUFFIX:-}"
  local sched_tag=""
  if [[ -n "${LR_SCHEDULER:-}" ]]; then
    sched_tag="-${LR_SCHEDULER}_w${WARMUP_RATIO:-0}"
  fi
  local run_name="${train_mode}-${optimizer}-lr_${lr}-rank_${lora_rank}${sched_tag}${name_suffix}"
  local device="${DEVICE:-2}"
  local vllm_url="${VLLM_URL:-http://localhost:8000}"
  local -a vllm_url_args=()
  local -a cfg_suffix_args=()
  if [[ -n "${CFG_SUFFIX:-}" ]]; then
    # Intended for trusted local overrides, e.g. CFG_SUFFIX="--flag --arg value".
    read -r -a cfg_suffix_args <<< "${CFG_SUFFIX}"
  fi
  if [[ "$train_mode" == "lora" || "$train_mode" == "dora" || "$train_mode" == "randlora" ]]; then
    vllm_url_args=(--vllm-url "$vllm_url")
  fi

  CUDA_VISIBLE_DEVICES="$device" uv run run_rl.py \
    --train-mode "$train_mode" \
    --lr "$lr" \
    --optimizer "$optimizer" \
    --lora-rank "$lora_rank" \
    --trainable-type "$trainable_type" \
    "${MEM_FLAGS[@]}" \
    "${SCHED_FLAGS[@]}" \
    "${vllm_url_args[@]}" \
    --model-id "$MODEL_ID" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-run-name "$run_name" \
    "${cfg_suffix_args[@]}"
}

run_blocktt()
{
  local train_mode="blocktt"
  local lr="${LR:-1e-4}"
  local optimizer="${OPTIMIZER:-adamw}"
  local decomp_mode="${DECOMP_MODE:-output_one_block}"
  local train_position="${TRAIN_POSITION:-small}"
  local s_merged_to="${S_MERGED_TO:-keep_trainable}"
  local device="${DEVICE:-2}"
  local name_suffix="${NAME_SUFFIX:-}"
  local sched_tag=""
  if [[ -n "${LR_SCHEDULER:-}" ]]; then
    sched_tag="-${LR_SCHEDULER}_w${WARMUP_RATIO:-0}"
  fi
  local run_name="${train_mode}-${optimizer}-lr_${lr}-${decomp_mode}-s_to_${s_merged_to}-train_${train_position}${sched_tag}${name_suffix}"
  local -a cfg_suffix_args=()
  if [[ -n "${CFG_SUFFIX:-}" ]]; then
    # Intended for trusted local overrides, e.g. CFG_SUFFIX="--flag --arg value".
    read -r -a cfg_suffix_args <<< "${CFG_SUFFIX}"
  fi

  if [[ "$train_position" != "small" && "$train_position" != "large" && "$train_position" != "both" ]]; then
    echo "Invalid BlockTT train position: $train_position (expected: small|large|both)"
    return 1
  fi

  CUDA_VISIBLE_DEVICES="$device" uv run run_rl.py \
    --train-mode "$train_mode" \
    --lr "$lr" \
    --optimizer "$optimizer" \
    --trainable-type all \
    --decomp-mode "$decomp_mode" \
    --s-merged-to "$s_merged_to" \
    --train-position "$train_position" \
    "${MEM_FLAGS[@]}" \
    "${SCHED_FLAGS[@]}" \
    --model-id "$MODEL_ID" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-run-name "$run_name" \
    "${cfg_suffix_args[@]}"
}


if [[ "$TRAIN_MODE" == "full" ]]; then
    run_full
elif [[ "$TRAIN_MODE" == "lora" ]]; then
    run_lora
elif [[ "$TRAIN_MODE" == "dora" ]]; then
    run_lora
elif [[ "$TRAIN_MODE" == "blocktt" ]]; then
    run_blocktt
else
    echo "Unsupported train mode: $TRAIN_MODE"
    echo "Use TRAIN_MODE=full|lora|dora|blocktt"
    exit 1
fi


### Example invocations
#
# DEVICE=3 LR=1e-5 TRAIN_MODE=full bash run_rl_7B.sh
# DEVICE=3 LR=1e-4 LORA_RANK=64 TRAIN_MODE=lora bash run_rl_7B.sh
# DEVICE=3 LR=1e-4 LORA_RANK=64 TRAIN_MODE=dora bash run_rl_7B.sh
# DEVICE=3 LR=1e-4 TRAIN_MODE=blocktt bash run_rl_7B.sh
