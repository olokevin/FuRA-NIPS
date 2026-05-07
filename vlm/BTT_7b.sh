# BlockTT (FuRA) launcher for LLaVA-v1.5-7B visual instruction tuning.
# Replaces the DoRA PEFT wrapper with a direct Linear->BTT conversion;
# all other hyperparameters (lr, epoch, batch size, ...) are unchanged.
#
# Config: rank=full, decomp=output_one_block, train_position=small,
#         s_merged_to=keep_trainable (singular values stay trainable).

deepspeed llava/train/train_mem_btt.py \
    --btt_enable True \
    --btt_rank full \
    --btt_decomp_mode output_one_block \
    --btt_train_position small \
    --btt_s_merged_to keep_trainable \
    --btt_trainable_type all \
    --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path ./playground/data/llava_v1_5_mix665k.json \
    --image_folder ./playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-7b-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-btt-full-output_one_block-small-keep_trainable \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 2 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name ${WANDB_NAME:-llava-btt-full-ob-small-kt}
