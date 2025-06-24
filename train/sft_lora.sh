# Reference Running: bash train/sft_lora.sh
# LoRA training script for s1 project
uid="$(date +%Y%m%d_%H%M%S)"
base_model="Qwen/Qwen2.5-32B-Instruct"
lr=3e-4  # Higher learning rate for LoRA
min_lr=0
epochs=5
weight_decay=1e-4
micro_batch_size=1
gradient_accumulation_steps=1
max_steps=-1
gpu_count=$(nvidia-smi -L | wc -l)
push_to_hub=false

# LoRA specific parameters
rank=16  # Default rank, can be overridden via command line
alpha=32  # Alpha = 2*rank (for default rank=16)

torchrun --nproc-per-node ${gpu_count} --master_port 12345 \
    train/sft_lora.py \
    --block_size=32768 \
    --per_device_train_batch_size=${micro_batch_size} \
    --per_device_eval_batch_size=${micro_batch_size} \
    --gradient_accumulation_steps=${gradient_accumulation_steps} \
    --num_train_epochs=${epochs} \
    --train_file_path="simplescaling/s1K_tokenized" \
    --model_name=${base_model} \
    --warmup_ratio=0.05 \
    --fsdp="full_shard auto_wrap" \
    --fsdp_config="train/fsdp_config_qwen.json" \
    --bf16=True \
    --eval_strategy="no" \
    --logging_steps=1 \
    --save_strategy="no" \
    --lr_scheduler_type="cosine" \
    --learning_rate=${lr} \
    --weight_decay=${weight_decay} \
    --adam_beta1=0.9 \
    --adam_beta2=0.95 \
    --use_lora=True \
    --rank=${rank} \
    --alpha=${alpha} \
    --output_dir="ckpts/s1-lora-r${rank}-${uid}" \
    --push_to_hub=${push_to_hub} \
    --save_only_model=True
    # --gradient_checkpointing=True \ # Likely not needed with LoRA due to reduced memory usage