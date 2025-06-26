# Reference Running: bash train/sft_lora.sh
# LoRA training script for s1 project
uid="$(date +%Y%m%d_%H%M%S)"
model_size="32B"  # Default model size, can be overridden via --model_size
base_model="Qwen/Qwen2.5-${model_size}-Instruct"
block_size=20000
lr=5e-4
min_lr=0
epochs=5
weight_decay=1e-4
batch_size=16
micro_batch_size=1
max_steps=-1
gpu_count=$(nvidia-smi -L | wc -l)
gradient_accumulation_steps=$((batch_size / (micro_batch_size * gpu_count)))
push_to_hub=false

# LoRA specific parameters
rank=128  # Default rank, can be overridden via command line
alpha=16

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_size)
            model_size="$2"
            base_model="Qwen/Qwen2.5-${model_size}-Instruct"
            shift 2
            ;;
        --rank)
            rank="$2"
            # alpha remains fixed at 16 for RSLoRA
            shift 2
            ;;
        --lr|--learning_rate)
            lr="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

torchrun --nproc-per-node ${gpu_count} --master_port 12345 \
    train/sft_lora.py \
    --block_size=${block_size} \
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
    --output_dir="ckpts/s1-lora-${model_size}-r${rank}-${uid}" \
    --push_to_hub=${push_to_hub} \
    --save_only_model=True \
    --gradient_checkpointing=True