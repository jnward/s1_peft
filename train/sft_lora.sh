# Reference Running: bash train/sft_lora.sh
# LoRA training script for s1 project
uid="$(date +%Y%m%d_%H%M%S)"
model_size="32B"  # Default model size, can be overridden via --model_size
base_model="Qwen/Qwen2.5-${model_size}-Instruct"
block_size=20000
lr=1e-3
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
rank=1  # Default rank, can be overridden via command line
alpha=16
layer_start=""  # Default: train all layers
layer_end=""    # Default: train all layers

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
        --layer_start)
            layer_start="$2"
            shift 2
            ;;
        --layer_end)
            layer_end="$2"
            shift 2
            ;;
        --last_third)
            # For 64 layer model, train last 21 layers (43-64)
            layer_start="43"
            layer_end="64"
            shift 1
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build layer range suffix for output directory
layer_suffix=""
if [ -n "$layer_start" ] || [ -n "$layer_end" ]; then
    # Use actual values or defaults for display
    display_start=${layer_start:-0}
    display_end=${layer_end:-all}
    layer_suffix="-l${display_start}-${display_end}"
fi

# Build command with optional layer range parameters
cmd="torchrun --nproc-per-node ${gpu_count} --master_port 12345 \
    train/sft_lora.py \
    --block_size=${block_size} \
    --per_device_train_batch_size=${micro_batch_size} \
    --per_device_eval_batch_size=${micro_batch_size} \
    --gradient_accumulation_steps=${gradient_accumulation_steps} \
    --num_train_epochs=${epochs} \
    --train_file_path=\"simplescaling/s1K-1.1_tokenized\" \
    --model_name=${base_model} \
    --warmup_ratio=0.05 \
    --fsdp=\"full_shard auto_wrap\" \
    --fsdp_config=\"train/fsdp_config_qwen.json\" \
    --bf16=True \
    --eval_strategy=\"no\" \
    --logging_steps=1 \
    --save_strategy=\"no\" \
    --lr_scheduler_type=\"cosine\" \
    --learning_rate=${lr} \
    --weight_decay=${weight_decay} \
    --adam_beta1=0.9 \
    --adam_beta2=0.95 \
    --use_lora=True \
    --rank=${rank} \
    --alpha=${alpha}"

# Add layer range parameters if specified
if [ -n "$layer_start" ]; then
    cmd="$cmd --layer_start=${layer_start}"
fi
if [ -n "$layer_end" ]; then
    cmd="$cmd --layer_end=${layer_end}"
fi

cmd="$cmd \
    --output_dir=\"ckpts_1.1/s1-lora-${model_size}-r${rank}-mlp_only${layer_suffix}-${uid}\" \
    --push_to_hub=${push_to_hub} \
    --save_only_model=True \
    --gradient_checkpointing=True"

# Execute the command
eval $cmd