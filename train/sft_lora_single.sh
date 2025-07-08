#!/bin/bash
# Training script for single example experiment
# Trains on one example (index 732) while evaluating on 100 random examples

# Fix for H200 NCCL CUDA error 999 - disable NVLS (NVLink SHARP)
export NCCL_NVLS_ENABLE=0
export NCCL_ALGO=Ring
export NCCL_DEBUG=WARN
export CUDA_LAUNCH_BLOCKING=0

uid="$(date +%Y%m%d_%H%M%S)"
model_size="32B"  # Default model size, can be overridden via --model_size
base_model="Qwen/Qwen2.5-${model_size}-Instruct"
block_size=20000
lr=1e-4
min_lr=0
epochs=1000
weight_decay=1e-4
batch_size=1  # Single example, so batch size is 1
micro_batch_size=1
eval_batch_size=2  # Larger batch size for faster evaluation
max_steps=-1
gpu_count=$(nvidia-smi -L | wc -l)
gradient_accumulation_steps=$((batch_size / (micro_batch_size * gpu_count)))
push_to_hub=false

# LoRA specific parameters
rank=1  # Default rank, can be overridden via command line
alpha=16
layer_start=""  # Default: train all layers
layer_end=""    # Default: train all layers
layer_indices=""  # Specific layers to train (comma-separated)
layer_stride=""   # Train every nth layer
mlp_only=false  # Default: train all layers (attention + MLP)

# Single example training parameters
train_example_idx=732  # Default training example index
num_eval_examples=50  # Number of validation examples
eval_steps=50  # Evaluate every N steps
save_steps=50  # Save checkpoint every N steps

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
        --layer_indices)
            layer_indices="$2"
            shift 2
            ;;
        --layer_stride|--every_nth)
            layer_stride="$2"
            shift 2
            ;;
        --epochs)
            epochs="$2"
            shift 2
            ;;
        --train_example_idx)
            train_example_idx="$2"
            shift 2
            ;;
        --num_eval_examples)
            num_eval_examples="$2"
            shift 2
            ;;
        --eval_steps)
            eval_steps="$2"
            shift 2
            ;;
        --save_steps)
            save_steps="$2"
            shift 2
            ;;
        --mlp_only)
            mlp_only=true
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
if [ -n "$layer_indices" ]; then
    # Specific indices - abbreviate if too many
    num_indices=$(echo "$layer_indices" | tr -cd ',' | wc -c)
    num_indices=$((num_indices + 1))
    if [ $num_indices -gt 5 ]; then
        first_idx=$(echo "$layer_indices" | cut -d',' -f1)
        last_idx=$(echo "$layer_indices" | rev | cut -d',' -f1 | rev)
        layer_suffix="-l${first_idx}...${last_idx}_${num_indices}layers"
    else
        layer_suffix="-l${layer_indices//,/_}"
    fi
elif [ -n "$layer_stride" ]; then
    display_start=${layer_start:-0}
    display_end=${layer_end:-all}
    layer_suffix="-l${display_start}-${display_end}-stride${layer_stride}"
elif [ -n "$layer_start" ] || [ -n "$layer_end" ]; then
    # Use actual values or defaults for display
    display_start=${layer_start:-0}
    display_end=${layer_end:-all}
    layer_suffix="-l${display_start}-${display_end}"
fi

# Build command with optional layer range parameters
cmd="torchrun --nproc-per-node ${gpu_count} --master_port 12345 \
    train/sft_lora_single.py \
    --block_size=${block_size} \
    --per_device_train_batch_size=${micro_batch_size} \
    --per_device_eval_batch_size=${eval_batch_size} \
    --gradient_accumulation_steps=${gradient_accumulation_steps} \
    --num_train_epochs=${epochs} \
    --train_file_path=\"simplescaling/s1K-1.1_tokenized\" \
    --model_name=${base_model} \
    --warmup_ratio=0.05 \
    --fsdp=\"full_shard auto_wrap\" \
    --fsdp_config=\"train/fsdp_config_qwen.json\" \
    --bf16=True \
    --eval_strategy=\"steps\" \
    --eval_steps=${eval_steps} \
    --save_strategy=\"steps\" \
    --save_steps=${save_steps} \
    --logging_steps=1 \
    --lr_scheduler_type=\"cosine\" \
    --learning_rate=${lr} \
    --weight_decay=${weight_decay} \
    --adam_beta1=0.9 \
    --adam_beta2=0.95 \
    --use_lora=True \
    --rank=${rank} \
    --alpha=${alpha} \
    --train_example_idx=${train_example_idx} \
    --num_eval_examples=${num_eval_examples} \
    --mlp_only=${mlp_only}"

# Add layer range parameters if specified
if [ -n "$layer_start" ]; then
    cmd="$cmd --layer_start=${layer_start}"
fi
if [ -n "$layer_end" ]; then
    cmd="$cmd --layer_end=${layer_end}"
fi
if [ -n "$layer_indices" ]; then
    cmd="$cmd --layer_indices=\"${layer_indices}\""
fi
if [ -n "$layer_stride" ]; then
    cmd="$cmd --layer_stride=${layer_stride}"
fi

# Note the single example and eval setup in the output directory
cmd="$cmd \
    --output_dir=\"ckpts_1.1/s1-lora-${model_size}-r${rank}-single_idx${train_example_idx}-eval${num_eval_examples}${layer_suffix}-${uid}\" \
    --push_to_hub=${push_to_hub} \
    --save_only_model=True \
    --gradient_checkpointing=True"

echo "Training on single example at index ${train_example_idx}"
echo "Evaluating on ${num_eval_examples} random examples (excluding training example)"
echo "Saving checkpoints every ${save_steps} steps"
echo "Evaluating every ${eval_steps} steps"
echo ""

# Execute the command
eval $cmd