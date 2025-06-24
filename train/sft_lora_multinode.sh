#!/bin/bash
# Multi-node LoRA training script for s1 project
# This script expects environment variables set by your cluster scheduler:
# - NUM_NODES: total number of nodes
# - REPLICA_RANK: rank of this node (0 for master, 1+ for workers)
# - REPLICA_HOSTNAME: hostname/IP of the master node

uid="$(date +%Y%m%d_%H%M%S)"
base_model="Qwen/Qwen2.5-7B-Instruct"
block_size=20000
lr=5e-4
min_lr=0
epochs=5
weight_decay=1e-4
micro_batch_size=1  # If 2 nodes with 8 gpus each, batch_size will be 16
gradient_accumulation_steps=1
max_steps=-1
gpu_count=$(nvidia-smi -L | wc -l)
push_to_hub=false

# LoRA specific parameters
rank=16
alpha=16  # Fixed alpha for RSLoRA

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
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

echo "Running on node ${REPLICA_RANK} of ${NUM_NODES} with ${gpu_count} GPUs"
echo "Master node: ${REPLICA_HOSTNAME}:29401"

torchrun \
    --nnodes ${NUM_NODES}:${NUM_NODES} \
    --node_rank=$REPLICA_RANK \
    --nproc-per-node ${gpu_count} \
    --rdzv_id=12347 \
    --rdzv_backend=c10d \
    --rdzv_conf='read_timeout=420' \
    --rdzv_endpoint=$REPLICA_HOSTNAME:29401 \
    train/sft_lora.py \
    --block_size=${block_size} \
    --per_device_train_batch_size=${micro_batch_size} \
    --per_device_eval_batch_size=${micro_batch_size} \
    --gradient_accumulation_steps=${gradient_accumulation_steps} \
    --num_train_epochs=${epochs} \
    --train_file_path="simplescaling/s1K-1.1_tokenized" \
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
    --save_only_model=True \
    --gradient_checkpointing=True