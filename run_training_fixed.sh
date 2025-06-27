#\!/bin/bash

# Fix for H200 NCCL CUDA error 999 - disable NVLS (NVLink SHARP)
export NCCL_NVLS_ENABLE=0
export NCCL_ALGO=Ring
export NCCL_DEBUG=WARN
export CUDA_LAUNCH_BLOCKING=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Run the training script
bash train/sft_lora.sh "$@"
# bash train/sft.sh "$@"./