#!/bin/bash
# Evaluation script for LoRA models with vLLM
# Usage: bash eval_lora.sh ckpts/s1-lora-r16-20250624

# Fix for H200 NCCL CUDA error 999 - disable NVLS (NVLink SHARP)
export NCCL_NVLS_ENABLE=0
export NCCL_ALGO=Ring
export NCCL_DEBUG=WARN
export CUDA_LAUNCH_BLOCKING=0

ADAPTER_PATH=$1
if [ -z "$ADAPTER_PATH" ]; then
    echo "Usage: bash eval_lora.sh <adapter_path>"
    exit 1
fi

# Strip trailing slashes from adapter path
ADAPTER_PATH=${ADAPTER_PATH%/}

# Extract rank and model size from path (format: s1-lora-{size}-r{rank}-{timestamp})
RANK=$(echo $ADAPTER_PATH | sed -n 's/.*-r\([0-9]*\)-.*/\1/p')
MODEL_SIZE=$(echo $ADAPTER_PATH | sed -n 's/.*s1-lora-\([0-9]*B\)-r.*/\1/p')
MERGED_PATH="${ADAPTER_PATH}-merged"

# Validate extraction
if [ -z "$MODEL_SIZE" ]; then
    echo "Error: Could not extract model size from adapter path: $ADAPTER_PATH"
    echo "Expected format: s1-lora-{size}-r{rank}-{timestamp} (e.g., s1-lora-7B-r16-20241224_123456)"
    exit 1
fi

if [ -z "$RANK" ]; then
    echo "Error: Could not extract rank from adapter path: $ADAPTER_PATH"
    echo "Expected format: s1-lora-{size}-r{rank}-{timestamp} (e.g., s1-lora-7B-r16-20241224_123456)"
    exit 1
fi

echo "Evaluating LoRA model: ${MODEL_SIZE} with rank ${RANK}"
echo "Adapter path: $ADAPTER_PATH"
echo "Merged model path: $MERGED_PATH"

# Step 1: Merge LoRA adapter with base model
echo "Step 1: Merging LoRA adapter..."
python merge_lora.py --adapter_path $ADAPTER_PATH --output_path $MERGED_PATH
if [ $? -ne 0 ]; then
    echo "Error: Failed to merge LoRA adapter"
    exit 1
fi

# Step 2: Run evaluation with vLLM
echo "Step 2: Running evaluation..."
cd eval/lm-evaluation-harness


# You need to set your OpenAI API key for answer extraction
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Warning: OPENAI_API_KEY not set. Answer extraction may fail."
fi

PROCESSOR=gpt-4o-mini lm_eval \
    --model vllm \
    --model_args "pretrained=${PWD}/../../${MERGED_PATH},dtype=bfloat16,tensor_parallel_size=8" \
    --tasks aime24_nofigures,openai_math \
    --batch_size auto \
    --apply_chat_template \
    --output_path "../../results/lora_${MODEL_SIZE}_r${RANK}" \
    --log_samples \
    --gen_kwargs "max_gen_toks=20000"

cd ../..

echo "Evaluation complete! Results saved to: results/lora_${MODEL_SIZE}_r${RANK}"

# Optional: Clean up merged model to save space
rm -rf $MERGED_PATH