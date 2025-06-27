export NCCL_NVLS_ENABLE=0
export NCCL_ALGO=Ring
export NCCL_DEBUG=WARN
export CUDA_LAUNCH_BLOCKING=0

MODEL_PATH=$1
if [ -z "$MODEL_PATH" ]; then
    echo "Usage: bash eval.sh <model_path>"
    exit 1
fi

# Strip trailing slashes from adapter path
MODEL_PATH=${MODEL_PATH%/}

MODEL_SIZE=32B

cd eval/lm-evaluation-harness

PROCESSOR=gpt-4o-mini lm_eval \
    --model vllm \
    --model_args "pretrained=${PWD}/../../${MODEL_PATH},dtype=bfloat16,tensor_parallel_size=2" \
    --tasks aime24_nofigures,openai_math \
    --batch_size auto \
    --apply_chat_template \
    --output_path "../../results/${MODEL_SIZE}_ft" \
    --log_samples \
    --gen_kwargs "max_gen_toks=20000"

cd ../..

echo "Evaluation complete! Results saved to: results/${MODEL_SIZE}_ft"
