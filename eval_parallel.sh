#!/bin/bash
# Parallel evaluation script for running benchmarks on split GPUs
# Usage: bash eval_parallel.sh /path/to/model

MODEL_PATH=$1
if [ -z "$MODEL_PATH" ]; then
    echo "Usage: bash eval_parallel.sh /path/to/model"
    exit 1
fi

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable is not set"
    echo "Please run: export OPENAI_API_KEY=your_key_here"
    exit 1
fi

echo "Starting parallel evaluation of model: $MODEL_PATH"
echo "Using GPUs 0-3 for AIME + GPQA"
echo "Using GPUs 4-7 for MATH500"
echo ""

# Create results directory if it doesn't exist
mkdir -p results

# Function to run evaluation in background
run_eval() {
    local gpu_devices=$1
    local tasks=$2
    local output_name=$3
    local log_file="results/${output_name}_eval.log"
    
    echo "Starting $output_name on GPUs $gpu_devices..."
    echo "Log file: $log_file"
    
    CUDA_VISIBLE_DEVICES=$gpu_devices PROCESSOR=gpt-4o-mini lm_eval \
        --model vllm \
        --model_args pretrained=$MODEL_PATH,dtype=bfloat16,tensor_parallel_size=4 \
        --tasks $tasks \
        --batch_size auto \
        --apply_chat_template \
        --output_path results/$output_name \
        --log_samples \
        --gen_kwargs "max_gen_toks=20000" \
        > $log_file 2>&1 &
    
    echo "PID: $!"
    return $!
}

# Change to evaluation directory
cd eval/lm-evaluation-harness

# Start AIME + GPQA evaluation on GPUs 0-3
run_eval "0,1,2,3" "aime24_nofigures,gpqa_diamond_openai" "aime_gpqa"
PID1=$!

# Start MATH500 evaluation on GPUs 4-7
run_eval "4,5,6,7" "openai_math" "math500"
PID2=$!

# Monitor progress
echo ""
echo "Both evaluations running in parallel..."
echo "Monitor progress with:"
echo "  tail -f results/aime_gpqa_eval.log"
echo "  tail -f results/math500_eval.log"
echo ""

# Wait for both to complete
echo "Waiting for evaluations to complete..."
wait $PID1
STATUS1=$?
wait $PID2
STATUS2=$?

# Check results
echo ""
echo "Evaluation complete!"
echo "AIME + GPQA exit status: $STATUS1"
echo "MATH500 exit status: $STATUS2"

# Show results summary if successful
if [ $STATUS1 -eq 0 ] && [ $STATUS2 -eq 0 ]; then
    echo ""
    echo "Results saved to:"
    echo "  results/aime_gpqa/"
    echo "  results/math500/"
    echo ""
    echo "To view results:"
    echo "  cat results/aime_gpqa/results.json"
    echo "  cat results/math500/results.json"
else
    echo ""
    echo "One or more evaluations failed. Check log files for errors:"
    echo "  results/aime_gpqa_eval.log"
    echo "  results/math500_eval.log"
fi

cd ../..