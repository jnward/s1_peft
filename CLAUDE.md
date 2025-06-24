# S1 LoRA Implementation Project

## Project Overview
This project extends the s1 (Simple test-time scaling) work by implementing LoRA (Low-Rank Adaptation) training instead of full model fine-tuning. The goal is to study how different LoRA ranks affect reasoning performance while dramatically reducing computational requirements.

**Original s1 project**: Achieves o1-preview level reasoning with just 1,000 training examples using test-time scaling and budget forcing techniques.

**Our extension**: Replace full fine-tuning with LoRA to enable efficient rank experiments and reduce memory/compute requirements.

## Key Decisions Made

### LoRA Configuration
- **Fixed alpha = 16**: Instead of scaling with rank (alpha = 2*rank), we use a fixed value for cleaner rank comparisons
- **No dropout**: Set to 0.0 across all ranks, especially important for rank-1 experiments where dropout would be counterproductive
- **Target modules**: Include all linear layers - attention (q_proj, k_proj, v_proj, o_proj) AND MLP (gate_proj, up_proj, down_proj)
- **Rank experiments**: Test ranks 1, 2, 4, 8, 16, 32, 64, 128 to study parameter efficiency vs capacity tradeoffs

### Training Parameters
- **Learning rate**: 3e-4 (3x higher than full fine-tuning's 1e-5) to account for LoRA training dynamics
- **Base model**: Qwen/Qwen2.5-32B-Instruct (same as original)
- **Dataset**: s1K tokenized dataset (1,000 examples)
- **Epochs**: 5 (same as original)
- **FSDP compatibility**: Works with existing distributed training setup

### Evaluation Strategy
- **No budget forcing**: We're not interested in test-time compute scaling, focusing purely on parameter efficiency
- **Merge-then-vLLM approach**: Merge LoRA adapters with base model for 10x faster evaluation vs HuggingFace backend
- **Benchmarks**: AIME24, MATH500, GPQA for comprehensive reasoning evaluation
- **Parallel evaluation**: Split benchmarks across GPU groups to maximize utilization

## Implementation Files

### Training
- `train/sft_lora.py` - LoRA training script with PEFT integration
- `train/sft_lora.sh` - Launch script with configurable rank parameter
- `requirements.txt` - Updated with `peft>=0.7.0` dependency

### Evaluation  
- `merge_lora.py` - Merges LoRA adapters with base model for vLLM compatibility
- `eval_lora.sh` - Streamlined evaluation workflow
- `eval_parallel.sh` - Parallel evaluation on split GPU groups
- `inference_lora.py` - Simple inference script for testing adapters

### Documentation
- `LORA_USAGE.md` - Usage guide and parameter explanations
- `CLAUDE.md` - This project summary (for future Claude sessions)

## Workflow

### Training Different Ranks
```bash
# Rank 1 (extreme parameter efficiency)
bash train/sft_lora.sh --lora_rank 1 --learning_rate 1e-3

# Rank 2
bash train/sft_lora.sh --lora_rank 2 --learning_rate 1e-3

# Standard ranks
bash train/sft_lora.sh --lora_rank 4
bash train/sft_lora.sh --lora_rank 8
bash train/sft_lora.sh --lora_rank 16  # default
bash train/sft_lora.sh --lora_rank 32
bash train/sft_lora.sh --lora_rank 64
bash train/sft_lora.sh --lora_rank 128
```

### Evaluation
```bash
# Merge adapter for vLLM compatibility
python merge_lora.py --adapter_path ckpts/s1-lora-r16-timestamp

# Fast evaluation with vLLM
bash eval_lora.sh ckpts/s1-lora-r16-timestamp

# Or parallel evaluation across GPUs
export OPENAI_API_KEY=your_key
bash eval_parallel.sh /path/to/model
```

## Technical Insights

### Memory Requirements
- **Full fine-tuning**: ~350GB across 16 H100s
- **LoRA training**: ~165GB, likely fits on 1 node (8 H100s)
- **Parameter reduction**: 99.95% fewer trainable parameters for rank-1

## Research Questions
1. What's the minimum LoRA rank that maintains reasonable s1 reasoning performance?
2. How does rank scale with performance on mathematical reasoning tasks?
3. Can rank-1 LoRA learn meaningful adaptations on s1K dataset?
4. What's the performance/efficiency Pareto frontier for different ranks?

## Repository Structure
- Fork of original s1 repo: https://github.com/jnward/s1_peft
- Maintains compatibility with original evaluation infrastructure
- Adds LoRA training and evaluation capabilities
- Preserves all original s1 functionality

## Next Steps
- Train models across different ranks
- Run comprehensive evaluation on AIME/MATH500/GPQA
- Analyze parameter efficiency vs performance tradeoffs
- Compare against original full fine-tuning results