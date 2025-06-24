# LoRA Training for s1 Project

This document describes how to use the LoRA (Low-Rank Adaptation) training implementation for the s1 project.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt  # Includes peft>=0.7.0
```

### 2. Train LoRA Models with Different Ranks

```bash
# Rank 1 (minimal parameters)
bash train/sft_lora.sh --lora_rank 1 --learning_rate 1e-3

# Rank 8 
bash train/sft_lora.sh --lora_rank 8

# Rank 16 (default)
bash train/sft_lora.sh

# Rank 32
bash train/sft_lora.sh --lora_rank 32

# Rank 64
bash train/sft_lora.sh --lora_rank 64
```

### 3. Test Trained Models

```bash
python inference_lora.py \
  --adapter_path ckpts/s1-lora-r16-20250624_123456 \
  --prompt "How many r in raspberry"
```

## Configuration

### LoRA Parameters
- **Rank (r)**: 1, 8, 16, 32, 64 (controls model capacity)
- **Alpha**: Fixed at 16 (scaling factor)
- **Dropout**: 0.0 (no dropout for cleaner experiments)
- **Target modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

### Training Parameters
- **Learning rate**: 3e-4 (auto-adjusted from 1e-5 for full fine-tuning)
- **Epochs**: 5 (same as original)
- **Batch size**: Same as original setup
- **Dataset**: s1K tokenized dataset

## Files Created

1. `train/sft_lora.py` - LoRA training script
2. `train/sft_lora.sh` - Launch script for LoRA training
3. `inference_lora.py` - Simple inference script for testing
4. `requirements.txt` - Updated with PEFT dependency

## Expected Benefits

- **Memory**: ~10-20x reduction in trainable parameters
- **Speed**: Faster training, especially for multiple rank experiments
- **Storage**: Adapter files are only 10-100MB vs full model GBs
- **Flexibility**: Easy to compare different ranks and switch adapters

## Parameter Counts (Approximate)

For Qwen2.5-32B:
- Rank 1: ~0.01% trainable parameters
- Rank 8: ~0.08% trainable parameters  
- Rank 16: ~0.16% trainable parameters
- Rank 32: ~0.32% trainable parameters
- Rank 64: ~0.64% trainable parameters

vs Full fine-tuning: 100% trainable parameters