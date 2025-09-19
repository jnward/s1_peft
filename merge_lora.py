#!/usr/bin/env python3
"""
Merge LoRA adapter with base model for vLLM evaluation
Usage: python merge_lora.py --adapter_path ckpts/s1-lora-r16-20250624
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument("--base_model", type=str, default=None,
                       help="Base model name (auto-detected from adapter if not specified)")
    parser.add_argument("--adapter_path", type=str, required=True,
                       help="Path to LoRA adapter checkpoint")
    parser.add_argument("--output_path", type=str, default=None,
                       help="Output path for merged model (default: adapter_path + '-merged')")
    parser.add_argument("--push_to_hub", action="store_true",
                       help="Push merged model to HuggingFace Hub")
    parser.add_argument("--hub_repo", type=str, default=None,
                       help="HuggingFace Hub repository name")
    args = parser.parse_args()

    # Strip trailing slashes from adapter path
    args.adapter_path = args.adapter_path.rstrip('/')

    # Set output path
    if args.output_path is None:
        args.output_path = args.adapter_path + "-merged"
    
    # Auto-detect base model from adapter path if not specified
    if args.base_model is None:
        import re
        # Extract model size from path (format: s1-lora-{size}-r{rank}-{timestamp})
        match = re.search(r's1-lora-(\d+(?:\.\d+)?B)-r\d+', args.adapter_path)
        if not match:
            raise ValueError(
                f"Could not extract model size from adapter path: {args.adapter_path}\n"
                "Expected format: s1-lora-{{size}}-r{{rank}}-{{timestamp}} (e.g., s1-lora-7B-r16-20241224_123456)\n"
                "Please specify --base_model explicitly."
            )
        
        model_size = match.group(1)
        args.base_model = f"Qwen/Qwen2.5-{model_size}-Instruct"
        print(f"Auto-detected base model from path: {args.base_model}")
    
    print(f"Loading base model: {args.base_model}")
    # Load base model in the same precision as training
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print(f"Loading LoRA adapter: {args.adapter_path}")
    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, args.adapter_path)
    
    print("Merging adapter with base model...")
    # Merge LoRA weights into base model
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to: {args.output_path}")
    # Save merged model
    merged_model.save_pretrained(args.output_path)
    
    # Also save tokenizer for convenience
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.save_pretrained(args.output_path)
    
    if args.push_to_hub and args.hub_repo:
        print(f"Pushing to HuggingFace Hub: {args.hub_repo}")
        merged_model.push_to_hub(args.hub_repo)
        tokenizer.push_to_hub(args.hub_repo)
    
    print(f"\nMerge complete! Model saved to: {args.output_path}")

if __name__ == "__main__":
    main()