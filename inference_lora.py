#!/usr/bin/env python3
"""
Simple LoRA inference script for s1 models
Usage: python inference_lora.py --adapter_path ckpts/s1-lora-r16-20250101_120000 --prompt "How many r in raspberry"
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser(description="Inference with LoRA-trained s1 models")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-32B-Instruct", 
                       help="Base model name")
    parser.add_argument("--adapter_path", type=str, required=True,
                       help="Path to LoRA adapter checkpoint")
    parser.add_argument("--prompt", type=str, default="How many r in raspberry",
                       help="Input prompt")
    parser.add_argument("--max_tokens", type=int, default=1000,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Generation temperature")
    args = parser.parse_args()

    print(f"Loading base model: {args.base_model}")
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print(f"Loading LoRA adapter: {args.adapter_path}")
    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, args.adapter_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    # Format prompt in s1 style
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": args.prompt}
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    print(f"\nPrompt:\n{prompt}")
    print("\nGenerating response...")
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            do_sample=args.temperature > 0,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    print(f"\nResponse:\n{response}")
    print(f"\nResponse length: {len(response.split())} words")

if __name__ == "__main__":
    main()