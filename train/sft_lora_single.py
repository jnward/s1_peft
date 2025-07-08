import os
from dataclasses import dataclass, field, asdict
from typing import Optional
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from datasets import load_dataset, concatenate_datasets, DatasetDict
import transformers
import trl
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np

@dataclass
class TrainingConfig:
    model_name: str = field(default="Qwen/Qwen2.5-32B-Instruct")
    block_size: int = field(default=32768)
    wandb_project: Optional[str] = field(default="s1-peft")
    wandb_entity: Optional[str] = field(default="jnward")
    train_file_path: Optional[str] = field(default='simplescaling/s1K_tokenized')
    dagger: bool = field(default=False)
    # LoRA specific parameters
    use_lora: bool = field(default=True)
    rank: int = field(default=16)
    alpha: int = field(default=32)  # Default alpha = 2*default_rank
    layer_start: Optional[int] = field(default=None)  # Starting layer index (0-based)
    layer_end: Optional[int] = field(default=None)    # Ending layer index (exclusive)
    layer_indices: Optional[str] = field(default=None)  # Comma-separated list of specific layers
    layer_stride: Optional[int] = field(default=None)   # Train every nth layer
    # Single example training parameters
    train_example_idx: int = field(default=732)  # Index of the single example to train on
    num_eval_examples: int = field(default=100)  # Number of random validation examples
    eval_seed: int = field(default=42)  # Seed for random validation sampling
    mlp_only: bool = field(default=False)  # If True, only train MLP layers; if False, train all layers

    def __post_init__(self):
        os.environ['WANDB_PROJECT'] = self.wandb_project
        os.environ['WANDB_ENTITY'] = self.wandb_entity

def train():
    # parsing input
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, args = parser.parse_args_into_dataclasses()
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")

    # loading model
    kwargs = {}
    if "70B" in config.model_name:
        # Removed "low_cpu_mem_usage": True, for 70B, since by default we are in FSDP,
        # it's more efficient to do  "cpu_ram_efficient_loading": true, in fsdp_config.json
        kwargs = {"device_map": "auto", "torch_dtype": "auto",
                  "attn_implementation": "flash_attention_2", "use_cache": False}
        model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name, **kwargs)
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name)

    # Enable gradient checkpointing if requested (must be done BEFORE LoRA)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        # Required for gradient checkpointing to work with PEFT
        model.enable_input_require_grads()
        logging.info("Enabled gradient checkpointing with use_reentrant=False for PEFT compatibility")

    # Apply LoRA if enabled
    if config.use_lora:
        # Choose target modules based on mlp_only flag
        if config.mlp_only:
            # Only target MLP projection layers
            target_modules = ["gate_proj", "up_proj", "down_proj"]
            logging.info("Training MLP layers only")
        else:
            # Target all linear layers (attention + MLP)
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            logging.info("Training all layers (attention + MLP)")
        
        # Determine which layers to train
        if any([config.layer_start is not None, config.layer_end is not None, 
                config.layer_indices is not None, config.layer_stride is not None]):
            # Get total number of layers
            if hasattr(model.config, 'num_hidden_layers'):
                total_layers = model.config.num_hidden_layers
            else:
                # Fallback for different model architectures
                total_layers = len([n for n, _ in model.named_modules() if '.mlp.gate_proj' in n])
            
            # Determine which layers to train based on configuration
            if config.layer_indices is not None:
                # Specific layer indices provided
                layer_indices = [int(idx.strip()) for idx in config.layer_indices.split(',')]
                # Validate indices
                for idx in layer_indices:
                    if idx < 0 or idx >= total_layers:
                        raise ValueError(f"Invalid layer index {idx} for model with {total_layers} layers")
                logging.info(f"Training specific layers: {layer_indices}")
            elif config.layer_stride is not None:
                # Train every nth layer
                layer_start = config.layer_start if config.layer_start is not None else 0
                layer_end = config.layer_end if config.layer_end is not None else total_layers
                layer_indices = list(range(layer_start, layer_end, config.layer_stride))
                logging.info(f"Training every {config.layer_stride} layer(s) from {layer_start} to {layer_end}: {layer_indices}")
            else:
                # Standard range-based selection
                layer_start = config.layer_start if config.layer_start is not None else 0
                layer_end = config.layer_end if config.layer_end is not None else total_layers
                
                # Validate range
                if layer_start < 0 or layer_end > total_layers or layer_start >= layer_end:
                    raise ValueError(f"Invalid layer range: {layer_start}-{layer_end} for model with {total_layers} layers")
                
                layer_indices = list(range(layer_start, layer_end))
                logging.info(f"Training layers {layer_start}-{layer_end} out of {total_layers} total layers")
            
            # Build specific layer patterns
            layer_specific_modules = []
            for layer_idx in layer_indices:
                for module in target_modules:
                    # Pattern varies by model architecture
                    if "Qwen" in config.model_name:
                        if module in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                            layer_specific_modules.append(f"model.layers.{layer_idx}.self_attn.{module}")
                        else:
                            layer_specific_modules.append(f"model.layers.{layer_idx}.mlp.{module}")
                    elif "Llama" in config.model_name:
                        if module in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                            layer_specific_modules.append(f"model.layers.{layer_idx}.self_attn.{module}")
                        else:
                            layer_specific_modules.append(f"model.layers.{layer_idx}.mlp.{module}")
                    else:
                        # Generic pattern
                        if module in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                            layer_specific_modules.append(f"layers.{layer_idx}.self_attn.{module}")
                        else:
                            layer_specific_modules.append(f"layers.{layer_idx}.mlp.{module}")
            
            target_modules = layer_specific_modules
            logging.info(f"Total modules to be adapted: {len(target_modules)}")
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.rank,
            lora_alpha=config.alpha,
            lora_dropout=0.0,  # No dropout as discussed
            target_modules=target_modules,
            bias="none",
            use_rslora=True,  # Use rank-stabilized LoRA scaling
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        # Adjust learning rate for LoRA if not manually set
        if args.learning_rate == 2e-5:  # Default transformers learning rate
            args.learning_rate = 3e-4
            logging.info(f"Adjusted learning rate to {args.learning_rate} for LoRA training")

    # Load dataset
    dataset = load_dataset(config.train_file_path)
    
    # Extract single training example
    train_example_idx = config.train_example_idx
    if train_example_idx >= len(dataset['train']):
        raise ValueError(f"Train example index {train_example_idx} is out of bounds. Dataset has {len(dataset['train'])} examples.")
    
    # Get the single training example
    single_example = dataset['train'].select([train_example_idx])
    
    # Print the question from the training example
    logging.info("="*80)
    logging.info(f"Training on single example at index {train_example_idx}")
    logging.info(f"Question: {single_example[0]['question']}")
    logging.info(f"Solution: {single_example[0]['solution']}")
    logging.info("="*80)
    
    # Create validation dataset by excluding the training example
    all_indices = list(range(len(dataset['train'])))
    all_indices.remove(train_example_idx)
    
    # Randomly sample num_eval_examples from remaining indices
    np.random.seed(config.eval_seed)
    eval_indices = np.random.choice(all_indices, size=min(config.num_eval_examples, len(all_indices)), replace=False)
    eval_dataset = dataset['train'].select(eval_indices.tolist())
    
    logging.info(f"Created validation dataset with {len(eval_dataset)} examples (excluding training example)")
    
    # Configure wandb run name
    if args.report_to and "wandb" in args.report_to:
        import wandb
        # Build run name with single example and layer range
        run_name_parts = [f"s1-lora-r{config.rank}-single_idx{train_example_idx}"]
        
        # Add layer specification to run name
        if config.layer_indices is not None:
            # Specific indices - abbreviate if too many
            indices = [int(idx.strip()) for idx in config.layer_indices.split(',')]
            if len(indices) > 5:
                run_name_parts.append(f"l{indices[0]}...{indices[-1]}_{len(indices)}layers")
            else:
                run_name_parts.append(f"l{','.join(map(str, indices))}")
        elif config.layer_stride is not None:
            layer_start = config.layer_start if config.layer_start is not None else 0
            layer_end = config.layer_end if config.layer_end is not None else "all"
            run_name_parts.append(f"l{layer_start}-{layer_end}-stride{config.layer_stride}")
        elif config.layer_start is not None or config.layer_end is not None:
            layer_start = config.layer_start if config.layer_start is not None else 0
            layer_end = config.layer_end if config.layer_end is not None else "all"
            run_name_parts.append(f"l{layer_start}-{layer_end}")
        
        # Add timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name_parts.append(timestamp)
        
        args.run_name = "-".join(run_name_parts)
        logging.info(f"Wandb run name: {args.run_name}")

    # setting up trainer
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if "Llama" in config.model_name:
        instruction_template = "<|start_header_id|>user<|end_header_id|>"
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        # Use a token that is never used
        tokenizer.pad_token = "<|reserved_special_token_5|>"
    elif "Qwen" in config.model_name:
        instruction_template = "<|im_start|>user"
        response_template = "<|im_start|>assistant\n"
        # Use a token that is never used
        tokenizer.pad_token = "<|fim_pad|>"

    # Only compute loss over assistant responses
    # Verified that it precisely starts where the thinking tokens start and ends with the first pad token
    # via labels being set to -100
    collator = trl.DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )
    args.dataset_text_field = 'text'
    args.max_seq_length = config.block_size
    
    # Override evaluation strategy to use steps instead of epochs
    # Since we only have 1 example, we need to be explicit about when to evaluate
    if args.eval_strategy == "epoch":
        args.eval_strategy = "steps"
        if not hasattr(args, 'eval_steps') or args.eval_steps is None:
            args.eval_steps = 50  # Default to evaluate every 50 steps
        logging.info(f"Changed eval_strategy to 'steps' with eval_steps={args.eval_steps}")
    
    # Also ensure save strategy is step-based
    if args.save_strategy == "epoch":
        args.save_strategy = "steps"
        if not hasattr(args, 'save_steps') or args.save_steps is None:
            args.save_steps = args.eval_steps  # Save at same frequency as eval
        logging.info(f"Changed save_strategy to 'steps' with save_steps={args.save_steps}")
    
    trainer = trl.SFTTrainer(
        model,
        train_dataset=single_example,  # Single example dataset
        eval_dataset=eval_dataset,     # Random validation set
        args=args,
        data_collator=collator
    )

    trainer.train()
    
    # Save model - PEFT will automatically save only the adapter weights
    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()