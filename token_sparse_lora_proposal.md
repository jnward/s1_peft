# Token-Sparse LoRA: Interpretable Rank-1 Adapters with Token-wise Sparsity

## Executive Summary

This proposal outlines a novel approach to make LoRA (Low-Rank Adaptation) more interpretable by enforcing token-wise sparsity constraints. Instead of allowing LoRA neurons to activate for all tokens, we restrict each neuron to activate for only the top-k most relevant tokens in a sequence. This creates specialized, interpretable feature detectors while maintaining the efficiency benefits of LoRA.

## Motivation

### Current Limitations
- Standard LoRA neurons activate for all tokens, making it difficult to understand what each neuron represents
- Dense activations provide no insight into which tokens are most important for each learned feature
- Attribution analysis shows that most neuron activations contribute minimally to model outputs

### Proposed Solution
Apply a top-k sparsity constraint over the token dimension:
- Each LoRA neuron can only activate for k tokens (e.g., 16 out of 512)
- Forces neurons to specialize and become selective feature detectors
- Creates interpretable activation patterns showing exactly which tokens each neuron "cares about"

## Technical Approach

### Core Concept

For a rank-1 LoRA adapter with activation shape `[batch_size, seq_len, 1]`:
1. Compute activations normally through the LoRA A matrix
2. Apply ReLU to ensure positivity (optional but recommended)
3. Select top-k activations across the sequence dimension
4. Zero out all non-top-k activations
5. Continue with LoRA B matrix computation

### Mathematical Formulation

Given:
- Input: `x ∈ ℝ^{batch × seq_len × d_model}`
- LoRA A matrix: `W_A ∈ ℝ^{d_model × r}` (r=1 for rank-1)
- LoRA B matrix: `W_B ∈ ℝ^{r × d_model}`
- Sparsity parameter: `k` (number of active tokens)

Forward pass:
```
a = ReLU(xW_A)                    # [batch, seq_len, 1]
mask = TopK(a, k, dim=1)          # Binary mask [batch, seq_len, 1]
a_sparse = a ⊙ mask               # Element-wise multiplication
output = a_sparse W_B × scaling   # [batch, seq_len, d_model]
```

## Implementation Details

### 1. Configuration

```python
@dataclass
class TokenSparseLoRAConfig:
    """Configuration for token-sparse LoRA training"""
    
    # Standard LoRA parameters
    r: int = 1                      # LoRA rank
    lora_alpha: float = 1.0         # LoRA scaling factor
    lora_dropout: float = 0.0       # Dropout probability
    target_modules: List[str] = field(
        default_factory=lambda: ["gate_proj", "up_proj", "down_proj"]
    )
    
    # Sparsity parameters
    k_tokens: int = 16              # Max active tokens per neuron
    sparsity_mode: str = "hard"     # "hard" for discrete, "soft" for continuous
    enforce_positive: bool = True    # Apply ReLU before sparsity
    
    # Training dynamics
    sparsity_warmup_steps: int = 1000   # Steps to reach target sparsity
    sparsity_lambda: float = 0.1         # Weight for sparsity regularization
    
    # Adaptive sparsity
    scale_k_with_seqlen: bool = True    # Scale k based on sequence length
    min_k_ratio: float = 0.03           # Minimum 3% of tokens active
    max_k_ratio: float = 0.1            # Maximum 10% of tokens active
```

### 2. Core Layer Implementation

```python
class TokenSparseLoRALayer(nn.Module):
    """LoRA layer with token-wise sparsity constraint"""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: TokenSparseLoRAConfig,
    ):
        super().__init__()
        self.config = config
        
        # Initialize LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, config.r))
        self.lora_B = nn.Parameter(torch.zeros(config.r, out_features))
        
        # Initialize with Kaiming for A, zeros for B
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        self.scaling = config.lora_alpha / config.r
        self.dropout = nn.Dropout(config.lora_dropout)
        
        # For tracking statistics
        self.register_buffer('activation_counts', torch.zeros(1))
        self.register_buffer('sparsity_mask', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with token-wise sparsity"""
        
        # Standard LoRA A projection
        x_dropout = self.dropout(x)
        a_output = F.linear(x_dropout, self.lora_A.T)  # [batch, seq_len, r]
        
        # Apply positivity constraint
        if self.config.enforce_positive:
            a_output = F.relu(a_output)
        
        # Apply token-wise sparsity
        a_sparse, mask = self._apply_token_sparsity(a_output)
        
        # Store mask for analysis
        self.sparsity_mask = mask.detach()
        
        # LoRA B projection
        output = F.linear(a_sparse, self.lora_B.T)  # [batch, seq_len, out_features]
        
        return output * self.scaling
    
    def _apply_token_sparsity(
        self, 
        activations: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply top-k sparsity over sequence dimension"""
        
        batch_size, seq_len, rank = activations.shape
        
        # Determine k based on sequence length
        if self.config.scale_k_with_seqlen:
            min_k = int(seq_len * self.config.min_k_ratio)
            max_k = int(seq_len * self.config.max_k_ratio)
            k = max(min_k, min(self.config.k_tokens, max_k))
        else:
            k = min(self.config.k_tokens, seq_len)
        
        if self.config.sparsity_mode == "hard":
            # Hard top-k with straight-through estimator
            mask = self._hard_topk_mask(activations, k)
            sparse_output = activations * mask
            
        else:  # soft mode
            # Differentiable relaxation using sigmoid
            sparse_output, mask = self._soft_topk(activations, k)
        
        # Update statistics
        self.activation_counts += mask.sum().detach()
        
        return sparse_output, mask
    
    def _hard_topk_mask(
        self, 
        activations: torch.Tensor, 
        k: int
    ) -> torch.Tensor:
        """Create binary mask for top-k activations"""
        
        # Get top-k values and indices
        topk_values, topk_indices = torch.topk(
            activations, 
            k=k, 
            dim=1, 
            largest=True
        )
        
        # Create binary mask
        mask = torch.zeros_like(activations)
        mask.scatter_(1, topk_indices, 1.0)
        
        # Detach mask to implement straight-through estimator
        mask = mask.detach()
        
        return mask
    
    def _soft_topk(
        self, 
        activations: torch.Tensor, 
        k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Differentiable approximation of top-k"""
        
        # Find the k-th largest value (threshold)
        kth_values, _ = torch.kthvalue(
            activations, 
            activations.size(1) - k + 1, 
            dim=1, 
            keepdim=True
        )
        
        # Soft mask using sigmoid
        temperature = 0.1  # Controls sharpness
        soft_mask = torch.sigmoid((activations - kth_values) / temperature)
        
        sparse_output = activations * soft_mask
        
        return sparse_output, soft_mask
```

### 3. Integration with PEFT

```python
class TokenSparsePeftModel(PeftModel):
    """PEFT model with token-sparse LoRA layers"""
    
    @classmethod
    def from_pretrained(
        cls,
        model: PreTrainedModel,
        model_id: str,
        config: TokenSparseLoRAConfig,
        **kwargs
    ):
        """Load pretrained model and add sparse LoRA adapters"""
        
        # Create PEFT config
        peft_config = LoraConfig(
            r=config.r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
        )
        
        # Initialize PEFT model
        peft_model = super().from_pretrained(
            model, 
            model_id, 
            config=peft_config,
            **kwargs
        )
        
        # Replace LoRA layers with sparse versions
        cls._replace_with_sparse_lora(peft_model, config)
        
        return peft_model
    
    @staticmethod
    def _replace_with_sparse_lora(
        model: nn.Module, 
        config: TokenSparseLoRAConfig
    ):
        """Replace standard LoRA layers with sparse versions"""
        
        for name, module in model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                # Extract dimensions
                in_features = module.lora_A[module.active_adapter].in_features
                out_features = module.lora_B[module.active_adapter].out_features
                
                # Create sparse layer
                sparse_layer = TokenSparseLoRALayer(
                    in_features, 
                    out_features, 
                    config
                )
                
                # Copy weights
                sparse_layer.lora_A.data = module.lora_A[module.active_adapter].weight.data.T
                sparse_layer.lora_B.data = module.lora_B[module.active_adapter].weight.data.T
                
                # Replace in parent
                parent_name = '.'.join(name.split('.')[:-1])
                attr_name = name.split('.')[-1]
                parent = model.get_submodule(parent_name)
                setattr(parent, attr_name, sparse_layer)
```

### 4. Training Components

#### Sparsity Scheduler

```python
class SparsityWarmupScheduler:
    """Gradually increase sparsity during training"""
    
    def __init__(
        self,
        model: nn.Module,
        config: TokenSparseLoRAConfig,
        num_training_steps: int
    ):
        self.model = model
        self.config = config
        self.num_training_steps = num_training_steps
        self.warmup_steps = config.sparsity_warmup_steps
        
    def step(self, current_step: int):
        """Update sparsity based on training progress"""
        
        if current_step >= self.warmup_steps:
            return  # Warmup complete
        
        # Linear warmup from 50% sparsity to target
        progress = current_step / self.warmup_steps
        
        for module in self.model.modules():
            if isinstance(module, TokenSparseLoRALayer):
                # Temporarily adjust k_tokens
                original_k = module.config.k_tokens
                warmup_k = int(original_k * (2 - progress))  # 2x to 1x
                module.current_k = warmup_k
```

#### Regularization

```python
class SparsityRegularizer:
    """Regularization terms to encourage useful sparsity patterns"""
    
    def __init__(self, config: TokenSparseLoRAConfig):
        self.config = config
        
    def compute_loss(self, model: nn.Module) -> torch.Tensor:
        """Compute sparsity-related regularization losses"""
        
        total_loss = 0.0
        
        for name, module in model.named_modules():
            if isinstance(module, TokenSparseLoRALayer):
                # L1 regularization on activations (encourage sparsity)
                if hasattr(module, 'last_activations'):
                    l1_loss = module.last_activations.abs().mean()
                    total_loss += self.config.sparsity_lambda * l1_loss
                
                # Entropy regularization (encourage decisive on/off)
                if module.sparsity_mask is not None:
                    # Compute entropy of activation pattern
                    p = module.sparsity_mask.mean(dim=0)  # [seq_len, 1]
                    entropy = -(p * torch.log(p + 1e-8) + 
                              (1-p) * torch.log(1-p + 1e-8)).mean()
                    total_loss -= 0.01 * entropy  # Negative to minimize entropy
                
                # Coverage regularization (ensure neurons use their budget)
                if module.activation_counts > 0:
                    usage_rate = module.sparsity_mask.sum() / module.sparsity_mask.numel()
                    target_rate = module.config.k_tokens / module.sparsity_mask.size(1)
                    coverage_loss = (usage_rate - target_rate).pow(2)
                    total_loss += 0.1 * coverage_loss
        
        return total_loss
```

#### Metrics and Monitoring

```python
class SparsityMetrics:
    """Track and analyze sparsity patterns"""
    
    @staticmethod
    def compute_metrics(model: nn.Module) -> Dict[str, float]:
        """Compute comprehensive sparsity metrics"""
        
        metrics = {}
        all_sparsity_rates = []
        all_coverage_rates = []
        all_specialization_scores = []
        
        for name, module in model.named_modules():
            if isinstance(module, TokenSparseLoRALayer):
                if module.sparsity_mask is None:
                    continue
                
                mask = module.sparsity_mask  # [batch, seq_len, 1]
                
                # Sparsity rate (fraction of zeros)
                sparsity_rate = 1 - mask.mean().item()
                all_sparsity_rates.append(sparsity_rate)
                
                # Coverage (fraction of tokens each neuron uses)
                coverage = mask.sum(dim=1).float() / mask.size(1)
                coverage_rate = coverage.mean().item()
                all_coverage_rates.append(coverage_rate)
                
                # Specialization (how concentrated activations are)
                # High score = activations concentrated on few tokens
                if mask.sum() > 0:
                    activation_probs = mask / (mask.sum(dim=1, keepdim=True) + 1e-8)
                    entropy = -(activation_probs * torch.log(activation_probs + 1e-8)).sum(dim=1)
                    max_entropy = torch.log(torch.tensor(mask.size(1), dtype=torch.float32))
                    specialization = 1 - (entropy / max_entropy).mean().item()
                    all_specialization_scores.append(specialization)
                
                # Per-layer metrics
                metrics[f"{name}/sparsity_rate"] = sparsity_rate
                metrics[f"{name}/coverage_rate"] = coverage_rate
                metrics[f"{name}/specialization"] = specialization
        
        # Aggregate metrics
        if all_sparsity_rates:
            metrics["global/mean_sparsity"] = np.mean(all_sparsity_rates)
            metrics["global/mean_coverage"] = np.mean(all_coverage_rates)
            metrics["global/mean_specialization"] = np.mean(all_specialization_scores)
        
        return metrics
    
    @staticmethod
    def visualize_activation_patterns(
        model: nn.Module,
        tokenizer,
        sample_text: str,
        save_path: str
    ):
        """Visualize which tokens activate which neurons"""
        
        # Tokenize
        inputs = tokenizer(sample_text, return_tensors="pt")
        
        # Forward pass
        with torch.no_grad():
            _ = model(**inputs)
        
        # Collect activation patterns
        activation_patterns = {}
        
        for name, module in model.named_modules():
            if isinstance(module, TokenSparseLoRALayer):
                if module.sparsity_mask is not None:
                    mask = module.sparsity_mask[0, :, 0].cpu().numpy()
                    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
                    
                    # Find active tokens
                    active_indices = np.where(mask > 0.5)[0]
                    active_tokens = [tokens[i] for i in active_indices]
                    
                    activation_patterns[name] = {
                        'active_indices': active_indices.tolist(),
                        'active_tokens': active_tokens,
                        'activation_strength': mask[active_indices].tolist()
                    }
        
        # Save visualization
        import json
        with open(save_path, 'w') as f:
            json.dump({
                'text': sample_text,
                'tokens': tokenizer.convert_ids_to_tokens(inputs.input_ids[0]),
                'activation_patterns': activation_patterns
            }, f, indent=2)
```

### 5. Complete Training Pipeline

```python
def train_token_sparse_lora(
    model_name: str,
    dataset_name: str,
    output_dir: str,
    config: TokenSparseLoRAConfig,
    training_args: TrainingArguments
):
    """Complete training pipeline for token-sparse LoRA"""
    
    # 1. Load base model and tokenizer
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 2. Create token-sparse LoRA model
    print("Initializing token-sparse LoRA...")
    model = create_token_sparse_lora_model(base_model, config)
    
    # 3. Load and preprocess dataset
    print("Loading dataset...")
    dataset = load_dataset(dataset_name)
    
    def preprocess_function(examples):
        # Tokenize and prepare for causal LM
        model_inputs = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512
        )
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs
    
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    # 4. Initialize training components
    sparsity_scheduler = SparsityWarmupScheduler(
        model, 
        config, 
        training_args.max_steps
    )
    sparsity_regularizer = SparsityRegularizer(config)
    
    # 5. Custom trainer with sparsity support
    class TokenSparseTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            # Standard language modeling loss
            outputs = model(**inputs)
            lm_loss = outputs.loss
            
            # Add sparsity regularization
            sparsity_loss = sparsity_regularizer.compute_loss(model)
            
            loss = lm_loss + sparsity_loss
            
            return (loss, outputs) if return_outputs else loss
        
        def training_step(self, model, inputs):
            # Standard training step
            loss = super().training_step(model, inputs)
            
            # Update sparsity schedule
            sparsity_scheduler.step(self.state.global_step)
            
            # Log sparsity metrics
            if self.state.global_step % 100 == 0:
                metrics = SparsityMetrics.compute_metrics(model)
                self.log(metrics)
            
            return loss
    
    # 6. Initialize trainer
    trainer = TokenSparseTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
    )
    
    # 7. Train
    print("Starting training...")
    trainer.train()
    
    # 8. Save model
    print("Saving model...")
    trainer.save_model(output_dir)
    
    # 9. Analyze learned patterns
    print("Analyzing activation patterns...")
    SparsityMetrics.visualize_activation_patterns(
        model,
        tokenizer,
        "The quick brown fox jumps over the lazy dog.",
        os.path.join(output_dir, "activation_patterns.json")
    )
    
    return model, trainer

# Example usage
if __name__ == "__main__":
    config = TokenSparseLoRAConfig(
        r=1,
        k_tokens=16,
        target_modules=["gate_proj", "up_proj", "down_proj"],
        sparsity_mode="hard",
        enforce_positive=True,
        scale_k_with_seqlen=True,
        sparsity_warmup_steps=1000,
        sparsity_lambda=0.1
    )
    
    training_args = TrainingArguments(
        output_dir="./sparse-lora-qwen",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=100,
        save_steps=1000,
        evaluation_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        push_to_hub=False,
        report_to=["tensorboard"],
        gradient_checkpointing=True,
        bf16=True,
    )
    
    model, trainer = train_token_sparse_lora(
        model_name="Qwen/Qwen2.5-32B-Instruct",
        dataset_name="HuggingFaceH4/ultrachat_200k",
        output_dir="./sparse-lora-output",
        config=config,
        training_args=training_args
    )
```

## Expected Benefits

### 1. Interpretability
- Each neuron becomes a specialized feature detector
- Clear activation patterns show which tokens trigger each neuron
- Can identify neurons that detect specific concepts, syntax patterns, or semantic features

### 2. Efficiency
- Sparse computations can be optimized with specialized kernels
- Reduced memory bandwidth during inference
- Potential for pruning inactive neurons

### 3. Controllability
- Can selectively activate/deactivate neurons based on their learned specializations
- Easier to debug and understand model behavior
- Potential for targeted interventions

## Evaluation Strategy

### 1. Performance Metrics
- **Perplexity**: Ensure sparse LoRA maintains competitive language modeling performance
- **Downstream Tasks**: Evaluate on standard benchmarks (MMLU, GSM8K, etc.)
- **Sparsity-Performance Trade-off**: Plot performance vs. sparsity level

### 2. Interpretability Metrics
- **Neuron Specialization Score**: Entropy of activation patterns
- **Token Coverage**: Distribution of how many neurons activate per token
- **Semantic Coherence**: Whether neurons activate for semantically related tokens

### 3. Ablation Studies
- Effect of k (number of active tokens)
- Hard vs. soft sparsity
- With/without positivity constraint
- Different warmup schedules

## Implementation Timeline

### Phase 1: Prototype (Week 1-2)
- Implement basic TokenSparseLoRALayer
- Test gradient flow and training stability
- Validate on small model (Qwen-1.8B)

### Phase 2: Integration (Week 3-4)
- Full PEFT integration
- Implement schedulers and regularizers
- Add comprehensive metrics

### Phase 3: Training (Week 5-8)
- Train on Qwen-32B with MATH dataset
- Hyperparameter tuning
- Performance optimization

### Phase 4: Analysis (Week 9-10)
- Analyze learned activation patterns
- Create visualization tools
- Document findings

## Open Questions and Considerations

### 1. Gradient Dynamics
- How does the straight-through estimator affect training stability?
- Should we use a soft relaxation during early training?

### 2. Hyperparameter Selection
- Optimal k for different sequence lengths
- How to handle very long sequences (>2048 tokens)
- Regularization weight tuning

### 3. Architecture Variants
- Should different layers have different k values?
- Potential for hierarchical sparsity patterns
- Integration with other LoRA variants (DoRA, AdaLoRA)

### 4. Inference Optimization
- Custom CUDA kernels for sparse operations
- Caching strategies for common activation patterns
- Dynamic k selection based on input

## Conclusion

Token-sparse LoRA represents a promising direction for creating more interpretable and efficient language model adaptations. By forcing neurons to be selective about which tokens they process, we can create specialized feature detectors that are both powerful and understandable. This approach maintains the parameter efficiency of LoRA while adding a new dimension of interpretability that could be valuable for understanding and controlling model behavior.