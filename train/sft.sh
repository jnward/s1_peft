# Reference Running: bash train/sft.sh
# {'train_runtime': 5268.8407, 'train_samples_per_second': 0.949, 'train_steps_per_second': 0.119, 'train_loss': 0.1172730620391667, 'epoch': 5.0}
uid="$(date +%Y%m%d_%H%M%S)"
base_model="Qwen/Qwen2.5-32B-Instruct"
block_size=20000
lr=1e-5
min_lr=0
epochs=5
weight_decay=1e-4 # -> the same training pipe as slurm_training
batch_size=16
micro_batch_size=1
max_steps=-1
gpu_count=$(nvidia-smi -L | wc -l)
gradient_accumulation_steps=$((batch_size / (micro_batch_size * gpu_count)))
push_to_hub=false

torchrun --nproc-per-node ${gpu_count} --master_port 12345 \
    train/sft.py \
    --block_size=${block_size} \
    --per_device_train_batch_size=${micro_batch_size} \
    --per_device_eval_batch_size=${micro_batch_size} \
    --gradient_accumulation_steps=${gradient_accumulation_steps} \
    --num_train_epochs=${epochs} \
    --train_file_path="simplescaling/s1K-1.1_tokenized" \
    --model_name=${base_model} \
    --warmup_ratio=0.05 \
    --fsdp="full_shard auto_wrap" \
    --fsdp_config="train/fsdp_config_qwen.json" \
    --bf16=True \
    --eval_strategy="no" \
    --logging_steps=1 \
    --save_strategy="no" \
    --lr_scheduler_type="cosine" \
    --learning_rate=${lr} \
    --weight_decay=${weight_decay} \
    --adam_beta1=0.9 \
    --adam_beta2=0.95 \
    --output_dir="ckpts_1.1/s1-${uid}" \
    --push_to_hub=${push_to_hub} \
    --save_only_model=True \
    --gradient_checkpointing=True \
    --accelerator_config='{"gradient_accumulation_kwargs": {"sync_each_batch": true}}'

