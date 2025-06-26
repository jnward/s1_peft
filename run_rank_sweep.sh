#!/bin/bash

# Script to run LoRA training with different rank values
# Continues to next rank even if a run fails

echo "Starting LoRA rank sweep experiment"
echo "Testing ranks: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512"
echo "================================================"

# Array of ranks to test
ranks=(1 4 16 64 256)

# Loop through each rank
for rank in "${ranks[@]}"; do
    echo ""
    echo "================================================"
    echo "Starting training with rank=$rank"
    echo "Timestamp: $(date)"
    echo "================================================"
    
    # Determine learning rate based on rank
    # Keep the same LR for rank 1, scale down for higher ranks
    if [[ $rank -le 64 ]]; then
        lr=1e-3  # Keep original for rank 1
    elif [[ $rank -le 256 ]]; then
        lr=5e-4
    else
        # rank >= 256 (e.g., 512)
        lr=2.5e-4
    fi
    
    echo "Using learning rate: $lr for rank=$rank"
    
    # Run training with current rank and learning rate
    # The || true ensures the script continues even if training fails
    ./run_training_fixed.sh --rank $rank --lr $lr || {
        echo "Training with rank=$rank failed, continuing to next rank..."
        echo "Exit code: $?"
    }
    
    echo "Completed rank=$rank at $(date)"
done

echo ""
echo "================================================"
echo "Rank sweep experiment completed!"
echo "Final timestamp: $(date)"
echo "================================================"