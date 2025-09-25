#!/bin/bash
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=36:00:00
#SBATCH --job-name=lstm-optimized
#SBATCH --output=./slurm_output/%j.out
#SBATCH --mail-user=zmou1@jh.edu
#SBATCH --mail-type=END,FAIL

echo "Starting MEMORY-OPTIMIZED LSTM training..."
echo "GPU allocation:"
nvidia-smi

# Set memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

echo "Environment variables set for memory optimization"

# Try the optimized version with reduced batch size
echo "Attempting optimized training with batch_size=8..."
python3 /home/zmou1/scratchenalisn1/ziyao/l2d-cog/l2d_project/scripts/train_defer_lstm_optimized.py \
    --hidden_dim 1024 \
    --lr 0.0001 \
    --batch_size 8 \
    --seq_len 30 \
    --step 30 \
    --lstm_layers 1 \
    --checkpoint_dir checkpoints/lstm-optimized \
    --model_name densenet_lstm_fatigue_optimized \
    --pretrained_epochs 3 \
    --finetuned_epochs 4

echo "Optimized LSTM training attempt completed." 