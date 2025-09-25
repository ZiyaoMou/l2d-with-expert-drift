#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --job-name=lstm-reduced
#SBATCH --output=./slurm_output/%j.out
#SBATCH --mail-user=zmou1@jh.edu
#SBATCH --mail-type=END,FAIL

echo "Starting reduced memory LSTM training..."
nvidia-smi

# Set memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Try progressively smaller configurations until one works
echo "Attempting configuration 1: batch_size=4, seq_len=20, hidden_dim=128"
python3 /home/zmou1/scratchenalisn1/ziyao/l2d-cog/l2d_project/scripts/train_defer_lstm_reduced.py \
    --batch_size 4 \
    --seq_len 20 \
    --hidden_dim 128 \
    --train_size 0.01 \
    --pretrained_epochs 1 \
    --finetuned_epochs 1

if [ $? -ne 0 ]; then
    echo "Configuration 1 failed, trying configuration 2: batch_size=2, seq_len=10, hidden_dim=64"
    python3 /home/zmou1/scratchenalisn1/ziyao/l2d-cog/l2d_project/scripts/train_defer_lstm_reduced.py \
        --batch_size 2 \
        --seq_len 10 \
        --hidden_dim 64 \
        --train_size 0.005 \
        --pretrained_epochs 1 \
        --finetuned_epochs 1
fi

if [ $? -ne 0 ]; then
    echo "Configuration 2 failed, trying configuration 3: batch_size=1, seq_len=5, hidden_dim=32"
    python3 /home/zmou1/scratchenalisn1/ziyao/l2d-cog/l2d_project/scripts/train_defer_lstm_reduced.py \
        --batch_size 1 \
        --seq_len 5 \
        --hidden_dim 32 \
        --train_size 0.001 \
        --pretrained_epochs 1 \
        --finetuned_epochs 1
fi

echo "Training completed or all configurations failed." 