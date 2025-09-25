#!/bin/bash
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=36:00:00
#SBATCH --job-name=lstm-a100
#SBATCH --output=./slurm_output/%j.out
#SBATCH --mail-user=zmou1@jh.edu
#SBATCH --mail-type=END,FAIL

echo "Starting LSTM training on A100 GPU..."
nvidia-smi

# Set memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Original parameters that should work on A100
python3 /home/zmou1/scratchenalisn1/ziyao/l2d-cog/l2d_project/scripts/train_defer_lstm.py \
    --hidden_dim 1024 \
    --lr 0.0001 \
    --batch_size 16 \
    --seq_len 100 \
    --step 100 \
    --lstm_layers 1 \
    --checkpoint_dir checkpoints/lstm-a100 \
    --model_name densenet_lstm_fatigue_a100 \
    --pretrained_epochs 3 \
    --finetuned_epochs 4

echo "A100 training completed!" 