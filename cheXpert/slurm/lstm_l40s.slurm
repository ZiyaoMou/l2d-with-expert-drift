#!/bin/bash
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=36:00:00
#SBATCH --job-name=lstm-l40s
#SBATCH --output=./slurm_output/%j.out
#SBATCH --mail-user=zmou1@jh.edu
#SBATCH --mail-type=END,FAIL

echo "Starting LSTM training on L40S GPU..."
nvidia-smi

# Set memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Original parameters that should work on L40S (48GB memory)
python3 /home/zmou1/scratchenalisn1/ziyao/l2d-cog/l2d_project/scripts/train_defer_lstm.py \
    --hidden_dim 1024 \
    --lr 0.0001 \
    --batch_size 16 \
    --seq_len 100 \
    --step 100 \
    --lstm_layers 1 \
    --checkpoint_dir checkpoints/lstm-l40s \
    --model_name densenet_lstm_fatigue_l40s \
    --pretrained_epochs 3 \
    --finetuned_epochs 4

echo "L40S training completed!" 