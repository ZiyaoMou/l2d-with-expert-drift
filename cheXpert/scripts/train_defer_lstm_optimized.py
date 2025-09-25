import sys
import os
import torch
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.densenet_lstm_optimized import DenseLSTMDeferOptimized, CheXpertTrainerDeferLSTMOptimized
from data.cheXpert_bias import split_dataset_seq
from experts.fake_bias import ExpertModelBiased

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train optimized DenseLSTM Defer model')
parser.add_argument('--hidden_dim', type=int, default=1024, help='Hidden dimension for LSTM (default: 1024)')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate (default: 0.0001)')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size (default: 8)')
parser.add_argument('--seq_len', type=int, default=100, help='Sequence length (default: 100)')
parser.add_argument('--step', type=int, default=100, help='Step size (default: 100)')
parser.add_argument('--lstm_layers', type=int, default=1, help='Number of LSTM layers (default: 1)')
parser.add_argument('--train_size', type=float, default=0.999, help='Training data size ratio (default: 0.999)')
parser.add_argument('--random_seed', type=int, default=66, help='Random seed (default: 66)')
parser.add_argument('--pretrained_epochs', type=int, default=3, help='Number of epochs for pretrained (default: 3)')
parser.add_argument('--finetuned_epochs', type=int, default=4, help='Number of epochs for finetuned (default: 4)')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/lstm_optimized', help='Checkpoint directory')
parser.add_argument('--model_name', type=str, default='densenet_lstm_optimized', help='Model name for saving')

args = parser.parse_args()

print("=" * 60)
print("MEMORY-OPTIMIZED LSTM TRAINING")
print("=" * 60)
print(f"Batch size: {args.batch_size}")
print(f"Sequence length: {args.seq_len}")
print(f"Hidden dim: {args.hidden_dim}")
print(f"Train size: {args.train_size}")
print("Memory optimizations enabled:")
print("  - Gradient checkpointing")
print("  - Mixed precision training")
print("  - Chunk-based processing")
print("  - Gradient accumulation")
print("  - Frequent cache clearing")
print("=" * 60)

# Set memory optimization environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Enable memory optimization
torch.backends.cudnn.benchmark = False  # Disable to save memory
torch.backends.cudnn.deterministic = True

nnClassCount = 14
seq_len      = args.seq_len
step         = args.step
batch_size   = args.batch_size
hidden_dim   = args.hidden_dim

rootDir      = "/home/zmou1/scratchenalisn1/ziyao/l2d-data/CheXpert/"
pathFileTrain= os.path.join(rootDir, "CheXpert-v1.0-small/train.csv")
pathFileValid= os.path.join(rootDir, "CheXpert-v1.0-small/valid.csv")

# Clear GPU cache before starting
torch.cuda.empty_cache()

print("Initializing optimized model...")
model = DenseLSTMDeferOptimized(num_classes=nnClassCount,
                               lstm_hidden=hidden_dim,
                               lstm_layers=args.lstm_layers).cuda()

# Enable gradient checkpointing for the CNN backbone
model.cnn.requires_grad_(True)

exp_biased = ExpertModelBiased(p_confound=0.7,
                                p_nonconfound=1,
                                decay_confound=0.0035,
                                decay_nonconfound=0.005,
                                confounding_class=13,
                                use_fatigue=True,
                                seq_len=seq_len,
                                num_classes=nnClassCount)

print("Loading data...")
dataLoaderTrain, dataLoaderVal, dataLoaderTest = split_dataset_seq(
    train_size=args.train_size, 
    random_seed=args.random_seed,
    root_dir=rootDir, 
    pathFileValid=pathFileValid, 
    pathFileTrain=pathFileTrain, 
    exp_fake=exp_biased, 
    trBatchSize=batch_size, 
    seq_len=seq_len, 
    step=step
)

trainer = CheXpertTrainerDeferLSTMOptimized(
    model, 
    exp_biased, 
    torch.device("cuda"), 
    lr=args.lr, 
    pretrained_epochs=args.pretrained_epochs, 
    finetuned_epochs=args.finetuned_epochs
)

try:
    print("Starting training...")
    trainer.train_defer_lstm(dataLoaderTrain, dataLoaderTest)
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    model_path = os.path.join(args.checkpoint_dir, f"{args.model_name}-{hidden_dim}-unit-{args.lstm_layers}-layers-{args.pretrained_epochs}-pretrained-{args.finetuned_epochs}-seq{seq_len}-step{step}-optimized.pth")
    torch.save(model.state_dict(), model_path)
    
    print("=" * 60)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print(f"Model saved to: {model_path}")
    print("=" * 60)
    
except RuntimeError as e:
    if "out of memory" in str(e):
        print("=" * 60)
        print("❌ MEMORY ERROR STILL OCCURRED!")
        print("=" * 60)
        print("Even with optimizations, memory is insufficient.")
        print("Try these further reductions:")
        print(f"1. --batch_size (current: {batch_size}, try: {max(1, batch_size//2)})")
        print(f"2. --seq_len (current: {seq_len}, try: {seq_len//2})")
        print(f"3. --hidden_dim (current: {hidden_dim}, try: {hidden_dim//2})")
        print("4. Use --train_size 0.1 for testing")
        print("=" * 60)
        raise
    else:
        raise

finally:
    # Always clear cache at the end
    torch.cuda.empty_cache() 