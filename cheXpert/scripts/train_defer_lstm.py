import sys
import os
import torch
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.densenet_lstm import DenseLSTMDefer, CheXpertTrainerDeferLSTM
from data.cheXpert_bias import split_dataset_seq
from experts.fake_bias import ExpertModelBiased

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train DenseLSTM Defer model with CheXpert dataset')
parser.add_argument('--hidden_dim', type=int, default=32, help='Hidden dimension for LSTM (default: 32)')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.0001)')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size (default: 16)')
parser.add_argument('--seq_len', type=int, default=10, help='Sequence length (default: 10)')
parser.add_argument('--step', type=int, default=10, help='Step size (default: 10)')
parser.add_argument('--lstm_layers', type=int, default=1, help='Number of LSTM layers (default: 1)')
parser.add_argument('--train_size', type=float, default=0.999, help='Training data size ratio (default: 0.1)')
parser.add_argument('--random_seed', type=int, default=66, help='Random seed (default: 66)')
parser.add_argument('--pretrained_epochs', type=int, default=3, help='Number of epochs for pretrained (default: 10)')
parser.add_argument('--finetuned_epochs', type=int, default=4, help='Number of epochs for finetuned (default: 10)')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/lstm', help='Checkpoint directory (default: checkpoints/lstm)')
parser.add_argument('--model_name', type=str, default='densenet_lstm_fatigue_1', help='Model name for saving (default: densenet_lstm_fatigue_1)')

args = parser.parse_args()

print(args)

nnClassCount = 14
seq_len      = args.seq_len
step         = args.step
batch_size   = args.batch_size
hidden_dim   = args.hidden_dim


rootDir      = "/home/zmou1/scratchenalisn1/ziyao/l2d-data/CheXpert/"
pathFileTrain= os.path.join(rootDir, "CheXpert-v1.0-small/train.csv")
pathFileValid= os.path.join(rootDir, "CheXpert-v1.0-small/valid.csv")


model = DenseLSTMDefer(num_classes=nnClassCount,
                       lstm_hidden=hidden_dim,
                       lstm_layers=args.lstm_layers).cuda()

exp_biased = ExpertModelBiased(p_confound=0.7,
                                p_nonconfound=1,
                                decay_confound=0.0035,
                                decay_nonconfound=0.005,
                                confounding_class=13,
                                use_fatigue=True,
                                seq_len=seq_len,
                                num_classes=nnClassCount)

dataLoaderTrain, dataLoaderVal, dataLoaderTest = split_dataset_seq(train_size=args.train_size, random_seed=args.random_seed,root_dir=rootDir, pathFileValid=pathFileValid, pathFileTrain=pathFileTrain, exp_fake=exp_biased, trBatchSize=batch_size, seq_len=seq_len, step=step)

trainer = CheXpertTrainerDeferLSTM(model, exp_biased, torch.device("cuda"), lr=args.lr, pretrained_epochs=args.pretrained_epochs, finetuned_epochs=args.finetuned_epochs, seq_len=seq_len, step=step)

trainer.train_defer_lstm(dataLoaderTrain, dataLoaderTest)

os.makedirs(args.checkpoint_dir, exist_ok=True)

torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"{args.model_name}-{hidden_dim}-unit-{args.lstm_layers}-layers-{args.pretrained_epochs}-pretrained-{args.finetuned_epochs}-finetuned-no-alpha-{seq_len}-steps.pth"))

print("Model trained")

