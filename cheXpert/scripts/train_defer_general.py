import sys
import os
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.densenet_defer_seq import DenseNet121SeqDefer, SeqTrainerDefer

from data.cheXpert_bias import split_dataset_seq
from experts.fake_bias import ExpertModelBiased

nnClassCount = 14
seq_len      = 10
step         = 10
batch_size   = 16

rootDir      = "/home/zmou1/scratchenalisn1/ziyao/l2d-data/CheXpert/"
pathFileTrain= os.path.join(rootDir, "CheXpert-v1.0-small/train.csv")
pathFileValid= os.path.join(rootDir, "CheXpert-v1.0-small/valid.csv")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = DenseNet121SeqDefer(num_classes=nnClassCount).cuda()
exp_biased = ExpertModelBiased(p_confound=0.7,
                                p_nonconfound=1,
                                decay_confound=0.0035,
                                decay_nonconfound=0.005,
                                confounding_class=13,
                                use_fatigue=True,
                                seq_len=seq_len,
                                num_classes=nnClassCount)
                                
dataLoaderTrain, dataLoaderVal, dataLoaderTest = split_dataset_seq(train_size=0.999, random_seed=66,root_dir=rootDir, pathFileValid=pathFileValid, pathFileTrain=pathFileTrain, exp_fake=exp_biased, trBatchSize=batch_size, seq_len=seq_len, step=step)

trainer = SeqTrainerDefer(model, exp_biased, DEVICE)

trainer.fit(dataLoaderTrain, dataLoaderTest, pretained_epochs=3, finetuned_epochs=1, lr=1e-4)

torch.save(model.state_dict(), "checkpoints/densenet_seq_defer_full_no_alpha_100_steps.pth")

