"""
Train CE baseline model
Reference: train_defer.py and notebook implementation
"""

import sys
import os

# Add l2d_project to import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import torch
from models.densenet_ce import CheXpertTrainer_CE, DenseNet121_CE 
from data.cheXpert import split_dataset
from experts.fake import ExpertModel_fake
from utils.calibrate import ModelWithTemperature

NN_CLASS_COUNT = 14
EPOCHS_STAGE1 = 3
BATCH_SIZE = 16
RANDOM_SEED = 66
TRAIN_SIZE = 0.999

ROOT_DIR = "/home/zmou1/scratchenalisn1/ziyao/l2d-data/CheXpert/"
PATH_FILE_TRAIN = "/home/zmou1/scratchenalisn1/ziyao/l2d-data/CheXpert/CheXpert-v1.0-small/train.csv"
PATH_FILE_VALID = "/home/zmou1/scratchenalisn1/ziyao/l2d-data/CheXpert/CheXpert-v1.0-small/valid.csv"
CHECKPOINT_DIR = "/home/zmou1/scratchenalisn1/ziyao/l2d-cog/l2d_project/checkpoints"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

p_confound = 0.7
p_nonconfound = 1.0
exp_fake = ExpertModel_fake(
    confounding_class=13,
    p_confound=p_confound,
    p_nonconfound=p_nonconfound
)

# Data loading
dl_train, dl_val, dl_test, dl_official_val, _ = split_dataset(
    train_size=TRAIN_SIZE,
    random_seed=RANDOM_SEED,
    root_dir=ROOT_DIR,
    pathFileTrain=PATH_FILE_TRAIN,
    pathFileValid=PATH_FILE_VALID,
    exp_fake=exp_fake,
    trBatchSize=BATCH_SIZE
)

model_ce = DenseNet121_CE(NN_CLASS_COUNT).to(device)
model_ce = torch.nn.DataParallel(model_ce).to(device)

timestamp = time.strftime("%d%m%Y-%H%M%S")
batch, losst, losse = CheXpertTrainer_CE.train(
    model=model_ce,
    dataLoaderTrain=dl_train,
    dataLoaderVal=dl_val,
    dataLoaderTest=dl_test,
    nnClassCount=NN_CLASS_COUNT,
    trMaxEpoch=EPOCHS_STAGE1,
    launchTimestamp=timestamp,
    checkpoint=None
)

torch.save(
    model_ce.state_dict(),
    os.path.join(CHECKPOINT_DIR, "densenet_model_ce.pth")
)
print("CE baseline training finished and checkpoint saved.")