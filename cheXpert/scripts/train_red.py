"""
Train RED baseline model
Reference: train_ce.py and train_defer.py implementation
"""

import sys
import os

# Add l2d_project to import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import torch
from models.densenet_red import CheXpertTrainer_rad
from models.densenet_ce import DenseNet121_CE
from data.cheXpert import split_dataset
from experts.fake import ExpertModel_fake
from utils.calibrate import ModelWithTemperature_rad

NN_CLASS_COUNT = 14
EPOCHS_STAGE1 = 3
BATCH_SIZE = 16
RANDOM_SEED = 66
TRAIN_SIZE = 0.999
RAD_INDEX = 1  # Which radiologist to use (1, 2, or 3)

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

dl_train, dl_val, dl_test, dl_official_val, _ = split_dataset(
    train_size=TRAIN_SIZE,
    random_seed=RANDOM_SEED,
    root_dir=ROOT_DIR,
    pathFileTrain=PATH_FILE_TRAIN,
    pathFileValid=PATH_FILE_VALID,
    exp_fake=exp_fake,
    trBatchSize=BATCH_SIZE
)

model_red = DenseNet121_CE (NN_CLASS_COUNT).to(device)
model_red = torch.nn.DataParallel(model_red).to(device)

timestamp = time.strftime("%d%m%Y-%H%M%S")
batch, losst, losse = CheXpertTrainer_rad.train(
    model=model_red,
    rad_index=RAD_INDEX,
    dataLoaderTrain=dl_train,
    dataLoaderTest=dl_test,
    nnClassCount=NN_CLASS_COUNT,
    trMaxEpoch=EPOCHS_STAGE1,
    launchTimestamp=timestamp,
    checkpoint=None
)

torch.save(
    model_red.state_dict(),
    os.path.join(CHECKPOINT_DIR, "densenet_red.pth")
)
print("RED baseline training finished and checkpoint saved.")