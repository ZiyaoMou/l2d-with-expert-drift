import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
from models.densenet_defer import DenseNet121_defer, CheXpertTrainer_defer
import torch
from data.cheXpert import split_dataset
from experts.fake import ExpertModel_fake

nnClassCount = 14
p_confound = 0.7
p_nonconfound = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = [
    'No Finding','Enlarged Cardiomediastinum','Cardiomegaly',
    'Lung Opacity','Lung Lesion','Edema','Consolidation',
    'Pneumonia','Atelectasis','Pneumothorax','Pleural Effusion',
    'Pleural Other','Fracture','Support Devices'
]

rootDir = "/home/zmou1/scratchenalisn1/ziyao/l2d-data/CheXpert/"
pathFileTrain = "/home/zmou1/scratchenalisn1/ziyao/l2d-data/CheXpert/CheXpert-v1.0-small/train.csv"
pathFileValid = "/home/zmou1/scratchenalisn1/ziyao/l2d-data/CheXpert/CheXpert-v1.0-small/valid.csv"

model_defer = DenseNet121_defer(nnClassCount).cuda()
model_defer = torch.nn.DataParallel(model_defer).cuda()
timestampTime = time.strftime("%H%M%S")
timestampDate = time.strftime("%d%m%Y")
timestampLaunch = timestampDate + '-' + timestampTime

exp_fake = ExpertModel_fake(13,p_confound,p_nonconfound)

dataLoaderTrain, dataLoaderVal, dataLoaderTest, dataLoaderOfficialValid, patients_train_leftout = split_dataset(train_size=0.999, random_seed=66,root_dir=rootDir, pathFileValid=pathFileValid, pathFileTrain=pathFileTrain, exp_fake=exp_fake, trBatchSize=16)

batch, losst, losse = CheXpertTrainer_defer.train_defer(
    model_defer,
    rad_index=1,
    learn_to_defer=False,
    dataLoadertrain=dataLoaderTrain,
    dataLoaderVal=dataLoaderVal,
    nnClassCount=nnClassCount,
    trMaxEpoch=3,
    launchTimestamp=timestampLaunch,
    alpha=[1] * 14,
    checkpoint='/home/zmou1/scratchenalisn1/ziyao/l2d-cog/l2d_project/checkpoints/densenet_defer.pth'
)

print("Model trained")

alpha = [1, 1, 0.1, 1, 1, 0.4, 0.2, 1, 0.1, 1, 0.1, 1, 1, 1]
timestampTime = time.strftime("%H%M%S")
timestampDate = time.strftime("%d%m%Y")
timestampLaunch = timestampDate + '-' + timestampTime

batch, losst, losse = CheXpertTrainer_defer.train_defer(
    model_defer,
    rad_index=1,
    learn_to_defer=True,
    dataLoadertrain=dataLoaderTrain,
    dataLoaderVal=dataLoaderVal,
    nnClassCount=nnClassCount,
    trMaxEpoch=1,
    launchTimestamp=timestampLaunch,
    alpha=alpha,
    checkpoint='/home/zmou1/scratchenalisn1/ziyao/l2d-cog/l2d_project/checkpoints/densenet_defer.pth'
)
print("Model trained")
trainer = CheXpertTrainer_defer()
loss, auc, auc_exp, auc_sys, defer_rate, auprc_cls, auprc_exp, auprc_sys = trainer.test_epoch_defer(model_defer, dataLoaderTest, alpha, DEVICE, rad_index=1, use_defer=True)

# torch.save(model_defer.state_dict(), f"/home/zmou1/scratchenalisn1/ziyao/l2d-cog/l2d_project/checkpoints/densenet_defer.pth")