import sys
import os
import torch
import time
import pandas as pd
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from data.cheXpert_random import split_dataset_shuffle
from data.cheXpert import split_dataset
from experts.fake import ExpertModel_fake
from models.densenet_defer import CheXpertTrainer_defer, DenseNet121_defer

nnClassCount = 14
seq_len      = 10
step         = 10
batch_size   = 16
train_ratio  = 0.01
random_seed  = 66

rootDir      = "/home/zmou1/scratchenalisn1/ziyao/l2d-data/CheXpert/"
pathFileTrain= os.path.join(rootDir, "CheXpert-v1.0-small/train.csv")
pathFileValid= os.path.join(rootDir, "CheXpert-v1.0-small/valid.csv")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

alpha = [1, 1, 0.1, 1, 1, 0.4, 0.2, 1, 0.1, 1, 0.1, 1, 1, 1]
# alpha = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
timestampTime = time.strftime("%H%M%S")
timestampDate = time.strftime("%d%m%Y")
timestampLaunch = timestampDate + '-' + timestampTime

pretrained_model = "/home/zmou1/scratchenalisn1/ziyao/l2d-cog/l2d_project/checkpoints/pretrained_step_model_full_pretrained-100-steps.pth"

# Main training loop with progress bar
df = pd.DataFrame()
for t in tqdm(range(0, 100), desc="Training steps", unit="step"):
    model_defer = DenseNet121_defer(nnClassCount).to(DEVICE)
    model_defer = torch.nn.DataParallel(model_defer).to(DEVICE)
    
    expert_t = ExpertModel_fake(
        confounding_class=13,
        p_confound=0.7-0.0035 * t,
        p_nonconfound=1.0-0.005 * t
    )
    
    trainer_t = CheXpertTrainer_defer()
    dataLoaderTrain_t, dataLoaderVal_t, dataLoaderTest_t, _, _ = split_dataset(
        exp_fake=expert_t,
        train_size=0.999,
        random_seed=random_seed,
        root_dir=rootDir,
        pathFileValid=pathFileValid,
        pathFileTrain=pathFileTrain,
    )
    
    model_defer = DenseNet121_defer(nnClassCount).cuda()
    model_defer = torch.nn.DataParallel(model_defer).cuda()
    batch, losst, losse = CheXpertTrainer_defer.train_defer(
            model_defer,
            rad_index=1,
            learn_to_defer=False,
            dataLoadertrain=dataLoaderTrain_t,
            dataLoaderVal=dataLoaderTest_t,
            nnClassCount=nnClassCount,
            trMaxEpoch=3,
            launchTimestamp=timestampLaunch,
            alpha=1*[alpha],
            checkpoint=pretrained_model
    )
    # loss, auc, auc_exp, auc_sys, defer_rate = trainer_t.test_epoch_defer(model_defer, dataLoaderTest_t, alpha, DEVICE, rad_index=1, use_defer=True)
    # print(f"Pretrained model trained with loss {loss}, auc {auc}, auc_exp {auc_exp}, auc_sys {auc_sys}, defer_rate {defer_rate}")
    torch.save(model_defer.state_dict(), pretrained_model)

    base_path = "/home/zmou1/scratchenalisn1/ziyao/l2d-cog/l2d_project/checkpoints/perstep-full-100-steps"

    dense_per_step_model = f"/home/zmou1/scratchenalisn1/ziyao/l2d-cog/l2d_project/checkpoints/perstep-full-100-steps/densenet_defer_step_{t}.pth"

    os.makedirs(base_path, exist_ok=True)

    batch, losst, losse = CheXpertTrainer_defer.train_defer(
            model_defer,
            rad_index=1,
            learn_to_defer=True,
            dataLoadertrain=dataLoaderTrain_t,
            dataLoaderVal=dataLoaderTest_t,
            nnClassCount=nnClassCount,
            trMaxEpoch=1,
            launchTimestamp=timestampLaunch,
            alpha=alpha,
            checkpoint=dense_per_step_model
        )
    loss, auc_cls_per_class, auc_exp_per_class, auc_sys_per_class, defer_rates, auprc_cls_per_class, auprc_exp_per_class, auprc_sys_per_class = trainer_t.test_epoch_defer(model_defer, dataLoaderTest_t, alpha, DEVICE, rad_index=1, use_defer=True)

    for i in range(14):
        auc_i = auc_cls_per_class[i]
        auc_exp_i = auc_exp_per_class[i]
        auc_sys_i = auc_sys_per_class[i]
        defer_rate_i = defer_rates[i]
        auprc_cls_i = auprc_cls_per_class[i]
        auprc_exp_i = auprc_exp_per_class[i]
        auprc_sys_i = auprc_sys_per_class[i]
        new_row = pd.DataFrame({
            'Timestep': [t],
            'Class': [i],
            'AUC_cls': [auc_i],
            'AUC_exp': [auc_exp_i],
            'AUC_sys': [auc_sys_i],
            'Defer Rate': [defer_rate_i],
            'AUPRC_cls': [auprc_cls_i],
            'AUPRC_exp': [auprc_exp_i],
            'AUPRC_sys': [auprc_sys_i]
        })
        df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(f"/home/zmou1/scratchenalisn1/ziyao/l2d-cog/l2d_project/results/perstep-full-100-steps.csv", index=False)

    torch.save(model_defer.state_dict(), dense_per_step_model)