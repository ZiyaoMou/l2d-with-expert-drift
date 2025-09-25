import os
import torch
import sys
import pandas as pd
import ast
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from models.densenet_lstm import DenseLSTMDefer, CheXpertTrainerDeferLSTM, l2d_loss
from models.densenet_defer_seq import DenseNet121SeqDefer, SeqTrainerDefer
from experts.fake_bias import ExpertModelBiased
from experts.fake import ExpertModel_fake
from data.cheXpert_bias import split_dataset_seq
from data.cheXpert_random import split_dataset_shuffle
from models.densenet_defer import DenseNet121_defer, CheXpertTrainer_defer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = "checkpoints/densenet_lstm_fatigue_new.pth"
PRIOR_CHECKPOINT = "checkpoints/densenet_seq_defer.pth"
PERSTEP_DIR = "checkpoints/perstep-new"

ROOT_DIR = "/home/zmou1/scratchenalisn1/ziyao/l2d-data/CheXpert/"
TRAIN_CSV = os.path.join(ROOT_DIR, "CheXpert-v1.0-small/train.csv")
VALID_CSV = os.path.join(ROOT_DIR, "CheXpert-v1.0-small/valid.csv")
alpha = [1, 1, 0.1, 1, 1, 0.4, 0.2, 1, 0.1, 1, 0.1, 1, 1, 1]

BATCH_SIZE = 16
SEQ_LEN = 10
STEP = 10
N_CLASS = 14

# Load models
state = torch.load(CHECKPOINT, map_location=DEVICE)
prior_state = torch.load(PRIOR_CHECKPOINT, map_location=DEVICE)

model = DenseLSTMDefer(num_classes=N_CLASS, lstm_hidden=512, lstm_layers=1).to(DEVICE)
model_prior = DenseNet121SeqDefer(num_classes=N_CLASS).to(DEVICE)
model.load_state_dict(state)
model_prior.load_state_dict(prior_state)

exp_biased = ExpertModelBiased(p_confound=0.7,
                                p_nonconfound=1,
                                decay_confound=0.07,
                                decay_nonconfound=0.1,
                                confounding_class=13,
                                use_fatigue=True,
                                seq_len=SEQ_LEN,
                                num_classes=N_CLASS)

# Create data loaders
_, _, test_loader = split_dataset_seq(
    train_size=0.01, random_seed=66,
    root_dir=ROOT_DIR,
    pathFileTrain=TRAIN_CSV, pathFileValid=VALID_CSV,
    exp_fake=exp_biased, trBatchSize=BATCH_SIZE,
    seq_len=SEQ_LEN, step=STEP
)

# Create trainers
lstm_trainer = CheXpertTrainerDeferLSTM(model, exp_biased, DEVICE)
seq_trainer = SeqTrainerDefer(model_prior, exp_biased, DEVICE)

perstep_sys_aurocs_all = []
perstep_defer_rates_all = []
perstep_covs_all = []
perstep_aucs_all = []
perstep_best_aucs_all = []
perstep_best_covs_all = []

# Evaluate LSTM model
# print("Evaluating LSTM model...")
lstm_loss, lstm_cls_auc, lstm_exp_auc, lstm_sys_aurocs, lstm_defer_rates = lstm_trainer.test_epoch(test_loader, loss_fn=l2d_loss)
all_covs_lstm, all_aucs_lstm, all_best_auc_lstm, all_best_cov_lstm = lstm_trainer.test_epoch_best_coverage(test_loader, rad_index=1)

df_lstm = pd.DataFrame()
df_lstm["Defer Rate"] = lstm_defer_rates
df_lstm["System AUC"] = lstm_sys_aurocs
df_lstm["Expert AUC"] = lstm_exp_auc
df_lstm["Classifier AUC"] = lstm_cls_auc

df_lstm["Best AUC"] = all_best_auc_lstm
df_lstm["Best Coverage"] = all_best_cov_lstm
df_lstm["Timestep"] = np.arange(1, len(lstm_sys_aurocs)+1)
df_lstm['all_aucs'] = all_aucs_lstm

float_cols = df_lstm.select_dtypes(include=['float']).columns
df_lstm[float_cols] = df_lstm[float_cols].round(3)

df_lstm["all_aucs"] = df_lstm["all_aucs"].apply(
    lambda x: np.round(ast.literal_eval(x), 3).tolist() if isinstance(x, str) else np.round(x, 3).tolist()
)

df_lstm.to_csv("results/lstm_results.csv", index=False)


# # Evaluate Seq model  
# print("\nEvaluating Seq model...")
seq_loss, seq_cls_auc, seq_exp_auc, seq_sys_aurocs, seq_defer_rates = seq_trainer.validate_epoch(test_loader, use_defer=True)
all_covs_seq, all_aucs_seq, all_best_auc_seq, all_best_cov_seq = seq_trainer.validate_epoch_best_coverage(test_loader, rad_index=1)

print("\nEvaluating Per-Step Models...")
for t in range(SEQ_LEN):
    ckpt_path = os.path.join(PERSTEP_DIR, f"densenet_defer_step_{t}.pth")
    if not os.path.exists(ckpt_path):
        print(f"Warning: checkpoint not found for step {t}: {ckpt_path}")
        perstep_sys_aurocs_all.append(np.nan)
        perstep_defer_rates_all.append(np.nan)
        continue

    # Create expert for this timestep (matching train_defer_perstep.py)
    expert_t = ExpertModel_fake(
        confounding_class=13,
        p_confound=0.7-0.07*t,
        p_nonconfound=1-0.1*t
    )

    # Create data loaders for this timestep (matching train_defer_perstep.py)
    _, _, dataLoaderTest_t, _, _ = split_dataset_shuffle(
        exp_fake=expert_t,
        train_size=0.1,
        random_seed=66,
        root_dir=ROOT_DIR,
        pathFileValid=VALID_CSV,
        pathFileTrain=TRAIN_CSV,
    )

    model_t = DenseNet121_defer(out_size=N_CLASS).to(DEVICE)
    model_t = torch.nn.DataParallel(model_t).to(DEVICE)
    model_t.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model_t.eval()

    trainer_t = CheXpertTrainer_defer()
    loss_t, auc_cls_t, auc_exp_t, auc_sys_t, defer_rate_t = trainer_t.test_epoch_defer(
        model=model_t,
        loader=dataLoaderTest_t,
        device=DEVICE,
        alpha=alpha,
        rad_index=1)
    
    all_covs_perstep, all_aucs_perstep, all_best_auc_perstep, all_best_cov_perstep = trainer_t.test_epoch_defer_best_coverage(model=model_t, loader=dataLoaderTest_t, device=DEVICE, rad_index=1)
    perstep_sys_aurocs_all.append(np.nanmean(auc_sys_t))
    perstep_defer_rates_all.append(defer_rate_t)
    perstep_covs_all.append(all_covs_perstep)
    perstep_aucs_all.append(all_aucs_perstep)
    perstep_best_aucs_all.append(all_best_auc_perstep)
    perstep_best_covs_all.append(all_best_cov_perstep)


plt.figure(figsize=(6,4))
ts = np.arange(1, len(lstm_sys_aurocs)+1)
plt.plot(ts, lstm_sys_aurocs, marker='o', label='LSTM-L2D AUROC')
plt.plot(ts, seq_sys_aurocs, marker='s', label='L2D AUROC')
plt.plot(ts, perstep_sys_aurocs_all, marker='^', label='Per-Step-L2D AUROC')
plt.xlabel("Timestep t")
plt.ylabel("System AUROC")
plt.title("Different L2D System AUROC over time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/1-fatigue_plot_with_prior.png")
plt.show()

print(f"LSTM System AUROCs: {lstm_sys_aurocs}")
print(f"LSTM Defer Rates: {lstm_defer_rates}")
print(f"Seq System AUROCs: {seq_sys_aurocs}") 
print(f"Seq Defer Rates: {seq_defer_rates}")
print(f"Per-Step System AUROCs: {perstep_sys_aurocs_all}")
print(f"Per-Step Defer Rates: {perstep_defer_rates_all}")

plt.figure(figsize=(6,4))
plt.plot(ts, lstm_defer_rates, marker='o', label='LSTM-L2D Defer Rate')
plt.plot(ts, seq_defer_rates, marker='s', label='L2D Defer Rate')
plt.plot(ts, perstep_defer_rates_all, marker='^', label='Per-Step-L2D Defer Rate')
plt.xlabel("Timestep t")
plt.ylabel("Defer Rate")
plt.title("Defer Rate over time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/1-fatigue_plot_with_prior_time.png")
plt.show()

plt.figure(figsize=(6,4))
plt.plot(ts, all_best_auc_lstm, marker='o', label='LSTM-L2D Best AUC')
plt.plot(ts, all_best_auc_seq, marker='s', label='L2D Best AUC')
plt.plot(ts, perstep_best_aucs_all, marker='^', label='Per-Step-L2D Best AUC')
plt.xlabel("Timestep t")
plt.ylabel("Best AUC")
plt.title("Different L2D Best AUC over time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/1-fatigue_plot_with_prior_best_auc.png")
plt.show()


plt.figure(figsize=(6,4))
plt.plot(ts, all_best_cov_lstm, marker='o', label='LSTM-L2D Best Coverage')
plt.plot(ts, all_best_cov_seq, marker='s', label='L2D Best Coverage')
plt.plot(ts, perstep_best_covs_all, marker='^', label='Per-Step-L2D Best Coverage')
plt.xlabel("Timestep t")
plt.ylabel("Best Coverage")
plt.title("Different L2D Best Coverage over time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/1-fatigue_plot_with_prior_best_cov.png")
plt.show()


