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

# # Load models
# state = torch.load(CHECKPOINT, map_location=DEVICE)
# prior_state = torch.load(PRIOR_CHECKPOINT, map_location=DEVICE)

# model = DenseLSTMDefer(num_classes=N_CLASS, lstm_hidden=512, lstm_layers=1).to(DEVICE)
# model_prior = DenseNet121SeqDefer(num_classes=N_CLASS).to(DEVICE)
# model.load_state_dict(state)
# model_prior.load_state_dict(prior_state)

# exp_biased = ExpertModelBiased(p_confound=0.7,
#                                 p_nonconfound=1,
#                                 decay_confound=0.07,
#                                 decay_nonconfound=0.1,
#                                 confounding_class=13,
#                                 use_fatigue=True,
#                                 seq_len=SEQ_LEN,
#                                 num_classes=N_CLASS)

# # Create data loaders
# _, _, test_loader = split_dataset_seq(
#     train_size=0.01, random_seed=66,
#     root_dir=ROOT_DIR,
#     pathFileTrain=TRAIN_CSV, pathFileValid=VALID_CSV,
#     exp_fake=exp_biased, trBatchSize=BATCH_SIZE,
#     seq_len=SEQ_LEN, step=STEP
# )

# # Create trainers
# lstm_trainer = CheXpertTrainerDeferLSTM(model, exp_biased, DEVICE)
# seq_trainer = SeqTrainerDefer(model_prior, exp_biased, DEVICE)

# perstep_sys_aurocs_all = []
# perstep_defer_rates_all = []
# perstep_cls_aucs_all = []
# perstep_exp_aucs_all = []
# perstep_covs_all = []
# perstep_aucs_all = []
# perstep_best_aucs_all = []
# perstep_best_covs_all = []


# lstm_loss, lstm_cls_auc, lstm_exp_auc, lstm_sys_aurocs, lstm_defer_rates = lstm_trainer.test_epoch_cumulative(test_loader, loss_fn=l2d_loss)

# seq_loss, seq_cls_auc, seq_exp_auc, seq_sys_aurocs, seq_defer_rates = seq_trainer.validate_epoch_cumulative(test_loader, use_defer=True)



# print("\nEvaluating Per-Step Models with Cumulative Averaging...")
# # Store individual step results for cumulative averaging
# step_sys_aurocs = []
# step_cls_aucs = []
# step_exp_aucs = []
# step_defer_rates = []

# for t in range(SEQ_LEN):
#     ckpt_path = os.path.join(PERSTEP_DIR, f"densenet_defer_step_{t}.pth")
#     if not os.path.exists(ckpt_path):
#         print(f"Warning: checkpoint not found for step {t}: {ckpt_path}")
#         step_sys_aurocs.append(np.nan)
#         step_defer_rates.append(np.nan)
#         step_cls_aucs.append(np.nan)
#         step_exp_aucs.append(np.nan)
#         continue

#     # Create expert for this timestep (matching train_defer_perstep.py)
#     expert_t = ExpertModel_fake(
#         confounding_class=13,
#         p_confound=0.7-0.07*t,
#         p_nonconfound=1-0.1*t
#     )

#     # Create data loaders for this timestep (matching train_defer_perstep.py)
#     _, _, dataLoaderTest_t, _, _ = split_dataset_shuffle(
#         exp_fake=expert_t,
#         train_size=0.1,
#         random_seed=66,
#         root_dir=ROOT_DIR,
#         pathFileValid=VALID_CSV,
#         pathFileTrain=TRAIN_CSV,
#     )

#     model_t = DenseNet121_defer(out_size=N_CLASS).to(DEVICE)
#     model_t = torch.nn.DataParallel(model_t).to(DEVICE)
#     model_t.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
#     model_t.eval()

#     trainer_t = CheXpertTrainer_defer()
#     loss_t, auc_cls_t, auc_exp_t, auc_sys_t, defer_rate_t = trainer_t.test_epoch_defer(
#         model=model_t,
#         loader=dataLoaderTest_t,
#         device=DEVICE,
#         alpha=alpha,
#         rad_index=1)
    
#     # Store individual step results
#     step_sys_aurocs.append(auc_sys_t)
#     step_defer_rates.append(defer_rate_t)
#     step_cls_aucs.append(auc_cls_t)
#     step_exp_aucs.append(auc_exp_t)


# Create results directory if it doesn't exist
# os.makedirs('results', exist_ok=True)

# # # Create DataFrames and save results
# timesteps = list(range(1, SEQ_LEN + 1))

# # Use the recalculated results instead of reading from CSV
# # For LSTM and Sequential, we'll still read from CSV for now
# lstm_path = 'results/lstm_cumulative_results.csv'
# seq_path = 'results/seq_cumulative_results.csv'
# perstep_path = 'results/perstep_cumulative_results.csv'

# if os.path.exists(lstm_path):
#     lstm_df = pd.read_csv(lstm_path)
#     lstm_sys_aurocs = lstm_df['system_auroc']
#     lstm_cls_auc = lstm_df['cls_auroc']
#     lstm_exp_auc = lstm_df['exp_auroc']
#     lstm_defer_rates = lstm_df['defer_rate']
# else:
#     print("Warning: LSTM results not found, using empty lists")
#     lstm_sys_aurocs = []
#     lstm_cls_auc = []
#     lstm_exp_auc = []
#     lstm_defer_rates = []

# if os.path.exists(seq_path):
#     seq_df = pd.read_csv(seq_path)
#     seq_sys_aurocs = seq_df['system_auroc']
#     seq_cls_auc = seq_df['cls_auroc']
#     seq_exp_auc = seq_df['exp_auroc']
#     seq_defer_rates = seq_df['defer_rate']
# else:
#     print("Warning: Sequential results not found, using empty lists")
#     seq_sys_aurocs = []
#     seq_cls_auc = []
#     seq_exp_auc = []
#     seq_defer_rates = []

# if os.path.exists(perstep_path):
#     perstep_df = pd.read_csv(perstep_path)
#     perstep_sys_aurocs = perstep_df['system_auroc']
#     perstep_cls_auc = perstep_df['cls_auroc']
#     perstep_exp_auc = perstep_df['exp_auroc']
#     perstep_defer_rates = perstep_df['defer_rate']
#     perstep_sys_aurocs_new = []
#     perstep_defer_rates_new = []
#     perstep_cls_auc_new = []
#     perstep_exp_auc_new = []
#     for t in range(SEQ_LEN):
#         # Calculate cumulative average up to current step
#         valid_sys = [x for x in perstep_sys_aurocs[:t+1] if not np.isnan(x)]
#         valid_defer = [x for x in perstep_defer_rates[:t+1] if not np.isnan(x)]
#         valid_cls = [x for x in perstep_cls_auc[:t+1] if not np.isnan(x)]
#         valid_exp = [x for x in perstep_exp_auc[:t+1] if not np.isnan(x)]
        
#         if valid_sys:
#             perstep_sys_aurocs_new.append(np.mean(valid_sys))
#         else:
#             perstep_sys_aurocs_new.append(np.nan)
            
#         if valid_defer:
#             perstep_defer_rates_new.append(np.mean(valid_defer))
#         else:
#             perstep_defer_rates_new.append(np.nan)
            
#         if valid_cls:
#             perstep_cls_auc_new.append(np.mean(valid_cls))
#         else:
#             perstep_cls_auc_new.append(np.nan)
            
#         if valid_exp:
#             perstep_exp_auc_new.append(np.mean(valid_exp))
#         else:
#             perstep_exp_auc_new.append(np.nan)

# else:
#     print("Warning: Per-step results not found, using empty lists")
#     perstep_sys_aurocs_new = []
#     perstep_cls_auc_new = []
#     perstep_exp_auc_new = []
#     perstep_defer_rates_new = []

# # # LSTM results
# # if len(lstm_sys_aurocs) > 0:
# #     lstm_df = pd.DataFrame({
# #         'timestep': timesteps,
# #         'system_auroc': lstm_sys_aurocs,
# #         'cls_auroc': lstm_cls_auc,
# #         'exp_auroc': lstm_exp_auc,
# #         'defer_rate': lstm_defer_rates
# #     })
# #     lstm_df.to_csv('results/lstm_cumulative_results.csv', index=False)
# #     print(f"LSTM results saved to results/lstm_cumulative_results.csv")

# # # Sequential results
# # if len(seq_sys_aurocs) > 0:
# #     seq_df = pd.DataFrame({
# #         'timestep': timesteps,
# #         'system_auroc': seq_sys_aurocs,
# #         'cls_auroc': seq_cls_auc,
# #         'exp_auroc': seq_exp_auc,
# #         'defer_rate': seq_defer_rates
# #     })
# #     seq_df.to_csv('results/seq_cumulative_results.csv', index=False)
# #     print(f"Sequential results saved to results/seq_cumulative_results.csv")

# # Per-step results
# perstep_df = pd.DataFrame({
#     'timestep': timesteps,
#     'system_auroc': perstep_sys_aurocs_new,
#     'cls_auroc': perstep_cls_auc_new,
#     'exp_auroc': perstep_exp_auc_new,
#     'defer_rate': perstep_defer_rates_new
# })
# perstep_df.to_csv('results/perstep_cumulative_results.csv', index=False)
# print(f"Per-step results saved to results/perstep_cumulative_results.csv")


# Combined results for comparison
# combined_df = pd.DataFrame({
#     'timestep': timesteps,
#     'lstm_system_auroc': lstm_sys_aurocs,
#     'lstm_cls_auroc': lstm_cls_auc,
#     'lstm_exp_auroc': lstm_exp_auc,
#     'lstm_defer_rate': lstm_defer_rates,
#     'seq_system_auroc': seq_sys_aurocs,
#     'seq_cls_auroc': seq_cls_auc,
#     'seq_exp_auroc': seq_exp_auc,
#     'seq_defer_rate': seq_defer_rates,
#     'perstep_system_auroc': perstep_sys_aurocs_new,
#     'perstep_cls_auroc': perstep_cls_auc_new,
#     'perstep_exp_auroc': perstep_exp_auc_new,
#     'perstep_defer_rate': perstep_defer_rates_new
# })
# combined_df.to_csv('results/combined_cumulative_results.csv', index=False)
# print(f"Combined results saved to results/combined_cumulative_results.csv")

combined_df = pd.read_csv('results/combined_cumulative_results.csv')
lstm_sys_aurocs = combined_df['lstm_system_auroc']
lstm_cls_auc = combined_df['lstm_cls_auroc']
lstm_exp_auc = combined_df['lstm_exp_auroc']
lstm_defer_rates = combined_df['lstm_defer_rate']
seq_sys_aurocs = combined_df['seq_system_auroc']
seq_cls_auc = combined_df['seq_cls_auroc']
seq_exp_auc = combined_df['seq_exp_auroc']
seq_defer_rates = combined_df['seq_defer_rate']
perstep_sys_aurocs_new = combined_df['perstep_system_auroc']
perstep_cls_auc_new = combined_df['perstep_cls_auroc']
perstep_exp_auc_new = combined_df['perstep_exp_auroc']
perstep_defer_rates_new = combined_df['perstep_defer_rate']



print(f"LSTM Cumulative System AUROCs: {lstm_sys_aurocs}")
print(f"LSTM Cumulative Defer Rates: {lstm_defer_rates}")
print(f"Seq Cumulative System AUROCs: {seq_sys_aurocs}") 
print(f"Seq Cumulative Defer Rates: {seq_defer_rates}")
print(f"Per-Step Cumulative System AUROCs: {perstep_sys_aurocs_new}")
print(f"Per-Step Cumulative Defer Rates: {perstep_defer_rates_new}")

# Only plot if we have data
if len(lstm_sys_aurocs) > 0 or len(seq_sys_aurocs) > 0 or len(perstep_sys_aurocs_new) > 0:
    plt.figure(figsize=(6,4))
    ts = np.arange(1, SEQ_LEN + 1)
    if len(lstm_sys_aurocs) > 0:
        plt.plot(ts, lstm_sys_aurocs, marker='o', label='LSTM-L2D Cumulative AUROC')
    if len(seq_sys_aurocs) > 0:
        plt.plot(ts, seq_sys_aurocs, marker='s', label='L2D Cumulative AUROC')
    if len(perstep_sys_aurocs_new) > 0:
        plt.plot(ts, perstep_sys_aurocs_new, marker='^', label='Per-Step-L2D Cumulative AUROC')
plt.xlabel("Timestep t")
plt.ylabel("Cumulative System AUROC")
plt.title("Cumulative L2D System AUROC over time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/fatigue_plot_with_prior_cumulative.png")
plt.show()

plt.figure(figsize=(6,4))
plt.plot(ts, lstm_defer_rates, marker='o', label='LSTM-L2D Cumulative Defer Rate')
plt.plot(ts, seq_defer_rates, marker='s', label='L2D Cumulative Defer Rate')
plt.plot(ts, perstep_defer_rates_new, marker='^', label='Per-Step-L2D Cumulative Defer Rate')
plt.xlabel("Timestep t")
plt.ylabel("Cumulative Defer Rate")
plt.title("Cumulative Defer Rate over time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/fatigue_plot_with_prior_cumulative_time.png")
plt.show()

plt.figure(figsize=(6,4))
plt.plot(ts, lstm_cls_auc, marker='o', label='LSTM-L2D Classifier AUROC')
plt.plot(ts, lstm_exp_auc, marker='o', label='LSTM-L2D Expert AUROC')
plt.plot(ts, seq_cls_auc, marker='s', label='L2D Classifier AUROC')
plt.plot(ts, seq_exp_auc, marker='s', label='L2D Expert AUROC')
plt.plot(ts, perstep_cls_auc_new, marker='^', label='Per-Step-L2D Classifier AUROC')
plt.plot(ts, perstep_exp_auc_new, marker='^', label='Per-Step-L2D Expert AUROC')
plt.xlabel("Timestep t")
plt.ylabel("Cumulative AUROC")
plt.title("Cumulative Expert AUROC and Classifier AUROC over time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/fatigue_plot_with_prior_cumulative_time_exp_cls.png")
plt.show()



plt.figure(figsize=(6,4))
plt.plot(ts, lstm_defer_rates, marker='o', label='LSTM-L2D Cumulative Defer Rate')
plt.plot(ts, seq_defer_rates, marker='s', label='L2D Cumulative Defer Rate')
plt.plot(ts, perstep_defer_rates_new, marker='^', label='Per-Step-L2D Cumulative Defer Rate')
plt.xlabel("Timestep t")
plt.ylabel("Cumulative Defer Rate")
plt.title("Cumulative Defer Rate over time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/fatigue_plot_with_prior_cumulative_time_defer.png")
plt.show()