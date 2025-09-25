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
from scipy.stats import sem

N_RUNS = 10
SEQ_LEN = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

alpha = [1, 1, 0.1, 1, 1, 0.4, 0.2, 1, 0.1, 1, 0.1, 1, 1, 1]

ROOT_DIR = "/home/zmou1/scratchenalisn1/ziyao/l2d-data/CheXpert/"
TRAIN_CSV = os.path.join(ROOT_DIR, "CheXpert-v1.0-small/train.csv")
VALID_CSV = os.path.join(ROOT_DIR, "CheXpert-v1.0-small/valid.csv")

def init_metric_dict():
    return {
        "lstm_sys_aurocs": [],
        "lstm_defer_rates": [],
        "lstm_best_aucs": [],
        "lstm_best_covs": [],
        "seq_sys_aurocs": [],
        "seq_defer_rates": [],
        "seq_best_aucs": [],
        "seq_best_covs": [],
        "perstep_sys_aurocs": [[] for _ in range(SEQ_LEN)],
        "perstep_defer_rates": [[] for _ in range(SEQ_LEN)],
        "perstep_best_aucs": [[] for _ in range(SEQ_LEN)],
        "perstep_best_covs": [[] for _ in range(SEQ_LEN)],
    }

results = init_metric_dict()

for run in range(N_RUNS):
    print(f"\n==== Run {run+1}/{N_RUNS} ====")

    seed = 66
    model = DenseLSTMDefer(num_classes=14).to(DEVICE)
    model.load_state_dict(torch.load("checkpoints/densenet_lstm_fatigue_new.pth", map_location=DEVICE))
    model_prior = DenseNet121SeqDefer(num_classes=14).to(DEVICE)
    model_prior.load_state_dict(torch.load("checkpoints/densenet_seq_defer.pth", map_location=DEVICE))

    exp_biased = ExpertModelBiased(p_confound=0.7,
                                p_nonconfound=1,
                                decay_confound=0.07,
                                decay_nonconfound=0.1,
                                confounding_class=13,
                                use_fatigue=True,
                                seq_len=SEQ_LEN,
                                num_classes=14)
    _, _, test_loader = split_dataset_seq(random_seed=seed, exp_fake=exp_biased,
        train_size=0.1,
        root_dir=ROOT_DIR,
        pathFileValid=VALID_CSV,
        pathFileTrain=TRAIN_CSV)

    lstm_trainer = CheXpertTrainerDeferLSTM(model, exp_biased, DEVICE)
    seq_trainer = SeqTrainerDefer(model_prior, exp_biased, DEVICE)

    _, _, _, sys_aurocs, defer_rates = lstm_trainer.test_epoch(test_loader, loss_fn=l2d_loss)
    _, _, best_aucs, best_covs = lstm_trainer.test_epoch_best_coverage(test_loader, rad_index=1)

    results["lstm_sys_aurocs"].append(sys_aurocs)
    results["lstm_defer_rates"].append(defer_rates)
    results["lstm_best_aucs"].append(best_aucs)
    results["lstm_best_covs"].append(best_covs)

    _, _, _, sys_aurocs_seq, defer_rates_seq = seq_trainer.validate_epoch(test_loader, use_defer=True)
    _, _, best_aucs_seq, best_covs_seq = seq_trainer.validate_epoch_best_coverage(test_loader, rad_index=1)

    results["seq_sys_aurocs"].append(sys_aurocs_seq)
    results["seq_defer_rates"].append(defer_rates_seq)
    results["seq_best_aucs"].append(best_aucs_seq)
    results["seq_best_covs"].append(best_covs_seq)

    for t in range(SEQ_LEN):
        expert_t = ExpertModel_fake(
            confounding_class=13,
            p_confound=0.7-0.07*t,
            p_nonconfound=1-0.1*t
        )
        _, _, dataLoaderTest_t, _, _ = split_dataset_shuffle(
        exp_fake=expert_t,
        train_size=0.1,
        random_seed=66,
        root_dir=ROOT_DIR,
        pathFileValid=VALID_CSV,
        pathFileTrain=TRAIN_CSV,
        )

        model_t = DenseNet121_defer(out_size=14).to(DEVICE)
        model_t.load_state_dict(torch.load(f"checkpoints/perstep-new/densenet_defer_step_{t}.pth", map_location=DEVICE))
        model_t.eval()

        trainer_t = CheXpertTrainer_defer()
        _, _, _, sys_auc_t, defer_rate_t = trainer_t.test_epoch_defer(model_t, dataLoaderTest_t, DEVICE, alpha, rad_index=1)
        _, _, best_auc_t, best_cov_t = trainer_t.test_epoch_defer_best_coverage(model_t, dataLoaderTest_t, DEVICE, rad_index=1)

        results["perstep_sys_aurocs"][t].append(sys_auc_t)
        results["perstep_defer_rates"][t].append(defer_rate_t)
        results["perstep_best_aucs"][t].append(best_auc_t)
        results["perstep_best_covs"][t].append(best_cov_t)


def to_np_arr(metric_list):
    return np.array(metric_list)

def agg_and_plot(key_prefix, ylabel, filename):
    mean = to_np_arr(results[f"{key_prefix}"]).mean(axis=0)
    stderr = sem(to_np_arr(results[f"{key_prefix}"]), axis=0)

    ts = np.arange(1, SEQ_LEN+1)
    plt.figure(figsize=(6, 4))
    plt.plot(ts, mean, label=f"{key_prefix} mean", marker='o')
    plt.fill_between(ts, mean - stderr, mean + stderr, alpha=0.2)
    plt.xlabel("Timestep")
    plt.ylabel(ylabel)
    plt.title(f"{key_prefix} over time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"figures/{filename}.png")
    plt.show()

agg_and_plot("lstm_sys_aurocs", "System AUROC", "lstm_sys_auc")
agg_and_plot("lstm_defer_rates", "Defer Rate", "lstm_defer")
agg_and_plot("lstm_best_aucs", "Best AUROC", "lstm_best_auc")
agg_and_plot("lstm_best_covs", "Best Coverage", "lstm_best_cov")

agg_and_plot("seq_sys_aurocs", "System AUROC", "seq_sys_auc")
agg_and_plot("seq_defer_rates", "Defer Rate", "seq_defer")
agg_and_plot("seq_best_aucs", "Best AUROC", "seq_best_auc")
agg_and_plot("seq_best_covs", "Best Coverage", "seq_best_cov")

for t in range(SEQ_LEN):
    def step_plot(metric_key, ylabel, title_suffix):
        vals = np.array(results[metric_key][t])
        mean, std = np.mean(vals), sem(vals)
        return mean, std

   
ts = np.arange(1, SEQ_LEN+1)

df = pd.DataFrame({
    "Timestep": np.arange(1, SEQ_LEN+1),
    "LSTM AUROC Mean": np.mean(results["lstm_sys_aurocs"], axis=0),
    "LSTM AUROC StdErr": sem(results["lstm_sys_aurocs"], axis=0),
    "LSTM Defer Mean": np.mean(results["lstm_defer_rates"], axis=0),
    "LSTM Defer StdErr": sem(results["lstm_defer_rates"], axis=0),
    "Seq AUROC Mean": np.mean(results["seq_sys_aurocs"], axis=0),
    "Seq AUROC StdErr": sem(results["seq_sys_aurocs"], axis=0),
    "Seq Defer Mean": np.mean(results["seq_defer_rates"], axis=0),
    "Seq Defer StdErr": sem(results["seq_defer_rates"], axis=0),
})

df.to_csv("results/l2d_compare_avg.csv", index=False)



df_lstm = pd.DataFrame({
    "Timestep": ts,
    "System AUROC Mean": np.mean(results["lstm_sys_aurocs"], axis=0),
    "System AUROC StdErr": sem(results["lstm_sys_aurocs"], axis=0),
    "Defer Rate Mean": np.mean(results["lstm_defer_rates"], axis=0),
    "Defer Rate StdErr": sem(results["lstm_defer_rates"], axis=0),
    "Best AUC Mean": np.mean(results["lstm_best_aucs"], axis=0),
    "Best AUC StdErr": sem(results["lstm_best_aucs"], axis=0),
    "Best Coverage Mean": np.mean(results["lstm_best_covs"], axis=0),
    "Best Coverage StdErr": sem(results["lstm_best_covs"], axis=0),
})
df_lstm.to_csv("results/lstm_l2d_summary.csv", index=False)

# Seq-L2D
df_seq = pd.DataFrame({
    "Timestep": ts,
    "System AUROC Mean": np.mean(results["seq_sys_aurocs"], axis=0),
    "System AUROC StdErr": sem(results["seq_sys_aurocs"], axis=0),
    "Defer Rate Mean": np.mean(results["seq_defer_rates"], axis=0),
    "Defer Rate StdErr": sem(results["seq_defer_rates"], axis=0),
    "Best AUC Mean": np.mean(results["seq_best_aucs"], axis=0),
    "Best AUC StdErr": sem(results["seq_best_aucs"], axis=0),
    "Best Coverage Mean": np.mean(results["seq_best_covs"], axis=0),
    "Best Coverage StdErr": sem(results["seq_best_covs"], axis=0),
})
df_seq.to_csv("results/seq_l2d_summary.csv", index=False)

# Per-Step-L2D
df_perstep = pd.DataFrame({
    "Timestep": ts,
    "System AUROC Mean": [np.mean(results["perstep_sys_aurocs"][t]) for t in range(SEQ_LEN)],
    "System AUROC StdErr": [sem(results["perstep_sys_aurocs"][t]) for t in range(SEQ_LEN)],
    "Defer Rate Mean": [np.mean(results["perstep_defer_rates"][t]) for t in range(SEQ_LEN)],
    "Defer Rate StdErr": [sem(results["perstep_defer_rates"][t]) for t in range(SEQ_LEN)],
    "Best AUC Mean": [np.mean(results["perstep_best_aucs"][t]) for t in range(SEQ_LEN)],
    "Best AUC StdErr": [sem(results["perstep_best_aucs"][t]) for t in range(SEQ_LEN)],
    "Best Coverage Mean": [np.mean(results["perstep_best_covs"][t]) for t in range(SEQ_LEN)],
    "Best Coverage StdErr": [sem(results["perstep_best_covs"][t]) for t in range(SEQ_LEN)],
})

df_perstep.to_csv("results/perstep_l2d_summary.csv", index=False)