import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
gru_dir = "/home/zmou1/scratchenalisn1/ziyao/l2d-cog/cifar10H/results/gru_curve_ft_test/20250924_101859"
perstep_dir = "/home/zmou1/scratchenalisn1/ziyao/l2d-cog/cifar10H/results/perstep_curve_test"
general_dir = "/home/zmou1/scratchenalisn1/ziyao/l2d-cog/cifar10H/results/general_curve_v2_test/20250924_110543"

METRICS = {
    "Coverage": "coverage",
    "System accuracy (%)": "system_acc",
    "Alone-classifier accuracy (%)": "alone_classifier",
}

def _pick_step_col(df):
    for c in ["step", "t", "timestep", "time_step"]:
        if c in df.columns:
            return c
    raise ValueError("找不到时间步列（需要列名 'step' 或 't' 或 'timestep'）")

def _load_gru_frames(gru_dir):
    # 匹配 general_model_curve_mixed_seedxx.csv
    dfs = []
    for fp in glob.glob(os.path.join(gru_dir, "general_model_curve_mixed_seed*.csv")):
        df = pd.read_csv(fp)
        step_col = _pick_step_col(df)
        df = df.rename(columns={step_col: "step"})
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError("GRU 结果未找到：general_model_curve_mixed_seed*.csv")
    return dfs

def _load_perstep_frames(perstep_dir):
    # 匹配 seed_xx/results.csv
    dfs = []
    for seed_dir in glob.glob(os.path.join(perstep_dir, "seed_*")):
        fp = os.path.join(seed_dir, "results.csv")
        if os.path.exists(fp):
            df = pd.read_csv(fp)
            step_col = _pick_step_col(df)
            df = df.rename(columns={step_col: "step"})
            dfs.append(df)
    if not dfs:
        raise FileNotFoundError("Per-step 结果未找到：seed_*/result.csv")
    return dfs

def _align_and_stack(dfs, value_col):
    all_steps = sorted(set().union(*[df["step"].unique().tolist() for df in dfs]))
    series_list = []
    for df in dfs:
        s = df.set_index("step")[value_col]
        s = s.reindex(all_steps)
        series_list.append(s.to_numpy())
    M = np.vstack(series_list)  # [n_seeds, n_steps]
    return np.array(all_steps), M

def _mean_ste(M):
    mu = np.nanmean(M, axis=0)
    ste = np.nanstd(M, axis=0, ddof=1) / np.sqrt(M.shape[0])
    return mu, ste
gru_dfs = _load_gru_frames(gru_dir)
per_dfs = _load_perstep_frames(perstep_dir)
general_dfs = _load_gru_frames(general_dir)

for y_label, col in METRICS.items():
    # GRU
    steps_gru, M_gru = _align_and_stack(gru_dfs, col)
    mu_gru, ste_gru = _mean_ste(M_gru)

    # Per-step
    steps_per, M_per = _align_and_stack(per_dfs, col)
    mu_per, ste_per = _mean_ste(M_per)

    steps_general, M_general = _align_and_stack(general_dfs, col)
    mu_general, ste_general = _mean_ste(M_general)

    # 画图
    plt.figure()
    plt.plot(steps_gru, mu_gru, label="GRU / temporal L2D")
    plt.fill_between(steps_gru, mu_gru - ste_gru, mu_gru + ste_gru, alpha=0.2)

    plt.plot(steps_per, mu_per, label="Per-step L2D")
    plt.fill_between(steps_per, mu_per - ste_per, mu_per + ste_per, alpha=0.2)

    plt.plot(steps_general, mu_general, label="General L2D")
    plt.fill_between(steps_general, mu_general - ste_general, mu_general + ste_general, alpha=0.2)

    plt.xlabel("timestep")
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # 文件名友好
    safe_name = re.sub(r"[^a-zA-Z0-9_]+", "_", y_label).strip("_").lower()
    out_png = f"figures/{safe_name}_vs_timestep.png"
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"Saved: {out_png}")