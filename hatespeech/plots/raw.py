import os
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *

df = pd.read_csv("results/combined_raw_results_new.csv")

system_acc_dict = {}
coverage_dict = {}
system_acc_ste_dict = {}
coverage_ste_dict = {}

for f in ["general", "lstm", "perstep"]:
    df_f = df[df["model"] == f]
    system_acc_list = []
    coverage_list = []
    system_acc_ste_list = []
    coverage_ste_list = []
    for t in range(10):
        df_t = df_f[df_f["timestep"] == t]
        system_acc = df_t["system_acc"].mean()
        coverage = df_t["coverage"].mean()
        system_acc_ste = df_t["system_acc"].std() / np.sqrt(df_t["system_acc"].count())
        coverage_ste = df_t["coverage"].std() / np.sqrt(df_t["coverage"].count())
        print(f"Accumulated system accuracy at timestep {t}: {system_acc} ± {system_acc_ste}")
        print(f"Accumulated coverage at timestep {t}: {coverage} ± {coverage_ste}")
        system_acc_list.append(system_acc)
        coverage_list.append(coverage)
        system_acc_ste_list.append(system_acc_ste)
        coverage_ste_list.append(coverage_ste)

    timestep_list = list(range(10))
    system_acc_list = [acc/ 100 for acc in system_acc_list]
    coverage_list = [acc/ 100 for acc in coverage_list]
    system_acc_ste_list = [acc / 100 for acc in system_acc_ste_list]
    coverage_ste_list = [acc / 100 for acc in coverage_ste_list]

    system_acc_dict[f] = system_acc_list
    coverage_dict[f] = coverage_list
    system_acc_ste_dict[f] = system_acc_ste_list
    coverage_ste_dict[f] = coverage_ste_list


os.makedirs("figures", exist_ok=True)

plt.figure(figsize=(6, 6))
for f in ["general", "lstm", "perstep"]:
    plt.plot(timestep_list, system_acc_dict[f], label=f)
    system_acc_dict_bottom = [acc - acc_ste for acc, acc_ste in zip(system_acc_dict[f], system_acc_ste_dict[f])]
    system_acc_dict_top = [acc + acc_ste for acc, acc_ste in zip(system_acc_dict[f], system_acc_ste_dict[f])]
    plt.fill_between(timestep_list, system_acc_dict_bottom, system_acc_dict_top, alpha=0.2)
plt.legend()
plt.show()
plt.savefig("figures/raw_system_accuracy_vs_timestep_new.png")

plt.figure(figsize=(6, 6))
for f in ["general", "lstm", "perstep"]:
    plt.plot(timestep_list, coverage_dict[f], label=f)
    coverage_dict_bottom = [acc - acc_ste for acc, acc_ste in zip(coverage_dict[f], coverage_ste_dict[f])]
    coverage_dict_top = [acc + acc_ste for acc, acc_ste in zip(coverage_dict[f], coverage_ste_dict[f])]
    plt.fill_between(timestep_list, coverage_dict_bottom, coverage_dict_top, alpha=0.2) 
plt.legend()
plt.show()
plt.savefig("figures/raw_coverage_vs_timestep_new.png")

def plot_box_by_timestep(df, metric_col, title, outfile):
    plt.figure(figsize=(9, 5))
    ax = plt.gca()
    # 按模型分三列小图（更清楚）
    models = ["general","lstm","perstep"]
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4), sharey=True)
    for j, m in enumerate(models):
        sub = df[df["model"]==m]
        # 准备每个 t 的数组列表
        data_by_t = [sub[sub["timestep"]==t][metric_col].dropna().values for t in range(10)]
        axes[j].boxplot(data_by_t, positions=list(range(10)), showfliers=False)
        # 种子点（抖动）
        for t in range(10):
            ys = data_by_t[t]
            xs = np.random.normal(t, 0.05, size=len(ys))
            axes[j].scatter(xs, ys, s=10, alpha=0.6)
        axes[j].set_title(m)
        axes[j].set_xlabel("timestep")
        axes[j].grid(alpha=0.3)
    axes[0].set_ylabel(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    print("[SAVE]", outfile)


def agg_median_iqr(df, value_col):
    # 必要列检查
    need = {"model", "timestep", value_col}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"missing columns: {sorted(missing)}")
    s = pd.to_numeric(df[value_col], errors="coerce")
    tmp = df[["model", "timestep"]].copy()
    tmp[value_col] = s

    g = tmp.groupby(["model", "timestep"])[value_col]
    g_med = g.median().rename("median")
    g_q1  = g.quantile(0.25).rename("q1")
    g_q3  = g.quantile(0.75).rename("q3")

    out = pd.concat([g_med, g_q1, g_q3], axis=1).reset_index()
    out = out.sort_values(["model", "timestep"], kind="mergesort")
    return out

def plot_median_iqr(df, value_col, ylabel, outfile, ylim_min=0, ylim_max=1):
    plt.figure(figsize=(5,5))
    for m in ["general","lstm","perstep"]:
        g = agg_median_iqr(df[df["model"]==m], value_col)
        if g.empty: 
            continue
        g = g.sort_values("timestep")
        x = g["timestep"].values
        y  = (g["median"].values / 100.0).astype(float)
        y1 = (g["q1"].values     / 100.0).astype(float)
        y3 = (g["q3"].values     / 100.0).astype(float)
        # 避免 NaN 影响绘图
        y  = np.nan_to_num(y,  nan=np.nanmean(y))
        y1 = np.nan_to_num(y1, nan=np.nanmean(y1))
        y3 = np.nan_to_num(y3, nan=np.nanmean(y3))
        plt.plot(x, y, label=m, linewidth=2)
        plt.fill_between(x, y1, y3, alpha=0.20)
    plt.xlabel("timestep"); plt.ylabel(ylabel)
    plt.ylim(ylim_min,ylim_max); 
    plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(outfile, dpi=200); print("[SAVE]", outfile)

plot_median_iqr(df, "system_acc", "system accuracy", "figures/median_raw_system_acc_by_timestep_new.png", ylim_min=0.88, ylim_max=0.98)
plot_median_iqr(df, "coverage", "coverage", "figures/median_raw_coverage_by_timestep_new.png", ylim_min=0.3, ylim_max=0.85)
plot_box_by_timestep(df, "system_acc", "system accuracy", "figures/raw_system_acc_by_timestep_new.png")
plot_box_by_timestep(df, "coverage", "coverage", "figures/raw_coverage_by_timestep_new.png")