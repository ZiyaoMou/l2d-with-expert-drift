import os
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *

df = pd.read_csv("results/combined_cumulative_results_new.csv")

accum_system_acc_dict = {}
accum_coverage_dict = {}
accum_system_acc_ste_dict = {}
accum_coverage_ste_dict = {}

for f in ["general", "lstm", "perstep"]:
    df_f = df[df["model"] == f]
    accum_system_acc_list = []
    accum_coverage_list = []
    accum_system_acc_ste_list = []
    accum_coverage_ste_list = []
    for t in range(10):
        df_t = df_f[df_f["timestep"] == t]
        accum_system_acc = df_t["accum_system_acc"].mean()
        accum_coverage = df_t["accum_coverage"].mean()
        accum_system_acc_ste = df_t["accum_system_acc"].std() / np.sqrt(df_t["accum_system_acc"].count())
        accum_coverage_ste = df_t["accum_coverage"].std() / np.sqrt(df_t["accum_coverage"].count())
        print(f"Accumulated system accuracy at timestep {t}: {accum_system_acc} ± {accum_system_acc_ste}")
        print(f"Accumulated coverage at timestep {t}: {accum_coverage} ± {accum_coverage_ste}")
        accum_system_acc_list.append(accum_system_acc)
        accum_coverage_list.append(accum_coverage)
        accum_system_acc_ste_list.append(accum_system_acc_ste)
        accum_coverage_ste_list.append(accum_coverage_ste)

    timestep_list = list(range(10))
    accum_system_acc_list = [acc/ 100 for acc in accum_system_acc_list]
    accum_coverage_list = [acc/ 100 for acc in accum_coverage_list]
    accum_system_acc_ste_list = [acc / 100 for acc in accum_system_acc_ste_list]
    accum_coverage_ste_list = [acc / 100 for acc in accum_coverage_ste_list]

    accum_system_acc_dict[f] = accum_system_acc_list
    accum_coverage_dict[f] = accum_coverage_list
    accum_system_acc_ste_dict[f] = accum_system_acc_ste_list
    accum_coverage_ste_dict[f] = accum_coverage_ste_list


os.makedirs("figures", exist_ok=True)

plt.figure(figsize=(6, 6))
for f in ["general", "lstm", "perstep"]:
    plt.plot(timestep_list, accum_system_acc_dict[f], label=f)
    accum_system_acc_dict_bottom = [acc - acc_ste for acc, acc_ste in zip(accum_system_acc_dict[f], accum_system_acc_ste_dict[f])]
    accum_system_acc_dict_top = [acc + acc_ste for acc, acc_ste in zip(accum_system_acc_dict[f], accum_system_acc_ste_dict[f])]
    plt.fill_between(timestep_list, accum_system_acc_dict_bottom, accum_system_acc_dict_top, alpha=0.2)
plt.legend()
plt.show()
plt.savefig("figures/accumulated_system_accuracy_vs_timestep_new.png")

plt.figure(figsize=(6, 6))
for f in ["general", "lstm", "perstep"]:
    plt.plot(timestep_list, accum_coverage_dict[f], label=f)
    accum_coverage_dict_bottom = [acc - acc_ste for acc, acc_ste in zip(accum_coverage_dict[f], accum_coverage_ste_dict[f])]
    accum_coverage_dict_top = [acc + acc_ste for acc, acc_ste in zip(accum_coverage_dict[f], accum_coverage_ste_dict[f])]
    plt.fill_between(timestep_list, accum_coverage_dict_bottom, accum_coverage_dict_top, alpha=0.2) 
plt.legend()
plt.show()
plt.savefig("figures/accumulated_coverage_vs_timestep_new.png")