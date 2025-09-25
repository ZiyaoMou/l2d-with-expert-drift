import os
import pandas as pd
import matplotlib.pyplot as plt

out_rows_csv = '/home/zmou1/scratchenalisn1/ziyao/l2d-cog/cifar10H/results/mixed_general_mixed_perstep.csv'
df = pd.read_csv(out_rows_csv).sort_values('step').reset_index(drop=True)

trial_index = df["step"]
S = df["system_acc"].astype(float)
M = df["classifier_acc"].astype(float)
H = df["expert_acc"].astype(float)
D = df["defer_rate"].astype(float)


os.makedirs("figures/mixed_curves", exist_ok=True)

plt.figure(figsize=(10, 6))
plt.plot(trial_index, S, marker='o', label='System Accuracy')
plt.plot(trial_index, M, marker='o', label='Classifier Accuracy')
plt.plot(trial_index, H, marker='o', label='Human Accuracy')
plt.xlabel("Trial Index")
plt.ylabel("Accuracy")
plt.title("General L2D Model Accuracy on CIFAR-10H Over Trial Index")
plt.legend(loc='lower right', fontsize=8)
plt.ylim(0.5, 1.0)
plt.tight_layout()
plt.savefig("figures/mixed_curves/general_model_accuracy_over_trial_index.png", dpi=200)

WINDOW = 7
S_s = S.rolling(window=WINDOW, center=True, min_periods=1).mean()
M_s = M.rolling(window=WINDOW, center=True, min_periods=1).mean()
H_s = H.rolling(window=WINDOW, center=True, min_periods=1).mean()
D_s = D.rolling(window=WINDOW, center=True, min_periods=1).mean()

plt.figure(figsize=(10, 6))
plt.plot(trial_index, S_s, linewidth=2.2, label="System Accuracy (smoothed)")
plt.plot(trial_index, M_s, linewidth=2.0, label="Classifier Accuracy (smoothed)")
plt.plot(trial_index, H_s, linewidth=2.0, label="Human Accuracy (smoothed)")


plt.scatter(trial_index, S, s=10, alpha=0.2)
plt.scatter(trial_index, M, s=10, alpha=0.2)
plt.scatter(trial_index, H, s=10, alpha=0.2)

plt.xlabel("Trial Index")
plt.ylabel("Accuracy")
plt.title("General L2D Model Accuracy on CIFAR-10H Over Trial Index (Smoothed)")
plt.ylim(0.5, 1.0)
plt.grid(True, alpha=0.25, linestyle="--")
plt.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("figures/mixed_curves/general_model_accuracy_over_trial_index_smooth.png", dpi=200)
# plt.show()

plt.plot(trial_index, D_s, linewidth=2.0, label="Defer Rate (smoothed)")
plt.scatter(trial_index, D, s=10, alpha=0.2)
plt.xlabel("Trial Index")
plt.ylabel("Defer Rate")
plt.title("General L2D Model Defer Rate on CIFAR-10H Over Trial Index")
plt.ylim(0.0, 1.0)
plt.grid(True, alpha=0.25, linestyle="--")
plt.legend(loc="lower right", fontsize=9)
plt.tight_layout()
os.makedirs("figures/mixed_curves", exist_ok=True)
plt.savefig("figures/mixed_curves/general_model_defer_rate_over_trial_index_smooth.png", dpi=200)
# plt.show()