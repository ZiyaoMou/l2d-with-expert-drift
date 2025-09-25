import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F
from data.cheXpert import split_dataset
from models.__init__ import model_loader
from experts.fake import ExpertModel_fake
from sklearn.metrics import roc_auc_score, average_precision_score

# hyper-parameters
N_CLASS     = 14
BATCH_SIZE  = 16
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
ROOT_DIR    = "/home/zmou1/scratchenalisn1/ziyao/l2d-data/CheXpert/"
TRAIN_CSV   = f"{ROOT_DIR}CheXpert-v1.0-small/train.csv"
VALID_CSV   = f"{ROOT_DIR}CheXpert-v1.0-small/valid.csv"
MAX_TRIALS  = 2

# alpha values for annotation
alpha = [1,1,0.1,1,1,0.4,0.2,1,0.1,1,0.1,1,1,1]

# Class names for titles
CLASS_NAMES = [
    'No Finding','Enlarged Cardiomediastinum','Cardiomegaly',
    'Lung Opacity','Lung Lesion','Edema','Consolidation',
    'Pneumonia','Atelectasis','Pneumothorax','Pleural Effusion',
    'Pleural Other','Fracture','Support Devices'
]

def run_trial(mdl_cls, mdl_cls_cal, mdl_rad, mdl_rad_cal, mdl_def, dl_test):
    for m in (mdl_cls, mdl_cls_cal, mdl_rad, mdl_rad_cal, mdl_def):
        m.to(DEVICE).eval()

    # storage
    scores = {
        "defer": {"sc": [[] for _ in range(N_CLASS)], "fb": [[] for _ in range(N_CLASS)]},
        "mc":    {"sc": [[] for _ in range(N_CLASS)], "fb": [[] for _ in range(N_CLASS)]},
        "conf":  {"sc": [[] for _ in range(N_CLASS)], "fb": [[] for _ in range(N_CLASS)]},
    }
    expert = [[] for _ in range(N_CLASS)]
    target = [[] for _ in range(N_CLASS)]
    rad_index = 1

    with torch.no_grad():
        for imgs, y, r1, r2, r3, w in dl_test:
            imgs = imgs.to(DEVICE); y_np = y.cpu().numpy()
            out_def = mdl_def(imgs.view(-1, *imgs.shape[1:]))
            out_cls = mdl_cls(imgs.view(-1, *imgs.shape[1:]))
            out_c   = mdl_cls_cal(imgs.view(-1, *imgs.shape[1:]))
            out_r   = mdl_rad_cal(imgs.view(-1, *imgs.shape[1:]))

            bs = imgs.size(0)
            for b in range(bs):
                jd, jc, jf = 0, 0, 0
                for cls in range(N_CLASS):
                    if w[b, cls].item() == 1:
                        # Defer
                        pd = F.softmax(out_def[b, jd:jd+3], dim=0)
                        scores["defer"]["sc"][cls].append((pd[2] - pd[:2].max()).item())
                        scores["defer"]["fb"][cls].append(pd[1].item())
                        # ModelConfidence
                        pc = F.softmax(out_cls[b, jc:jc+2], dim=0)
                        scores["mc"]["sc"][cls].append((1 - pc.max()).item())
                        scores["mc"]["fb"][cls].append(pc[1].item())
                        # ConfidenceDifference
                        sc = F.softmax(out_c[b, jf:jf+2], dim=0)
                        sr = F.softmax(out_r[b, jf:jf+2], dim=0)
                        scores["conf"]["sc"][cls].append((sr[1] - sc.max()).item())
                        scores["conf"]["fb"][cls].append(sc[1].item())
                        # expert & target
                        r = r1 if rad_index==1 else (r2 if rad_index==2 else r3)
                        expert[cls].append(r[b, cls].item())
                        target[cls].append(y_np[b, cls])
                    jd += 3; jc += 2; jf += 2

    # compute curves
    covs = np.linspace(0,1,101)
    auc_curves = {k: np.zeros((N_CLASS, len(covs))) for k in scores}
    ap_curves  = {k: np.zeros((N_CLASS, len(covs))) for k in scores}
    for key in scores:
        for cls in range(N_CLASS):
            sc = np.array(scores[key]["sc"][cls])
            fb = np.array(scores[key]["fb"][cls])
            ex = np.array(expert[cls])
            tg = np.array(target[cls])
            idx = np.argsort(sc)
            for i, c in enumerate(covs):
                tau = sc[idx][int(c*(len(sc)-1))]
                pred = np.where(sc>=tau, ex, fb)
                auc_curves[key][cls,i] = roc_auc_score(tg, pred)
                ap_curves [key][cls,i] = average_precision_score(tg, pred)
    return covs, auc_curves, ap_curves

def main():
    all_auc = {k: [] for k in ("defer","mc","conf")}
    all_ap  = {k: [] for k in ("defer","mc","conf")}
    for _ in range(MAX_TRIALS):
        exp = ExpertModel_fake(confounding_class=13, p_confound=0.7, p_nonconfound=1.0)
        _, dl_val, dl_test, _, _ = split_dataset(
            train_size=0.999, random_seed=66, root_dir=ROOT_DIR,
            pathFileTrain=TRAIN_CSV, pathFileValid=VALID_CSV,
            exp_fake=exp, trBatchSize=BATCH_SIZE
        )
        mdl_cls, mdl_cls_cal, mdl_rad, mdl_rad_cal, mdl_def = model_loader(nnClassCount=N_CLASS, dataLoaderValidTrain=dl_val)
        covs, aucs, aps = run_trial(mdl_cls, mdl_cls_cal, mdl_rad, mdl_rad_cal, mdl_def, dl_test)
        for k in aucs:
            all_auc[k].append(aucs[k])
            all_ap [k].append(aps[k])

    mean_auc = {k: np.mean(all_auc[k], axis=0) for k in all_auc}
    std_auc  = {k: np.std (all_auc[k], axis=0) for k in all_auc}
    print(mean_auc)
    print(std_auc)
    
    # Debug: print some statistics to check if error bars have meaningful values
    print("Debug: Error bar statistics for class 2 (Cardiomegaly):")
    for k in ["defer", "conf", "mc"]:
        print(f"{k}: mean={mean_auc[k][2].mean():.4f}, std={std_auc[k][2].mean():.4f}, max_std={std_auc[k][2].max():.4f}")

    # plotting identical to original
    classes = [2,5,6,8,10]
    j = 0
    for k in range(N_CLASS):
        if k not in classes: continue
        ax = plt.subplot(1,5,j+1)
        ax.set_title(CLASS_NAMES[k], fontsize=12, fontweight='bold')
        # ours
        m = mean_auc["defer"][k]; s = std_auc["defer"][k]
        ax.errorbar(covs, m, yerr=s,
                    label=f'ours $\\alpha$={alpha[k]} {m.max():.3f}',
                    color='maroon', ecolor='lightcoral', elinewidth=4, capsize=0)
        # ConfidenceDifference
        m2 = mean_auc["conf"][k]; s2 = std_auc["conf"][k]
        ax.errorbar(covs, m2, yerr=s2,
                    label=f'Confidence {m2.max():.3f}',
                    color='blue', ecolor='lightblue', elinewidth=4, capsize=0)
        # ModelConfidence
        m3 = mean_auc["mc"][k]; s3 = std_auc["mc"][k]
        ax.errorbar(covs, m3, yerr=s3,
                    label=f'ModelConfidence {m3.max():.3f}',
                    color='black', ecolor='lightgray', elinewidth=4, capsize=0)

        ax.legend(loc='lower left', fontsize=9)
        ax.set_xlim([0,1]); ax.set_ylim([0.7,0.95])
        if j == 0:
            ax.annotate('Expert',
                        xy=(covs[0], m[0]),
                        xytext=(covs[0]+0.1, m[0]-0.05),
                        arrowprops=dict(facecolor='black', width=0.2, headwidth=3, headlength=6))
            ax.annotate('Classifier',
                        xy=(covs[-1], m[-1]-0.002),
                        xytext=(covs[-1]-0.4, m[-1]-0.05),
                        arrowprops=dict(facecolor='black', width=0.2, headwidth=3, headlength=6))
            ax.set_ylabel('AU-ROC')
        else:
            ax.set_yticks([0.7,0.75,0.8,0.85])
            ax.set_yticklabels([])
        ax.set_xlabel('Coverage')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        j += 1

    plt.gcf().set_size_inches(30,7)
    plt.tight_layout()
    plt.savefig("figures/auc_vs_cov_simplified.png", dpi=1000)
    plt.show()

if __name__ == "__main__":
    main()