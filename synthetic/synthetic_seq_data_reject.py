# seq_l2d_experiment.py
# --------------------------------------------------------------
# 200-trial Sequential Learn-to-Defer on synthetic sequence data.
# Compares Global-L2D / Per-step-L2D / RNN-L2D and aggregates stats.
# Also prints a single-row RNN-L2D table with user-provided metrics.
# --------------------------------------------------------------

import os
import csv
import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Subset
from tqdm import tqdm

# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(123)

# ==============================================================
# Data: sequential synthetic dataset with population-specific decay
# ==============================================================

def _sample_gaussian(mu: torch.Tensor, var: torch.Tensor, n: int) -> torch.Tensor:
    # IID dimensions with provided mean/var per feature
    return torch.stack([torch.normal(mu, var.sqrt()) for _ in range(n)], 0)

def gen_seq_dataset(
    d: int = 10,
    N: int = 6000,
    T: int = 10,
    p_flip: float = 0.03,
    sigma: float = 0.5,
    p0_A: float = 0.90,
    p0_B: float = 0.85,
    lam_A: float = 0.15,
    lam_B: float = 0.60,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      X: [N, T, d]  features
      Y: [N, T]     labels {0,1}
      E: [N, T]     expert preds {0,1} with time-decay accuracy
      W: [N, T]     weights {0,1}
      G: [N]        group id {0(A),1(B)}
    """
    # population proportion in [0.02, 0.98]
    prop = float(np.clip(np.random.uniform(), 0.02, 0.98))
    nA = int(round(N * prop))
    nB = N - nA
    G = torch.zeros(N, dtype=torch.long)
    G[nA:] = 1  # A=0, B=1

    def rand_mv():
        mu, var = torch.rand(d) * d, torch.rand(d) * d
        return mu, var

    # 4 clusters: (A/B) x (y=1/y=0)
    muA1, varA1 = rand_mv()
    muA0, varA0 = rand_mv()
    muB1, varB1 = rand_mv()
    muB0, varB0 = rand_mv()

    # Initial labels within each population
    nA1 = nA // 2
    nA0 = nA - nA1
    nB1 = nB // 2
    nB0 = nB - nB1

    XA1_0 = _sample_gaussian(muA1, varA1, nA1)
    XA0_0 = _sample_gaussian(muA0, varA0, nA0)
    XB1_0 = _sample_gaussian(muB1, varB1, nB1)
    XB0_0 = _sample_gaussian(muB0, varB0, nB0)

    X0 = torch.cat([XA1_0, XA0_0, XB1_0, XB0_0], 0)
    Y0 = torch.cat([
        torch.ones(nA1), torch.zeros(nA0),
        torch.ones(nB1), torch.zeros(nB0)
    ], 0).long()

    # shuffle initial assignment
    perm = torch.randperm(N)
    X0, Y0, G = X0[perm], Y0[perm], G[perm]

    drift_A = 0.02 * torch.randn(d)
    drift_B = 0.05 * torch.randn(d)

    X = torch.zeros(N, T, d)
    Y = torch.zeros(N, T, dtype=torch.long)
    E = torch.zeros(N, T, dtype=torch.long)
    W = torch.ones (N, T, dtype=torch.long)

    X[:, 0] = X0
    Y[:, 0] = Y0

    for t in range(T):
        if t > 0:
            noise = sigma * torch.randn(N, d)
            drift = torch.where(G.unsqueeze(1) == 0, drift_A, drift_B)  # [N,d]
            X[:, t] = X[:, t - 1] + noise + drift
            flip = torch.bernoulli(torch.full((N,), p_flip)).long()
            Y[:, t] = torch.where(flip == 1, 1 - Y[:, t - 1], Y[:, t - 1])

        # time-decayed expert accuracy
        pA = p0_A * math.exp(-lam_A * t / float(T))
        pB = p0_B * math.exp(-lam_B * t / float(T))
        p_correct = torch.where(G == 0, torch.full((N,), pA), torch.full((N,), pB))
        corr = torch.bernoulli(p_correct).long()
        E[:, t] = torch.where(corr == 1, Y[:, t], 1 - Y[:, t])

    return X.to(device), Y.to(device), E.to(device), W.to(device), G.to(device)

# ==============================================================
# Models
# ==============================================================

class GlobalHead(nn.Module):
    """One shared linear head for all timesteps; produces 3 logits (0/1/defer)."""
    def __init__(self, d: int):
        super().__init__()
        self.fc = nn.Linear(d, 3)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, T, d = X.shape
        out = self.fc(X.view(B * T, d)).view(B, T, 3)
        return out

class PerStepHead(nn.Module):
    """One independent head per time step; each produces 3 logits."""
    def __init__(self, d: int, T: int):
        super().__init__()
        self.T = T
        self.heads = nn.ModuleList([nn.Linear(d, 3) for _ in range(T)])

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, T, d = X.shape
        outs = [self.heads[t](X[:, t, :]) for t in range(T)]
        return torch.stack(outs, dim=1)  # [B,T,3]

class RNNL2D(nn.Module):
    """
    - Classifier head: linear -> 2 logits (class 0/1).
    - Defer head: GRU over [x_t, e_{t-1}] -> 1 logit (defer).
    Final logits: concat [g0, g1, g_defer].
    """
    def __init__(self, d: int, hidden: int = 64):
        super().__init__()
        self.cls = nn.Linear(d, 2)
        self.rnn = nn.GRU(input_size=d + 1, hidden_size=hidden, batch_first=True)
        self.def_head = nn.Linear(hidden, 1)

    def forward(self, X: torch.Tensor, E_prev: torch.Tensor = None) -> torch.Tensor:
        B, T, d = X.shape
        g_cls = self.cls(X)  # [B,T,2]
        if E_prev is None:
            E_prev = torch.zeros(B, T, 1, device=X.device)
        rnn_in = torch.cat([X, E_prev], dim=-1)  # [B,T,d+1]
        h, _ = self.rnn(rnn_in)                  # [B,T,H]
        g_def = self.def_head(h)                 # [B,T,1]
        return torch.cat([g_cls, g_def], dim=-1) # [B,T,3]

# ==============================================================
# Loss (L2D for sequence, single binary task per timestep)
# ==============================================================

def l2d_loss_seq(
    logits: torch.Tensor,  # [B,T,3] (class0, class1, defer)
    labels: torch.Tensor,  # [B,T]   {0,1}
    expert: torch.Tensor,  # [B,T]   {0,1}
    weights: torch.Tensor, # [B,T]
    alpha: float = 1.0
) -> torch.Tensor:
    p = F.softmax(logits, dim=-1)          # [B,T,3]
    p_true  = p.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # [B,T]
    p_defer = p[..., 2]                                       # [B,T]

    expert_correct = (expert == labels).float()               # [B,T]
    w_cls = alpha * expert_correct + (1.0 - expert_correct)   # [B,T]

    loss_cls = - w_cls * weights * torch.log(p_true + 1e-12)
    loss_def = - weights * expert_correct * torch.log(p_defer + 1e-12)

    denom = weights.sum().clamp(min=1e-12)
    return (loss_cls.sum() + loss_def.sum()) / denom

# ==============================================================
# Metrics
# ==============================================================

@dataclass
class Metrics:
    coverage: float
    system_acc: float
    expert_acc: float
    classifier_acc: float

def evaluate(
    logits: torch.Tensor,   # [B,T,3]
    labels: torch.Tensor,   # [B,T]
    expert: torch.Tensor,   # [B,T]
) -> Metrics:
    with torch.no_grad():
        p = F.softmax(logits, dim=-1)
        p0 = p[..., 0]
        p1 = p[..., 1]
        pdef = p[..., 2]
        mask_defer = pdef > torch.maximum(p0, p1)  # [B,T] bool

        # classifier decision where not deferred
        pred_cls = (p1 > 0.5).long()
        sys_pred = torch.where(mask_defer, expert, pred_cls)

        total = labels.numel()

        # accuracies
        correct_sys = (sys_pred == labels).sum().item()

        cls_mask = (~mask_defer)
        exp_mask = (mask_defer)

        classifier_acc = (pred_cls[cls_mask] == labels[cls_mask]).float().mean().item() if cls_mask.any() else 0.0
        expert_acc     = (expert[exp_mask]    == labels[exp_mask]).float().mean().item() if exp_mask.any() else 0.0
        system_acc     = correct_sys / total
        coverage       = (cls_mask.sum().item()) / total

        return Metrics(coverage=coverage,
                       system_acc=system_acc,
                       expert_acc=expert_acc,
                       classifier_acc=classifier_acc)

# ==============================================================
# Training loops
# ==============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    method: str,  # "global" | "perstep" | "rnn"
    alpha: float = 1.0
) -> float:
    model.train()
    running = 0.0
    n = 0
    for X, Y, E, W in loader:
        X, Y, E, W = X.to(device), Y.to(device), E.to(device), W.to(device)

        if method in ("global", "perstep"):
            logits = model(X)  # [B,T,3]
        elif method == "rnn":
            E_prev = torch.zeros_like(E, dtype=torch.float32).unsqueeze(-1)
            E_prev[:, 1:, 0] = E[:, :-1].float()
            logits = model(X, E_prev)  # [B,T,3]
        else:
            raise ValueError("Unknown method")

        loss = l2d_loss_seq(logits, Y, E, W, alpha=alpha)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running += loss.item()
        n += 1
    return running / max(n, 1)

@torch.no_grad()
def eval_model(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    method: str
) -> Metrics:
    model.eval()
    all_metrics: List[Metrics] = []
    for X, Y, E, W in loader:
        X, Y, E = X.to(device), Y.to(device), E.to(device)

        if method in ("global", "perstep"):
            logits = model(X)
        elif method == "rnn":
            E_prev = torch.zeros_like(E, dtype=torch.float32).unsqueeze(-1)
            E_prev[:, 1:, 0] = E[:, :-1].float()
            logits = model(X, E_prev)
        else:
            raise ValueError("Unknown method")

        all_metrics.append(evaluate(logits, Y, E))

    cov = float(np.mean([m.coverage for m in all_metrics]))
    sys = float(np.mean([m.system_acc for m in all_metrics]))
    exp = float(np.mean([m.expert_acc for m in all_metrics]))
    clf = float(np.mean([m.classifier_acc for m in all_metrics]))
    return Metrics(cov, sys, exp, clf)

# ==============================================================
# Data helpers
# ==============================================================

def make_loaders(
    X: torch.Tensor, Y: torch.Tensor, E: torch.Tensor, W: torch.Tensor,
    batch_size: int = 128, splits=(0.7, 0.15, 0.15), seed: int = 123
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    N = X.shape[0]
    idx = np.arange(N)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    n_train = int(N * splits[0])
    n_val   = int(N * splits[1])
    id_train = idx[:n_train]
    id_val   = idx[n_train:n_train+n_val]
    id_test  = idx[n_train+n_val:]

    ds = TensorDataset(X, Y, E, W)
    ds_train = Subset(ds, id_train.tolist())
    ds_val   = Subset(ds, id_val.tolist())
    ds_test  = Subset(ds, id_test .tolist())

    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=False)
    loader_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, drop_last=False)
    loader_test  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False, drop_last=False)
    return loader_train, loader_val, loader_test

# ==============================================================
# Single-trial run (returns Metrics for 3 methods)
# ==============================================================

def run_single_trial(
    seed: int,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    d: int = 10, N: int = 6000, T: int = 10,
    epochs: int = 5, batch_size: int = 128, alpha: float = 1.0
) -> Dict[str, Metrics]:
    set_seed(seed)
    X, Y, E, W, G = gen_seq_dataset(d=d, N=N, T=T, device=device)
    loader_tr, loader_va, loader_te = make_loaders(X, Y, E, W, batch_size=batch_size)

    # Global-L2D
    model_global = GlobalHead(d).to(device)
    opt_g = torch.optim.Adam(model_global.parameters(), lr=1e-3, weight_decay=1e-5)
    for _ in range(epochs):
        train_one_epoch(model_global, loader_tr, opt_g, device, method="global", alpha=alpha)
    m_global = eval_model(model_global, loader_te, device, method="global")

    # Per-step-L2D
    model_per = PerStepHead(d, T).to(device)
    opt_p = torch.optim.Adam(model_per.parameters(), lr=1e-3, weight_decay=1e-5)
    for _ in range(epochs):
        train_one_epoch(model_per, loader_tr, opt_p, device, method="perstep", alpha=alpha)
    m_per = eval_model(model_per, loader_te, device, method="perstep")

    # RNN-L2D
    model_rnn = RNNL2D(d, hidden=64).to(device)
    opt_r = torch.optim.Adam(model_rnn.parameters(), lr=1e-3, weight_decay=1e-5)
    for _ in range(epochs):
        train_one_epoch(model_rnn, loader_tr, opt_r, device, method="rnn", alpha=alpha)
    m_rnn = eval_model(model_rnn, loader_te, device, method="rnn")

    return {"Global": m_global, "PerStep": m_per, "RNN": m_rnn}

# ==============================================================
# Trials aggregation
# ==============================================================

def agg_mean_std(samples: List[float]) -> Tuple[float, float]:
    arr = np.asarray(samples, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1) if arr.size > 1 else 0.0)

def print_agg_table(results: List[Dict[str, Metrics]]):
    # Collect per method arrays
    methods = ["Global", "PerStep", "RNN"]
    fields = ["coverage", "system_acc", "expert_acc", "classifier_acc"]

    print("\n=== Aggregated test metrics over trials (mean ± ste) ===")
    header = "| Method | coverage | system_acc | expert_acc | classifier_acc |"
    sep    = "|:--|--:|--:|--:|--:|"
    print(header)
    print(sep)
    for m in methods:
        cols = {f: [] for f in fields}
        for r in results:
            met: Metrics = r[m]
            cols["coverage"].append(met.coverage)
            cols["system_acc"].append(met.system_acc)
            cols["expert_acc"].append(met.expert_acc)
            cols["classifier_acc"].append(met.classifier_acc)
        stats = []
        for f in fields:
            mu, sd = agg_mean_std(cols[f])
            stats.append(f"{mu * 100:.2f} ± {sd * 100 / np.sqrt(len(cols[f])):.2f}")
        print(f"| {m} | " + " | ".join(stats) + " |")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def write_rnn_trials_csv(results: List[Dict[str, Metrics]], out_path="synthetic/results/rnn_l2d_trials.csv"):
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial", "coverage", "system_acc", "expert_acc", "classifier_acc"])
        for i, r in enumerate(results):
            m = r["RNN"]
            w.writerow([i+1, f"{m.coverage:.6f}", f"{m.system_acc:.6f}", f"{m.expert_acc:.6f}", f"{m.classifier_acc:.6f}"])
    print(f"\nSaved per-trial RNN-L2D metrics to: {out_path}")


# ==============================================================
# Main
# ==============================================================

def run_trials(
    trials: int = 200,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    d: int = 10, N: int = 6000, T: int = 10,
    epochs: int = 5, batch_size: int = 128, alpha: float = 1.0,
    base_seed: int = 40
):
    all_results: List[Dict[str, Metrics]] = []
    print(f"Device: {device}")
    print(f"Running {trials} trials ...")
    for t in tqdm(range(trials), ncols=100):
        seed = base_seed + t
        res = run_single_trial(seed, device=device, d=d, N=N, T=T,
                               epochs=epochs, batch_size=batch_size, alpha=alpha)
        all_results.append(res)

    print_agg_table(all_results)
    write_rnn_trials_csv(all_results)
    # print_user_row()

if __name__ == "__main__":
    # You can tweak epochs to control runtime; 5 is a good balance for 200 trials.
    run_trials(trials=200, epochs=5, alpha=1.0)