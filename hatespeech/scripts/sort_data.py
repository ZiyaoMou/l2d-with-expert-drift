import os
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

GENERAL_BASE = "/home/zmou1/scratchenalisn1/ziyao/l2d-cog/hatespeech/model/general_models_new/alpha1.0_seq_len10"
LSTM_BASE    = "/home/zmou1/scratchenalisn1/ziyao/l2d-cog/hatespeech/model/lstm_models_new/alpha1.0_seq_len10"
PERSTEP_BASE = "/home/zmou1/scratchenalisn1/ziyao/l2d-cog/hatespeech/model/perstep_models/alpha1.0_seq_len10"

# Exact per-seed file patterns
GENERAL_PATTERN = "seed{seed}/metrics_test.csv"
LSTM_PATTERN    = "seed{seed}/metrics_test_fullseq_per_t.csv"      # LSTM lacks sample_count in your case
PERSTEP_PATTERN = "seed{seed}/epoch5_metrics_test.csv"

# Timestep column detection
TIMESTEP_CANDIDATES = ["timestep", "step", "time", "t", "idx"]

# Output
OUT_DIR = "./figs"

def find_timestep_col(df):
    for c in TIMESTEP_CANDIDATES:
        if c in df.columns:
            return c
    return None

def find_all_seeds():
    seeds = set()
    for base in [GENERAL_BASE, LSTM_BASE, PERSTEP_BASE]:
        if not os.path.isdir(base):
            print(f"[MISS] Base not found: {base}")
            continue
        for p in glob.glob(os.path.join(base, "seed*")):
            name = os.path.basename(p)
            if name.startswith("seed"):
                try:
                    seeds.add(int(name[4:]))
                except ValueError:
                    pass
    seeds = sorted(seeds)
    if not seeds:
        print("[ERROR] No seed* directories found.")
    else:
        print(f"[INFO] Found seeds: {seeds}")
    return seeds

def safe_read_csv(path):
    if os.path.isfile(path):
        try:
            return pd.read_csv(path)
        except Exception as e:
            print(f"[WARN] Failed to read CSV: {path} -> {e}")
    else:
        print(f"[MISS] File not found: {path}")
    return None

def _accumulate_with_counts(df):
    """Exact accumulation using sample_count (percent -> counts -> cumulative -> percent)."""
    df = df.copy()
    df["accepted_count"]    = df["coverage"]   * df["sample_count"] / 100.0
    df["correct_sys_count"] = df["system_acc"] * df["sample_count"] / 100.0
    df["cum_samples"]       = df["sample_count"].cumsum()
    df["cum_accepted"]      = df["accepted_count"].cumsum()
    df["cum_correct_sys"]   = df["correct_sys_count"].cumsum()
    eps = 1e-9
    df["accum_coverage"]    = 100.0 * df["cum_accepted"]    / (df["cum_samples"] + eps)
    df["accum_system_acc"]  = 100.0 * df["cum_correct_sys"] / (df["cum_samples"] + eps)
    return df[["timestep", "accum_system_acc", "accum_coverage"]]

def _accumulate_running_mean(df, model_name, seed):
    """Approximation when sample_count is missing: running mean over timesteps."""
    print(f"[WARN] {model_name}-seed{seed}: 'sample_count' missing -> using running-average approximation.")
    df = df.copy()
    df["accum_coverage"]   = df["coverage"].expanding().mean()
    df["accum_system_acc"] = df["system_acc"].expanding().mean()
    return df[["timestep", "accum_system_acc", "accum_coverage"]]

def load_one_seed_table(model_name, base, pattern, seed):
    """Load a single seed CSV and return per-seed cumulative DataFrame (accum_*)."""
    path = os.path.join(base, pattern.format(seed=seed))
    df = safe_read_csv(path)
    if df is None:
        return None

    tcol = find_timestep_col(df)
    if tcol is None:
        print(f"[WARN] {model_name}-seed{seed}: no timestep-like column; skipped.")
        return None

    # Minimal required columns for both paths
    needed = {"system_acc", "coverage"}
    if not needed.issubset(df.columns):
        print(f"[WARN] {model_name}-seed{seed}: missing columns {sorted(list(needed - set(df.columns)))}; skipped.")
        return None

    df = df[[tcol, "system_acc", "coverage"] + ([ "sample_count"] if "sample_count" in df.columns else [])].copy()
    df = df.rename(columns={tcol: "timestep"})
    df = df.dropna(subset=["timestep", "system_acc", "coverage"])
    df["timestep"] = pd.to_numeric(df["timestep"], errors="coerce")
    df = df.dropna(subset=["timestep"]).sort_values("timestep")

    if "sample_count" in df.columns:
        df = df.dropna(subset=["sample_count"])
        out = _accumulate_with_counts(df)
    else:
        out = _accumulate_running_mean(df, model_name, seed)

    out["model"] = model_name
    out["seed"]  = seed
    return out

def load_all_cumulative():
    """Return a long DataFrame with: model, seed, timestep, accum_system_acc, accum_coverage."""
    seeds = find_all_seeds()
    chunks = []
    for seed in seeds:
        for model_name, base, pattern in [
            ("general", GENERAL_BASE, GENERAL_PATTERN),
            ("lstm",    LSTM_BASE,    LSTM_PATTERN),
            ("perstep", PERSTEP_BASE, PERSTEP_PATTERN),
        ]:
            if not os.path.isdir(base):
                print(f"[MISS] Model base not found: {model_name} -> {base}")
                continue
            df = load_one_seed_table(model_name, base, pattern, seed)
            if df is not None:
                chunks.append(df)

    if not chunks:
        raise RuntimeError("No CSVs could be loaded. Check paths and file names.")
    all_df = pd.concat(chunks, ignore_index=True).drop_duplicates()
    all_df["timestep"] = pd.to_numeric(all_df["timestep"], errors="coerce")
    all_df = all_df.dropna(subset=["timestep"]).sort_values(["model", "seed", "timestep"])
    print(all_df)
    return all_df

def agg_mean_ste(df, value_col):
    """Compute mean ± STE across seeds, grouped by (model, timestep)."""
    grp = (
        df.groupby(["model", "timestep"], as_index=False)[value_col]
          .agg(mean="mean", std="std", n="count")
    )
    grp["ste"] = grp.apply(lambda r: (r["std"] / math.sqrt(r["n"])) if r["n"] > 0 else np.nan, axis=1)
    return grp.sort_values(["model", "timestep"])

def plot_combined(df, value_col, y_label, title_suffix, outfile):
    """Overlay the three models with mean line and ±STE band."""
    plt.figure(figsize=(10, 6))
    for model in sorted(df["model"].unique()):
        sub = df[df["model"] == model]
        agg = agg_mean_ste(sub, value_col)
        if agg.empty:
            print(f"[WARN] No data to plot for model={model}, metric={value_col}")
            continue
        x = agg["timestep"].values
        y = agg["mean"].values
        ste = agg["ste"].values
        plt.plot(x, y, linewidth=2, label=model)
        plt.fill_between(x, y - ste, y + ste, alpha=0.2)

    plt.xlabel("timestep")
    plt.ylabel(y_label)
    plt.title(f"{value_col} vs timestep (accumulated; mean ± STE across seeds) — {title_suffix}")
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend(title="model")
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, outfile)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"[SAVE] {out_path}")
    plt.show()

def load_one_seed_table_raw(model_name, base, pattern, seed):
    """
    Read single seed CSV and return non-cumulative raw table:
    Columns: timestep, system_acc, coverage, model, seed
    """
    path = os.path.join(base, pattern.format(seed=seed))
    df = safe_read_csv(path)
    if df is None:
        return None

    tcol = find_timestep_col(df)
    if tcol is None:
        print(f"[WARN] {model_name}-seed{seed}: no timestep-like column; skipped.")
        return None

    # Only need raw columns
    needed = {"system_acc", "coverage"}
    if not needed.issubset(df.columns):
        print(f"[WARN] {model_name}-seed{seed}: missing columns {sorted(list(needed - set(df.columns)))}; skipped.")
        return None

    df = df[[tcol, "system_acc", "coverage"]].copy()
    df = df.rename(columns={tcol: "timestep"})
    df = df.dropna(subset=["timestep", "system_acc", "coverage"])
    df["timestep"] = pd.to_numeric(df["timestep"], errors="coerce")
    df = df.dropna(subset=["timestep"]).sort_values("timestep")

    df["model"] = model_name
    df["seed"]  = seed
    return df

def load_all_raw():
    """
    Aggregate all model/seed non-cumulative data:
    Columns: model, seed, timestep, system_acc, coverage
    """
    seeds = find_all_seeds()
    chunks = []
    for seed in seeds:
        for model_name, base, pattern in [
            ("general", GENERAL_BASE, GENERAL_PATTERN),
            ("lstm",    LSTM_BASE,    LSTM_PATTERN),
            ("perstep", PERSTEP_BASE, PERSTEP_PATTERN),
        ]:
            if not os.path.isdir(base):
                print(f"[MISS] Model base not found: {model_name} -> {base}")
                continue
            df = load_one_seed_table_raw(model_name, base, pattern, seed)
            if df is not None:
                chunks.append(df)

    if not chunks:
        raise RuntimeError("No CSVs could be loaded. Check paths and file names.")
    all_df = pd.concat(chunks, ignore_index=True).drop_duplicates()
    all_df["timestep"] = pd.to_numeric(all_df["timestep"], errors="coerce")
    all_df = all_df.dropna(subset=["timestep"]).sort_values(["model", "seed", "timestep"])
    print(all_df)
    return all_df

def main():
    # 1) Non-cumulative: output by t as-is + sorted
    print("[INFO] Loading all RAW (non-cumulative) data...")
    raw_df = load_all_raw()
    os.makedirs("results", exist_ok=True)
    raw_out = "results/combined_raw_results.csv"
    raw_df.to_csv(raw_out, index=False)
    print(f"[INFO] Saved RAW (non-cumulative) to {raw_out}")

    # 2) If you want to keep previous cumulative export, you can keep it:
    print("[INFO] Loading all cumulative data...")
    cum_df = load_all_cumulative()
    cum_out = "results/combined_cumulative_results_new.csv"
    cum_df.to_csv(cum_out, index=False)
    print(f"[INFO] Saved cumulative to {cum_out}")

def main():
    print("[INFO] Loading all cumulative data...")
    df = load_all_cumulative()
    raw_df = load_all_raw()
    print(f"[INFO] Loaded cumulative rows: {len(df)}")
    df.to_csv('results/combined_cumulative_results_new.csv', index=False)
    raw_df.to_csv('results/combined_raw_results_new.csv', index=False)
    print("[INFO] Saved to results/combined_cumulative_results_new.csv")

if __name__ == "__main__":
    main()

