"""
Single-probe Mouse01-Coor1 — Manifold Geometry Layer 1a (Dimensionality)
========================================================================
Port of `dp_manifold_geometry.py` (Layer 1a) for single-probe LHA & RSP.

Inputs : cached preprocessed matrices in
         data/sp_coor1_dynamics/_cache/session_{N}_{LHA,RSP}.npy
         (50ms bins, sigma=1 Gaussian smooth, per-unit z-scored)
Outputs:
  - data/sp_coor1_dynamics/manifold_layer1a.csv
  - data/sp_coor1_dynamics/manifold_layer1a.json
  - figures/sp_coor1_dynamics/manifold_layer1a_dimensionality.png
"""

import json
import time as timer
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import pdist
from scipy.stats import linregress
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


# ---- Constants (mirrors dp_manifold_geometry.py) ----
BIN_MS = 50.0
BLOCK_SIZE = 200            # ~10 s at 50 ms bins
N_BOOT = 100                # PR + Two-NN
N_BOOT_CORRDIM = 50
N_BOOT_ISOMAP = 20
CORRDIM_SUBSAMPLE = 3000
ISOMAP_SUBSAMPLE = 3000
ISOMAP_K_VALUES = [10, 15, 20, 30]
ISOMAP_K_PRIMARY = 15

REGIONS = ("LHA", "RSP")
SESSIONS = list(range(1, 9))
STATE_OF = {1: "fed", 2: "fed", 3: "fed", 4: "fed",
            5: "fasted", 6: "fasted", 7: "fasted", 8: "fasted"}

REPO = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO / "data" / "sp_coor1_dynamics" / "_cache"
OUT_DIR = REPO / "data" / "sp_coor1_dynamics"
FIG_DIR = REPO / "figures" / "sp_coor1_dynamics"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Dimensionality estimators
# ---------------------------------------------------------------------------

def participation_ratio(X):
    pca = PCA(n_components=min(X.shape))
    pca.fit(X)
    evals = pca.explained_variance_
    pr = (evals.sum()) ** 2 / (evals ** 2).sum()
    return float(pr), evals


def two_nn_dimension(X):
    nn = NearestNeighbors(n_neighbors=3, algorithm="auto")
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    r1 = distances[:, 1]
    r2 = distances[:, 2]
    valid = r1 > 0
    mu = r2[valid] / r1[valid]
    mu = np.sort(mu)
    N = len(mu)
    if N < 20:
        return np.nan
    F = np.arange(1, N + 1) / N
    mask = (F > 0.01) & (F < 0.90)
    if mask.sum() < 10:
        return np.nan
    log_mu = np.log(mu[mask])
    log_surv = np.log(1.0 - F[mask])
    slope, _, _, _, _ = linregress(log_mu, log_surv)
    return float(-slope)


def correlation_dimension(X, n_sub=CORRDIM_SUBSAMPLE):
    if len(X) > n_sub:
        idx = np.random.choice(len(X), n_sub, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X
    dists = pdist(X_sub)
    n_pairs = len(dists)
    r_lo = np.percentile(dists, 5)
    r_hi = np.percentile(dists, 95)
    if r_lo <= 0:
        positive = dists[dists > 0]
        if len(positive) == 0:
            return np.nan
        r_lo = positive.min()
    r_vals = np.logspace(np.log10(r_lo), np.log10(r_hi), 40)
    C_r = np.array([np.sum(dists < r) / n_pairs for r in r_vals])
    valid = (C_r > 0.02) & (C_r < 0.80)
    if valid.sum() < 5:
        return np.nan
    log_r = np.log10(r_vals[valid])
    log_C = np.log10(C_r[valid])
    slope, _, _, _, _ = linregress(log_r, log_C)
    return float(slope)


def isomap_dimension(X, k=ISOMAP_K_PRIMARY, n_sub=ISOMAP_SUBSAMPLE):
    if len(X) > n_sub:
        idx = np.random.choice(len(X), n_sub, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
    nn.fit(X_sub)
    knn_graph = nn.kneighbors_graph(mode="distance")
    knn_graph = knn_graph.maximum(knn_graph.T)
    geo_dist = shortest_path(knn_graph, method="D", directed=False)
    if np.any(np.isinf(geo_dist)):
        connected = np.all(np.isfinite(geo_dist), axis=1)
        if connected.sum() < 100:
            return np.nan
        geo_dist = geo_dist[np.ix_(connected, connected)]
    n = len(geo_dist)
    D2 = geo_dist ** 2
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ D2 @ H
    eigenvalues = np.linalg.eigvalsh(B)
    eigenvalues = np.sort(eigenvalues)[::-1]
    pos_evals = eigenvalues[eigenvalues > 0]
    if len(pos_evals) == 0:
        return np.nan
    threshold = 0.05 * pos_evals[0]
    return int(np.sum(pos_evals > threshold))


# ---------------------------------------------------------------------------
# Block bootstrap
# ---------------------------------------------------------------------------

def block_bootstrap_resample(X, block_size=BLOCK_SIZE):
    T, _N = X.shape
    n_blocks = int(np.ceil(T / block_size))
    block_starts = np.arange(0, T, block_size)
    chosen = np.random.choice(len(block_starts), n_blocks, replace=True)
    rows = []
    for b in chosen:
        start = block_starts[b]
        end = min(start + block_size, T)
        rows.append(X[start:end])
    return np.concatenate(rows, axis=0)[:T]


def _ci(values, lo=2.5, hi=97.5):
    arr = np.array([v for v in values if not np.isnan(v)])
    if len(arr) == 0:
        return [np.nan, np.nan]
    return [float(np.percentile(arr, lo)), float(np.percentile(arr, hi))]


def run_dimensionality_suite(X, label):
    T, N = X.shape
    print(f"  {label}: {T} bins x {N} units")

    # 1) PR
    t0 = timer.time()
    pr, evals = participation_ratio(X)
    boot_pr = []
    for _ in range(N_BOOT):
        Xb = block_bootstrap_resample(X)
        pr_b, _ = participation_ratio(Xb)
        boot_pr.append(pr_b)
    pr_ci = _ci(boot_pr)
    print(f"    PR     = {pr:6.2f}  CI [{pr_ci[0]:.2f}, {pr_ci[1]:.2f}]  ({timer.time()-t0:.1f}s)")

    # 2) Two-NN
    t0 = timer.time()
    tnn = two_nn_dimension(X)
    boot_tnn = []
    for _ in range(N_BOOT):
        Xb = block_bootstrap_resample(X)
        Xb = Xb + np.random.randn(*Xb.shape) * 1e-6
        boot_tnn.append(two_nn_dimension(Xb))
    tnn_ci = _ci(boot_tnn)
    print(f"    TwoNN  = {tnn:6.2f}  CI [{tnn_ci[0]:.2f}, {tnn_ci[1]:.2f}]  ({timer.time()-t0:.1f}s)")

    # 3) CorrDim
    t0 = timer.time()
    cd = correlation_dimension(X)
    boot_cd = []
    for _ in range(N_BOOT_CORRDIM):
        Xb = block_bootstrap_resample(X)
        boot_cd.append(correlation_dimension(Xb))
    cd_ci = _ci(boot_cd)
    cd_str = f"{cd:6.2f}" if not np.isnan(cd) else "  nan"
    print(f"    CorrDim= {cd_str}  CI [{cd_ci[0]:.2f}, {cd_ci[1]:.2f}]  ({timer.time()-t0:.1f}s)")

    # 4) Isomap
    t0 = timer.time()
    iso_sensitivity = {k: isomap_dimension(X, k=k) for k in ISOMAP_K_VALUES}
    boot_iso = []
    for _ in range(N_BOOT_ISOMAP):
        Xb = block_bootstrap_resample(X)
        d_b = isomap_dimension(Xb, k=ISOMAP_K_PRIMARY)
        if not (isinstance(d_b, float) and np.isnan(d_b)):
            boot_iso.append(d_b)
    iso_ci = _ci(boot_iso) if boot_iso else [np.nan, np.nan]
    iso_val = iso_sensitivity[ISOMAP_K_PRIMARY]
    iso_str = f"{iso_val}" if iso_val is not None and not (isinstance(iso_val, float) and np.isnan(iso_val)) else "nan"
    print(f"    Isomap(k={ISOMAP_K_PRIMARY}) = {iso_str}  CI [{iso_ci[0]:.0f}, {iso_ci[1]:.0f}]  ({timer.time()-t0:.1f}s)")
    print(f"      k-sens: " + ", ".join(f"k={k}:{v}" for k, v in iso_sensitivity.items()))

    return {
        "n_bins": int(T),
        "n_units": int(N),
        "PR": pr,
        "PR_ci": pr_ci,
        "TwoNN": tnn,
        "TwoNN_ci": tnn_ci,
        "CorrDim": cd,
        "CorrDim_ci": cd_ci,
        "Isomap": iso_val,
        "Isomap_ci": iso_ci,
        "Isomap_k_sensitivity": iso_sensitivity,
        "eigenvalues_top50": evals.tolist()[:50],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    np.random.seed(20260428)

    rows = []
    full = {}
    for session in SESSIONS:
        for region in REGIONS:
            cache_path = CACHE_DIR / f"session_{session}_{region}.npy"
            if not cache_path.exists():
                print(f"[skip] missing cache: {cache_path}")
                continue
            X = np.load(cache_path)
            print(f"\n=== S{session} {region} ({STATE_OF[session]}) ===")
            res = run_dimensionality_suite(X, label=f"S{session}_{region}")
            full[f"S{session}_{region}"] = res
            rows.append({
                "session": session,
                "state": STATE_OF[session],
                "region": region,
                "n_bins": res["n_bins"],
                "n_units": res["n_units"],
                "PR": res["PR"],
                "PR_lo": res["PR_ci"][0],
                "PR_hi": res["PR_ci"][1],
                "TwoNN": res["TwoNN"],
                "TwoNN_lo": res["TwoNN_ci"][0],
                "TwoNN_hi": res["TwoNN_ci"][1],
                "CorrDim": res["CorrDim"],
                "CorrDim_lo": res["CorrDim_ci"][0],
                "CorrDim_hi": res["CorrDim_ci"][1],
                "Isomap": res["Isomap"],
                "Isomap_lo": res["Isomap_ci"][0],
                "Isomap_hi": res["Isomap_ci"][1],
                "Isomap_k10": res["Isomap_k_sensitivity"].get(10),
                "Isomap_k15": res["Isomap_k_sensitivity"].get(15),
                "Isomap_k20": res["Isomap_k_sensitivity"].get(20),
                "Isomap_k30": res["Isomap_k_sensitivity"].get(30),
            })

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "manifold_layer1a.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    json_path = OUT_DIR / "manifold_layer1a.json"
    with open(json_path, "w") as f:
        json.dump(full, f, indent=2, default=lambda o: None if isinstance(o, float) and np.isnan(o) else o)
    print(f"Saved {json_path}")

    make_summary_figure(df)
    return df


def make_summary_figure(df):
    metrics = [("PR", "Participation Ratio"),
               ("TwoNN", "Two-NN Dim"),
               ("CorrDim", "Correlation Dim"),
               ("Isomap", "Isomap Dim (k=15)")]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    color = {"fed": "#2c7fb8", "fasted": "#d95f0e"}
    marker = {"LHA": "o", "RSP": "s"}
    for ax, (m, title) in zip(axes, metrics):
        for region in REGIONS:
            sub = df[df["region"] == region]
            for _, row in sub.iterrows():
                lo = row[f"{m}_lo"]
                hi = row[f"{m}_hi"]
                v = row[m]
                if pd.isna(v):
                    continue
                # Guard: TwoNN bootstrap CIs collapse near 0 due to block-resample
                # duplicates; clip yerr to non-negative.
                if not (pd.isna(lo) or pd.isna(hi)):
                    err_lo = max(v - lo, 0.0)
                    err_hi = max(hi - v, 0.0)
                    yerr = [[err_lo], [err_hi]]
                else:
                    yerr = None
                ax.errorbar(row["session"], v, yerr=yerr,
                            fmt=marker[region], color=color[row["state"]],
                            ecolor="gray", capsize=3, ms=8, mfc="white" if region == "RSP" else color[row["state"]],
                            mew=1.5)
        ax.set_xlabel("Session")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xticks(SESSIONS)
        ax.axvline(4.5, ls="--", color="k", alpha=0.3)
        ax.grid(alpha=0.3)
    # custom legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="o", color=color["fed"], lw=0, ms=8, label="LHA fed"),
        Line2D([], [], marker="o", color=color["fasted"], lw=0, ms=8, label="LHA fasted"),
        Line2D([], [], marker="s", color=color["fed"], lw=0, ms=8, mfc="white", mew=1.5, label="RSP fed"),
        Line2D([], [], marker="s", color=color["fasted"], lw=0, ms=8, mfc="white", mew=1.5, label="RSP fasted"),
    ]
    axes[0].legend(handles=handles, fontsize=8, loc="best")
    fig.suptitle("Single-probe Mouse01-Coor1 — Manifold Layer 1a (Intrinsic Dimensionality)", fontweight="bold")
    plt.tight_layout()
    fig_path = FIG_DIR / "manifold_layer1a_dimensionality.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {fig_path}")

    # State-level summary table figure
    summary = df.groupby(["region", "state"])[["PR", "TwoNN", "CorrDim", "Isomap"]].mean().round(2)
    print("\nState-level mean (across sessions):")
    print(summary)


if __name__ == "__main__":
    main()
