"""
Single-probe Mouse01-Coor1 — Manifold Geometry Layer 1b (Persistent Homology)
==============================================================================
Port of `dp_manifold_layer1b.py` for single-probe LHA & RSP, all 8 sessions.

Pipeline per (session, region):
  1. Load cached matrix (50ms bins, sigma=1, z-scored) -> (T, N)
  2. PCA -> K_PCS components (LHA: 5, RSP: 10 -- mirrors dual-probe defaults;
     can be overridden via --k-lha / --k-rsp once Layer 1a reports)
  3. Maxmin landmark subsample (SUBSAMPLE_N points)
  4. Vietoris-Rips persistent homology (H0, H1, H2) via ripser
  5. Circular-shift shuffle null (N_SHUFFLES draws)
  6. Persistence-stat null comparison + per-session figure

Inputs : data/sp_coor1_dynamics/_cache/session_{N}_{LHA,RSP}.npy
Outputs:
  - data/sp_coor1_dynamics/manifold_layer1b.csv (one row per session-region)
  - data/sp_coor1_dynamics/S{N}_{region}_layer1b.json (per session-region)
  - figures/sp_coor1_dynamics/S{N}_{region}_persistent_homology.png
"""

import argparse
import json
import time as timer
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from ripser import ripser

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

import warnings
warnings.filterwarnings("ignore")


SUBSAMPLE_N = 600         # Reduced from 1000 (Rips is O(N^3) on point count)
N_SHUFFLES = 15           # Reduced from 20 (nulls dominate runtime)
MAX_DIM = 2
DEFAULT_K_PCS = {"LHA": 5, "RSP": 10}

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


def circular_shift_shuffle(X, rng):
    Xs = np.empty_like(X)
    for j in range(X.shape[1]):
        shift = rng.integers(1, X.shape[0])
        Xs[:, j] = np.roll(X[:, j], shift)
    return Xs


def maxmin_subsample(X, n, seed=42):
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    if N <= n:
        return X.copy()
    indices = [int(rng.integers(N))]
    dists = np.full(N, np.inf)
    for _ in range(n - 1):
        last = X[indices[-1]]
        d = np.sum((X - last) ** 2, axis=1)
        dists = np.minimum(dists, d)
        indices.append(int(np.argmax(dists)))
    return X[np.array(indices)]


def compute_persistence(X_sub, max_dim=MAX_DIM):
    return ripser(X_sub, maxdim=max_dim, do_cocycles=False)["dgms"]


def persistence_stats(dgms):
    stats = {}
    for k in range(len(dgms)):
        dgm = dgms[k]
        finite = dgm[np.isfinite(dgm[:, 1])]
        if len(finite) == 0:
            lifetimes = np.array([0.0])
        else:
            lifetimes = finite[:, 1] - finite[:, 0]
        stats[f"H{k}_n_features"] = int(len(finite))
        stats[f"H{k}_max_persistence"] = float(np.max(lifetimes)) if len(lifetimes) > 0 else 0.0
        stats[f"H{k}_total_persistence"] = float(np.sum(lifetimes))
        stats[f"H{k}_mean_persistence"] = float(np.mean(lifetimes)) if len(lifetimes) > 0 else 0.0
        top3 = np.sort(lifetimes)[::-1][:3]
        stats[f"H{k}_top3"] = top3.tolist()
    return stats


def betti_curve(dgms, n_steps=200, eps_max=None):
    if eps_max is None:
        all_deaths = np.concatenate([d[np.isfinite(d[:, 1]), 1]
                                     for d in dgms if len(d[np.isfinite(d[:, 1])]) > 0])
        if len(all_deaths) == 0:
            return np.linspace(0, 1, n_steps), np.zeros((len(dgms), n_steps))
        eps_max = float(np.max(all_deaths) * 1.1)
    eps_range = np.linspace(0, eps_max, n_steps)
    curves = np.zeros((len(dgms), n_steps))
    for k, dgm in enumerate(dgms):
        for birth, death in dgm:
            if not np.isfinite(death):
                continue
            alive = (eps_range >= birth) & (eps_range < death)
            curves[k] += alive
    return eps_range, curves


def run_region(session_num, region, K, rng):
    cache_path = CACHE_DIR / f"session_{session_num}_{region}.npy"
    if not cache_path.exists():
        print(f"[skip] missing cache: {cache_path}")
        return None
    matrix = np.load(cache_path)
    T, N = matrix.shape
    print(f"\n  S{session_num}_{region}: {T} bins x {N} units, K={K}")

    pca = PCA(n_components=K)
    X_pca = pca.fit_transform(matrix)
    var_expl = float(np.sum(pca.explained_variance_ratio_) * 100.0)
    print(f"    PCA K={K}: {var_expl:.1f}% variance explained")

    t0 = timer.time()
    X_sub = maxmin_subsample(X_pca, SUBSAMPLE_N)
    print(f"    Maxmin subsample ({SUBSAMPLE_N}): {timer.time()-t0:.1f}s")

    t0 = timer.time()
    dgms = compute_persistence(X_sub)
    print(f"    Rips H0..H{MAX_DIM}: {timer.time()-t0:.1f}s")
    stats_data = persistence_stats(dgms)
    for k in range(MAX_DIM + 1):
        top3 = stats_data[f"H{k}_top3"]
        top3_str = ", ".join(f"{v:.3f}" for v in top3)
        print(f"      H{k}: n={stats_data[f'H{k}_n_features']}, "
              f"max={stats_data[f'H{k}_max_persistence']:.3f}, "
              f"total={stats_data[f'H{k}_total_persistence']:.3f}, top3=[{top3_str}]")

    eps_data, betti_data = betti_curve(dgms)

    # Null
    print(f"    Null: {N_SHUFFLES} circular-shift shuffles")
    null_max = {f"H{k}": [] for k in range(MAX_DIM + 1)}
    null_total = {f"H{k}": [] for k in range(MAX_DIM + 1)}
    null_betti_all = []
    for i in range(N_SHUFFLES):
        X_shuf = circular_shift_shuffle(matrix, rng)
        X_shuf_pca = pca.transform(X_shuf)
        X_shuf_sub = maxmin_subsample(X_shuf_pca, SUBSAMPLE_N, seed=42 + i)
        dgms_shuf = compute_persistence(X_shuf_sub)
        ss = persistence_stats(dgms_shuf)
        for k in range(MAX_DIM + 1):
            null_max[f"H{k}"].append(ss[f"H{k}_max_persistence"])
            null_total[f"H{k}"].append(ss[f"H{k}_total_persistence"])
        _, bc = betti_curve(dgms_shuf, n_steps=len(eps_data),
                            eps_max=float(eps_data[-1]))
        null_betti_all.append(bc)

    sig_results = {}
    for k in range(MAX_DIM + 1):
        nmax = np.array(null_max[f"H{k}"])
        ntot = np.array(null_total[f"H{k}"])
        dmax = stats_data[f"H{k}_max_persistence"]
        dtot = stats_data[f"H{k}_total_persistence"]
        p_max = float(np.mean(nmax >= dmax))
        p_tot = float(np.mean(ntot >= dtot))
        print(f"      H{k} null: max_pers data={dmax:.3f} vs null={nmax.mean():.3f}+/-{nmax.std():.3f} (p={p_max:.3f})  "
              f"total data={dtot:.3f} vs null={ntot.mean():.3f}+/-{ntot.std():.3f} (p={p_tot:.3f})")
        sig_results[f"H{k}"] = {
            "max_pers_data": dmax, "max_pers_null_mean": float(nmax.mean()),
            "max_pers_null_std": float(nmax.std()), "max_pers_p": p_max,
            "total_pers_data": dtot, "total_pers_null_mean": float(ntot.mean()),
            "total_pers_null_std": float(ntot.std()), "total_pers_p": p_tot,
        }

    null_betti = np.array(null_betti_all)

    _make_figure(session_num, region, K, var_expl, dgms, eps_data, betti_data, null_betti)

    summary = {
        "session": session_num,
        "state": STATE_OF[session_num],
        "region": region,
        "n_units": int(N),
        "K_pcs": int(K),
        "var_expl_pct": var_expl,
    }
    for k in range(MAX_DIM + 1):
        summary[f"H{k}_n"] = stats_data[f"H{k}_n_features"]
        summary[f"H{k}_max_pers"] = stats_data[f"H{k}_max_persistence"]
        summary[f"H{k}_total_pers"] = stats_data[f"H{k}_total_persistence"]
        summary[f"H{k}_max_pers_null_mean"] = sig_results[f"H{k}"]["max_pers_null_mean"]
        summary[f"H{k}_max_pers_null_std"] = sig_results[f"H{k}"]["max_pers_null_std"]
        summary[f"H{k}_max_pers_p"] = sig_results[f"H{k}"]["max_pers_p"]
        summary[f"H{k}_total_pers_p"] = sig_results[f"H{k}"]["total_pers_p"]

    json_path = OUT_DIR / f"S{session_num}_{region}_layer1b.json"
    with open(json_path, "w") as f:
        json.dump({"stats": stats_data, "null": sig_results, "K": K,
                   "var_expl_pct": var_expl, "N_landmarks": SUBSAMPLE_N,
                   "N_shuffles": N_SHUFFLES, "n_units": int(N)}, f, indent=2)
    print(f"    Saved {json_path}")
    return summary


def _make_figure(session_num, region, K, var_expl, dgms, eps_data, betti_data, null_betti):
    fig = plt.figure(figsize=(20, 11))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    colors = ["tab:blue", "tab:orange", "tab:green"]
    labels = ["H0", "H1", "H2"]

    ax = fig.add_subplot(gs[0, 0])
    max_val = 0.0
    for k, dgm in enumerate(dgms):
        finite = dgm[np.isfinite(dgm[:, 1])]
        if len(finite) > 0:
            ax.scatter(finite[:, 0], finite[:, 1], s=15, alpha=0.6,
                       c=colors[k], label=f"{labels[k]} ({len(finite)})", zorder=3)
            max_val = max(max_val, float(finite.max()))
    ax.plot([0, max_val * 1.1], [0, max_val * 1.1], "k--", alpha=0.3, lw=1)
    ax.set_xlabel("Birth"); ax.set_ylabel("Death")
    ax.set_title("Persistence Diagram"); ax.legend(fontsize=9)

    ax = fig.add_subplot(gs[0, 1])
    y = 0
    for k, dgm in enumerate(dgms):
        finite = dgm[np.isfinite(dgm[:, 1])]
        if len(finite) == 0:
            continue
        lifetimes = finite[:, 1] - finite[:, 0]
        order = np.argsort(lifetimes)[::-1][:15]
        for idx in order:
            ax.barh(y, lifetimes[idx], left=finite[idx, 0],
                    height=0.8, color=colors[k], alpha=0.7)
            y += 1
        y += 1
    ax.set_xlabel("Filtration scale"); ax.set_ylabel("Top-15 features per Hk")
    ax.set_title("Barcode")
    ax.legend(handles=[Patch(color=colors[k], label=labels[k]) for k in range(len(dgms))], fontsize=9)

    ax = fig.add_subplot(gs[0, 2])
    for k, dgm in enumerate(dgms):
        finite = dgm[np.isfinite(dgm[:, 1])]
        if len(finite) > 0:
            lifetimes = finite[:, 1] - finite[:, 0]
            ax.hist(lifetimes, bins=30, alpha=0.5, color=colors[k], label=labels[k])
    ax.set_xlabel("Persistence"); ax.set_ylabel("Count")
    ax.set_title("Lifetime distribution"); ax.legend(fontsize=9)

    for k in range(MAX_DIM + 1):
        ax = fig.add_subplot(gs[1, k])
        if null_betti.shape[2] == len(eps_data):
            null_k = null_betti[:, k, :]
            ax.fill_between(eps_data, np.percentile(null_k, 2.5, axis=0),
                            np.percentile(null_k, 97.5, axis=0),
                            alpha=0.2, color="gray", label="Null 95% CI")
            ax.plot(eps_data, null_k.mean(axis=0), color="gray", alpha=0.5, lw=1, label="Null mean")
        ax.plot(eps_data, betti_data[k], color=colors[k], lw=2, label=f"{labels[k]} data")
        ax.set_xlabel("epsilon"); ax.set_ylabel(f"Betti-{k}")
        ax.set_title(f"Betti-{k} curve"); ax.legend(fontsize=8)

    fig.suptitle(f"Persistent Homology — S{session_num} {region} ({STATE_OF[session_num]})  "
                 f"K={K} PCs ({var_expl:.1f}% var), {SUBSAMPLE_N} landmarks, {N_SHUFFLES} null",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig_path = FIG_DIR / f"S{session_num}_{region}_persistent_homology.png"
    plt.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"    Saved {fig_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k-lha", type=int, default=DEFAULT_K_PCS["LHA"])
    parser.add_argument("--k-rsp", type=int, default=DEFAULT_K_PCS["RSP"])
    parser.add_argument("--sessions", type=str, default="all",
                        help="comma-separated session numbers, or 'all'")
    args = parser.parse_args()

    if args.sessions == "all":
        sessions = SESSIONS
    else:
        sessions = [int(s) for s in args.sessions.split(",")]

    K_PCS = {"LHA": args.k_lha, "RSP": args.k_rsp}
    print(f"K_PCS = {K_PCS}")
    rng = np.random.default_rng(20260428)

    rows = []
    for s in sessions:
        for r in REGIONS:
            t0 = timer.time()
            res = run_region(s, r, K_PCS[r], rng)
            if res is not None:
                rows.append(res)
            print(f"    [S{s}_{r}] elapsed {(timer.time()-t0)/60:.1f} min")

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "manifold_layer1b.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")
    print(df[["session", "state", "region", "n_units", "K_pcs",
              "H1_max_pers", "H1_max_pers_null_mean", "H1_max_pers_p",
              "H2_max_pers", "H2_max_pers_p"]].to_string(index=False))


if __name__ == "__main__":
    main()
