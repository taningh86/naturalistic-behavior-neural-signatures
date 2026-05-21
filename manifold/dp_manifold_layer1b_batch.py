"""
Dual-probe Layer 1b — Batch driver for fasted + HFD sessions.

Reuses functions from `dp_manifold_layer1b.py` but with matched parameters
(SUBSAMPLE_N=600, N_SHUFFLES=15) for direct comparability with the single-probe
pipeline (`analysis/sp_coor1_dynamics/sp_manifold_layer1b.py`).

Sessions covered:
  - Fed reference (re-run for matched params): S3, S4
  - Fasted: S11-S16  (6 sessions)
  - HFD:    S19-S24  (6 sessions)

Outputs:
  - data/manifold/S{N}_{ACA,LHA}_layer1b_batch.json
  - data/manifold/manifold_layer1b_batch.csv
  - figures/manifold/S{N}_{ACA,LHA}_persistent_homology_batch.png
"""

import argparse
import json
import time as timer
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Reuse helpers from dp_manifold_layer1b.py
import dp_manifold_layer1b as dpL1b
from dp_manifold_layer1b import (
    load_and_preprocess,
    circular_shift_shuffle,
    maxmin_subsample,
    compute_persistence,
    persistence_stats,
    betti_curve,
)

# Override defaults to match the single-probe run
dpL1b.SUBSAMPLE_N = 600
dpL1b.N_SHUFFLES = 15
SUBSAMPLE_N = 600
N_SHUFFLES = 15
MAX_DIM = 2
K_PCS = {"ACA": 10, "LHA": 5}

OUT_DIR = Path("data/manifold")
FIG_DIR = Path("figures/manifold")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

STATE_OF = {
    1: "fed", 3: "fed", 4: "fed", 5: "fed", 6: "fed",
    7: "fed", 8: "fed", 9: "fed", 10: "fed",
    11: "fasted", 12: "fasted", 13: "fasted", 14: "fasted",
    15: "fasted", 16: "fasted",
    19: "HFD", 20: "HFD", 21: "HFD", 22: "HFD", 23: "HFD", 24: "HFD",
}
PHASE_OF = {
    1: "exp", 3: "exp", 4: "for", 5: "exp", 6: "for", 7: "exp", 8: "for",
    9: "exp", 10: "for",
    11: "exp", 12: "for", 13: "exp", 14: "for", 15: "exp", 16: "for",
    19: "exp", 20: "for", 21: "exp", 22: "for", 23: "exp", 24: "for",
}


def run_region_batch(session_num, region, K, rng):
    matrix, n_units = load_and_preprocess(session_num, region)
    T = matrix.shape[0]
    print(f"\n  S{session_num}_{region} ({STATE_OF[session_num]}/{PHASE_OF[session_num]}): "
          f"{T} bins x {n_units} units, K={K}")

    pca = PCA(n_components=K)
    X_pca = pca.fit_transform(matrix)
    var_expl = float(np.sum(pca.explained_variance_ratio_) * 100.0)
    print(f"    PCA K={K}: {var_expl:.1f}% var")

    t0 = timer.time()
    X_sub = maxmin_subsample(X_pca, SUBSAMPLE_N)
    print(f"    Maxmin subsample ({SUBSAMPLE_N}): {timer.time()-t0:.1f}s")

    t0 = timer.time()
    dgms = compute_persistence(X_sub)
    print(f"    Rips H0..H{MAX_DIM}: {timer.time()-t0:.1f}s")
    stats_data = persistence_stats(dgms)
    for k in range(MAX_DIM + 1):
        print(f"      H{k}: n={stats_data[f'H{k}_n_features']}, "
              f"max={stats_data[f'H{k}_max_persistence']:.3f}, "
              f"total={stats_data[f'H{k}_total_persistence']:.3f}")

    eps_data, _ = betti_curve(dgms)
    print(f"    Null: {N_SHUFFLES} shuffles")
    null_max = {f"H{k}": [] for k in range(MAX_DIM + 1)}
    null_total = {f"H{k}": [] for k in range(MAX_DIM + 1)}
    for i in range(N_SHUFFLES):
        X_shuf = circular_shift_shuffle(matrix)
        X_shuf_pca = pca.transform(X_shuf)
        X_shuf_sub = maxmin_subsample(X_shuf_pca, SUBSAMPLE_N)
        dgms_shuf = compute_persistence(X_shuf_sub)
        ss = persistence_stats(dgms_shuf)
        for k in range(MAX_DIM + 1):
            null_max[f"H{k}"].append(ss[f"H{k}_max_persistence"])
            null_total[f"H{k}"].append(ss[f"H{k}_total_persistence"])

    sig = {}
    for k in range(MAX_DIM + 1):
        nmax = np.array(null_max[f"H{k}"])
        ntot = np.array(null_total[f"H{k}"])
        dmax = stats_data[f"H{k}_max_persistence"]
        dtot = stats_data[f"H{k}_total_persistence"]
        p_max = float(np.mean(nmax >= dmax))
        p_tot = float(np.mean(ntot >= dtot))
        print(f"      H{k} null: max data={dmax:.3f} vs null={nmax.mean():.3f} (p={p_max:.3f})  "
              f"total data={dtot:.3f} vs null={ntot.mean():.3f} (p={p_tot:.3f})")
        sig[f"H{k}"] = {
            "max_pers_data": float(dmax),
            "max_pers_null_mean": float(nmax.mean()),
            "max_pers_null_std": float(nmax.std()),
            "max_pers_p": p_max,
            "total_pers_data": float(dtot),
            "total_pers_null_mean": float(ntot.mean()),
            "total_pers_null_std": float(ntot.std()),
            "total_pers_p": p_tot,
        }

    summary = {
        "session": int(session_num),
        "state": STATE_OF[session_num],
        "phase": PHASE_OF[session_num],
        "region": region,
        "n_units": int(n_units),
        "K_pcs": int(K),
        "var_expl_pct": var_expl,
    }
    for k in range(MAX_DIM + 1):
        summary[f"H{k}_n"] = int(stats_data[f"H{k}_n_features"])
        summary[f"H{k}_max_pers"] = float(stats_data[f"H{k}_max_persistence"])
        summary[f"H{k}_total_pers"] = float(stats_data[f"H{k}_total_persistence"])
        summary[f"H{k}_max_pers_null_mean"] = sig[f"H{k}"]["max_pers_null_mean"]
        summary[f"H{k}_max_pers_p"] = sig[f"H{k}"]["max_pers_p"]
        summary[f"H{k}_total_pers_null_mean"] = sig[f"H{k}"]["total_pers_null_mean"]
        summary[f"H{k}_total_pers_p"] = sig[f"H{k}"]["total_pers_p"]

    json_path = OUT_DIR / f"S{session_num}_{region}_layer1b_batch.json"
    with open(json_path, "w") as f:
        json.dump({"stats": stats_data, "null": sig, "K": K,
                   "var_expl_pct": var_expl, "N_landmarks": SUBSAMPLE_N,
                   "N_shuffles": N_SHUFFLES, "n_units": int(n_units)}, f, indent=2)
    print(f"    Saved {json_path}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=str,
                        default="3,4,11,12,13,14,15,16,19,20,21,22,23,24",
                        help="comma-separated dual-probe session numbers")
    args = parser.parse_args()

    sessions = [int(s) for s in args.sessions.split(",")]
    print(f"Running dual-probe Layer 1b batch on sessions: {sessions}")
    print(f"Parameters: SUBSAMPLE_N={SUBSAMPLE_N}, N_SHUFFLES={N_SHUFFLES}, "
          f"K_PCS={K_PCS}")
    rng = np.random.default_rng(20260429)

    rows = []
    for s in sessions:
        for region in ("ACA", "LHA"):
            t0 = timer.time()
            try:
                row = run_region_batch(s, region, K_PCS[region], rng)
                rows.append(row)
            except Exception as e:
                print(f"    [S{s}_{region}] FAILED: {e}")
            print(f"    [S{s}_{region}] elapsed {(timer.time()-t0)/60:.1f} min")

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "manifold_layer1b_batch.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")
    print(df[["session", "state", "phase", "region", "n_units",
              "H1_max_pers", "H1_max_pers_null_mean", "H1_max_pers_p",
              "H2_max_pers", "H2_max_pers_p"]].to_string(index=False))


if __name__ == "__main__":
    main()
