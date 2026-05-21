"""
Dual-probe Layer 1b state contrast — fed vs fasted vs HFD on ACA + LHA topology.

Inputs: data/manifold/manifold_layer1b_batch.csv
Outputs:
  - data/manifold/manifold_layer1b_batch_state_contrast.csv  (pairwise diffs)
  - data/manifold/manifold_layer1b_batch_state_summary.csv   (per-state means)

Pairwise contrasts: fed vs fasted, fed vs HFD, fasted vs HFD.
Bootstrap: 5000 resamples of group-mean difference, n=2 fed (S3,S4) vs n=6
fasted (S11-S16) vs n=6 HFD (S19-S24). CIs at 2.5/97.5 percentile.
"""
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd

CSV_IN = Path("data/manifold/manifold_layer1b_batch.csv")
CSV_OUT_CONTRAST = Path("data/manifold/manifold_layer1b_batch_state_contrast.csv")
CSV_OUT_SUMMARY = Path("data/manifold/manifold_layer1b_batch_state_summary.csv")
N_BOOT = 5000
SEED = 20260429


def boot_diff(a, b, rng):
    a = np.asarray(a)
    b = np.asarray(b)
    obs = float(a.mean() - b.mean())
    boots = np.empty(N_BOOT)
    for i in range(N_BOOT):
        ai = rng.choice(a, size=len(a), replace=True)
        bi = rng.choice(b, size=len(b), replace=True)
        boots[i] = ai.mean() - bi.mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return obs, float(lo), float(hi)


def main():
    df = pd.read_csv(CSV_IN)
    rng = np.random.default_rng(SEED)

    metrics = ("n_units", "var_expl_pct",
               "H0_max_pers", "H0_total_pers",
               "H1_max_pers", "H1_total_pers",
               "H2_max_pers", "H2_total_pers")
    states = ("fed", "fasted", "HFD")

    summary_rows = []
    for region, state, m in product(("ACA", "LHA"), states, metrics):
        sub = df[(df["region"] == region) & (df["state"] == state)][m].values
        summary_rows.append({
            "region": region, "state": state, "metric": m,
            "n": int(len(sub)),
            "mean": float(sub.mean()) if len(sub) else np.nan,
            "sd": float(sub.std(ddof=1)) if len(sub) > 1 else np.nan,
        })
    pd.DataFrame(summary_rows).to_csv(CSV_OUT_SUMMARY, index=False)

    contrast_rows = []
    for region in ("ACA", "LHA"):
        for s1, s2 in combinations(states, 2):
            for m in metrics:
                sub = df[df["region"] == region]
                a = sub[sub["state"] == s1][m].values
                b = sub[sub["state"] == s2][m].values
                if len(a) < 2 or len(b) < 2:
                    continue
                diff, lo, hi = boot_diff(a, b, rng)
                excl = (lo > 0) or (hi < 0)
                contrast_rows.append({
                    "region": region, "metric": m,
                    "contrast": f"{s1}_minus_{s2}",
                    "n_a": int(len(a)), "n_b": int(len(b)),
                    "mean_a": float(a.mean()), "mean_b": float(b.mean()),
                    "diff": diff, "ci_lo": lo, "ci_hi": hi,
                    "excludes_zero": bool(excl),
                    "pct_vs_b": 100.0 * diff / b.mean() if b.mean() != 0 else np.nan,
                })
    out = pd.DataFrame(contrast_rows)
    out.to_csv(CSV_OUT_CONTRAST, index=False)

    print("=== Per-state summary ===")
    print(pd.DataFrame(summary_rows).round(3).to_string(index=False))
    print(f"\nSaved {CSV_OUT_SUMMARY}")

    print("\n=== Pairwise state contrasts ===")
    sig = out[out["excludes_zero"]]
    print("Significant contrasts (CI excludes 0):")
    if len(sig):
        print(sig[["region", "metric", "contrast", "mean_a", "mean_b",
                   "diff", "ci_lo", "ci_hi", "pct_vs_b"]].round(3).to_string(index=False))
    else:
        print("  (none)")

    print(f"\nSaved {CSV_OUT_CONTRAST}")


if __name__ == "__main__":
    main()
