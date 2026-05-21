"""
State contrast (fed vs fasted) bootstrap on Layer 1a dimensionality estimates.

For each region x metric: 5000-resample bootstrap of group-mean diff (n=4 vs n=4).
Reports point estimate, 95% CI, percent change.

NOTE: Fed sessions have ~1.5-1.7x more units than fasted, so all dim estimates
(except potentially TwoNN) carry an N confound. This script reports the raw
contrast as-is; interpret in light of unit-count differences.

Outputs:
  - data/sp_coor1_dynamics/manifold_layer1a_state_contrast.csv
"""
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CSV_IN = REPO / "data" / "sp_coor1_dynamics" / "manifold_layer1a.csv"
CSV_OUT = REPO / "data" / "sp_coor1_dynamics" / "manifold_layer1a_state_contrast.csv"
N_BOOT = 5000
SEED = 20260428


def boot_diff(a, b, n_boot=N_BOOT, rng=None):
    rng = rng or np.random.default_rng(SEED)
    a = np.asarray(a)
    b = np.asarray(b)
    obs = float(a.mean() - b.mean())
    boots = np.empty(n_boot)
    for i in range(n_boot):
        ai = rng.choice(a, size=len(a), replace=True)
        bi = rng.choice(b, size=len(b), replace=True)
        boots[i] = ai.mean() - bi.mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return obs, float(lo), float(hi)


def main():
    df = pd.read_csv(CSV_IN)
    rows = []
    rng = np.random.default_rng(SEED)
    for region, metric in product(("LHA", "RSP"),
                                  ("PR", "TwoNN", "CorrDim", "Isomap", "n_units")):
        sub = df[df["region"] == region]
        fed = sub[sub["state"] == "fed"][metric].dropna().values
        fas = sub[sub["state"] == "fasted"][metric].dropna().values
        if len(fed) == 0 or len(fas) == 0:
            continue
        diff, lo, hi = boot_diff(fed, fas, rng=rng)
        excludes_zero = (lo > 0) or (hi < 0)
        pct = 100.0 * diff / fas.mean() if fas.mean() != 0 else np.nan
        rows.append({
            "region": region,
            "metric": metric,
            "fed_mean": float(fed.mean()),
            "fed_sd": float(fed.std(ddof=1)),
            "fas_mean": float(fas.mean()),
            "fas_sd": float(fas.std(ddof=1)),
            "diff_fed_minus_fas": diff,
            "ci_lo": lo,
            "ci_hi": hi,
            "excludes_zero": excludes_zero,
            "pct_vs_fas": pct,
        })
    out = pd.DataFrame(rows)
    out.to_csv(CSV_OUT, index=False)
    print(out.round(3).to_string(index=False))
    print(f"\nSaved {CSV_OUT}")


if __name__ == "__main__":
    main()
