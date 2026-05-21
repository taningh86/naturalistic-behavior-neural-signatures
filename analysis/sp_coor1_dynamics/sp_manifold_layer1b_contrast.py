"""State contrast on Layer 1b persistent-homology metrics."""
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CSV_IN = REPO / "data" / "sp_coor1_dynamics" / "manifold_layer1b.csv"
CSV_OUT = REPO / "data" / "sp_coor1_dynamics" / "manifold_layer1b_state_contrast.csv"
N_BOOT = 5000
SEED = 20260428


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
    rows = []
    metrics = ("H0_max_pers", "H0_total_pers",
               "H1_max_pers", "H1_total_pers",
               "H2_max_pers", "H2_total_pers")
    for region, m in product(("LHA", "RSP"), metrics):
        sub = df[df["region"] == region]
        fed = sub[sub["state"] == "fed"][m].values
        fas = sub[sub["state"] == "fasted"][m].values
        diff, lo, hi = boot_diff(fed, fas, rng)
        excl = (lo > 0) or (hi < 0)
        rows.append({
            "region": region, "metric": m,
            "fed_mean": float(fed.mean()), "fed_sd": float(fed.std(ddof=1)),
            "fas_mean": float(fas.mean()), "fas_sd": float(fas.std(ddof=1)),
            "diff_fed_minus_fas": diff,
            "ci_lo": lo, "ci_hi": hi,
            "excludes_zero": excl,
            "pct_vs_fas": 100.0 * diff / fas.mean() if fas.mean() != 0 else np.nan,
        })
    out = pd.DataFrame(rows)
    out.to_csv(CSV_OUT, index=False)
    print(out.round(3).to_string(index=False))
    print(f"\nSaved {CSV_OUT}")


if __name__ == "__main__":
    main()
