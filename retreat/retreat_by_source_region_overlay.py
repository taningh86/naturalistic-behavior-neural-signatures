"""Peri-retreat by source zone × metabolic state, with LHA vs RSP overlaid.

Same pooled data as retreat_by_source_fed_vs_fasted.py, but the layout puts
LHA and RSP traces in the SAME panel (different colors) and uses fed vs
fasted as external panels (separate columns).

Layout:
  Rows = source zone (transition, corner, arena_center, pot_area)
  Cols = (fed Pop FR, fed PC1, fasted Pop FR, fasted PC1)
  Lines: LHA (orange) vs RSP (purple), mean ± SEM

Outputs:
  figures/retreat_by_source_lha_vs_rsp.png
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


REPO = Path(__file__).resolve().parent
NPZ = REPO / "data" / "retreat_peri_event_all_sessions.npz"
CSV = REPO / "data" / "retreat_transitions_all_sessions.csv"
FIG = REPO / "figures" / "retreat_by_source_lha_vs_rsp.png"

SOURCE_ORDER = ["transition", "corner", "arena_center", "pot_area"]
SOURCE_LABEL = {
    "transition": "Transition zone",
    "corner": "Corner",
    "arena_center": "Arena center",
    "pot_area": "Pot area",
}
SMOOTH_SIGMA = 1.5

REGION_COLOR = {"lha": "#e67e22", "rsp": "#8e44ad"}
REGION_LABEL = {"lha": "LHA", "rsp": "RSP"}


def main():
    d = np.load(NPZ, allow_pickle=True)
    t = pd.read_csv(CSV)
    peri_time = d["peri_time"]

    pools = {}    # (region, metric, source, state) -> list of arrays
    for region in ("lha", "rsp"):
        for snum in sorted(t["session"].unique()):
            sub = t[t["session"] == snum].reset_index(drop=True)
            state = sub["state"].iloc[0]
            pop = d[f"s{snum}_{region}_peri_pop"]
            pc = d[f"s{snum}_{region}_peri_pc"]
            for src in SOURCE_ORDER:
                mask = (sub["source_category"] == src).values
                if not mask.any():
                    continue
                pools.setdefault((region, "pop", src, state), []).append(pop[mask])
                pools.setdefault((region, "pc1", src, state), []).append(pc[mask, :, 0])
    pooled = {k: np.concatenate(v, axis=0) for k, v in pools.items()}

    n_rows = len(SOURCE_ORDER)
    fig, axes = plt.subplots(n_rows, 4,
                              figsize=(18, 3.2 * n_rows),
                              sharex=True)

    col_specs = [
        ("fed", "pop", "Fed — Population FR"),
        ("fed", "pc1", "Fed — PC1"),
        ("fasted", "pop", "Fasted — Population FR"),
        ("fasted", "pc1", "Fasted — PC1"),
    ]

    for r, src in enumerate(SOURCE_ORDER):
        for c, (state, metric, label) in enumerate(col_specs):
            ax = axes[r, c]
            for region in ("lha", "rsp"):
                arr = pooled.get((region, metric, src, state))
                if arr is None or len(arr) == 0:
                    continue
                m = gaussian_filter1d(arr.mean(axis=0), SMOOTH_SIGMA)
                if len(arr) >= 2:
                    s = gaussian_filter1d(arr.std(axis=0) / np.sqrt(len(arr)),
                                            SMOOTH_SIGMA)
                    ax.fill_between(peri_time, m - s, m + s,
                                      alpha=0.25, color=REGION_COLOR[region])
                ls = "-" if len(arr) >= 2 else ":"
                lw = 1.6 if len(arr) >= 2 else 1.0
                ax.plot(peri_time, m, color=REGION_COLOR[region], ls=ls, lw=lw,
                          label=f"{REGION_LABEL[region]} (n={len(arr)})")
            ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.5)
            ax.set_title(f"{SOURCE_LABEL[src]} — {label}", fontsize=10)
            if c == 0:
                ax.set_ylabel(SOURCE_LABEL[src], fontsize=10)
            if r == n_rows - 1:
                ax.set_xlabel("Time from retreat (s)", fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    fig.suptitle("Peri-retreat by source zone — LHA (orange) vs RSP (purple), "
                  "split by fed vs fasted (cols), pooled across 8 sessions",
                  fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIG}")


if __name__ == "__main__":
    main()
