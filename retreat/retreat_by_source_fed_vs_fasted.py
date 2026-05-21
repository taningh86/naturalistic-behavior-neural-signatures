"""Peri-retreat by source zone, fed vs fasted, pooled across sessions.

Uses the saved per-session peri-event arrays (`data/retreat_peri_event_all_sessions.npz`)
and the matching per-transition metadata (`data/retreat_transitions_all_sessions.csv`)
to produce a single figure with one row per source zone (transition, corner,
arena_center, optionally pot_area), one column per (region × metric), and two
overlaid traces per panel: fed (blue) vs fasted (red).

Outputs:
  figures/retreat_by_source_fed_vs_fasted.png
  data/retreat_by_source_fed_vs_fasted_n.csv     (n_transitions per cell)
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
FIG = REPO / "figures" / "retreat_by_source_fed_vs_fasted.png"
N_CSV = REPO / "data" / "retreat_by_source_fed_vs_fasted_n.csv"

SOURCE_ORDER = ["transition", "corner", "arena_center", "pot_area"]
SOURCE_LABEL = {
    "transition": "Transition zone",
    "corner": "Corner",
    "arena_center": "Arena center",
    "pot_area": "Pot area",
}
MIN_N_PER_CELL = 1             # include all source zones, flag low n in panel title
SMOOTH_SIGMA = 1.5             # bins (=150 ms at 100 ms time-step)


def main():
    d = np.load(NPZ, allow_pickle=True)
    t = pd.read_csv(CSV)
    peri_time = d["peri_time"]

    # Build per-source × fed/fasted pools of peri-event arrays
    pools = {}                  # (region, metric, source, state) -> list of arrays
    for region in ("lha", "rsp"):
        for snum in sorted(t["session"].unique()):
            sub = t[t["session"] == snum].reset_index(drop=True)
            state = sub["state"].iloc[0]
            pop = d[f"s{snum}_{region}_peri_pop"]    # (n, T)
            pc = d[f"s{snum}_{region}_peri_pc"]      # (n, T, n_pcs)
            assert pop.shape[0] == len(sub), (snum, pop.shape, len(sub))
            for src in SOURCE_ORDER:
                mask = (sub["source_category"] == src).values
                if not mask.any():
                    continue
                pools.setdefault((region, "pop", src, state), []).append(pop[mask])
                pools.setdefault((region, "pc1", src, state), []).append(pc[mask, :, 0])

    # Concatenate
    pooled = {k: np.concatenate(v, axis=0) for k, v in pools.items()}

    # Determine which sources have enough data in BOTH fed and fasted
    plottable = []
    n_summary_rows = []
    for src in SOURCE_ORDER:
        n_fed = len(pooled.get(("lha", "pop", src, "fed"), np.array([])))
        n_fas = len(pooled.get(("lha", "pop", src, "fasted"), np.array([])))
        n_summary_rows.append(dict(source=src, n_fed=int(n_fed), n_fasted=int(n_fas)))
        if n_fed >= MIN_N_PER_CELL and n_fas >= MIN_N_PER_CELL:
            plottable.append(src)
    pd.DataFrame(n_summary_rows).to_csv(N_CSV, index=False)
    print(f"Plottable sources (>={MIN_N_PER_CELL} per state): {plottable}")
    print(pd.DataFrame(n_summary_rows).to_string(index=False))

    if not plottable:
        print("No source has enough data to plot. Exiting.")
        return

    # Figure layout: rows = sources, cols = {LHA Pop, LHA PC1, RSP Pop, RSP PC1}
    n_rows = len(plottable)
    fig, axes = plt.subplots(n_rows, 4,
                              figsize=(18, 3.2 * n_rows),
                              sharex=True)
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    state_colors = {"fed": "#1f77b4", "fasted": "#d62728"}

    for r, src in enumerate(plottable):
        for c, (region, metric, label) in enumerate([
            ("lha", "pop", "LHA — Population FR"),
            ("lha", "pc1", "LHA — PC1"),
            ("rsp", "pop", "RSP — Population FR"),
            ("rsp", "pc1", "RSP — PC1"),
        ]):
            ax = axes[r, c]
            for state in ("fed", "fasted"):
                arr = pooled.get((region, metric, src, state))
                if arr is None or len(arr) == 0:
                    continue
                m = gaussian_filter1d(arr.mean(axis=0), SMOOTH_SIGMA)
                if len(arr) >= 2:
                    s = gaussian_filter1d(arr.std(axis=0) / np.sqrt(len(arr)),
                                            SMOOTH_SIGMA)
                    ax.fill_between(peri_time, m - s, m + s,
                                      alpha=0.25, color=state_colors[state])
                else:
                    # single trace — show as thinner line, no SEM
                    pass
                ax.plot(peri_time, m, color=state_colors[state],
                          lw=1.6 if len(arr) >= 2 else 1.0,
                          ls="-" if len(arr) >= 2 else ":",
                          label=f"{state} (n={len(arr)})")
            ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.5)
            ax.set_title(f"{SOURCE_LABEL[src]} — {label}", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{SOURCE_LABEL[src]}", fontsize=10)
            if r == n_rows - 1:
                ax.set_xlabel("Time from retreat (s)", fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    fig.suptitle("Peri-retreat by source zone — fed (blue) vs fasted (red), "
                  "pooled across 8 sessions",
                  fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIG}")
    print(f"Saved {N_CSV}")


if __name__ == "__main__":
    main()
