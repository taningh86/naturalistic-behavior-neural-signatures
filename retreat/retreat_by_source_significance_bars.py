"""Per-zone significance bar plots for retreat data.

Two missing comparisons added:
  Figure 1: fed vs fasted within each (source zone, region, metric) — Mann-Whitney U
  Figure 2: LHA vs RSP within each (source zone, state, metric) — Wilcoxon paired

Per-retreat scalar = mean post-retreat (0 to +5 s) value of the metric (Pop FR or PC1).

Outputs:
  figures/retreat_sig_fed_vs_fasted_by_source.png
  figures/retreat_sig_lha_vs_rsp_by_source.png
  data/retreat_significance_summary.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, wilcoxon


REPO = Path(__file__).resolve().parent
NPZ = REPO / "data" / "retreat_peri_event_all_sessions.npz"
CSV = REPO / "data" / "retreat_transitions_all_sessions.csv"
FIG_FED_FAS = REPO / "figures" / "retreat_sig_fed_vs_fasted_by_source.png"
FIG_LHA_RSP = REPO / "figures" / "retreat_sig_lha_vs_rsp_by_source.png"
OUT_CSV = REPO / "data" / "retreat_significance_summary.csv"

SOURCE_ORDER = ["transition", "corner", "arena_center", "pot_area"]
SOURCE_LABEL = {
    "transition": "Transition\nzone",
    "corner": "Corner",
    "arena_center": "Arena\ncenter",
    "pot_area": "Pot\narea",
}
POST_WINDOW = (0.0, 5.0)             # seconds — post-retreat scalar = mean here
STATE_COLOR = {"fed": "#1f77b4", "fasted": "#d62728"}
REGION_COLOR = {"lha": "#e67e22", "rsp": "#8e44ad"}


def sig_stars(p):
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def per_retreat_scalars(peri_time, arr, window):
    """Reduce (n_retreats, n_timepoints) to (n_retreats,) by mean over window."""
    mask = (peri_time >= window[0]) & (peri_time <= window[1])
    return arr[:, mask].mean(axis=1)


def main():
    d = np.load(NPZ, allow_pickle=True)
    t = pd.read_csv(CSV)
    peri_time = d["peri_time"]

    # Build per-retreat scalars: {(region, metric, src, state): array of values}
    # Also track paired arrays for LHA/RSP: {(metric, src, state): {lha:[], rsp:[]}}
    pools = {}
    paired_pools = {}        # (metric, src, state) -> dict 'lha' and 'rsp' lists, aligned
    for snum in sorted(t["session"].unique()):
        sub = t[t["session"] == snum].reset_index(drop=True)
        state = sub["state"].iloc[0]
        # Compute per-retreat scalars per region per metric
        scalars = {}
        for region in ("lha", "rsp"):
            pop = d[f"s{snum}_{region}_peri_pop"]
            pc = d[f"s{snum}_{region}_peri_pc"][:, :, 0]
            scalars[(region, "pop")] = per_retreat_scalars(peri_time, pop, POST_WINDOW)
            scalars[(region, "pc1")] = per_retreat_scalars(peri_time, pc, POST_WINDOW)
        # Distribute into pools per source category
        for src in SOURCE_ORDER:
            mask = (sub["source_category"] == src).values
            if not mask.any():
                continue
            for region in ("lha", "rsp"):
                for metric in ("pop", "pc1"):
                    pools.setdefault((region, metric, src, state), []).append(
                        scalars[(region, metric)][mask]
                    )
            # paired LHA/RSP
            for metric in ("pop", "pc1"):
                p = paired_pools.setdefault((metric, src, state),
                                              {"lha": [], "rsp": []})
                p["lha"].append(scalars[("lha", metric)][mask])
                p["rsp"].append(scalars[("rsp", metric)][mask])

    pooled = {k: np.concatenate(v) for k, v in pools.items()}
    paired = {k: {"lha": np.concatenate(v["lha"]),
                   "rsp": np.concatenate(v["rsp"])}
              for k, v in paired_pools.items()}

    # =============== Stats ===============
    rows = []
    for metric in ("pop", "pc1"):
        for src in SOURCE_ORDER:
            # Fed vs fasted (per region)
            for region in ("lha", "rsp"):
                fed = pooled.get((region, metric, src, "fed"), np.array([]))
                fas = pooled.get((region, metric, src, "fasted"), np.array([]))
                if len(fed) >= 2 and len(fas) >= 2:
                    try:
                        _, p = mannwhitneyu(fed, fas, alternative="two-sided")
                    except Exception:
                        p = np.nan
                else:
                    p = np.nan
                rows.append(dict(
                    test="fed_vs_fasted", metric=metric, source=src,
                    region=region, state=None,
                    n_fed=int(len(fed)), n_fasted=int(len(fas)),
                    fed_mean=float(fed.mean()) if len(fed) else np.nan,
                    fasted_mean=float(fas.mean()) if len(fas) else np.nan,
                    fed_sem=float(fed.std()/np.sqrt(max(1, len(fed))))
                       if len(fed) else np.nan,
                    fasted_sem=float(fas.std()/np.sqrt(max(1, len(fas))))
                       if len(fas) else np.nan,
                    p_value=float(p) if np.isfinite(p) else np.nan,
                    stars=sig_stars(p),
                ))
            # LHA vs RSP (paired per state)
            for state in ("fed", "fasted"):
                pp = paired.get((metric, src, state))
                if pp is None:
                    continue
                lha = pp["lha"]; rsp = pp["rsp"]
                if len(lha) >= 2 and len(rsp) >= 2:
                    try:
                        _, p = wilcoxon(lha, rsp, alternative="two-sided")
                    except Exception:
                        p = np.nan
                else:
                    p = np.nan
                rows.append(dict(
                    test="lha_vs_rsp", metric=metric, source=src,
                    region=None, state=state,
                    n_fed=int(len(lha)) if state == "fed" else None,
                    n_fasted=int(len(lha)) if state == "fasted" else None,
                    fed_mean=None, fasted_mean=None,
                    fed_sem=None, fasted_sem=None,
                    lha_mean=float(lha.mean()),
                    rsp_mean=float(rsp.mean()),
                    lha_sem=float(lha.std() / np.sqrt(len(lha))),
                    rsp_sem=float(rsp.std() / np.sqrt(len(rsp))),
                    n_pairs=int(len(lha)),
                    p_value=float(p) if np.isfinite(p) else np.nan,
                    stars=sig_stars(p),
                ))
    stats = pd.DataFrame(rows)
    stats.to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_CSV}")

    # =============== Figure 1: Fed vs fasted per (region, metric, zone) ===============
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for row_i, metric in enumerate(("pop", "pc1")):
        for col_i, region in enumerate(("lha", "rsp")):
            ax = axes[row_i, col_i]
            x = np.arange(len(SOURCE_ORDER))
            width = 0.38
            fed_means = []; fed_sems = []
            fas_means = []; fas_sems = []
            stars_list = []
            for src in SOURCE_ORDER:
                rr = stats[(stats.test == "fed_vs_fasted") & (stats.metric == metric)
                            & (stats.source == src) & (stats.region == region)].iloc[0]
                fed_means.append(rr["fed_mean"])
                fed_sems.append(rr["fed_sem"])
                fas_means.append(rr["fasted_mean"])
                fas_sems.append(rr["fasted_sem"])
                stars_list.append(rr["stars"])
            ax.bar(x - width/2, fed_means, width, yerr=fed_sems,
                    color=STATE_COLOR["fed"], label="fed", capsize=4)
            ax.bar(x + width/2, fas_means, width, yerr=fas_sems,
                    color=STATE_COLOR["fasted"], label="fasted", capsize=4)
            # Significance annotations
            for xi, sm in enumerate(stars_list):
                if not sm or sm == "ns":
                    continue
                top = max(fed_means[xi] + (fed_sems[xi] or 0),
                           fas_means[xi] + (fas_sems[xi] or 0))
                ax.text(xi, top * 1.1 if top > 0 else top * 0.9,
                          sm, ha="center", fontsize=11, fontweight="bold")
            ax.axhline(0, color="k", lw=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels([SOURCE_LABEL[s] for s in SOURCE_ORDER], fontsize=9)
            ax.set_title(f"{region.upper()} — {'Pop FR' if metric=='pop' else 'PC1'}",
                          fontsize=11)
            if col_i == 0:
                ax.set_ylabel(f"Post-retreat mean (z-FR / PC1)", fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
    fig.suptitle(f"Fed vs fasted retreat response by source zone "
                  f"(post window {POST_WINDOW[0]}–{POST_WINDOW[1]} s; "
                  f"Mann-Whitney U on per-retreat scalars; *p<0.05, **p<0.01, ***p<0.001)",
                  fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG_FED_FAS, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIG_FED_FAS}")

    # =============== Figure 2: LHA vs RSP per (state, metric, zone) ===============
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for row_i, metric in enumerate(("pop", "pc1")):
        for col_i, state in enumerate(("fed", "fasted")):
            ax = axes[row_i, col_i]
            x = np.arange(len(SOURCE_ORDER))
            width = 0.38
            lha_m, lha_s = [], []
            rsp_m, rsp_s = [], []
            stars_list = []; n_list = []
            for src in SOURCE_ORDER:
                rr = stats[(stats.test == "lha_vs_rsp") & (stats.metric == metric)
                            & (stats.source == src) & (stats.state == state)].iloc[0]
                lha_m.append(rr["lha_mean"]); lha_s.append(rr["lha_sem"])
                rsp_m.append(rr["rsp_mean"]); rsp_s.append(rr["rsp_sem"])
                stars_list.append(rr["stars"])
                n_list.append(rr["n_pairs"])
            ax.bar(x - width/2, lha_m, width, yerr=lha_s,
                    color=REGION_COLOR["lha"], label="LHA", capsize=4)
            ax.bar(x + width/2, rsp_m, width, yerr=rsp_s,
                    color=REGION_COLOR["rsp"], label="RSP", capsize=4)
            for xi, (sm, np_) in enumerate(zip(stars_list, n_list)):
                if sm and sm != "ns":
                    top = max((lha_m[xi] or 0) + (lha_s[xi] or 0),
                               (rsp_m[xi] or 0) + (rsp_s[xi] or 0))
                    ax.text(xi, top * 1.1 if top > 0 else top * 0.9,
                              sm, ha="center", fontsize=11, fontweight="bold")
                ax.text(xi, ax.get_ylim()[0] if False else
                              min((lha_m[xi] or 0) - (lha_s[xi] or 0),
                                  (rsp_m[xi] or 0) - (rsp_s[xi] or 0)) * 1.15,
                          f"n={np_}", ha="center", fontsize=7, color="gray")
            ax.axhline(0, color="k", lw=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels([SOURCE_LABEL[s] for s in SOURCE_ORDER], fontsize=9)
            ax.set_title(f"{state.capitalize()} — "
                          f"{'Pop FR' if metric=='pop' else 'PC1'}",
                          fontsize=11)
            if col_i == 0:
                ax.set_ylabel(f"Post-retreat mean", fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
    fig.suptitle(f"LHA vs RSP retreat response by source zone "
                  f"(post window {POST_WINDOW[0]}–{POST_WINDOW[1]} s; "
                  f"Wilcoxon paired on per-retreat scalars; *p<0.05, **p<0.01, ***p<0.001)",
                  fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG_LHA_RSP, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIG_LHA_RSP}")

    # Print summary
    print("\nSignificant cells (p<0.05):")
    sig = stats[stats.p_value < 0.05][["test", "metric", "source", "region",
                                          "state", "p_value", "stars"]]
    print(sig.to_string(index=False))


if __name__ == "__main__":
    main()
