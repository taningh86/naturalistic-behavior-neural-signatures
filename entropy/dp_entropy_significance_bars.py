"""Significance bar plots for entropy inflection-locked neural responses.

Two missing summary figures added on top of the existing trace plots:
  Figure 1: 3-state comparison (fed / fasted / fed-HFD) per metric (ACA FR,
    LHA FR, ACA PC1, LHA PC1, Velocity, Entropy) split by inflection
    (peak / trough). Kruskal-Wallis across states + pairwise Mann-Whitney.
  Figure 2: LHA vs ACA paired comparison within state, per metric type
    (FR, PC1) split by inflection. Wilcoxon signed-rank across sessions.

Uses session-level deltas (post_mean - pre_mean) from
`data/dp_entropy_inflection_stats.csv` — one number per session × inflection
× metric. n_per_group: fed 8, fasted 5, fed-HFD 4.

Outputs:
  figures/dp_entropy_sig_states_by_metric.png
  figures/dp_entropy_sig_lha_vs_aca_paired.png
  data/dp_entropy_significance_summary.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kruskal, mannwhitneyu, wilcoxon


REPO = Path(__file__).resolve().parent
SRC_CSV = REPO / "data" / "dp_entropy_inflection_stats.csv"
FIG_STATES = REPO / "figures" / "dp_entropy_sig_states_by_metric.png"
FIG_LHA_ACA = REPO / "figures" / "dp_entropy_sig_lha_vs_aca_paired.png"
OUT_CSV = REPO / "data" / "dp_entropy_significance_summary.csv"

STATES = ["fed", "fasted", "fed-HFD"]
STATE_LABEL = {"fed": "Fed", "fasted": "Fasted", "fed-HFD": "HFD"}
STATE_COLOR = {"fed": "#1f77b4", "fasted": "#d62728", "fed-HFD": "#9b59b6"}
INFLECTIONS = ["peak", "trough"]
METRICS_ALL = ["Entropy", "Velocity", "ACA FR", "LHA FR", "ACA PC1", "LHA PC1"]
REGION_PAIRS = [("ACA FR", "LHA FR", "FR"),
                  ("ACA PC1", "LHA PC1", "PC1")]
REGION_COLOR = {"ACA": "#8e44ad", "LHA": "#e67e22"}


def stars(p):
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def main():
    df = pd.read_csv(SRC_CSV)

    # Aggregate: one row per (session, inflection, metric, state) with `delta`
    # (already at that granularity in the input)
    # Drop session_20 truncated stuff if needed — the data already has it.

    summary_rows = []

    # ============= Figure 1: across-state per (inflection, metric) =============
    n_metrics = len(METRICS_ALL)
    fig, axes = plt.subplots(2, n_metrics, figsize=(3 * n_metrics, 8),
                               sharex=False)
    for r, infl in enumerate(INFLECTIONS):
        for c, metric in enumerate(METRICS_ALL):
            ax = axes[r, c]
            sub = df[(df.inflection == infl) & (df.metric == metric)]
            # Collect per-state deltas
            data_per_state = {}
            means, sems, ns = [], [], []
            for st in STATES:
                vals = sub[sub.state == st]["delta"].dropna().values
                data_per_state[st] = vals
                means.append(float(vals.mean()) if len(vals) else np.nan)
                sems.append(float(vals.std() / np.sqrt(max(1, len(vals)))) if len(vals) else 0)
                ns.append(int(len(vals)))
            # Kruskal-Wallis across all 3 states
            try:
                valid_groups = [v for v in data_per_state.values() if len(v) >= 2]
                if len(valid_groups) >= 2:
                    _, kw_p = kruskal(*valid_groups)
                else:
                    kw_p = np.nan
            except Exception:
                kw_p = np.nan
            # Pairwise MWU (no MC correction; small n)
            pairwise = {}
            pairs = [("fed", "fasted"), ("fed", "fed-HFD"), ("fasted", "fed-HFD")]
            for a, b in pairs:
                va, vb = data_per_state[a], data_per_state[b]
                if len(va) >= 2 and len(vb) >= 2:
                    try:
                        _, p = mannwhitneyu(va, vb, alternative="two-sided")
                    except Exception:
                        p = np.nan
                else:
                    p = np.nan
                pairwise[(a, b)] = p
                summary_rows.append(dict(
                    test="state_pairwise_MWU", inflection=infl, metric=metric,
                    state_a=a, state_b=b, n_a=len(va), n_b=len(vb),
                    mean_a=float(va.mean()) if len(va) else np.nan,
                    mean_b=float(vb.mean()) if len(vb) else np.nan,
                    p_value=p, stars=stars(p),
                ))
            summary_rows.append(dict(
                test="state_kw", inflection=infl, metric=metric,
                state_a=None, state_b=None,
                n_a=ns[0], n_b=ns[1],
                mean_a=means[0], mean_b=means[1],
                p_value=kw_p, stars=stars(kw_p),
            ))
            # Draw bars
            x = np.arange(len(STATES))
            ax.bar(x, means, yerr=sems,
                    color=[STATE_COLOR[s] for s in STATES],
                    edgecolor="black", capsize=4)
            # Annotate n
            for xi, n in enumerate(ns):
                ax.text(xi, ax.get_ylim()[0] if False else
                              (means[xi] + sems[xi] * np.sign(means[xi]) + 0.001
                               if means[xi] is not None else 0),
                          f"n={n}", ha="center", fontsize=7, color="gray")
            # KW p
            kw_label = f"KW p={kw_p:.3f}" if np.isfinite(kw_p) else "KW p=ns"
            ax.set_title(f"{metric}\n{infl}: {kw_label}", fontsize=9)
            ax.axhline(0, color="k", lw=0.6)
            ax.set_xticks(x); ax.set_xticklabels([STATE_LABEL[s] for s in STATES],
                                                   rotation=30, fontsize=8)
            if c == 0:
                ax.set_ylabel(f"{infl} delta\n(post − pre)", fontsize=10)
            # Pairwise sig markers above bars
            y_top = max(means[i] + sems[i] for i in range(len(STATES))
                          if means[i] is not None and np.isfinite(means[i]))
            y_bot = min(means[i] - sems[i] for i in range(len(STATES))
                          if means[i] is not None and np.isfinite(means[i]))
            y_range = y_top - y_bot if y_top > y_bot else abs(y_top) + 0.01
            y_offset = y_top + y_range * 0.1
            for (a, b), p in pairwise.items():
                if not np.isfinite(p) or p >= 0.05:
                    continue
                ia = STATES.index(a); ib = STATES.index(b)
                ax.plot([ia, ib], [y_offset, y_offset], color="k", lw=0.6)
                ax.text((ia + ib) / 2, y_offset + y_range * 0.02, stars(p),
                          ha="center", fontsize=10, fontweight="bold")
                y_offset += y_range * 0.12
            ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Entropy-inflection neural response per metric — fed / fasted / HFD comparison\n"
                  "(session-level delta = post_mean − pre_mean; Kruskal-Wallis + pairwise Mann-Whitney)",
                  fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG_STATES, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIG_STATES}")

    # ============= Figure 2: LHA vs ACA paired per (state, inflection) =============
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=False)
    for r, infl in enumerate(INFLECTIONS):
        for c, (aca_metric, lha_metric, label) in enumerate(REGION_PAIRS):
            ax = axes[r, c]
            x_positions = []
            bar_means_aca = []; bar_means_lha = []
            bar_sems_aca = []; bar_sems_lha = []
            bar_labels = []
            stars_text = []
            for si, st in enumerate(STATES):
                sub_aca = df[(df.state == st) & (df.inflection == infl)
                              & (df.metric == aca_metric)]
                sub_lha = df[(df.state == st) & (df.inflection == infl)
                              & (df.metric == lha_metric)]
                # Align by session
                merged = sub_aca[["session", "delta"]].merge(
                    sub_lha[["session", "delta"]],
                    on="session", suffixes=("_aca", "_lha"))
                if not len(merged):
                    continue
                aca_vals = merged["delta_aca"].dropna().values
                lha_vals = merged["delta_lha"].dropna().values
                bar_means_aca.append(aca_vals.mean())
                bar_sems_aca.append(aca_vals.std() / np.sqrt(max(1, len(aca_vals))))
                bar_means_lha.append(lha_vals.mean())
                bar_sems_lha.append(lha_vals.std() / np.sqrt(max(1, len(lha_vals))))
                bar_labels.append(f"{STATE_LABEL[st]}\nn={len(aca_vals)}")
                x_positions.append(si)
                if len(aca_vals) >= 2 and len(lha_vals) >= 2:
                    try:
                        _, p = wilcoxon(aca_vals, lha_vals, alternative="two-sided")
                    except Exception:
                        p = np.nan
                else:
                    p = np.nan
                stars_text.append(stars(p))
                summary_rows.append(dict(
                    test="lha_vs_aca_paired_wilcoxon", inflection=infl, metric=label,
                    state_a=f"ACA {label}", state_b=f"LHA {label}",
                    n_a=int(len(aca_vals)), n_b=int(len(lha_vals)),
                    mean_a=float(aca_vals.mean()), mean_b=float(lha_vals.mean()),
                    p_value=p, stars=stars(p),
                ))
            x = np.array(x_positions)
            width = 0.36
            ax.bar(x - width/2, bar_means_aca, width, yerr=bar_sems_aca,
                    color=REGION_COLOR["ACA"], edgecolor="black",
                    capsize=4, label="ACA")
            ax.bar(x + width/2, bar_means_lha, width, yerr=bar_sems_lha,
                    color=REGION_COLOR["LHA"], edgecolor="black",
                    capsize=4, label="LHA")
            # Sig markers
            for xi, sm in zip(x, stars_text):
                if sm:
                    y_top = max(
                        bar_means_aca[list(x).index(xi)] + bar_sems_aca[list(x).index(xi)],
                        bar_means_lha[list(x).index(xi)] + bar_sems_lha[list(x).index(xi)],
                    )
                    ax.text(xi, y_top + 0.05, sm, ha="center",
                              fontsize=11, fontweight="bold")
            ax.axhline(0, color="k", lw=0.6)
            ax.set_xticks(x); ax.set_xticklabels(bar_labels, fontsize=8)
            ax.set_title(f"{infl} — ACA vs LHA {label}", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{infl} delta", fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("ACA vs LHA peri-inflection response, paired Wilcoxon (across sessions per state)",
                  fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG_LHA_ACA, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIG_LHA_ACA}")

    pd.DataFrame(summary_rows).to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_CSV}")

    # Print significant cells
    sig = pd.DataFrame(summary_rows)
    sig = sig[(sig.p_value.notna()) & (sig.p_value < 0.05)]
    if len(sig):
        print("\nSignificant cells (p<0.05):")
        print(sig[["test", "inflection", "metric", "state_a", "state_b",
                     "p_value", "stars"]].to_string(index=False))
    else:
        print("\nNo cells reach p<0.05.")


if __name__ == "__main__":
    main()
