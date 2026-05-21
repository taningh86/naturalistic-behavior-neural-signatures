"""07 — Compare HMM state usage between fed and fasted foraging sessions.

For each session computes per-state occupancy, mean dwell, and the empirical
transition matrix (from Viterbi). Then tests fed vs fasted at the session level
(Mann-Whitney U on session means; per-pair counting for transitions).

Outputs:
  data/HMM/fed_vs_fasted.csv   — per-session × state metrics
  data/HMM/fed_vs_fasted_transitions.csv — per-session × (i,j) transition probs
  data/HMM/fed_vs_fasted_stats.csv — per-state state-comparison stats
  figures/HMM/fed_vs_fasted.png — bar/scatter overview
"""
import pickle
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import load_config, session_list, ensure_dir, REPO_ROOT


def runs(viterbi):
    out = []
    if len(viterbi) == 0:
        return out
    cur = viterbi[0]
    n = 1
    for v in viterbi[1:]:
        if v == cur:
            n += 1
        else:
            out.append((int(cur), int(n)))
            cur = v
            n = 1
    out.append((int(cur), int(n)))
    return out


def transition_matrix_empirical(viterbi, N):
    """Row-stochastic empirical transition matrix from a single Viterbi sequence."""
    counts = np.zeros((N, N), dtype=np.float64)
    for a, b in zip(viterbi[:-1], viterbi[1:]):
        counts[a, b] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return counts / row_sums


def mwu(a, b):
    """Mann-Whitney U two-sided. Returns (U, p, n_a, n_b)."""
    a = np.asarray(a)
    b = np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan"), len(a), len(b)
    try:
        U, p = mannwhitneyu(a, b, alternative="two-sided")
    except ValueError:
        return float("nan"), float("nan"), len(a), len(b)
    return float(U), float(p), len(a), len(b)


def main():
    cfg = load_config()
    prepared_dir = REPO_ROOT / cfg["out_dirs"]["prepared"]
    posteriors_dir = REPO_ROOT / cfg["out_dirs"]["posteriors"]
    bin_s = cfg["target_bin_ms"] / 1000.0

    model_path = REPO_ROOT / cfg["out_dirs"]["final_model"]
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    N = bundle["N"]

    sess = session_list(cfg)

    occ_rows = []  # per-session × state
    trans_rows = []  # per-session × (i, j)
    for session_num, state in sess:
        post_p = posteriors_dir / f"session_{session_num}.csv"
        if not post_p.exists():
            print(f"  SKIP S{session_num}: missing {post_p}")
            continue
        post_df = pd.read_csv(post_p)
        soft = post_df[[f"p_state_{k}" for k in range(N)]].values
        vit = post_df["viterbi"].values.astype(np.int64)
        T = len(vit)

        # Soft + hard occupancy.
        occ_soft = soft.sum(axis=0) / soft.sum()
        occ_hard = np.array([(vit == k).mean() for k in range(N)])

        # Mean dwell (bins) per state from Viterbi.
        runs_all = runs(vit)
        dwell_by_state = {k: [] for k in range(N)}
        for st, length in runs_all:
            dwell_by_state[st].append(length)
        mean_dwell_bins = np.array([
            float(np.mean(dwell_by_state[k])) if dwell_by_state[k] else 0.0
            for k in range(N)
        ])

        for k in range(N):
            occ_rows.append(dict(
                session_num=session_num,
                state_label=state,
                hmm_state=k,
                T=T,
                occupancy_soft=float(occ_soft[k]),
                occupancy_hard=float(occ_hard[k]),
                mean_dwell_bins=float(mean_dwell_bins[k]),
                mean_dwell_s=float(mean_dwell_bins[k] * bin_s),
            ))

        P_emp = transition_matrix_empirical(vit, N)
        for i in range(N):
            for j in range(N):
                trans_rows.append(dict(
                    session_num=session_num,
                    state_label=state,
                    from_state=i,
                    to_state=j,
                    p=float(P_emp[i, j]),
                ))

    occ_df = pd.DataFrame(occ_rows)
    trans_df = pd.DataFrame(trans_rows)

    occ_csv = REPO_ROOT / cfg["out_dirs"]["fed_vs_fasted_csv"]
    ensure_dir(occ_csv.parent)
    occ_df.to_csv(occ_csv, index=False)
    print(f"Saved {occ_csv}")

    trans_csv = occ_csv.parent / "fed_vs_fasted_transitions.csv"
    trans_df.to_csv(trans_csv, index=False)
    print(f"Saved {trans_csv}")

    # Per-state fed-vs-fasted comparison (occupancy & dwell).
    stat_rows = []
    for k in range(N):
        sub = occ_df[occ_df["hmm_state"] == k]
        fed_occ = sub.loc[sub["state_label"] == "fed", "occupancy_soft"].values
        fas_occ = sub.loc[sub["state_label"] == "fasted", "occupancy_soft"].values
        fed_dw = sub.loc[sub["state_label"] == "fed", "mean_dwell_s"].values
        fas_dw = sub.loc[sub["state_label"] == "fasted", "mean_dwell_s"].values

        U_o, p_o, n_f, n_fa = mwu(fed_occ, fas_occ)
        U_d, p_d, _, _ = mwu(fed_dw, fas_dw)

        stat_rows.append(dict(
            hmm_state=k,
            n_fed=n_f, n_fasted=n_fa,
            mean_occ_fed=float(fed_occ.mean()) if len(fed_occ) else float("nan"),
            mean_occ_fasted=float(fas_occ.mean()) if len(fas_occ) else float("nan"),
            occ_U=U_o, occ_p=p_o,
            mean_dwell_s_fed=float(fed_dw.mean()) if len(fed_dw) else float("nan"),
            mean_dwell_s_fasted=float(fas_dw.mean()) if len(fas_dw) else float("nan"),
            dwell_U=U_d, dwell_p=p_d,
        ))

    stat_df = pd.DataFrame(stat_rows)
    stat_csv = occ_csv.parent / "fed_vs_fasted_stats.csv"
    stat_df.to_csv(stat_csv, index=False)
    print(f"Saved {stat_csv}")
    print("\nPer-state state-comparison stats:")
    print(stat_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Plot: occupancy + mean dwell, fed vs fasted, with session points.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    states_idx = np.arange(N)
    width = 0.36
    colors = {"fed": "steelblue", "fasted": "firebrick"}

    for ax, metric, ylabel in [
        (axes[0], "occupancy_soft", "Occupancy (soft)"),
        (axes[1], "mean_dwell_s", "Mean dwell (s)"),
    ]:
        for offset, group in [(-width / 2, "fed"), (+width / 2, "fasted")]:
            sub = occ_df[occ_df["state_label"] == group]
            grouped = sub.groupby("hmm_state")[metric]
            mean = grouped.mean().reindex(states_idx).values
            sem = grouped.sem().reindex(states_idx).values
            ax.bar(states_idx + offset, mean, width=width,
                   yerr=sem, label=group, color=colors[group],
                   alpha=0.75, capsize=3)
            # session-level scatter
            for k in states_idx:
                vals = sub.loc[sub["hmm_state"] == k, metric].values
                ax.scatter(np.full_like(vals, k + offset, dtype=float),
                           vals, color="k", s=12, alpha=0.7, zorder=3)
        # significance markers
        for k in states_idx:
            if metric == "occupancy_soft":
                p = stat_df.loc[stat_df["hmm_state"] == k, "occ_p"].iloc[0]
            else:
                p = stat_df.loc[stat_df["hmm_state"] == k, "dwell_p"].iloc[0]
            if np.isfinite(p) and p < 0.05:
                ymax = max(occ_df.loc[occ_df["hmm_state"] == k, metric].max(), 0)
                ax.text(k, ymax * 1.05, "*", ha="center", fontsize=14)

        ax.set_xticks(states_idx)
        ax.set_xticklabels([f"S{k}" for k in states_idx])
        ax.set_xlabel("HMM state")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False)
        ax.grid(alpha=0.25, axis="y")

    fig.suptitle("Fed vs Fasted — HMM state usage (session-level)")
    fig.tight_layout()
    fig_path = REPO_ROOT / cfg["out_dirs"]["fed_vs_fasted_fig"]
    ensure_dir(fig_path.parent)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved {fig_path}")

    # Mean transition matrix per group + difference plot.
    P_fed = trans_df[trans_df["state_label"] == "fed"].groupby(
        ["from_state", "to_state"])["p"].mean().unstack().reindex(
        index=range(N), columns=range(N)).fillna(0.0).values
    P_fas = trans_df[trans_df["state_label"] == "fasted"].groupby(
        ["from_state", "to_state"])["p"].mean().unstack().reindex(
        index=range(N), columns=range(N)).fillna(0.0).values
    dP = P_fas - P_fed

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, M, title, vmax in [
        (axes[0], P_fed, "Fed mean P", 1.0),
        (axes[1], P_fas, "Fasted mean P", 1.0),
        (axes[2], dP, "Fasted − Fed", float(np.abs(dP).max() or 1e-3)),
    ]:
        if title.startswith("Fasted − Fed"):
            im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
        else:
            im = ax.imshow(M, cmap="viridis", vmin=0, vmax=vmax, aspect="equal")
        ax.set_xticks(range(N))
        ax.set_yticks(range(N))
        ax.set_xticklabels([f"S{k}" for k in range(N)], rotation=45)
        ax.set_yticklabels([f"S{k}" for k in range(N)])
        ax.set_xlabel("to state")
        ax.set_ylabel("from state")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig2_path = fig_path.parent / "fed_vs_fasted_transitions.png"
    fig.savefig(fig2_path, dpi=150)
    plt.close(fig)
    print(f"Saved {fig2_path}")

    n_fed_sessions = occ_df.loc[occ_df["state_label"] == "fed", "session_num"].nunique()
    n_fas_sessions = occ_df.loc[occ_df["state_label"] == "fasted", "session_num"].nunique()
    print(f"\nDone. Compared fed (n={n_fed_sessions}) vs fasted (n={n_fas_sessions}) "
          f"foraging sessions across {N} HMM states.")


if __name__ == "__main__":
    main()
