"""07 (dynamax) — Fed vs fasted comparison: occupancy, dwell, transition matrix.

Per session:
  - soft occupancy = mean(gamma) per state
  - hard occupancy = fraction of bins assigned by Viterbi
  - mean Viterbi dwell per state (s)
  - empirical transition matrix from Viterbi sequence (row-normalized)

Per state:
  - Mann-Whitney U on session-level fed-vs-fasted occupancy and dwell

Plots:
  - bar plots (occupancy + dwell) with session-level scatter
  - transition matrix triptych (fed mean, fasted mean, diff)
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import load_config, ensure_dir, REPO_ROOT


def empirical_transition_matrix(viterbi: np.ndarray, N: int) -> np.ndarray:
    """Row-normalized empirical transition matrix from a Viterbi sequence."""
    T = len(viterbi)
    A = np.zeros((N, N), dtype=np.float64)
    for t in range(T - 1):
        A[viterbi[t], viterbi[t + 1]] += 1
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return A / row_sums


def mean_dwell_bins(viterbi: np.ndarray, N: int) -> np.ndarray:
    """Mean Viterbi dwell per state (in bins)."""
    out = np.zeros(N, dtype=np.float64)
    if len(viterbi) == 0:
        return out
    run_idx = np.flatnonzero(np.diff(viterbi, prepend=-1, append=-1) != 0)
    starts = run_idx[:-1]
    ends = run_idx[1:]
    run_states = viterbi[starts]
    run_lens = ends - starts
    for k in range(N):
        m = run_states == k
        if m.any():
            out[k] = run_lens[m].mean()
    return out


def main():
    cfg = load_config()
    posteriors_dir = REPO_ROOT / cfg["dynamax_dirs"]["posteriors"]
    final_npz = REPO_ROOT / cfg["dynamax_dirs"]["final_model"]
    fvf_csv = REPO_ROOT / cfg["dynamax_dirs"]["fed_vs_fasted_csv"]
    fvf_trans_csv = REPO_ROOT / cfg["dynamax_dirs"]["fed_vs_fasted_transitions_csv"]
    fvf_stats_csv = REPO_ROOT / cfg["dynamax_dirs"]["fed_vs_fasted_stats_csv"]
    fvf_fig = REPO_ROOT / cfg["dynamax_dirs"]["fed_vs_fasted_fig"]
    fvf_trans_fig = REPO_ROOT / cfg["dynamax_dirs"]["fed_vs_fasted_transitions_fig"]
    ensure_dir(fvf_csv.parent)
    ensure_dir(fvf_fig.parent)

    z = np.load(final_npz, allow_pickle=True)
    N = int(z["K"])

    fed = cfg["sessions"]["fed"]
    fasted = cfg["sessions"]["fasted"]
    bin_size_s = float(cfg["target_bin_ms"]) / 1000.0

    rows = []
    trans_rows = []
    for state, sessions in [("fed", fed), ("fasted", fasted)]:
        for sn in sessions:
            df = pd.read_csv(posteriors_dir / f"session_{sn}.csv")
            gamma = df[[f"p_state_{k}" for k in range(N)]].values
            vit = df["viterbi"].values
            T = len(vit)
            soft_occ = gamma.mean(axis=0)
            hard_occ = np.bincount(vit, minlength=N) / T
            dwell_bins = mean_dwell_bins(vit, N)
            dwell_s = dwell_bins * bin_size_s
            for k in range(N):
                rows.append(dict(
                    session=sn, state_label=state, hmm_state=k,
                    soft_occupancy=float(soft_occ[k]),
                    hard_occupancy=float(hard_occ[k]),
                    mean_dwell_bins=float(dwell_bins[k]),
                    mean_dwell_s=float(dwell_s[k]),
                ))
            tm = empirical_transition_matrix(vit, N)
            for i in range(N):
                for j in range(N):
                    trans_rows.append(dict(
                        session=sn, state_label=state,
                        from_state=i, to_state=j, prob=float(tm[i, j]),
                    ))

    fvf = pd.DataFrame(rows)
    fvf.to_csv(fvf_csv, index=False)
    fvf_trans = pd.DataFrame(trans_rows)
    fvf_trans.to_csv(fvf_trans_csv, index=False)
    print(f"Per-session metrics → {fvf_csv}")
    print(f"Per-session transitions → {fvf_trans_csv}")

    # ---- Mann-Whitney U per state on soft_occupancy and mean_dwell_s ----
    stats_rows = []
    for k in range(N):
        for metric in ("soft_occupancy", "mean_dwell_s"):
            fed_vals = fvf.loc[(fvf.hmm_state == k) & (fvf.state_label == "fed"),
                               metric].values
            fasted_vals = fvf.loc[(fvf.hmm_state == k) & (fvf.state_label == "fasted"),
                                  metric].values
            try:
                U, p = mannwhitneyu(fed_vals, fasted_vals, alternative="two-sided")
            except ValueError:
                U, p = np.nan, np.nan
            stats_rows.append(dict(
                state=k, metric=metric,
                fed_mean=float(np.mean(fed_vals)),
                fasted_mean=float(np.mean(fasted_vals)),
                fed_n=len(fed_vals), fasted_n=len(fasted_vals),
                U=float(U) if U == U else np.nan,
                p=float(p) if p == p else np.nan,
            ))
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(fvf_stats_csv, index=False)
    print(f"Stats → {fvf_stats_csv}")

    # ---- Bar plots: occupancy and dwell ----
    fig, axes = plt.subplots(2, 1, figsize=(max(8, 0.7 * N), 7), sharex=True)
    states = np.arange(N)
    width = 0.35
    for ax, metric, ylab in [
        (axes[0], "soft_occupancy", "Soft occupancy"),
        (axes[1], "mean_dwell_s", "Mean dwell (s)"),
    ]:
        fed_per_state = [fvf.loc[(fvf.hmm_state == k) & (fvf.state_label == "fed"),
                                 metric].values for k in range(N)]
        fas_per_state = [fvf.loc[(fvf.hmm_state == k) & (fvf.state_label == "fasted"),
                                 metric].values for k in range(N)]
        fed_means = [np.mean(v) if len(v) else 0 for v in fed_per_state]
        fas_means = [np.mean(v) if len(v) else 0 for v in fas_per_state]
        fed_se = [np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0
                  for v in fed_per_state]
        fas_se = [np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0
                  for v in fas_per_state]
        ax.bar(states - width / 2, fed_means, width=width, yerr=fed_se,
               color="#4477aa", alpha=0.75, label="fed", capsize=2.5)
        ax.bar(states + width / 2, fas_means, width=width, yerr=fas_se,
               color="#cc6677", alpha=0.75, label="fasted", capsize=2.5)
        for k in range(N):
            ax.scatter(np.full(len(fed_per_state[k]), k - width / 2),
                       fed_per_state[k], color="#1f4060", s=12, zorder=3)
            ax.scatter(np.full(len(fas_per_state[k]), k + width / 2),
                       fas_per_state[k], color="#7a2c39", s=12, zorder=3)
        # significance stars
        for k in range(N):
            row = stats_df[(stats_df.state == k) & (stats_df.metric == metric)]
            if len(row):
                p = row.iloc[0]["p"]
                if p < 0.05:
                    star = "*"
                elif p < 0.1:
                    star = "."
                else:
                    star = ""
                if star:
                    y = max(fed_means[k] + fed_se[k], fas_means[k] + fas_se[k]) * 1.05
                    ax.text(k, y, star, ha="center", fontsize=12)
        ax.set_ylabel(ylab)
        ax.set_xticks(states)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    axes[1].set_xlabel("HMM state")
    fig.suptitle(f"dynamax MixedHMM — fed vs fasted (N={N})")
    fig.tight_layout()
    fig.savefig(fvf_fig, dpi=150)
    plt.close(fig)
    print(f"Bar plot → {fvf_fig}")

    # ---- Transition matrices ----
    fed_T = np.stack([np.array(fvf_trans[(fvf_trans.session == sn) &
                                          (fvf_trans.state_label == "fed")]
                                .pivot(index="from_state", columns="to_state",
                                       values="prob"))
                      for sn in fed], axis=0).mean(axis=0)
    fas_T = np.stack([np.array(fvf_trans[(fvf_trans.session == sn) &
                                          (fvf_trans.state_label == "fasted")]
                                .pivot(index="from_state", columns="to_state",
                                       values="prob"))
                      for sn in fasted], axis=0).mean(axis=0)
    diff = fas_T - fed_T

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    cmap_main = "viridis"
    cmap_diff = "RdBu_r"
    vmax = max(fed_T.max(), fas_T.max())
    im0 = axes[0].imshow(fed_T, vmin=0, vmax=vmax, cmap=cmap_main)
    axes[0].set_title("Fed (mean across sessions)")
    im1 = axes[1].imshow(fas_T, vmin=0, vmax=vmax, cmap=cmap_main)
    axes[1].set_title("Fasted (mean across sessions)")
    vmax_d = max(abs(diff.min()), abs(diff.max())) + 1e-9
    im2 = axes[2].imshow(diff, vmin=-vmax_d, vmax=vmax_d, cmap=cmap_diff)
    axes[2].set_title("Fasted − Fed")
    for ax, im, lab in [(axes[0], im0, "P"), (axes[1], im1, "P"), (axes[2], im2, "Δ")]:
        ax.set_xlabel("to state"); ax.set_ylabel("from state")
        plt.colorbar(im, ax=ax, fraction=0.046, label=lab)
    fig.suptitle(f"Empirical transition matrices — N={N}")
    fig.tight_layout()
    fig.savefig(fvf_trans_fig, dpi=150)
    plt.close(fig)
    print(f"Transition figure → {fvf_trans_fig}")

    # Print top differences
    print("\nTop fed-vs-fasted differences (Mann-Whitney U):")
    sorted_stats = stats_df.sort_values("p").head(10)
    print(sorted_stats.to_string(index=False))


if __name__ == "__main__":
    main()
