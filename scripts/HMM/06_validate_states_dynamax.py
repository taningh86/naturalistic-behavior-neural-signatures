"""06 (dynamax) — State validation: posterior-weighted profiles, timelines, warnings.

Computes per-state behavioral profiles (posterior-weighted means/probs across
all sessions), produces a consolidated heatmap of states × features, per-session
timeline plots (posterior heatmap + Viterbi ribbon + dig/feed event overlays),
and prints warnings for low-occupancy / redundant / flickering states.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import load_config, ensure_dir, REPO_ROOT


def load_emissions_csvs(params_dir: Path):
    cont = pd.read_csv(params_dir / "emissions_continuous.csv")
    zone = pd.read_csv(params_dir / "emissions_zone.csv")
    ev = pd.read_csv(params_dir / "emissions_events.csv")
    return cont, zone, ev


def main():
    cfg = load_config()
    prepared_dir = REPO_ROOT / cfg["dynamax_dirs"]["prepared"]
    posteriors_dir = REPO_ROOT / cfg["dynamax_dirs"]["posteriors"]
    params_dir = REPO_ROOT / cfg["dynamax_dirs"]["final_params"]
    profiles_csv = REPO_ROOT / cfg["dynamax_dirs"]["state_profiles_csv"]
    dwell_csv = REPO_ROOT / cfg["dynamax_dirs"]["state_dwell_occupancy_csv"]
    profiles_fig = REPO_ROOT / cfg["dynamax_dirs"]["state_profiles_fig"]
    timeline_dir = ensure_dir(REPO_ROOT / cfg["dynamax_dirs"]["timeline_fig_dir"])

    fed = cfg["sessions"]["fed"]
    fasted = cfg["sessions"]["fasted"]
    all_sessions = fed + fasted

    cont_em, zone_em, ev_em = load_emissions_csvs(params_dir)
    N = zone_em.shape[0]
    zone_labels = [c for c in zone_em.columns if c != "state"]
    event_names = [c for c in ev_em.columns if c != "state"]

    # Aggregate posterior-weighted profiles across all sessions
    sum_gamma = np.zeros(N)                            # (K,)
    sum_cont = np.zeros((N, 2))                        # (K, 2) — speed_z, dist_z
    sum_cont_sq = np.zeros((N, 2))
    sum_zone = np.zeros((N, len(zone_labels)))         # (K, K_zone)
    sum_events = np.zeros((N, len(event_names)))       # (K, n_ev)

    # Also aggregate transition counts from Viterbi for dwell
    viterbi_per_session = {}
    posteriors_per_session = {}
    times_per_session = {}
    states_present_total = np.zeros(N)
    bin_size_s = float(cfg["target_bin_ms"]) / 1000.0

    for sn in all_sessions:
        prep = np.load(prepared_dir / f"session_{sn}.npz", allow_pickle=True)
        post_df = pd.read_csv(posteriors_dir / f"session_{sn}.csv")
        gamma = post_df[[f"p_state_{k}" for k in range(N)]].values   # (T, K)
        viterbi = post_df["viterbi"].values
        time_s = post_df["time_s"].values

        x_cont = np.asarray(prep["X_continuous"], dtype=np.float64)
        x_zone = np.asarray(prep["X_zone"], dtype=np.int64)
        x_events = np.asarray(prep["X_events"], dtype=np.float64)
        T = x_cont.shape[0]

        sum_gamma += gamma.sum(axis=0)
        sum_cont += gamma.T @ x_cont
        sum_cont_sq += gamma.T @ (x_cont ** 2)
        zone_oh = np.zeros((T, len(zone_labels)))
        zone_oh[np.arange(T), x_zone] = 1.0
        sum_zone += gamma.T @ zone_oh
        sum_events += gamma.T @ x_events

        for k in np.unique(viterbi):
            states_present_total[k] += 1
        viterbi_per_session[sn] = viterbi
        posteriors_per_session[sn] = gamma
        times_per_session[sn] = time_s

    # Posterior-weighted summaries
    gs = sum_gamma + 1e-12
    cont_mean = sum_cont / gs[:, None]
    cont_var = sum_cont_sq / gs[:, None] - cont_mean ** 2
    cont_std = np.sqrt(np.maximum(cont_var, 0))
    zone_prob = sum_zone / gs[:, None]
    event_prob = sum_events / gs[:, None]
    soft_occupancy = sum_gamma / sum_gamma.sum()

    # Per-state dwell from Viterbi (across sessions, in bins, then convert to s)
    dwell_per_state = {k: [] for k in range(N)}
    for sn, vit in viterbi_per_session.items():
        # find runs
        run_idx = np.flatnonzero(np.diff(vit, prepend=-1, append=-1) != 0)
        # run_idx alternates start/end of each run if we use diff trick
        starts = run_idx[:-1]
        ends = run_idx[1:]
        run_states = vit[starts]
        run_lens = ends - starts
        for k in range(N):
            mask = run_states == k
            dwell_per_state[k].extend(run_lens[mask].tolist())
    mean_dwell_bins = np.array([np.mean(dwell_per_state[k]) if dwell_per_state[k] else 0.0
                                for k in range(N)])
    mean_dwell_s = mean_dwell_bins * bin_size_s

    # Save profile + dwell tables
    profile_rows = []
    for k in range(N):
        row = dict(state=k,
                   soft_occupancy=float(soft_occupancy[k]),
                   mean_dwell_bins=float(mean_dwell_bins[k]),
                   mean_dwell_s=float(mean_dwell_s[k]),
                   speed_z_mean=float(cont_mean[k, 0]),
                   speed_z_std=float(cont_std[k, 0]),
                   dist_z_mean=float(cont_mean[k, 1]),
                   dist_z_std=float(cont_std[k, 1]))
        for zi, zl in enumerate(zone_labels):
            row[f"zone_{zl}_prob"] = float(zone_prob[k, zi])
        for ei, en in enumerate(event_names):
            row[f"event_{en}_prob"] = float(event_prob[k, ei])
        profile_rows.append(row)
    pd.DataFrame(profile_rows).to_csv(profiles_csv, index=False)
    print(f"State profiles → {profiles_csv}")

    pd.DataFrame({"state": np.arange(N),
                  "soft_occupancy": soft_occupancy,
                  "mean_dwell_bins": mean_dwell_bins,
                  "mean_dwell_s": mean_dwell_s}).to_csv(dwell_csv, index=False)
    print(f"Dwell/occupancy → {dwell_csv}")

    # Build state × feature heatmap
    feat_labels = (
        ["speed_z", "dist_z"]
        + [f"zone:{l}" for l in zone_labels]
        + [f"event:{e}" for e in event_names]
    )
    M = np.column_stack([
        cont_mean[:, 0], cont_mean[:, 1],
        zone_prob, event_prob,
    ])  # (N, 2 + K_zone + n_ev)

    fig, axes = plt.subplots(1, 3, figsize=(2 + 0.45 * len(feat_labels), 0.5 + 0.4 * N),
                              gridspec_kw={"width_ratios": [2, len(zone_labels), len(event_names)]})
    # 1. Continuous
    cont_M = cont_mean.copy()
    vmax_c = np.abs(cont_M).max() + 1e-9
    axes[0].imshow(cont_M, aspect="auto", cmap="RdBu_r", vmin=-vmax_c, vmax=vmax_c)
    axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["speed_z", "dist_z"], rotation=30, ha="right")
    axes[0].set_yticks(np.arange(N)); axes[0].set_yticklabels([f"S{k}" for k in range(N)])
    axes[0].set_title("Continuous (z)")
    for i in range(N):
        for j in range(2):
            axes[0].text(j, i, f"{cont_M[i,j]:.1f}", ha="center", va="center",
                         fontsize=7, color="black" if abs(cont_M[i,j]) < 0.6*vmax_c else "white")

    # 2. Zone probabilities
    axes[1].imshow(zone_prob, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    axes[1].set_xticks(np.arange(len(zone_labels)))
    axes[1].set_xticklabels(zone_labels, rotation=30, ha="right")
    axes[1].set_yticks([])
    axes[1].set_title("Zone P")
    for i in range(N):
        for j in range(len(zone_labels)):
            v = zone_prob[i, j]
            axes[1].text(j, i, f"{v:.2f}", ha="center", va="center",
                         fontsize=7, color="black" if v < 0.6 else "white")

    # 3. Event probabilities
    axes[2].imshow(event_prob, aspect="auto", cmap="Reds", vmin=0, vmax=event_prob.max() + 1e-9)
    axes[2].set_xticks(np.arange(len(event_names)))
    axes[2].set_xticklabels(event_names, rotation=30, ha="right")
    axes[2].set_yticks([])
    axes[2].set_title("Event P")
    for i in range(N):
        for j in range(len(event_names)):
            v = event_prob[i, j]
            axes[2].text(j, i, f"{v:.2f}", ha="center", va="center",
                         fontsize=7,
                         color="black" if v < 0.6 * event_prob.max() else "white")

    fig.suptitle(f"dynamax MixedHMM — N={N} state profiles "
                 f"(posterior-weighted)", y=0.995)
    fig.tight_layout()
    fig.savefig(profiles_fig, dpi=150)
    plt.close(fig)
    print(f"Profile figure → {profiles_fig}")

    # ---- Warnings ----
    print("\n--- State diagnostics ---")
    for k in range(N):
        if soft_occupancy[k] < 0.02:
            print(f"  WARNING low-occupancy state {k}: {soft_occupancy[k]*100:.2f}%")
        if mean_dwell_bins[k] < 2:
            print(f"  WARNING flickering state {k}: mean dwell {mean_dwell_bins[k]:.2f} bins")
    # redundant pairs by cosine similarity on full profile vector
    M_full = np.column_stack([cont_mean, zone_prob, event_prob])
    norms = np.linalg.norm(M_full, axis=1, keepdims=True) + 1e-12
    cos = (M_full / norms) @ (M_full / norms).T
    for i in range(N):
        for j in range(i + 1, N):
            if cos[i, j] > 0.95:
                print(f"  WARNING redundant pair ({i}, {j}): cosine={cos[i,j]:.3f}")

    # ---- Per-session timelines ----
    cmap_states = plt.cm.tab20 if N <= 20 else plt.cm.gist_ncar
    for sn in all_sessions:
        prep = np.load(prepared_dir / f"session_{sn}.npz", allow_pickle=True)
        gamma = posteriors_per_session[sn]   # (T, K)
        vit = viterbi_per_session[sn]
        time_s = times_per_session[sn]
        x_events = np.asarray(prep["X_events"], dtype=np.float64)
        ev_idx_dig = event_names.index("digging_sand")
        ev_idx_feed = event_names.index("feeding")
        dig_mask = x_events[:, ev_idx_dig] > 0.5
        feed_mask = x_events[:, ev_idx_feed] > 0.5
        state = str(prep["state"])

        fig, axes = plt.subplots(2, 1, figsize=(13, 3.4),
                                  gridspec_kw={"height_ratios": [3, 1]},
                                  sharex=True)
        axes[0].imshow(gamma.T, aspect="auto", origin="lower",
                       extent=[time_s[0], time_s[-1], -0.5, N - 0.5],
                       cmap="viridis", interpolation="nearest", vmin=0, vmax=1)
        axes[0].set_ylabel("State (posterior)")
        axes[0].set_title(f"S{sn} ({state}) — N={N}")
        # Viterbi ribbon
        axes[1].imshow(vit[None, :], aspect="auto",
                       extent=[time_s[0], time_s[-1], 0, 1],
                       cmap=cmap_states, vmin=-0.5, vmax=N - 0.5,
                       interpolation="nearest")
        axes[1].set_yticks([])
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Viterbi")
        # Event overlays as tick marks at top of axes[1]
        if dig_mask.any():
            axes[1].scatter(time_s[dig_mask],
                             np.full(dig_mask.sum(), 1.05),
                             marker="|", s=40, color="red", clip_on=False, label="dig")
        if feed_mask.any():
            axes[1].scatter(time_s[feed_mask],
                             np.full(feed_mask.sum(), 1.15),
                             marker="|", s=40, color="orange", clip_on=False, label="feed")
        if dig_mask.any() or feed_mask.any():
            axes[1].legend(loc="upper right", fontsize=8, frameon=False)
        fig.tight_layout()
        out = timeline_dir / f"session_{sn}.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"  S{sn} timeline → {out.name}")


if __name__ == "__main__":
    main()
