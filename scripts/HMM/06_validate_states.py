"""06 — Validate the fitted HMM states.

Computes per-state behavioral profiles, plots a states×features heatmap, and
draws per-session timeline plots with dig/feeding events overlaid. Prints
warnings for unstable, redundant, or flickering states.

Outputs:
  data/HMM/state_profiles.csv
  data/HMM/state_dwell_occupancy.csv
  figures/HMM/state_profiles.png
  figures/HMM/timelines/session_{N}.png
"""
import pickle
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import load_config, session_list, ensure_dir, REPO_ROOT


def runs(viterbi):
    """Return list of (state, length) for contiguous runs."""
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


def main():
    cfg = load_config()
    prepared_dir = REPO_ROOT / cfg["out_dirs"]["prepared"]
    posteriors_dir = REPO_ROOT / cfg["out_dirs"]["posteriors"]
    bin_s = cfg["target_bin_ms"] / 1000.0

    model_path = REPO_ROOT / cfg["out_dirs"]["final_model"]
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    N = bundle["N"]
    feature_names = bundle["feature_names"]
    print(f"Validating model: N={N}, D={bundle['D']}")

    sess = session_list(cfg)

    # Aggregate posteriors and X across sessions.
    Xs = []
    posts = []
    viterbis = []
    sids = []
    states = []
    for session_num, state in sess:
        prep_p = prepared_dir / f"session_{session_num}.npz"
        post_p = posteriors_dir / f"session_{session_num}.csv"
        if not prep_p.exists() or not post_p.exists():
            print(f"  SKIP S{session_num}: missing inputs")
            continue
        z = np.load(prep_p, allow_pickle=True)
        post_df = pd.read_csv(post_p)
        post = post_df[[f"p_state_{k}" for k in range(N)]].values
        viterbi = post_df["viterbi"].values.astype(np.int64)
        Xs.append(z["X"])
        posts.append(post)
        viterbis.append(viterbi)
        sids.append(session_num)
        states.append(state)

    if not Xs:
        raise SystemExit("No posteriors found; run 05 first.")

    X_all = np.concatenate(Xs, axis=0)
    post_all = np.concatenate(posts, axis=0)
    vit_all = np.concatenate(viterbis, axis=0)
    T_all = X_all.shape[0]

    # Posterior-weighted per-state feature profile (mean & sd).
    weights = post_all  # (T, N)
    w_sum = weights.sum(axis=0)  # (N,)
    means = (weights.T @ X_all) / w_sum[:, None]  # (N, D)
    var = (weights.T @ (X_all ** 2)) / w_sum[:, None] - means ** 2
    var = np.clip(var, 0.0, None)
    sds = np.sqrt(var)

    profile = pd.DataFrame(means, columns=feature_names)
    profile.insert(0, "state", np.arange(N))
    sd_df = pd.DataFrame(sds, columns=[f"{c}_sd" for c in feature_names])
    profile = pd.concat([profile, sd_df], axis=1)

    # Total occupancy (Viterbi & soft).
    occ_soft = w_sum / w_sum.sum()
    occ_hard = np.array([(vit_all == k).mean() for k in range(N)])
    profile.insert(1, "occupancy_soft", occ_soft)
    profile.insert(2, "occupancy_hard", occ_hard)

    # Mean dwell (in bins, from Viterbi).
    runs_all = runs(vit_all)
    dwell_by_state = {k: [] for k in range(N)}
    for st, length in runs_all:
        dwell_by_state[st].append(length)
    mean_dwell_bins = np.array([
        float(np.mean(dwell_by_state[k])) if dwell_by_state[k] else 0.0
        for k in range(N)
    ])
    n_runs = np.array([len(dwell_by_state[k]) for k in range(N)])
    profile.insert(3, "mean_dwell_bins", mean_dwell_bins)
    profile.insert(4, "mean_dwell_s", mean_dwell_bins * bin_s)
    profile.insert(5, "n_runs", n_runs)

    out_csv = REPO_ROOT / "data" / "HMM" / "state_profiles.csv"
    ensure_dir(out_csv.parent)
    profile.to_csv(out_csv, index=False)
    print(f"\nState profiles → {out_csv}")
    print(profile[["state", "occupancy_soft", "occupancy_hard",
                   "mean_dwell_bins", "mean_dwell_s", "n_runs"]].to_string(index=False))

    # Heatmap of state × feature.
    # Z-score each feature column across states for visual clarity.
    M = means.copy()
    col_mu = M.mean(axis=0)
    col_sd = M.std(axis=0)
    col_sd = np.where(col_sd > 0, col_sd, 1.0)
    Mz = (M - col_mu) / col_sd

    fig_h = max(4.5, 0.45 * N + 1.5)
    fig_w = max(7.5, 0.55 * len(feature_names) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cmap = LinearSegmentedColormap.from_list("rb", ["#3a4ea8", "#f7f7f7", "#c0392b"])
    im = ax.imshow(Mz, aspect="auto", cmap=cmap, vmin=-2.5, vmax=2.5)
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(np.arange(N))
    ax.set_yticklabels([f"S{k} ({occ_soft[k]*100:.1f}%)" for k in range(N)])
    ax.set_xlabel("Feature")
    ax.set_ylabel("State")
    ax.set_title("HMM state behavioral profiles (z-scored across states)")
    fig.colorbar(im, ax=ax, label="z (across states)")
    fig.tight_layout()
    fig_path = REPO_ROOT / cfg["out_dirs"]["state_profiles_fig"]
    ensure_dir(fig_path.parent)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved {fig_path}")

    # Save dwell/occupancy table.
    docc = pd.DataFrame({
        "state": np.arange(N),
        "occupancy_soft": occ_soft,
        "occupancy_hard": occ_hard,
        "mean_dwell_bins": mean_dwell_bins,
        "mean_dwell_s": mean_dwell_bins * bin_s,
        "n_runs": n_runs,
    })
    docc_path = REPO_ROOT / "data" / "HMM" / "state_dwell_occupancy.csv"
    docc.to_csv(docc_path, index=False)
    print(f"Saved {docc_path}")

    # Per-session timeline plots.
    timeline_dir = ensure_dir(REPO_ROOT / cfg["out_dirs"]["timeline_fig_dir"])
    # Map feature index for events of interest.
    feat_to_idx = {f: i for i, f in enumerate(feature_names)}
    dig_idx = feat_to_idx.get("event_digging_sand")
    feed_idx = feat_to_idx.get("event_feeding")

    cmap_states = plt.get_cmap("tab20", N)

    for session_num, state, X, post, vit in zip(sids, states, Xs, posts, viterbis):
        T = X.shape[0]
        t = np.arange(T) * bin_s
        fig, axes = plt.subplots(
            2, 1, figsize=(11, 5.5),
            gridspec_kw=dict(height_ratios=[3, 1]),
            sharex=True,
        )
        ax0 = axes[0]
        # Stack of posterior probabilities (rows = states), as image
        im = ax0.imshow(
            post.T, aspect="auto", cmap="viridis",
            extent=[t[0], t[-1], N - 0.5, -0.5],
            vmin=0.0, vmax=1.0,
        )
        ax0.set_yticks(np.arange(N))
        ax0.set_yticklabels([f"S{k}" for k in range(N)])
        ax0.set_ylabel("State posterior")
        ax0.set_title(f"Session {session_num} ({state}) — HMM posteriors + Viterbi")
        fig.colorbar(im, ax=ax0, label="p(state | obs)", pad=0.01)

        # Overlay digging and feeding ticks.
        if dig_idx is not None:
            dig_t = t[X[:, dig_idx] > 0.5]
            for tt in dig_t:
                ax0.axvline(tt, color="orange", alpha=0.35, lw=0.6)
        if feed_idx is not None:
            feed_t = t[X[:, feed_idx] > 0.5]
            for tt in feed_t:
                ax0.axvline(tt, color="red", alpha=0.55, lw=0.8)

        # Bottom: Viterbi as colored ribbon.
        ax1 = axes[1]
        for k in range(N):
            mask = vit == k
            if not mask.any():
                continue
            ax1.fill_between(t, 0, 1, where=mask, color=cmap_states(k),
                             step="pre", linewidth=0)
        ax1.set_ylim(0, 1)
        ax1.set_yticks([])
        ax1.set_ylabel("Viterbi")
        ax1.set_xlabel("Time (s)")

        # Custom legend for state colors + event ticks.
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        handles = [Patch(facecolor=cmap_states(k), label=f"S{k}") for k in range(N)]
        handles += [
            Line2D([0], [0], color="orange", lw=1.5, label="dig"),
            Line2D([0], [0], color="red", lw=1.5, label="feed"),
        ]
        ax1.legend(
            handles=handles, loc="lower center",
            bbox_to_anchor=(0.5, -1.05),
            ncol=min(N + 2, 8), fontsize=8, frameon=False,
        )
        fig.tight_layout()
        out = timeline_dir / f"session_{session_num}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  S{session_num}: {out.name}")

    # Warnings.
    print("\n--- Warnings ---")
    flagged = False

    # 1. Low-occupancy states.
    for k in range(N):
        if occ_soft[k] < 0.02:
            print(f"  [LOW OCC] State {k}: soft occupancy {occ_soft[k]*100:.2f}% (<2%)")
            flagged = True

    # 2. Flickering states (mean dwell < 2 bins).
    for k in range(N):
        if mean_dwell_bins[k] < 2.0:
            print(f"  [FLICKER] State {k}: mean dwell {mean_dwell_bins[k]:.2f} bins "
                  f"({mean_dwell_bins[k]*bin_s:.2f}s) (<2 bins)")
            flagged = True

    # 3. Redundant pairs (high cosine similarity in z-scored mean profile).
    def cos(a, b):
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0

    for i in range(N):
        for j in range(i + 1, N):
            c = cos(Mz[i], Mz[j])
            if c > 0.95:
                print(f"  [REDUNDANT] States {i} & {j}: cosine sim of "
                      f"z-profiles = {c:.3f} (>0.95)")
                flagged = True

    if not flagged:
        print("  (no warnings)")

    print(f"\nDone. Validated N={N} states across {len(sids)} sessions ({T_all} bins).")


if __name__ == "__main__":
    main()
