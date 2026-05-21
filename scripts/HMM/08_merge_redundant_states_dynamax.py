"""08 (dynamax) — Iterative merging of near-redundant states.

Uses cosine similarity on the concatenated profile vector
[continuous means, zone P, event P]. Merges the most-similar pair (above
cfg["merge_cosine_threshold"]) at each iteration, weights by occupancy,
recomputes the similarity matrix, and stops when no pair exceeds threshold.

Updates posteriors (column sum), Viterbi (relabel), and the empirical
transition matrix (rows weighted-averaged, columns summed, then row-renormed).

Regenerates state-profile heatmap, per-session timelines, and fed-vs-fasted
comparison on the merged states.

CLI:
  --threshold T   override cfg["merge_cosine_threshold"]
"""
from pathlib import Path
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import load_config, ensure_dir, REPO_ROOT


# -----------------------------------------------------------------------------
# Profile vector + cosine similarity
# -----------------------------------------------------------------------------
def build_profile_matrix(profile_df: pd.DataFrame, zone_labels, event_names):
    """Return (K, F) profile matrix and the column names."""
    cont = np.column_stack([
        profile_df["speed_z_mean"].values,
        profile_df["dist_z_mean"].values,
    ])
    zone = np.column_stack([profile_df[f"zone_{z}_prob"].values for z in zone_labels])
    ev = np.column_stack([profile_df[f"event_{e}_prob"].values for e in event_names])
    M = np.column_stack([cont, zone, ev])
    cols = (["speed_z", "dist_z"]
            + [f"zone:{z}" for z in zone_labels]
            + [f"event:{e}" for e in event_names])
    return M, cols


def cosine_similarity_matrix(M: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    Mn = M / norms
    S = Mn @ Mn.T
    np.fill_diagonal(S, -np.inf)  # exclude self-pairs from "max"
    return S


def find_most_similar_pair(S: np.ndarray):
    i, j = np.unravel_index(np.argmax(S), S.shape)
    return int(min(i, j)), int(max(i, j)), float(S[i, j])


# -----------------------------------------------------------------------------
# Merging
# -----------------------------------------------------------------------------
def merge_state_pair(profile_M, occ, A, posteriors, viterbi_dict, i, j):
    """
    Merge state j into state i (i < j). Returns updated copies of
    profile_M, occ, A, posteriors (per-session arrays), viterbi_dict.

    - Profile: occupancy-weighted mean of rows i and j.
    - Occupancy: sum.
    - Transition matrix:
        A_new[i, k]   = (occ[i]*A[i,k] + occ[j]*A[j,k]) / (occ[i]+occ[j]) for k != i, j
        A_new[k, i]   = A[k, i] + A[k, j]                                    for k != i, j
        A_new[i, i]   = (occ[i]*(A[i,i]+A[i,j]) + occ[j]*(A[j,i]+A[j,j])) / (occ[i]+occ[j])
        Then row-renormalize as a safety pass.
    - Posteriors: column i = col i + col j.
    - Viterbi: any v == j is relabelled to i. Then drop index j and shift labels > j down by 1.
    """
    K = profile_M.shape[0]
    wi, wj = occ[i], occ[j]
    wsum = wi + wj
    if wsum < 1e-12:
        wsum = 1.0
        wi = wj = 0.5

    # --- new profile row ---
    new_profile = (wi * profile_M[i] + wj * profile_M[j]) / wsum

    # --- new transition row/col entries for merged state (still at index i pre-deletion) ---
    A_new = A.copy()
    # rows from i (other targets k): weighted average of rows i and j
    for k in range(K):
        if k in (i, j):
            continue
        A_new[i, k] = (wi * A[i, k] + wj * A[j, k]) / wsum
    # columns into i (other sources k): sum
    for k in range(K):
        if k in (i, j):
            continue
        A_new[k, i] = A[k, i] + A[k, j]
    # self-transition merged
    A_new[i, i] = (wi * (A[i, i] + A[i, j]) + wj * (A[j, i] + A[j, j])) / wsum

    # delete row j and col j
    keep = np.array([k for k in range(K) if k != j])
    A_new = A_new[np.ix_(keep, keep)]
    # row-renormalize (numerical safety)
    row_sums = A_new.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    A_new = A_new / row_sums

    # --- profile + occupancy ---
    profile_new = profile_M.copy()
    profile_new[i] = new_profile
    profile_new = profile_new[keep]
    occ_new = occ.copy()
    occ_new[i] = wsum
    occ_new = occ_new[keep]

    # --- per-session posteriors (T, K) ---
    posteriors_new = {}
    for sn, gamma in posteriors.items():
        gnew = gamma.copy()
        gnew[:, i] = gamma[:, i] + gamma[:, j]
        gnew = np.delete(gnew, j, axis=1)
        posteriors_new[sn] = gnew

    # --- per-session Viterbi: relabel j→i, then shift labels > j down by 1 ---
    viterbi_new = {}
    for sn, v in viterbi_dict.items():
        vnew = v.copy()
        vnew[vnew == j] = i
        vnew[vnew > j] -= 1
        viterbi_new[sn] = vnew

    return profile_new, occ_new, A_new, posteriors_new, viterbi_new


# -----------------------------------------------------------------------------
# Recompute per-session profile/dwell/occupancy from updated posteriors+viterbi
# -----------------------------------------------------------------------------
def mean_dwell_per_state(viterbi: np.ndarray, K: int) -> np.ndarray:
    out = np.zeros(K)
    if len(viterbi) == 0:
        return out
    run_idx = np.flatnonzero(np.diff(viterbi, prepend=-1, append=-1) != 0)
    starts = run_idx[:-1]
    ends = run_idx[1:]
    run_states = viterbi[starts]
    run_lens = ends - starts
    for k in range(K):
        m = run_states == k
        if m.any():
            out[k] = run_lens[m].mean()
    return out


def empirical_transition_matrix(viterbi: np.ndarray, K: int) -> np.ndarray:
    A = np.zeros((K, K))
    for t in range(len(viterbi) - 1):
        A[viterbi[t], viterbi[t + 1]] += 1
    rs = A.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return A / rs


# -----------------------------------------------------------------------------
# Plot helpers (mirror script 06/07)
# -----------------------------------------------------------------------------
def plot_state_profile_heatmap(profile_M, occ, dwell_s, zone_labels, event_names,
                                out_path):
    K = profile_M.shape[0]
    cont_M = profile_M[:, :2]
    n_z = len(zone_labels)
    n_e = len(event_names)
    zone_prob = profile_M[:, 2:2 + n_z]
    event_prob = profile_M[:, 2 + n_z:]

    fig, axes = plt.subplots(
        1, 3, figsize=(2 + 0.45 * (2 + n_z + n_e), 0.5 + 0.4 * K),
        gridspec_kw={"width_ratios": [2, n_z, n_e]},
    )
    vmax_c = np.abs(cont_M).max() + 1e-9
    axes[0].imshow(cont_M, aspect="auto", cmap="RdBu_r", vmin=-vmax_c, vmax=vmax_c)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["speed_z", "dist_z"], rotation=30, ha="right")
    axes[0].set_yticks(np.arange(K))
    axes[0].set_yticklabels([f"S{k} ({occ[k]*100:.1f}% / {dwell_s[k]:.1f}s)"
                              for k in range(K)])
    axes[0].set_title("Continuous (z)")
    for i in range(K):
        for j in range(2):
            axes[0].text(j, i, f"{cont_M[i,j]:.1f}", ha="center", va="center",
                         fontsize=7,
                         color="black" if abs(cont_M[i, j]) < 0.6 * vmax_c else "white")

    axes[1].imshow(zone_prob, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    axes[1].set_xticks(np.arange(n_z))
    axes[1].set_xticklabels(zone_labels, rotation=30, ha="right")
    axes[1].set_yticks([])
    axes[1].set_title("Zone P")
    for i in range(K):
        for j in range(n_z):
            v = zone_prob[i, j]
            axes[1].text(j, i, f"{v:.2f}", ha="center", va="center",
                         fontsize=7, color="black" if v < 0.6 else "white")

    em_max = event_prob.max() + 1e-9
    axes[2].imshow(event_prob, aspect="auto", cmap="Reds", vmin=0, vmax=em_max)
    axes[2].set_xticks(np.arange(n_e))
    axes[2].set_xticklabels(event_names, rotation=30, ha="right")
    axes[2].set_yticks([])
    axes[2].set_title("Event P")
    for i in range(K):
        for j in range(n_e):
            v = event_prob[i, j]
            axes[2].text(j, i, f"{v:.2f}", ha="center", va="center",
                         fontsize=7, color="black" if v < 0.6 * em_max else "white")
    fig.suptitle(f"Merged states (K={K}) — posterior-weighted profiles", y=0.99)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_session_timeline(sn, state, time_s, gamma, viterbi, x_events, event_names,
                           out_path, K):
    ev_idx_dig = event_names.index("digging_sand")
    ev_idx_feed = event_names.index("feeding")
    dig_mask = x_events[:, ev_idx_dig] > 0.5
    feed_mask = x_events[:, ev_idx_feed] > 0.5

    cmap_states = plt.cm.tab20 if K <= 20 else plt.cm.gist_ncar
    fig, axes = plt.subplots(
        2, 1, figsize=(13, 3.4),
        gridspec_kw={"height_ratios": [3, 1]}, sharex=True,
    )
    axes[0].imshow(gamma.T, aspect="auto", origin="lower",
                    extent=[time_s[0], time_s[-1], -0.5, K - 0.5],
                    cmap="viridis", interpolation="nearest", vmin=0, vmax=1)
    axes[0].set_ylabel("State (posterior)")
    axes[0].set_title(f"S{sn} ({state}) — merged K={K}")
    axes[1].imshow(viterbi[None, :], aspect="auto",
                    extent=[time_s[0], time_s[-1], 0, 1],
                    cmap=cmap_states, vmin=-0.5, vmax=K - 0.5,
                    interpolation="nearest")
    axes[1].set_yticks([])
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Viterbi")
    if dig_mask.any():
        axes[1].scatter(time_s[dig_mask], np.full(dig_mask.sum(), 1.05),
                         marker="|", s=40, color="red", clip_on=False, label="dig")
    if feed_mask.any():
        axes[1].scatter(time_s[feed_mask], np.full(feed_mask.sum(), 1.15),
                         marker="|", s=40, color="orange", clip_on=False, label="feed")
    if dig_mask.any() or feed_mask.any():
        axes[1].legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_fed_vs_fasted_bars(per_session_df, stats_df, K, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(max(8, 0.7 * K), 7), sharex=True)
    states = np.arange(K)
    width = 0.35
    for ax, metric, ylab in [
        (axes[0], "soft_occupancy", "Soft occupancy"),
        (axes[1], "mean_dwell_s", "Mean dwell (s)"),
    ]:
        fed_per = [per_session_df.loc[
            (per_session_df.merged_state == k) & (per_session_df.state_label == "fed"),
            metric].values for k in range(K)]
        fas_per = [per_session_df.loc[
            (per_session_df.merged_state == k) & (per_session_df.state_label == "fasted"),
            metric].values for k in range(K)]
        fed_means = [np.mean(v) if len(v) else 0 for v in fed_per]
        fas_means = [np.mean(v) if len(v) else 0 for v in fas_per]
        fed_se = [np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0 for v in fed_per]
        fas_se = [np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0 for v in fas_per]
        ax.bar(states - width / 2, fed_means, width=width, yerr=fed_se,
                color="#4477aa", alpha=0.75, label="fed", capsize=2.5)
        ax.bar(states + width / 2, fas_means, width=width, yerr=fas_se,
                color="#cc6677", alpha=0.75, label="fasted", capsize=2.5)
        for k in range(K):
            ax.scatter(np.full(len(fed_per[k]), k - width / 2), fed_per[k],
                        color="#1f4060", s=12, zorder=3)
            ax.scatter(np.full(len(fas_per[k]), k + width / 2), fas_per[k],
                        color="#7a2c39", s=12, zorder=3)
        for k in range(K):
            row = stats_df[(stats_df.state == k) & (stats_df.metric == metric)]
            if len(row):
                p = row.iloc[0]["p"]
                star = "*" if p < 0.05 else "." if p < 0.1 else ""
                if star:
                    y = max(fed_means[k] + fed_se[k], fas_means[k] + fas_se[k]) * 1.05
                    ax.text(k, y, star, ha="center", fontsize=12)
        ax.set_ylabel(ylab)
        ax.set_xticks(states)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    axes[1].set_xlabel("Merged HMM state")
    fig.suptitle(f"dynamax MixedHMM (merged) — fed vs fasted (K={K})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config()
    threshold = args.threshold if args.threshold is not None \
        else float(cfg["merge_cosine_threshold"])

    # paths
    profiles_csv = REPO_ROOT / cfg["dynamax_dirs"]["state_profiles_csv"]
    posteriors_dir = REPO_ROOT / cfg["dynamax_dirs"]["posteriors"]
    final_npz = REPO_ROOT / cfg["dynamax_dirs"]["final_model"]
    final_params_dir = REPO_ROOT / cfg["dynamax_dirs"]["final_params"]
    prepared_dir = REPO_ROOT / cfg["dynamax_dirs"]["prepared"]

    out_params_dir = ensure_dir(REPO_ROOT / cfg["merge_dirs"]["params"])
    out_post_dir = ensure_dir(REPO_ROOT / cfg["merge_dirs"]["posteriors"])
    profiles_out = REPO_ROOT / cfg["merge_dirs"]["state_profiles_csv"]
    log_out = REPO_ROOT / cfg["merge_dirs"]["merge_log_csv"]
    fvf_csv = REPO_ROOT / cfg["merge_dirs"]["fed_vs_fasted_csv"]
    fvf_stats_csv = REPO_ROOT / cfg["merge_dirs"]["fed_vs_fasted_stats_csv"]
    profiles_fig = REPO_ROOT / cfg["merge_dirs"]["state_profiles_fig"]
    timeline_dir = ensure_dir(REPO_ROOT / cfg["merge_dirs"]["timeline_fig_dir"])
    fvf_fig = REPO_ROOT / cfg["merge_dirs"]["fed_vs_fasted_fig"]

    # load model + posteriors + profile
    z = np.load(final_npz, allow_pickle=True)
    A = np.asarray(z["A"], dtype=np.float64)
    K_orig = int(z["K"])
    profiles_df = pd.read_csv(profiles_csv)
    bin_size_s = float(cfg["target_bin_ms"]) / 1000.0

    # load zone/event labels via emissions CSVs
    zone_em = pd.read_csv(final_params_dir / "emissions_zone.csv")
    ev_em = pd.read_csv(final_params_dir / "emissions_events.csv")
    zone_labels = [c for c in zone_em.columns if c != "state"]
    event_names = [c for c in ev_em.columns if c != "state"]

    profile_M, profile_cols = build_profile_matrix(profiles_df, zone_labels, event_names)
    occ = profiles_df["soft_occupancy"].values.astype(np.float64)

    fed = cfg["sessions"]["fed"]
    fasted = cfg["sessions"]["fasted"]
    all_sessions = fed + fasted

    posteriors = {}
    viterbi_dict = {}
    times = {}
    state_session = {}
    for sn in all_sessions:
        df = pd.read_csv(posteriors_dir / f"session_{sn}.csv")
        posteriors[sn] = df[[f"p_state_{k}" for k in range(K_orig)]].values.astype(np.float64)
        viterbi_dict[sn] = df["viterbi"].values.astype(np.int64)
        times[sn] = df["time_s"].values.astype(np.float64)

    # state_label per session
    state_label = {sn: ("fed" if sn in fed else "fasted") for sn in all_sessions}

    # iteratively merge
    print(f"Initial K = {K_orig}, threshold = {threshold}")
    S = cosine_similarity_matrix(profile_M)
    merge_log_rows = []
    # cluster_sets: list of sets, one per current state, naming the original states it contains
    cluster_sets = [{k} for k in range(K_orig)]
    iter_idx = 0

    # Print all initial pairs above threshold
    print("Initial pairs above threshold (sorted):")
    upper = np.triu_indices_from(S, k=1)
    pairs_init = [(i, j, S[i, j]) for i, j in zip(*upper) if S[i, j] > threshold]
    pairs_init.sort(key=lambda x: -x[2])
    for (i, j, c) in pairs_init:
        print(f"  ({i:>2}, {j:>2})  cos={c:.4f}")
    if not pairs_init:
        print("  (none — no merging will happen)")

    while True:
        i, j, max_cos = find_most_similar_pair(S)
        if max_cos < threshold:
            break
        iter_idx += 1
        print(f"  iter {iter_idx}: merge ({i}, {j})  cos={max_cos:.4f}  "
              f"orig={cluster_sets[i] | cluster_sets[j]}")
        merge_log_rows.append(dict(
            iter=iter_idx,
            merged_state_a=i,
            merged_state_b=j,
            cosine=float(max_cos),
            cluster_after_merge=sorted(cluster_sets[i] | cluster_sets[j]),
        ))
        profile_M, occ, A, posteriors, viterbi_dict = merge_state_pair(
            profile_M, occ, A, posteriors, viterbi_dict, i, j,
        )
        # update cluster_sets
        new_cluster_sets = []
        for k_idx in range(len(cluster_sets)):
            if k_idx == i:
                new_cluster_sets.append(cluster_sets[i] | cluster_sets[j])
            elif k_idx == j:
                continue
            else:
                new_cluster_sets.append(cluster_sets[k_idx])
        cluster_sets = new_cluster_sets
        S = cosine_similarity_matrix(profile_M)

    K_final = profile_M.shape[0]
    print(f"\nFinal K = {K_final} (merged {K_orig - K_final} pairs)")

    # ---- Recompute mean dwell using merged Viterbi (per-session per-state) ----
    dwell_per_state_pooled = np.zeros(K_final)
    occ_per_state_pooled = np.zeros(K_final)
    total_T = 0
    for sn in all_sessions:
        v = viterbi_dict[sn]
        T = len(v)
        total_T += T
        dwell = mean_dwell_per_state(v, K_final) * bin_size_s
        # weight by session length when pooling — simple average gives equal weight
        # Use simple mean of session-level dwells when available
        # Here we compute pooled via concatenated Viterbi later
    # Pooled mean dwell across all sessions (concatenated):
    big_v = np.concatenate([viterbi_dict[sn] for sn in all_sessions])
    big_dwell = mean_dwell_per_state(big_v, K_final) * bin_size_s

    # Pooled soft occupancy = weighted mean of per-session posteriors
    big_gamma = np.concatenate([posteriors[sn] for sn in all_sessions], axis=0)
    soft_occ_pooled = big_gamma.mean(axis=0)

    # ---- Save merge log ----
    pd.DataFrame(merge_log_rows).to_csv(log_out, index=False)
    print(f"Merge log → {log_out}")

    # ---- Build the merged state profile table ----
    # Recompute per-state continuous, zone, event from posterior weights
    cont_pool = np.concatenate(
        [np.load(prepared_dir / f"session_{sn}.npz", allow_pickle=True)["X_continuous"]
         for sn in all_sessions], axis=0,
    )
    zone_pool = np.concatenate(
        [np.load(prepared_dir / f"session_{sn}.npz", allow_pickle=True)["X_zone"]
         for sn in all_sessions], axis=0,
    ).astype(np.int64)
    ev_pool = np.concatenate(
        [np.load(prepared_dir / f"session_{sn}.npz", allow_pickle=True)["X_events"]
         for sn in all_sessions], axis=0,
    ).astype(np.float64)

    sum_gamma = big_gamma.sum(axis=0) + 1e-12
    cont_mean = (big_gamma.T @ cont_pool) / sum_gamma[:, None]
    cont_var = (big_gamma.T @ (cont_pool ** 2)) / sum_gamma[:, None] - cont_mean ** 2
    cont_std = np.sqrt(np.maximum(cont_var, 0))
    n_z = len(zone_labels)
    zone_oh = np.zeros((zone_pool.shape[0], n_z))
    zone_oh[np.arange(zone_pool.shape[0]), zone_pool] = 1.0
    zone_prob = (big_gamma.T @ zone_oh) / sum_gamma[:, None]
    event_prob = (big_gamma.T @ ev_pool) / sum_gamma[:, None]

    rows = []
    for k in range(K_final):
        row = dict(
            state=k,
            soft_occupancy=float(soft_occ_pooled[k]),
            mean_dwell_bins=float(big_dwell[k] / bin_size_s),
            mean_dwell_s=float(big_dwell[k]),
            speed_z_mean=float(cont_mean[k, 0]),
            speed_z_std=float(cont_std[k, 0]),
            dist_z_mean=float(cont_mean[k, 1]),
            dist_z_std=float(cont_std[k, 1]),
        )
        for zi, zl in enumerate(zone_labels):
            row[f"zone_{zl}_prob"] = float(zone_prob[k, zi])
        for ei, en in enumerate(event_names):
            row[f"event_{en}_prob"] = float(event_prob[k, ei])
        rows.append(row)
    merged_profile_df = pd.DataFrame(rows)
    merged_profile_df.to_csv(profiles_out, index=False)
    print(f"Merged state profiles → {profiles_out}")

    # ---- Save merged params (transition matrix + emissions) ----
    pd.DataFrame(A,
                 index=[f"from_state_{i}" for i in range(K_final)],
                 columns=[f"to_state_{j}" for j in range(K_final)]).to_csv(
        out_params_dir / "transition_matrix.csv"
    )
    pd.DataFrame(zone_prob, columns=zone_labels).assign(state=np.arange(K_final))[
        ["state"] + zone_labels
    ].to_csv(out_params_dir / "emissions_zone.csv", index=False)
    pd.DataFrame(event_prob, columns=event_names).assign(state=np.arange(K_final))[
        ["state"] + event_names
    ].to_csv(out_params_dir / "emissions_events.csv", index=False)
    cont_rows = []
    for k in range(K_final):
        for d, nm in enumerate(["speed_z", "distance_to_pot_z"]):
            cont_rows.append(dict(state=k, feature=nm,
                                   mu=float(cont_mean[k, d]),
                                   sigma=float(cont_std[k, d])))
    pd.DataFrame(cont_rows).to_csv(out_params_dir / "emissions_continuous.csv", index=False)
    print(f"Merged params → {out_params_dir}")

    # ---- Save merged per-session posteriors ----
    for sn in all_sessions:
        gamma = posteriors[sn]
        v = viterbi_dict[sn]
        time_s = times[sn]
        T = len(v)
        df = pd.DataFrame(gamma, columns=[f"p_state_{k}" for k in range(K_final)])
        df.insert(0, "viterbi", v)
        df.insert(0, "time_s", time_s)
        df.insert(0, "bin", np.arange(T))
        df.to_csv(out_post_dir / f"session_{sn}.csv", index=False)
    print(f"Merged posteriors → {out_post_dir}")

    # ---- Heatmap ----
    plot_state_profile_heatmap(
        np.column_stack([cont_mean, zone_prob, event_prob]),
        soft_occ_pooled, big_dwell, zone_labels, event_names,
        profiles_fig,
    )
    print(f"Profile figure → {profiles_fig}")

    # ---- Per-session timelines ----
    for sn in all_sessions:
        prep = np.load(prepared_dir / f"session_{sn}.npz", allow_pickle=True)
        x_events = np.asarray(prep["X_events"], dtype=np.float64)
        plot_session_timeline(
            sn, state_label[sn], times[sn], posteriors[sn], viterbi_dict[sn],
            x_events, event_names, timeline_dir / f"session_{sn}.png", K_final,
        )
    print(f"Timelines → {timeline_dir}")

    # ---- Fed vs fasted on merged states ----
    fvf_rows = []
    for sn in all_sessions:
        gamma = posteriors[sn]
        v = viterbi_dict[sn]
        T = len(v)
        soft_occ = gamma.mean(axis=0)
        hard_occ = np.bincount(v, minlength=K_final) / T
        dwell_s = mean_dwell_per_state(v, K_final) * bin_size_s
        for k in range(K_final):
            fvf_rows.append(dict(
                session=sn, state_label=state_label[sn], merged_state=k,
                soft_occupancy=float(soft_occ[k]),
                hard_occupancy=float(hard_occ[k]),
                mean_dwell_s=float(dwell_s[k]),
            ))
    fvf_df = pd.DataFrame(fvf_rows)
    fvf_df.to_csv(fvf_csv, index=False)
    print(f"Fed vs fasted → {fvf_csv}")

    # Mann-Whitney
    stats_rows = []
    for k in range(K_final):
        for metric in ("soft_occupancy", "mean_dwell_s"):
            fed_vals = fvf_df.loc[(fvf_df.merged_state == k) &
                                   (fvf_df.state_label == "fed"), metric].values
            fas_vals = fvf_df.loc[(fvf_df.merged_state == k) &
                                   (fvf_df.state_label == "fasted"), metric].values
            try:
                U, p = mannwhitneyu(fed_vals, fas_vals, alternative="two-sided")
            except ValueError:
                U, p = np.nan, np.nan
            stats_rows.append(dict(
                state=k, metric=metric,
                fed_mean=float(np.mean(fed_vals)),
                fasted_mean=float(np.mean(fas_vals)),
                fed_n=len(fed_vals), fasted_n=len(fas_vals),
                U=float(U) if U == U else np.nan,
                p=float(p) if p == p else np.nan,
            ))
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(fvf_stats_csv, index=False)
    print(f"Stats → {fvf_stats_csv}")

    plot_fed_vs_fasted_bars(fvf_df, stats_df, K_final, fvf_fig)
    print(f"Fed vs fasted figure → {fvf_fig}")

    # ---- Summary ----
    print("\n========== SUMMARY ==========")
    print(f"Original N = {K_orig}, final K = {K_final} "
          f"({K_orig - K_final} merges)")
    print(f"Threshold = {threshold}")
    print(f"\nMerges performed:")
    for r in merge_log_rows:
        print(f"  iter {r['iter']}: merged states {r['merged_state_a']} & "
              f"{r['merged_state_b']} (cos={r['cosine']:.3f})  "
              f"→ originals: {r['cluster_after_merge']}")
    print("\nFinal cluster mapping (merged_state → original ssm/dynamax states):")
    for k_new, cluster in enumerate(cluster_sets):
        print(f"  merged S{k_new}: original {sorted(cluster)}")

    print("\nTop 10 fed-vs-fasted differences (merged):")
    print(stats_df.sort_values("p").head(10).to_string(index=False))

    # Save final cluster mapping as a small CSV too
    mapping_rows = []
    for k_new, cluster in enumerate(cluster_sets):
        for orig in sorted(cluster):
            mapping_rows.append(dict(merged_state=k_new, original_state=int(orig)))
    pd.DataFrame(mapping_rows).to_csv(
        out_params_dir / "cluster_mapping.csv", index=False,
    )
    print(f"\nCluster mapping → {out_params_dir / 'cluster_mapping.csv'}")


if __name__ == "__main__":
    main()
