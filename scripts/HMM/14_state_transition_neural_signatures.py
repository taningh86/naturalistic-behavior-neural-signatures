"""14 — Neural signatures of HMM state transitions (pre-exit analysis).

Tests two parallel hypotheses about neural signatures of upcoming state
transitions, using ONLY bins still in the source state:

  A1 — Generic switch signal:
       Within state i, last K_pre bins of each run (pre-exit) vs earlier bins
       of each run (stay). Different = "I'm about to leave state i" signal,
       irrespective of destination.

  A2 — Pair-specific switch signal:
       Within state i's pre-exit bins, ending in destination j vs ending in
       destinations other than j. Different = "I'm about to enter j
       specifically" signal, encoded while still in i.

Both compared against 100-iteration circular-shift Viterbi shuffles that
preserve run structure & marginal occupancy but break alignment with neural
data. All 6 foraging sessions × 2 regions. Same QC-filtered units as Track B.
"""
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "HMM"))

import spikeinterface.extractors as se
from dp_avalanche_criticality import (
    get_good_units_p0,
    get_good_units_p1_lha,
    load_spike_times_for_region,
)
from _utils import load_config


# ---- Constants ----
HMM_BIN_S = 0.480
NEURAL_BIN_SMALL_S = 0.1
SESSIONS = [4, 6, 8, 12, 14, 16]
K_PRE = 3                              # last 3 bins of each run = pre-exit (~1.4 s)
A1_MIN_STAY = 30
A1_MIN_PRE = 30
A2_MIN_J = 20
A2_MIN_NON_J = 20
N_SHUFFLES = 100
SHUFFLE_MIN_OFFSET = 200
SHUFFLE_MARGIN = 200
SHUFFLE_SEED = 20260508
FDR_ALPHA = 0.05
N_PCS = 5
REPLICATION_THRESHOLD = 3              # sessions passing for cross-session pass


def out_dirs():
    base_out = REPO_ROOT / "data" / "HMM" / "neural_alignment" / "state_transitions"
    base_fig = REPO_ROOT / "figures" / "HMM" / "neural_alignment" / "state_transitions"
    base_out.mkdir(parents=True, exist_ok=True)
    base_fig.mkdir(parents=True, exist_ok=True)
    return base_out, base_fig


# ---- Helpers ----
def fdr_adjust(pvals, q=FDR_ALPHA):
    p = np.asarray(pvals, dtype=np.float64)
    valid = np.isfinite(p)
    out = np.full(p.shape, np.nan)
    if not valid.any():
        return out
    _, p_adj, _, _ = multipletests(p[valid], alpha=q, method="fdr_bh")
    out[valid] = p_adj
    return out


def fdr_pass_count(pvals, q=FDR_ALPHA):
    p = np.asarray(pvals, dtype=np.float64)
    valid = np.isfinite(p)
    if not valid.any():
        return np.zeros(p.shape, dtype=bool)
    rej, _, _, _ = multipletests(p[valid], alpha=q, method="fdr_bh")
    sig = np.zeros(p.shape, dtype=bool)
    sig[valid] = rej
    return sig


def rebin_100ms_to_480ms(rates_100ms, n_hmm_bins):
    n_100ms = rates_100ms.shape[1]
    centers_s = (np.arange(n_100ms) + 0.5) * NEURAL_BIN_SMALL_S
    hmm_idx = np.floor(centers_s / HMM_BIN_S).astype(np.int64)
    valid = (hmm_idx >= 0) & (hmm_idx < n_hmm_bins)
    hmm_idx = hmm_idx[valid]
    rates_100ms = rates_100ms[:, valid]
    n_units = rates_100ms.shape[0]
    out = np.zeros((n_units, n_hmm_bins), dtype=np.float64)
    counts = np.bincount(hmm_idx, minlength=n_hmm_bins)
    for u in range(n_units):
        sums = np.bincount(hmm_idx, weights=rates_100ms[u].astype(np.float64),
                            minlength=n_hmm_bins)
        out[u] = sums / np.maximum(counts, 1)
    return out * (1.0 / NEURAL_BIN_SMALL_S)


def compute_pca_projection(rates):
    mu = rates.mean(axis=1, keepdims=True)
    sig = rates.std(axis=1, keepdims=True) + 1e-9
    z = (rates - mu) / sig
    X = z.T
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pcs = U[:, :N_PCS] * S[:N_PCS]
    return pcs


# ---- Run / bin labeling ----
def label_bins(viterbi, K_pre=K_PRE):
    """Return dict of per-bin arrays:
      bin_group: "stay" | "pre_exit" | "excluded"
      destination: int (state at bin immediately after the run; -1 if undefined)
      run_id: int (-1 if excluded)
      run_position: int 0..L-1 (-1 if excluded)
      run_length: int (-1 if excluded)
    A run is the maximal contiguous stretch with same Viterbi label.
    Runs with L < 2*K_pre are skipped (excluded).
    """
    n = len(viterbi)
    diff = np.diff(viterbi, prepend=-1, append=-1)
    boundaries = np.flatnonzero(diff != 0)
    starts = boundaries[:-1]
    ends = boundaries[1:]

    bin_group = np.array(["excluded"] * n, dtype=object)
    destination = np.full(n, -1, dtype=np.int64)
    run_id = np.full(n, -1, dtype=np.int64)
    run_position = np.full(n, -1, dtype=np.int64)
    run_length = np.full(n, -1, dtype=np.int64)

    for i, (s, e) in enumerate(zip(starts, ends)):
        L = int(e - s)
        run_id[s:e] = i
        run_position[s:e] = np.arange(L)
        run_length[s:e] = L
        if L < 2 * K_pre:
            continue
        bin_group[s : e - K_pre] = "stay"
        bin_group[e - K_pre : e] = "pre_exit"
        if e < n:
            destination[s:e] = int(viterbi[e])
        else:
            destination[s:e] = -1
    return dict(bin_group=bin_group, destination=destination,
                run_id=run_id, run_position=run_position,
                run_length=run_length)


# ---- A1 — generic stay vs pre-exit ----
def a1_per_unit_tests(rates, labels, viterbi, K, region):
    n_units = rates.shape[0]
    bin_group = labels["bin_group"]
    rows = []
    for state in range(K):
        in_state = (viterbi == state)
        stay_idx = np.flatnonzero(in_state & (bin_group == "stay"))
        pre_idx = np.flatnonzero(in_state & (bin_group == "pre_exit"))
        n_s = len(stay_idx); n_p = len(pre_idx)
        if n_s < A1_MIN_STAY or n_p < A1_MIN_PRE:
            continue
        for u in range(n_units):
            fr_s = rates[u, stay_idx]
            fr_p = rates[u, pre_idx]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    _, p = mannwhitneyu(fr_s, fr_p, alternative="two-sided")
                except Exception:
                    p = np.nan
            rows.append(dict(unit=u, region=region, state=state,
                              n_stay=n_s, n_pre_exit=n_p,
                              mean_FR_stay=float(fr_s.mean()),
                              mean_FR_pre_exit=float(fr_p.mean()),
                              p=p))
    df = pd.DataFrame(rows)
    if len(df):
        df["p_fdr"] = fdr_adjust(df["p"].values)
        df["sig_fdr"] = df["p_fdr"] < FDR_ALPHA
    else:
        df["p_fdr"] = []; df["sig_fdr"] = []
    return df


def a1_population_centroids(pcs, labels, viterbi, K):
    bin_group = labels["bin_group"]
    rows = []
    for state in range(K):
        in_state = (viterbi == state)
        stay_idx = np.flatnonzero(in_state & (bin_group == "stay"))
        pre_idx = np.flatnonzero(in_state & (bin_group == "pre_exit"))
        n_s = len(stay_idx); n_p = len(pre_idx)
        if n_s < A1_MIN_STAY or n_p < A1_MIN_PRE:
            continue
        c_stay = pcs[stay_idx, :3].mean(axis=0)
        c_pre = pcs[pre_idx, :3].mean(axis=0)
        d = float(np.linalg.norm(c_stay - c_pre))
        rows.append(dict(state=state, n_stay=n_s, n_pre_exit=n_p,
                          centroid_distance=d))
    return pd.DataFrame(rows)


# ---- A2 — pair-specific destination ----
def a2_per_unit_tests(rates, labels, viterbi, K, region):
    n_units = rates.shape[0]
    bin_group = labels["bin_group"]
    destination = labels["destination"]
    rows = []
    for source in range(K):
        in_pre_source = (viterbi == source) & (bin_group == "pre_exit")
        if not in_pre_source.any():
            continue
        dests = np.unique(destination[in_pre_source])
        dests = [int(d) for d in dests if d >= 0]
        for j in dests:
            to_j = in_pre_source & (destination == j)
            to_other = in_pre_source & (destination != j)
            n_j = int(to_j.sum()); n_o = int(to_other.sum())
            if n_j < A2_MIN_J or n_o < A2_MIN_NON_J:
                continue
            j_idx = np.flatnonzero(to_j)
            o_idx = np.flatnonzero(to_other)
            for u in range(n_units):
                fr_j = rates[u, j_idx]
                fr_o = rates[u, o_idx]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        _, p = mannwhitneyu(fr_j, fr_o, alternative="two-sided")
                    except Exception:
                        p = np.nan
                rows.append(dict(unit=u, region=region,
                                  source=source, destination=j,
                                  n_to_j=n_j, n_to_non_j=n_o,
                                  mean_FR_to_j=float(fr_j.mean()),
                                  mean_FR_to_non_j=float(fr_o.mean()),
                                  p=p))
    df = pd.DataFrame(rows)
    if len(df):
        df["p_fdr"] = fdr_adjust(df["p"].values)
        df["sig_fdr"] = df["p_fdr"] < FDR_ALPHA
    else:
        df["p_fdr"] = []; df["sig_fdr"] = []
    return df


def a2_population_centroids(pcs, labels, viterbi, K):
    bin_group = labels["bin_group"]
    destination = labels["destination"]
    rows = []
    for source in range(K):
        in_pre_source = (viterbi == source) & (bin_group == "pre_exit")
        if not in_pre_source.any():
            continue
        dests = [int(d) for d in np.unique(destination[in_pre_source]) if d >= 0]
        for j in dests:
            to_j = np.flatnonzero(in_pre_source & (destination == j))
            to_o = np.flatnonzero(in_pre_source & (destination != j))
            if len(to_j) < A2_MIN_J or len(to_o) < A2_MIN_NON_J:
                continue
            c_j = pcs[to_j, :3].mean(axis=0)
            c_o = pcs[to_o, :3].mean(axis=0)
            rows.append(dict(source=source, destination=j,
                              n_to_j=int(len(to_j)), n_to_non_j=int(len(to_o)),
                              centroid_distance=float(np.linalg.norm(c_j - c_o))))
    return pd.DataFrame(rows)


# ---- Shuffle helpers (lightweight — count sig units only) ----
def a1_shuffle_kernel(rates, viterbi, K):
    """Return per-state count of FDR-sig units."""
    labels = label_bins(viterbi)
    df = a1_per_unit_tests(rates, labels, viterbi, K, region="x")
    if not len(df):
        return {}
    sig = df[df["sig_fdr"]]
    counts = sig.groupby("state").size().to_dict()
    return counts


def a2_shuffle_kernel(rates, viterbi, K):
    """Return per-(source,destination) count of FDR-sig units."""
    labels = label_bins(viterbi)
    df = a2_per_unit_tests(rates, labels, viterbi, K, region="x")
    if not len(df):
        return {}
    sig = df[df["sig_fdr"]]
    counts = sig.groupby(["source", "destination"]).size().to_dict()
    return counts


# ---- Per-session orchestrator ----
def load_session(sn, cfg, paths_data):
    s_paths = (paths_data["double_probe"]["coordinates_1"]["mouse01"]
                ["sessions"][f"session_{sn}"])
    aca_sorted = Path(s_paths["probe_0_aca"]["sorted"])
    lha_sorted = Path(s_paths["probe_1_lha_rsp"]["sorted"])
    aca_uids = [int(u) for u in get_good_units_p0(aca_sorted)]
    lha_uids = [int(u) for u in get_good_units_p1_lha(lha_sorted)]
    aca_sorting = se.read_kilosort(aca_sorted)
    lha_sorting = se.read_kilosort(lha_sorted)
    aca_uids = [u for u in aca_uids if u in set(aca_sorting.get_unit_ids())]
    lha_uids = [u for u in lha_uids if u in set(lha_sorting.get_unit_ids())]
    aca_spikes = load_spike_times_for_region(aca_sorting, aca_uids)
    lha_spikes = load_spike_times_for_region(lha_sorting, lha_uids)

    binned = np.load(REPO_ROOT / cfg["out_dirs"]["binned"]
                      / f"session_{sn}.npz", allow_pickle=True)
    trial_time = np.asarray(binned["trial_time"], dtype=np.float64)
    n_hmm = len(trial_time)
    duration_s = float(trial_time[-1] + HMM_BIN_S)
    n_100ms = int(np.ceil(duration_s / NEURAL_BIN_SMALL_S))
    edges_100ms = np.arange(n_100ms + 1) * NEURAL_BIN_SMALL_S

    aca_uid_list = sorted(aca_spikes.keys())
    lha_uid_list = sorted(lha_spikes.keys())
    n_aca = len(aca_uid_list); n_lha = len(lha_uid_list)
    aca_100 = np.zeros((n_aca, n_100ms))
    lha_100 = np.zeros((n_lha, n_100ms))
    for i, uid in enumerate(aca_uid_list):
        aca_100[i] = np.histogram(aca_spikes[uid], edges_100ms)[0]
    for i, uid in enumerate(lha_uid_list):
        lha_100[i] = np.histogram(lha_spikes[uid], edges_100ms)[0]
    aca_rates = rebin_100ms_to_480ms(aca_100, n_hmm)
    lha_rates = rebin_100ms_to_480ms(lha_100, n_hmm)

    post = pd.read_csv(REPO_ROOT / cfg["merge_dirs"]["posteriors"]
                        / f"session_{sn}.csv")
    viterbi = post["viterbi"].values.astype(np.int64)
    K = max(int(viterbi.max()) + 1,
            sum(1 for c in post.columns if c.startswith("p_state_")))
    n = min(n_hmm, len(viterbi))
    aca_rates = aca_rates[:, :n]; lha_rates = lha_rates[:, :n]
    viterbi = viterbi[:n]

    history = pd.read_csv(REPO_ROOT / cfg["commitment_dirs"]["out"]
                           / "sampling_history.csv")
    s_hist = history[history.session == sn].iloc[0]
    return dict(sn=sn, K=K, n_bins=n,
                 aca_rates=aca_rates, lha_rates=lha_rates,
                 viterbi=viterbi,
                 metabolic_state=s_hist["state"], food_pot=int(s_hist["food_pot"])
                  if not pd.isna(s_hist["food_pot"]) else None,
                 n_aca=n_aca, n_lha=n_lha)


def run_session(session, base_out_dir, base_fig_dir, rng):
    sn = session["sn"]; K = session["K"]
    out_dir = base_out_dir / f"session_{sn}"
    fig_dir = base_fig_dir / f"session_{sn}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    aca_rates = session["aca_rates"]; lha_rates = session["lha_rates"]
    viterbi = session["viterbi"]; n = session["n_bins"]
    n_aca = session["n_aca"]; n_lha = session["n_lha"]

    print(f"\n--- S{sn} ({session['metabolic_state']}) "
          f"K={K} bins={n} ACA={n_aca} LHA={n_lha} ---", flush=True)
    t0 = time.time()

    # ---- Bin labels ----
    labels = label_bins(viterbi)
    bg_counts = pd.Series(labels["bin_group"]).value_counts().to_dict()
    print(f"  Bin groups: {bg_counts}", flush=True)

    # Save per-bin labels
    df_lab = pd.DataFrame({
        "bin": np.arange(n),
        "time_s": np.arange(n) * HMM_BIN_S,
        "viterbi": viterbi,
        "run_id": labels["run_id"],
        "run_position": labels["run_position"],
        "run_length": labels["run_length"],
        "bin_group": labels["bin_group"],
        "destination_state": labels["destination"],
    })
    df_lab.to_csv(out_dir / f"session_{sn}_bin_labels.csv", index=False)

    # ---- Observed A1 ----
    pcs_aca = compute_pca_projection(aca_rates)
    pcs_lha = compute_pca_projection(lha_rates)

    df_a1_aca = a1_per_unit_tests(aca_rates, labels, viterbi, K, "ACA")
    df_a1_lha = a1_per_unit_tests(lha_rates, labels, viterbi, K, "LHA")
    df_a1 = pd.concat([df_a1_aca, df_a1_lha], ignore_index=True)
    df_a1.to_csv(out_dir / "A1_generic_switch_per_unit.csv", index=False)
    df_a1c_aca = a1_population_centroids(pcs_aca, labels, viterbi, K)
    df_a1c_lha = a1_population_centroids(pcs_lha, labels, viterbi, K)
    df_a1c_aca["region"] = "ACA"; df_a1c_lha["region"] = "LHA"
    df_a1c = pd.concat([df_a1c_aca, df_a1c_lha], ignore_index=True)
    df_a1c.to_csv(out_dir / "A1_centroid_distances.csv", index=False)
    print(f"  A1 done. Tested {len(df_a1)} (unit, state) pairs; "
          f"{int(df_a1['sig_fdr'].sum()) if len(df_a1) else 0} FDR-sig "
          f"[{time.time()-t0:.0f}s]", flush=True)

    # ---- Observed A2 ----
    df_a2_aca = a2_per_unit_tests(aca_rates, labels, viterbi, K, "ACA")
    df_a2_lha = a2_per_unit_tests(lha_rates, labels, viterbi, K, "LHA")
    df_a2 = pd.concat([df_a2_aca, df_a2_lha], ignore_index=True)
    df_a2.to_csv(out_dir / "A2_pair_specific_per_unit.csv", index=False)
    df_a2c_aca = a2_population_centroids(pcs_aca, labels, viterbi, K)
    df_a2c_lha = a2_population_centroids(pcs_lha, labels, viterbi, K)
    df_a2c_aca["region"] = "ACA"; df_a2c_lha["region"] = "LHA"
    df_a2c = pd.concat([df_a2c_aca, df_a2c_lha], ignore_index=True)
    df_a2c.to_csv(out_dir / "A2_centroid_distances.csv", index=False)
    print(f"  A2 done. Tested {len(df_a2)} (unit, src, dst) triples; "
          f"{int(df_a2['sig_fdr'].sum()) if len(df_a2) else 0} FDR-sig "
          f"[{time.time()-t0:.0f}s]", flush=True)

    # ---- Per-state significant unit counts plot (A1) ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, df_r, region, n_units in [
        (axes[0], df_a1[df_a1.region == "ACA"], "ACA", n_aca),
        (axes[1], df_a1[df_a1.region == "LHA"], "LHA", n_lha),
    ]:
        if len(df_r):
            sig_per_state = df_r[df_r["sig_fdr"]].groupby("state").size()
        else:
            sig_per_state = pd.Series(dtype=int)
        states = sorted(set(df_r["state"]) if len(df_r) else range(K))
        counts = [int(sig_per_state.get(s, 0)) for s in states]
        ax.bar(states, counts, color="#cc4444")
        ax.set_xticks(states)
        ax.set_xticklabels([f"S{s}" for s in states])
        ax.set_xlabel("State"); ax.set_ylabel("# FDR-sig units")
        ax.set_title(f"{region}: A1 generic switch (n_units={n_units})")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"S{sn}: A1 — generic stay vs pre-exit, FDR-sig unit counts", y=1.0)
    fig.tight_layout()
    fig.savefig(fig_dir / "A1_per_state_significant_unit_counts.png", dpi=130)
    plt.close(fig)

    # ---- Per (source, destination) significance heatmap (A2) ----
    if len(df_a2) == 0:
        # Guarantee column presence for downstream guard
        df_a2 = pd.DataFrame(columns=["unit", "region", "source", "destination",
                                         "n_to_j", "n_to_non_j",
                                         "mean_FR_to_j", "mean_FR_to_non_j",
                                         "p", "p_fdr", "sig_fdr"])
    for region in ("ACA", "LHA"):
        sub = df_a2[df_a2.region == region]
        if not len(sub):
            continue
        sig = sub[sub["sig_fdr"]]
        mat = np.zeros((K, K), dtype=int)
        for _, r in sig.groupby(["source", "destination"]).size().reset_index(name="n").iterrows():
            mat[int(r["source"]), int(r["destination"])] = int(r["n"])
        fig, ax = plt.subplots(figsize=(0.5 * K + 2, 0.5 * K + 1.5))
        im = ax.imshow(mat, aspect="auto", cmap="Reds")
        ax.set_xticks(range(K)); ax.set_xticklabels([f"S{j}" for j in range(K)])
        ax.set_yticks(range(K)); ax.set_yticklabels([f"S{i}" for i in range(K)])
        ax.set_xlabel("Destination"); ax.set_ylabel("Source")
        plt.colorbar(im, ax=ax, label="# FDR-sig units")
        ax.set_title(f"S{sn} {region}: A2 destination-specific (FDR-sig units)")
        fig.tight_layout()
        fig.savefig(fig_dir / f"A2_destination_significance_heatmap_{region}.png",
                     dpi=130)
        plt.close(fig)

    # ---- Shuffles ----
    print(f"  Running {N_SHUFFLES} circular-shift shuffles...", flush=True)
    boundary_lo = SHUFFLE_MIN_OFFSET
    boundary_hi = n - SHUFFLE_MARGIN

    a1_shuf_aca = []      # list of dicts (iter, state, count)
    a1_shuf_lha = []
    a2_shuf_aca = []      # list of dicts (iter, source, destination, count)
    a2_shuf_lha = []

    for it in range(N_SHUFFLES):
        offset = int(rng.integers(boundary_lo, boundary_hi))
        v_shuf = np.roll(viterbi, offset)
        # A1 shuffle
        c_aca = a1_shuffle_kernel(aca_rates, v_shuf, K)
        c_lha = a1_shuffle_kernel(lha_rates, v_shuf, K)
        for st, cnt in c_aca.items():
            a1_shuf_aca.append(dict(iter=it, state=int(st), n_sig=int(cnt)))
        for st, cnt in c_lha.items():
            a1_shuf_lha.append(dict(iter=it, state=int(st), n_sig=int(cnt)))
        # A2 shuffle
        c2_aca = a2_shuffle_kernel(aca_rates, v_shuf, K)
        c2_lha = a2_shuffle_kernel(lha_rates, v_shuf, K)
        for (src, dst), cnt in c2_aca.items():
            a2_shuf_aca.append(dict(iter=it, source=int(src),
                                       destination=int(dst), n_sig=int(cnt)))
        for (src, dst), cnt in c2_lha.items():
            a2_shuf_lha.append(dict(iter=it, source=int(src),
                                       destination=int(dst), n_sig=int(cnt)))
        if (it + 1) % 20 == 0:
            print(f"    shuffle iter {it+1}/{N_SHUFFLES} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    df_a1s_aca = pd.DataFrame(a1_shuf_aca)
    df_a1s_lha = pd.DataFrame(a1_shuf_lha)
    df_a2s_aca = pd.DataFrame(a2_shuf_aca)
    df_a2s_lha = pd.DataFrame(a2_shuf_lha)
    pd.concat([
        df_a1s_aca.assign(region="ACA"),
        df_a1s_lha.assign(region="LHA"),
    ], ignore_index=True).to_csv(out_dir / "shuffle_A1_summary.csv", index=False)
    pd.concat([
        df_a2s_aca.assign(region="ACA"),
        df_a2s_lha.assign(region="LHA"),
    ], ignore_index=True).to_csv(out_dir / "shuffle_A2_summary.csv", index=False)

    # ---- Per (state, region) sig vs shuffle p95 — A1 ----
    a1_pass = []
    for region, df_obs, df_shuf in [
        ("ACA", df_a1[df_a1.region == "ACA"], df_a1s_aca),
        ("LHA", df_a1[df_a1.region == "LHA"], df_a1s_lha),
    ]:
        states_tested = sorted(set(df_obs["state"]) if len(df_obs) else [])
        for st in states_tested:
            obs_n = int(df_obs[(df_obs.state == st) & df_obs["sig_fdr"]].shape[0])
            shuf_n = (df_shuf[df_shuf.state == st]["n_sig"].values
                      if len(df_shuf) else np.array([]))
            # iterations that didn't include state st have implicit 0
            full = np.zeros(N_SHUFFLES, dtype=int)
            for _, r in df_shuf[df_shuf.state == st].iterrows():
                full[int(r["iter"])] = int(r["n_sig"])
            shuf_p95 = float(np.percentile(full, 95)) if len(full) else 0.0
            a1_pass.append(dict(region=region, state=st,
                                  observed=obs_n, shuf_mean=float(full.mean()),
                                  shuf_p95=shuf_p95,
                                  obs_pctile=float((full <= obs_n).mean() * 100),
                                  exceeds_p95=bool(obs_n > shuf_p95)))
    df_a1pass = pd.DataFrame(a1_pass)
    if len(df_a1pass) == 0:
        df_a1pass = pd.DataFrame(columns=["region", "state", "observed",
                                              "shuf_mean", "shuf_p95",
                                              "obs_pctile", "exceeds_p95"])
    df_a1pass.to_csv(out_dir / "A1_pass_summary.csv", index=False)

    # ---- Per (source, destination, region) sig vs shuffle p95 — A2 ----
    a2_pass = []
    for region, df_obs, df_shuf in [
        ("ACA", df_a2[df_a2.region == "ACA"], df_a2s_aca),
        ("LHA", df_a2[df_a2.region == "LHA"], df_a2s_lha),
    ]:
        if not len(df_obs):
            continue
        pairs = sorted(set(zip(df_obs["source"].astype(int),
                                df_obs["destination"].astype(int))))
        for (src, dst) in pairs:
            obs_n = int(df_obs[(df_obs.source == src) &
                                  (df_obs.destination == dst) &
                                  df_obs["sig_fdr"]].shape[0])
            full = np.zeros(N_SHUFFLES, dtype=int)
            sub = df_shuf[(df_shuf.source == src)
                            & (df_shuf.destination == dst)] if len(df_shuf) else df_shuf
            for _, r in sub.iterrows():
                full[int(r["iter"])] = int(r["n_sig"])
            shuf_p95 = float(np.percentile(full, 95)) if len(full) else 0.0
            a2_pass.append(dict(region=region, source=src, destination=dst,
                                  observed=obs_n, shuf_mean=float(full.mean()),
                                  shuf_p95=shuf_p95,
                                  obs_pctile=float((full <= obs_n).mean() * 100),
                                  exceeds_p95=bool(obs_n > shuf_p95)))
    df_a2pass = pd.DataFrame(a2_pass)
    if len(df_a2pass) == 0:
        df_a2pass = pd.DataFrame(columns=["region", "source", "destination",
                                              "observed", "shuf_mean", "shuf_p95",
                                              "obs_pctile", "exceeds_p95"])
    df_a2pass.to_csv(out_dir / "A2_pass_summary.csv", index=False)

    # Distribution figures (per region)
    for region, df_pass, fig_name in [
        ("ACA", df_a1pass[df_a1pass.region == "ACA"], "shuffle_A1_distributions_ACA.png"),
        ("LHA", df_a1pass[df_a1pass.region == "LHA"], "shuffle_A1_distributions_LHA.png"),
    ]:
        if not len(df_pass):
            continue
        states_tested = sorted(df_pass["state"].astype(int).unique())
        n_st = len(states_tested)
        ncol = min(4, n_st); nrow = (n_st + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.6 * nrow))
        axes_flat = np.atleast_1d(axes).flatten()
        df_shuf = df_a1s_aca if region == "ACA" else df_a1s_lha
        for ax, st in zip(axes_flat, states_tested):
            full = np.zeros(N_SHUFFLES, dtype=int)
            for _, r in df_shuf[df_shuf.state == st].iterrows():
                full[int(r["iter"])] = int(r["n_sig"])
            obs = int(df_pass[df_pass.state == st]["observed"].iloc[0])
            p95 = float(df_pass[df_pass.state == st]["shuf_p95"].iloc[0])
            pct = float(df_pass[df_pass.state == st]["obs_pctile"].iloc[0])
            ax.hist(full, bins=20, color="#9999cc", edgecolor="white")
            ax.axvline(obs, color="red", lw=2, label=f"obs={obs}")
            ax.axvline(p95, color="black", lw=1, ls="--", label=f"p95={p95:.0f}")
            ax.set_title(f"State {st} (pctile {pct:.0f})", fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        for ax in axes_flat[len(states_tested):]:
            ax.axis("off")
        fig.suptitle(f"S{sn} {region} — A1 shuffle null per state", y=1.0)
        fig.tight_layout()
        fig.savefig(fig_dir / fig_name, dpi=130)
        plt.close(fig)

    # A2 distribution figure (top 8 pairs by observed)
    for region, df_pass, fig_name in [
        ("ACA", df_a2pass[df_a2pass.region == "ACA"], "shuffle_A2_distributions_ACA.png"),
        ("LHA", df_a2pass[df_a2pass.region == "LHA"], "shuffle_A2_distributions_LHA.png"),
    ]:
        if not len(df_pass):
            continue
        top = df_pass.nlargest(8, "observed")
        nrow = 2; ncol = 4
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.6 * nrow))
        axes_flat = axes.flatten()
        df_shuf = df_a2s_aca if region == "ACA" else df_a2s_lha
        for ax, (_, row) in zip(axes_flat, top.iterrows()):
            src = int(row["source"]); dst = int(row["destination"])
            full = np.zeros(N_SHUFFLES, dtype=int)
            sub = df_shuf[(df_shuf.source == src) & (df_shuf.destination == dst)]
            for _, r in sub.iterrows():
                full[int(r["iter"])] = int(r["n_sig"])
            obs = int(row["observed"])
            p95 = float(row["shuf_p95"])
            pct = float(row["obs_pctile"])
            ax.hist(full, bins=20, color="#9999cc", edgecolor="white")
            ax.axvline(obs, color="red", lw=2, label=f"obs={obs}")
            ax.axvline(p95, color="black", lw=1, ls="--", label=f"p95={p95:.0f}")
            ax.set_title(f"S{src}→S{dst} (pctile {pct:.0f})", fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        for ax in axes_flat[len(top):]:
            ax.axis("off")
        fig.suptitle(f"S{sn} {region} — A2 shuffle null (top 8 pairs)", y=1.0)
        fig.tight_layout()
        fig.savefig(fig_dir / fig_name, dpi=130)
        plt.close(fig)

    print(f"  Done S{sn} [{time.time()-t0:.0f}s]", flush=True)

    return dict(
        sn=sn, K=K, n_aca=n_aca, n_lha=n_lha,
        metabolic_state=session["metabolic_state"],
        df_a1pass=df_a1pass, df_a2pass=df_a2pass,
        df_a1=df_a1, df_a2=df_a2,
        df_a1c=df_a1c, df_a2c=df_a2c,
    )


# ---- Cross-session aggregation ----
def cross_session(per_sess, base_out_dir, base_fig_dir):
    # A1 replication
    rows = []
    for r in per_sess:
        if not len(r["df_a1pass"]):
            continue
        sub = r["df_a1pass"].copy()
        sub["session"] = r["sn"]
        sub["metabolic_state"] = r["metabolic_state"]
        rows.append(sub)
    cross_a1 = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(cross_a1):
        cross_a1.to_csv(base_out_dir / "A1_cross_session.csv", index=False)

    rep_a1 = []
    if len(cross_a1):
        for region in ("ACA", "LHA"):
            sub = cross_a1[cross_a1.region == region]
            for st in sorted(sub["state"].unique()):
                sub_st = sub[sub.state == st]
                rep_a1.append(dict(
                    region=region, state=int(st),
                    n_sessions_tested=int(len(sub_st)),
                    n_sessions_passing=int(sub_st["exceeds_p95"].sum()),
                    sessions_passing=",".join(str(int(x)) for x in
                                                sub_st.loc[sub_st["exceeds_p95"], "session"]),
                ))
    rep_a1_df = pd.DataFrame(rep_a1)
    if len(rep_a1_df):
        rep_a1_df.to_csv(base_out_dir / "replication_A1.csv", index=False)

    # A2 replication
    rows = []
    for r in per_sess:
        if not len(r["df_a2pass"]):
            continue
        sub = r["df_a2pass"].copy()
        sub["session"] = r["sn"]
        sub["metabolic_state"] = r["metabolic_state"]
        rows.append(sub)
    cross_a2 = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(cross_a2):
        cross_a2.to_csv(base_out_dir / "A2_cross_session.csv", index=False)

    rep_a2 = []
    if len(cross_a2):
        for region in ("ACA", "LHA"):
            sub = cross_a2[cross_a2.region == region]
            for (src, dst), pair_sub in sub.groupby(["source", "destination"]):
                rep_a2.append(dict(
                    region=region, source=int(src), destination=int(dst),
                    n_sessions_tested=int(len(pair_sub)),
                    n_sessions_passing=int(pair_sub["exceeds_p95"].sum()),
                    sessions_passing=",".join(str(int(x)) for x in
                                                pair_sub.loc[pair_sub["exceeds_p95"], "session"]),
                ))
    rep_a2_df = pd.DataFrame(rep_a2)
    if len(rep_a2_df):
        rep_a2_df.to_csv(base_out_dir / "replication_A2.csv", index=False)

    # Replication heatmap A1: states × sessions, color = passes shuffle
    if len(cross_a1):
        sess_order = sorted(cross_a1["session"].unique().astype(int))
        states = sorted(cross_a1["state"].unique().astype(int))
        for region in ("ACA", "LHA"):
            sub = cross_a1[cross_a1.region == region]
            mat = np.zeros((len(states), len(sess_order)), dtype=int)
            for i, st in enumerate(states):
                for j, sn in enumerate(sess_order):
                    row = sub[(sub.state == st) & (sub.session == sn)]
                    if len(row) and bool(row.iloc[0]["exceeds_p95"]):
                        mat[i, j] = 1
            fig, ax = plt.subplots(figsize=(2 + 0.6 * len(sess_order),
                                              1.5 + 0.4 * len(states)))
            ax.imshow(mat, aspect="auto", cmap="Reds", vmin=0, vmax=1,
                       interpolation="nearest")
            ax.set_xticks(range(len(sess_order)))
            ax.set_xticklabels([f"S{s}" for s in sess_order])
            ax.set_yticks(range(len(states)))
            ax.set_yticklabels([f"S{s}" for s in states])
            ax.set_xlabel("Session"); ax.set_ylabel("HMM state")
            ax.set_title(f"{region}: A1 generic switch passes shuffle p95")
            for i in range(len(states)):
                for j in range(len(sess_order)):
                    if mat[i, j]:
                        ax.text(j, i, "✓", ha="center", va="center",
                                 color="white", fontweight="bold")
            fig.tight_layout()
            fig.savefig(base_fig_dir / f"replication_heatmap_A1_{region}.png", dpi=130)
            plt.close(fig)

    # Replication heatmap A2 — top 20 pairs by sessions_passing
    if len(rep_a2_df):
        for region in ("ACA", "LHA"):
            sub = rep_a2_df[rep_a2_df.region == region]
            top = sub.nlargest(20, "n_sessions_passing")
            if not len(top):
                continue
            sess_order = sorted(cross_a2["session"].unique().astype(int))
            mat = np.zeros((len(top), len(sess_order)), dtype=int)
            labels_y = []
            for i, (_, row) in enumerate(top.iterrows()):
                src = int(row["source"]); dst = int(row["destination"])
                labels_y.append(f"S{src}→S{dst}")
                for j, sn in enumerate(sess_order):
                    pair_row = cross_a2[(cross_a2.region == region)
                                          & (cross_a2.source == src)
                                          & (cross_a2.destination == dst)
                                          & (cross_a2.session == sn)]
                    if len(pair_row) and bool(pair_row.iloc[0]["exceeds_p95"]):
                        mat[i, j] = 1
            fig, ax = plt.subplots(figsize=(2 + 0.6 * len(sess_order),
                                              1.5 + 0.4 * len(top)))
            ax.imshow(mat, aspect="auto", cmap="Reds", vmin=0, vmax=1,
                       interpolation="nearest")
            ax.set_xticks(range(len(sess_order)))
            ax.set_xticklabels([f"S{s}" for s in sess_order])
            ax.set_yticks(range(len(top)))
            ax.set_yticklabels(labels_y, fontsize=8)
            ax.set_xlabel("Session"); ax.set_ylabel("Source → destination")
            ax.set_title(f"{region}: A2 destination-specific passes shuffle p95")
            for i in range(len(top)):
                for j in range(len(sess_order)):
                    if mat[i, j]:
                        ax.text(j, i, "✓", ha="center", va="center",
                                 color="white", fontweight="bold")
            fig.tight_layout()
            fig.savefig(base_fig_dir / f"replication_heatmap_A2_{region}.png", dpi=130)
            plt.close(fig)

    return rep_a1_df, rep_a2_df


# ---- Main ----
def main():
    cfg = load_config()
    base_out, base_fig = out_dirs()
    with open(REPO_ROOT / cfg["paths_yaml"]) as f:
        paths_data = yaml.safe_load(f)

    rng = np.random.default_rng(SHUFFLE_SEED)

    per_sess = []
    for sn in SESSIONS:
        print(f"\n========== Loading S{sn} ==========", flush=True)
        try:
            session = load_session(sn, cfg, paths_data)
        except Exception as e:
            print(f"  ERROR loading S{sn}: {e}", flush=True)
            continue
        result = run_session(session, base_out, base_fig, rng)
        per_sess.append(result)

    print("\n========== Cross-session aggregation ==========", flush=True)
    rep_a1, rep_a2 = cross_session(per_sess, base_out, base_fig)

    print("\n========== HEADLINE — A1 ==========")
    if len(rep_a1):
        rep_a1_top = rep_a1.sort_values(["n_sessions_passing", "region", "state"],
                                          ascending=[False, True, True])
        print(rep_a1_top.to_string(index=False))
        print(f"\nStates with A1 pass in ≥{REPLICATION_THRESHOLD} sessions:")
        rep_strong = rep_a1[rep_a1["n_sessions_passing"] >= REPLICATION_THRESHOLD]
        if len(rep_strong):
            print(rep_strong.to_string(index=False))
        else:
            print("  (none)")
    else:
        print("  No A1 results")

    print("\n========== HEADLINE — A2 ==========")
    if len(rep_a2):
        rep_a2_top = rep_a2.sort_values(["n_sessions_passing", "region",
                                            "source", "destination"],
                                          ascending=[False, True, True, True])
        print(f"Top 20 (source, destination) pairs by A2 sessions passing:")
        print(rep_a2_top.head(20).to_string(index=False))
        print(f"\nPairs with A2 pass in ≥{REPLICATION_THRESHOLD} sessions:")
        rep_strong2 = rep_a2[rep_a2["n_sessions_passing"] >= REPLICATION_THRESHOLD]
        if len(rep_strong2):
            print(rep_strong2.to_string(index=False))
        else:
            print("  (none)")
    else:
        print("  No A2 results")

    print(f"\nDone. Outputs in {base_out} and {base_fig}", flush=True)


if __name__ == "__main__":
    main()
