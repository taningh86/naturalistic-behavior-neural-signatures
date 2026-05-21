"""15 — Six population-level metrics on the stay vs pre-exit contrast.

Extends script 14's per-unit-firing-rate Mann-Whitney test with five additional
population-level metrics that capture different aspects of the population
state. All applied to the same A1 contrast (stay bins vs pre-exit bins,
pooled across destinations) on states that already replicated in 14.

  M1 — Fano factor:        per-unit dispersion (count residuals); per-state
                              FDR-sig unit count from per-bin Mann-Whitney
                              on |count − mean| / √mean.
  M2 — ISI CV:              per-state scalar = CV(ISI_pre_exit) − CV(ISI_stay)
                              from concatenated within-bin spike trains.
  M3 — PC trajectory speed: per-state scalar = mean step length in PC1-3
                              within-run, pre-exit minus stay.
  M4 — Participation ratio: per-state scalar = PR(C_pre) − PR(C_stay) where
                              C is the unit×unit covariance.
  M5 — Pairwise corr norm:  per-state scalar = ||C_pre − C_stay||_F /
                              mean(|C_stay| + |C_pre|).
  M6 — Cross-region corr:   per-state scalar = mean(|r_ACA-LHA|_pre) −
                              mean(|r_ACA-LHA|_stay).

100 circular-shift Viterbi shuffles per session, same protocol as script 14.
Per-state pass = observed metric exceeds shuffle 95th percentile (two-tailed
for M2-M6, one-tailed upper for M1 unit count).

States analyzed (per script 14 results):
  ACA: S2, S3, S4, S6, S8, S9, S12
  LHA: S2, S3
  M6 cross-region: union of the above source lists.

Spec deviations:
  - M2 done at the per-state scalar level (CV difference) rather than per-unit
    FDR-sig unit count — running 100 shuffle CV resamples per unit (250 × 100 =
    25k) is tractable but very slow; the per-state scalar with shuffle null
    is the same kind of test as M3-M6 and matches their replication test.
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
K_PRE = 3
N_SHUFFLES = 100
SHUFFLE_MIN_OFFSET = 200
SHUFFLE_MARGIN = 200
SHUFFLE_SEED = 20260508
FDR_ALPHA = 0.05
N_PCS = 5

ACA_STATES = [2, 3, 4, 6, 8, 9, 12]
LHA_STATES = [2, 3]
M6_STATES = sorted(set(ACA_STATES) | set(LHA_STATES))
MIN_BINS = 30


def out_dirs():
    base_out = REPO_ROOT / "data" / "HMM" / "neural_alignment" / "state_transitions" / "population_metrics"
    base_fig = REPO_ROOT / "figures" / "HMM" / "neural_alignment" / "state_transitions" / "population_metrics"
    base_out.mkdir(parents=True, exist_ok=True)
    base_fig.mkdir(parents=True, exist_ok=True)
    return base_out, base_fig


# ---- Helpers ----
def fdr_pass_mask(pvals, q=FDR_ALPHA):
    p = np.asarray(pvals, dtype=np.float64)
    valid = np.isfinite(p)
    sig = np.zeros(p.shape, dtype=bool)
    if not valid.any():
        return sig
    rej, _, _, _ = multipletests(p[valid], alpha=q, method="fdr_bh")
    sig[valid] = rej
    return sig


def rebin_100ms_to_480ms(rates_100ms, n_hmm_bins):
    n_100ms = rates_100ms.shape[1]
    centers_s = (np.arange(n_100ms) + 0.5) * NEURAL_BIN_SMALL_S
    hmm_idx = np.floor(centers_s / HMM_BIN_S).astype(np.int64)
    valid = (hmm_idx >= 0) & (hmm_idx < n_hmm_bins)
    hmm_idx = hmm_idx[valid]
    rates_100ms = rates_100ms[:, valid]
    out = np.zeros((rates_100ms.shape[0], n_hmm_bins), dtype=np.float64)
    counts = np.bincount(hmm_idx, minlength=n_hmm_bins)
    for u in range(rates_100ms.shape[0]):
        sums = np.bincount(hmm_idx, weights=rates_100ms[u].astype(np.float64),
                            minlength=n_hmm_bins)
        out[u] = sums / np.maximum(counts, 1)
    return out * (1.0 / NEURAL_BIN_SMALL_S)


def label_bins(viterbi, K_pre=K_PRE):
    n = len(viterbi)
    diff = np.diff(viterbi, prepend=-1, append=-1)
    boundaries = np.flatnonzero(diff != 0)
    starts = boundaries[:-1]; ends = boundaries[1:]
    bin_group = np.array(["excluded"] * n, dtype=object)
    run_id = np.full(n, -1, dtype=np.int64)
    run_position = np.full(n, -1, dtype=np.int64)
    run_length = np.full(n, -1, dtype=np.int64)
    for i, (s, e) in enumerate(zip(starts, ends)):
        L = int(e - s)
        run_id[s:e] = i; run_position[s:e] = np.arange(L); run_length[s:e] = L
        if L < 2 * K_pre:
            continue
        bin_group[s : e - K_pre] = "stay"
        bin_group[e - K_pre : e] = "pre_exit"
    return dict(bin_group=bin_group, run_id=run_id,
                run_position=run_position, run_length=run_length)


def compute_pca_projection(rates):
    mu = rates.mean(axis=1, keepdims=True)
    sig = rates.std(axis=1, keepdims=True) + 1e-9
    z = (rates - mu) / sig
    X = z.T
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return U[:, :N_PCS] * S[:N_PCS]


# ---- Metrics ----
def m1_fano_per_unit_pvals(counts, viterbi, labels, state):
    """Per-unit Mann-Whitney on |count - cond_mean| / sqrt(cond_mean) between
    stay and pre-exit bins. Returns (n_units,) p-values, NaN where untested."""
    n_units = counts.shape[0]
    in_state = (viterbi == state)
    stay_idx = np.flatnonzero(in_state & (labels["bin_group"] == "stay"))
    pre_idx = np.flatnonzero(in_state & (labels["bin_group"] == "pre_exit"))
    pvals = np.full(n_units, np.nan)
    if len(stay_idx) < MIN_BINS or len(pre_idx) < MIN_BINS:
        return pvals
    stay_counts = counts[:, stay_idx]
    pre_counts = counts[:, pre_idx]
    mean_stay = stay_counts.mean(axis=1, keepdims=True) + 1e-9
    mean_pre = pre_counts.mean(axis=1, keepdims=True) + 1e-9
    res_stay = np.abs(stay_counts - mean_stay) / np.sqrt(mean_stay)
    res_pre = np.abs(pre_counts - mean_pre) / np.sqrt(mean_pre)
    for u in range(n_units):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                _, p = mannwhitneyu(res_stay[u], res_pre[u],
                                      alternative="two-sided")
                pvals[u] = p
            except Exception:
                pass
    return pvals


def m2_isi_cv_diff(spike_times_dict, viterbi, labels, state, edges_480ms):
    """Per-state scalar: CV(ISI_pre) - CV(ISI_stay) pooled across units
    and bins (each unit's within-bin ISIs concatenated within condition)."""
    in_state = (viterbi == state)
    stay_idx = np.flatnonzero(in_state & (labels["bin_group"] == "stay"))
    pre_idx = np.flatnonzero(in_state & (labels["bin_group"] == "pre_exit"))
    if len(stay_idx) < MIN_BINS or len(pre_idx) < MIN_BINS:
        return np.nan, 0, 0

    def collect_isis(bin_indices):
        all_isis = []
        for uid, st in spike_times_dict.items():
            if len(st) == 0:
                continue
            for b in bin_indices:
                t0 = edges_480ms[b]; t1 = edges_480ms[b + 1]
                # spikes in [t0, t1)
                lo = np.searchsorted(st, t0, side="left")
                hi = np.searchsorted(st, t1, side="left")
                spikes_in = st[lo:hi]
                if len(spikes_in) >= 2:
                    isis = np.diff(spikes_in)
                    all_isis.extend(isis.tolist())
        return np.asarray(all_isis)

    isis_stay = collect_isis(stay_idx)
    isis_pre = collect_isis(pre_idx)
    if len(isis_stay) < 50 or len(isis_pre) < 50:
        return np.nan, len(isis_stay), len(isis_pre)
    cv_stay = float(isis_stay.std() / (isis_stay.mean() + 1e-12))
    cv_pre = float(isis_pre.std() / (isis_pre.mean() + 1e-12))
    return cv_pre - cv_stay, len(isis_stay), len(isis_pre)


def m3_pc_speed_diff(pcs, viterbi, labels, state):
    """Per-state scalar: mean PC1-3 step length pre_exit - stay, computed
    only between consecutive bins of the same condition within the same run."""
    bin_group = labels["bin_group"]
    run_id = labels["run_id"]
    in_state_pre = (viterbi == state) & (bin_group == "pre_exit")
    in_state_stay = (viterbi == state) & (bin_group == "stay")
    if in_state_stay.sum() < MIN_BINS or in_state_pre.sum() < MIN_BINS:
        return np.nan, 0, 0

    def step_lengths(mask):
        idx = np.flatnonzero(mask)
        if len(idx) < 2:
            return np.array([])
        # consecutive-within-run steps
        keep = (np.diff(idx) == 1) & (run_id[idx[1:]] == run_id[idx[:-1]])
        if not keep.any():
            return np.array([])
        a = idx[:-1][keep]; b = idx[1:][keep]
        return np.linalg.norm(pcs[b, :3] - pcs[a, :3], axis=1)

    s_stay = step_lengths(in_state_stay)
    s_pre = step_lengths(in_state_pre)
    if len(s_stay) < 5 or len(s_pre) < 5:
        return np.nan, len(s_stay), len(s_pre)
    return float(s_pre.mean() - s_stay.mean()), len(s_stay), len(s_pre)


def m4_pr_diff(rates, viterbi, labels, state):
    """Participation ratio difference (pre_exit - stay) on n_units × n_bins."""
    in_state = (viterbi == state)
    stay_idx = np.flatnonzero(in_state & (labels["bin_group"] == "stay"))
    pre_idx = np.flatnonzero(in_state & (labels["bin_group"] == "pre_exit"))
    if len(stay_idx) < MIN_BINS or len(pre_idx) < MIN_BINS:
        return np.nan, 0, 0
    def pr(X):
        X = X - X.mean(axis=1, keepdims=True)
        if X.shape[1] < 2:
            return np.nan
        cov = (X @ X.T) / (X.shape[1] - 1)
        eigs = np.linalg.eigvalsh(cov)
        eigs = np.maximum(eigs, 0)
        s1 = eigs.sum(); s2 = (eigs ** 2).sum()
        if s2 < 1e-12:
            return np.nan
        return (s1 ** 2) / s2
    pr_stay = pr(rates[:, stay_idx])
    pr_pre = pr(rates[:, pre_idx])
    if not (np.isfinite(pr_stay) and np.isfinite(pr_pre)):
        return np.nan, len(stay_idx), len(pre_idx)
    return float(pr_pre - pr_stay), len(stay_idx), len(pre_idx)


def m5_corr_struct_norm(rates, viterbi, labels, state):
    """Frobenius norm of (C_pre - C_stay), normalized by mean(|C_stay|+|C_pre|)."""
    in_state = (viterbi == state)
    stay_idx = np.flatnonzero(in_state & (labels["bin_group"] == "stay"))
    pre_idx = np.flatnonzero(in_state & (labels["bin_group"] == "pre_exit"))
    if len(stay_idx) < MIN_BINS or len(pre_idx) < MIN_BINS:
        return np.nan, 0, 0
    def corr(X):
        X = X - X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1) + 1e-9
        Xn = X / sd[:, None]
        return (Xn @ Xn.T) / X.shape[1]
    C_stay = corr(rates[:, stay_idx])
    C_pre = corr(rates[:, pre_idx])
    diff = C_pre - C_stay
    np.fill_diagonal(diff, 0)
    norm_diff = np.linalg.norm(diff, ord="fro")
    norm_avg = (np.abs(C_stay).mean() + np.abs(C_pre).mean()) / 2.0
    if norm_avg < 1e-12:
        return np.nan, len(stay_idx), len(pre_idx)
    return float(norm_diff / (norm_avg * (rates.shape[0]))), len(stay_idx), len(pre_idx)


def m6_cross_region_corr_diff(aca_rates, lha_rates, viterbi, labels, state):
    """Mean |r_ACA-LHA pair| difference (pre - stay)."""
    in_state = (viterbi == state)
    stay_idx = np.flatnonzero(in_state & (labels["bin_group"] == "stay"))
    pre_idx = np.flatnonzero(in_state & (labels["bin_group"] == "pre_exit"))
    if len(stay_idx) < MIN_BINS or len(pre_idx) < MIN_BINS:
        return np.nan, 0, 0
    def cross_corrs(A, L):
        A = A - A.mean(axis=1, keepdims=True)
        L = L - L.mean(axis=1, keepdims=True)
        sa = A.std(axis=1) + 1e-9; sl = L.std(axis=1) + 1e-9
        An = A / sa[:, None]; Ln = L / sl[:, None]
        return (An @ Ln.T) / A.shape[1]
    R_stay = cross_corrs(aca_rates[:, stay_idx], lha_rates[:, stay_idx])
    R_pre = cross_corrs(aca_rates[:, pre_idx], lha_rates[:, pre_idx])
    return (float(np.abs(R_pre).mean() - np.abs(R_stay).mean()),
            len(stay_idx), len(pre_idx))


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
    n_100 = int(np.ceil(duration_s / NEURAL_BIN_SMALL_S))
    edges_100 = np.arange(n_100 + 1) * NEURAL_BIN_SMALL_S
    edges_480 = np.arange(n_hmm + 1) * HMM_BIN_S

    aca_uid_list = sorted(aca_spikes.keys())
    lha_uid_list = sorted(lha_spikes.keys())
    n_aca = len(aca_uid_list); n_lha = len(lha_uid_list)
    aca_100 = np.zeros((n_aca, n_100)); lha_100 = np.zeros((n_lha, n_100))
    aca_counts_480 = np.zeros((n_aca, n_hmm), dtype=np.int64)
    lha_counts_480 = np.zeros((n_lha, n_hmm), dtype=np.int64)
    for i, uid in enumerate(aca_uid_list):
        aca_100[i] = np.histogram(aca_spikes[uid], edges_100)[0]
        aca_counts_480[i] = np.histogram(aca_spikes[uid], edges_480)[0]
    for i, uid in enumerate(lha_uid_list):
        lha_100[i] = np.histogram(lha_spikes[uid], edges_100)[0]
        lha_counts_480[i] = np.histogram(lha_spikes[uid], edges_480)[0]
    aca_rates = rebin_100ms_to_480ms(aca_100, n_hmm)
    lha_rates = rebin_100ms_to_480ms(lha_100, n_hmm)

    post = pd.read_csv(REPO_ROOT / cfg["merge_dirs"]["posteriors"]
                        / f"session_{sn}.csv")
    viterbi = post["viterbi"].values.astype(np.int64)
    K = max(int(viterbi.max()) + 1,
            sum(1 for c in post.columns if c.startswith("p_state_")))
    n = min(n_hmm, len(viterbi))
    aca_rates = aca_rates[:, :n]; lha_rates = lha_rates[:, :n]
    aca_counts_480 = aca_counts_480[:, :n]
    lha_counts_480 = lha_counts_480[:, :n]
    viterbi = viterbi[:n]

    history = pd.read_csv(REPO_ROOT / cfg["commitment_dirs"]["out"]
                           / "sampling_history.csv")
    s_hist = history[history.session == sn].iloc[0]
    return dict(sn=sn, K=K, n_bins=n,
                 aca_rates=aca_rates, lha_rates=lha_rates,
                 aca_counts=aca_counts_480, lha_counts=lha_counts_480,
                 aca_spikes=aca_spikes, lha_spikes=lha_spikes,
                 viterbi=viterbi, edges_480=edges_480,
                 metabolic_state=s_hist["state"],
                 n_aca=n_aca, n_lha=n_lha)


def run_session(session, base_out, base_fig, rng):
    sn = session["sn"]; K = session["K"]
    out_dir = base_out / f"session_{sn}"
    fig_dir = base_fig / f"session_{sn}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    aca_rates = session["aca_rates"]; lha_rates = session["lha_rates"]
    aca_counts = session["aca_counts"]; lha_counts = session["lha_counts"]
    aca_spikes = session["aca_spikes"]; lha_spikes = session["lha_spikes"]
    viterbi = session["viterbi"]
    edges_480 = session["edges_480"]
    n = session["n_bins"]
    n_aca = session["n_aca"]; n_lha = session["n_lha"]

    print(f"\n--- S{sn} ({session['metabolic_state']}) "
          f"K={K} bins={n} ACA={n_aca} LHA={n_lha} ---", flush=True)
    t0 = time.time()
    pcs_aca = compute_pca_projection(aca_rates)
    pcs_lha = compute_pca_projection(lha_rates)

    # ---- Observed ----
    obs_rows = []
    obs_labels = label_bins(viterbi)
    print("  Computing observed metrics...", flush=True)
    for region, rates, counts, spikes, pcs, n_units, states in [
        ("ACA", aca_rates, aca_counts, aca_spikes, pcs_aca, n_aca, ACA_STATES),
        ("LHA", lha_rates, lha_counts, lha_spikes, pcs_lha, n_lha, LHA_STATES),
    ]:
        for state in states:
            # M1
            pvals = m1_fano_per_unit_pvals(counts, viterbi, obs_labels, state)
            sig = fdr_pass_mask(pvals, FDR_ALPHA) if np.isfinite(pvals).any() else np.zeros_like(pvals, dtype=bool)
            obs_m1 = int(sig.sum())
            # M2
            isi_diff, n_isi_stay, n_isi_pre = m2_isi_cv_diff(
                spikes, viterbi, obs_labels, state, edges_480)
            # M3
            pc_speed_diff, _, _ = m3_pc_speed_diff(pcs, viterbi, obs_labels, state)
            # M4
            pr_diff, _, _ = m4_pr_diff(rates, viterbi, obs_labels, state)
            # M5
            corr_norm, _, _ = m5_corr_struct_norm(rates, viterbi, obs_labels, state)
            obs_rows.append(dict(region=region, state=state,
                                   M1_n_sig_units=obs_m1,
                                   M2_cv_isi_diff=isi_diff,
                                   M3_pc_speed_diff=pc_speed_diff,
                                   M4_pr_diff=pr_diff,
                                   M5_corr_norm=corr_norm))
    # M6 cross-region (one row per state)
    for state in M6_STATES:
        m6_diff, _, _ = m6_cross_region_corr_diff(aca_rates, lha_rates,
                                                     viterbi, obs_labels, state)
        obs_rows.append(dict(region="ACA-LHA", state=state,
                              M1_n_sig_units=np.nan,
                              M2_cv_isi_diff=np.nan,
                              M3_pc_speed_diff=np.nan,
                              M4_pr_diff=np.nan,
                              M5_corr_norm=np.nan,
                              M6_cross_corr_diff=m6_diff))
    df_obs = pd.DataFrame(obs_rows)
    df_obs["session"] = sn
    df_obs.to_csv(out_dir / "observed_metrics.csv", index=False)
    print(f"  Observed done [{time.time()-t0:.0f}s]", flush=True)

    # ---- Shuffles ----
    print(f"  Running {N_SHUFFLES} shuffles...", flush=True)
    boundary_lo = SHUFFLE_MIN_OFFSET
    boundary_hi = n - SHUFFLE_MARGIN
    shuf_records = []   # one row per (iter, region, state) with all metric values

    for it in range(N_SHUFFLES):
        offset = int(rng.integers(boundary_lo, boundary_hi))
        v_shuf = np.roll(viterbi, offset)
        labels_shuf = label_bins(v_shuf)
        # Per-region M1-M5
        for region, rates, counts, spikes, pcs, states in [
            ("ACA", aca_rates, aca_counts, aca_spikes, pcs_aca, ACA_STATES),
            ("LHA", lha_rates, lha_counts, lha_spikes, pcs_lha, LHA_STATES),
        ]:
            for state in states:
                pvals = m1_fano_per_unit_pvals(counts, v_shuf, labels_shuf, state)
                sig = fdr_pass_mask(pvals, FDR_ALPHA) if np.isfinite(pvals).any() else np.zeros_like(pvals, dtype=bool)
                m1_count = int(sig.sum())
                m2_d, _, _ = m2_isi_cv_diff(spikes, v_shuf, labels_shuf, state, edges_480)
                m3_d, _, _ = m3_pc_speed_diff(pcs, v_shuf, labels_shuf, state)
                m4_d, _, _ = m4_pr_diff(rates, v_shuf, labels_shuf, state)
                m5_d, _, _ = m5_corr_struct_norm(rates, v_shuf, labels_shuf, state)
                shuf_records.append(dict(iter=it, region=region, state=state,
                                            M1_n_sig_units=m1_count,
                                            M2_cv_isi_diff=m2_d,
                                            M3_pc_speed_diff=m3_d,
                                            M4_pr_diff=m4_d,
                                            M5_corr_norm=m5_d,
                                            M6_cross_corr_diff=np.nan))
        # M6 (cross-region)
        for state in M6_STATES:
            m6_d, _, _ = m6_cross_region_corr_diff(aca_rates, lha_rates,
                                                     v_shuf, labels_shuf, state)
            shuf_records.append(dict(iter=it, region="ACA-LHA", state=state,
                                        M1_n_sig_units=np.nan,
                                        M2_cv_isi_diff=np.nan,
                                        M3_pc_speed_diff=np.nan,
                                        M4_pr_diff=np.nan,
                                        M5_corr_norm=np.nan,
                                        M6_cross_corr_diff=m6_d))
        if (it + 1) % 20 == 0:
            print(f"    shuffle iter {it+1}/{N_SHUFFLES} ({time.time()-t0:.0f}s)",
                  flush=True)
    df_shuf = pd.DataFrame(shuf_records)
    df_shuf["session"] = sn
    df_shuf.to_csv(out_dir / "shuffle_records.csv", index=False)

    # ---- Per-(region, state, metric) pass test ----
    pass_rows = []
    metrics_one_tail_upper = {"M1_n_sig_units"}    # only M1 is one-tailed
    metrics_two_tail = {"M2_cv_isi_diff", "M3_pc_speed_diff",
                         "M4_pr_diff", "M5_corr_norm", "M6_cross_corr_diff"}
    for _, row in df_obs.iterrows():
        region = row["region"]; state = int(row["state"])
        for metric in (list(metrics_one_tail_upper) + list(metrics_two_tail)):
            if metric not in row.index:
                continue
            obs = row[metric]
            if pd.isna(obs):
                continue
            shuf = df_shuf[(df_shuf.region == region)
                              & (df_shuf.state == state)][metric].dropna().values
            if len(shuf) == 0:
                continue
            if metric in metrics_one_tail_upper:
                p95 = float(np.percentile(shuf, 95))
                passes = bool(obs > p95)
                obs_pct = float((shuf <= obs).mean() * 100)
            else:
                p95 = float(np.percentile(np.abs(shuf), 95))
                passes = bool(abs(obs) > p95)
                obs_pct = float((np.abs(shuf) <= abs(obs)).mean() * 100)
            pass_rows.append(dict(session=sn, region=region, state=state,
                                    metric=metric,
                                    observed=float(obs),
                                    shuffle_p95_abs=p95,
                                    obs_pctile=obs_pct,
                                    exceeds_p95=passes))
    df_pass = pd.DataFrame(pass_rows)
    df_pass.to_csv(out_dir / "pass_summary.csv", index=False)
    print(f"  Pass summary saved ({len(df_pass)} rows) [{time.time()-t0:.0f}s]",
          flush=True)
    return dict(sn=sn, df_obs=df_obs, df_pass=df_pass,
                  metabolic_state=session["metabolic_state"])


# ---- Cross-session aggregation ----
def cross_session(per_sess, base_out, base_fig):
    rows = []
    for r in per_sess:
        sub = r["df_pass"].copy()
        sub["metabolic_state"] = r["metabolic_state"]
        rows.append(sub)
    cross = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not len(cross):
        return None
    cross.to_csv(base_out / "cross_session_pass.csv", index=False)

    # Master replication: count of sessions passing per (region, state, metric)
    master_rows = []
    for (region, state, metric), grp in cross.groupby(["region", "state", "metric"]):
        n_tested = len(grp)
        n_pass = int(grp["exceeds_p95"].sum())
        sessions_passing = ",".join(str(x) for x in
                                       grp.loc[grp["exceeds_p95"], "session"].astype(int))
        master_rows.append(dict(region=region, state=int(state), metric=metric,
                                  n_sessions_tested=n_tested,
                                  n_sessions_passing=n_pass,
                                  sessions_passing=sessions_passing))
    master = pd.DataFrame(master_rows)
    master.to_csv(base_out / "master_replication_table.csv", index=False)

    # Master heatmap: rows = (region, state), cols = M1..M6. Cell value = sessions passing.
    metric_order = ["M1_n_sig_units", "M2_cv_isi_diff", "M3_pc_speed_diff",
                     "M4_pr_diff", "M5_corr_norm", "M6_cross_corr_diff"]
    rs_pairs = sorted({(row["region"], int(row["state"])) for _, row in master.iterrows()},
                       key=lambda x: (x[0], x[1]))
    mat = np.full((len(rs_pairs), len(metric_order)), np.nan)
    n_tested_mat = np.zeros((len(rs_pairs), len(metric_order)), dtype=int)
    for i, (region, state) in enumerate(rs_pairs):
        for j, m in enumerate(metric_order):
            sel = master[(master.region == region) & (master.state == state)
                          & (master.metric == m)]
            if len(sel):
                mat[i, j] = int(sel.iloc[0]["n_sessions_passing"])
                n_tested_mat[i, j] = int(sel.iloc[0]["n_sessions_tested"])

    fig, ax = plt.subplots(figsize=(2 + 1.0 * len(metric_order),
                                      0.6 * len(rs_pairs) + 1.5))
    im = ax.imshow(mat, aspect="auto", cmap="Reds", vmin=0, vmax=6,
                    interpolation="nearest")
    ax.set_xticks(range(len(metric_order)))
    ax.set_xticklabels([m.split("_")[0] for m in metric_order])
    ax.set_yticks(range(len(rs_pairs)))
    ax.set_yticklabels([f"{r}/S{s}" for r, s in rs_pairs])
    plt.colorbar(im, ax=ax, label="# sessions passing")
    ax.set_title("Master replication — # sessions where observed exceeds shuffle p95")
    for i in range(len(rs_pairs)):
        for j in range(len(metric_order)):
            v = mat[i, j]; nt = n_tested_mat[i, j]
            if not np.isnan(v):
                txt_color = "white" if v >= 4 else "black"
                ax.text(j, i, f"{int(v)}/{int(nt)}",
                         ha="center", va="center", fontsize=8, color=txt_color)
    fig.tight_layout()
    fig.savefig(base_fig / "master_replication_heatmap.png", dpi=130)
    plt.close(fig)

    return master


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
    master = cross_session(per_sess, base_out, base_fig)

    if master is not None:
        print("\n========== HEADLINE: replication per (region, state, metric) ==========")
        # Print reorganized: region, state with a row of metric counts
        metric_order = ["M1_n_sig_units", "M2_cv_isi_diff", "M3_pc_speed_diff",
                         "M4_pr_diff", "M5_corr_norm", "M6_cross_corr_diff"]
        for region in sorted(master["region"].unique()):
            print(f"\n  {region}:")
            sub = master[master.region == region]
            states = sorted(sub["state"].unique())
            print("    state " + "".join(f"  {m.split('_')[0]:>4}" for m in metric_order))
            for st in states:
                vals = []
                for m in metric_order:
                    sel = sub[(sub.state == st) & (sub.metric == m)]
                    if len(sel):
                        vals.append(f"{int(sel.iloc[0]['n_sessions_passing'])}/{int(sel.iloc[0]['n_sessions_tested'])}")
                    else:
                        vals.append("  - ")
                print(f"    S{st:<4} " + "".join(f"  {v:>4}" for v in vals))

        print("\n  States passing in ≥3 sessions (region, state, metric):")
        rep_strong = master[master["n_sessions_passing"] >= 3]
        if len(rep_strong):
            print(rep_strong.sort_values(
                ["region", "n_sessions_passing", "metric"],
                ascending=[True, False, True]
            ).to_string(index=False))
        else:
            print("    (none)")

    print(f"\nDone.", flush=True)


if __name__ == "__main__":
    main()
