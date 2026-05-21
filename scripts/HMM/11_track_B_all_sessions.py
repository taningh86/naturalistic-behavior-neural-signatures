"""11 — Multi-session Track B + shuffle controls across foraging sessions.

Runs the full Track B (B1, B2, B3, B4) pipeline plus the 10c-style
(circular-shift) and 10d-style (fake-discovery) shuffles for sessions
[4, 6, 8, 14, 16]. Session 12 results are loaded from the existing 10b/10c/10d
outputs (no recompute). S10 is excluded (no-food extinction).

Aggregates per-session results into:
  - cross_session_summary.csv   (one row per session, key metrics)
  - replication_count_per_state.csv
  - replication_heatmap_{ACA,LHA}.png
  - preferred_state_counts_all_sessions.png
  - fed_vs_fasted_aggregate_metrics.png

S4 uses the manual-override discovery time (bin 916, t=439.8 s) from
sampling_history.csv.

Runtime: ~5-6 min per new session (× 5) = 25-30 min. Set verbose flushing in
calling shell so progress streams through tee.
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
from matplotlib.colors import to_rgba
from scipy.stats import f_oneway, mannwhitneyu, norm
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


# =====  Constants  =====
HMM_BIN_S = 0.480
NEURAL_BIN_SMALL_S = 0.1
N_SHUFFLES = 100
SEED_10C = 20260506
SEED_10D = 20260507
ALPHA = 0.01
FDR_ALPHA = 0.05
GLM_Z_THRESHOLD = 2.5
B2_MIN_BINS = 30
B4_MIN_BINS = 5
N_PCS = 5
MIN_SPIKE_TOTAL = 10
GLM_MAX_ITER = 15
GLM_TOL = 1e-6
FAKE_BOUND_MIN = 500
FAKE_BOUND_MAX_OFFSET = 500
EXCLUSION_HALF = 20

NEW_SESSIONS = [4, 6, 8, 14, 16]
S12_NUM = 12
ALL_SESSIONS = sorted(NEW_SESSIONS + [S12_NUM])

POT_ZONE_STATES = {6, 7, 8, 9, 10, 13}   # for replication tabulation


# =====  Generic helpers  =====
def fdr_pass(pvals, q=FDR_ALPHA):
    p = np.asarray(pvals, dtype=np.float64)
    valid = np.isfinite(p)
    sig = np.zeros(p.shape, dtype=bool)
    if valid.sum() == 0:
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
    n_units = rates_100ms.shape[0]
    rates_480 = np.zeros((n_units, n_hmm_bins), dtype=np.float64)
    counts_per = np.bincount(hmm_idx, minlength=n_hmm_bins)
    for u in range(n_units):
        sums = np.bincount(hmm_idx, weights=rates_100ms[u].astype(np.float64),
                            minlength=n_hmm_bins)
        rates_480[u] = sums / np.maximum(counts_per, 1)
    return rates_480 * (1.0 / NEURAL_BIN_SMALL_S)


def compute_pca_projection(rates):
    mu = rates.mean(axis=1, keepdims=True)
    sig = rates.std(axis=1, keepdims=True) + 1e-9
    z = (rates - mu) / sig
    X = z.T
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pcs = U[:, :N_PCS] * S[:N_PCS]
    var = (S ** 2) / (S ** 2).sum()
    return pcs, var[:N_PCS]


# =====  Poisson IRLS (for B3)  =====
def poisson_irls(X, y, offset, max_iter=GLM_MAX_ITER, tol=GLM_TOL):
    n, p = X.shape
    mean_y = max(float(y.mean()), 0.5)
    beta = np.zeros(p)
    if p > 0:
        beta[0] = np.log(mean_y) - float(offset.mean())
    for it in range(max_iter):
        eta = X @ beta + offset
        mu = np.exp(np.clip(eta, -30, 30))
        mu = np.clip(mu, 1e-10, None)
        z = eta + (y - mu) / mu - offset
        Xw = X * mu[:, None]
        XtWX = X.T @ Xw
        XtWz = X.T @ (mu * z)
        try:
            beta_new = np.linalg.solve(XtWX, XtWz)
        except np.linalg.LinAlgError:
            return np.full(p, np.nan), False
        if np.max(np.abs(beta_new - beta)) < tol:
            return beta_new, True
        beta = beta_new
    return beta, True


def poisson_se(X, beta, offset):
    if np.any(~np.isfinite(beta)):
        return np.full(beta.shape, np.nan)
    eta = X @ beta + offset
    mu = np.exp(np.clip(eta, -30, 30))
    mu = np.clip(mu, 1e-10, None)
    XtWX = (X * mu[:, None]).T @ X
    try:
        cov = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        return np.full(beta.shape, np.nan)
    return np.sqrt(np.maximum(np.diag(cov), 0))


# =====  Single-session loader  =====
def load_session(sn, cfg, paths_data):
    """Returns dict with rates, counts, viterbi, posteriors, K, discovery_bin,
    food_pot, metabolic_state, n_units."""
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
    n_hmm_bins = len(trial_time)
    duration_s = float(trial_time[-1] + HMM_BIN_S)
    n_100ms = int(np.ceil(duration_s / NEURAL_BIN_SMALL_S))
    edges_100ms = np.arange(n_100ms + 1) * NEURAL_BIN_SMALL_S
    edges_480ms = np.arange(n_hmm_bins + 1) * HMM_BIN_S

    aca_uid_list = sorted(aca_spikes.keys())
    lha_uid_list = sorted(lha_spikes.keys())
    n_aca = len(aca_uid_list); n_lha = len(lha_uid_list)
    aca_100 = np.zeros((n_aca, n_100ms), dtype=np.float64)
    lha_100 = np.zeros((n_lha, n_100ms), dtype=np.float64)
    aca_counts = np.zeros((n_aca, n_hmm_bins), dtype=np.int64)
    lha_counts = np.zeros((n_lha, n_hmm_bins), dtype=np.int64)
    for i, uid in enumerate(aca_uid_list):
        aca_100[i] = np.histogram(aca_spikes[uid], edges_100ms)[0]
        aca_counts[i] = np.histogram(aca_spikes[uid], edges_480ms)[0]
    for i, uid in enumerate(lha_uid_list):
        lha_100[i] = np.histogram(lha_spikes[uid], edges_100ms)[0]
        lha_counts[i] = np.histogram(lha_spikes[uid], edges_480ms)[0]
    aca_rates = rebin_100ms_to_480ms(aca_100, n_hmm_bins)
    lha_rates = rebin_100ms_to_480ms(lha_100, n_hmm_bins)

    post_csv = REPO_ROOT / cfg["merge_dirs"]["posteriors"] / f"session_{sn}.csv"
    post_df = pd.read_csv(post_csv)
    viterbi = post_df["viterbi"].values.astype(np.int64)
    K = max(int(viterbi.max()) + 1,
            sum(1 for c in post_df.columns if c.startswith("p_state_")))
    posteriors = np.column_stack([post_df[f"p_state_{k}"].values for k in range(K)])

    n = min(n_hmm_bins, len(viterbi))
    aca_rates = aca_rates[:, :n]
    lha_rates = lha_rates[:, :n]
    aca_counts = aca_counts[:, :n]
    lha_counts = lha_counts[:, :n]
    viterbi = viterbi[:n]
    posteriors = posteriors[:n]

    history = pd.read_csv(REPO_ROOT / cfg["commitment_dirs"]["out"]
                           / "sampling_history.csv")
    s_hist = history[history.session == sn].iloc[0]
    discovery_bin = int(s_hist["discovery_bin"])
    metabolic_state = s_hist["state"]
    food_pot = (None if pd.isna(s_hist["food_pot"]) else int(s_hist["food_pot"]))

    return dict(
        sn=sn, n_bins=n, K=K,
        aca_rates=aca_rates, lha_rates=lha_rates,
        aca_counts=aca_counts, lha_counts=lha_counts,
        viterbi=viterbi, posteriors=posteriors,
        discovery_bin=discovery_bin,
        metabolic_state=metabolic_state, food_pot=food_pot,
        n_aca=n_aca, n_lha=n_lha,
    )


# =====  Per-analysis kernels  =====
def b1_anova_per_unit(rates, viterbi, K):
    n_units = rates.shape[0]
    pvals = np.full(n_units, np.nan)
    fvals = np.full(n_units, np.nan)
    group_idx = [np.flatnonzero(viterbi == k) for k in range(K)]
    valid = [g for g in group_idx if len(g) >= 2]
    if len(valid) < 2:
        return pvals, fvals
    for u in range(n_units):
        groups = [rates[u, g] for g in valid]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                f, p = f_oneway(*groups)
                pvals[u] = p; fvals[u] = f
            except Exception:
                pass
    return pvals, fvals


def b1_per_unit_state_means(rates, viterbi, K):
    n_units = rates.shape[0]
    means = np.full((n_units, K), np.nan)
    for k in range(K):
        idx = np.flatnonzero(viterbi == k)
        if not len(idx):
            continue
        means[:, k] = rates[:, idx].mean(axis=1)
    return means


def b2_per_unit_state(rates, viterbi, discovery_bin, K):
    """Returns df with columns unit, state, FR_pre, FR_post, delta, n_pre,
    n_post, p, plus arrays for per-unit FDR sig (any state)."""
    n_units, n_bins = rates.shape
    pre_mask = np.arange(n_bins) < discovery_bin
    rows = []
    for k in range(K):
        in_state = (viterbi == k)
        pre_idx = np.flatnonzero(in_state & pre_mask)
        post_idx = np.flatnonzero(in_state & ~pre_mask)
        n_pre = len(pre_idx); n_post = len(post_idx)
        for u in range(n_units):
            fr_pre = rates[u, pre_idx]
            fr_post = rates[u, post_idx]
            if n_pre >= B2_MIN_BINS and n_post >= B2_MIN_BINS:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        _, p = mannwhitneyu(fr_pre, fr_post, alternative="two-sided")
                    except Exception:
                        p = np.nan
            else:
                p = np.nan
            rows.append(dict(
                unit_id=u, state=k,
                FR_pre=float(fr_pre.mean()) if n_pre else np.nan,
                FR_post=float(fr_post.mean()) if n_post else np.nan,
                delta=(float(fr_post.mean()) if n_post else np.nan)
                       - (float(fr_pre.mean()) if n_pre else np.nan),
                n_pre=n_pre, n_post=n_post,
                p=p,
            ))
    df = pd.DataFrame(rows)
    df["sig_fdr"] = fdr_pass(df["p"].values, FDR_ALPHA)
    return df


def b3_glm_per_unit(counts, posteriors, viterbi, K):
    """Returns df + max_abs_z + sig_any per unit."""
    n_units, n_bins = counts.shape
    state_counts = np.bincount(viterbi, minlength=K)
    ref_state = int(np.argmax(state_counts))
    keep_states = [k for k in range(K) if k != ref_state]
    X = np.column_stack([np.ones(n_bins), posteriors[:, keep_states]])
    offset = np.full(n_bins, np.log(HMM_BIN_S))
    n_keep = len(keep_states)
    z_mat = np.full((n_units, n_keep), np.nan)
    p_mat = np.full((n_units, n_keep), np.nan)
    rows = []
    for u in range(n_units):
        y = counts[u]
        if int(y.sum()) < MIN_SPIKE_TOTAL:
            continue
        beta, ok = poisson_irls(X, y.astype(np.float64), offset)
        if not ok or np.any(~np.isfinite(beta)):
            continue
        se_arr = poisson_se(X, beta, offset)
        if np.any(~np.isfinite(se_arr[1:])):
            continue
        coefs = beta[1:]
        zvals = coefs / np.where(se_arr[1:] > 0, se_arr[1:], 1)
        pvals = 2.0 * norm.sf(np.abs(zvals))
        z_mat[u] = zvals
        p_mat[u] = pvals
        for j, k in enumerate(keep_states):
            rows.append(dict(unit_id=u, state=k,
                              beta=float(coefs[j]),
                              se=float(se_arr[1 + j]),
                              z=float(zvals[j]),
                              p=float(pvals[j])))
    df = pd.DataFrame(rows)
    df["p_fdr"] = np.nan
    valid = df["p"].notna() if len(df) else pd.Series([], dtype=bool)
    if valid.any():
        _, p_adj_arr, _, _ = multipletests(df.loc[valid, "p"].values,
                                              alpha=FDR_ALPHA, method="fdr_bh")
        df.loc[valid, "p_fdr"] = p_adj_arr
    df["sig"] = (df["p_fdr"] < FDR_ALPHA) & (df["z"].abs() > GLM_Z_THRESHOLD)
    max_abs_z = np.nanmax(np.abs(z_mat), axis=1)
    sig_any = np.zeros(n_units, dtype=bool)
    if len(df):
        sig_any_idx = df.loc[df["sig"], "unit_id"].unique()
        sig_any[sig_any_idx] = True
    return df, max_abs_z, sig_any, ref_state


def b4_pre_post_centroid_shift(pcs, viterbi, discovery_bin, K):
    n_bins = pcs.shape[0]
    pre_mask = np.arange(n_bins) < discovery_bin
    shifts = np.full(K, np.nan)
    n_pre_arr = np.zeros(K, dtype=int)
    n_post_arr = np.zeros(K, dtype=int)
    for k in range(K):
        m_pre = (viterbi == k) & pre_mask
        m_post = (viterbi == k) & ~pre_mask
        n_pre_arr[k] = int(m_pre.sum())
        n_post_arr[k] = int(m_post.sum())
        if m_pre.sum() < B4_MIN_BINS or m_post.sum() < B4_MIN_BINS:
            continue
        c_pre = pcs[m_pre, :3].mean(axis=0)
        c_post = pcs[m_post, :3].mean(axis=0)
        shifts[k] = float(np.linalg.norm(c_post - c_pre))
    return shifts, n_pre_arr, n_post_arr


# =====  Shuffle 10c kernel (circular shift)  =====
def b3_shuffle_kernel(counts, posteriors, viterbi, K):
    """Used inside the shuffle loop. Returns max_abs_z + sig_any."""
    df, max_abs_z, sig_any, _ = b3_glm_per_unit(counts, posteriors, viterbi, K)
    return max_abs_z, sig_any


def run_shuffle_10c(session, n_aca, n_lha, K, fig_path, out_dir):
    """100 circular shuffles. Compare B1 ANOVA + B3 sig-coef counts and
    per-unit max |z| to observed."""
    rng = np.random.default_rng(SEED_10C)
    n = session["n_bins"]
    aca_rates = session["aca_rates"]; lha_rates = session["lha_rates"]
    aca_counts = session["aca_counts"]; lha_counts = session["lha_counts"]
    viterbi = session["viterbi"]; posteriors = session["posteriors"]

    # Observed (re-derive with same kernels)
    pa, _ = b1_anova_per_unit(aca_rates, viterbi, K)
    pl, _ = b1_anova_per_unit(lha_rates, viterbi, K)
    obs_b1_aca = int(fdr_pass(pa).sum())
    obs_b1_lha = int(fdr_pass(pl).sum())
    mz_a_real, sig_a_real = b3_shuffle_kernel(aca_counts, posteriors, viterbi, K)
    mz_l_real, sig_l_real = b3_shuffle_kernel(lha_counts, posteriors, viterbi, K)
    obs_b3_aca = int(np.nansum(sig_a_real))
    obs_b3_lha = int(np.nansum(sig_l_real))

    rows = []
    max_z_aca_shuf = np.zeros((N_SHUFFLES, n_aca))
    max_z_lha_shuf = np.zeros((N_SHUFFLES, n_lha))
    for it in range(N_SHUFFLES):
        offset = int(rng.integers(100, n - 100))
        v = np.roll(viterbi, offset)
        post = np.roll(posteriors, offset, axis=0)
        pa_s, _ = b1_anova_per_unit(aca_rates, v, K)
        pl_s, _ = b1_anova_per_unit(lha_rates, v, K)
        mz_a, sig_a = b3_shuffle_kernel(aca_counts, post, v, K)
        mz_l, sig_l = b3_shuffle_kernel(lha_counts, post, v, K)
        rows.append(dict(iter=it, offset=offset,
                          n_b1_aca=int(fdr_pass(pa_s).sum()),
                          n_b1_lha=int(fdr_pass(pl_s).sum()),
                          n_b3_aca=int(np.nansum(sig_a)),
                          n_b3_lha=int(np.nansum(sig_l))))
        max_z_aca_shuf[it] = mz_a
        max_z_lha_shuf[it] = mz_l

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "shuffle_10c_summary.csv", index=False)

    # Per-unit max |z| comparison
    p95_aca = np.nanpercentile(max_z_aca_shuf, 95, axis=0)
    p95_lha = np.nanpercentile(max_z_lha_shuf, 95, axis=0)
    rows_z = []
    for u in range(n_aca):
        rows_z.append(dict(unit_id=u, region="ACA",
                            real_max_abs_z=float(mz_a_real[u]),
                            shuf_p95_max_abs_z=float(p95_aca[u]),
                            exceeds_shuf_p95=bool(mz_a_real[u] > p95_aca[u])))
    for u in range(n_lha):
        rows_z.append(dict(unit_id=u, region="LHA",
                            real_max_abs_z=float(mz_l_real[u]),
                            shuf_p95_max_abs_z=float(p95_lha[u]),
                            exceeds_shuf_p95=bool(mz_l_real[u] > p95_lha[u])))
    df_z = pd.DataFrame(rows_z)
    df_z.to_csv(out_dir / "shuffle_10c_per_unit_max_z.csv", index=False)

    # Distribution figure (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    panels = [
        (axes[0, 0], df["n_b1_aca"].values, obs_b1_aca, n_aca, "B1 — ACA"),
        (axes[0, 1], df["n_b1_lha"].values, obs_b1_lha, n_lha, "B1 — LHA"),
        (axes[1, 0], df["n_b3_aca"].values, obs_b3_aca, n_aca, "B3 — ACA"),
        (axes[1, 1], df["n_b3_lha"].values, obs_b3_lha, n_lha, "B3 — LHA"),
    ]
    for ax, vals, observed, total, label in panels:
        ax.hist(vals, bins=20, color="#9999cc", edgecolor="white")
        ax.axvline(observed, color="red", lw=2, label=f"obs={observed}/{total}")
        pct = float((vals <= observed).mean() * 100.0)
        ax.set_title(f"{label} (obs at {pct:.0f}th pctile)", fontsize=10)
        ax.set_xlabel("# sig units"); ax.set_ylabel("# shuffles")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(f"S{session['sn']} 10c shuffle (circular shift)", y=1.0)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)

    n_above_aca = int((mz_a_real > p95_aca).sum())
    n_above_lha = int((mz_l_real > p95_lha).sum())
    return dict(
        obs_b1_aca=obs_b1_aca, obs_b1_lha=obs_b1_lha,
        obs_b3_aca=obs_b3_aca, obs_b3_lha=obs_b3_lha,
        b1_aca_pctile=float((df["n_b1_aca"].values <= obs_b1_aca).mean() * 100),
        b1_lha_pctile=float((df["n_b1_lha"].values <= obs_b1_lha).mean() * 100),
        b3_aca_pctile=float((df["n_b3_aca"].values <= obs_b3_aca).mean() * 100),
        b3_lha_pctile=float((df["n_b3_lha"].values <= obs_b3_lha).mean() * 100),
        n_above_p95_aca=n_above_aca,
        n_above_p95_lha=n_above_lha,
    )


# =====  Shuffle 10d kernel (fake-discovery)  =====
def b2_one_iter(rates, viterbi, fake_bin, K):
    n_units, n_bins = rates.shape
    pre_mask = np.arange(n_bins) < fake_bin
    pvals = []; keys = []
    for k in range(K):
        in_state = (viterbi == k)
        pre_idx = np.flatnonzero(in_state & pre_mask)
        post_idx = np.flatnonzero(in_state & ~pre_mask)
        if len(pre_idx) < B2_MIN_BINS or len(post_idx) < B2_MIN_BINS:
            continue
        pre_block = rates[:, pre_idx]
        post_block = rates[:, post_idx]
        for u in range(n_units):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    _, p = mannwhitneyu(pre_block[u], post_block[u],
                                         alternative="two-sided")
                except Exception:
                    p = np.nan
            pvals.append(p); keys.append((u, k))
    pvals = np.asarray(pvals)
    sig = fdr_pass(pvals, FDR_ALPHA)
    sig_units = set()
    for (u, _), s in zip(keys, sig):
        if s:
            sig_units.add(u)
    return len(sig_units)


def run_shuffle_10d(session, fig_path_b2, fig_path_b4, out_dir):
    """100 fake-discovery shuffles. Compare B2 + B4 to observed."""
    n = session["n_bins"]
    K = session["K"]
    discovery_bin = session["discovery_bin"]
    aca_rates = session["aca_rates"]; lha_rates = session["lha_rates"]
    viterbi = session["viterbi"]
    pcs_aca, _ = compute_pca_projection(aca_rates)
    pcs_lha, _ = compute_pca_projection(lha_rates)

    obs_b2_aca = b2_one_iter(aca_rates, viterbi, discovery_bin, K)
    obs_b2_lha = b2_one_iter(lha_rates, viterbi, discovery_bin, K)
    obs_b4_aca, _, _ = b4_pre_post_centroid_shift(pcs_aca, viterbi, discovery_bin, K)
    obs_b4_lha, _, _ = b4_pre_post_centroid_shift(pcs_lha, viterbi, discovery_bin, K)

    rng = np.random.default_rng(SEED_10D)
    bnd_lo = max(FAKE_BOUND_MIN, discovery_bin - 5000)  # ensure feasible
    bnd_hi = n - FAKE_BOUND_MAX_OFFSET
    if bnd_lo >= bnd_hi:
        bnd_lo = max(50, n // 6)
        bnd_hi = max(bnd_lo + 200, n - 50)
    excl_lo = discovery_bin - EXCLUSION_HALF
    excl_hi = discovery_bin + EXCLUSION_HALF

    rows_b2 = []; rows_b4 = []
    for it in range(N_SHUFFLES):
        while True:
            fb = int(rng.integers(bnd_lo, bnd_hi))
            if fb < excl_lo or fb > excl_hi:
                break
        n_aca_b2 = b2_one_iter(aca_rates, viterbi, fb, K)
        n_lha_b2 = b2_one_iter(lha_rates, viterbi, fb, K)
        sh_aca, _, _ = b4_pre_post_centroid_shift(pcs_aca, viterbi, fb, K)
        sh_lha, _, _ = b4_pre_post_centroid_shift(pcs_lha, viterbi, fb, K)
        rows_b2.append(dict(iter=it, fake_boundary_bin=fb,
                              n_sig_aca=n_aca_b2, n_sig_lha=n_lha_b2))
        for k in range(K):
            rows_b4.append(dict(iter=it, fake_boundary_bin=fb, region="ACA",
                                  state=k, centroid_shift=float(sh_aca[k])
                                  if np.isfinite(sh_aca[k]) else np.nan))
            rows_b4.append(dict(iter=it, fake_boundary_bin=fb, region="LHA",
                                  state=k, centroid_shift=float(sh_lha[k])
                                  if np.isfinite(sh_lha[k]) else np.nan))

    df_b2 = pd.DataFrame(rows_b2); df_b4 = pd.DataFrame(rows_b4)
    df_b2.to_csv(out_dir / "shuffle_10d_B2_summary.csv", index=False)

    # Per-state significance
    sig_rows = []
    for region, observed in [("ACA", obs_b4_aca), ("LHA", obs_b4_lha)]:
        for k in range(K):
            shuf = df_b4[(df_b4.region == region) & (df_b4.state == k)]["centroid_shift"].dropna().values
            if len(shuf) == 0 or not np.isfinite(observed[k]):
                continue
            obs = float(observed[k])
            shuf_p95 = float(np.percentile(shuf, 95))
            sig_rows.append(dict(region=region, state=k,
                                   observed_shift=obs,
                                   shuffle_mean=float(shuf.mean()),
                                   shuffle_p95=shuf_p95,
                                   obs_pctile=float((shuf <= obs).mean() * 100),
                                   exceeds_p95=bool(obs > shuf_p95)))
    df_sig = pd.DataFrame(sig_rows)
    df_sig.to_csv(out_dir / "shuffle_10d_B4_state_significance.csv", index=False)

    # B2 distribution figure
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    n_aca_units = aca_rates.shape[0]; n_lha_units = lha_rates.shape[0]
    for ax, vals, observed, total, label in [
        (axes[0], df_b2["n_sig_aca"].values, obs_b2_aca, n_aca_units, "ACA"),
        (axes[1], df_b2["n_sig_lha"].values, obs_b2_lha, n_lha_units, "LHA"),
    ]:
        ax.hist(vals, bins=20, color="#9999cc", edgecolor="white")
        ax.axvline(observed, color="red", lw=2, label=f"obs={observed}/{total}")
        pct = float((vals <= observed).mean() * 100.0)
        ax.set_title(f"B2 — {label} (obs at {pct:.0f}th pctile)", fontsize=10)
        ax.set_xlabel("# sig units"); ax.set_ylabel("# shuffles")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(f"S{session['sn']} 10d B2 shuffle (fake discovery)", y=1.0)
    fig.tight_layout()
    fig.savefig(fig_path_b2, dpi=130)
    plt.close(fig)

    # B4 per-state figure
    fig, axes = plt.subplots(2, 1, figsize=(max(8, 0.6 * K + 2), 8), sharex=True)
    for ax, region, observed in [(axes[0], "ACA", obs_b4_aca),
                                    (axes[1], "LHA", obs_b4_lha)]:
        states = np.arange(K)
        obs_arr = np.array(observed, dtype=np.float64)
        ax.bar(states - 0.18, obs_arr, width=0.36,
               color="#cc4444", alpha=0.85, label="observed")
        p95_arr = np.full(K, np.nan); sh_mean = np.full(K, np.nan)
        for k in range(K):
            ss = df_b4[(df_b4.region == region) & (df_b4.state == k)]["centroid_shift"].dropna().values
            if len(ss):
                p95_arr[k] = np.percentile(ss, 95)
                sh_mean[k] = ss.mean()
        ax.bar(states + 0.18, p95_arr, width=0.36,
               color="#888888", alpha=0.85, label="shuffle p95")
        ax.scatter(states + 0.18, sh_mean, color="white",
                    edgecolors="black", s=20, zorder=5, label="shuffle mean")
        for k in range(K):
            if (np.isfinite(obs_arr[k]) and np.isfinite(p95_arr[k])
                    and obs_arr[k] > p95_arr[k]):
                ax.text(k - 0.18, obs_arr[k] + 0.03, "*",
                         ha="center", fontsize=14, color="red")
        ax.set_xticks(states)
        ax.set_xticklabels([f"S{k}" for k in range(K)])
        ax.set_ylabel("PC1-3 centroid shift")
        ax.set_title(f"{region}: pre/post centroid shift (* = obs > p95)")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(axis="y", alpha=0.3)
    axes[1].set_xlabel("Merged HMM state")
    fig.suptitle(f"S{session['sn']} B4 shuffle (fake discovery)", y=1.0)
    fig.tight_layout()
    fig.savefig(fig_path_b4, dpi=130)
    plt.close(fig)

    aca_pass = sorted([int(s) for s in df_sig.loc[
        (df_sig.region == "ACA") & df_sig["exceeds_p95"], "state"]])
    lha_pass = sorted([int(s) for s in df_sig.loc[
        (df_sig.region == "LHA") & df_sig["exceeds_p95"], "state"]])
    return dict(
        obs_b2_aca=obs_b2_aca, obs_b2_lha=obs_b2_lha,
        b2_aca_pctile=float((df_b2["n_sig_aca"].values <= obs_b2_aca).mean() * 100),
        b2_lha_pctile=float((df_b2["n_sig_lha"].values <= obs_b2_lha).mean() * 100),
        aca_b4_pass=aca_pass,
        lha_b4_pass=lha_pass,
    )


# =====  Plots: per-session B1/B2/B4 outputs  =====
def plot_B1_heatmap(mean_mat, region, K, out_path):
    n_units = mean_mat.shape[0]
    if n_units == 0:
        return
    row_means = np.nanmean(mean_mat, axis=1, keepdims=True)
    row_stds = np.nanstd(mean_mat, axis=1, keepdims=True) + 1e-9
    z_mat = (mean_mat - row_means) / row_stds
    pref = np.argmax(np.where(np.isfinite(mean_mat), mean_mat, -np.inf), axis=1)
    order = np.lexsort((np.arange(n_units), pref))
    fig, ax = plt.subplots(figsize=(0.4 * K + 2, 0.04 * n_units + 1.2))
    vmax = np.nanmax(np.abs(z_mat))
    im = ax.imshow(z_mat[order], aspect="auto", cmap="RdBu_r",
                    vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_xticks(np.arange(K))
    ax.set_xticklabels([f"S{k}" for k in range(K)])
    ax.set_xlabel("State")
    ax.set_ylabel(f"{region} units (sorted)")
    plt.colorbar(im, ax=ax, label="z (per row)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_state_preference_counts(pref_aca, pref_lha, K, out_path, sess_label):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, pref, region in [(axes[0], pref_aca, "ACA"),
                              (axes[1], pref_lha, "LHA")]:
        counts = np.bincount(pref, minlength=K)
        ax.bar(np.arange(K), counts, color="#4477aa")
        ax.set_xticks(np.arange(K))
        ax.set_xticklabels([f"S{k}" for k in range(K)])
        ax.set_xlabel("Preferred state")
        ax.set_ylabel("# units")
        ax.set_title(f"{region}: preferred-state counts")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"{sess_label}: B1 preferred-state distribution")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_B4_pca_state(pcs, viterbi, K, region, out_path, var_top, sess_label):
    cmap = plt.cm.tab20 if K <= 20 else plt.cm.gist_ncar
    colors = np.array([cmap(int(k) % cmap.N) for k in viterbi])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (xi, yi) in zip(axes, [(0, 1), (1, 2)]):
        ax.scatter(pcs[:, xi], pcs[:, yi], c=colors, s=4, alpha=0.45,
                    rasterized=True, edgecolors="none")
        ax.set_xlabel(f"PC{xi+1} ({var_top[xi]*100:.1f}%)")
        ax.set_ylabel(f"PC{yi+1} ({var_top[yi]*100:.1f}%)")
        ax.set_title(f"{region}: PC{xi+1} vs PC{yi+1}")
    handles = [plt.Line2D([0], [0], marker="o", linestyle="",
                            markerfacecolor=cmap(k % cmap.N),
                            markersize=8, label=f"S{k}") for k in range(K)]
    axes[1].legend(handles=handles, fontsize=8, loc="upper left",
                    bbox_to_anchor=(1.02, 1.0))
    fig.suptitle(f"{sess_label} {region} PCA, by state")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_B4_pre_post(pcs, viterbi, discovery_bin, K, region, out_path,
                       var_top, sess_label):
    n_bins = pcs.shape[0]
    pre_mask = np.arange(n_bins) < discovery_bin
    cmap = plt.cm.tab20 if K <= 20 else plt.cm.gist_ncar
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (xi, yi) in zip(axes, [(0, 1), (1, 2)]):
        c_pre = np.array([to_rgba(cmap(int(k) % cmap.N), alpha=0.18)
                            for k in viterbi[pre_mask]])
        ax.scatter(pcs[pre_mask, xi], pcs[pre_mask, yi], c=c_pre, s=3,
                    rasterized=True, edgecolors="none")
        c_post = np.array([to_rgba(cmap(int(k) % cmap.N), alpha=0.85)
                            for k in viterbi[~pre_mask]])
        ax.scatter(pcs[~pre_mask, xi], pcs[~pre_mask, yi], c=c_post, s=4,
                    rasterized=True, edgecolors="none")
        ax.set_xlabel(f"PC{xi+1} ({var_top[xi]*100:.1f}%)")
        ax.set_ylabel(f"PC{yi+1} ({var_top[yi]*100:.1f}%)")
        ax.set_title(f"{region}: pre (faint) vs post (solid)")
    fig.suptitle(f"{sess_label} {region} pre/post overlay")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# =====  Per-session orchestrator  =====
def run_session(session, base_out_dir, base_fig_dir):
    sn = session["sn"]
    out_dir = base_out_dir / f"session_{sn}"
    fig_dir = base_fig_dir / f"session_{sn}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    K = session["K"]
    aca_rates = session["aca_rates"]; lha_rates = session["lha_rates"]
    aca_counts = session["aca_counts"]; lha_counts = session["lha_counts"]
    viterbi = session["viterbi"]; posteriors = session["posteriors"]
    discovery_bin = session["discovery_bin"]
    n_aca = session["n_aca"]; n_lha = session["n_lha"]
    label = f"S{sn} ({session['metabolic_state']}, food=" \
            f"{('P'+str(session['food_pot'])) if session['food_pot'] else 'NONE'})"

    print(f"\n--- {label} ---")
    t0 = time.time()

    # B1
    pa, _ = b1_anova_per_unit(aca_rates, viterbi, K)
    pl, _ = b1_anova_per_unit(lha_rates, viterbi, K)
    sig_b1_aca = fdr_pass(pa, FDR_ALPHA)
    sig_b1_lha = fdr_pass(pl, FDR_ALPHA)
    means_aca = b1_per_unit_state_means(aca_rates, viterbi, K)
    means_lha = b1_per_unit_state_means(lha_rates, viterbi, K)
    pref_aca = np.argmax(np.where(np.isfinite(means_aca), means_aca, -np.inf), axis=1)
    pref_lha = np.argmax(np.where(np.isfinite(means_lha), means_lha, -np.inf), axis=1)
    pd.DataFrame(dict(unit_id=np.arange(n_aca), region="ACA",
                       p=pa, sig_fdr=sig_b1_aca,
                       preferred_state=pref_aca)).to_csv(
        out_dir / "B1_selectivity_summary_ACA.csv", index=False)
    pd.DataFrame(dict(unit_id=np.arange(n_lha), region="LHA",
                       p=pl, sig_fdr=sig_b1_lha,
                       preferred_state=pref_lha)).to_csv(
        out_dir / "B1_selectivity_summary_LHA.csv", index=False)
    plot_B1_heatmap(means_aca, "ACA", K, fig_dir / "B1_heatmap_ACA.png")
    plot_B1_heatmap(means_lha, "LHA", K, fig_dir / "B1_heatmap_LHA.png")
    plot_state_preference_counts(pref_aca, pref_lha, K,
                                  fig_dir / "B1_state_preference_counts.png", label)
    print(f"  B1 done (sig: ACA {int(sig_b1_aca.sum())}/{n_aca}, "
          f"LHA {int(sig_b1_lha.sum())}/{n_lha})  [{time.time()-t0:.0f}s]")

    # B2
    df_b2_aca = b2_per_unit_state(aca_rates, viterbi, discovery_bin, K)
    df_b2_lha = b2_per_unit_state(lha_rates, viterbi, discovery_bin, K)
    df_b2_aca.to_csv(out_dir / "B2_pre_vs_post_per_state_ACA.csv", index=False)
    df_b2_lha.to_csv(out_dir / "B2_pre_vs_post_per_state_LHA.csv", index=False)
    n_b2_aca = int(df_b2_aca.loc[df_b2_aca["sig_fdr"], "unit_id"].nunique())
    n_b2_lha = int(df_b2_lha.loc[df_b2_lha["sig_fdr"], "unit_id"].nunique())
    print(f"  B2 done (units with sig: ACA {n_b2_aca}/{n_aca}, "
          f"LHA {n_b2_lha}/{n_lha})  [{time.time()-t0:.0f}s]")

    # B3
    df_b3_aca, _, sig_b3_aca, ref_aca = b3_glm_per_unit(aca_counts, posteriors, viterbi, K)
    df_b3_lha, _, sig_b3_lha, ref_lha = b3_glm_per_unit(lha_counts, posteriors, viterbi, K)
    df_b3_aca.to_csv(out_dir / "B3_glm_coefficients_ACA.csv", index=False)
    df_b3_lha.to_csv(out_dir / "B3_glm_coefficients_LHA.csv", index=False)
    print(f"  B3 done (units with sig: ACA {int(sig_b3_aca.sum())}/{n_aca}, "
          f"LHA {int(sig_b3_lha.sum())}/{n_lha}; ref ACA=S{ref_aca}, "
          f"LHA=S{ref_lha})  [{time.time()-t0:.0f}s]")

    # B4
    pcs_aca, var_aca = compute_pca_projection(aca_rates)
    pcs_lha, var_lha = compute_pca_projection(lha_rates)
    shifts_aca, npre_aca, npost_aca = b4_pre_post_centroid_shift(
        pcs_aca, viterbi, discovery_bin, K)
    shifts_lha, npre_lha, npost_lha = b4_pre_post_centroid_shift(
        pcs_lha, viterbi, discovery_bin, K)
    pd.DataFrame(dict(state=np.arange(K),
                       n_pre=npre_aca, n_post=npost_aca,
                       centroid_shift_PC123=shifts_aca)).to_csv(
        out_dir / "B4_pre_post_centroid_shift_ACA.csv", index=False)
    pd.DataFrame(dict(state=np.arange(K),
                       n_pre=npre_lha, n_post=npost_lha,
                       centroid_shift_PC123=shifts_lha)).to_csv(
        out_dir / "B4_pre_post_centroid_shift_LHA.csv", index=False)
    plot_B4_pca_state(pcs_aca, viterbi, K, "ACA",
                       fig_dir / "B4_pca_state_colored_ACA.png", var_aca, label)
    plot_B4_pca_state(pcs_lha, viterbi, K, "LHA",
                       fig_dir / "B4_pca_state_colored_LHA.png", var_lha, label)
    plot_B4_pre_post(pcs_aca, viterbi, discovery_bin, K, "ACA",
                      fig_dir / "B4_pca_pre_post_ACA.png", var_aca, label)
    plot_B4_pre_post(pcs_lha, viterbi, discovery_bin, K, "LHA",
                      fig_dir / "B4_pca_pre_post_LHA.png", var_lha, label)
    print(f"  B4 done  [{time.time()-t0:.0f}s]")

    # 10c circular shuffle
    print(f"  10c shuffle...", flush=True)
    summ_10c = run_shuffle_10c(session, n_aca, n_lha, K,
                                  fig_dir / "shuffle_10c_distributions.png",
                                  out_dir)
    print(f"  10c done. B1 obs pctile ACA={summ_10c['b1_aca_pctile']:.0f}, "
          f"LHA={summ_10c['b1_lha_pctile']:.0f}; B3 ACA={summ_10c['b3_aca_pctile']:.0f}, "
          f"LHA={summ_10c['b3_lha_pctile']:.0f}; per-unit p95 above: "
          f"ACA {summ_10c['n_above_p95_aca']}/{n_aca}, "
          f"LHA {summ_10c['n_above_p95_lha']}/{n_lha}  [{time.time()-t0:.0f}s]")

    # 10d fake-discovery shuffle
    print(f"  10d shuffle...", flush=True)
    summ_10d = run_shuffle_10d(session,
                                  fig_dir / "shuffle_10d_B2_distributions.png",
                                  fig_dir / "shuffle_10d_B4_per_state.png",
                                  out_dir)
    print(f"  10d done. B2 pctile ACA={summ_10d['b2_aca_pctile']:.0f}, "
          f"LHA={summ_10d['b2_lha_pctile']:.0f}; B4 pass: "
          f"ACA={summ_10d['aca_b4_pass']}, "
          f"LHA={summ_10d['lha_b4_pass']}  [{time.time()-t0:.0f}s]")

    return dict(
        sn=sn,
        metabolic_state=session["metabolic_state"],
        food_pot=session["food_pot"],
        n_aca=n_aca, n_lha=n_lha,
        b1_obs_aca=int(sig_b1_aca.sum()), b1_obs_lha=int(sig_b1_lha.sum()),
        b1_per_unit_above_aca=summ_10c["n_above_p95_aca"],
        b1_per_unit_above_lha=summ_10c["n_above_p95_lha"],
        b3_obs_aca=int(sig_b3_aca.sum()), b3_obs_lha=int(sig_b3_lha.sum()),
        b3_aca_pctile=summ_10c["b3_aca_pctile"],
        b3_lha_pctile=summ_10c["b3_lha_pctile"],
        b2_obs_aca=n_b2_aca, b2_obs_lha=n_b2_lha,
        b2_aca_pctile=summ_10d["b2_aca_pctile"],
        b2_lha_pctile=summ_10d["b2_lha_pctile"],
        b4_aca_pass=",".join(str(s) for s in summ_10d["aca_b4_pass"]),
        b4_lha_pass=",".join(str(s) for s in summ_10d["lha_b4_pass"]),
        n_aca_b4_pass=len(summ_10d["aca_b4_pass"]),
        n_lha_b4_pass=len(summ_10d["lha_b4_pass"]),
        pref_aca=pref_aca, pref_lha=pref_lha,
        K=K,
    )


# =====  S12 result loader  =====
def load_s12_results():
    """Read existing 10b/10c/10d outputs to build the same summary as run_session."""
    cm_dir = REPO_ROOT / "data" / "HMM" / "neural_alignment"
    sc_dir = cm_dir / "state_conditioned_S12"
    sh_c_dir = cm_dir / "shuffle_control_S12"
    sh_d_dir = cm_dir / "shuffle_control_B2_B4_S12"
    history = pd.read_csv(REPO_ROOT / "data" / "HMM" / "commitment_markers"
                           / "sampling_history.csv")
    s12_hist = history[history.session == 12].iloc[0]

    sel = pd.read_csv(sc_dir / "B1_selectivity_summary.csv")
    aca_sel = sel[sel.region == "ACA"]; lha_sel = sel[sel.region == "LHA"]
    n_aca = len(aca_sel); n_lha = len(lha_sel)

    df_z = pd.read_csv(sh_c_dir / "shuffle_B3_max_z_per_unit.csv")
    above_aca = int((df_z[df_z.region == "ACA"]["exceeds_shuf_p95"]).sum())
    above_lha = int((df_z[df_z.region == "LHA"]["exceeds_shuf_p95"]).sum())
    df_b1_shuf = pd.read_csv(sh_c_dir / "shuffle_B1_summary.csv")
    df_b3_shuf = pd.read_csv(sh_c_dir / "shuffle_B3_summary.csv")

    sig_b1_aca = int(aca_sel["sig_fdr"].sum())
    sig_b1_lha = int(lha_sel["sig_fdr"].sum())

    # B3 observed from existing 10b
    b3_aca = pd.read_csv(sc_dir / "B3_glm_coefficients_ACA.csv")
    b3_lha = pd.read_csv(sc_dir / "B3_glm_coefficients_LHA.csv")
    sig_b3_aca = int(b3_aca[b3_aca.sig]["unit_id"].nunique())
    sig_b3_lha = int(b3_lha[b3_lha.sig]["unit_id"].nunique())

    # B2 observed from existing 10b
    b2_aca = pd.read_csv(sc_dir / "B2_pre_vs_post_per_state_ACA.csv")
    b2_lha = pd.read_csv(sc_dir / "B2_pre_vs_post_per_state_LHA.csv")
    n_b2_aca = int(b2_aca[b2_aca.sig_fdr]["unit_id"].nunique())
    n_b2_lha = int(b2_lha[b2_lha.sig_fdr]["unit_id"].nunique())
    df_b2_shuf = pd.read_csv(sh_d_dir / "shuffle_B2_summary.csv")
    b2_aca_pct = float((df_b2_shuf["n_sig_aca"].values <= n_b2_aca).mean() * 100)
    b2_lha_pct = float((df_b2_shuf["n_sig_lha"].values <= n_b2_lha).mean() * 100)

    # B4 pass states from existing 10d
    df_sig = pd.read_csv(sh_d_dir / "shuffle_B4_state_significance.csv")
    aca_pass = sorted([int(s) for s in
                        df_sig[(df_sig.region == "ACA") & df_sig["exceeds_p95"]]["state"]])
    lha_pass = sorted([int(s) for s in
                        df_sig[(df_sig.region == "LHA") & df_sig["exceeds_p95"]]["state"]])

    K = int(max(b3_aca["state"].max(), b3_lha["state"].max())) + 1

    pref_aca = aca_sel["preferred_state"].values
    pref_lha = lha_sel["preferred_state"].values

    obs_b1_aca = sig_b1_aca; obs_b1_lha = sig_b1_lha
    b1_aca_pct = float((df_b1_shuf["n_sig_aca"].values <= obs_b1_aca).mean() * 100)
    b1_lha_pct = float((df_b1_shuf["n_sig_lha"].values <= obs_b1_lha).mean() * 100)
    b3_aca_pct = float((df_b3_shuf["n_sig_aca"].values <= sig_b3_aca).mean() * 100)
    b3_lha_pct = float((df_b3_shuf["n_sig_lha"].values <= sig_b3_lha).mean() * 100)

    return dict(
        sn=12,
        metabolic_state=s12_hist["state"],
        food_pot=int(s12_hist["food_pot"]),
        n_aca=n_aca, n_lha=n_lha,
        b1_obs_aca=sig_b1_aca, b1_obs_lha=sig_b1_lha,
        b1_per_unit_above_aca=above_aca, b1_per_unit_above_lha=above_lha,
        b3_obs_aca=sig_b3_aca, b3_obs_lha=sig_b3_lha,
        b3_aca_pctile=b3_aca_pct, b3_lha_pctile=b3_lha_pct,
        b2_obs_aca=n_b2_aca, b2_obs_lha=n_b2_lha,
        b2_aca_pctile=b2_aca_pct, b2_lha_pctile=b2_lha_pct,
        b4_aca_pass=",".join(str(s) for s in aca_pass),
        b4_lha_pass=",".join(str(s) for s in lha_pass),
        n_aca_b4_pass=len(aca_pass),
        n_lha_b4_pass=len(lha_pass),
        pref_aca=pref_aca, pref_lha=pref_lha,
        K=K,
    )


# =====  Cross-session aggregation  =====
def cross_session_aggregate(per_sess, base_out_dir, base_fig_dir):
    rows = []
    for s in per_sess:
        rows.append({k: v for k, v in s.items()
                      if k not in ("pref_aca", "pref_lha", "K")})
    df = pd.DataFrame(rows)
    df.to_csv(base_out_dir / "cross_session_summary.csv", index=False)
    print(f"\nCross-session summary → {base_out_dir / 'cross_session_summary.csv'}")
    print(df.to_string(index=False))

    # Replication count per state for B4 pass
    K = max(s["K"] for s in per_sess)
    rep_rows = []
    for region in ("ACA", "LHA"):
        col = "b4_aca_pass" if region == "ACA" else "b4_lha_pass"
        per_state_count = np.zeros(K, dtype=int)
        per_state_session_list = [[] for _ in range(K)]
        for s in per_sess:
            states_str = s[col]
            if not states_str:
                continue
            for k in [int(x) for x in states_str.split(",") if x]:
                per_state_count[k] += 1
                per_state_session_list[k].append(s["sn"])
        for k in range(K):
            rep_rows.append(dict(region=region, state=k,
                                   n_sessions_passing=int(per_state_count[k]),
                                   sessions=",".join(str(x) for x in per_state_session_list[k])))
    rep_df = pd.DataFrame(rep_rows)
    rep_df.to_csv(base_out_dir / "replication_count_per_state.csv", index=False)

    # Replication heatmap
    sess_order = sorted([s["sn"] for s in per_sess])
    for region in ("ACA", "LHA"):
        col = "b4_aca_pass" if region == "ACA" else "b4_lha_pass"
        mat = np.zeros((K, len(sess_order)), dtype=int)
        for j, sn in enumerate(sess_order):
            s = next(s for s in per_sess if s["sn"] == sn)
            states_str = s[col]
            if not states_str:
                continue
            for k in [int(x) for x in states_str.split(",") if x]:
                mat[k, j] = 1
        fig, ax = plt.subplots(figsize=(2 + 0.5 * len(sess_order),
                                          0.4 * K + 1.5))
        ax.imshow(mat, aspect="auto", cmap="Reds", vmin=0, vmax=1,
                   interpolation="nearest")
        ax.set_xticks(np.arange(len(sess_order)))
        # Color session ticks by state
        sess_states = {s["sn"]: s["metabolic_state"] for s in per_sess}
        labels = []
        for sn in sess_order:
            colsts = sess_states[sn]
            tag = "fed" if colsts == "fed" else "fas"
            labels.append(f"S{sn}\n{tag}")
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_yticks(np.arange(K))
        ax.set_yticklabels([f"S{k}" for k in range(K)])
        ax.set_xlabel("Session")
        ax.set_ylabel("HMM state")
        ax.set_title(f"{region}: B4 pre/post centroid shift passes p95\n"
                     "(red = passes shuffle null in that session)")
        for i in range(K):
            for j in range(len(sess_order)):
                if mat[i, j]:
                    ax.text(j, i, "✓", ha="center", va="center",
                             fontsize=11, color="white", fontweight="bold")
        fig.tight_layout()
        fig.savefig(base_fig_dir / f"replication_heatmap_{region}.png", dpi=130)
        plt.close(fig)

    # Preferred-state counts grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharey=False)
    for ax, s in zip(axes.flat, sorted(per_sess, key=lambda x: x["sn"])):
        K_s = s["K"]
        cnt_aca = np.bincount(s["pref_aca"], minlength=K_s)
        cnt_lha = np.bincount(s["pref_lha"], minlength=K_s)
        x = np.arange(K_s); w = 0.4
        ax.bar(x - w/2, cnt_aca, width=w, color="#4477aa", label=f"ACA n={s['n_aca']}")
        ax.bar(x + w/2, cnt_lha, width=w, color="#cc6677", label=f"LHA n={s['n_lha']}")
        ax.set_xticks(x); ax.set_xticklabels([f"S{k}" for k in range(K_s)], fontsize=8)
        ax.set_xlabel("Preferred state")
        ax.set_ylabel("# units")
        col = "blue" if s["metabolic_state"] == "fed" else "red"
        ax.set_title(f"S{s['sn']} ({s['metabolic_state']}, food="
                     f"{('P'+str(s['food_pot'])) if s['food_pot'] else 'NONE'})",
                     color=col)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("B1 preferred-state distribution per session "
                 "(blue title = fed, red = fasted)", y=1.0)
    fig.tight_layout()
    fig.savefig(base_fig_dir / "preferred_state_counts_all_sessions.png", dpi=130)
    plt.close(fig)

    # Fed vs fasted aggregate dots
    metric_specs = [
        ("b1_obs_aca", "B1 ACA sig units"),
        ("b1_obs_lha", "B1 LHA sig units"),
        ("b1_per_unit_above_aca", "B1 ACA units > 10c p95"),
        ("b1_per_unit_above_lha", "B1 LHA units > 10c p95"),
        ("b2_obs_aca", "B2 ACA sig units"),
        ("b2_obs_lha", "B2 LHA sig units"),
        ("n_aca_b4_pass", "B4 ACA states passing p95"),
        ("n_lha_b4_pass", "B4 LHA states passing p95"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, (m, lab) in zip(axes.flat, metric_specs):
        for s in per_sess:
            color = "#4477aa" if s["metabolic_state"] == "fed" else "#cc6677"
            ax.scatter([0 if s["metabolic_state"] == "fed" else 1],
                        [s[m]], color=color, s=80, edgecolors="black",
                        linewidths=0.6)
            ax.text(0.05 if s["metabolic_state"] == "fed" else 1.05,
                     s[m], f"S{s['sn']}", fontsize=8, va="center")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["fed", "fasted"])
        ax.set_ylabel(lab)
        ax.set_xlim(-0.4, 1.4)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Per-session metric values by metabolic state "
                 "(no formal stats — n=3 vs 3)", y=1.0)
    fig.tight_layout()
    fig.savefig(base_fig_dir / "fed_vs_fasted_aggregate_metrics.png", dpi=130)
    plt.close(fig)


# =====  Main  =====
def main():
    cfg = load_config()
    base_out = REPO_ROOT / "data" / "HMM" / "neural_alignment" / "track_B_all_sessions"
    base_fig = REPO_ROOT / "figures" / "HMM" / "neural_alignment" / "track_B_all_sessions"
    base_out.mkdir(parents=True, exist_ok=True)
    base_fig.mkdir(parents=True, exist_ok=True)

    with open(REPO_ROOT / cfg["paths_yaml"]) as f:
        paths_data = yaml.safe_load(f)

    per_sess = []

    # Run new sessions
    for sn in NEW_SESSIONS:
        print(f"\n========== Loading S{sn} ==========")
        try:
            session = load_session(sn, cfg, paths_data)
        except Exception as e:
            print(f"  ERROR loading S{sn}: {e}")
            continue
        result = run_session(session, base_out, base_fig)
        per_sess.append(result)

    # Load S12
    print(f"\n========== Loading S12 from existing 10b/10c/10d outputs ==========")
    try:
        s12 = load_s12_results()
        per_sess.append(s12)
        print(f"  S12 loaded: B1 obs ACA={s12['b1_obs_aca']}/{s12['n_aca']}, "
              f"LHA={s12['b1_obs_lha']}/{s12['n_lha']}; B4 pass "
              f"ACA={s12['b4_aca_pass']}, LHA={s12['b4_lha_pass']}")
    except Exception as e:
        print(f"  ERROR loading S12: {e}")

    # Aggregate
    print("\n========== Cross-session aggregation ==========")
    cross_session_aggregate(per_sess, base_out, base_fig)

    # Headline summary
    print("\n========== HEADLINE: replication of B4 pot-zone passes ==========")
    print("Across sessions, count of fed/fasted sessions where each state passes B4 shuffle p95:")
    for region in ("ACA", "LHA"):
        col = "b4_aca_pass" if region == "ACA" else "b4_lha_pass"
        per_state_fed = np.zeros(20, dtype=int)
        per_state_fasted = np.zeros(20, dtype=int)
        for s in per_sess:
            if not s[col]:
                continue
            states = [int(x) for x in s[col].split(",") if x]
            for k in states:
                if s["metabolic_state"] == "fed":
                    per_state_fed[k] += 1
                else:
                    per_state_fasted[k] += 1
        K = max(s["K"] for s in per_sess)
        print(f"\n  {region}:")
        print(f"    {'state':>5}  {'fed':>3}  {'fasted':>6}  pot-zone?")
        for k in range(K):
            tag = " (pot-zone)" if k in POT_ZONE_STATES else ""
            print(f"    S{k:<4}  {per_state_fed[k]:>3}  {per_state_fasted[k]:>6}{tag}")


if __name__ == "__main__":
    main()
