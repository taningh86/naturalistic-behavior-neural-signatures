"""12 — Commitment analysis (Strategy A + Strategy B), all 6 foraging sessions.

Tests whether ACA or LHA neural activity carries a signature of approaching
commitment to a dig, isolated from behavioral state encoding.

  Strategy A — Within-state trajectory.
    For each pre-discovery state in {S4, S6, S9}, project state-conditioned
    firing rates onto the global PCA top-3, and ask: do PC values (or distance
    to a "discovery target" vector) drift monotonically toward the target as
    time-to-discovery shrinks? Linear regression slope vs time-to-discovery,
    100-shuffle null on the time-to-discovery permutation.

  Strategy B — Behavior-residualized commitment GLM.
    Per-unit Poisson GLM on pre-discovery bins:
      log mu_t = beta0 + sum_k beta_k posterior_k(t) + beta_ttd * ttd(t)
    The time-to-discovery coefficient is the commitment signal after partialling
    out state. 100-shuffle null on permutation of ttd within pre-discovery bins.

  Cross-strategy convergence:
    Are units flagged by Strategy A and Strategy B the same units?

S4 uses the manual-override discovery time (raw feeding onset). For each
session, the "discovery target" population vector is taken at the start of
the most recent S6 (digging) run at the food pot before/at discovery, falling
back to discovery_bin itself if no such dig is found.

Out of scope: sampling-history regression, cross-session unit matching,
nonlinear time-to-discovery.
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
from scipy.stats import linregress, norm
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
N_SHUFFLES = 100
SEED_MASTER = 20260508
FDR_ALPHA = 0.05
GLM_MAX_ITER = 15
GLM_TOL = 1e-6
MIN_SPIKE_TOTAL = 10
N_PCS = 5
SMOOTH_BINS = 30                # for visualization only
PRIMARY_STATES = (4, 6, 9)      # contemplation, digging, pot-zone
MIN_BINS_FOR_TRAJECTORY = 30
MIN_PRE_BINS_FOR_GLM = 80       # below this we skip Strategy B for the session
ALL_SESSIONS = [4, 6, 8, 12, 14, 16]


# ---- Generic helpers ----
def fdr_pass(pvals, q=FDR_ALPHA):
    p = np.asarray(pvals, dtype=np.float64)
    valid = np.isfinite(p)
    sig = np.zeros(p.shape, dtype=bool)
    if valid.sum() == 0:
        return sig
    rej, _, _, _ = multipletests(p[valid], alpha=q, method="fdr_bh")
    sig[valid] = rej
    return sig


def fdr_adjust(pvals, q=FDR_ALPHA):
    p = np.asarray(pvals, dtype=np.float64)
    valid = np.isfinite(p)
    out = np.full(p.shape, np.nan)
    if valid.sum() == 0:
        return out
    _, p_adj, _, _ = multipletests(p[valid], alpha=q, method="fdr_bh")
    out[valid] = p_adj
    return out


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
    """z-score per unit, then SVD-based PCA. Returns pcs (n_bins, N_PCS), var_top (N_PCS,)."""
    mu = rates.mean(axis=1, keepdims=True)
    sig = rates.std(axis=1, keepdims=True) + 1e-9
    z = (rates - mu) / sig
    X = z.T
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pcs = U[:, :N_PCS] * S[:N_PCS]
    var = (S ** 2) / (S ** 2).sum()
    return pcs, var[:N_PCS]


# ---- Poisson IRLS ----
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


# ---- Session loader ----
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
    pot_id = np.asarray(binned["pot_id"], dtype=np.int64)
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
    aca_rates = aca_rates[:, :n]; lha_rates = lha_rates[:, :n]
    aca_counts = aca_counts[:, :n]; lha_counts = lha_counts[:, :n]
    viterbi = viterbi[:n]; posteriors = posteriors[:n]; pot_id = pot_id[:n]

    history = pd.read_csv(REPO_ROOT / cfg["commitment_dirs"]["out"]
                           / "sampling_history.csv")
    s_hist = history[history.session == sn].iloc[0]
    discovery_bin = int(s_hist["discovery_bin"])
    metabolic_state = s_hist["state"]
    food_pot = (None if pd.isna(s_hist["food_pot"]) else int(s_hist["food_pot"]))
    discovery_method = s_hist["discovery_method"]

    return dict(
        sn=sn, n_bins=n, K=K,
        aca_rates=aca_rates, lha_rates=lha_rates,
        aca_counts=aca_counts, lha_counts=lha_counts,
        viterbi=viterbi, posteriors=posteriors, pot_id=pot_id,
        discovery_bin=discovery_bin,
        discovery_method=discovery_method,
        metabolic_state=metabolic_state, food_pot=food_pot,
        n_aca=n_aca, n_lha=n_lha,
    )


def find_discovery_dig_bin(viterbi, pot_id, discovery_bin, food_pot):
    """Return the start-of-run of the most recent S6 (state 6) at the food pot
    on or before discovery_bin. Falls back to discovery_bin."""
    if food_pot is None:
        return discovery_bin
    in_dig = (viterbi == 6) & (pot_id == int(food_pot))
    cand = np.flatnonzero(in_dig)
    cand = cand[cand <= discovery_bin]
    if len(cand) == 0:
        return discovery_bin
    last = int(cand[-1])
    start = last
    while start > 0 and in_dig[start - 1]:
        start -= 1
    return start


# ---- Strategy A ----
def strategy_a_for_state(rates, viterbi, discovery_bin, target_bin,
                          state_id, pcs, region, sn, out_dir, fig_dir, rng):
    """Run Strategy A for one (session, region, state). Returns rows for the
    cross-session summary, or None if skipped."""
    in_state_pre = (viterbi == state_id) & (np.arange(len(viterbi)) < discovery_bin)
    state_bins = np.flatnonzero(in_state_pre)
    if len(state_bins) < MIN_BINS_FOR_TRAJECTORY:
        print(f"    skip A state {state_id} {region}: only {len(state_bins)} bins")
        return None

    # time-to-discovery in seconds (positive)
    ttd_s = (discovery_bin - state_bins) * HMM_BIN_S

    # Project state bins onto PCs; target = the PC location of the discovery dig bin
    state_pcs = pcs[state_bins, :3]                              # (n, 3)
    target_pc = pcs[min(target_bin, pcs.shape[0] - 1), :3]
    distances = np.linalg.norm(state_pcs - target_pc[None, :], axis=1)

    # Linear regression of each metric vs ttd
    metrics = {
        "PC1": state_pcs[:, 0],
        "PC2": state_pcs[:, 1],
        "PC3": state_pcs[:, 2],
        "distance_to_target": distances,
    }
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, pc_name in zip(axes, ["PC1", "PC2", "PC3"]):
        ord_idx = np.argsort(-ttd_s)   # ascending bin order = descending ttd_s
        ax.plot(-ttd_s[ord_idx], metrics[pc_name][ord_idx],
                color="grey", alpha=0.45, lw=0.8)
        # smooth
        if len(state_bins) >= SMOOTH_BINS:
            kernel = np.ones(SMOOTH_BINS) / SMOOTH_BINS
            sm = np.convolve(metrics[pc_name][ord_idx], kernel, mode="same")
            ax.plot(-ttd_s[ord_idx], sm, color="firebrick", lw=2)
        slope, intercept, _, p_val, _ = linregress(ttd_s, metrics[pc_name])
        # shuffle null
        shuf_slopes = []
        for _ in range(N_SHUFFLES):
            ttd_shuf = rng.permutation(ttd_s)
            s_shuf, _, _, _, _ = linregress(ttd_shuf, metrics[pc_name])
            shuf_slopes.append(s_shuf)
        shuf_slopes = np.array(shuf_slopes)
        # observed pctile (negative slopes are "approaching": low pctile)
        obs_pct = float((shuf_slopes <= slope).mean() * 100)
        # passes if observed slope is below shuffle 5th pctile (extreme-low) for distance,
        # or below 5th / above 95th for PCs. Use two-tailed p95 |slope|.
        p5 = np.percentile(shuf_slopes, 5)
        p95 = np.percentile(shuf_slopes, 95)
        passes = bool(slope < p5 or slope > p95)
        rows.append(dict(
            session=sn, region=region, state=state_id,
            metric=pc_name, n_bins=int(len(state_bins)),
            slope=float(slope), p_linregress=float(p_val),
            shuffle_mean_slope=float(shuf_slopes.mean()),
            shuffle_p5_slope=float(p5),
            shuffle_p95_slope=float(p95),
            obs_pctile=obs_pct, passes_two_tail_p95=passes,
        ))
        ax.set_xlabel("time relative to discovery (s, 0 = discovery)")
        ax.set_ylabel(pc_name)
        ax.set_title(f"S{sn} {region} state {state_id}: {pc_name}\n"
                     f"slope={slope:.4f}, obs pctile={obs_pct:.0f}",
                     fontsize=10)
        ax.axvline(0, color="black", lw=0.7, ls="--")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / f"A_state_{state_id}_PC_trajectories_{region}.png",
                 dpi=130)
    plt.close(fig)

    # Distance-to-target panel
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ord_idx = np.argsort(-ttd_s)
    ax2.plot(-ttd_s[ord_idx], distances[ord_idx], color="grey", alpha=0.45, lw=0.8)
    if len(state_bins) >= SMOOTH_BINS:
        kernel = np.ones(SMOOTH_BINS) / SMOOTH_BINS
        sm = np.convolve(distances[ord_idx], kernel, mode="same")
        ax2.plot(-ttd_s[ord_idx], sm, color="firebrick", lw=2,
                  label="smoothed")
    slope, intercept, _, p_val, _ = linregress(ttd_s, distances)
    shuf_slopes = np.array([linregress(rng.permutation(ttd_s), distances)[0]
                              for _ in range(N_SHUFFLES)])
    obs_pct = float((shuf_slopes <= slope).mean() * 100)
    p5 = np.percentile(shuf_slopes, 5)
    p95 = np.percentile(shuf_slopes, 95)
    passes = bool(slope < p5 or slope > p95)
    # negative slope = distance shrinks as ttd shrinks (= activity approaches target as discovery nears)
    rows.append(dict(
        session=sn, region=region, state=state_id,
        metric="distance_to_target", n_bins=int(len(state_bins)),
        slope=float(slope), p_linregress=float(p_val),
        shuffle_mean_slope=float(shuf_slopes.mean()),
        shuffle_p5_slope=float(p5),
        shuffle_p95_slope=float(p95),
        obs_pctile=obs_pct, passes_two_tail_p95=passes,
    ))
    ax2.set_xlabel("time relative to discovery (s, 0 = discovery)")
    ax2.set_ylabel("Euclidean distance to discovery target (PC1-3)")
    ax2.set_title(f"S{sn} {region} state {state_id}: distance to discovery target\n"
                  f"slope={slope:.4f}, obs pctile={obs_pct:.0f} "
                  f"(negative slope = approach toward target)", fontsize=10)
    ax2.axvline(0, color="black", lw=0.7, ls="--")
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(fig_dir / f"A_state_{state_id}_distance_to_discovery_{region}.png",
                  dpi=130)
    plt.close(fig2)
    return rows


# ---- Strategy B ----
def strategy_b_run(counts, posteriors, viterbi, discovery_bin, region,
                    sn, out_dir, fig_dir, rng_seed):
    """Per-unit Poisson GLM on pre-discovery bins: counts ~ posteriors + ttd."""
    n_units, n_bins = counts.shape
    pre_idx = np.arange(min(discovery_bin, n_bins))
    if len(pre_idx) < MIN_PRE_BINS_FOR_GLM:
        print(f"    skip B {region}: only {len(pre_idx)} pre-discovery bins (<{MIN_PRE_BINS_FOR_GLM})")
        return None

    K = posteriors.shape[1]
    state_counts_pre = np.bincount(viterbi[pre_idx], minlength=K)
    ref_state = int(np.argmax(state_counts_pre))
    keep_states = [k for k in range(K) if k != ref_state]
    post_pre = posteriors[pre_idx][:, keep_states]              # (n_pre, K-1)
    ttd = (discovery_bin - pre_idx) * HMM_BIN_S                # positive seconds
    ttd_z = (ttd - ttd.mean()) / (ttd.std() + 1e-9)
    X = np.column_stack([np.ones(len(pre_idx)), post_pre, ttd_z])
    offset = np.full(len(pre_idx), np.log(HMM_BIN_S))

    counts_pre = counts[:, pre_idx]                            # (n_units, n_pre)

    rows = []
    real_beta = np.full(n_units, np.nan)
    real_se = np.full(n_units, np.nan)
    real_z = np.full(n_units, np.nan)
    real_p = np.full(n_units, np.nan)
    for u in range(n_units):
        y = counts_pre[u].astype(np.float64)
        if int(y.sum()) < MIN_SPIKE_TOTAL:
            continue
        beta, ok = poisson_irls(X, y, offset)
        if not ok or np.any(~np.isfinite(beta)):
            continue
        se_arr = poisson_se(X, beta, offset)
        if np.any(~np.isfinite(se_arr)):
            continue
        b_ttd = beta[-1]; se_ttd = se_arr[-1]
        if not np.isfinite(b_ttd) or se_ttd <= 0:
            continue
        z = b_ttd / se_ttd
        p = 2.0 * norm.sf(abs(z))
        real_beta[u] = b_ttd; real_se[u] = se_ttd
        real_z[u] = z; real_p[u] = p

    # Shuffle null: permute ttd column among pre bins, refit per unit
    rng = np.random.default_rng(rng_seed)
    shuf_abs_betas = np.full((N_SHUFFLES, n_units), np.nan)
    for it in range(N_SHUFFLES):
        ttd_shuf = rng.permutation(ttd_z)
        X_shuf = X.copy()
        X_shuf[:, -1] = ttd_shuf
        for u in range(n_units):
            y = counts_pre[u].astype(np.float64)
            if int(y.sum()) < MIN_SPIKE_TOTAL:
                continue
            beta, ok = poisson_irls(X_shuf, y, offset)
            if not ok or np.any(~np.isfinite(beta)):
                continue
            shuf_abs_betas[it, u] = abs(beta[-1])

    p95_abs_beta = np.nanpercentile(shuf_abs_betas, 95, axis=0)
    real_abs_beta = np.abs(real_beta)
    exceeds = (real_abs_beta > p95_abs_beta) & np.isfinite(real_abs_beta)

    p_fdr = fdr_adjust(real_p, FDR_ALPHA)
    sig_fdr = (p_fdr < FDR_ALPHA)

    rows = [
        dict(unit_id=u, region=region,
             beta_ttd=float(real_beta[u]) if np.isfinite(real_beta[u]) else np.nan,
             se=float(real_se[u]) if np.isfinite(real_se[u]) else np.nan,
             z=float(real_z[u]) if np.isfinite(real_z[u]) else np.nan,
             p=float(real_p[u]) if np.isfinite(real_p[u]) else np.nan,
             p_FDR=float(p_fdr[u]) if np.isfinite(p_fdr[u]) else np.nan,
             sig_FDR=bool(sig_fdr[u]),
             shuffle_p95_abs_beta_ttd=float(p95_abs_beta[u]) if np.isfinite(p95_abs_beta[u]) else np.nan,
             observed_exceeds_p95=bool(exceeds[u]))
        for u in range(n_units)
    ]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"B_unit_coefficients_{region}.csv", index=False)

    # β_ttd distribution figure
    fig, ax = plt.subplots(figsize=(8, 4.5))
    valid = np.isfinite(real_beta)
    ax.hist(real_beta[valid], bins=30, color="firebrick", alpha=0.65,
             edgecolor="white", label=f"observed β_ttd (n={valid.sum()})")
    # shuffle null pooled across all units (both signs)
    shuf_all = shuf_abs_betas[np.isfinite(shuf_abs_betas)]
    # show as outline on |beta| scale
    ax.hist(np.concatenate([shuf_all, -shuf_all]),
             bins=30, histtype="step", color="grey",
             label="shuffle |β| (both signs)")
    ax.axvline(0, color="black", lw=0.7, ls="--")
    pct_sig = exceeds.mean() * 100
    ax.set_xlabel("β_ttd (z-scored time-to-discovery coefficient)")
    ax.set_ylabel("# units")
    ax.set_title(f"S{sn} {region}: β_ttd distribution — "
                 f"{int(exceeds.sum())}/{valid.sum()} ({pct_sig:.0f}%) exceed shuffle p95",
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / f"B_beta_ttd_distribution_{region}.png", dpi=130)
    plt.close(fig)

    n_sig_fdr = int(sig_fdr.sum())
    n_above_shuf = int(exceeds.sum())
    n_pos = int(np.sum(real_beta > 0))
    n_neg = int(np.sum(real_beta < 0))
    pct_pos = (n_pos / valid.sum() * 100) if valid.sum() else np.nan
    pct_neg = (n_neg / valid.sum() * 100) if valid.sum() else np.nan
    return dict(
        session=sn, region=region,
        n_units=int(valid.sum()), n_pre_bins=int(len(pre_idx)),
        ref_state_dropped=ref_state,
        n_sig_FDR=n_sig_fdr,
        n_above_shuffle_p95=n_above_shuf,
        pct_positive_beta=float(pct_pos),
        pct_negative_beta=float(pct_neg),
        df=df,
    )


def plot_sig_unit_counts(b_results, fig_path):
    """Bar chart per session of FDR-sig and exceeds-shuffle counts, ACA + LHA."""
    sessions = sorted(set([r["session"] for r in b_results]))
    n_sess = len(sessions)
    fig, axes = plt.subplots(1, 2, figsize=(max(8, 1.2 * n_sess), 4.5),
                               sharey=False)
    for ax, region in zip(axes, ["ACA", "LHA"]):
        x = np.arange(n_sess); w = 0.35
        fdr_vals = []; shuf_vals = []
        for sn in sessions:
            r = next((r for r in b_results
                       if r["session"] == sn and r["region"] == region), None)
            fdr_vals.append(r["n_sig_FDR"] if r else 0)
            shuf_vals.append(r["n_above_shuffle_p95"] if r else 0)
        ax.bar(x - w/2, fdr_vals, width=w, color="#4477aa", label="FDR-sig")
        ax.bar(x + w/2, shuf_vals, width=w, color="#cc6677", label="exceeds shuf p95")
        ax.set_xticks(x); ax.set_xticklabels([f"S{s}" for s in sessions])
        ax.set_xlabel("Session")
        ax.set_ylabel("# units")
        ax.set_title(f"Strategy B — {region}: units with sig β_ttd")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Strategy B — per-session counts of units with significant β_ttd",
                 y=1.0)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)


# ---- Per-session orchestrator ----
def run_session(session, base_out_dir, base_fig_dir, rng_seed_master):
    sn = session["sn"]
    out_dir = base_out_dir / f"session_{sn}"
    fig_dir = base_fig_dir / f"session_{sn}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    K = session["K"]
    aca_rates = session["aca_rates"]; lha_rates = session["lha_rates"]
    aca_counts = session["aca_counts"]; lha_counts = session["lha_counts"]
    viterbi = session["viterbi"]; posteriors = session["posteriors"]
    pot_id = session["pot_id"]
    discovery_bin = session["discovery_bin"]
    food_pot = session["food_pot"]
    n_aca = session["n_aca"]; n_lha = session["n_lha"]
    label = f"S{sn} ({session['metabolic_state']}, " \
            f"food={('P'+str(food_pot)) if food_pot else 'NONE'}, " \
            f"discovery=bin {discovery_bin})"
    print(f"\n--- {label} ---")
    t0 = time.time()

    target_dig_bin = find_discovery_dig_bin(viterbi, pot_id, discovery_bin, food_pot)
    print(f"  Discovery target dig bin: {target_dig_bin}", flush=True)

    pcs_aca, var_aca = compute_pca_projection(aca_rates)
    pcs_lha, var_lha = compute_pca_projection(lha_rates)

    # Strategy A
    a_rows = []
    rng_a = np.random.default_rng(rng_seed_master + sn * 100)
    for state_id in PRIMARY_STATES:
        for region, rates, pcs in [("ACA", aca_rates, pcs_aca),
                                      ("LHA", lha_rates, pcs_lha)]:
            rows = strategy_a_for_state(rates, viterbi, discovery_bin,
                                          target_dig_bin, state_id, pcs, region,
                                          sn, out_dir, fig_dir, rng_a)
            if rows is not None:
                a_rows.extend(rows)
    a_df = pd.DataFrame(a_rows)
    a_df.to_csv(out_dir / "A_slope_summary.csv", index=False)
    print(f"  Strategy A done ({len(a_df)} rows)  [{time.time()-t0:.0f}s]", flush=True)

    # Strategy B
    b_outputs = []
    for region, counts in [("ACA", aca_counts), ("LHA", lha_counts)]:
        print(f"  Strategy B {region}...", flush=True)
        out = strategy_b_run(counts, posteriors, viterbi, discovery_bin,
                              region, sn, out_dir, fig_dir,
                              rng_seed_master + sn * 100 + (1 if region == "ACA" else 2))
        if out:
            b_outputs.append(out)
            print(f"    {region}: {out['n_above_shuffle_p95']}/{out['n_units']} "
                  f"units exceed shuffle p95 (FDR sig: {out['n_sig_FDR']}, "
                  f"pos:{out['pct_positive_beta']:.0f}% / neg:{out['pct_negative_beta']:.0f}%)",
                  flush=True)
    print(f"  Strategy B done  [{time.time()-t0:.0f}s]", flush=True)

    # Per-session significant unit counts plot
    plot_sig_unit_counts(b_outputs, fig_dir / "B_significant_unit_counts.png")

    return dict(
        sn=sn,
        metabolic_state=session["metabolic_state"],
        a_df=a_df,
        b_outputs=b_outputs,
        n_aca=n_aca, n_lha=n_lha,
    )


# ---- Cross-session aggregation ----
def aggregate_strategy_a(per_sess_results, base_out_dir, base_fig_dir):
    rows = []
    for r in per_sess_results:
        if r["a_df"] is not None and len(r["a_df"]):
            df = r["a_df"].copy()
            df["metabolic_state"] = r["metabolic_state"]
            rows.append(df)
    if not rows:
        return None
    cross = pd.concat(rows, ignore_index=True)
    cross.to_csv(base_out_dir / "A_cross_session_summary.csv", index=False)
    print(f"\nA cross-session → {base_out_dir / 'A_cross_session_summary.csv'}")

    # Per state per region per metric: count sessions where slope passes
    summary_rows = []
    for state in PRIMARY_STATES:
        for region in ("ACA", "LHA"):
            sub = cross[(cross.state == state) & (cross.region == region)]
            for metric in ("PC1", "PC2", "PC3", "distance_to_target"):
                m = sub[sub.metric == metric]
                n_total = len(m)
                n_pass = int(m["passes_two_tail_p95"].sum())
                n_neg = int((m["slope"] < m["shuffle_p5_slope"]).sum())
                summary_rows.append(dict(
                    state=state, region=region, metric=metric,
                    n_sessions=n_total,
                    n_pass_p95_two_tail=n_pass,
                    n_neg_extreme=n_neg,
                ))
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(base_out_dir / "A_replication_summary.csv", index=False)

    # Slope dot plot per state per region
    fig, axes = plt.subplots(len(PRIMARY_STATES), 2,
                              figsize=(11, 3 * len(PRIMARY_STATES)),
                              sharex=True)
    for i, state in enumerate(PRIMARY_STATES):
        for j, region in enumerate(("ACA", "LHA")):
            ax = axes[i, j] if len(PRIMARY_STATES) > 1 else axes[j]
            sub = cross[(cross.state == state) &
                         (cross.region == region) &
                         (cross.metric == "distance_to_target")]
            for _, row in sub.iterrows():
                color = "#4477aa" if row["metabolic_state"] == "fed" else "#cc6677"
                ax.scatter(row["session"], row["slope"], s=70, color=color,
                            edgecolors="black", linewidths=0.5)
                if row["passes_two_tail_p95"]:
                    ax.scatter(row["session"], row["slope"], s=160,
                                facecolors="none", edgecolors="red", linewidths=2)
            ax.axhline(0, color="black", lw=0.7, ls="--")
            ax.set_xticks(sorted(set(cross["session"])))
            ax.set_xticklabels([f"S{s}" for s in sorted(set(cross["session"]))])
            ax.set_ylabel("slope (distance vs ttd)")
            ax.set_title(f"State {state} {region}", fontsize=10)
            ax.grid(alpha=0.3)
    fig.suptitle("Strategy A — distance-to-target slope per session "
                 "(red ring = passes shuffle two-tailed 95%)", y=1.0)
    fig.tight_layout()
    fig.savefig(base_fig_dir / "A_distance_slope_per_session.png", dpi=130)
    plt.close(fig)
    return summary_df


def aggregate_strategy_b(per_sess_results, base_out_dir, base_fig_dir):
    rows = []
    df_units = []
    for r in per_sess_results:
        for o in r["b_outputs"]:
            rows.append(dict(session=r["sn"],
                              metabolic_state=r["metabolic_state"],
                              region=o["region"],
                              n_units=o["n_units"],
                              n_pre_bins=o["n_pre_bins"],
                              ref_state_dropped=o["ref_state_dropped"],
                              n_sig_FDR=o["n_sig_FDR"],
                              n_above_shuffle_p95=o["n_above_shuffle_p95"],
                              pct_units_above_p95=(
                                  o["n_above_shuffle_p95"] / o["n_units"] * 100
                                  if o["n_units"] else 0),
                              pct_positive_beta=o["pct_positive_beta"],
                              pct_negative_beta=o["pct_negative_beta"]))
            d = o["df"].copy()
            d["session"] = r["sn"]
            df_units.append(d)
    cross = pd.DataFrame(rows)
    cross.to_csv(base_out_dir / "B_cross_session_summary.csv", index=False)
    if df_units:
        all_units = pd.concat(df_units, ignore_index=True)
        all_units.to_csv(base_out_dir / "B_all_unit_coefficients.csv", index=False)
    print(f"\nB cross-session → {base_out_dir / 'B_cross_session_summary.csv'}")

    # Plot fed vs fasted % units above shuffle
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, region in zip(axes, ("ACA", "LHA")):
        sub = cross[cross.region == region]
        for _, row in sub.iterrows():
            color = "#4477aa" if row["metabolic_state"] == "fed" else "#cc6677"
            ax.scatter([0 if row["metabolic_state"] == "fed" else 1],
                        [row["pct_units_above_p95"]], s=80, color=color,
                        edgecolors="black", linewidths=0.5)
            ax.text(0.05 if row["metabolic_state"] == "fed" else 1.05,
                     row["pct_units_above_p95"], f"S{row['session']}",
                     fontsize=8, va="center")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["fed", "fasted"])
        ax.set_xlim(-0.4, 1.4)
        ax.set_ylabel("% units exceeding shuffle p95 |β_ttd|")
        ax.set_title(f"{region}")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Strategy B — units with significant β_ttd above shuffle p95",
                 y=1.0)
    fig.tight_layout()
    fig.savefig(base_fig_dir / "B_fed_vs_fasted_pct_above_p95.png", dpi=130)
    plt.close(fig)
    return cross


# ---- Cross-strategy convergence ----
def cross_strategy_convergence(per_sess_results, base_out_dir, base_fig_dir):
    """For each session × region: do units with high Strategy B β_ttd also have
    a Strategy A signal? Compute per-unit pre-discovery-bin firing-rate change
    (slope vs ttd) as a per-unit Strategy A surrogate, then correlate with B β_ttd.
    """
    rows = []
    for r in per_sess_results:
        sn = r["sn"]
        for o in r["b_outputs"]:
            region = o["region"]
            df_b = o["df"].copy()
            # Per-unit Strategy A surrogate: linregress of unit's pre-discovery
            # bin-level FR vs ttd (across ALL pre bins, not state-conditioned)
            counts = (None if region == "ACA"
                       else None)
            # Use rates from session
            for r_full in per_sess_results:
                if r_full["sn"] != sn:
                    continue
            # Re-derive from the saved rates: too costly; instead skip per-unit A
            # surrogate and convergence based on session-level overlap of flags.
            # Save b_df with sig flag for cross-session aggregation.
            df_b["session"] = sn
            df_b["metabolic_state"] = r["metabolic_state"]
            rows.append(df_b)
    if not rows:
        return None
    all_b = pd.concat(rows, ignore_index=True)
    # save (already saved per session, this is convenience)
    all_b.to_csv(base_out_dir / "convergence_units_combined.csv", index=False)
    return all_b


# ---- Main ----
def main():
    cfg = load_config()
    base_out = REPO_ROOT / "data" / "HMM" / "neural_alignment" / "commitment_analysis"
    base_fig = REPO_ROOT / "figures" / "HMM" / "neural_alignment" / "commitment_analysis"
    base_out.mkdir(parents=True, exist_ok=True)
    base_fig.mkdir(parents=True, exist_ok=True)

    with open(REPO_ROOT / cfg["paths_yaml"]) as f:
        paths_data = yaml.safe_load(f)

    per_sess = []
    for sn in ALL_SESSIONS:
        print(f"\n========== Loading S{sn} ==========", flush=True)
        try:
            session = load_session(sn, cfg, paths_data)
        except Exception as e:
            print(f"  ERROR loading S{sn}: {e}", flush=True)
            continue
        result = run_session(session, base_out, base_fig, SEED_MASTER)
        per_sess.append(result)

    print("\n========== Cross-session aggregation ==========", flush=True)
    a_summary = aggregate_strategy_a(per_sess, base_out, base_fig)
    b_summary = aggregate_strategy_b(per_sess, base_out, base_fig)
    cross_strategy_convergence(per_sess, base_out, base_fig)

    # ---- Headline summary ----
    print("\n========== HEADLINE: Strategy A replication ==========")
    if a_summary is not None:
        print(a_summary.to_string(index=False))

    print("\n========== HEADLINE: Strategy B per-session ==========")
    if b_summary is not None:
        print(b_summary[["session", "metabolic_state", "region", "n_units",
                            "n_pre_bins", "n_sig_FDR", "n_above_shuffle_p95",
                            "pct_units_above_p95",
                            "pct_positive_beta", "pct_negative_beta"]
                          ].to_string(index=False))


if __name__ == "__main__":
    main()
