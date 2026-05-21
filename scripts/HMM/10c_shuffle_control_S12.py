"""10c — Shuffle control for Track B (S12 only, B1 + B3).

Tests whether the pervasive state-encoding result from 10b survives the
destruction of temporal alignment between neural activity and HMM states.

Strategy: circularly shift the Viterbi sequence and the posterior matrix by a
random offset in [100, T-100] bins. Circular shifts preserve state dwell-time
and marginal occupancy but destroy the bin-by-bin alignment with neural data.
Re-run B1 (per-unit ANOVA) and B3 (Poisson GLM on posteriors) under each of
100 such shuffles, with FDR within region.

Implementation notes:
- B3 uses a custom vectorized Poisson IRLS (~5-10 ms/fit). statsmodels is too
  slow for 25,400 fits in serial. The IRLS gives the same MLE as
  sm.GLM(... family=Poisson()) on convergence; SE is from the inverse Fisher
  information (X^T W X) at the MLE.
- Observed B1/B3 stats are re-computed within this script using the same
  IRLS path so the observed and shuffled distributions are apples-to-apples.
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
from scipy.stats import f_oneway
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
SESSION_NUM = 12
HMM_BIN_S = 0.480
NEURAL_BIN_SMALL_S = 0.1
N_SHUFFLES = 100
SHUFFLE_SEED = 20260506
SHUFFLE_MIN_OFFSET = 100        # bins
SHUFFLE_MARGIN = 100            # bins reserved on the high side
ALPHA = 0.01                    # B1 uncorrected
FDR_ALPHA = 0.05
GLM_Z_THRESHOLD = 2.5
MIN_SPIKE_TOTAL = 10            # skip B3 if total < this
GLM_MAX_ITER = 15
GLM_TOL = 1e-6


# ---- Fast Poisson IRLS ----
def poisson_irls(X, y, offset, max_iter=GLM_MAX_ITER, tol=GLM_TOL):
    """Poisson regression: log mu = X @ beta + offset.

    Returns (beta, converged). beta is np.nan if it failed.
    """
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
    """Standard errors from inverse Fisher info at the MLE."""
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


def chi2_pvalue_from_z(z):
    """Two-sided Wald p from |z|. Using survival of standard normal."""
    from scipy.stats import norm
    return 2.0 * norm.sf(np.abs(z))


# ---- Helpers ----
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


# ---- B1 + B3 wrappers ----
def b1_anova(rates, viterbi, K):
    """Per-unit one-way ANOVA across states.
    Returns p-value array (n_units,)."""
    n_units = rates.shape[0]
    pvals = np.full(n_units, np.nan)
    # Pre-compute group indices once
    group_idx = [np.flatnonzero(viterbi == k) for k in range(K)]
    valid_groups = [g for g in group_idx if len(g) >= 2]
    if len(valid_groups) < 2:
        return pvals
    for u in range(n_units):
        groups = [rates[u, g] for g in valid_groups]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                _, p = f_oneway(*groups)
                pvals[u] = p
            except Exception:
                pass
    return pvals


def b3_glm(counts, posteriors, viterbi, K):
    """Poisson GLM per unit with most-occupied state dropped as ref.

    Returns:
      max_abs_z: (n_units,)         max |z| across non-ref states
      sig_any: (n_units,) bool      ≥1 sig coefficient (FDR within region AND |z|>threshold)
    """
    n_units, n_bins = counts.shape
    state_counts = np.bincount(viterbi, minlength=K)
    ref_state = int(np.argmax(state_counts))
    keep_states = [k for k in range(K) if k != ref_state]
    X_post = posteriors[:, keep_states]
    X = np.column_stack([np.ones(n_bins), X_post])
    offset = np.full(n_bins, np.log(HMM_BIN_S))

    n_keep = len(keep_states)
    z_mat = np.full((n_units, n_keep), np.nan)
    p_mat = np.full((n_units, n_keep), np.nan)
    for u in range(n_units):
        y = counts[u]
        if int(y.sum()) < MIN_SPIKE_TOTAL:
            continue
        beta, ok = poisson_irls(X, y.astype(np.float64), offset)
        if not ok or np.any(~np.isfinite(beta)):
            continue
        se = poisson_se(X, beta, offset)
        if np.any(~np.isfinite(se[1:])):
            continue
        coefs = beta[1:]
        zvals = coefs / np.where(se[1:] > 0, se[1:], 1)
        z_mat[u] = zvals
        p_mat[u] = chi2_pvalue_from_z(zvals)

    # Per-unit FDR over coefficients flattened, then check threshold
    max_abs_z = np.nanmax(np.abs(z_mat), axis=1)
    p_flat = p_mat.flatten()
    sig_flat = fdr_pass(p_flat, FDR_ALPHA)
    sig_mat = sig_flat.reshape(z_mat.shape) & (np.abs(z_mat) > GLM_Z_THRESHOLD)
    sig_any = sig_mat.any(axis=1)
    return max_abs_z, sig_any, ref_state


# ---- Main ----
def main():
    cfg = load_config()
    out_dir = REPO_ROOT / "data" / "HMM" / "neural_alignment" / f"shuffle_control_S{SESSION_NUM}"
    fig_dir = REPO_ROOT / "figures" / "HMM" / "neural_alignment" / f"shuffle_control_S{SESSION_NUM}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Resolve session paths
    with open(REPO_ROOT / cfg["paths_yaml"]) as f:
        paths_data = yaml.safe_load(f)
    s12_paths = (paths_data["double_probe"]["coordinates_1"]["mouse01"]
                  ["sessions"][f"session_{SESSION_NUM}"])
    aca_sorted = Path(s12_paths["probe_0_aca"]["sorted"])
    lha_sorted = Path(s12_paths["probe_1_lha_rsp"]["sorted"])

    # Load good units and spikes
    print("=== Loading neural data ===")
    aca_uids = [int(u) for u in get_good_units_p0(aca_sorted)]
    lha_uids = [int(u) for u in get_good_units_p1_lha(lha_sorted)]
    aca_sorting = se.read_kilosort(aca_sorted)
    lha_sorting = se.read_kilosort(lha_sorted)
    aca_uids = [u for u in aca_uids if u in set(aca_sorting.get_unit_ids())]
    lha_uids = [u for u in lha_uids if u in set(lha_sorting.get_unit_ids())]
    aca_spikes = load_spike_times_for_region(aca_sorting, aca_uids)
    lha_spikes = load_spike_times_for_region(lha_sorting, lha_uids)
    n_aca = len(aca_uids); n_lha = len(lha_uids)
    print(f"  ACA: {n_aca} units, LHA: {n_lha} units")

    # Bin to 480 ms (rates Hz from 100 ms aggregation; counts directly)
    binned = np.load(REPO_ROOT / cfg["out_dirs"]["binned"]
                      / f"session_{SESSION_NUM}.npz", allow_pickle=True)
    trial_time = np.asarray(binned["trial_time"], dtype=np.float64)
    n_hmm_bins = len(trial_time)
    duration_s = float(trial_time[-1] + HMM_BIN_S)
    n_100ms = int(np.ceil(duration_s / NEURAL_BIN_SMALL_S))
    edges_100ms = np.arange(n_100ms + 1) * NEURAL_BIN_SMALL_S
    edges_480ms = np.arange(n_hmm_bins + 1) * HMM_BIN_S

    aca_uid_list = sorted(aca_spikes.keys())
    lha_uid_list = sorted(lha_spikes.keys())
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

    # Load merged Viterbi + posteriors
    post_csv = REPO_ROOT / cfg["merge_dirs"]["posteriors"] / f"session_{SESSION_NUM}.csv"
    post_df = pd.read_csv(post_csv)
    viterbi = post_df["viterbi"].values.astype(np.int64)
    K = max(int(viterbi.max()) + 1,
            sum(1 for c in post_df.columns if c.startswith("p_state_")))
    posteriors = np.column_stack([post_df[f"p_state_{k}"].values for k in range(K)])

    # Align
    n = min(n_hmm_bins, len(viterbi))
    aca_rates = aca_rates[:, :n]; lha_rates = lha_rates[:, :n]
    aca_counts = aca_counts[:, :n]; lha_counts = lha_counts[:, :n]
    viterbi = viterbi[:n]
    posteriors = posteriors[:n]
    print(f"  Bins: {n}, K={K}, "
          f"max offset window: [{SHUFFLE_MIN_OFFSET}, {n - SHUFFLE_MARGIN}]")

    # ---- Observed (real, no shuffle) ----
    print("\n=== Observed (real) ===")
    t0 = time.time()
    pvals_aca = b1_anova(aca_rates, viterbi, K)
    pvals_lha = b1_anova(lha_rates, viterbi, K)
    sig_b1_aca_real = int(fdr_pass(pvals_aca).sum())
    sig_b1_lha_real = int(fdr_pass(pvals_lha).sum())
    print(f"  B1 FDR-sig: ACA {sig_b1_aca_real}/{n_aca}, LHA {sig_b1_lha_real}/{n_lha}")

    max_z_aca_real, sig_b3_aca_real_mask, ref_aca = b3_glm(
        aca_counts, posteriors, viterbi, K)
    max_z_lha_real, sig_b3_lha_real_mask, ref_lha = b3_glm(
        lha_counts, posteriors, viterbi, K)
    sig_b3_aca_real = int(np.nansum(sig_b3_aca_real_mask))
    sig_b3_lha_real = int(np.nansum(sig_b3_lha_real_mask))
    print(f"  B3 FDR-sig (|z|>{GLM_Z_THRESHOLD}): "
          f"ACA {sig_b3_aca_real}/{n_aca}, LHA {sig_b3_lha_real}/{n_lha}")
    print(f"  Reference state dropped (most occupied): ACA=S{ref_aca}, LHA=S{ref_lha}")
    print(f"  Observed pass: {time.time()-t0:.1f} s")

    # ---- Shuffles ----
    print(f"\n=== {N_SHUFFLES} circular-shift shuffles ===")
    rng = np.random.default_rng(SHUFFLE_SEED)
    shuffle_b1 = []   # rows: dict iter, n_sig_aca, n_sig_lha, offset
    shuffle_b3 = []   # rows: dict iter, n_sig_aca, n_sig_lha, offset
    max_z_aca_shuf = np.zeros((N_SHUFFLES, n_aca), dtype=np.float64)
    max_z_lha_shuf = np.zeros((N_SHUFFLES, n_lha), dtype=np.float64)

    t0 = time.time()
    for it in range(N_SHUFFLES):
        offset = int(rng.integers(SHUFFLE_MIN_OFFSET, n - SHUFFLE_MARGIN))
        v_shuf = np.roll(viterbi, offset)
        post_shuf = np.roll(posteriors, offset, axis=0)

        # B1
        pa = b1_anova(aca_rates, v_shuf, K)
        pl = b1_anova(lha_rates, v_shuf, K)
        n_sig_b1_a = int(fdr_pass(pa).sum())
        n_sig_b1_l = int(fdr_pass(pl).sum())
        shuffle_b1.append(dict(iter=it, offset=offset,
                                 n_sig_aca=n_sig_b1_a,
                                 n_sig_lha=n_sig_b1_l))

        # B3
        mz_a, sig_a, _ = b3_glm(aca_counts, post_shuf, v_shuf, K)
        mz_l, sig_l, _ = b3_glm(lha_counts, post_shuf, v_shuf, K)
        n_sig_b3_a = int(np.nansum(sig_a))
        n_sig_b3_l = int(np.nansum(sig_l))
        shuffle_b3.append(dict(iter=it, offset=offset,
                                 n_sig_aca=n_sig_b3_a,
                                 n_sig_lha=n_sig_b3_l))
        max_z_aca_shuf[it] = mz_a
        max_z_lha_shuf[it] = mz_l

        if (it + 1) % 10 == 0 or it == 0:
            elapsed = time.time() - t0
            est_total = elapsed * N_SHUFFLES / (it + 1)
            print(f"  iter {it+1:>3}/{N_SHUFFLES}  offset={offset:>5}  "
                  f"B1 sig: ACA {n_sig_b1_a:>3} LHA {n_sig_b1_l:>3}  "
                  f"B3 sig: ACA {n_sig_b3_a:>3} LHA {n_sig_b3_l:>3}  "
                  f"({elapsed:.0f}s elapsed, ~{est_total:.0f}s total)")

    df_b1 = pd.DataFrame(shuffle_b1)
    df_b3 = pd.DataFrame(shuffle_b3)
    df_b1.to_csv(out_dir / "shuffle_B1_summary.csv", index=False)
    df_b3.to_csv(out_dir / "shuffle_B3_summary.csv", index=False)
    print(f"\n  B1 summary → {out_dir / 'shuffle_B1_summary.csv'}")
    print(f"  B3 summary → {out_dir / 'shuffle_B3_summary.csv'}")

    # ---- Per-unit max |z| comparison ----
    p95_aca = np.nanpercentile(max_z_aca_shuf, 95, axis=0)
    p95_lha = np.nanpercentile(max_z_lha_shuf, 95, axis=0)
    rows = []
    for u in range(n_aca):
        rows.append(dict(unit_id=u, region="ACA",
                          real_max_abs_z=float(max_z_aca_real[u]),
                          shuf_p95_max_abs_z=float(p95_aca[u]),
                          exceeds_shuf_p95=bool(max_z_aca_real[u] > p95_aca[u])))
    for u in range(n_lha):
        rows.append(dict(unit_id=u, region="LHA",
                          real_max_abs_z=float(max_z_lha_real[u]),
                          shuf_p95_max_abs_z=float(p95_lha[u]),
                          exceeds_shuf_p95=bool(max_z_lha_real[u] > p95_lha[u])))
    df_z = pd.DataFrame(rows)
    df_z.to_csv(out_dir / "shuffle_B3_max_z_per_unit.csv", index=False)
    print(f"  max |z| comparison → {out_dir / 'shuffle_B3_max_z_per_unit.csv'}")

    # ---- Distribution figures ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [
        (axes[0, 0], df_b1["n_sig_aca"].values, sig_b1_aca_real, n_aca,
         "B1 — ACA", "n FDR-sig units (ANOVA)"),
        (axes[0, 1], df_b1["n_sig_lha"].values, sig_b1_lha_real, n_lha,
         "B1 — LHA", "n FDR-sig units (ANOVA)"),
        (axes[1, 0], df_b3["n_sig_aca"].values, sig_b3_aca_real, n_aca,
         "B3 — ACA", f"n units with sig coef (FDR & |z|>{GLM_Z_THRESHOLD})"),
        (axes[1, 1], df_b3["n_sig_lha"].values, sig_b3_lha_real, n_lha,
         "B3 — LHA", f"n units with sig coef (FDR & |z|>{GLM_Z_THRESHOLD})"),
    ]
    for ax, vals, observed, total, label, xlab in panels:
        ax.hist(vals, bins=20, color="#9999cc", edgecolor="white", alpha=0.85)
        ax.axvline(observed, color="red", lw=2,
                    label=f"observed = {observed}/{total}")
        # Percentile of observed within shuffle null
        pct = float((vals <= observed).mean() * 100.0)
        ax.set_title(f"{label}\nshuffled mean={vals.mean():.1f} "
                     f"[95% CI {np.percentile(vals,2.5):.1f}-"
                     f"{np.percentile(vals,97.5):.1f}], "
                     f"observed at {pct:.1f}th pctile",
                     fontsize=10)
        ax.set_xlabel(xlab)
        ax.set_ylabel("# shuffles")
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(alpha=0.3)
    fig.suptitle(f"S{SESSION_NUM} shuffle control — Track B "
                 f"({N_SHUFFLES} circular-shift iterations)",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(fig_dir / "shuffle_distributions.png", dpi=140)
    plt.close(fig)

    # ---- max |z| scatter ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, real, p95, region, total in [
        (axes[0], max_z_aca_real, p95_aca, "ACA", n_aca),
        (axes[1], max_z_lha_real, p95_lha, "LHA", n_lha),
    ]:
        n_above = int((real > p95).sum())
        ax.scatter(p95, real, s=18, color="#4477aa", alpha=0.65,
                    edgecolors="none")
        lim = max(np.nanmax(p95), np.nanmax(real)) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=1)
        ax.set_xlabel("95th percentile of shuffled max |z|")
        ax.set_ylabel("observed max |z|")
        ax.set_title(f"{region}: {n_above}/{total} units exceed shuffle 95th pctile "
                     f"({n_above/total*100:.1f}%)")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.grid(alpha=0.3)
    fig.suptitle("Per-unit max |z| comparison: observed vs shuffled 95th percentile",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "shuffle_max_z_comparison.png", dpi=140)
    plt.close(fig)

    # ---- Final summary ----
    print("\n========== SUMMARY ==========")
    for region, vals_b1, vals_b3, obs_b1, obs_b3, n_real_max, n_p95, n_total in [
        ("ACA", df_b1["n_sig_aca"].values, df_b3["n_sig_aca"].values,
         sig_b1_aca_real, sig_b3_aca_real,
         max_z_aca_real, p95_aca, n_aca),
        ("LHA", df_b1["n_sig_lha"].values, df_b3["n_sig_lha"].values,
         sig_b1_lha_real, sig_b3_lha_real,
         max_z_lha_real, p95_lha, n_lha),
    ]:
        print(f"\n{region} ({n_total} units):")
        print(f"  B1 observed: {obs_b1}/{n_total} FDR-sig units")
        print(f"  B1 shuffled: mean={vals_b1.mean():.1f}, "
              f"95% CI [{np.percentile(vals_b1, 2.5):.0f}, "
              f"{np.percentile(vals_b1, 97.5):.0f}], "
              f"range [{vals_b1.min()}, {vals_b1.max()}]")
        b1_pct = float((vals_b1 <= obs_b1).mean() * 100)
        print(f"  B1 observed percentile within null: {b1_pct:.1f}%")

        print(f"  B3 observed: {obs_b3}/{n_total} units with sig coef")
        print(f"  B3 shuffled: mean={vals_b3.mean():.1f}, "
              f"95% CI [{np.percentile(vals_b3, 2.5):.0f}, "
              f"{np.percentile(vals_b3, 97.5):.0f}], "
              f"range [{vals_b3.min()}, {vals_b3.max()}]")
        b3_pct = float((vals_b3 <= obs_b3).mean() * 100)
        print(f"  B3 observed percentile within null: {b3_pct:.1f}%")

        n_exceed = int((n_real_max > n_p95).sum())
        print(f"  B3 max |z|: {n_exceed}/{n_total} units exceed shuffle 95th pctile "
              f"({n_exceed/n_total*100:.1f}%)")

    print(f"\nDone. Outputs in {out_dir} and {fig_dir}")


if __name__ == "__main__":
    main()
