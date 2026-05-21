"""10b — Track B: state-conditioned neural analyses for Session 12.

Asks: do single units or population structure encode HMM behavioral states
across the whole session, regardless of event timing?

Four analyses, all on session 12 (fasted, food P4, discovery t=597.1 s):
  B1: per-unit state-selectivity (one-way ANOVA across states)
  B2: pre vs post-discovery within-state shift (Mann-Whitney per state)
  B3: Poisson GLM with state posteriors as predictors
  B4: PCA trajectories colored by Viterbi state (and pre/post overlay)

Neural binning matches the HMM bin grid (480 ms). Spikes are re-binned by
averaging the 100 ms count series produced by Track A, mapped to 480 ms HMM
bins via 100 ms bin-center membership.

Same QC-filtered units as Track A (165 ACA, 89 LHA).
"""
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from scipy.stats import f_oneway, mannwhitneyu
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm

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
NEURAL_BIN_SMALL_S = 0.1     # Track A's 100 ms grid (we'll re-bin from this)
HMM_BIN_S = 0.480            # 480 ms behavioral bin
ALPHA = 0.01                 # uncorrected p threshold for B1
FDR_ALPHA = 0.05             # FDR Q for B1/B2/B3
MIN_BINS_PER_STATE = 30      # B2 minimum bins per side per state
N_PCS = 5
GLM_Z_THRESHOLD = 2.5        # for B3 highlight (post-FDR)
N_BOOT_GLM_FALLBACK = 0      # not used; statsmodels analytical SEs


# ---- Re-binning ----
def rebin_100ms_to_480ms(rates_100ms, n_hmm_bins):
    """Average 100 ms spike counts within each 480 ms HMM bin (by 100 ms
    bin-center membership). Output is in Hz (mean counts/100 ms × 10).

    rates_100ms: (n_units, n_100ms) spike counts per 100 ms bin.
    Returns: (n_units, n_hmm_bins) firing rate in Hz.
    """
    n_100ms = rates_100ms.shape[1]
    centers_s = (np.arange(n_100ms) + 0.5) * NEURAL_BIN_SMALL_S
    # For each 100 ms center, which 480 ms bin does it fall in?
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
    return rates_480 * (1.0 / NEURAL_BIN_SMALL_S)   # to Hz


# ---- Helpers ----
def fdr_mask(pvals, q=FDR_ALPHA):
    """BH-FDR. NaN-safe: NaNs treated as non-significant."""
    p = np.asarray(pvals, dtype=np.float64)
    valid = np.isfinite(p)
    out_p = np.full(p.shape, np.nan)
    out_sig = np.zeros(p.shape, dtype=bool)
    if valid.sum() == 0:
        return out_sig, out_p
    rej, p_adj, _, _ = multipletests(p[valid], alpha=q, method="fdr_bh")
    out_sig[valid] = rej
    out_p[valid] = p_adj
    return out_sig, out_p


def state_color_map(K):
    cmap = plt.cm.tab20 if K <= 20 else plt.cm.gist_ncar
    return {k: cmap(k % cmap.N) for k in range(K)}


# ---- B1: state selectivity ----
def run_B1(rates, viterbi, region, K, out_dir, fig_dir):
    """Per-unit ANOVA across states + units × states heatmap."""
    n_units, _ = rates.shape

    # Mean / SE per state for each unit
    mean_mat = np.full((n_units, K), np.nan)
    se_mat = np.full((n_units, K), np.nan)
    n_per_state = np.zeros(K, dtype=int)
    for k in range(K):
        idx = np.flatnonzero(viterbi == k)
        n_per_state[k] = len(idx)
        if len(idx) == 0:
            continue
        sub = rates[:, idx]
        mean_mat[:, k] = sub.mean(axis=1)
        se_mat[:, k] = sub.std(axis=1, ddof=1) / np.sqrt(len(idx))

    # ANOVA per unit
    pvals = np.full(n_units, np.nan)
    fvals = np.full(n_units, np.nan)
    for u in range(n_units):
        groups = [rates[u, viterbi == k] for k in range(K) if (viterbi == k).sum() >= 2]
        if len(groups) < 2:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                f, p = f_oneway(*groups)
                pvals[u] = p
                fvals[u] = f
            except Exception:
                pass
    sig_uncorr = np.where(np.isfinite(pvals) & (pvals < ALPHA))[0]
    sig_fdr_mask, p_fdr = fdr_mask(pvals, FDR_ALPHA)
    sig_fdr = np.flatnonzero(sig_fdr_mask)

    # Preferred / second-preferred state
    pref = np.argmax(np.where(np.isfinite(mean_mat), mean_mat, -np.inf), axis=1)
    second = np.zeros(n_units, dtype=int)
    for u in range(n_units):
        order = np.argsort(-np.where(np.isfinite(mean_mat[u]),
                                      mean_mat[u], -np.inf))
        second[u] = int(order[1]) if len(order) > 1 else int(order[0])

    # Save matrices
    states_cols = [f"state_{k}" for k in range(K)]
    raw_df = pd.DataFrame(mean_mat, columns=states_cols)
    raw_df.insert(0, "unit_id", np.arange(n_units))
    raw_df.insert(1, "region", region)
    raw_df.to_csv(out_dir / f"B1_state_selectivity_matrix_{region}.csv", index=False)

    # Z-score per unit (row-wise) for visualization
    row_means = np.nanmean(mean_mat, axis=1, keepdims=True)
    row_stds = np.nanstd(mean_mat, axis=1, keepdims=True) + 1e-9
    z_mat = (mean_mat - row_means) / row_stds
    z_df = pd.DataFrame(z_mat, columns=[f"z_state_{k}" for k in range(K)])
    z_df.insert(0, "unit_id", np.arange(n_units))
    z_df.insert(1, "region", region)
    z_df.to_csv(out_dir / f"B1_state_selectivity_zscored_{region}.csv", index=False)

    # Selectivity summary
    summary = pd.DataFrame(dict(
        unit_id=np.arange(n_units),
        region=region,
        f_value=fvals,
        p_value_anova=pvals,
        p_fdr=p_fdr,
        sig_uncorr=(np.isfinite(pvals) & (pvals < ALPHA)),
        sig_fdr=sig_fdr_mask,
        preferred_state=pref,
        second_state=second,
        n_bins_total=int((viterbi >= 0).sum()),
    ))

    # ---- Heatmap (units × states), z-scored, sorted by preferred state ----
    order = np.lexsort((np.arange(n_units), pref))
    fig, ax = plt.subplots(figsize=(0.4 * K + 2, 0.04 * n_units + 1.2))
    vmax = np.nanmax(np.abs(z_mat))
    im = ax.imshow(z_mat[order], aspect="auto", cmap="RdBu_r",
                    vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_xticks(np.arange(K))
    ax.set_xticklabels([f"S{k}" for k in range(K)])
    ax.set_xlabel("Merged HMM state")
    ax.set_ylabel(f"{region} units (sorted by preferred state)")
    ax.set_title(f"{region}: state-selectivity z-scored mean FR (n_units={n_units})")
    plt.colorbar(im, ax=ax, label="z-score (per row)")
    fig.tight_layout()
    fig.savefig(fig_dir / f"B1_heatmap_{region}.png", dpi=140)
    plt.close(fig)

    return summary, mean_mat, n_per_state


def plot_state_preference_counts(summary_aca, summary_lha, K, n_per_state_aca,
                                  n_per_state_lha, fig_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for ax, summary, region, n_per in [
        (axes[0], summary_aca, "ACA", n_per_state_aca),
        (axes[1], summary_lha, "LHA", n_per_state_lha),
    ]:
        counts = np.bincount(summary["preferred_state"].values, minlength=K)
        ax.bar(np.arange(K), counts, color="#4477aa", alpha=0.85)
        # Overlay state-FDR-significant counts
        sig = summary[summary["sig_fdr"]]
        sig_counts = np.bincount(sig["preferred_state"].values, minlength=K)
        ax.bar(np.arange(K), sig_counts, color="#cc4444", alpha=0.95,
               label="FDR-sig units")
        ax.set_xticks(np.arange(K))
        ax.set_xticklabels([f"S{k}" for k in range(K)], rotation=0)
        ax.set_xlabel("Preferred state")
        ax.set_ylabel("# units")
        ax.set_title(f"{region}: preferred-state counts")
        ax.legend(fontsize=8)
        for k in range(K):
            ax.text(k, counts[k] + 0.5, f"({n_per[k]})",
                    ha="center", fontsize=7, color="grey")
    fig.suptitle("B1: per-region preferred-state distribution "
                 "(grey numbers = bins assigned to that state)")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)


# ---- B2: pre vs post within state ----
def run_B2(rates, viterbi, discovery_bin, region, K, out_dir, fig_dir):
    """Per-unit per-state pre/post Mann-Whitney."""
    n_units, n_bins = rates.shape
    pre_mask = np.arange(n_bins) < discovery_bin
    post_mask = ~pre_mask

    rows = []
    delta_mat = np.full((n_units, K), np.nan)
    pmat = np.full((n_units, K), np.nan)
    for k in range(K):
        in_state = (viterbi == k)
        pre_idx = np.flatnonzero(in_state & pre_mask)
        post_idx = np.flatnonzero(in_state & post_mask)
        n_pre = len(pre_idx); n_post = len(post_idx)
        for u in range(n_units):
            fr_pre = rates[u, pre_idx]
            fr_post = rates[u, post_idx]
            delta = (fr_post.mean() if n_post else np.nan) \
                    - (fr_pre.mean() if n_pre else np.nan)
            delta_mat[u, k] = delta
            if n_pre >= MIN_BINS_PER_STATE and n_post >= MIN_BINS_PER_STATE:
                try:
                    _, p = mannwhitneyu(fr_pre, fr_post, alternative="two-sided")
                except Exception:
                    p = np.nan
            else:
                p = np.nan
            pmat[u, k] = p
            rows.append(dict(
                unit_id=u, region=region, state=k,
                FR_pre=float(fr_pre.mean()) if n_pre else np.nan,
                FR_post=float(fr_post.mean()) if n_post else np.nan,
                delta=delta,
                n_bins_pre=n_pre, n_bins_post=n_post,
                p=p,
            ))
    df = pd.DataFrame(rows)

    # FDR within region across all (unit, state) tests with valid p
    df["p_fdr"] = np.nan
    valid = df["p"].notna()
    if valid.any():
        _, p_adj = fdr_mask(df.loc[valid, "p"].values, FDR_ALPHA)
        df.loc[valid, "p_fdr"] = p_adj
    df["sig_fdr"] = df["p_fdr"] < FDR_ALPHA
    df.to_csv(out_dir / f"B2_pre_vs_post_per_state_{region}.csv", index=False)

    # Heatmap delta sorted by mean delta
    mean_delta_per_unit = np.nanmean(delta_mat, axis=1)
    order = np.argsort(mean_delta_per_unit)
    fig, ax = plt.subplots(figsize=(0.4 * K + 2, 0.04 * n_units + 1.2))
    vmax = np.nanmax(np.abs(delta_mat)) + 1e-9
    im = ax.imshow(delta_mat[order], aspect="auto", cmap="RdBu_r",
                    vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_xticks(np.arange(K))
    ax.set_xticklabels([f"S{k}" for k in range(K)])
    ax.set_xlabel("Merged HMM state")
    ax.set_ylabel(f"{region} units (sorted by mean Δ)")
    ax.set_title(f"{region}: B2 within-state Δ FR (post − pre, Hz)")
    plt.colorbar(im, ax=ax, label="ΔFR (Hz)")
    fig.tight_layout()
    fig.savefig(fig_dir / f"B2_delta_heatmap_{region}.png", dpi=140)
    plt.close(fig)

    return df


def plot_B2_significant_counts(df_aca, df_lha, K, fig_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for ax, df, region in [(axes[0], df_aca, "ACA"), (axes[1], df_lha, "LHA")]:
        sig = df[df["sig_fdr"]].groupby("state").size().reindex(np.arange(K), fill_value=0)
        ax.bar(np.arange(K), sig.values, color="#4477aa")
        ax.set_xticks(np.arange(K))
        ax.set_xticklabels([f"S{k}" for k in range(K)])
        ax.set_xlabel("State")
        ax.set_ylabel("# units with FDR-sig pre→post change")
        ax.set_title(f"{region}: B2 FDR-significant unit counts per state")
    fig.suptitle("B2: per-region count of units showing pre/post within-state shift "
                 "(BH-FDR q<0.05)")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)


# ---- B3: Poisson GLM ----
def run_B3(counts, posteriors, region, K, out_dir, fig_dir, viterbi):
    """Poisson GLM per unit: log(FR) = β0 + Σ β_k posterior_k (drop most common state).
    `counts`: (n_units, n_bins) integer spike counts per HMM bin.
    `posteriors`: (n_bins, K) p_state matrix.
    """
    n_units, n_bins = counts.shape
    state_counts = np.bincount(viterbi, minlength=K)
    ref_state = int(np.argmax(state_counts))
    keep_states = [k for k in range(K) if k != ref_state]
    X = posteriors[:, keep_states]
    X = sm.add_constant(X, has_constant="add")
    offset = np.full(n_bins, np.log(HMM_BIN_S))

    rows = []
    coef_mat = np.full((n_units, len(keep_states)), np.nan)
    z_mat = np.full((n_units, len(keep_states)), np.nan)
    p_mat = np.full((n_units, len(keep_states)), np.nan)
    n_failed = 0
    for u in range(n_units):
        y = counts[u].astype(np.int64)
        if y.sum() < 10:
            n_failed += 1
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
                res = model.fit(maxiter=50)
        except Exception:
            n_failed += 1
            continue
        coefs = res.params[1:]   # skip intercept
        bse = res.bse[1:]
        pvals = res.pvalues[1:]
        zvals = coefs / np.where(bse > 0, bse, 1)
        coef_mat[u] = coefs
        z_mat[u] = zvals
        p_mat[u] = pvals
        for j, k in enumerate(keep_states):
            rows.append(dict(
                unit_id=u, region=region, state=k,
                beta=float(coefs[j]),
                se=float(bse[j]),
                z=float(zvals[j]),
                p=float(pvals[j]),
            ))

    df = pd.DataFrame(rows)
    df["p_fdr"] = np.nan
    valid = df["p"].notna()
    if valid.any():
        _, p_adj = fdr_mask(df.loc[valid, "p"].values, FDR_ALPHA)
        df.loc[valid, "p_fdr"] = p_adj
    # Significance: FDR significant AND |z| > threshold
    df["sig"] = (df["p_fdr"] < FDR_ALPHA) & (df["z"].abs() > GLM_Z_THRESHOLD)
    df.to_csv(out_dir / f"B3_glm_coefficients_{region}.csv", index=False)

    # Coefficient heatmap (units × states), sorted by max |coef|
    order = np.argsort(-np.nanmax(np.abs(coef_mat), axis=1))
    fig, ax = plt.subplots(figsize=(0.4 * len(keep_states) + 2,
                                      0.04 * n_units + 1.4))
    vmax = np.nanmax(np.abs(coef_mat)) + 1e-9
    im = ax.imshow(coef_mat[order], aspect="auto", cmap="RdBu_r",
                    vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_xticks(np.arange(len(keep_states)))
    ax.set_xticklabels([f"S{k}" for k in keep_states])
    ax.set_xlabel(f"Merged HMM state (reference = S{ref_state}, dropped)")
    ax.set_ylabel(f"{region} units (sorted by max |β|)")
    ax.set_title(f"{region}: B3 Poisson GLM β coefficients "
                 f"(skipped {n_failed} units)")
    plt.colorbar(im, ax=ax, label="β (log Hz)")
    fig.tight_layout()
    fig.savefig(fig_dir / f"B3_coefficient_heatmap_{region}.png", dpi=140)
    plt.close(fig)

    return df, ref_state, keep_states


def plot_B3_summary(df_aca, df_lha, K, ref_aca, ref_lha, out_dir):
    rows = []
    for region, df, ref in [("ACA", df_aca, ref_aca), ("LHA", df_lha, ref_lha)]:
        sig_units = df[df["sig"]].groupby("unit_id")["state"].apply(
            lambda x: ",".join(map(str, sorted(set(x))))
        )
        for u, states in sig_units.items():
            rows.append(dict(unit_id=int(u), region=region,
                              n_sig_states=len(states.split(",")),
                              sig_states=states,
                              reference_state_dropped=ref))
    out = pd.DataFrame(rows).sort_values(["region", "unit_id"])
    out.to_csv(out_dir / "B3_significant_coefficients_summary.csv", index=False)
    return out


# ---- B4: PCA + state coloring ----
def run_B4(rates, viterbi, discovery_bin, region, K, out_dir, fig_dir):
    n_units, n_bins = rates.shape
    # z-score each unit
    mu = rates.mean(axis=1, keepdims=True)
    sig = rates.std(axis=1, keepdims=True) + 1e-9
    z = (rates - mu) / sig            # (n_units, n_bins)
    X = z.T                            # (n_bins, n_units)

    # PCA via SVD on mean-centered X
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pcs = U[:, :N_PCS] * S[:N_PCS]     # (n_bins, N_PCS)
    var_explained = (S ** 2) / (S ** 2).sum()
    var_top = var_explained[:N_PCS]

    # Save loadings
    loadings = Vt[:N_PCS].T            # (n_units, N_PCS)
    df_load = pd.DataFrame(loadings, columns=[f"PC{i+1}" for i in range(N_PCS)])
    df_load.insert(0, "unit_id", np.arange(n_units))
    df_load.insert(1, "region", region)
    df_load.to_csv(out_dir / f"B4_pca_loadings_{region}.csv", index=False)

    # ---- Plot: PC1 vs PC2 + PC2 vs PC3 colored by state ----
    cmap = state_color_map(K)
    colors = np.array([cmap[k] for k in viterbi])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (xi, yi) in zip(axes, [(0, 1), (1, 2)]):
        ax.scatter(pcs[:, xi], pcs[:, yi], c=colors, s=4, alpha=0.45,
                    rasterized=True, edgecolors="none")
        ax.set_xlabel(f"PC{xi+1} ({var_top[xi]*100:.1f}% var)")
        ax.set_ylabel(f"PC{yi+1} ({var_top[yi]*100:.1f}% var)")
        ax.set_title(f"{region}: PC{xi+1} vs PC{yi+1}, by state")
    handles = [plt.Line2D([0], [0], marker="o", linestyle="",
                            markerfacecolor=cmap[k], markeredgecolor="none",
                            markersize=8, label=f"S{k}") for k in range(K)]
    axes[1].legend(handles=handles, fontsize=8, loc="upper left",
                    bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    fig.tight_layout()
    fig.savefig(fig_dir / f"B4_pca_state_colored_{region}.png", dpi=140,
                 bbox_inches="tight")
    plt.close(fig)

    # ---- Centroid distances in (PC1, PC2, PC3) ----
    cents = np.full((K, 3), np.nan)
    for k in range(K):
        m = viterbi == k
        if m.sum() >= 5:
            cents[k] = pcs[m][:, :3].mean(axis=0)
    dist = np.full((K, K), np.nan)
    for i in range(K):
        for j in range(K):
            if np.all(np.isfinite(cents[i])) and np.all(np.isfinite(cents[j])):
                dist[i, j] = np.linalg.norm(cents[i] - cents[j])
    rows = []
    for i in range(K):
        for j in range(i + 1, K):
            rows.append(dict(state_a=i, state_b=j, distance=float(dist[i, j])))
    pd.DataFrame(rows).to_csv(out_dir / f"B4_centroid_distances_{region}.csv",
                                index=False)

    # ---- Pre vs post overlay (transparency by pre/post) ----
    pre_mask = np.arange(n_bins) < discovery_bin
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (xi, yi) in zip(axes, [(0, 1), (1, 2)]):
        # pre = transparent
        c_pre = np.array([to_rgba(cmap[k], alpha=0.18) for k in viterbi[pre_mask]])
        ax.scatter(pcs[pre_mask, xi], pcs[pre_mask, yi], c=c_pre, s=3,
                    rasterized=True, edgecolors="none", label="pre")
        # post = solid
        c_post = np.array([to_rgba(cmap[k], alpha=0.85) for k in viterbi[~pre_mask]])
        ax.scatter(pcs[~pre_mask, xi], pcs[~pre_mask, yi], c=c_post, s=4,
                    rasterized=True, edgecolors="none", label="post")
        ax.set_xlabel(f"PC{xi+1} ({var_top[xi]*100:.1f}%)")
        ax.set_ylabel(f"PC{yi+1} ({var_top[yi]*100:.1f}%)")
        ax.set_title(f"{region}: pre (faint) vs post (solid)")
    fig.tight_layout()
    fig.savefig(fig_dir / f"B4_pca_pre_post_{region}.png", dpi=140,
                 bbox_inches="tight")
    plt.close(fig)

    # Pre/post centroid shift per state (in 3D PC space)
    shift_rows = []
    for k in range(K):
        m_pre = (viterbi == k) & pre_mask
        m_post = (viterbi == k) & ~pre_mask
        if m_pre.sum() >= 5 and m_post.sum() >= 5:
            c_pre = pcs[m_pre][:, :3].mean(axis=0)
            c_post = pcs[m_post][:, :3].mean(axis=0)
            shift = float(np.linalg.norm(c_post - c_pre))
        else:
            shift = np.nan
        shift_rows.append(dict(state=k,
                                n_pre=int(m_pre.sum()),
                                n_post=int(m_post.sum()),
                                centroid_shift_PC123=shift))
    shifts = pd.DataFrame(shift_rows)
    shifts.to_csv(out_dir / f"B4_pre_post_centroid_shift_{region}.csv",
                    index=False)

    return var_top, dist, shifts


# ---- Main ----
def main():
    cfg = load_config()
    out_dir = REPO_ROOT / "data" / "HMM" / "neural_alignment" / f"state_conditioned_S{SESSION_NUM}"
    fig_dir = REPO_ROOT / "figures" / "HMM" / "neural_alignment" / f"state_conditioned_S{SESSION_NUM}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---- Resolve session paths ----
    with open(REPO_ROOT / cfg["paths_yaml"]) as f:
        paths_data = yaml.safe_load(f)
    s12_paths = (paths_data["double_probe"]["coordinates_1"]["mouse01"]
                  ["sessions"][f"session_{SESSION_NUM}"])
    aca_sorted = Path(s12_paths["probe_0_aca"]["sorted"])
    lha_sorted = Path(s12_paths["probe_1_lha_rsp"]["sorted"])

    # ---- Load good units + spikes ----
    print("=== Loading neural data ===")
    aca_uids = [int(u) for u in get_good_units_p0(aca_sorted)]
    lha_uids = [int(u) for u in get_good_units_p1_lha(lha_sorted)]
    aca_sorting = se.read_kilosort(aca_sorted)
    lha_sorting = se.read_kilosort(lha_sorted)
    aca_uids = [u for u in aca_uids if u in set(aca_sorting.get_unit_ids())]
    lha_uids = [u for u in lha_uids if u in set(lha_sorting.get_unit_ids())]
    aca_spikes = load_spike_times_for_region(aca_sorting, aca_uids)
    lha_spikes = load_spike_times_for_region(lha_sorting, lha_uids)
    print(f"  ACA units: {len(aca_uids)}, LHA units: {len(lha_uids)}")

    # ---- Bin to 100 ms (Track A grid), then aggregate to 480 ms ----
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
    aca_100 = np.zeros((len(aca_uid_list), n_100ms), dtype=np.float64)
    lha_100 = np.zeros((len(lha_uid_list), n_100ms), dtype=np.float64)
    for i, uid in enumerate(aca_uid_list):
        aca_100[i] = np.histogram(aca_spikes[uid], edges_100ms)[0]
    for i, uid in enumerate(lha_uid_list):
        lha_100[i] = np.histogram(lha_spikes[uid], edges_100ms)[0]

    aca_rates = rebin_100ms_to_480ms(aca_100, n_hmm_bins)   # Hz, (n_units, n_hmm_bins)
    lha_rates = rebin_100ms_to_480ms(lha_100, n_hmm_bins)

    # Spike counts per HMM bin (for B3 GLM): bin spikes directly at 480 ms edges.
    aca_counts = np.zeros((len(aca_uid_list), n_hmm_bins), dtype=np.int64)
    lha_counts = np.zeros((len(lha_uid_list), n_hmm_bins), dtype=np.int64)
    for i, uid in enumerate(aca_uid_list):
        aca_counts[i] = np.histogram(aca_spikes[uid], edges_480ms)[0]
    for i, uid in enumerate(lha_uid_list):
        lha_counts[i] = np.histogram(lha_spikes[uid], edges_480ms)[0]

    # ---- Load merged Viterbi + posteriors ----
    post_csv = REPO_ROOT / cfg["merge_dirs"]["posteriors"] / f"session_{SESSION_NUM}.csv"
    post_df = pd.read_csv(post_csv)
    viterbi = post_df["viterbi"].values.astype(np.int64)
    K = max(int(viterbi.max()) + 1,
            sum(1 for c in post_df.columns if c.startswith("p_state_")))
    posteriors = np.column_stack([post_df[f"p_state_{k}"].values for k in range(K)])

    # Sanity: align lengths
    n_align = min(n_hmm_bins, len(viterbi))
    aca_rates = aca_rates[:, :n_align]
    lha_rates = lha_rates[:, :n_align]
    aca_counts = aca_counts[:, :n_align]
    lha_counts = lha_counts[:, :n_align]
    viterbi = viterbi[:n_align]
    posteriors = posteriors[:n_align]
    print(f"  HMM bins: {n_align}, K = {K}")

    history = pd.read_csv(REPO_ROOT / cfg["commitment_dirs"]["out"]
                           / "sampling_history.csv")
    s12 = history[history.session == SESSION_NUM].iloc[0]
    discovery_bin = int(s12["discovery_bin"])
    print(f"  Discovery bin: {discovery_bin}, t={s12['discovery_time_s']:.2f} s")

    # ===== B1 =====
    print("\n=== B1: state-selectivity (one-way ANOVA per unit) ===")
    sum_aca, mean_aca, npb_aca = run_B1(aca_rates, viterbi, "ACA", K, out_dir, fig_dir)
    sum_lha, mean_lha, npb_lha = run_B1(lha_rates, viterbi, "LHA", K, out_dir, fig_dir)
    sel_summary = pd.concat([sum_aca, sum_lha], ignore_index=True)
    sel_summary.to_csv(out_dir / "B1_selectivity_summary.csv", index=False)
    plot_state_preference_counts(sum_aca, sum_lha, K, npb_aca, npb_lha,
                                  fig_dir / "B1_state_preference_counts.png")
    for region, summ in [("ACA", sum_aca), ("LHA", sum_lha)]:
        n_uncorr = int(summ["sig_uncorr"].sum())
        n_fdr = int(summ["sig_fdr"].sum())
        pref_dist = summ["preferred_state"].value_counts().sort_index().to_dict()
        print(f"  {region}: ANOVA uncorrected p<{ALPHA}: {n_uncorr}/{len(summ)}, "
              f"FDR-sig: {n_fdr}/{len(summ)}")
        print(f"    Preferred-state counts: {pref_dist}")

    # ===== B2 =====
    print("\n=== B2: pre vs post-discovery within-state shift ===")
    df_b2_aca = run_B2(aca_rates, viterbi, discovery_bin, "ACA", K, out_dir, fig_dir)
    df_b2_lha = run_B2(lha_rates, viterbi, discovery_bin, "LHA", K, out_dir, fig_dir)
    plot_B2_significant_counts(df_b2_aca, df_b2_lha, K,
                                fig_dir / "B2_significant_unit_counts.png")
    for region, df in [("ACA", df_b2_aca), ("LHA", df_b2_lha)]:
        n_units_total = df["unit_id"].nunique()
        sig_units = df.loc[df["sig_fdr"], "unit_id"].nunique()
        per_state = df[df["sig_fdr"]].groupby("state").size().to_dict()
        print(f"  {region}: units with FDR-sig pre/post change in any state: "
              f"{sig_units}/{n_units_total}")
        print(f"    sig (unit, state) pairs per state: {per_state}")

    # ===== B3 =====
    print("\n=== B3: Poisson GLM per unit on state posteriors ===")
    df_b3_aca, ref_aca, keep_aca = run_B3(aca_counts, posteriors, "ACA", K,
                                            out_dir, fig_dir, viterbi)
    df_b3_lha, ref_lha, keep_lha = run_B3(lha_counts, posteriors, "LHA", K,
                                            out_dir, fig_dir, viterbi)
    glm_summary = plot_B3_summary(df_b3_aca, df_b3_lha, K, ref_aca, ref_lha, out_dir)
    print(f"  Reference state dropped: ACA = S{ref_aca}, LHA = S{ref_lha}")
    for region, df in [("ACA", df_b3_aca), ("LHA", df_b3_lha)]:
        n_units_total = df["unit_id"].nunique()
        sig_units = df.loc[df["sig"], "unit_id"].nunique()
        sig_per_state = df[df["sig"]].groupby("state").size().to_dict()
        print(f"  {region}: units with ≥1 sig coefficient (FDR q<{FDR_ALPHA}, "
              f"|z|>{GLM_Z_THRESHOLD}): {sig_units}/{n_units_total}")
        print(f"    sig (unit, state) pairs per state: {sig_per_state}")

    # ===== B4 =====
    print("\n=== B4: PCA, colored by state, with pre/post overlay ===")
    var_aca, dist_aca, shifts_aca = run_B4(aca_rates, viterbi, discovery_bin,
                                              "ACA", K, out_dir, fig_dir)
    var_lha, dist_lha, shifts_lha = run_B4(lha_rates, viterbi, discovery_bin,
                                              "LHA", K, out_dir, fig_dir)
    print(f"  ACA top-{N_PCS} var explained: "
          f"{[f'{v*100:.1f}%' for v in var_aca]} (cumulative {var_aca.sum()*100:.1f}%)")
    print(f"  LHA top-{N_PCS} var explained: "
          f"{[f'{v*100:.1f}%' for v in var_lha]} (cumulative {var_lha.sum()*100:.1f}%)")
    print(f"  ACA top 3 pre/post centroid shifts (state, shift):")
    print(f"    {shifts_aca.dropna().nlargest(3, 'centroid_shift_PC123')[['state','centroid_shift_PC123','n_pre','n_post']].to_string(index=False)}")
    print(f"  LHA top 3 pre/post centroid shifts (state, shift):")
    print(f"    {shifts_lha.dropna().nlargest(3, 'centroid_shift_PC123')[['state','centroid_shift_PC123','n_pre','n_post']].to_string(index=False)}")

    # ---- Final summary ----
    print(f"\nDone. Outputs in {out_dir} and {fig_dir}")


if __name__ == "__main__":
    main()
