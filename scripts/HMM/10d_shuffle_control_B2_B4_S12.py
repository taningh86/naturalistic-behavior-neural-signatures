"""10d — Shuffle control for Track B's B2 (within-state pre/post) and B4
(PC centroid pre/post shift) on session 12.

Both analyses depend on the pre/post-discovery boundary (real bin = 1244,
t=597.12 s). This script tests whether their effects survive replacing that
boundary with a random "fake discovery" bin while keeping every other axis
intact.

Strategy (per iteration):
  - Pick fake_boundary_bin uniformly in [500, T-500], avoiding ±20 bins around
    the real discovery (1244). Range [500, 3250] excludes the very early/late
    session edges where one side would have too few bins.
  - B2: recompute per-(unit, state) Mann-Whitney pre vs post on bin-level FR
    using the fake boundary. FDR within region. Count units with ≥1 sig state.
  - B4: project the (unchanged) z-scored rates onto the (unchanged) real PCA
    loadings, split bins by the fake boundary, recompute per-state pre/post
    centroid distance in PC1-3 space.

The Viterbi sequence and unit firing rates are NOT shuffled — only the pre/post
label is randomized. This isolates the pre/post-discovery contrast as the
source of any effect.

100 iterations, master seed = 20260507.
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
SESSION_NUM = 12
HMM_BIN_S = 0.480
NEURAL_BIN_SMALL_S = 0.1
N_SHUFFLES = 100
SHUFFLE_SEED = 20260507
FAKE_BOUND_MIN = 500
FAKE_BOUND_MAX_OFFSET = 500   # max bin = T - 500
EXCLUSION_HALF = 20            # ±N bins around real discovery to avoid landing on truth
REAL_DISCOVERY_BIN = 1244     # for reference + exclusion + observed pass
FDR_ALPHA = 0.05
B2_MIN_BINS = 30
B4_MIN_BINS = 5
N_PCS = 5


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


# ---- B2 / B4 single-iteration kernels ----
def b2_one_iter(rates, viterbi, fake_bin, K):
    """Return (n_sig_units, n_sig_per_state np.array of length K).
    Mann-Whitney pre vs post on rates within each state, FDR within region.
    """
    n_units, n_bins = rates.shape
    pre_mask = np.arange(n_bins) < fake_bin
    pvals = []
    keys = []
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
            pvals.append(p)
            keys.append((u, k))
    pvals = np.asarray(pvals)
    sig = fdr_pass(pvals, FDR_ALPHA)
    sig_units = set()
    sig_per_state = np.zeros(K, dtype=int)
    for (u, k), s in zip(keys, sig):
        if s:
            sig_units.add(u)
            sig_per_state[k] += 1
    return len(sig_units), sig_per_state


def b4_one_iter(pcs, viterbi, fake_bin, K):
    """Return per-state centroid shift (PC1-3) np.array length K with NaN
    for states with fewer than B4_MIN_BINS bins on either side.
    """
    n_bins = pcs.shape[0]
    pre_mask = np.arange(n_bins) < fake_bin
    shifts = np.full(K, np.nan)
    for k in range(K):
        m_pre = (viterbi == k) & pre_mask
        m_post = (viterbi == k) & ~pre_mask
        if m_pre.sum() < B4_MIN_BINS or m_post.sum() < B4_MIN_BINS:
            continue
        c_pre = pcs[m_pre, :3].mean(axis=0)
        c_post = pcs[m_post, :3].mean(axis=0)
        shifts[k] = float(np.linalg.norm(c_post - c_pre))
    return shifts


def compute_pca_projection(rates):
    mu = rates.mean(axis=1, keepdims=True)
    sig = rates.std(axis=1, keepdims=True) + 1e-9
    z = (rates - mu) / sig
    X = z.T   # (bins, units)
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pcs = U[:, :N_PCS] * S[:N_PCS]
    return pcs


# ---- Main ----
def main():
    cfg = load_config()
    out_dir = REPO_ROOT / "data" / "HMM" / "neural_alignment" / f"shuffle_control_B2_B4_S{SESSION_NUM}"
    fig_dir = REPO_ROOT / "figures" / "HMM" / "neural_alignment" / f"shuffle_control_B2_B4_S{SESSION_NUM}"
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

    # Bin to 480 ms
    binned = np.load(REPO_ROOT / cfg["out_dirs"]["binned"]
                      / f"session_{SESSION_NUM}.npz", allow_pickle=True)
    trial_time = np.asarray(binned["trial_time"], dtype=np.float64)
    n_hmm_bins = len(trial_time)
    duration_s = float(trial_time[-1] + HMM_BIN_S)
    n_100ms = int(np.ceil(duration_s / NEURAL_BIN_SMALL_S))
    edges_100ms = np.arange(n_100ms + 1) * NEURAL_BIN_SMALL_S

    aca_uid_list = sorted(aca_spikes.keys())
    lha_uid_list = sorted(lha_spikes.keys())
    aca_100 = np.zeros((n_aca, n_100ms), dtype=np.float64)
    lha_100 = np.zeros((n_lha, n_100ms), dtype=np.float64)
    for i, uid in enumerate(aca_uid_list):
        aca_100[i] = np.histogram(aca_spikes[uid], edges_100ms)[0]
    for i, uid in enumerate(lha_uid_list):
        lha_100[i] = np.histogram(lha_spikes[uid], edges_100ms)[0]
    aca_rates = rebin_100ms_to_480ms(aca_100, n_hmm_bins)
    lha_rates = rebin_100ms_to_480ms(lha_100, n_hmm_bins)

    # Merged Viterbi
    post_csv = REPO_ROOT / cfg["merge_dirs"]["posteriors"] / f"session_{SESSION_NUM}.csv"
    post_df = pd.read_csv(post_csv)
    viterbi = post_df["viterbi"].values.astype(np.int64)
    K = max(int(viterbi.max()) + 1,
            sum(1 for c in post_df.columns if c.startswith("p_state_")))

    # Align lengths
    n = min(n_hmm_bins, len(viterbi))
    aca_rates = aca_rates[:, :n]
    lha_rates = lha_rates[:, :n]
    viterbi = viterbi[:n]
    print(f"  Bins: {n}, K={K}, real discovery bin: {REAL_DISCOVERY_BIN}")

    # ---- PCA fit (once, on real data; reused across all shuffles) ----
    pcs_aca = compute_pca_projection(aca_rates)    # (n_bins, N_PCS)
    pcs_lha = compute_pca_projection(lha_rates)
    print(f"  PCA fit (top-{N_PCS} PCs) — ACA / LHA")

    # ---- Observed pass (real discovery bin) ----
    print("\n=== Observed (real discovery bin = 1244) ===")
    obs_b2_aca, obs_b2_aca_per_state = b2_one_iter(aca_rates, viterbi,
                                                     REAL_DISCOVERY_BIN, K)
    obs_b2_lha, obs_b2_lha_per_state = b2_one_iter(lha_rates, viterbi,
                                                     REAL_DISCOVERY_BIN, K)
    print(f"  B2 observed: ACA {obs_b2_aca}/{n_aca}, LHA {obs_b2_lha}/{n_lha}")

    obs_b4_aca = b4_one_iter(pcs_aca, viterbi, REAL_DISCOVERY_BIN, K)
    obs_b4_lha = b4_one_iter(pcs_lha, viterbi, REAL_DISCOVERY_BIN, K)
    print(f"  B4 observed centroid shifts (top 3 per region):")
    for region, shifts in [("ACA", obs_b4_aca), ("LHA", obs_b4_lha)]:
        order = np.argsort(-np.where(np.isfinite(shifts), shifts, -np.inf))[:3]
        print(f"    {region}: " +
              ", ".join(f"S{int(k)}={shifts[k]:.2f}"
                          for k in order if np.isfinite(shifts[k])))

    # ---- Shuffles ----
    print(f"\n=== {N_SHUFFLES} fake-discovery shuffles ===")
    rng = np.random.default_rng(SHUFFLE_SEED)
    rows_b2 = []
    rows_b2_per_state = []
    rows_b4 = []
    boundary_lo = FAKE_BOUND_MIN
    boundary_hi = n - FAKE_BOUND_MAX_OFFSET
    excl_lo = REAL_DISCOVERY_BIN - EXCLUSION_HALF
    excl_hi = REAL_DISCOVERY_BIN + EXCLUSION_HALF
    print(f"  fake-boundary range: [{boundary_lo}, {boundary_hi}], "
          f"excluding [{excl_lo}, {excl_hi}]")

    t0 = time.time()
    for it in range(N_SHUFFLES):
        # rejection sample to avoid the exclusion zone
        while True:
            fb = int(rng.integers(boundary_lo, boundary_hi))
            if fb < excl_lo or fb > excl_hi:
                break

        # B2 ACA + LHA
        n_aca_sig, aca_per_state = b2_one_iter(aca_rates, viterbi, fb, K)
        n_lha_sig, lha_per_state = b2_one_iter(lha_rates, viterbi, fb, K)
        rows_b2.append(dict(iter=it, fake_boundary_bin=fb,
                              n_sig_aca=n_aca_sig, n_sig_lha=n_lha_sig))
        for k in range(K):
            rows_b2_per_state.append(dict(iter=it, region="ACA", state=k,
                                            n_FDR_sig=int(aca_per_state[k])))
            rows_b2_per_state.append(dict(iter=it, region="LHA", state=k,
                                            n_FDR_sig=int(lha_per_state[k])))

        # B4 ACA + LHA
        sh_aca = b4_one_iter(pcs_aca, viterbi, fb, K)
        sh_lha = b4_one_iter(pcs_lha, viterbi, fb, K)
        for k in range(K):
            rows_b4.append(dict(iter=it, fake_boundary_bin=fb, region="ACA",
                                  state=k, centroid_shift=float(sh_aca[k])
                                  if np.isfinite(sh_aca[k]) else np.nan))
            rows_b4.append(dict(iter=it, fake_boundary_bin=fb, region="LHA",
                                  state=k, centroid_shift=float(sh_lha[k])
                                  if np.isfinite(sh_lha[k]) else np.nan))

        if (it + 1) % 10 == 0 or it == 0:
            print(f"  iter {it+1:>3}/{N_SHUFFLES}  fb={fb:>5}  "
                  f"B2 sig: ACA {n_aca_sig:>3} LHA {n_lha_sig:>3}  "
                  f"({time.time()-t0:.0f}s)")

    df_b2 = pd.DataFrame(rows_b2)
    df_b2_per_state = pd.DataFrame(rows_b2_per_state)
    df_b4 = pd.DataFrame(rows_b4)
    df_b2.to_csv(out_dir / "shuffle_B2_summary.csv", index=False)
    df_b2_per_state.to_csv(out_dir / "shuffle_B2_per_state.csv", index=False)
    df_b4.to_csv(out_dir / "shuffle_B4_summary.csv", index=False)
    print(f"\n  → {out_dir / 'shuffle_B2_summary.csv'}, "
          f"shuffle_B2_per_state.csv, shuffle_B4_summary.csv")

    # ---- B4 per-state significance summary ----
    sig_rows = []
    for region, observed in [("ACA", obs_b4_aca), ("LHA", obs_b4_lha)]:
        for k in range(K):
            shuf = df_b4[(df_b4.region == region) & (df_b4.state == k)]["centroid_shift"].dropna().values
            if len(shuf) == 0:
                continue
            obs = observed[k]
            if not np.isfinite(obs):
                continue
            shuf_mean = float(shuf.mean())
            shuf_p95 = float(np.percentile(shuf, 95))
            obs_pct = float((shuf <= obs).mean() * 100)
            sig_rows.append(dict(
                region=region, state=k,
                observed_shift=float(obs),
                shuffle_mean=shuf_mean,
                shuffle_p95=shuf_p95,
                obs_pctile=obs_pct,
                exceeds_p95=bool(obs > shuf_p95),
            ))
    df_sig = pd.DataFrame(sig_rows)
    df_sig.to_csv(out_dir / "shuffle_B4_state_significance.csv", index=False)

    # ---- B2 distribution figure ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, vals, observed, total, label in [
        (axes[0], df_b2["n_sig_aca"].values, obs_b2_aca, n_aca, "ACA"),
        (axes[1], df_b2["n_sig_lha"].values, obs_b2_lha, n_lha, "LHA"),
    ]:
        ax.hist(vals, bins=20, color="#9999cc", edgecolor="white", alpha=0.85)
        ax.axvline(observed, color="red", lw=2,
                    label=f"observed = {observed}/{total}")
        pct = float((vals <= observed).mean() * 100.0)
        ax.set_title(f"B2 — {label}\nshuffled mean={vals.mean():.1f} "
                     f"[95% CI {np.percentile(vals,2.5):.1f}-"
                     f"{np.percentile(vals,97.5):.1f}], "
                     f"observed at {pct:.1f}th pctile",
                     fontsize=10)
        ax.set_xlabel("n units with ≥1 FDR-sig pre/post state change")
        ax.set_ylabel("# shuffles")
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(alpha=0.3)
    fig.suptitle(f"S{SESSION_NUM} fake-discovery shuffle — B2 "
                 f"({N_SHUFFLES} iter, real bin = {REAL_DISCOVERY_BIN})",
                 fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig(fig_dir / "shuffle_B2_distributions.png", dpi=140)
    plt.close(fig)

    # ---- B4 per-state plot ----
    fig, axes = plt.subplots(2, 1, figsize=(max(8, 0.6 * K + 2), 8), sharex=True)
    for ax, region, observed in [(axes[0], "ACA", obs_b4_aca),
                                    (axes[1], "LHA", obs_b4_lha)]:
        states = np.arange(K)
        # observed bar
        obs_arr = np.array(observed, dtype=np.float64)
        bar_obs = ax.bar(states - 0.18, obs_arr, width=0.36,
                          color="#cc4444", alpha=0.85, label="observed")
        # shuffle p95 bar
        p95_arr = np.full(K, np.nan)
        sh_mean = np.full(K, np.nan)
        for k in range(K):
            ss = df_b4[(df_b4.region == region) & (df_b4.state == k)]["centroid_shift"].dropna().values
            if len(ss):
                p95_arr[k] = np.percentile(ss, 95)
                sh_mean[k] = ss.mean()
        ax.bar(states + 0.18, p95_arr, width=0.36,
               color="#888888", alpha=0.85, label="shuffle 95th pctile")
        ax.scatter(states + 0.18, sh_mean, color="white",
                    edgecolors="black", s=20, zorder=5,
                    label="shuffle mean")
        # Highlight states where observed > p95
        for k in range(K):
            if (np.isfinite(obs_arr[k]) and np.isfinite(p95_arr[k])
                    and obs_arr[k] > p95_arr[k]):
                ax.text(k - 0.18, obs_arr[k] + 0.03, "*",
                         ha="center", fontsize=14, color="red")
        ax.set_xticks(states)
        ax.set_xticklabels([f"S{k}" for k in range(K)])
        ax.set_ylabel("PC1-3 centroid shift")
        ax.set_title(f"{region}: per-state pre/post centroid shift "
                     "(* = observed > shuffle 95th pctile)")
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(axis="y", alpha=0.3)
    axes[1].set_xlabel("Merged HMM state")
    fig.suptitle("B4 per-state pre/post centroid shift — observed vs shuffle null",
                 fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig(fig_dir / "shuffle_B4_per_state.png", dpi=140)
    plt.close(fig)

    # ---- Final summary ----
    print("\n========== SUMMARY ==========")
    for label, vals, obs, total in [
        ("B2 ACA", df_b2["n_sig_aca"].values, obs_b2_aca, n_aca),
        ("B2 LHA", df_b2["n_sig_lha"].values, obs_b2_lha, n_lha),
    ]:
        pct = float((vals <= obs).mean() * 100)
        print(f"\n{label}:")
        print(f"  observed: {obs}/{total} ({obs/total*100:.0f}%)")
        print(f"  shuffle: mean={vals.mean():.1f}, "
              f"95% CI [{np.percentile(vals, 2.5):.0f}, "
              f"{np.percentile(vals, 97.5):.0f}], "
              f"range [{vals.min()}, {vals.max()}]")
        print(f"  observed percentile in null: {pct:.1f}%")

    print("\nB4 per-state significance (observed shift vs shuffle null):")
    for region, observed in [("ACA", obs_b4_aca), ("LHA", obs_b4_lha)]:
        sub = df_sig[df_sig.region == region].sort_values("observed_shift",
                                                            ascending=False)
        print(f"\n  {region} (top 6 states by observed shift):")
        print("    " + sub.head(6)[["state", "observed_shift", "shuffle_mean",
                                       "shuffle_p95", "obs_pctile",
                                       "exceeds_p95"]].to_string(index=False).replace("\n",
                                                                                        "\n    "))
        # Highlight S8 / S9 specifically
        for k in (8, 9):
            row = sub[sub.state == k]
            if not len(row):
                continue
            r = row.iloc[0]
            tag = "✓ exceeds" if r["exceeds_p95"] else "✗ within null"
            print(f"    S{k}: observed={r['observed_shift']:.2f}, "
                  f"p95={r['shuffle_p95']:.2f}, pctile={r['obs_pctile']:.0f}%  "
                  f"[{tag}]")

    # below-median check
    print("\nStates where observed shift falls below shuffle median (50th pctile):")
    below = df_sig[df_sig.obs_pctile < 50.0]
    if len(below):
        print(below[["region", "state", "observed_shift", "shuffle_mean",
                       "obs_pctile"]].to_string(index=False))
    else:
        print("  (none — every observed shift is at or above shuffle median)")

    print(f"\nDone. Outputs in {out_dir} and {fig_dir}")


if __name__ == "__main__":
    main()
