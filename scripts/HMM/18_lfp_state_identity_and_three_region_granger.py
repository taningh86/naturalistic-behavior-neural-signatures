"""18 — LFP state-identity + three-region Granger.

Two analyses on top of the corrected (script 17 v2) bipolar-referenced regional
LFP traces in `data/HMM/neural_alignment/lfp_spectral/preprocessed_v2/`.

A1 — LFP State-Identity
  For each (session, region, band): Kruskal-Wallis across the three behavioral
  categories home(S3) / feeding(S2) / transition_zone(S4) on per-bin band
  power (multitaper PSD over 480 ms HMM bins). 100 circular-shift shuffles
  for replication. Pairwise Mann-Whitney + FDR when omnibus passes.

A2 — Three-Region Granger
  Extends script 17 M4 (ACA-LHA only) to all three region pairs (ACA-LHA,
  ACA-RSP, LHA-RSP). Hilbert envelope per band on S3 stay + 5 s post-exit
  segments. Bivariate VAR with BIC lag selection 1-20 samples at 500 Hz.
  100 circular-shift shuffles per direction. Sign test across sessions per
  band × pair.

Inputs (per session N in {4,6,8,12,14,16}):
  data/HMM/neural_alignment/lfp_spectral/preprocessed_v2/
    session_{N}_{ACA,LHA,RSP}_regional.npy        (float32, 500 Hz)
    session_{N}_{ACA,LHA,RSP}_artifact_mask.npy   (bool,   500 Hz)
  data/HMM/merged_posteriors_dynamax/session_{N}.csv   (Viterbi col)

Outputs:
  data/HMM/neural_alignment/lfp_state_identity/...
  data/HMM/neural_alignment/lfp_three_region_granger/...
  figures/HMM/neural_alignment/lfp_state_identity/...
  figures/HMM/neural_alignment/lfp_three_region_granger/...
"""
from pathlib import Path
import argparse
import sys
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, hilbert, windows
from scipy.stats import kruskal, mannwhitneyu, f as f_dist, binomtest
from statsmodels.stats.multitest import multipletests

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "HMM"))
from _utils import load_config


# ---- Constants ----
FS_LFP_DS = 500.0
HMM_BIN_S = 0.480
SAMPLES_PER_HMM_BIN = int(HMM_BIN_S * FS_LFP_DS)        # 240

SESSIONS = [4, 6, 8, 12, 14, 16]
SESSION_STATE = {4: "fed", 6: "fed", 8: "fed",
                  12: "fasted", 14: "fasted", 16: "fasted"}
REGIONS = ("ACA", "LHA", "RSP")

CATEGORY_STATES = {"home": 3, "feeding": 2, "transition_zone": 4}
CATEGORIES = list(CATEGORY_STATES.keys())

BANDS = {
    "delta": (1, 4),
    "theta": (4, 12),
    "beta": (15, 30),
    "low_gamma": (30, 60),
    "high_gamma": (60, 100),
}
BAND_NAMES = list(BANDS.keys())

# Multitaper
DPSS_NW = 3
DPSS_K = 5

MIN_BINS = 30
N_SHUFFLES = 100
SHUFFLE_MIN_OFFSET = 200
SHUFFLE_MARGIN = 200
SEED = 20260510
FDR_ALPHA = 0.05
REPLICATION_THRESHOLD = 4    # 4/6 sessions

# Granger
S3_STATE = 3
POST_EXIT_S = 5.0
POST_EXIT_SAMPLES = int(POST_EXIT_S * FS_LFP_DS)        # 2500
GRANGER_MIN_S3_SAMPLES = int(5.0 * FS_LFP_DS)
GRANGER_LAG_RANGE = list(range(1, 21))                  # 2-40 ms at 500 Hz


# ---- Paths ----
def lfp_prep_dir(pairs_version="v1"):
    if pairs_version == "v1":
        return REPO_ROOT / "data" / "HMM" / "neural_alignment" / "lfp_spectral" / "preprocessed_v2"
    return REPO_ROOT / "data" / "HMM" / "neural_alignment" / "lfp_spectral_v3" / "preprocessed_v3"


def out_dirs(pairs_version="v1"):
    suffix = "" if pairs_version == "v1" else "_v2"
    a1_out = REPO_ROOT / "data" / "HMM" / "neural_alignment" / f"lfp_state_identity{suffix}"
    a2_out = REPO_ROOT / "data" / "HMM" / "neural_alignment" / f"lfp_three_region_granger{suffix}"
    a1_fig = REPO_ROOT / "figures" / "HMM" / "neural_alignment" / f"lfp_state_identity{suffix}"
    a2_fig = REPO_ROOT / "figures" / "HMM" / "neural_alignment" / f"lfp_three_region_granger{suffix}"
    for d in (a1_out, a2_out, a1_fig, a2_fig):
        d.mkdir(parents=True, exist_ok=True)
    return a1_out, a2_out, a1_fig, a2_fig


# ---- LFP loading ----
def load_regional(sn, prep_dir):
    """Return dict of region → regional float32 trace (500 Hz, foraging-truncated),
    and dict region → artifact mask. Aligned to min common length."""
    rec_regional = {}
    rec_artifact = {}
    for r in REGIONS:
        rec_regional[r] = np.load(prep_dir / f"session_{sn}_{r}_regional.npy")
        rec_artifact[r] = np.load(prep_dir / f"session_{sn}_{r}_artifact_mask.npy")
    n_common = min(len(v) for v in rec_regional.values())
    rec_regional = {r: v[:n_common] for r, v in rec_regional.items()}
    rec_artifact = {r: v[:n_common] for r, v in rec_artifact.items()}
    return rec_regional, rec_artifact, n_common


def load_viterbi(sn, cfg, n_samples_ds):
    """Load Viterbi from merged posteriors, truncate to len consistent with LFP."""
    post = pd.read_csv(REPO_ROOT / cfg["merge_dirs"]["posteriors"]
                        / f"session_{sn}.csv")
    viterbi = post["viterbi"].values.astype(np.int64)
    n_hmm = min(len(viterbi), n_samples_ds // SAMPLES_PER_HMM_BIN)
    return viterbi[:n_hmm]


def per_bin_artifact_mask(artifact_ds, n_hmm):
    """Bin-level artifact: True if >10% samples in bin flagged."""
    mask = np.zeros(n_hmm, dtype=bool)
    for t in range(n_hmm):
        s = t * SAMPLES_PER_HMM_BIN
        e = s + SAMPLES_PER_HMM_BIN
        if e <= len(artifact_ds):
            mask[t] = artifact_ds[s:e].mean() > 0.1
    return mask


# ---- Multitaper PSD per bin ----
def multitaper_band_power_per_bin(signal, n_hmm, fs):
    """For each HMM bin (240 samples), compute multitaper PSD and aggregate to
    5 band-mean powers. Returns (n_hmm, n_bands) float32."""
    win = SAMPLES_PER_HMM_BIN
    tapers = windows.dpss(win, DPSS_NW, DPSS_K)
    freqs = np.fft.rfftfreq(win, d=1.0 / fs)
    band_masks = []
    for bn in BAND_NAMES:
        lo, hi = BANDS[bn]
        band_masks.append((freqs >= lo) & (freqs <= hi))
    out = np.full((n_hmm, len(BAND_NAMES)), np.nan, dtype=np.float32)
    for t in range(n_hmm):
        s = t * win
        e = s + win
        if e > len(signal):
            continue
        seg = signal[s:e]
        # average over tapers
        psd = np.zeros(len(freqs), dtype=np.float64)
        for k in range(DPSS_K):
            psd += np.abs(np.fft.rfft(seg * tapers[k])) ** 2
        psd /= DPSS_K
        for bi, m in enumerate(band_masks):
            if m.any():
                out[t, bi] = psd[m].mean()
    return out


# ---- Analysis 1: LFP state identity ----
def category_masks(viterbi, artifact_per_bin):
    """Per category: bool mask of valid bins."""
    out = {}
    for cat, st in CATEGORY_STATES.items():
        out[cat] = (viterbi == st) & ~artifact_per_bin
    return out


def kruskal_per_region_band(band_power_per_region, cat_masks):
    """Returns DataFrame: region, band, H, p, n per category."""
    rows = []
    for region in REGIONS:
        bp = band_power_per_region[region]
        for bi, bn in enumerate(BAND_NAMES):
            groups = []
            ns = {}
            for cat in CATEGORIES:
                m = cat_masks[cat]
                vals = bp[m, bi]
                vals = vals[np.isfinite(vals)]
                ns[cat] = int(len(vals))
                groups.append(vals)
            if any(n < MIN_BINS for n in ns.values()):
                rows.append(dict(region=region, band=bn, H=np.nan, p=np.nan,
                                  n_home=ns["home"], n_feeding=ns["feeding"],
                                  n_transition_zone=ns["transition_zone"]))
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    H, p = kruskal(*groups)
                except Exception:
                    H, p = np.nan, np.nan
            rows.append(dict(region=region, band=bn, H=float(H), p=float(p),
                              n_home=ns["home"], n_feeding=ns["feeding"],
                              n_transition_zone=ns["transition_zone"]))
    return pd.DataFrame(rows)


def pairwise_mw_per_region_band(band_power_per_region, cat_masks):
    """For each (region, band, pair): Mann-Whitney U. Returns DataFrame."""
    pairs = [("home", "feeding"),
             ("home", "transition_zone"),
             ("feeding", "transition_zone")]
    rows = []
    for region in REGIONS:
        bp = band_power_per_region[region]
        for bi, bn in enumerate(BAND_NAMES):
            for a, b in pairs:
                va = bp[cat_masks[a], bi]
                vb = bp[cat_masks[b], bi]
                va = va[np.isfinite(va)]
                vb = vb[np.isfinite(vb)]
                if len(va) < MIN_BINS or len(vb) < MIN_BINS:
                    rows.append(dict(region=region, band=bn,
                                      pair=f"{a}_vs_{b}",
                                      U=np.nan, p=np.nan,
                                      mean_a=float(va.mean()) if len(va) else np.nan,
                                      mean_b=float(vb.mean()) if len(vb) else np.nan,
                                      n_a=int(len(va)), n_b=int(len(vb))))
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        U, p = mannwhitneyu(va, vb, alternative="two-sided")
                    except Exception:
                        U, p = np.nan, np.nan
                rows.append(dict(region=region, band=bn,
                                  pair=f"{a}_vs_{b}",
                                  U=float(U), p=float(p),
                                  mean_a=float(va.mean()),
                                  mean_b=float(vb.mean()),
                                  n_a=int(len(va)), n_b=int(len(vb))))
    return pd.DataFrame(rows)


def fdr_within_session(df, p_col="p"):
    pv = df[p_col].values
    finite = np.isfinite(pv)
    p_adj = np.full(len(pv), np.nan)
    sig = np.zeros(len(pv), dtype=bool)
    if finite.any():
        rej, p_corr, _, _ = multipletests(pv[finite], alpha=FDR_ALPHA,
                                            method="fdr_bh")
        p_adj[finite] = p_corr
        sig[finite] = rej
    df = df.copy()
    df["p_FDR"] = p_adj
    df["sig_FDR"] = sig
    return df


def analysis_1_session(sn, regional, artifact, viterbi, artifact_per_bin,
                        a1_out, a1_fig, rng):
    print(f"  A1 S{sn}...", flush=True)
    n_hmm = len(viterbi)

    band_power = {r: multitaper_band_power_per_bin(regional[r], n_hmm, FS_LFP_DS)
                  for r in REGIONS}

    cat_masks = category_masks(viterbi, artifact_per_bin)
    for cat in CATEGORIES:
        print(f"    {cat}: {cat_masks[cat].sum()} valid bins", flush=True)

    # Mean power per category
    rows_bp = []
    for region in REGIONS:
        bp = band_power[region]
        for bi, bn in enumerate(BAND_NAMES):
            for cat in CATEGORIES:
                vals = bp[cat_masks[cat], bi]
                vals = vals[np.isfinite(vals)]
                rows_bp.append(dict(region=region, band=bn, category=cat,
                                      mean_power=float(vals.mean()) if len(vals) else np.nan,
                                      sd_power=float(vals.std()) if len(vals) else np.nan,
                                      n_bins=int(len(vals))))
    pd.DataFrame(rows_bp).to_csv(
        a1_out / f"A1_session_{sn}_band_power_per_category.csv", index=False)

    # Observed Kruskal-Wallis
    df_k = kruskal_per_region_band(band_power, cat_masks)
    df_k = fdr_within_session(df_k, p_col="p")

    # Pairwise MW
    df_pw = pairwise_mw_per_region_band(band_power, cat_masks)
    df_pw = fdr_within_session(df_pw, p_col="p")

    # Shuffles for omnibus: build (region, band) -> observed H, shuffle H array
    T = len(viterbi)
    H_shuf = {(region, bn): np.full(N_SHUFFLES, np.nan)
              for region in REGIONS for bn in BAND_NAMES}
    U_shuf = {(region, bn, pair): np.full(N_SHUFFLES, np.nan)
              for region in REGIONS for bn in BAND_NAMES
              for pair in ("home_vs_feeding", "home_vs_transition_zone",
                            "feeding_vs_transition_zone")}

    for it in range(N_SHUFFLES):
        offset = int(rng.integers(SHUFFLE_MIN_OFFSET, T - SHUFFLE_MARGIN))
        v_shuf = np.roll(viterbi, offset)
        cm = category_masks(v_shuf, artifact_per_bin)

        for region in REGIONS:
            bp = band_power[region]
            for bi, bn in enumerate(BAND_NAMES):
                groups = []
                ns = []
                for cat in CATEGORIES:
                    vals = bp[cm[cat], bi]
                    vals = vals[np.isfinite(vals)]
                    groups.append(vals); ns.append(len(vals))
                if any(n < MIN_BINS for n in ns):
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        H, _ = kruskal(*groups)
                    except Exception:
                        H = np.nan
                H_shuf[(region, bn)][it] = H
                # Pairwise U for each pair
                for a, b in [("home", "feeding"),
                              ("home", "transition_zone"),
                              ("feeding", "transition_zone")]:
                    va = bp[cm[a], bi]; vb = bp[cm[b], bi]
                    va = va[np.isfinite(va)]; vb = vb[np.isfinite(vb)]
                    if len(va) < MIN_BINS or len(vb) < MIN_BINS:
                        continue
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        try:
                            U, _ = mannwhitneyu(va, vb, alternative="two-sided")
                        except Exception:
                            U = np.nan
                    U_shuf[(region, bn, f"{a}_vs_{b}")][it] = U

    # Compute shuffle percentile per cell
    H_pct = []
    H_pass = []
    for _, r in df_k.iterrows():
        key = (r["region"], r["band"])
        shuf = H_shuf[key]
        if not np.isfinite(r["H"]) or not np.isfinite(shuf).any():
            H_pct.append(np.nan); H_pass.append(False)
            continue
        valid = shuf[np.isfinite(shuf)]
        pct = float((valid < r["H"]).mean() * 100)
        p95 = float(np.percentile(valid, 95))
        H_pct.append(pct)
        H_pass.append(bool(r["H"] > p95))
    df_k["observed_pctile_shuffle"] = H_pct
    df_k["exceeds_p95"] = H_pass
    df_k.to_csv(a1_out / f"A1_session_{sn}_kruskal.csv", index=False)

    U_pct = []
    U_pass = []
    for _, r in df_pw.iterrows():
        key = (r["region"], r["band"], r["pair"])
        shuf = U_shuf[key]
        if not np.isfinite(r["U"]) or not np.isfinite(shuf).any():
            U_pct.append(np.nan); U_pass.append(False)
            continue
        valid = shuf[np.isfinite(shuf)]
        # Two-sided: compare |U - shuffle median| vs |shuf - shuf median|
        shuf_med = float(np.median(valid))
        obs_abs = abs(r["U"] - shuf_med)
        shuf_abs = np.abs(valid - shuf_med)
        pct = float((shuf_abs < obs_abs).mean() * 100)
        p95 = float(np.percentile(shuf_abs, 95))
        U_pct.append(pct)
        U_pass.append(bool(obs_abs > p95))
    df_pw["observed_pctile_shuffle"] = U_pct
    df_pw["exceeds_p95"] = U_pass
    df_pw.to_csv(a1_out / f"A1_session_{sn}_pairwise.csv", index=False)

    # Spectral fingerprint figure per region
    for region in REGIONS:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        x = np.arange(len(BAND_NAMES))
        for cat in CATEGORIES:
            means = [df_k.loc[
                (df_k.region == region), :
            ].pipe(lambda d: d).query(f"band==@bn")
                     for bn in BAND_NAMES]  # unused, fallback below
        # Use the band-power-per-category CSV rows directly
        bp_df = pd.DataFrame(rows_bp)
        sub = bp_df[bp_df.region == region]
        for cat in CATEGORIES:
            cdf = sub[sub.category == cat].set_index("band").reindex(BAND_NAMES)
            ax.errorbar(x, cdf["mean_power"], yerr=cdf["sd_power"]
                          / np.sqrt(cdf["n_bins"].replace(0, np.nan)),
                          marker="o", capsize=3, label=cat)
        ax.set_xticks(x); ax.set_xticklabels(BAND_NAMES, rotation=30)
        ax.set_ylabel("Mean band power (µV² / Hz)")
        ax.set_yscale("log")
        ax.set_title(f"S{sn} ({SESSION_STATE[sn]}) — {region} spectral fingerprint")
        ax.legend()
        fig.tight_layout()
        fig.savefig(a1_fig / f"A1_session_{sn}_spectral_fingerprint_{region}.png",
                     dpi=130)
        plt.close(fig)

    return dict(sn=sn, df_k=df_k, df_pw=df_pw, band_power_summary=pd.DataFrame(rows_bp))


# ---- Analysis 1: cross-session aggregation ----
def analysis_1_cross_session(per_sess, a1_out, a1_fig):
    print("  A1 cross-session aggregation...", flush=True)
    # Kruskal replication
    rep_k_rows = []
    for region in REGIONS:
        for bn in BAND_NAMES:
            n_tested = 0; n_passing = 0
            for r in per_sess:
                row = r["df_k"]
                rec = row[(row.region == region) & (row.band == bn)]
                if not len(rec):
                    continue
                if not np.isfinite(rec.iloc[0]["H"]):
                    continue
                n_tested += 1
                if bool(rec.iloc[0]["exceeds_p95"]):
                    n_passing += 1
            rep_k_rows.append(dict(region=region, band=bn,
                                    n_sessions_tested=n_tested,
                                    n_passing=n_passing,
                                    replicates=bool(n_passing >= REPLICATION_THRESHOLD)))
    df_rep_k = pd.DataFrame(rep_k_rows)
    df_rep_k.to_csv(a1_out / "A1_kruskal_replication.csv", index=False)

    # Pairwise replication
    rep_pw_rows = []
    for region in REGIONS:
        for bn in BAND_NAMES:
            for pair in ("home_vs_feeding", "home_vs_transition_zone",
                          "feeding_vs_transition_zone"):
                n_tested = 0; n_passing = 0
                for r in per_sess:
                    row = r["df_pw"]
                    rec = row[(row.region == region) & (row.band == bn)
                               & (row.pair == pair)]
                    if not len(rec):
                        continue
                    if not np.isfinite(rec.iloc[0]["U"]):
                        continue
                    n_tested += 1
                    if bool(rec.iloc[0]["exceeds_p95"]):
                        n_passing += 1
                rep_pw_rows.append(dict(region=region, band=bn, pair=pair,
                                          n_sessions_tested=n_tested,
                                          n_passing=n_passing,
                                          replicates=bool(n_passing >= REPLICATION_THRESHOLD)))
    df_rep_pw = pd.DataFrame(rep_pw_rows)
    df_rep_pw.to_csv(a1_out / "A1_pairwise_replication.csv", index=False)

    # Replication heatmap
    cols = ["kruskal", "home_vs_feeding", "home_vs_transition_zone",
            "feeding_vs_transition_zone"]
    rows = []
    for region in REGIONS:
        for bn in BAND_NAMES:
            d = dict(label=f"{region}_{bn}")
            kr = df_rep_k[(df_rep_k.region == region) & (df_rep_k.band == bn)]
            d["kruskal"] = int(kr.iloc[0]["n_passing"]) if len(kr) else 0
            for pair in cols[1:]:
                pr = df_rep_pw[(df_rep_pw.region == region) & (df_rep_pw.band == bn)
                                  & (df_rep_pw.pair == pair)]
                d[pair] = int(pr.iloc[0]["n_passing"]) if len(pr) else 0
            rows.append(d)
    df_heat = pd.DataFrame(rows).set_index("label")[cols]
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(df_heat.values, aspect="auto", cmap="Reds",
                    vmin=0, vmax=len(per_sess))
    ax.set_yticks(np.arange(len(df_heat)))
    ax.set_yticklabels(df_heat.index, fontsize=8)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=9)
    for i in range(len(df_heat)):
        for j in range(len(cols)):
            ax.text(j, i, str(df_heat.values[i, j]),
                     ha="center", va="center", fontsize=7,
                     color="white" if df_heat.values[i, j] >= 4 else "black")
    ax.set_title(f"A1 replication count per cell (n_sessions/{len(per_sess)})")
    plt.colorbar(im, ax=ax, label="n sessions passing shuffle p95")
    fig.tight_layout()
    fig.savefig(a1_fig / "A1_replication_heatmap.png", dpi=130)
    plt.close(fig)

    # Aggregate spectral fingerprints: 3 regions × 5 bands, 3 lines per region (cat)
    bp_all = pd.concat([r["band_power_summary"].assign(session=r["sn"])
                         for r in per_sess], ignore_index=True)
    bp_all.to_csv(a1_out / "A1_band_power_per_category_all_sessions.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    x = np.arange(len(BAND_NAMES))
    for ax, region in zip(axes, REGIONS):
        for cat in CATEGORIES:
            agg = bp_all[(bp_all.region == region) & (bp_all.category == cat)]
            grouped = agg.groupby("band")["mean_power"]
            means = [grouped.get_group(bn).mean() if bn in grouped.groups else np.nan
                     for bn in BAND_NAMES]
            sems = [grouped.get_group(bn).sem() if bn in grouped.groups else np.nan
                    for bn in BAND_NAMES]
            ax.errorbar(x, means, yerr=sems, marker="o", capsize=3, label=cat)
        ax.set_xticks(x); ax.set_xticklabels(BAND_NAMES, rotation=30)
        ax.set_yscale("log")
        ax.set_title(f"{region} (cross-session mean ± SEM)")
        ax.legend()
    axes[0].set_ylabel("Mean band power (µV²/Hz)")
    fig.suptitle("A1 spectral fingerprints — cross-session")
    fig.tight_layout()
    fig.savefig(a1_fig / "A1_spectral_fingerprints_aggregate.png", dpi=130)
    plt.close(fig)

    return df_rep_k, df_rep_pw


# ---- Analysis 2: three-region Granger ----
def build_S3_segments(viterbi, n_ds, artifact_any):
    """Find S3 runs (length ≥ 2*K=6 bins, S3 portion ≥ 5 s). Return list of
    (start_ds, end_ds) sample indices spanning S3 stay + 5 s post-exit."""
    K_PRE = 3
    diff = np.diff(viterbi, prepend=-1, append=-1)
    boundaries = np.flatnonzero(diff != 0)
    starts = boundaries[:-1]; ends = boundaries[1:]
    segs = []
    for s, e in zip(starts, ends):
        if int(viterbi[s]) != S3_STATE:
            continue
        L = e - s
        if L < 2 * K_PRE:
            continue
        s_ds = s * SAMPLES_PER_HMM_BIN
        e_ds = e * SAMPLES_PER_HMM_BIN
        if e_ds - s_ds < GRANGER_MIN_S3_SAMPLES:
            continue
        seg_end = min(n_ds, e_ds + POST_EXIT_SAMPLES)
        if seg_end - s_ds < GRANGER_MIN_S3_SAMPLES + 100:
            continue
        segs.append((s_ds, seg_end))
    return segs


def bandpass_envelope(signal, band, fs):
    f_lo, f_hi = band
    sos = butter(6, [f_lo / (fs / 2), f_hi / (fs / 2)],
                  btype="bandpass", output="sos")
    return np.abs(hilbert(sosfiltfilt(sos, signal)))


def build_lagged(seg_x, seg_y, p):
    Xf_list = []; Xr_list = []; y_list = []
    for sx, sy in zip(seg_x, seg_y):
        L = len(sy)
        if L <= p:
            continue
        n = L - p
        y_lags = np.column_stack([sy[p - k:L - k] for k in range(1, p + 1)])
        x_lags = np.column_stack([sx[p - k:L - k] for k in range(1, p + 1)])
        intercept = np.ones((n, 1))
        Xr_list.append(np.column_stack([y_lags, intercept]))
        Xf_list.append(np.column_stack([y_lags, x_lags, intercept]))
        y_list.append(sy[p:L])
    if not y_list:
        return None, None, None
    return np.vstack(Xf_list), np.vstack(Xr_list), np.concatenate(y_list)


def granger_F(seg_x, seg_y, p):
    Xu, Xr, y = build_lagged(seg_x, seg_y, p)
    if y is None or len(y) <= 2 * p + 2:
        return np.nan, np.nan
    try:
        beta_r, *_ = np.linalg.lstsq(Xr, y, rcond=None)
        beta_u, *_ = np.linalg.lstsq(Xu, y, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    RSS_r = float(np.sum((y - Xr @ beta_r) ** 2))
    RSS_u = float(np.sum((y - Xu @ beta_u) ** 2))
    n = len(y); df_diff = p; df_u = n - (2 * p + 1)
    if df_u <= 0 or RSS_u <= 1e-12:
        return np.nan, np.nan
    F = ((RSS_r - RSS_u) / df_diff) / (RSS_u / df_u)
    pval = 1.0 - f_dist.cdf(F, df_diff, df_u)
    return float(F), float(pval)


def select_lag_BIC(seg_x, seg_y):
    best_p = GRANGER_LAG_RANGE[0]; best_bic = np.inf
    for p in GRANGER_LAG_RANGE:
        Xu, _, y = build_lagged(seg_x, seg_y, p)
        if y is None or len(y) <= 2 * p + 2:
            continue
        try:
            beta_u, *_ = np.linalg.lstsq(Xu, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        RSS = float(np.sum((y - Xu @ beta_u) ** 2))
        if RSS <= 1e-12:
            continue
        n = len(y); k = 2 * p + 1
        bic = n * np.log(RSS / n) + k * np.log(n)
        if bic < best_bic:
            best_bic = bic; best_p = p
    return best_p


PAIR_DEFS = [
    ("ACA-LHA", "ACA", "LHA"),
    ("ACA-RSP", "ACA", "RSP"),
    ("LHA-RSP", "LHA", "RSP"),
]


def analysis_2_session(sn, regional, viterbi, a2_out, a2_fig, rng):
    print(f"  A2 S{sn}...", flush=True)
    n_ds = len(regional["ACA"])
    segs = build_S3_segments(viterbi, n_ds, None)
    print(f"    {len(segs)} S3 segments", flush=True)
    if not segs:
        return dict(sn=sn, df_obs=pd.DataFrame())

    rows = []
    T = n_ds
    for band_name, band_range in BANDS.items():
        envs = {r: bandpass_envelope(regional[r], band_range, FS_LFP_DS)
                for r in REGIONS}
        envs = {r: (envs[r] - envs[r].mean()) / (envs[r].std() + 1e-9)
                for r in REGIONS}

        seg_env = {r: [envs[r][s:e] for s, e in segs] for r in REGIONS}

        for pair_name, A, B in PAIR_DEFS:
            for direction in ("forward", "reverse"):
                src, tgt = (A, B) if direction == "forward" else (B, A)
                # BIC lag selection
                p = select_lag_BIC(seg_env[src], seg_env[tgt])
                F_obs, p_obs = granger_F(seg_env[src], seg_env[tgt], p)
                # Shuffles
                shuf_F = np.full(N_SHUFFLES, np.nan)
                for it in range(N_SHUFFLES):
                    offset = int(rng.integers(SHUFFLE_MIN_OFFSET,
                                                  T - SHUFFLE_MARGIN))
                    src_shuf = np.roll(envs[src], offset)
                    seg_src_shuf = [src_shuf[s:e] for s, e in segs]
                    shuf_F[it], _ = granger_F(seg_src_shuf, seg_env[tgt], p)
                valid = shuf_F[np.isfinite(shuf_F)]
                p95 = float(np.percentile(valid, 95)) if len(valid) else np.nan

                direction_label = f"{src}->{tgt}"
                rows.append(dict(band=band_name, pair=pair_name,
                                  direction=direction_label,
                                  selected_lag=int(p),
                                  observed_F=F_obs,
                                  F_p=p_obs,
                                  shuffle_p95=p95,
                                  exceeds_p95=bool(np.isfinite(F_obs)
                                                      and np.isfinite(p95)
                                                      and F_obs > p95)))
        print(f"    {band_name}: done", flush=True)
    df_obs = pd.DataFrame(rows)
    df_obs["session"] = sn
    df_obs.to_csv(a2_out / f"A2_session_{sn}_granger.csv", index=False)
    return dict(sn=sn, df_obs=df_obs)


def analysis_2_cross_session(per_sess, a2_out, a2_fig):
    print("  A2 cross-session aggregation...", flush=True)
    all_df = pd.concat([r["df_obs"] for r in per_sess if len(r["df_obs"])],
                        ignore_index=True)
    all_df.to_csv(a2_out / "A2_granger_F_table.csv", index=False)

    sign_rows = []
    for band in BAND_NAMES:
        for pair_name, A, B in PAIR_DEFS:
            forward = f"{A}->{B}"
            reverse = f"{B}->{A}"
            n_total = 0
            n_forward_leads = 0
            for sn in SESSIONS:
                fw = all_df[(all_df.session == sn) & (all_df.band == band)
                              & (all_df.pair == pair_name)
                              & (all_df.direction == forward)]["observed_F"]
                rv = all_df[(all_df.session == sn) & (all_df.band == band)
                              & (all_df.pair == pair_name)
                              & (all_df.direction == reverse)]["observed_F"]
                if not len(fw) or not len(rv):
                    continue
                fwv, rvv = float(fw.iloc[0]), float(rv.iloc[0])
                if not (np.isfinite(fwv) and np.isfinite(rvv)):
                    continue
                n_total += 1
                if fwv > rvv:
                    n_forward_leads += 1
            try:
                bp = (binomtest(n_forward_leads, n_total, 0.5,
                                  alternative="two-sided").pvalue
                      if n_total else np.nan)
            except Exception:
                bp = np.nan
            sign_rows.append(dict(band=band, pair=pair_name,
                                    forward=forward, reverse=reverse,
                                    n_sessions=n_total,
                                    n_forward_leads=n_forward_leads,
                                    n_reverse_leads=n_total - n_forward_leads,
                                    binom_p=bp))
    df_sign = pd.DataFrame(sign_rows)
    df_sign.to_csv(a2_out / "A2_sign_test.csv", index=False)

    rep_rows = []
    for band in BAND_NAMES:
        for pair_name, A, B in PAIR_DEFS:
            for direction in (f"{A}->{B}", f"{B}->{A}"):
                sub = all_df[(all_df.band == band) & (all_df.pair == pair_name)
                              & (all_df.direction == direction)]
                n_passing = int(sub["exceeds_p95"].sum())
                rep_rows.append(dict(band=band, pair=pair_name, direction=direction,
                                       n_sessions_tested=len(sub),
                                       n_passing=n_passing,
                                       replicates=bool(n_passing >= REPLICATION_THRESHOLD)))
    df_rep = pd.DataFrame(rep_rows)
    df_rep.to_csv(a2_out / "A2_replication.csv", index=False)

    # Figure 1: F per session per pair, bar charts
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=False)
    for ax, (pair_name, A, B) in zip(axes, PAIR_DEFS):
        x = np.arange(len(SESSIONS))
        width = 0.08
        for bi, band in enumerate(BAND_NAMES):
            for dj, direction in enumerate((f"{A}->{B}", f"{B}->{A}")):
                vals = []
                for sn in SESSIONS:
                    rec = all_df[(all_df.session == sn) & (all_df.band == band)
                                   & (all_df.pair == pair_name)
                                   & (all_df.direction == direction)]
                    vals.append(float(rec.iloc[0]["observed_F"])
                                if len(rec) and np.isfinite(rec.iloc[0]["observed_F"])
                                else 0)
                offset = (bi - 2) * 2 * width + (dj * width)
                color = plt.get_cmap("tab10")(bi)
                hatch = "" if dj == 0 else "//"
                ax.bar(x + offset, vals, width, color=color, hatch=hatch,
                        label=f"{band} {'fwd' if dj == 0 else 'rev'}"
                              if x[0] == 0 and pair_name == "ACA-LHA" else None)
        ax.set_xticks(x); ax.set_xticklabels([f"S{s}" for s in SESSIONS])
        ax.set_title(f"{pair_name}: fwd ({A}→{B}) plain, rev ({B}→{A}) hatched")
        ax.set_ylabel("F")
    fig.suptitle("A2 LFP Granger F values per session × band × direction")
    fig.tight_layout()
    fig.savefig(a2_fig / "A2_F_per_session_per_pair.png", dpi=130)
    plt.close(fig)

    # Figure 2: replication heatmap (pair × direction rows, band columns)
    row_labels = []
    grid = []
    for pair_name, A, B in PAIR_DEFS:
        for direction in (f"{A}->{B}", f"{B}->{A}"):
            row = []
            for band in BAND_NAMES:
                rec = df_rep[(df_rep.pair == pair_name) & (df_rep.direction == direction)
                               & (df_rep.band == band)]
                row.append(int(rec.iloc[0]["n_passing"]) if len(rec) else 0)
            row_labels.append(f"{pair_name} {direction}")
            grid.append(row)
    grid = np.array(grid)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(grid, aspect="auto", cmap="Reds", vmin=0, vmax=len(SESSIONS))
    ax.set_xticks(np.arange(len(BAND_NAMES)))
    ax.set_xticklabels(BAND_NAMES, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    for i in range(len(row_labels)):
        for j in range(len(BAND_NAMES)):
            ax.text(j, i, str(grid[i, j]), ha="center", va="center",
                     fontsize=8,
                     color="white" if grid[i, j] >= 4 else "black")
    plt.colorbar(im, ax=ax, label=f"n_sessions/{len(SESSIONS)} passing shuffle p95")
    ax.set_title("A2 replication count per pair × direction × band")
    fig.tight_layout()
    fig.savefig(a2_fig / "A2_replication_heatmap.png", dpi=130)
    plt.close(fig)

    # Figure 3: sign test summary (3 panels, one per pair)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    for ax, (pair_name, A, B) in zip(axes, PAIR_DEFS):
        sub = df_sign[df_sign.pair == pair_name].set_index("band").reindex(BAND_NAMES)
        x = np.arange(len(BAND_NAMES))
        ax.bar(x, sub["n_forward_leads"], color="C0",
                label=f"{A}→{B} leads")
        ax.bar(x, sub["n_reverse_leads"], bottom=sub["n_forward_leads"],
                color="C1", label=f"{B}→{A} leads")
        for i, bp in enumerate(sub["binom_p"].values):
            if pd.notna(bp):
                ax.text(i, len(SESSIONS) * 1.02, f"p={bp:.2g}",
                         ha="center", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(BAND_NAMES, rotation=30)
        ax.set_ylim(0, len(SESSIONS) * 1.15)
        ax.set_title(pair_name)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("n_sessions leading")
    fig.suptitle("A2 sign test: which direction has larger F per session")
    fig.tight_layout()
    fig.savefig(a2_fig / "A2_sign_test_summary.png", dpi=130)
    plt.close(fig)

    return df_rep, df_sign


# ---- Main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-version", choices=("v1", "v2"), default="v1",
                     help="v1: preprocessed_v2 LFP from script 17 v2; "
                          "v2: preprocessed_v3 LFP from script 17 v3 (refined LHA/RSP)")
    args = ap.parse_args()
    print(f"=== Pairs version: {args.pairs_version} ===")

    cfg = load_config()
    a1_out, a2_out, a1_fig, a2_fig = out_dirs(args.pairs_version)
    prep_dir = lfp_prep_dir(args.pairs_version)
    print(f"Preprocessed LFP from: {prep_dir}")
    print(f"A1 out: {a1_out}\nA2 out: {a2_out}")
    rng = np.random.default_rng(SEED)

    per_sess_a1 = []
    per_sess_a2 = []
    for sn in SESSIONS:
        print(f"\n========== S{sn} ==========", flush=True)
        t0 = time.time()
        regional, artifact, n_common = load_regional(sn, prep_dir)
        viterbi = load_viterbi(sn, cfg, n_common)
        n_hmm = len(viterbi)
        # Combined artifact mask (any region flagged)
        artifact_any = (artifact["ACA"] | artifact["LHA"] | artifact["RSP"])
        artifact_per_bin = per_bin_artifact_mask(artifact_any, n_hmm)
        print(f"  n_hmm={n_hmm} bins, artifact bins={artifact_per_bin.sum()} "
              f"({artifact_per_bin.mean()*100:.2f}%)", flush=True)

        r1 = analysis_1_session(sn, regional, artifact, viterbi, artifact_per_bin,
                                  a1_out, a1_fig, rng)
        per_sess_a1.append(r1)
        r2 = analysis_2_session(sn, regional, viterbi, a2_out, a2_fig, rng)
        per_sess_a2.append(r2)
        print(f"  S{sn} done [{time.time()-t0:.0f}s]", flush=True)

    print("\n========== A1 cross-session ==========", flush=True)
    df_rep_k, df_rep_pw = analysis_1_cross_session(per_sess_a1, a1_out, a1_fig)
    print("\n========== A2 cross-session ==========", flush=True)
    df_rep_a2, df_sign = analysis_2_cross_session(per_sess_a2, a2_out, a2_fig)

    # Print summary
    print("\n========== A1 summary ==========")
    for region in REGIONS:
        sub = df_rep_k[df_rep_k.region == region]
        n_rep = int(sub["replicates"].sum())
        print(f"  {region}: {n_rep}/5 bands replicate Kruskal (≥{REPLICATION_THRESHOLD}/{len(SESSIONS)})")
        for pair in ("home_vs_feeding", "home_vs_transition_zone",
                       "feeding_vs_transition_zone"):
            sub_pw = df_rep_pw[(df_rep_pw.region == region) & (df_rep_pw.pair == pair)]
            n_rep_pw = int(sub_pw["replicates"].sum())
            print(f"    {pair}: {n_rep_pw}/5 bands replicate")
    sub = df_rep_k.sort_values("n_passing", ascending=False).head(6)
    print(f"  Top Kruskal cells: {[(r['region'], r['band'], int(r['n_passing'])) for _, r in sub.iterrows()]}")

    print("\n========== A2 summary ==========")
    for pair_name, A, B in PAIR_DEFS:
        print(f"  {pair_name}:")
        for band in BAND_NAMES:
            rec = df_sign[(df_sign.pair == pair_name) & (df_sign.band == band)]
            if not len(rec):
                continue
            r = rec.iloc[0]
            print(f"    {band:10s}: {A}→{B} leads {r['n_forward_leads']}/{r['n_sessions']} "
                  f"(binom p={r['binom_p']:.3f})")
    print("Done.")


if __name__ == "__main__":
    main()
