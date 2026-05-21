"""17 — LFP spectral analysis at HMM state transitions, foraging sessions.

CORRECTED 2026-05-11: bipolar referencing now uses the static probe geometry
parsed in `scripts/HMM/lfp_parse_geometry.py` (see
`data/HMM/neural_alignment/lfp/bipolar_pairs_imec{0,1}.csv`). The old pipeline
paired channels by SpikeGLX file order (`raw[:,:-1] - raw[:,1:]`), which does
not match physical adjacency under the IMRO map. The S12 diagnostics
(`scripts/HMM/lfp_diagnostics_S12.py`) showed single-ended ACA-LHA delta
r ≈ 0.948 (extreme common-mode); geometry-correct bipolar reduces |r| to
~0.009 (99% reduction). All downstream M1-M4 analyses operate on the
geometry-bipolar regional means produced here.

Probe → region mapping (from static geometry):
  imec0 → ACA (all 384 channels, 370 within-shank ≤30 µm pairs)
  imec1 → LHA  (channels with y_um < 2500: 192 channels, 184 pairs)
          RSP  (channels with y_um ≥ 2500: 192 channels, 184 pairs)

Pipeline:
  Preprocess (per session × probe, chunked):
    Read int16 → gain → µV → notch 60/120 Hz → bipolar per static pair →
    mean within region → anti-alias LP @ 200 Hz → decimate to 500 Hz →
    artifact mask (envelope > 5σ proxy).
    Save: ACA, LHA, RSP regional traces + artifact masks.
    Sanity-check: regional ACA vs LHA Pearson r (should be ≪ 0.5).

  Analyses (unchanged from prior version, just use new regional traces):
    M1 — Band power per state × region (Welch on cached PSD).
    M2 — Event-aligned spectrograms (multitaper sliding window).
    M3 — ACA-LHA coherence per state, per band.
    M4 — LFP Granger per band (Hilbert envelope, S3 stay+post-exit segments).

Outputs land in:
  data/HMM/neural_alignment/lfp_spectral/preprocessed_v2/
  data/HMM/neural_alignment/lfp_spectral/session_{N}/*.csv
  figures/HMM/neural_alignment/lfp_spectral/session_{N}/*.png
"""
from pathlib import Path
import argparse
import re
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import (butter, sosfiltfilt, iirnotch, tf2sos,
                            welch, csd, hilbert, windows)
from scipy.stats import mannwhitneyu, f as f_dist, ttest_rel, binomtest, pearsonr
from statsmodels.stats.multitest import multipletests

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "HMM"))
from _utils import load_config


# ---- Constants ----
CATGT_ROOT = Path("H:/Neuropixels Data/Cat_GT_Out")
GEOMETRY_DIR = REPO_ROOT / "data" / "HMM" / "neural_alignment" / "lfp"

SESSION_DATE_MAP = {
    4:  ("6_17_25", "fed"),
    6:  ("6_24_25", "fed"),
    8:  ("6_30_25", "fed"),
    12: ("7_11_25", "fasted"),
    14: ("7_17_25", "fasted"),
    16: ("7_25_25", "fasted"),
}
SESSIONS = sorted(SESSION_DATE_MAP.keys())

FS_LFP_RAW = 2500.0
FS_LFP_DS = 500.0
DOWNSAMPLE_FACTOR = int(FS_LFP_RAW / FS_LFP_DS)
HMM_BIN_S = 0.480
SAMPLES_PER_HMM_BIN_DS = int(HMM_BIN_S * FS_LFP_DS)     # 240

N_NEURAL_CHANNELS = 384         # last channel is sync — skip
CHUNK_S = 30.0
CHUNK_SAMPLES_RAW = int(CHUNK_S * FS_LFP_RAW)
ARTIFACT_THRESHOLD_SD = 5.0

# Spectral bands
BANDS = {
    "delta": (1, 4),
    "theta": (4, 12),
    "beta": (15, 30),
    "low_gamma": (30, 60),
    "high_gamma": (60, 100),
}

# State analysis lists
ACA_STATES = [2, 3, 4, 6, 8, 9, 12]
LHA_STATES = [2, 3]
COH_STATES = sorted(set(ACA_STATES) | set(LHA_STATES))
MIN_BINS = 30
S3_STATE = 3
POST_EXIT_S = 5.0
POST_EXIT_BINS_DS = int(POST_EXIT_S * FS_LFP_DS)        # 2500
K_PRE_HMM = 3

# Shuffle
N_SHUFFLES = 100
SHUFFLE_MIN_OFFSET = 200
SHUFFLE_MARGIN = 200
SHUFFLE_SEED = 20260511     # bumped for the corrected pipeline
FDR_ALPHA = 0.05

# LFP Granger lag selection
GRANGER_LAG_RANGE = list(range(1, 21))
GRANGER_MIN_S3_DS = int(5.0 * FS_LFP_DS)


def out_dirs(pairs_version="v1"):
    suffix = "" if pairs_version == "v1" else "_v3"
    prep_subdir = "preprocessed_v2" if pairs_version == "v1" else "preprocessed_v3"
    base_out = REPO_ROOT / "data" / "HMM" / "neural_alignment" / f"lfp_spectral{suffix}"
    base_fig = REPO_ROOT / "figures" / "HMM" / "neural_alignment" / f"lfp_spectral{suffix}"
    (base_out / prep_subdir).mkdir(parents=True, exist_ok=True)
    (base_out / "spectrograms").mkdir(parents=True, exist_ok=True)
    base_fig.mkdir(parents=True, exist_ok=True)
    return base_out, base_fig, prep_subdir


# ---- Static geometry / bipolar pair tables ----
def load_static_geometry(pairs_version="v1"):
    """Load probe_geometry.csv + bipolar_pairs_imec{0,1}.csv. v1 uses original
    files; v2 uses _v2 files (LHA y<345, RSP y>4680). Returns dict keyed by
    (probe, region) → (channel_a_idx, channel_b_idx) int arrays."""
    geom_csv = GEOMETRY_DIR / "probe_geometry.csv"
    if pairs_version == "v1":
        p0_csv = GEOMETRY_DIR / "bipolar_pairs_imec0.csv"
        p1_csv = GEOMETRY_DIR / "bipolar_pairs_imec1.csv"
    else:
        p0_csv = GEOMETRY_DIR / "bipolar_pairs_imec0_v2.csv"
        p1_csv = GEOMETRY_DIR / "bipolar_pairs_imec1_v2.csv"
    for f in (geom_csv, p0_csv, p1_csv):
        if not f.exists():
            raise FileNotFoundError(
                f"Missing {f}. Run `scripts/HMM/lfp_parse_geometry.py` first."
            )
    geom = pd.read_csv(geom_csv)
    p0 = pd.read_csv(p0_csv)
    p1 = pd.read_csv(p1_csv)

    pairs_by_region = {}
    pairs_by_region[("imec0", "ACA")] = (
        p0["channel_a"].to_numpy(np.int64),
        p0["channel_b"].to_numpy(np.int64),
    )
    for region in ("LHA", "RSP"):
        sub = p1[p1["region"] == region]
        pairs_by_region[("imec1", region)] = (
            sub["channel_a"].to_numpy(np.int64),
            sub["channel_b"].to_numpy(np.int64),
        )

    print("Static geometry loaded:")
    for (probe, region), (a, b) in pairs_by_region.items():
        print(f"  {probe} {region}: {len(a)} bipolar pairs")
    return geom, pairs_by_region


# ---- LFP file discovery ----
def discover_lfp_files():
    rows = []
    cutoff = datetime(2025, 8, 29)
    if not CATGT_ROOT.exists():
        print(f"  WARNING: {CATGT_ROOT} not found")
        return pd.DataFrame()
    for folder in sorted(CATGT_ROOT.iterdir()):
        if not folder.is_dir() or not folder.name.startswith("catgt_DOUBLE_PROBE_"):
            continue
        name = folder.name
        if "HFD" in name or "_HOME" in name or "_EXP" in name:
            continue
        m = re.search(r"catgt_DOUBLE_PROBE_(\d+_\d+_\d+)_FOR", name)
        if not m:
            continue
        date_str = m.group(1)
        try:
            month, day, year = [int(x) for x in date_str.split("_")]
            dt = datetime(2000 + year, month, day)
        except Exception:
            continue
        if dt > cutoff:
            continue
        for probe in (0, 1):
            sub = folder / f"DOUBLE_PROBE_{date_str}_FOR_g0_imec{probe}"
            if not sub.exists():
                cand = list(folder.glob(f"DOUBLE_PROBE_{date_str}_FOR*_imec{probe}"))
                if not cand:
                    continue
                sub = cand[0]
            lf_bin = list(sub.glob("*.lf.bin"))
            lf_meta = list(sub.glob("*.lf.meta"))
            if not lf_bin or not lf_meta:
                continue
            rows.append(dict(
                date=date_str, folder=folder.name, probe=probe,
                lf_bin=str(lf_bin[0]), lf_meta=str(lf_meta[0]),
            ))
    return pd.DataFrame(rows)


def map_session_to_lfp(discover_df):
    out = {}
    for sn in SESSIONS:
        date_str, _ = SESSION_DATE_MAP[sn]
        rows = discover_df[discover_df["date"] == date_str]
        if not len(rows):
            month, day, year = date_str.split("_")
            for variant in (f"{int(month)}_{int(day)}_{year}",):
                rows = discover_df[discover_df["date"] == variant]
                if len(rows):
                    break
        if not len(rows):
            print(f"  WARNING: no LFP files for S{sn} ({date_str})")
            continue
        p0 = rows[rows.probe == 0]
        p1 = rows[rows.probe == 1]
        out[sn] = dict(
            imec0_bin=str(p0.iloc[0]["lf_bin"]) if len(p0) else None,
            imec0_meta=str(p0.iloc[0]["lf_meta"]) if len(p0) else None,
            imec1_bin=str(p1.iloc[0]["lf_bin"]) if len(p1) else None,
            imec1_meta=str(p1.iloc[0]["lf_meta"]) if len(p1) else None,
        )
    return out


# ---- LFP meta parsing ----
def parse_lf_meta(meta_path):
    meta = {}
    with open(meta_path, "r") as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                key, val = line.split("=", 1)
                meta[key] = val
    return meta


def lfp_gain_uV(meta):
    rng_max = float(meta.get("imAiRangeMax", 0.62))
    rng_min = float(meta.get("imAiRangeMin", -0.62))
    max_int = int(meta.get("imMaxInt", 2048))
    gain = 80.0
    if "imChan0lfGain" in meta:
        gain = float(meta["imChan0lfGain"])
    elif "imChan0apGain" in meta:
        gain = float(meta["imChan0apGain"])
    v_per_count = (rng_max - rng_min) / (2 * max_int * gain)
    return v_per_count * 1e6


# ---- Filter design ----
def design_notch_sos(f0, q, fs):
    b, a = iirnotch(f0, q, fs=fs)
    return tf2sos(b, a)


def design_lowpass_sos(cutoff, order, fs):
    return butter(order, cutoff / (fs / 2), btype="lowpass", output="sos")


# ---- Preprocessing ----
def preprocess_probe(bin_path, meta_path, probe_label, regions_pairs, sn,
                      out_dir, foraging_duration_s):
    """Read .lf.bin chunked, apply notch, compute per-region bipolar mean,
    anti-alias and decimate, store per-region traces and artifact masks.

    `regions_pairs` is a dict {region_name: (channel_a_arr, channel_b_arr)}
    holding the static bipolar pair channel indices for this probe.
    """
    meta = parse_lf_meta(meta_path)
    n_chan = int(meta.get("nSavedChans", N_NEURAL_CHANNELS + 1))
    fs_raw = float(meta.get("imSampRate", FS_LFP_RAW))
    gain = lfp_gain_uV(meta)

    file_size = Path(bin_path).stat().st_size
    n_samples = file_size // (n_chan * 2)
    print(f"    {probe_label}: {bin_path}", flush=True)
    print(f"      n_samples={n_samples}, n_chan={n_chan}, fs={fs_raw:.2f} Hz, "
          f"gain={gain:.4f} µV/cnt", flush=True)

    start_sample = 0
    end_sample = min(n_samples, int(foraging_duration_s * fs_raw)) if foraging_duration_s else n_samples
    n_truncated = end_sample - start_sample
    print(f"      Foraging truncation: 0..{end_sample} "
          f"({n_truncated} samples = {n_truncated/fs_raw:.1f}s)", flush=True)

    data = np.memmap(bin_path, dtype=np.int16, mode="r",
                       shape=(n_samples, n_chan))

    notch_60 = design_notch_sos(60.0, 30, fs_raw)
    notch_120 = design_notch_sos(120.0, 30, fs_raw)
    lp_aa = design_lowpass_sos(200.0, 8, fs_raw)

    n_ds = n_truncated // DOWNSAMPLE_FACTOR
    region_names = list(regions_pairs.keys())
    regional_ds = {r: np.zeros(n_ds, dtype=np.float32) for r in region_names}

    cursor_ds = 0
    n_chunks = int(np.ceil(n_truncated / CHUNK_SAMPLES_RAW))
    t0 = time.time()
    for ci in range(n_chunks):
        chunk_start = start_sample + ci * CHUNK_SAMPLES_RAW
        chunk_end = min(start_sample + (ci + 1) * CHUNK_SAMPLES_RAW, end_sample)
        n_chunk_raw = chunk_end - chunk_start
        n_chunk_ds = n_chunk_raw // DOWNSAMPLE_FACTOR
        if n_chunk_ds == 0:
            continue

        raw = np.asarray(data[chunk_start:chunk_end, :N_NEURAL_CHANNELS],
                          dtype=np.float32)
        raw *= gain                                # → µV
        raw = sosfiltfilt(notch_60, raw, axis=0)
        raw = sosfiltfilt(notch_120, raw, axis=0)

        for region, (ch_a, ch_b) in regions_pairs.items():
            bip = raw[:, ch_a] - raw[:, ch_b]      # (n_chunk_raw, n_pairs)
            reg = bip.mean(axis=1)                  # (n_chunk_raw,)
            reg_aa = sosfiltfilt(lp_aa, reg)
            regional_ds[region][cursor_ds:cursor_ds + n_chunk_ds] = (
                reg_aa[::DOWNSAMPLE_FACTOR][:n_chunk_ds]
            )

        cursor_ds += n_chunk_ds
        if (ci + 1) % 5 == 0 or ci == n_chunks - 1:
            print(f"      chunk {ci+1}/{n_chunks} ({time.time()-t0:.0f}s)",
                  flush=True)

    artifact_ds = {}
    for region in region_names:
        trace = regional_ds[region][:cursor_ds]
        regional_ds[region] = trace
        med = float(np.median(np.abs(trace)))
        mad = float(np.median(np.abs(trace - np.median(trace))))
        sd_proxy = 1.4826 * mad
        thr = med + ARTIFACT_THRESHOLD_SD * sd_proxy
        mask = np.abs(trace) > thr
        artifact_ds[region] = mask
        print(f"      {region}: threshold {thr:.2f} µV, "
              f"{mask.mean()*100:.2f}% bins flagged", flush=True)

    for region in region_names:
        np.save(out_dir / f"session_{sn}_{region}_regional.npy",
                 regional_ds[region])
        np.save(out_dir / f"session_{sn}_{region}_artifact_mask.npy",
                 artifact_ds[region])

    return regional_ds, artifact_ds


def sanity_check_bipolar(aca, lha, sn, out_dir):
    """Quick Pearson r between ACA and LHA bipolar means. Pre-correction
    single-ended was ~0.95; post-correction should be ≪ 0.5."""
    n = min(len(aca), len(lha))
    r, _ = pearsonr(aca[:n], lha[:n])
    rec = dict(session=sn, regional_ACA_vs_LHA_pearson_r=float(r))
    print(f"  Sanity: regional ACA vs LHA Pearson r = {r:.4f}", flush=True)
    return rec


# ---- Per-HMM-bin spectral cache ----
def compute_per_bin_psd(signal, n_bins, fs):
    samples_per_bin = int(HMM_BIN_S * fs)
    nperseg = min(samples_per_bin, 256)
    freqs = None
    psd_list = []
    for t in range(n_bins):
        s = t * samples_per_bin
        e = s + samples_per_bin
        if e > len(signal):
            psd_list.append(None)
            continue
        f, p = welch(signal[s:e], fs=fs, nperseg=nperseg,
                       noverlap=nperseg // 2, scaling="density")
        if freqs is None:
            freqs = f
        psd_list.append(p)
    psd_arr = np.full((n_bins, len(freqs)), np.nan, dtype=np.float32)
    for t, p in enumerate(psd_list):
        if p is not None:
            psd_arr[t] = p
    return freqs, psd_arr


def compute_per_bin_csd(sig_a, sig_b, n_bins, fs):
    samples_per_bin = int(HMM_BIN_S * fs)
    nperseg = min(samples_per_bin, 256)
    freqs = None
    csd_list, paa_list, pbb_list = [], [], []
    for t in range(n_bins):
        s = t * samples_per_bin
        e = s + samples_per_bin
        if e > len(sig_a):
            csd_list.append(None); paa_list.append(None); pbb_list.append(None)
            continue
        f, pxy = csd(sig_a[s:e], sig_b[s:e], fs=fs, nperseg=nperseg,
                       noverlap=nperseg // 2, scaling="density")
        f, pxx = welch(sig_a[s:e], fs=fs, nperseg=nperseg,
                         noverlap=nperseg // 2, scaling="density")
        f, pyy = welch(sig_b[s:e], fs=fs, nperseg=nperseg,
                         noverlap=nperseg // 2, scaling="density")
        if freqs is None:
            freqs = f
        csd_list.append(pxy); paa_list.append(pxx); pbb_list.append(pyy)
    n_f = len(freqs)
    csd_arr = np.full((n_bins, n_f), np.nan, dtype=np.complex64)
    paa = np.full((n_bins, n_f), np.nan, dtype=np.float32)
    pbb = np.full((n_bins, n_f), np.nan, dtype=np.float32)
    for t in range(n_bins):
        if csd_list[t] is not None:
            csd_arr[t] = csd_list[t]
            paa[t] = paa_list[t]
            pbb[t] = pbb_list[t]
    return freqs, csd_arr, paa, pbb


def band_power_per_bin(freqs, psd_per_bin, band):
    f_lo, f_hi = band
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if not mask.any():
        return np.full(psd_per_bin.shape[0], np.nan)
    return psd_per_bin[:, mask].mean(axis=1)


def coherence_per_bin(csd_arr, paa, pbb, band, freqs):
    f_lo, f_hi = band
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if not mask.any():
        return np.full(csd_arr.shape[0], np.nan)
    coh = (np.abs(csd_arr[:, mask]) ** 2 /
           (paa[:, mask] * pbb[:, mask] + 1e-12))
    return coh.mean(axis=1).real


# ---- Bin labels ----
def label_bins(viterbi, K_pre=K_PRE_HMM):
    n = len(viterbi)
    diff = np.diff(viterbi, prepend=-1, append=-1)
    boundaries = np.flatnonzero(diff != 0)
    starts = boundaries[:-1]; ends = boundaries[1:]
    bin_group = np.array(["excluded"] * n, dtype=object)
    for s, e in zip(starts, ends):
        L = int(e - s)
        if L < 2 * K_pre:
            continue
        bin_group[s:e - K_pre] = "stay"
        bin_group[e - K_pre:e] = "pre_exit"
    return bin_group


def fdr_pass_mask(pvals, q=FDR_ALPHA):
    p = np.asarray(pvals, dtype=np.float64)
    valid = np.isfinite(p)
    sig = np.zeros(p.shape, dtype=bool)
    if not valid.any():
        return sig
    rej, _, _, _ = multipletests(p[valid], alpha=q, method="fdr_bh")
    sig[valid] = rej
    return sig


# ---- Analysis 1: band power per state ----
def analysis_1(sn, freqs_aca, psd_aca, freqs_lha, psd_lha, viterbi,
                  bin_group, artifact_mask_per_bin, out_dir, fig_dir, rng):
    print(f"  M1: band power per state...", flush=True)
    band_names = list(BANDS.keys())
    bp_aca = {b: band_power_per_bin(freqs_aca, psd_aca, BANDS[b]) for b in band_names}
    bp_lha = {b: band_power_per_bin(freqs_lha, psd_lha, BANDS[b]) for b in band_names}

    def obs_test(region, bp_dict, states):
        out = []
        for state in states:
            in_state = (viterbi == state)
            stay_mask = in_state & (bin_group == "stay") & ~artifact_mask_per_bin
            pre_mask = in_state & (bin_group == "pre_exit") & ~artifact_mask_per_bin
            if stay_mask.sum() < MIN_BINS or pre_mask.sum() < MIN_BINS:
                continue
            for band in band_names:
                bp = bp_dict[band]
                fr_s = bp[stay_mask]; fr_p = bp[pre_mask]
                fr_s = fr_s[np.isfinite(fr_s)]; fr_p = fr_p[np.isfinite(fr_p)]
                if len(fr_s) < MIN_BINS or len(fr_p) < MIN_BINS:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        _, p = mannwhitneyu(fr_s, fr_p, alternative="two-sided")
                    except Exception:
                        p = np.nan
                log_ratio = float(np.log2(fr_p.mean() / (fr_s.mean() + 1e-12)))
                out.append(dict(region=region, state=state, band=band,
                                  mean_power_stay=float(fr_s.mean()),
                                  mean_power_pre_exit=float(fr_p.mean()),
                                  log2_fold_change=log_ratio, p=p,
                                  n_stay=int(len(fr_s)), n_pre=int(len(fr_p))))
        return out

    obs_rows = []
    obs_rows.extend(obs_test("ACA", bp_aca, ACA_STATES))
    obs_rows.extend(obs_test("LHA", bp_lha, LHA_STATES))
    df_obs = pd.DataFrame(obs_rows)
    for region in ("ACA", "LHA"):
        sub = df_obs[df_obs.region == region]
        if not len(sub):
            continue
        sig = fdr_pass_mask(sub["p"].values, FDR_ALPHA)
        _, p_adj, _, _ = multipletests(sub["p"].values, alpha=FDR_ALPHA,
                                          method="fdr_bh")
        df_obs.loc[df_obs.region == region, "p_fdr"] = p_adj
        df_obs.loc[df_obs.region == region, "sig_fdr"] = sig
    df_obs.to_csv(out_dir / f"session_{sn}_M1_band_power.csv", index=False)

    print(f"    {N_SHUFFLES} circular-shift shuffles...", flush=True)
    T = len(viterbi)
    sig_counts_obs = {}
    for region in ("ACA", "LHA"):
        sub = df_obs[df_obs.region == region]
        sig_counts_obs[region] = {(int(r["state"]), r["band"]): int(r["sig_fdr"])
                                     for _, r in sub.iterrows()}

    shuf_records = []
    for it in range(N_SHUFFLES):
        offset = int(rng.integers(SHUFFLE_MIN_OFFSET, T - SHUFFLE_MARGIN))
        v_shuf = np.roll(viterbi, offset)
        bg_shuf = label_bins(v_shuf)
        for region, bp_dict, states in [("ACA", bp_aca, ACA_STATES),
                                            ("LHA", bp_lha, LHA_STATES)]:
            pvals = []; keys = []
            for state in states:
                in_state = (v_shuf == state)
                stay_mask = in_state & (bg_shuf == "stay") & ~artifact_mask_per_bin
                pre_mask = in_state & (bg_shuf == "pre_exit") & ~artifact_mask_per_bin
                if stay_mask.sum() < MIN_BINS or pre_mask.sum() < MIN_BINS:
                    continue
                for band in band_names:
                    bp = bp_dict[band]
                    fr_s = bp[stay_mask]; fr_p = bp[pre_mask]
                    fr_s = fr_s[np.isfinite(fr_s)]; fr_p = fr_p[np.isfinite(fr_p)]
                    if len(fr_s) < MIN_BINS or len(fr_p) < MIN_BINS:
                        continue
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        try:
                            _, p = mannwhitneyu(fr_s, fr_p, alternative="two-sided")
                        except Exception:
                            p = np.nan
                    pvals.append(p); keys.append((state, band))
            pvals = np.asarray(pvals)
            sig = fdr_pass_mask(pvals, FDR_ALPHA) if len(pvals) else np.array([], dtype=bool)
            for (state, band), s in zip(keys, sig):
                shuf_records.append(dict(iter=it, region=region,
                                            state=int(state), band=band,
                                            sig=int(s)))
    df_shuf = pd.DataFrame(shuf_records)
    df_shuf.to_csv(out_dir / f"session_{sn}_M1_shuffle.csv", index=False)

    rows_pass = []
    for region in ("ACA", "LHA"):
        states = ACA_STATES if region == "ACA" else LHA_STATES
        for state in states:
            for band in band_names:
                key = (state, band)
                obs_sig = sig_counts_obs[region].get(key, 0)
                shuf = df_shuf[(df_shuf.region == region)
                                  & (df_shuf.state == state)
                                  & (df_shuf.band == band)]
                shuf_rate = float(shuf["sig"].mean()) if len(shuf) else 0.0
                rows_pass.append(dict(region=region, state=state, band=band,
                                        observed_sig=obs_sig,
                                        shuf_pass_rate=shuf_rate,
                                        exceeds_p95=bool(obs_sig == 1
                                                          and shuf_rate < 0.05)))
    df_pass = pd.DataFrame(rows_pass)
    df_pass.to_csv(out_dir / f"session_{sn}_M1_pass.csv", index=False)
    return df_obs, df_pass


# ---- Analysis 2: event-aligned spectrograms ----
def analysis_2(sn, regional_aca, regional_lha, viterbi, base_out, fig_dir):
    print(f"  M2: event-aligned spectrograms...", flush=True)
    pre_s = 3.0; post_s = 3.0
    pre_samples = int(pre_s * FS_LFP_DS)
    post_samples = int(post_s * FS_LFP_DS)
    win_s = 0.5; step_s = 0.1
    win_samples = int(win_s * FS_LFP_DS)
    step_samples = int(step_s * FS_LFP_DS)
    NW = 3; K_tap = 5
    tapers = windows.dpss(win_samples, NW, K_tap)
    f = np.fft.rfftfreq(win_samples, d=1 / FS_LFP_DS)
    f_mask = (f >= 1) & (f <= 100)
    f_use = f[f_mask]

    diff = np.diff(viterbi, prepend=-1, append=-1)
    boundaries = np.flatnonzero(diff != 0)
    starts = boundaries[:-1]; ends = boundaries[1:]
    states_to_run = sorted(set(ACA_STATES) | set(LHA_STATES))

    spec_results = {}
    for region, sig in [("ACA", regional_aca), ("LHA", regional_lha)]:
        for state in states_to_run:
            exit_bins = []
            for s, e in zip(starts, ends):
                if int(viterbi[s]) != state:
                    continue
                L = int(e - s)
                if L < 2 * K_PRE_HMM:
                    continue
                exit_bins.append(e)
            if len(exit_bins) < 3:
                continue
            samples_per_bin = int(HMM_BIN_S * FS_LFP_DS)
            event_samples = [b * samples_per_bin for b in exit_bins]
            event_samples = [s for s in event_samples
                                if s - pre_samples >= 0 and s + post_samples < len(sig)]
            if not event_samples:
                continue
            n_event = len(event_samples)
            n_steps = (pre_samples + post_samples - win_samples) // step_samples + 1
            spec = np.zeros((n_steps, np.sum(f_mask)), dtype=np.float32)
            time_axis = (np.arange(n_steps) * step_samples + win_samples / 2 - pre_samples) / FS_LFP_DS
            for ev in event_samples:
                seg = sig[ev - pre_samples:ev + post_samples]
                for ti in range(n_steps):
                    s0 = ti * step_samples
                    s1 = s0 + win_samples
                    if s1 > len(seg):
                        break
                    w = seg[s0:s1]
                    psds = []
                    for k in range(K_tap):
                        tapered = w * tapers[k]
                        Sk = np.abs(np.fft.rfft(tapered)) ** 2
                        psds.append(Sk)
                    psd_mt = np.mean(psds, axis=0)
                    spec[ti] += psd_mt[f_mask]
            spec /= n_event
            base_mask = (time_axis >= -3.0) & (time_axis <= -1.0)
            base = spec[base_mask].mean(axis=0, keepdims=True) + 1e-12
            spec_norm = np.log2(spec / base)
            spec_results[(region, state)] = dict(time_axis=time_axis,
                                                    freqs=f_use,
                                                    spec_log2_fc=spec_norm,
                                                    n_events=n_event)
            np.savez(base_out / "spectrograms"
                      / f"session_{sn}_state_{state}_{region}.npz",
                      time_axis=time_axis, freqs=f_use,
                      spec_log2_fc=spec_norm, n_events=n_event)
            fig, ax = plt.subplots(figsize=(7, 4))
            vmax = float(np.nanmax(np.abs(spec_norm)))
            im = ax.imshow(spec_norm.T, aspect="auto", origin="lower",
                            extent=[time_axis[0], time_axis[-1],
                                    f_use[0], f_use[-1]],
                            cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.axvline(0, color="black", lw=0.7, ls="--")
            ax.set_xlabel("Time relative to exit (s)")
            ax.set_ylabel("Frequency (Hz)")
            ax.set_title(f"S{sn} {region} state {state}: log2(power / [-3,-1] s) "
                         f"n_events={n_event}")
            plt.colorbar(im, ax=ax, label="log2 fold-change")
            fig.tight_layout()
            fig.savefig(fig_dir / f"session_{sn}_M2_state_{state}_{region}.png", dpi=130)
            plt.close(fig)
    return spec_results


# ---- Analysis 3: ACA-LHA coherence ----
def analysis_3(sn, freqs_csd, csd_arr, paa, pbb, viterbi,
                  bin_group, artifact_mask_per_bin, out_dir, fig_dir, rng):
    print(f"  M3: ACA-LHA coherence per state×band...", flush=True)
    band_names = list(BANDS.keys())
    coh = {b: coherence_per_bin(csd_arr, paa, pbb, BANDS[b], freqs_csd)
           for b in band_names}
    rows = []
    for state in COH_STATES:
        in_state = (viterbi == state)
        stay_mask = in_state & (bin_group == "stay") & ~artifact_mask_per_bin
        pre_mask = in_state & (bin_group == "pre_exit") & ~artifact_mask_per_bin
        if stay_mask.sum() < MIN_BINS or pre_mask.sum() < MIN_BINS:
            continue
        for band in band_names:
            c = coh[band]
            cs = c[stay_mask]; cp = c[pre_mask]
            cs = cs[np.isfinite(cs)]; cp = cp[np.isfinite(cp)]
            if len(cs) < MIN_BINS or len(cp) < MIN_BINS:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    _, p = mannwhitneyu(cs, cp, alternative="two-sided")
                except Exception:
                    p = np.nan
            rows.append(dict(state=state, band=band,
                              mean_coh_stay=float(cs.mean()),
                              mean_coh_pre_exit=float(cp.mean()),
                              delta=float(cp.mean() - cs.mean()), p=p,
                              n_stay=int(len(cs)), n_pre=int(len(cp))))
    df_obs = pd.DataFrame(rows)
    if len(df_obs):
        p_fdr_arr = np.full(len(df_obs), np.nan)
        if df_obs["p"].notna().any():
            _, p_adj, _, _ = multipletests(df_obs["p"].dropna().values,
                                              alpha=FDR_ALPHA, method="fdr_bh")
            p_fdr_arr[df_obs["p"].notna()] = p_adj
        df_obs["p_fdr"] = p_fdr_arr
        df_obs["sig_fdr"] = df_obs["p_fdr"] < FDR_ALPHA
    df_obs.to_csv(out_dir / f"session_{sn}_M3_coherence.csv", index=False)

    print(f"    {N_SHUFFLES} circular-shift shuffles...", flush=True)
    T = len(viterbi)
    shuf_records = []
    for it in range(N_SHUFFLES):
        offset = int(rng.integers(SHUFFLE_MIN_OFFSET, T - SHUFFLE_MARGIN))
        v_shuf = np.roll(viterbi, offset)
        bg_shuf = label_bins(v_shuf)
        for state in COH_STATES:
            in_state = (v_shuf == state)
            stay_mask = in_state & (bg_shuf == "stay") & ~artifact_mask_per_bin
            pre_mask = in_state & (bg_shuf == "pre_exit") & ~artifact_mask_per_bin
            if stay_mask.sum() < MIN_BINS or pre_mask.sum() < MIN_BINS:
                continue
            for band in band_names:
                c = coh[band]
                cs = c[stay_mask]; cp = c[pre_mask]
                cs = cs[np.isfinite(cs)]; cp = cp[np.isfinite(cp)]
                if len(cs) < MIN_BINS or len(cp) < MIN_BINS:
                    continue
                shuf_records.append(dict(iter=it, state=int(state), band=band,
                                            delta=float(cp.mean() - cs.mean())))
    df_shuf = pd.DataFrame(shuf_records)
    df_shuf.to_csv(out_dir / f"session_{sn}_M3_shuffle.csv", index=False)

    pass_rows = []
    for state in COH_STATES:
        for band in band_names:
            obs_row = df_obs[(df_obs.state == state) & (df_obs.band == band)]
            if not len(obs_row):
                continue
            obs_delta = float(obs_row.iloc[0]["delta"])
            shuf = df_shuf[(df_shuf.state == state)
                              & (df_shuf.band == band)]["delta"].values
            if not len(shuf):
                continue
            p95 = float(np.percentile(np.abs(shuf), 95))
            pass_rows.append(dict(state=int(state), band=band,
                                    observed_delta=obs_delta,
                                    shuf_p95_abs=p95,
                                    exceeds_p95=bool(abs(obs_delta) > p95)))
    df_pass = pd.DataFrame(pass_rows)
    df_pass.to_csv(out_dir / f"session_{sn}_M3_pass.csv", index=False)
    return df_obs, df_pass


# ---- Analysis 4: LFP Granger per band ----
def build_lagged_lfp(seg_x, seg_y, p):
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


def granger_F_lfp(seg_x, seg_y, p):
    Xu, Xr, y = build_lagged_lfp(seg_x, seg_y, p)
    if y is None or len(y) <= 2 * p + 2:
        return np.nan, np.nan
    try:
        beta_r, *_ = np.linalg.lstsq(Xr, y, rcond=None)
        beta_u, *_ = np.linalg.lstsq(Xu, y, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    RSS_r = float(np.sum((y - Xr @ beta_r) ** 2))
    RSS_u = float(np.sum((y - Xu @ beta_u) ** 2))
    n = len(y)
    df_diff = p
    df_u = n - (2 * p + 1)
    if df_u <= 0 or RSS_u <= 1e-12:
        return np.nan, np.nan
    F = ((RSS_r - RSS_u) / df_diff) / (RSS_u / df_u)
    pval = 1.0 - f_dist.cdf(F, df_diff, df_u)
    return float(F), float(pval)


def select_lag_BIC_lfp(seg_x, seg_y, lag_range=GRANGER_LAG_RANGE):
    best_p = lag_range[0]; best_bic = np.inf
    for p in lag_range:
        Xu, _, y = build_lagged_lfp(seg_x, seg_y, p)
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


def bandpass_envelope(signal, band, fs):
    f_lo, f_hi = band
    sos = butter(6, [f_lo / (fs / 2), f_hi / (fs / 2)], btype="bandpass",
                  output="sos")
    filt = sosfiltfilt(sos, signal)
    return np.abs(hilbert(filt))


def analysis_4(sn, regional_aca, regional_lha, viterbi,
                  out_dir, fig_dir, rng):
    print(f"  M4: LFP Granger per band on S3...", flush=True)
    diff = np.diff(viterbi, prepend=-1, append=-1)
    boundaries = np.flatnonzero(diff != 0)
    starts = boundaries[:-1]; ends = boundaries[1:]
    samples_per_bin_ds = int(HMM_BIN_S * FS_LFP_DS)
    seg_indices_ds = []
    for s, e in zip(starts, ends):
        if int(viterbi[s]) != S3_STATE:
            continue
        L = e - s
        if L < 2 * K_PRE_HMM:
            continue
        s_ds = s * samples_per_bin_ds
        e_ds = e * samples_per_bin_ds
        if e_ds - s_ds < GRANGER_MIN_S3_DS:
            continue
        seg_end = min(len(regional_aca), e_ds + POST_EXIT_BINS_DS)
        if seg_end - s_ds < GRANGER_MIN_S3_DS + 100:
            continue
        seg_indices_ds.append((s_ds, seg_end))
    if not seg_indices_ds:
        return None
    print(f"    {len(seg_indices_ds)} S3 segments", flush=True)

    rows = []
    shuf_rows = []
    T = len(regional_aca)
    for band_name, band_range in BANDS.items():
        env_aca = bandpass_envelope(regional_aca, band_range, FS_LFP_DS)
        env_lha = bandpass_envelope(regional_lha, band_range, FS_LFP_DS)
        env_aca = (env_aca - env_aca.mean()) / (env_aca.std() + 1e-9)
        env_lha = (env_lha - env_lha.mean()) / (env_lha.std() + 1e-9)

        seg_aca = []; seg_lha = []
        for s, e in seg_indices_ds:
            seg_aca.append(env_aca[s:e])
            seg_lha.append(env_lha[s:e])

        p = select_lag_BIC_lfp(seg_aca, seg_lha)
        F_a, p_a = granger_F_lfp(seg_aca, seg_lha, p)
        F_l, p_l = granger_F_lfp(seg_lha, seg_aca, p)

        shuf_F_a = np.full(N_SHUFFLES, np.nan)
        shuf_F_l = np.full(N_SHUFFLES, np.nan)
        for it in range(N_SHUFFLES):
            offset = int(rng.integers(SHUFFLE_MIN_OFFSET, T - SHUFFLE_MARGIN))
            env_aca_shuf = np.roll(env_aca, offset)
            env_lha_shuf = np.roll(env_lha, offset)
            seg_aca_shuf = [env_aca_shuf[s:e] for s, e in seg_indices_ds]
            seg_lha_shuf = [env_lha_shuf[s:e] for s, e in seg_indices_ds]
            shuf_F_a[it], _ = granger_F_lfp(seg_aca_shuf, seg_lha, p)
            shuf_F_l[it], _ = granger_F_lfp(seg_lha_shuf, seg_aca, p)
            shuf_rows.append(dict(iter=it, band=band_name,
                                    direction="ACA->LHA",
                                    shuffled_F=float(shuf_F_a[it])
                                       if np.isfinite(shuf_F_a[it]) else np.nan))
            shuf_rows.append(dict(iter=it, band=band_name,
                                    direction="LHA->ACA",
                                    shuffled_F=float(shuf_F_l[it])
                                       if np.isfinite(shuf_F_l[it]) else np.nan))

        for direction, F_obs, p_obs, shuf_F in [
            ("ACA->LHA", F_a, p_a, shuf_F_a),
            ("LHA->ACA", F_l, p_l, shuf_F_l),
        ]:
            valid = shuf_F[np.isfinite(shuf_F)]
            p95 = float(np.percentile(valid, 95)) if len(valid) else np.nan
            rows.append(dict(band=band_name, direction=direction,
                              lag_p=p, observed_F=F_obs, F_p_value=p_obs,
                              shuffle_p95=p95,
                              exceeds_p95=bool(np.isfinite(F_obs)
                                                  and np.isfinite(p95)
                                                  and F_obs > p95)))
        print(f"    {band_name}: ACA→LHA F={F_a:.2f} p={p_a:.3g} | "
              f"LHA→ACA F={F_l:.2f} p={p_l:.3g} (lag={p})", flush=True)

    df_obs = pd.DataFrame(rows)
    df_obs.to_csv(out_dir / f"session_{sn}_M4_granger.csv", index=False)
    pd.DataFrame(shuf_rows).to_csv(out_dir / f"session_{sn}_M4_shuffle.csv", index=False)
    return df_obs


# ---- Per-session orchestrator ----
def run_session(sn, lfp_paths, pairs_by_region, base_out, base_fig, rng, cfg,
                 prep_subdir="preprocessed_v2"):
    print(f"\n========== S{sn} ==========", flush=True)
    out_dir = base_out / f"session_{sn}"
    fig_dir = base_fig / f"session_{sn}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    binned = np.load(REPO_ROOT / cfg["out_dirs"]["binned"]
                      / f"session_{sn}.npz", allow_pickle=True)
    trial_time = np.asarray(binned["trial_time"], dtype=np.float64)
    foraging_duration_s = float(trial_time[-1] + HMM_BIN_S)
    print(f"  Foraging duration (from HMM): {foraging_duration_s:.1f}s", flush=True)

    prep_dir = base_out / prep_subdir
    # imec0 → ACA
    aca_regional_path = prep_dir / f"session_{sn}_ACA_regional.npy"
    lha_regional_path = prep_dir / f"session_{sn}_LHA_regional.npy"
    rsp_regional_path = prep_dir / f"session_{sn}_RSP_regional.npy"

    if (aca_regional_path.exists() and lha_regional_path.exists()
            and rsp_regional_path.exists()):
        print(f"  Loading cached preprocessed_v2", flush=True)
        regional_aca = np.load(aca_regional_path)
        artifact_aca = np.load(prep_dir / f"session_{sn}_ACA_artifact_mask.npy")
        regional_lha = np.load(lha_regional_path)
        artifact_lha = np.load(prep_dir / f"session_{sn}_LHA_artifact_mask.npy")
        regional_rsp = np.load(rsp_regional_path)
        artifact_rsp = np.load(prep_dir / f"session_{sn}_RSP_artifact_mask.npy")
    else:
        print(f"  Preprocessing imec0 (ACA)...", flush=True)
        imec0_pairs = {"ACA": pairs_by_region[("imec0", "ACA")]}
        reg0, art0 = preprocess_probe(
            lfp_paths["imec0_bin"], lfp_paths["imec0_meta"],
            "imec0", imec0_pairs, sn, prep_dir, foraging_duration_s,
        )
        regional_aca = reg0["ACA"]; artifact_aca = art0["ACA"]

        print(f"  Preprocessing imec1 (LHA + RSP)...", flush=True)
        imec1_pairs = {
            "LHA": pairs_by_region[("imec1", "LHA")],
            "RSP": pairs_by_region[("imec1", "RSP")],
        }
        reg1, art1 = preprocess_probe(
            lfp_paths["imec1_bin"], lfp_paths["imec1_meta"],
            "imec1", imec1_pairs, sn, prep_dir, foraging_duration_s,
        )
        regional_lha = reg1["LHA"]; artifact_lha = art1["LHA"]
        regional_rsp = reg1["RSP"]; artifact_rsp = art1["RSP"]

    n_common = min(len(regional_aca), len(regional_lha), len(regional_rsp))
    regional_aca = regional_aca[:n_common]
    regional_lha = regional_lha[:n_common]
    regional_rsp = regional_rsp[:n_common]
    artifact_mask = (artifact_aca[:n_common] | artifact_lha[:n_common])
    print(f"  Aligned to {n_common} samples ({n_common/FS_LFP_DS:.1f}s); "
          f"ACA∪LHA artifact fraction {artifact_mask.mean()*100:.2f}%",
          flush=True)

    sanity = sanity_check_bipolar(regional_aca, regional_lha, sn, out_dir)
    pd.DataFrame([sanity]).to_csv(out_dir / f"session_{sn}_sanity_bipolar_r.csv",
                                     index=False)

    post = pd.read_csv(REPO_ROOT / cfg["merge_dirs"]["posteriors"]
                        / f"session_{sn}.csv")
    viterbi = post["viterbi"].values.astype(np.int64)
    n_hmm = min(len(viterbi), n_common // SAMPLES_PER_HMM_BIN_DS)
    viterbi = viterbi[:n_hmm]

    artifact_per_bin = np.zeros(n_hmm, dtype=bool)
    for t in range(n_hmm):
        s_idx = t * SAMPLES_PER_HMM_BIN_DS
        e_idx = s_idx + SAMPLES_PER_HMM_BIN_DS
        if e_idx <= len(artifact_mask):
            artifact_per_bin[t] = bool(artifact_mask[s_idx:e_idx].mean() > 0.1)

    print(f"  Computing per-bin PSDs (ACA, LHA)...", flush=True)
    t_psd = time.time()
    freqs_aca, psd_aca = compute_per_bin_psd(regional_aca, n_hmm, FS_LFP_DS)
    freqs_lha, psd_lha = compute_per_bin_psd(regional_lha, n_hmm, FS_LFP_DS)
    print(f"    Done in {time.time()-t_psd:.0f}s. {len(freqs_aca)} freqs, "
          f"{n_hmm} bins", flush=True)
    np.savez(out_dir / f"session_{sn}_psd_cache.npz",
              freqs=freqs_aca, psd_aca=psd_aca, psd_lha=psd_lha,
              viterbi=viterbi)

    print(f"  Computing per-bin CSD (ACA-LHA)...", flush=True)
    t_csd = time.time()
    freqs_csd, csd_arr, paa, pbb = compute_per_bin_csd(
        regional_aca, regional_lha, n_hmm, FS_LFP_DS,
    )
    print(f"    Done in {time.time()-t_csd:.0f}s", flush=True)
    np.savez(out_dir / f"session_{sn}_csd_cache.npz",
              freqs=freqs_csd, csd_real=csd_arr.real, csd_imag=csd_arr.imag,
              paa=paa, pbb=pbb)

    bin_group = label_bins(viterbi)
    bin_group[artifact_per_bin] = "excluded"

    df_a1_obs, df_a1_pass = analysis_1(
        sn, freqs_aca, psd_aca, freqs_lha, psd_lha,
        viterbi, bin_group, artifact_per_bin, out_dir, fig_dir, rng,
    )
    spec_results = analysis_2(
        sn, regional_aca, regional_lha, viterbi, base_out, fig_dir,
    )
    df_a3_obs, df_a3_pass = analysis_3(
        sn, freqs_csd, csd_arr, paa, pbb,
        viterbi, bin_group, artifact_per_bin, out_dir, fig_dir, rng,
    )
    df_a4 = analysis_4(
        sn, regional_aca, regional_lha, viterbi, out_dir, fig_dir, rng,
    )

    print(f"  Done S{sn} [{time.time()-t0:.0f}s]", flush=True)
    return dict(sn=sn, sanity=sanity,
                 a1_obs=df_a1_obs, a1_pass=df_a1_pass,
                 spec_results=spec_results,
                 a3_obs=df_a3_obs, a3_pass=df_a3_pass,
                 a4_obs=df_a4)


# ---- Cross-session aggregation ----
def cross_session(per_sess, base_out, base_fig):
    sanity_rows = [r["sanity"] for r in per_sess if r is not None and "sanity" in r]
    if sanity_rows:
        pd.DataFrame(sanity_rows).to_csv(
            base_out / "sanity_bipolar_r_cross_session.csv", index=False)

    rows = []
    for r in per_sess:
        if r is None or r.get("a1_pass") is None or not len(r["a1_pass"]):
            continue
        sub = r["a1_pass"].copy(); sub["session"] = r["sn"]
        rows.append(sub)
    if rows:
        cross_a1 = pd.concat(rows, ignore_index=True)
        cross_a1.to_csv(base_out / "M1_cross_session.csv", index=False)
        rep_a1 = []
        for (region, state, band), grp in cross_a1.groupby(["region", "state", "band"]):
            rep_a1.append(dict(region=region, state=int(state), band=band,
                                n_tested=len(grp),
                                n_passing=int(grp["exceeds_p95"].sum())))
        pd.DataFrame(rep_a1).to_csv(base_out / "M1_replication.csv", index=False)

    rows = []
    for r in per_sess:
        if r is None or r.get("a3_pass") is None or not len(r["a3_pass"]):
            continue
        sub = r["a3_pass"].copy(); sub["session"] = r["sn"]
        rows.append(sub)
    if rows:
        cross_a3 = pd.concat(rows, ignore_index=True)
        cross_a3.to_csv(base_out / "M3_cross_session.csv", index=False)
        rep_a3 = []
        for (state, band), grp in cross_a3.groupby(["state", "band"]):
            rep_a3.append(dict(state=int(state), band=band,
                                n_tested=len(grp),
                                n_passing=int(grp["exceeds_p95"].sum())))
        pd.DataFrame(rep_a3).to_csv(base_out / "M3_replication.csv", index=False)

    rows = []
    for r in per_sess:
        if r is None or r.get("a4_obs") is None or not len(r["a4_obs"]):
            continue
        sub = r["a4_obs"].copy(); sub["session"] = r["sn"]
        rows.append(sub)
    if rows:
        cross_a4 = pd.concat(rows, ignore_index=True)
        cross_a4.to_csv(base_out / "M4_cross_session.csv", index=False)
        rep_a4 = []
        for (band, direction), grp in cross_a4.groupby(["band", "direction"]):
            rep_a4.append(dict(band=band, direction=direction,
                                n_tested=len(grp),
                                n_pass_shuf=int(grp["exceeds_p95"].sum()),
                                n_pass_F=int((grp["F_p_value"] < 0.05).sum())))
        pd.DataFrame(rep_a4).to_csv(base_out / "M4_replication.csv", index=False)

        sign_rows = []
        for band in BANDS.keys():
            sessions = sorted(set(cross_a4["session"].unique().astype(int)))
            n_aca_leads = 0; n_total = 0
            for sn in sessions:
                fa = cross_a4[(cross_a4.session == sn) & (cross_a4.band == band)
                                 & (cross_a4.direction == "ACA->LHA")]["observed_F"]
                fl = cross_a4[(cross_a4.session == sn) & (cross_a4.band == band)
                                 & (cross_a4.direction == "LHA->ACA")]["observed_F"]
                if not len(fa) or not len(fl):
                    continue
                if pd.isna(fa.iloc[0]) or pd.isna(fl.iloc[0]):
                    continue
                if float(fa.iloc[0]) > float(fl.iloc[0]):
                    n_aca_leads += 1
                n_total += 1
            try:
                bp = (binomtest(n_aca_leads, n_total, 0.5,
                                  alternative="two-sided").pvalue
                      if n_total else np.nan)
            except Exception:
                bp = np.nan
            sign_rows.append(dict(band=band, n_sessions=n_total,
                                    n_ACA_leads=n_aca_leads,
                                    n_LHA_leads=n_total - n_aca_leads,
                                    binom_p=bp))
        pd.DataFrame(sign_rows).to_csv(base_out / "M4_sign_test.csv", index=False)

    all_specs = {}
    for r in per_sess:
        if r is None:
            continue
        for key, val in (r.get("spec_results") or {}).items():
            all_specs.setdefault(key, []).append(val)
    for (region, state), specs in all_specs.items():
        if len(specs) < 2:
            continue
        shapes = set((s["spec_log2_fc"].shape, s["time_axis"].shape,
                       s["freqs"].shape) for s in specs)
        if len(shapes) != 1:
            continue
        avg = np.mean(np.stack([s["spec_log2_fc"] for s in specs]), axis=0)
        time_axis = specs[0]["time_axis"]
        freqs = specs[0]["freqs"]
        np.savez(base_out / "spectrograms"
                  / f"aggregate_state_{state}_{region}.npz",
                  time_axis=time_axis, freqs=freqs, spec_log2_fc=avg,
                  n_sessions=len(specs))
        base_mask = (time_axis >= -3.0) & (time_axis <= -1.0)
        pre_mask = (time_axis >= -1.4) & (time_axis <= 0.0)
        baseline_vals = np.stack([s["spec_log2_fc"][base_mask].mean(axis=0)
                                     for s in specs])
        pre_vals = np.stack([s["spec_log2_fc"][pre_mask].mean(axis=0)
                                 for s in specs])
        t_stats = []
        p_vals = []
        for fi in range(baseline_vals.shape[1]):
            try:
                t, p = ttest_rel(pre_vals[:, fi], baseline_vals[:, fi])
            except Exception:
                t, p = np.nan, np.nan
            t_stats.append(t); p_vals.append(p)
        fig, ax = plt.subplots(figsize=(7, 4))
        vmax = float(np.nanmax(np.abs(avg)))
        im = ax.imshow(avg.T, aspect="auto", origin="lower",
                        extent=[time_axis[0], time_axis[-1],
                                freqs[0], freqs[-1]],
                        cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.axvline(0, color="black", lw=0.7, ls="--")
        ax.set_xlabel("Time relative to exit (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(f"{region} state {state} aggregate "
                     f"(n={len(specs)} sessions)")
        plt.colorbar(im, ax=ax, label="log2 fold-change")
        fig.tight_layout()
        fig.savefig(base_fig / f"M2_aggregate_state_{state}_{region}.png", dpi=130)
        plt.close(fig)


# ---- Main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-version", choices=("v1", "v2"), default="v1",
                     help="v1: original bipolar_pairs CSVs (LHA y<2500); "
                          "v2: refined bipolar_pairs_v2 CSVs (LHA y<345, RSP y>4680)")
    args = ap.parse_args()
    print(f"=== Pairs version: {args.pairs_version} ===")

    cfg = load_config()
    base_out, base_fig, prep_subdir = out_dirs(args.pairs_version)
    print(f"Output dir: {base_out}")
    print("=== Loading static probe geometry ===")
    _, pairs_by_region = load_static_geometry(args.pairs_version)

    print("\n=== Discovering LFP files ===")
    discover_df = discover_lfp_files()
    discover_df.to_csv(base_out / "discovered_lfp_files.csv", index=False)
    print(f"  Found {len(discover_df)} LFP files")
    if not len(discover_df):
        print("No LFP files found. Aborting.")
        return

    lfp_map = map_session_to_lfp(discover_df)
    print(f"  Mapped to {len(lfp_map)} sessions")
    for sn in SESSIONS:
        if sn not in lfp_map:
            print(f"    WARNING: no LFP map for S{sn}")

    rng = np.random.default_rng(SHUFFLE_SEED)
    per_sess = []
    for sn in SESSIONS:
        if sn not in lfp_map:
            continue
        try:
            r = run_session(sn, lfp_map[sn], pairs_by_region,
                             base_out, base_fig, rng, cfg,
                             prep_subdir=prep_subdir)
            per_sess.append(r)
        except Exception as e:
            import traceback
            print(f"  ERROR S{sn}: {e}")
            traceback.print_exc()

    print("\n========== Cross-session aggregation ==========")
    cross_session(per_sess, base_out, base_fig)
    print("Done.")


if __name__ == "__main__":
    main()
