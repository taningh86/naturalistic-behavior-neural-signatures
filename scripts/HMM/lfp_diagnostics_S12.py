"""LFP diagnostics on session 12 — characterize raw single-ended LFP quality
and common-mode contamination before committing to a re-referencing scheme.

Five diagnostics:
  D1 — Per-channel power spectra (Welch). Detect dead/saturated/line-noisy
       channels.
  D2 — Cross-probe (cross-region) correlation: ACA-LHA, ACA-RSP, LHA-RSP
       per frequency band, raw and amplitude-envelope.
  D3 — Within-probe channel correlation matrix: tests whether locality is
       preserved (high near-diagonal) or common-mode dominates (uniform).
  D4 — Artifact prevalence by envelope thresholding (5 SD).
  D5 — Bipolar referencing preview on 20-channel subset per probe; compare
       cross-region correlations to D2.

S12 (7-11-25 FOR, fasted). Foraging-phase truncation via HMM bin duration.

Pragmatic: all diagnostics done on a 500 Hz downsampled version of the LFP
in memory. D3 correlation matrix is computed on 50 Hz further-downsampled
data to fit in memory.
"""
from pathlib import Path
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import (butter, sosfiltfilt, iirnotch, tf2sos, welch,
                            hilbert, decimate)
from scipy.stats import pearsonr

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "HMM"))
from _utils import load_config


SESSION_NUM = 12
SESSION_DATE = "7_11_25"
SESSION_STATE = "fasted"
N_CHANNELS = 384
FS_LFP_RAW = 2500.0
FS_DS = 500.0                              # Hz, working sample rate
DECIMATE_FACTOR = int(FS_LFP_RAW / FS_DS)  # 5
FS_CORR = 50.0                             # for D3 correlation matrix
DECIMATE_CORR = int(FS_DS / FS_CORR)       # 10

CHUNK_S = 30.0
CHUNK_SAMPLES_RAW = int(CHUNK_S * FS_LFP_RAW)

CATGT_ROOT = Path("H:/Neuropixels Data/Cat_GT_Out")

BANDS = {
    "delta": (1, 4),
    "theta": (4, 12),
    "beta": (15, 30),
    "low_gamma": (30, 60),
    "high_gamma": (60, 100),
}

RSP_Y_THRESHOLD_UM = 2500.0
N_BIPOLAR_PREVIEW = 20

ARTIFACT_THRESHOLD_SD = 5.0

# Default NPX 2.0 single-shank single-band geometry (probe type 2013):
#   192 rows × 2 columns × 15 µm vertical pitch, 32 µm horizontal.
def default_geometry(n_chan=N_CHANNELS):
    """Return (shank, x, y) arrays of length n_chan for default NPX 2.0 ss1b."""
    shank = np.zeros(n_chan, dtype=np.int64)
    rows = np.arange(n_chan) // 2
    col = np.arange(n_chan) % 2
    y = (rows * 15.0).astype(np.float64)
    x = np.where(col == 0, 11.0, 43.0)
    return shank, x, y


def out_dirs():
    base_out = REPO_ROOT / "data" / "HMM" / "neural_alignment" / "lfp_diagnostics" / f"S{SESSION_NUM}"
    base_fig = REPO_ROOT / "figures" / "HMM" / "neural_alignment" / "lfp_diagnostics" / f"S{SESSION_NUM}"
    base_out.mkdir(parents=True, exist_ok=True)
    base_fig.mkdir(parents=True, exist_ok=True)
    return base_out, base_fig


def find_lfp_paths():
    folder = CATGT_ROOT / f"catgt_DOUBLE_PROBE_{SESSION_DATE}_FOR_g0"
    if not folder.exists():
        return None
    out = {}
    for probe, region in [(0, "ACA"), (1, "LHA+RSP")]:
        sub = folder / f"DOUBLE_PROBE_{SESSION_DATE}_FOR_g0_imec{probe}"
        if not sub.exists():
            return None
        lf_bin = list(sub.glob("*.lf.bin"))
        lf_meta = list(sub.glob("*.lf.meta"))
        if not lf_bin or not lf_meta:
            return None
        out[f"imec{probe}"] = dict(
            bin=str(lf_bin[0]), meta=str(lf_meta[0]),
            region=region,
        )
    return out


def parse_meta(meta_path):
    out = {}
    with open(meta_path, "r") as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                key, val = line.split("=", 1)
                out[key] = val
    return out


def lfp_gain_uV(meta):
    rng_max = float(meta.get("imAiRangeMax", 0.62))
    rng_min = float(meta.get("imAiRangeMin", -0.62))
    max_int = int(meta.get("imMaxInt", 2048))
    gain = float(meta.get("imChan0lfGain", meta.get("imChan0apGain", 80.0)))
    return ((rng_max - rng_min) / (2 * max_int * gain)) * 1e6   # µV/count


def design_notch_sos(f0, q, fs):
    b, a = iirnotch(f0, q, fs=fs)
    return tf2sos(b, a)


def design_lowpass_sos(cutoff, order, fs):
    return butter(order, cutoff / (fs / 2), btype="lowpass", output="sos")


def design_highpass_sos(cutoff, order, fs):
    return butter(order, cutoff / (fs / 2), btype="highpass", output="sos")


def design_bandpass_sos(low, high, order, fs):
    return butter(order, [low / (fs / 2), high / (fs / 2)],
                   btype="bandpass", output="sos")


def stream_load_and_downsample(bin_path, meta_path, foraging_duration_s,
                                 verbose=True):
    """Read .lf.bin chunked, apply notch + decimate to 500 Hz, return
    (n_ds, n_chan) float32 array of microvolts."""
    meta = parse_meta(meta_path)
    fs_raw = float(meta.get("imSampRate", FS_LFP_RAW))
    n_saved = int(meta.get("nSavedChans", N_CHANNELS + 1))
    n_chan = N_CHANNELS   # use first 384; the +1 is usually the sync channel
    gain = lfp_gain_uV(meta)
    if verbose:
        print(f"    nSavedChans={n_saved}, fs={fs_raw:.2f} Hz, gain={gain:.4f} µV/cnt",
              flush=True)

    file_size = Path(bin_path).stat().st_size
    n_total = file_size // (n_saved * 2)
    if verbose:
        print(f"    file: {n_total} samples ({n_total/fs_raw:.1f}s total)",
              flush=True)

    end_sample = min(n_total, int(foraging_duration_s * fs_raw))
    n_truncated = end_sample
    if verbose:
        print(f"    truncated to {n_truncated} samples "
              f"({n_truncated/fs_raw:.1f}s foraging phase)", flush=True)

    n_ds = n_truncated // DECIMATE_FACTOR
    out = np.zeros((n_ds, n_chan), dtype=np.float32)

    notch_60 = design_notch_sos(60.0, 30, fs_raw)
    notch_120 = design_notch_sos(120.0, 30, fs_raw)
    lp_aa = design_lowpass_sos(200.0, 8, fs_raw)

    data = np.memmap(bin_path, dtype=np.int16, mode="r",
                       shape=(n_total, n_saved))

    cursor = 0
    n_chunks = int(np.ceil(n_truncated / CHUNK_SAMPLES_RAW))
    t0 = time.time()
    for ci in range(n_chunks):
        s0 = ci * CHUNK_SAMPLES_RAW
        s1 = min((ci + 1) * CHUNK_SAMPLES_RAW, n_truncated)
        n_chunk_raw = s1 - s0
        n_chunk_ds = n_chunk_raw // DECIMATE_FACTOR
        if n_chunk_ds == 0:
            continue
        raw = np.asarray(data[s0:s1, :n_chan], dtype=np.float32)
        raw *= gain   # → µV
        raw = sosfiltfilt(notch_60, raw, axis=0)
        raw = sosfiltfilt(notch_120, raw, axis=0)
        raw = sosfiltfilt(lp_aa, raw, axis=0)
        ds = raw[::DECIMATE_FACTOR][:n_chunk_ds]
        out[cursor:cursor + n_chunk_ds, :] = ds.astype(np.float32)
        cursor += n_chunk_ds
        if verbose and ((ci + 1) % 10 == 0 or ci == n_chunks - 1):
            print(f"    chunk {ci+1}/{n_chunks} ({time.time()-t0:.0f}s)",
                  flush=True)

    return out[:cursor]


# ---- Diagnostic 1: per-channel PSDs ----
def diagnostic_1(lfp_ds, probe_name, region_per_chan, shank, x, y,
                   out_dir, fig_dir):
    print(f"  D1: per-channel PSDs for {probe_name}...", flush=True)
    n_samples, n_chan = lfp_ds.shape
    # Welch per channel; 2s windows at 500 Hz = 1000 samples
    nperseg = int(2.0 * FS_DS)
    psd_list = []
    freqs = None
    for ch in range(n_chan):
        f, p = welch(lfp_ds[:, ch], fs=FS_DS, nperseg=nperseg,
                       noverlap=nperseg // 2, scaling="density")
        if freqs is None:
            freqs = f
        psd_list.append(p)
    psd = np.array(psd_list)   # (n_chan, n_freqs)

    # PSD heatmap, channels ordered by (shank, y)
    order = np.lexsort((y, shank))
    f_mask = (freqs >= 1) & (freqs <= 200)
    f_use = freqs[f_mask]
    psd_log = np.log10(psd[:, f_mask] + 1e-12)
    fig, ax = plt.subplots(figsize=(8, 9))
    vmax = np.percentile(psd_log, 99)
    vmin = np.percentile(psd_log, 5)
    im = ax.imshow(psd_log[order], aspect="auto", cmap="viridis",
                    vmin=vmin, vmax=vmax,
                    extent=[f_use[0], f_use[-1], n_chan, 0])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(f"Channel (sorted by shank, then y)")
    ax.set_title(f"{probe_name}: per-channel log10 PSD")
    plt.colorbar(im, ax=ax, label="log10 power (µV²/Hz)")
    fig.tight_layout()
    fig.savefig(fig_dir / f"D1_psd_heatmap_{probe_name}.png", dpi=130)
    plt.close(fig)

    # Per-channel summary
    total = psd[:, f_mask].sum(axis=1) * (f_use[1] - f_use[0])
    f_lt1 = freqs[(freqs < 1)]
    p60_mask = (freqs >= 58) & (freqs <= 62)
    p120_mask = (freqs >= 118) & (freqs <= 122)
    drift_mask = (freqs <= 1)
    band_p60 = psd[:, p60_mask].mean(axis=1)
    band_p120 = psd[:, p120_mask].mean(axis=1)
    drift_ratio = (psd[:, drift_mask].sum(axis=1) /
                    (psd[:, f_mask].sum(axis=1) + 1e-12))
    dead_thr = np.percentile(total, 1)
    sat_thr = np.percentile(total, 99)
    rows = []
    for ch in range(n_chan):
        rows.append(dict(
            channel=int(ch), shank=int(shank[ch]),
            x=float(x[ch]), y=float(y[ch]),
            region=region_per_chan[ch],
            total_power=float(total[ch]),
            power_60hz=float(band_p60[ch]),
            power_120hz=float(band_p120[ch]),
            drift_ratio=float(drift_ratio[ch]),
            dead_flag=bool(total[ch] < dead_thr * 0.5),
            saturated_flag=bool(total[ch] > sat_thr * 2.0),
        ))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"D1_channel_quality_{probe_name}.csv", index=False)
    return df, freqs, psd


# ---- Diagnostic 2: cross-probe correlation ----
def regional_mean(lfp_ds, region_per_chan, region_name):
    mask = np.array([r == region_name for r in region_per_chan])
    if not mask.any():
        return None
    return lfp_ds[:, mask].mean(axis=1)


def diagnostic_2(aca_ds, lha_rsp_ds, region_per_chan_lha_rsp, fs,
                   out_dir, fig_dir, label_suffix=""):
    print(f"  D2{label_suffix}: cross-region correlations...", flush=True)
    aca_mean = aca_ds.mean(axis=1)
    lha_mean = regional_mean(lha_rsp_ds, region_per_chan_lha_rsp, "LHA")
    rsp_mean = regional_mean(lha_rsp_ds, region_per_chan_lha_rsp, "RSP")

    region_means = {"ACA": aca_mean, "LHA": lha_mean, "RSP": rsp_mean}
    pairs = [("ACA", "LHA"), ("ACA", "RSP"), ("LHA", "RSP")]

    rows = []
    for sig_type in ("raw", "envelope"):
        for band_name, (f_lo, f_hi) in BANDS.items():
            sos = design_bandpass_sos(f_lo, f_hi, 6, fs)
            filtered = {}
            for region, sig in region_means.items():
                if sig is None:
                    continue
                bp = sosfiltfilt(sos, sig)
                if sig_type == "envelope":
                    bp = np.abs(hilbert(bp))
                filtered[region] = bp
            for a, b in pairs:
                if a not in filtered or b not in filtered:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r, p = pearsonr(filtered[a], filtered[b])
                rows.append(dict(region_pair=f"{a}-{b}", band=band_name,
                                  signal_type=sig_type,
                                  pearson_r=float(r),
                                  p=float(p)))
    df = pd.DataFrame(rows)
    fn = f"D2_cross_region_correlation{label_suffix}.csv"
    df.to_csv(out_dir / fn, index=False)

    # Heatmap (raw / envelope side-by-side)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, sig_type in zip(axes, ("raw", "envelope")):
        sub = df[df.signal_type == sig_type]
        if not len(sub):
            ax.axis("off"); continue
        pivot = sub.pivot(index="region_pair", columns="band", values="pearson_r")
        pivot = pivot.reindex(columns=list(BANDS.keys()))
        vmax = max(0.5, float(pivot.abs().to_numpy().max()))
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title(f"{sig_type}")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                             fontsize=8,
                             color="white" if abs(v) > 0.5 * vmax else "black")
        plt.colorbar(im, ax=ax, label="Pearson r")
    fig.suptitle(f"Cross-region correlations{label_suffix}", y=1.0)
    fig.tight_layout()
    fig.savefig(fig_dir / f"D2_correlation_heatmap{label_suffix}.png", dpi=130)
    plt.close(fig)
    return df


# ---- Diagnostic 3: within-probe correlation matrix ----
def diagnostic_3(lfp_ds, probe_name, shank, x, y, fs, out_dir, fig_dir):
    print(f"  D3: within-probe correlation matrix for {probe_name}...", flush=True)
    # 1 Hz high-pass to remove drift
    hp = design_highpass_sos(1.0, 4, fs)
    lp_aa2 = design_lowpass_sos(20.0, 8, fs)
    n_samples, n_chan = lfp_ds.shape
    # Further downsample to FS_CORR for correlation matrix
    X = np.zeros((n_samples // DECIMATE_CORR, n_chan), dtype=np.float32)
    for ch in range(n_chan):
        sig = sosfiltfilt(hp, lfp_ds[:, ch].astype(np.float64))
        sig = sosfiltfilt(lp_aa2, sig)
        X[:, ch] = sig[::DECIMATE_CORR][:X.shape[0]]
    # Correlation matrix
    C = np.corrcoef(X.T)
    np.save(out_dir / f"D3_correlation_matrix_{probe_name}.npy", C)

    # Order by shank then y
    order = np.lexsort((y, shank))
    fig, ax = plt.subplots(figsize=(7, 6.5))
    im = ax.imshow(C[np.ix_(order, order)], aspect="auto", cmap="RdBu_r",
                    vmin=-1, vmax=1)
    ax.set_xlabel("Channel (sorted by shank, then y)")
    ax.set_ylabel("Channel")
    ax.set_title(f"{probe_name}: within-probe correlation matrix "
                 "(after 1 Hz HP)")
    plt.colorbar(im, ax=ax, label="Pearson r")
    fig.tight_layout()
    fig.savefig(fig_dir / f"D3_within_probe_correlation_{probe_name}.png",
                 dpi=130)
    plt.close(fig)

    # Stats
    iu = np.triu_indices_from(C, k=1)
    rs = C[iu]
    # Pairwise physical distance
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    same_shank = (shank[iu[0]] == shank[iu[1]])
    dist = np.sqrt(dx ** 2 + dy ** 2)
    local_mask = (dist <= 30.0) & same_shank
    cross_shank_mask = ~same_shank
    mean_all = float(np.mean(rs))
    mean_local = float(np.mean(rs[local_mask])) if local_mask.any() else np.nan
    mean_cross = float(np.mean(rs[cross_shank_mask])) if cross_shank_mask.any() else np.nan
    return dict(probe=probe_name, mean_corr_all=mean_all,
                 mean_corr_local=mean_local,
                 mean_corr_cross_shank=mean_cross,
                 local_to_cross_ratio=(mean_local / mean_cross
                                        if mean_cross and mean_cross != 0 else np.nan))


# ---- Diagnostic 4: artifact prevalence ----
def diagnostic_4(lfp_ds, probe_name, fs, out_dir, fig_dir):
    print(f"  D4: artifact detection for {probe_name}...", flush=True)
    median_across_chan = np.median(lfp_ds, axis=1)
    envelope = np.abs(median_across_chan)
    # 500ms moving window smoothing
    w = int(0.5 * fs)
    kernel = np.ones(w) / w
    env_smooth = np.convolve(envelope, kernel, mode="same")
    med = float(np.median(env_smooth))
    mad = float(np.median(np.abs(env_smooth - med)))
    sd_proxy = 1.4826 * mad
    thr = med + ARTIFACT_THRESHOLD_SD * sd_proxy
    artifact_mask = env_smooth > thr
    np.save(out_dir / f"D4_artifact_mask_{probe_name}.npy", artifact_mask)
    n_total = len(artifact_mask)
    n_art = int(artifact_mask.sum())

    # Plot envelope timeline with threshold + artifact shading
    fig, ax = plt.subplots(figsize=(13, 3))
    t = np.arange(n_total) / fs
    ax.plot(t, env_smooth, lw=0.4, color="steelblue")
    ax.axhline(thr, color="red", lw=1, ls="--", label=f"thr={thr:.1f} µV")
    art_inds = np.flatnonzero(artifact_mask)
    if len(art_inds):
        # mark in groups
        breaks = np.flatnonzero(np.diff(art_inds) > 1)
        starts = [art_inds[0]] + list(art_inds[breaks + 1])
        ends = list(art_inds[breaks]) + [art_inds[-1]]
        for s, e in zip(starts, ends):
            ax.axvspan(t[s], t[e], color="red", alpha=0.18)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("|median LFP across channels| (µV, smoothed)")
    ax.set_title(f"{probe_name}: artifact mask "
                 f"({n_art}/{n_total} bins = {n_art/n_total*100:.2f}%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / f"D4_artifact_timeline_{probe_name}.png", dpi=130)
    plt.close(fig)
    return dict(probe=probe_name, total_bins=n_total,
                 artifact_bins=n_art,
                 fraction_artifact=float(n_art / n_total),
                 threshold_uV=float(thr))


# ---- Diagnostic 5: bipolar referencing preview ----
def diagnostic_5(aca_ds, lha_rsp_ds, region_per_chan_lha_rsp,
                   shank_aca, y_aca, shank_lha, y_lha,
                   ch_quality_aca, ch_quality_lha,
                   fs, out_dir, fig_dir):
    print(f"  D5: bipolar preview...", flush=True)
    rng = np.random.default_rng(20260511)

    def pick_subset(quality_df, shank_arr, y_arr, n_pick=N_BIPOLAR_PREVIEW):
        good = ~(quality_df["dead_flag"] | quality_df["saturated_flag"])
        good_idx = np.flatnonzero(good.values)
        if len(good_idx) < n_pick * 2:
            return None
        # Sort good channels by shank then y, then pick spaced subset of pairs
        ord_ = np.lexsort((y_arr[good_idx], shank_arr[good_idx]))
        sorted_idx = good_idx[ord_]
        # Pick pairs of adjacent good channels with distance <=30 µm in y, same shank
        pairs = []
        for i in range(len(sorted_idx) - 1):
            a = sorted_idx[i]; b = sorted_idx[i + 1]
            if shank_arr[a] != shank_arr[b]:
                continue
            d = abs(y_arr[a] - y_arr[b])
            if d <= 30.0 and d > 0.0:
                pairs.append((a, b))
            if len(pairs) >= n_pick * 3:
                break
        if len(pairs) < n_pick:
            return None
        sel = rng.choice(len(pairs), size=n_pick, replace=False)
        return [pairs[k] for k in sel]

    aca_pairs = pick_subset(ch_quality_aca, shank_aca, y_aca)
    lha_pairs_all = pick_subset(ch_quality_lha, shank_lha, y_lha,
                                  n_pick=N_BIPOLAR_PREVIEW * 2)
    if aca_pairs is None or lha_pairs_all is None:
        print("    insufficient good channels to build bipolar pairs", flush=True)
        return None, None

    # Split LHA+RSP pairs by region of the lower-y channel
    lha_pairs = []
    rsp_pairs = []
    for a, b in lha_pairs_all:
        # Use the deeper channel's region
        c = a if y_lha[a] < y_lha[b] else b
        if region_per_chan_lha_rsp[c] == "LHA":
            lha_pairs.append((a, b))
        else:
            rsp_pairs.append((a, b))
        if len(lha_pairs) >= N_BIPOLAR_PREVIEW and len(rsp_pairs) >= N_BIPOLAR_PREVIEW:
            break

    print(f"    bipolar pairs: ACA={len(aca_pairs)}, LHA={len(lha_pairs)}, "
          f"RSP={len(rsp_pairs)}", flush=True)

    def bipolar_regional_mean(lfp, pairs):
        if not pairs:
            return None
        bp_signals = np.zeros((lfp.shape[0], len(pairs)), dtype=np.float32)
        for i, (a, b) in enumerate(pairs):
            bp_signals[:, i] = lfp[:, a] - lfp[:, b]
        return bp_signals.mean(axis=1)

    aca_bp = bipolar_regional_mean(aca_ds, aca_pairs)
    lha_bp = bipolar_regional_mean(lha_rsp_ds, lha_pairs)
    rsp_bp = bipolar_regional_mean(lha_rsp_ds, rsp_pairs)

    region_means = {"ACA": aca_bp, "LHA": lha_bp, "RSP": rsp_bp}
    pairs_list = [("ACA", "LHA"), ("ACA", "RSP"), ("LHA", "RSP")]
    rows = []
    for sig_type in ("raw", "envelope"):
        for band_name, (f_lo, f_hi) in BANDS.items():
            sos = design_bandpass_sos(f_lo, f_hi, 6, fs)
            filtered = {}
            for region, sig in region_means.items():
                if sig is None:
                    continue
                bp = sosfiltfilt(sos, sig)
                if sig_type == "envelope":
                    bp = np.abs(hilbert(bp))
                filtered[region] = bp
            for a, b in pairs_list:
                if a not in filtered or b not in filtered:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r, p = pearsonr(filtered[a], filtered[b])
                rows.append(dict(region_pair=f"{a}-{b}", band=band_name,
                                  signal_type=sig_type,
                                  pearson_r=float(r),
                                  p=float(p)))
    df_bp = pd.DataFrame(rows)
    df_bp.to_csv(out_dir / "D5_bipolar_cross_region_correlation.csv", index=False)
    return df_bp, dict(aca_pairs=aca_pairs, lha_pairs=lha_pairs, rsp_pairs=rsp_pairs)


# ---- Main ----
def main():
    cfg = load_config()
    out_dir, fig_dir = out_dirs()
    print(f"=== LFP diagnostics: S{SESSION_NUM} ({SESSION_STATE}, {SESSION_DATE}) ===",
          flush=True)
    paths = find_lfp_paths()
    if paths is None:
        print(f"ERROR: LFP files not found"); return

    # Foraging-phase duration
    binned = np.load(REPO_ROOT / cfg["out_dirs"]["binned"]
                      / f"session_{SESSION_NUM}.npz", allow_pickle=True)
    trial_time = np.asarray(binned["trial_time"], dtype=np.float64)
    foraging_duration_s = float(trial_time[-1] + 0.480)
    print(f"  Foraging duration: {foraging_duration_s:.1f}s", flush=True)

    # ---- Load + downsample both probes ----
    t0 = time.time()
    print(f"\n  Loading + filtering ACA (imec0)...", flush=True)
    aca_ds = stream_load_and_downsample(paths["imec0"]["bin"],
                                          paths["imec0"]["meta"],
                                          foraging_duration_s)
    print(f"  Loading + filtering LHA+RSP (imec1)...", flush=True)
    lha_rsp_ds = stream_load_and_downsample(paths["imec1"]["bin"],
                                              paths["imec1"]["meta"],
                                              foraging_duration_s)
    # Align lengths
    n_min = min(aca_ds.shape[0], lha_rsp_ds.shape[0])
    aca_ds = aca_ds[:n_min]; lha_rsp_ds = lha_rsp_ds[:n_min]
    print(f"\n  Aligned to {n_min} samples ({n_min/FS_DS:.1f}s). "
          f"Total preprocessing time: {time.time()-t0:.0f}s", flush=True)

    # ---- Geometry ----
    shank_aca, x_aca, y_aca = default_geometry()
    shank_lha, x_lha, y_lha = default_geometry()
    region_per_chan_aca = ["ACA"] * N_CHANNELS
    region_per_chan_lha_rsp = ["LHA" if y_lha[ch] < RSP_Y_THRESHOLD_UM else "RSP"
                                for ch in range(N_CHANNELS)]
    n_lha = sum(1 for r in region_per_chan_lha_rsp if r == "LHA")
    n_rsp = N_CHANNELS - n_lha
    print(f"\n  imec1 region split (y threshold {RSP_Y_THRESHOLD_UM:.0f} µm): "
          f"LHA = {n_lha} channels (y < {RSP_Y_THRESHOLD_UM:.0f}), "
          f"RSP = {n_rsp} channels", flush=True)

    # ---- D1 ----
    df_q_aca, _, _ = diagnostic_1(aca_ds, "imec0", region_per_chan_aca,
                                     shank_aca, x_aca, y_aca, out_dir, fig_dir)
    df_q_lha, _, _ = diagnostic_1(lha_rsp_ds, "imec1", region_per_chan_lha_rsp,
                                     shank_lha, x_lha, y_lha, out_dir, fig_dir)

    # ---- D2 ----
    df_d2 = diagnostic_2(aca_ds, lha_rsp_ds, region_per_chan_lha_rsp, FS_DS,
                           out_dir, fig_dir)

    # ---- D3 ----
    stats_aca = diagnostic_3(aca_ds, "imec0", shank_aca, x_aca, y_aca,
                                FS_DS, out_dir, fig_dir)
    stats_lha = diagnostic_3(lha_rsp_ds, "imec1", shank_lha, x_lha, y_lha,
                                FS_DS, out_dir, fig_dir)
    pd.DataFrame([stats_aca, stats_lha]).to_csv(
        out_dir / "D3_correlation_summary.csv", index=False)

    # ---- D4 ----
    art_aca = diagnostic_4(aca_ds, "imec0", FS_DS, out_dir, fig_dir)
    art_lha = diagnostic_4(lha_rsp_ds, "imec1", FS_DS, out_dir, fig_dir)
    pd.DataFrame([art_aca, art_lha]).to_csv(
        out_dir / "D4_artifact_prevalence.csv", index=False)

    # ---- D5 ----
    df_d5, _ = diagnostic_5(aca_ds, lha_rsp_ds, region_per_chan_lha_rsp,
                              shank_aca, y_aca, shank_lha, y_lha,
                              df_q_aca, df_q_lha, FS_DS, out_dir, fig_dir)

    # Comparison plot D2 vs D5
    if df_d5 is not None:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
        for ax, sig_type in zip(axes, ("raw", "envelope")):
            d2 = df_d2[df_d2.signal_type == sig_type]
            d5 = df_d5[df_d5.signal_type == sig_type]
            band_order = list(BANDS.keys())
            x_pos = np.arange(len(band_order))
            width = 0.12
            pairs = ["ACA-LHA", "ACA-RSP", "LHA-RSP"]
            colors_d2 = ["#1f77b4", "#ff7f0e", "#2ca02c"]
            colors_d5 = ["#7799cc", "#ffb380", "#88cc88"]
            offsets = np.linspace(-2.5 * width, 2.5 * width, len(pairs) * 2)
            for i, pair in enumerate(pairs):
                v_d2 = [float(d2[(d2.region_pair == pair) & (d2.band == b)]
                                  ["pearson_r"].values[0])
                         if len(d2[(d2.region_pair == pair) & (d2.band == b)])
                         else np.nan for b in band_order]
                v_d5 = [float(d5[(d5.region_pair == pair) & (d5.band == b)]
                                  ["pearson_r"].values[0])
                         if len(d5[(d5.region_pair == pair) & (d5.band == b)])
                         else np.nan for b in band_order]
                ax.bar(x_pos + offsets[i * 2], v_d2, width=width,
                        color=colors_d2[i], label=f"{pair} (single-ended)" if sig_type == "raw" else None)
                ax.bar(x_pos + offsets[i * 2 + 1], v_d5, width=width,
                        color=colors_d5[i], hatch="//",
                        label=f"{pair} (bipolar)" if sig_type == "raw" else None)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(band_order, rotation=30, ha="right")
            ax.set_ylabel("Pearson r")
            ax.set_title(f"{sig_type}")
            ax.axhline(0, color="black", lw=0.5)
            ax.grid(axis="y", alpha=0.3)
        axes[0].legend(fontsize=8, loc="upper right")
        fig.suptitle("D5: Single-ended (solid) vs bipolar (hatched) cross-region "
                     "correlations", y=1.0)
        fig.tight_layout()
        fig.savefig(fig_dir / "D5_comparison.png", dpi=130)
        plt.close(fig)

    # ---- Print summary + recommendation ----
    print("\n========== SUMMARY ==========")
    for probe_name, df_q in [("imec0", df_q_aca), ("imec1", df_q_lha)]:
        n_dead = int(df_q["dead_flag"].sum())
        n_sat = int(df_q["saturated_flag"].sum())
        n_good = N_CHANNELS - n_dead - n_sat
        print(f"  {probe_name} channels: good={n_good}, dead={n_dead}, "
              f"saturated={n_sat}")
        for region, sub in df_q.groupby("region"):
            n_g = int((~sub["dead_flag"] & ~sub["saturated_flag"]).sum())
            print(f"    {region}: {len(sub)} total ({n_g} good)")

    raw_pairs = df_d2[df_d2.signal_type == "raw"].copy()
    raw_pairs["abs_r"] = raw_pairs["pearson_r"].abs()
    max_raw = raw_pairs.loc[raw_pairs["abs_r"].idxmax()]
    print(f"\n  D2 max raw |r|: {max_raw['region_pair']} / {max_raw['band']} "
          f"= {max_raw['pearson_r']:.3f}")

    print(f"\n  D3 within-probe correlation:")
    for probe_name, stats in [("imec0", stats_aca), ("imec1", stats_lha)]:
        print(f"    {probe_name}: mean_all={stats['mean_corr_all']:.3f}, "
              f"local={stats['mean_corr_local']:.3f}, "
              f"cross_shank={stats['mean_corr_cross_shank']:.3f}, "
              f"ratio={stats['local_to_cross_ratio']:.2f}")

    print(f"\n  D4 artifact prevalence:")
    for art in (art_aca, art_lha):
        print(f"    {art['probe']}: {art['fraction_artifact']*100:.2f}% bins "
              f"(threshold {art['threshold_uV']:.1f} µV)")

    if df_d5 is not None:
        print(f"\n  D5 bipolar improvement:")
        for band in BANDS.keys():
            for pair in ("ACA-LHA", "ACA-RSP", "LHA-RSP"):
                d2_r = float(df_d2[(df_d2.signal_type == "raw") &
                                       (df_d2.region_pair == pair) &
                                       (df_d2.band == band)]
                              ["pearson_r"].values[0])
                d5_r = float(df_d5[(df_d5.signal_type == "raw") &
                                       (df_d5.region_pair == pair) &
                                       (df_d5.band == band)]
                              ["pearson_r"].values[0]) if len(df_d5[(df_d5.signal_type == "raw") &
                                       (df_d5.region_pair == pair) &
                                       (df_d5.band == band)]) else np.nan
                if not np.isnan(d5_r):
                    print(f"    {pair} {band}: r_single={d2_r:+.3f} -> "
                          f"r_bipolar={d5_r:+.3f}  (Δ={d5_r-d2_r:+.3f})")

    # Recommendation
    print(f"\n  ========== RECOMMENDATION ==========")
    if max_raw["abs_r"] > 0.3:
        print(f"    Common-mode contamination is HIGH "
              f"(max raw |r| = {max_raw['abs_r']:.3f} > 0.3).")
        if df_d5 is not None:
            # Compare avg reduction in raw correlations
            d2_raw_mean = float(raw_pairs["abs_r"].mean())
            d5_raw_abs = df_d5[df_d5.signal_type == "raw"]["pearson_r"].abs().mean()
            print(f"    Bipolar reduces mean |r| from {d2_raw_mean:.3f} "
                  f"to {d5_raw_abs:.3f}.")
            if d5_raw_abs < d2_raw_mean * 0.7:
                print(f"    → Bipolar referencing is RECOMMENDED for all "
                      f"cross-region analyses.")
            else:
                print(f"    → Bipolar reduction is modest; consider "
                      f"median-CAR within probe as alternative.")
    elif max_raw["abs_r"] > 0.1:
        print(f"    Common-mode contamination is MODERATE "
              f"(max raw |r| = {max_raw['abs_r']:.3f}, 0.1-0.3). "
              f"Median within-probe referencing may suffice.")
    else:
        print(f"    Common-mode contamination is LOW "
              f"(max raw |r| = {max_raw['abs_r']:.3f} < 0.1). "
              f"No additional referencing needed beyond CatGT.")

    total_dead = int(df_q_aca["dead_flag"].sum() + df_q_lha["dead_flag"].sum())
    total_art_pct = (art_aca["fraction_artifact"] + art_lha["fraction_artifact"]) / 2 * 100
    if total_dead > 20:
        print(f"    ⚠ {total_dead} total dead channels across probes — review "
              f"D1 heatmaps.")
    if total_art_pct > 10:
        print(f"    ⚠ Mean artifact prevalence {total_art_pct:.1f}% > 10% — "
              f"threshold may need adjustment.")

    print(f"\nDone. Outputs in {out_dir} and {fig_dir}", flush=True)


if __name__ == "__main__":
    main()
