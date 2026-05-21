"""19 — Sharp-wave ripple (SWR) detection in ACA, LHA, RSP for foraging sessions.

Per-pair detection of 100-250 Hz transient events on bipolar-referenced LFP at
the native 2500 Hz sampling rate (must keep raw rate — decimating below 500 Hz
would alias the ripple band). The 100-250 Hz envelope itself is slow, so we
downsample the smoothed envelope to 500 Hz to keep per-session memory tractable.

Pipeline
--------
For each session × probe:
  Read raw .lf.bin at 2500 Hz (memmap, chunked)
  Apply gain → notch 60/120 Hz
  Apply static bipolar pairs (per data/HMM/neural_alignment/lfp/bipolar_pairs_*)
  Bandpass 100-250 Hz (order-4 butterworth, zero-phase)
  Hilbert envelope, Gaussian smooth σ=5 ms
  Decimate envelope to 500 Hz
  Z-score envelope within session per pair

For each pair:
  Threshold 4 SD on envelope, ≥30 ms sustained, 50 ms refractory merge
  Reject events within ±100 ms of broadband artifact
  Estimate peak frequency by re-reading raw .lf.bin in a ±50 ms window around
  the event peak and finding the spectral peak in 100-250 Hz

Regional aggregation:
  5 ms bins of per-pair event peaks → regional event when ≥10% of pairs report
  events within ±25 ms window (37/370 ACA, 18/184 LHA, 18/184 RSP)

Validation:
  Per regional event, count spikes in good QC units within ±50 ms; compare to
  shuffled-time controls (Mann-Whitney + per-event p95 flag)

Behavioral context:
  Map each event to the nearest HMM binned npz row → speed, zone, behavior flags

Cross-region co-occurrence:
  Per pair of regions, count events within ±50 ms vs 100 shuffles of one set

Decisions (pragmatic):
  - Do NOT persist per-pair raw bipolar signals at 2500 Hz (~80 GB). Smoothed
    envelope at 500 Hz is kept in memory per session, ~1.3 GB peak per region.
  - Peak frequency and example traces re-read raw .lf.bin on demand from the
    event timestamps.
  - Spike validation uses good-QC units per memory (P0 fr>0.2; P1 fr>0.2,
    amp>43; LHA depth 0-345, RSP depth 4680-5025); session 16 P1 uses KS4
    output (might lack cluster_info.tsv — handled gracefully).
"""
from pathlib import Path
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
from scipy.signal import butter, sosfiltfilt, iirnotch, tf2sos, hilbert, decimate
from scipy.ndimage import gaussian_filter1d
from scipy.stats import mannwhitneyu

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "HMM"))
from _utils import load_config


# ---- Constants ----
CATGT_ROOT = Path("H:/Neuropixels Data/Cat_GT_Out")
GEOMETRY_DIR = REPO_ROOT / "data" / "HMM" / "neural_alignment" / "lfp"
LFP_PREP_DIR = REPO_ROOT / "data" / "HMM" / "neural_alignment" / "lfp_spectral" / "preprocessed_v2"

SESSIONS = [4, 6, 8, 12, 14, 16]
SESSION_STATE = {4: "fed", 6: "fed", 8: "fed",
                  12: "fasted", 14: "fasted", 16: "fasted"}
SESSION_DATE_MAP = {
    4:  ("6_17_25", "fed"),
    6:  ("6_24_25", "fed"),
    8:  ("6_30_25", "fed"),
    12: ("7_11_25", "fasted"),
    14: ("7_17_25", "fasted"),
    16: ("7_25_25", "fasted"),
}

# LFP rates
FS_LFP_RAW = 2500.0
FS_ENV_DS = 500.0
ENV_DECIMATION = int(FS_LFP_RAW / FS_ENV_DS)            # 5

# Ripple band
RIPPLE_BAND = (100.0, 250.0)
ENV_SMOOTH_SIGMA_MS = 5.0
ENV_SMOOTH_SIGMA_SAMPLES_RAW = int(ENV_SMOOTH_SIGMA_MS * 1e-3 * FS_LFP_RAW)  # 12 at 2500 Hz

# Detection thresholds
DETECT_Z = 4.0
MIN_DURATION_MS = 30.0
REFRACTORY_MS = 50.0
ARTIFACT_REJECT_MS = 100.0          # ±100 ms from artifact flag → drop event
ARTIFACT_SD = 5.0                   # broadband artifact threshold

# Regional aggregation
REGIONAL_BIN_MS = 5.0
REGIONAL_PAIR_THRESHOLD_FRAC = 0.10
REGIONAL_COINCIDENCE_WINDOW_MS = 25.0   # ±25 ms

# Validation
SPIKE_WIN_MS = 50.0                  # ±50 ms around event peak
N_CONTROLS = 1000

# Cross-region
COOC_WINDOW_MS = 50.0                # ±50 ms between regions
N_COOC_SHUFFLES = 100

# Processing
CHUNK_S = 30.0
CHUNK_SAMPLES_RAW = int(CHUNK_S * FS_LFP_RAW)            # 75000
N_NEURAL_CHANNELS = 384
FILTER_PAD_SAMPLES = int(0.2 * FS_LFP_RAW)               # 500 samples (200 ms)

SEED = 20260512


# QC thresholds (from memory)
P0_MIN_FR = 0.2
P1_MIN_FR = 0.2
P1_MIN_AMP = 43
LHA_DEPTH_MIN = 0
LHA_DEPTH_MAX = 345
RSP_DEPTH_MIN = 4680
RSP_DEPTH_MAX = 5025


# ---- Paths ----
def out_dirs():
    base_out = REPO_ROOT / "data" / "HMM" / "neural_alignment" / "swr_detection"
    base_fig = REPO_ROOT / "figures" / "HMM" / "neural_alignment" / "swr_detection"
    (base_out / "example_traces").mkdir(parents=True, exist_ok=True)
    (base_fig / "example_traces").mkdir(parents=True, exist_ok=True)
    return base_out, base_fig


# ---- Geometry ----
def load_geometry():
    geom = pd.read_csv(GEOMETRY_DIR / "probe_geometry.csv")
    p0 = pd.read_csv(GEOMETRY_DIR / "bipolar_pairs_imec0.csv")
    p1 = pd.read_csv(GEOMETRY_DIR / "bipolar_pairs_imec1.csv")
    pairs = {
        "ACA": (p0, "imec0"),
        "LHA": (p1[p1.region == "LHA"].reset_index(drop=True), "imec1"),
        "RSP": (p1[p1.region == "RSP"].reset_index(drop=True), "imec1"),
    }
    return geom, pairs


# ---- LFP file discovery (reuse from 17) ----
def discover_lfp_files():
    rows = []
    cutoff = datetime(2025, 8, 29)
    if not CATGT_ROOT.exists():
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
            rows.append(dict(date=date_str, probe=probe,
                              lf_bin=str(lf_bin[0]),
                              lf_meta=str(lf_meta[0])))
    return pd.DataFrame(rows)


def map_session_to_lfp(disc):
    out = {}
    for sn in SESSIONS:
        date_str, _ = SESSION_DATE_MAP[sn]
        rows = disc[disc.date == date_str]
        if not len(rows):
            month, day, year = date_str.split("_")
            rows = disc[disc.date == f"{int(month)}_{int(day)}_{year}"]
        if not len(rows):
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


def design_bandpass_sos(lo, hi, order, fs):
    return butter(order, [lo / (fs / 2), hi / (fs / 2)],
                   btype="bandpass", output="sos")


# ---- Per-session preprocessing & envelope ----
def preprocess_envelope(bin_path, meta_path, pair_lookup, sn, foraging_duration_s,
                          probe_label):
    """Stream raw .lf.bin, apply notch + bipolar + bandpass + Hilbert + smooth +
    decimate. Returns dict region -> (n_samples_ds, n_pairs) float32 envelope at
    500 Hz, plus a regional artifact mask at 2500 Hz (envelope > 5 SD on broadband
    bipolar mean).

    `pair_lookup` is a dict region -> (channel_a_arr, channel_b_arr) for the
    pairs on this probe."""
    meta = parse_lf_meta(meta_path)
    n_chan = int(meta.get("nSavedChans", N_NEURAL_CHANNELS + 1))
    fs_raw = float(meta.get("imSampRate", FS_LFP_RAW))
    gain = lfp_gain_uV(meta)
    n_total = Path(bin_path).stat().st_size // (n_chan * 2)
    end_sample = min(n_total, int(foraging_duration_s * fs_raw))
    n_keep = end_sample
    print(f"    {probe_label}: n_samples={n_total}, fs={fs_raw:.2f} Hz, "
          f"gain={gain:.4f} µV/cnt, foraging={n_keep/fs_raw:.1f}s", flush=True)

    data = np.memmap(bin_path, dtype=np.int16, mode="r",
                       shape=(n_total, n_chan))

    notch_60 = design_notch_sos(60.0, 30, fs_raw)
    notch_120 = design_notch_sos(120.0, 30, fs_raw)
    bp_ripple = design_bandpass_sos(*RIPPLE_BAND, 4, fs_raw)

    # Output envelope per region at 500 Hz
    n_env = n_keep // ENV_DECIMATION
    region_env = {}
    for region in pair_lookup:
        n_pairs = len(pair_lookup[region][0])
        region_env[region] = np.zeros((n_env, n_pairs), dtype=np.float32)

    # Broadband regional mean for artifact detection (one per probe, not per region)
    bb_mean = np.zeros(n_keep, dtype=np.float32)

    cursor_env = 0
    n_chunks = int(np.ceil(n_keep / CHUNK_SAMPLES_RAW))
    t0 = time.time()
    for ci in range(n_chunks):
        s = ci * CHUNK_SAMPLES_RAW
        e = min((ci + 1) * CHUNK_SAMPLES_RAW, n_keep)
        n_chunk = e - s
        # Pad chunk on both sides for zero-phase filter; we will trim after
        pad_s = max(0, s - FILTER_PAD_SAMPLES)
        pad_e = min(n_total, e + FILTER_PAD_SAMPLES)
        raw = np.asarray(data[pad_s:pad_e, :N_NEURAL_CHANNELS], dtype=np.float32)
        raw *= gain
        raw = sosfiltfilt(notch_60, raw, axis=0)
        raw = sosfiltfilt(notch_120, raw, axis=0)

        # Broadband mean (for artifact mask)
        bb_chunk = raw.mean(axis=1)
        s_off = s - pad_s
        bb_mean[s:e] = bb_chunk[s_off:s_off + n_chunk]

        # Per region: bipolar, bandpass, Hilbert envelope, smooth, decimate
        for region, (ch_a, ch_b) in pair_lookup.items():
            bip = raw[:, ch_a] - raw[:, ch_b]              # (n_chunk_padded, n_pairs)
            bp = sosfiltfilt(bp_ripple, bip, axis=0)
            env = np.abs(hilbert(bp, axis=0))
            # Trim padding
            env = env[s_off:s_off + n_chunk]
            # Smooth (Gaussian, σ in samples at 2500 Hz)
            env = gaussian_filter1d(env, sigma=ENV_SMOOTH_SIGMA_SAMPLES_RAW,
                                       axis=0)
            # Decimate to 500 Hz via simple slicing
            env_ds = env[::ENV_DECIMATION]
            take = min(env_ds.shape[0], n_env - cursor_env)
            region_env[region][cursor_env:cursor_env + take] = env_ds[:take]

        cursor_env += n_chunk // ENV_DECIMATION
        if (ci + 1) % 5 == 0 or ci == n_chunks - 1:
            print(f"      chunk {ci+1}/{n_chunks} ({time.time()-t0:.0f}s)",
                  flush=True)

    # Build artifact mask at 2500 Hz
    bb_med = float(np.median(np.abs(bb_mean)))
    bb_mad = float(np.median(np.abs(bb_mean - np.median(bb_mean))))
    bb_sd = 1.4826 * bb_mad
    thr = bb_med + ARTIFACT_SD * bb_sd
    artifact_mask_raw = np.abs(bb_mean) > thr
    print(f"      broadband artifact threshold {thr:.1f} µV, "
          f"{artifact_mask_raw.mean()*100:.2f}% samples", flush=True)

    return region_env, artifact_mask_raw, n_keep, fs_raw, gain


# ---- Event detection per pair ----
def detect_events_per_pair(env_per_pair, fs_env, artifact_mask_env,
                              min_duration_ms, refractory_ms, threshold_z):
    """env_per_pair: (T, n_pairs) at fs_env. Returns list of per-pair event lists.
    Each event is dict(start_idx, end_idx, peak_idx, peak_z, duration_ms)."""
    n_t, n_pairs = env_per_pair.shape
    min_dur_samples = int(min_duration_ms * 1e-3 * fs_env)
    refr_samples = int(refractory_ms * 1e-3 * fs_env)

    # z-score per pair (median/MAD for robustness)
    med = np.median(env_per_pair, axis=0)
    mad = np.median(np.abs(env_per_pair - med), axis=0)
    sd_proxy = 1.4826 * mad + 1e-12
    z = (env_per_pair - med) / sd_proxy

    all_events = []
    for p in range(n_pairs):
        zp = z[:, p].copy()
        # mark artifact samples as below threshold (will not trigger event)
        zp[artifact_mask_env] = -np.inf
        above = zp > threshold_z
        if not above.any():
            all_events.append([])
            continue
        # find runs
        diff = np.diff(above.astype(np.int8), prepend=0, append=0)
        starts = np.flatnonzero(diff == 1)
        ends = np.flatnonzero(diff == -1)
        events = []
        for s, e in zip(starts, ends):
            if e - s < min_dur_samples:
                continue
            peak_off = int(np.argmax(zp[s:e]))
            peak_idx = s + peak_off
            peak_z = float(zp[peak_idx])
            duration_ms = (e - s) * 1000.0 / fs_env
            events.append(dict(start_idx=int(s), end_idx=int(e),
                                peak_idx=int(peak_idx),
                                peak_z=peak_z,
                                duration_ms=float(duration_ms)))
        # apply refractory: keep highest-peak event in each refractory cluster
        events.sort(key=lambda d: d["peak_idx"])
        merged = []
        for ev in events:
            if merged and ev["peak_idx"] - merged[-1]["peak_idx"] < refr_samples:
                if ev["peak_z"] > merged[-1]["peak_z"]:
                    merged[-1] = ev
                continue
            merged.append(ev)
        all_events.append(merged)
    return all_events


def reject_near_artifact(events_per_pair, artifact_mask_env, fs_env,
                          reject_ms):
    """Drop events whose peak is within reject_ms of any artifact bin."""
    reject_samples = int(reject_ms * 1e-3 * fs_env)
    n_t = len(artifact_mask_env)
    art_idx = np.flatnonzero(artifact_mask_env)
    if not len(art_idx):
        return events_per_pair
    out = []
    for evs in events_per_pair:
        keep = []
        for ev in evs:
            # Check if any artifact is within ±reject_samples of ev peak
            lo = max(0, ev["peak_idx"] - reject_samples)
            hi = min(n_t, ev["peak_idx"] + reject_samples + 1)
            if artifact_mask_env[lo:hi].any():
                continue
            keep.append(ev)
        out.append(keep)
    return out


def peak_frequency_from_raw(bin_path, meta_path, peak_time_s, fs_raw, n_chan,
                              ch_a, ch_b, gain, notch_60, notch_120, bp_ripple,
                              window_ms=50.0):
    """Re-read raw .lf.bin in a window around peak_time_s, apply notch +
    bipolar + bandpass, compute periodogram, return peak frequency in 100-250 Hz."""
    win_samples = int(window_ms * 1e-3 * fs_raw)
    center = int(peak_time_s * fs_raw)
    pad_s = max(0, center - win_samples - FILTER_PAD_SAMPLES)
    pad_e = min(int(Path(bin_path).stat().st_size // (n_chan * 2)),
                  center + win_samples + FILTER_PAD_SAMPLES)
    data = np.memmap(bin_path, dtype=np.int16, mode="r",
                       shape=(int(Path(bin_path).stat().st_size // (n_chan * 2)), n_chan))
    raw = np.asarray(data[pad_s:pad_e, [ch_a, ch_b]], dtype=np.float32) * gain
    raw = sosfiltfilt(notch_60, raw, axis=0)
    raw = sosfiltfilt(notch_120, raw, axis=0)
    bip = raw[:, 0] - raw[:, 1]
    filt = sosfiltfilt(bp_ripple, bip)
    s_off = center - pad_s - win_samples
    seg = filt[s_off:s_off + 2 * win_samples]
    if len(seg) < 32:
        return np.nan
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
    freqs = np.fft.rfftfreq(len(seg), d=1 / fs_raw)
    mask = (freqs >= RIPPLE_BAND[0]) & (freqs <= RIPPLE_BAND[1])
    if not mask.any():
        return np.nan
    return float(freqs[mask][np.argmax(spec[mask])])


# ---- Regional aggregation ----
def aggregate_regional_events(events_per_pair, n_pairs, fs_env, n_t):
    """Identify regional events: peaks in 5 ms binned per-pair event density
    where ≥ threshold * n_pairs report events within ±25 ms."""
    bin_samples = int(REGIONAL_BIN_MS * 1e-3 * fs_env)  # at 500 Hz, 5 ms = 2.5 → 2 samples
    bin_samples = max(1, bin_samples)
    n_bins = n_t // bin_samples
    coinc_window_bins = int(REGIONAL_COINCIDENCE_WINDOW_MS * 1e-3 * fs_env / bin_samples)
    coinc_window_bins = max(1, coinc_window_bins)
    threshold_count = int(np.ceil(REGIONAL_PAIR_THRESHOLD_FRAC * n_pairs))

    # Density of peaks per bin (across pairs)
    density = np.zeros(n_bins, dtype=np.int32)
    # Also keep per-pair contributing list per bin
    contrib = [list() for _ in range(n_bins)]
    for pi, evs in enumerate(events_per_pair):
        for ev in evs:
            b = ev["peak_idx"] // bin_samples
            if b >= n_bins:
                continue
            density[b] += 1
            contrib[b].append((pi, ev))

    # Sliding window count: how many UNIQUE pairs report events in ±coinc_window_bins
    # We use convolution on the per-pair binary "any event in bin" vector.
    # Per-pair binary: shape (n_pairs, n_bins) — too big maybe. Use density approximation:
    # density convolution gives sum of event counts, not unique pairs. For most ripples
    # one pair contributes at most 1 event in a 50 ms window, so density ≈ unique pairs.
    kernel = np.ones(2 * coinc_window_bins + 1, dtype=np.int32)
    smoothed = np.convolve(density, kernel, mode="same")

    # Find local maxima above threshold
    in_event = smoothed >= threshold_count
    diff = np.diff(in_event.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    regional_events = []
    for s, e in zip(starts, ends):
        # peak bin (where smoothed is highest)
        peak_bin = s + int(np.argmax(smoothed[s:e]))
        peak_t = peak_bin * bin_samples / fs_env
        # Collect contributing pairs within ±coinc_window_bins of peak_bin
        contributing = []
        for b in range(max(0, peak_bin - coinc_window_bins),
                       min(n_bins, peak_bin + coinc_window_bins + 1)):
            contributing.extend(contrib[b])
        if not contributing:
            continue
        n_active = len(set(pi for pi, _ in contributing))
        peak_zs = [ev["peak_z"] for _, ev in contributing]
        durs = [ev["duration_ms"] for _, ev in contributing]
        regional_events.append(dict(
            peak_time_s=float(peak_t),
            peak_bin=int(peak_bin),
            n_pairs_active=int(n_active),
            mean_peak_z=float(np.mean(peak_zs)),
            mean_duration_ms=float(np.mean(durs)),
            contributing_pairs=contributing,
        ))
    # apply refractory across regional events (50 ms)
    if regional_events:
        regional_events.sort(key=lambda d: d["peak_time_s"])
        merged = []
        refr_s = REFRACTORY_MS * 1e-3
        for ev in regional_events:
            if merged and ev["peak_time_s"] - merged[-1]["peak_time_s"] < refr_s:
                if ev["n_pairs_active"] > merged[-1]["n_pairs_active"]:
                    merged[-1] = ev
                continue
            merged.append(ev)
        regional_events = merged
    return regional_events, density


# ---- Spike validation ----
def load_good_units(sorted_path, region):
    sorted_path = Path(sorted_path)
    ci = sorted_path / "cluster_info.tsv"
    if not ci.exists():
        return np.array([], dtype=int)
    try:
        df = pd.read_csv(ci, sep="\t")
    except Exception:
        return np.array([], dtype=int)
    label_col = ("group" if ("group" in df.columns
                              and df["group"].eq("good").any())
                 else "KSLabel")
    if label_col not in df.columns:
        return np.array([], dtype=int)
    if region == "ACA":
        m = (df[label_col] == "good") & (df.get("fr", 0) > P0_MIN_FR)
    elif region == "LHA":
        m = ((df[label_col] == "good")
              & (df.get("fr", 0) > P1_MIN_FR)
              & (df.get("amp", 0) > P1_MIN_AMP)
              & (df.get("depth", -1) >= LHA_DEPTH_MIN)
              & (df.get("depth", -1) <= LHA_DEPTH_MAX))
    elif region == "RSP":
        m = ((df[label_col] == "good")
              & (df.get("fr", 0) > P1_MIN_FR)
              & (df.get("amp", 0) > P1_MIN_AMP)
              & (df.get("depth", -1) >= RSP_DEPTH_MIN)
              & (df.get("depth", -1) <= RSP_DEPTH_MAX))
    else:
        return np.array([], dtype=int)
    return df.loc[m, "cluster_id"].values.astype(int)


def load_spike_times(sorted_path, cluster_ids, fs_ap=30000.0):
    """Return concatenated spike times (s) for the given clusters."""
    sorted_path = Path(sorted_path)
    st_path = sorted_path / "spike_times.npy"
    sc_path = sorted_path / "spike_clusters.npy"
    if not st_path.exists() or not sc_path.exists():
        return np.array([])
    st = np.load(st_path).astype(np.int64).ravel()
    sc = np.load(sc_path).astype(np.int64).ravel()
    keep = np.isin(sc, cluster_ids)
    return st[keep] / fs_ap


def validate_events_with_spikes(regional_events, sorted_path, region,
                                  foraging_duration_s, rng):
    """For each regional event, count spikes in good units within ±SPIKE_WIN_MS
    of peak. Generate N_CONTROLS random control times (excluding ±200 ms of any
    detected event). Per-event validation: spike_count > control p95."""
    cluster_ids = load_good_units(sorted_path, region)
    if not len(cluster_ids):
        return pd.DataFrame(), dict(p_mw=np.nan, n_units=0)
    spikes = load_spike_times(sorted_path, cluster_ids)
    if not len(spikes):
        return pd.DataFrame(), dict(p_mw=np.nan, n_units=len(cluster_ids))
    win_s = SPIKE_WIN_MS * 1e-3

    # Per-event spike counts
    event_times = np.array([ev["peak_time_s"] for ev in regional_events])
    event_counts = []
    for t in event_times:
        c = np.sum((spikes >= t - win_s) & (spikes < t + win_s))
        event_counts.append(int(c))

    # Control times: random, excluding ±200 ms of any detected event
    exclude_pad_s = 0.2
    control_counts = []
    rng_local = np.random.default_rng(rng.integers(1, 1_000_000_000))
    n_tried = 0; n_kept = 0
    while n_kept < N_CONTROLS and n_tried < 10 * N_CONTROLS:
        n_tried += 1
        t = float(rng_local.uniform(exclude_pad_s, foraging_duration_s - exclude_pad_s))
        # Reject if within exclude_pad_s of any event
        if len(event_times):
            if (np.min(np.abs(event_times - t))) < exclude_pad_s:
                continue
        c = np.sum((spikes >= t - win_s) & (spikes < t + win_s))
        control_counts.append(int(c))
        n_kept += 1
    control_counts = np.array(control_counts)
    if len(control_counts):
        p95 = float(np.percentile(control_counts, 95))
    else:
        p95 = np.nan

    rows = []
    for ev, c in zip(regional_events, event_counts):
        rows.append(dict(
            peak_time_s=ev["peak_time_s"],
            event_locked_spikes=int(c),
            control_p95=p95,
            validated_flag=bool(c > p95),
        ))
    df = pd.DataFrame(rows)
    if len(event_counts) and len(control_counts):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                U, p_mw = mannwhitneyu(event_counts, control_counts,
                                          alternative="greater")
            except Exception:
                p_mw = np.nan
        n_pass = int(df["validated_flag"].sum())
    else:
        p_mw = np.nan; n_pass = 0
    summary = dict(p_mw=float(p_mw) if np.isfinite(p_mw) else np.nan,
                    n_units=int(len(cluster_ids)),
                    n_events=int(len(event_counts)),
                    n_validated=n_pass,
                    median_event_spikes=float(np.median(event_counts))
                       if event_counts else np.nan,
                    median_control_spikes=float(np.median(control_counts))
                       if len(control_counts) else np.nan)
    return df, summary


# ---- Behavioral characterization ----
def behavioral_context(regional_events, binned_npz_path):
    """Look up each event's behavioral context from the binned npz (480 ms bins)."""
    if not Path(binned_npz_path).exists():
        return pd.DataFrame()
    d = np.load(binned_npz_path, allow_pickle=True)
    trial_time = np.asarray(d["trial_time"], dtype=np.float64)
    speed = np.asarray(d.get("speed", np.zeros_like(trial_time)))
    zone = np.asarray(d.get("zone_int", np.full_like(trial_time, -1, dtype=int)))
    # zone label map
    ZONE_NAMES = {0: "home", 1: "transition", 2: "pot",
                   3: "pot_zone", 4: "arena", 5: "other"}
    # Behavior flags
    dig = np.asarray(d.get("digging_sand", np.zeros_like(trial_time)))
    feed = np.asarray(d.get("feeding", np.zeros_like(trial_time)))
    rear = np.asarray(d.get("rearing", np.zeros_like(trial_time)))
    explore = np.asarray(d.get("exploration_at_transition", np.zeros_like(trial_time)))
    contempl = np.asarray(d.get("contemplation_at_transition",
                                  np.zeros_like(trial_time)))

    rows = []
    for ev in regional_events:
        t = ev["peak_time_s"]
        i = int(np.clip(np.searchsorted(trial_time, t) - 1, 0, len(trial_time) - 1))
        sp = float(speed[i])
        if sp < 1.0:
            loc = "stationary"
        elif sp < 5.0:
            loc = "slow_locomotion"
        else:
            loc = "fast_locomotion"
        rows.append(dict(peak_time_s=t,
                          bin_idx=i,
                          speed=sp,
                          locomotion=loc,
                          zone=ZONE_NAMES.get(int(zone[i]), str(zone[i])),
                          dig=bool(dig[i]),
                          feed=bool(feed[i]),
                          rear=bool(rear[i]),
                          explore_at_transition=bool(explore[i]),
                          contemplation_at_transition=bool(contempl[i])))
    return pd.DataFrame(rows)


# ---- Cross-region co-occurrence ----
def cross_region_cooccurrence(events_by_region, foraging_duration_s, rng):
    rows = []
    PAIR_DEFS = [("ACA", "LHA"), ("ACA", "RSP"), ("LHA", "RSP")]
    win_s = COOC_WINDOW_MS * 1e-3
    for A, B in PAIR_DEFS:
        ta = np.array([e["peak_time_s"] for e in events_by_region.get(A, [])])
        tb = np.array([e["peak_time_s"] for e in events_by_region.get(B, [])])
        if not len(ta) or not len(tb):
            rows.append(dict(pair=f"{A}-{B}",
                              n_A=len(ta), n_B=len(tb),
                              obs_cooc_rate_A=np.nan,
                              obs_cooc_count=0,
                              shuf_p95=np.nan,
                              exceeds_p95=False))
            continue
        # Observed: count A events with any B within ±win_s
        cooc = 0
        for tA in ta:
            if np.any(np.abs(tb - tA) <= win_s):
                cooc += 1
        obs_rate = cooc / len(ta)
        # Shuffles: randomize B times in [0, T]
        shuf_rates = []
        for _ in range(N_COOC_SHUFFLES):
            tb_shuf = rng.uniform(0, foraging_duration_s, size=len(tb))
            c = 0
            for tA in ta:
                if np.any(np.abs(tb_shuf - tA) <= win_s):
                    c += 1
            shuf_rates.append(c / len(ta))
        shuf_rates = np.array(shuf_rates)
        p95 = float(np.percentile(shuf_rates, 95))
        rows.append(dict(pair=f"{A}-{B}",
                          n_A=int(len(ta)), n_B=int(len(tb)),
                          obs_cooc_rate_A=float(obs_rate),
                          obs_cooc_count=int(cooc),
                          shuf_mean=float(shuf_rates.mean()),
                          shuf_p95=p95,
                          exceeds_p95=bool(obs_rate > p95)))
    return pd.DataFrame(rows)


# ---- Example trace plot ----
def plot_example_trace(bin_path, meta_path, ch_a, ch_b, peak_time_s,
                        n_chan, gain, notch_60, notch_120, bp_ripple,
                        fs_raw, out_path, title=""):
    win_s = 0.15
    win_samples = int(win_s * fs_raw)
    n_total = int(Path(bin_path).stat().st_size // (n_chan * 2))
    center = int(peak_time_s * fs_raw)
    pad_s = max(0, center - win_samples - FILTER_PAD_SAMPLES)
    pad_e = min(n_total, center + win_samples + FILTER_PAD_SAMPLES)
    data = np.memmap(bin_path, dtype=np.int16, mode="r",
                       shape=(n_total, n_chan))
    raw = np.asarray(data[pad_s:pad_e, [ch_a, ch_b]], dtype=np.float32) * gain
    raw = sosfiltfilt(notch_60, raw, axis=0)
    raw = sosfiltfilt(notch_120, raw, axis=0)
    bip = raw[:, 0] - raw[:, 1]
    filt = sosfiltfilt(bp_ripple, bip)
    env = np.abs(hilbert(filt))
    s_off = center - pad_s - win_samples
    e_off = s_off + 2 * win_samples
    bip = bip[s_off:e_off]
    filt = filt[s_off:e_off]
    env = env[s_off:e_off]
    t = np.arange(len(bip)) / fs_raw - win_s

    fig, axes = plt.subplots(3, 1, figsize=(6.5, 5), sharex=True)
    axes[0].plot(t, bip, lw=0.6, color="black")
    axes[0].set_ylabel("Bipolar LFP (µV)")
    axes[1].plot(t, filt, lw=0.6, color="C3")
    axes[1].set_ylabel("100-250 Hz (µV)")
    axes[2].plot(t, env, lw=0.7, color="C2")
    axes[2].set_ylabel("Envelope (µV)")
    axes[2].set_xlabel("Time relative to peak (s)")
    for ax in axes:
        ax.axvline(0, color="k", lw=0.5, ls="--", alpha=0.5)
    axes[0].set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---- Per-session orchestrator ----
def run_session(sn, lfp_paths, sorted_paths, cfg, pairs_def, base_out, base_fig,
                 rng, paths_data):
    print(f"\n========== S{sn} ({SESSION_STATE[sn]}) ==========", flush=True)
    t0 = time.time()

    # Foraging duration from binned HMM
    binned_npz_path = (REPO_ROOT / cfg["out_dirs"]["binned"]
                         / f"session_{sn}.npz")
    binned = np.load(binned_npz_path, allow_pickle=True)
    trial_time = np.asarray(binned["trial_time"], dtype=np.float64)
    HMM_BIN_S = 0.480
    foraging_duration_s = float(trial_time[-1] + HMM_BIN_S)
    print(f"  Foraging duration: {foraging_duration_s:.1f}s", flush=True)

    # Lookups for bipolar pairs per probe
    imec0_lookup = {"ACA": (pairs_def["ACA"][0]["channel_a"].values.astype(np.int64),
                              pairs_def["ACA"][0]["channel_b"].values.astype(np.int64))}
    imec1_lookup = {
        "LHA": (pairs_def["LHA"][0]["channel_a"].values.astype(np.int64),
                pairs_def["LHA"][0]["channel_b"].values.astype(np.int64)),
        "RSP": (pairs_def["RSP"][0]["channel_a"].values.astype(np.int64),
                pairs_def["RSP"][0]["channel_b"].values.astype(np.int64)),
    }

    # Preprocess imec0
    print(f"  Preprocessing imec0 (ACA)...", flush=True)
    env0, art0_raw, n_keep0, fs_raw0, gain0 = preprocess_envelope(
        lfp_paths["imec0_bin"], lfp_paths["imec0_meta"],
        imec0_lookup, sn, foraging_duration_s, "imec0",
    )

    print(f"  Preprocessing imec1 (LHA + RSP)...", flush=True)
    env1, art1_raw, n_keep1, fs_raw1, gain1 = preprocess_envelope(
        lfp_paths["imec1_bin"], lfp_paths["imec1_meta"],
        imec1_lookup, sn, foraging_duration_s, "imec1",
    )

    # Build envelope-rate artifact masks (decimate raw mask by OR over chunks of 5)
    def downsample_mask(mask_raw, factor):
        n = len(mask_raw) // factor
        return mask_raw[:n * factor].reshape(n, factor).any(axis=1)

    art0_env = downsample_mask(art0_raw, ENV_DECIMATION)
    art1_env = downsample_mask(art1_raw, ENV_DECIMATION)

    # Detect events per pair
    fs_env = FS_ENV_DS
    print(f"  Detecting events per pair...", flush=True)
    events_ACA = detect_events_per_pair(
        env0["ACA"], fs_env, art0_env, MIN_DURATION_MS,
        REFRACTORY_MS, DETECT_Z)
    events_ACA = reject_near_artifact(events_ACA, art0_env, fs_env, ARTIFACT_REJECT_MS)
    events_LHA = detect_events_per_pair(
        env1["LHA"], fs_env, art1_env, MIN_DURATION_MS,
        REFRACTORY_MS, DETECT_Z)
    events_LHA = reject_near_artifact(events_LHA, art1_env, fs_env, ARTIFACT_REJECT_MS)
    events_RSP = detect_events_per_pair(
        env1["RSP"], fs_env, art1_env, MIN_DURATION_MS,
        REFRACTORY_MS, DETECT_Z)
    events_RSP = reject_near_artifact(events_RSP, art1_env, fs_env, ARTIFACT_REJECT_MS)

    for region, evs in [("ACA", events_ACA), ("LHA", events_LHA), ("RSP", events_RSP)]:
        total = sum(len(e) for e in evs)
        print(f"    {region}: {total} per-pair events ({len(evs)} pairs)",
              flush=True)

    # Filter banks for peak-freq and plotting
    notch60_0 = design_notch_sos(60.0, 30, fs_raw0)
    notch120_0 = design_notch_sos(120.0, 30, fs_raw0)
    bp_rip_0 = design_bandpass_sos(*RIPPLE_BAND, 4, fs_raw0)
    notch60_1 = design_notch_sos(60.0, 30, fs_raw1)
    notch120_1 = design_notch_sos(120.0, 30, fs_raw1)
    bp_rip_1 = design_bandpass_sos(*RIPPLE_BAND, 4, fs_raw1)
    fb0 = dict(notch_60=notch60_0, notch_120=notch120_0, bp_ripple=bp_rip_0,
                fs_raw=fs_raw0, gain=gain0,
                bin_path=lfp_paths["imec0_bin"],
                meta_path=lfp_paths["imec0_meta"])
    fb1 = dict(notch_60=notch60_1, notch_120=notch120_1, bp_ripple=bp_rip_1,
                fs_raw=fs_raw1, gain=gain1,
                bin_path=lfp_paths["imec1_bin"],
                meta_path=lfp_paths["imec1_meta"])
    fbs = {"ACA": fb0, "LHA": fb1, "RSP": fb1}

    # Save per-pair events to CSV
    def per_pair_to_df(events_per_pair, pairs_df, region):
        rows = []
        for pi, evs in enumerate(events_per_pair):
            pinfo = pairs_df.iloc[pi]
            for ev in evs:
                rows.append(dict(
                    pair_index=int(pinfo["pair_index"]),
                    channel_a=int(pinfo["channel_a"]),
                    channel_b=int(pinfo["channel_b"]),
                    shank=int(pinfo["shank"]),
                    mean_x_um=float(pinfo["mean_x_um"]),
                    mean_y_um=float(pinfo["mean_y_um"]),
                    region=region,
                    peak_time_s=ev["peak_idx"] / fs_env,
                    peak_amplitude_z=ev["peak_z"],
                    duration_ms=ev["duration_ms"],
                    start_time_s=ev["start_idx"] / fs_env,
                    end_time_s=ev["end_idx"] / fs_env,
                ))
        return pd.DataFrame(rows)

    pair_dfs = {}
    pair_dfs["ACA"] = per_pair_to_df(events_ACA, pairs_def["ACA"][0], "ACA")
    pair_dfs["LHA"] = per_pair_to_df(events_LHA, pairs_def["LHA"][0], "LHA")
    pair_dfs["RSP"] = per_pair_to_df(events_RSP, pairs_def["RSP"][0], "RSP")

    # Regional aggregation
    print(f"  Regional aggregation...", flush=True)
    regional_events_by_region = {}
    for region, ev_list, n_pairs in [("ACA", events_ACA, len(events_ACA)),
                                          ("LHA", events_LHA, len(events_LHA)),
                                          ("RSP", events_RSP, len(events_RSP))]:
        n_t = env0["ACA"].shape[0] if region == "ACA" else env1["LHA"].shape[0]
        reg_evs, density = aggregate_regional_events(ev_list, n_pairs, fs_env, n_t)
        print(f"    {region}: {len(reg_evs)} regional events (threshold "
              f"≥{int(np.ceil(REGIONAL_PAIR_THRESHOLD_FRAC * n_pairs))} pairs)",
              flush=True)
        regional_events_by_region[region] = reg_evs

    # Compute peak frequencies for each regional event
    print(f"  Computing peak frequencies (re-read raw)...", flush=True)
    for region, reg_evs in regional_events_by_region.items():
        fb = fbs[region]
        n_chan = int(parse_lf_meta(fb["meta_path"]).get("nSavedChans",
                                                            N_NEURAL_CHANNELS + 1))
        for ev in reg_evs:
            # Use first contributing pair for peak freq estimation
            if not ev.get("contributing_pairs"):
                ev["peak_frequency_hz"] = np.nan
                continue
            pi, pair_ev = ev["contributing_pairs"][0]
            pinfo = pairs_def[region][0].iloc[pi]
            ev["peak_frequency_hz"] = peak_frequency_from_raw(
                fb["bin_path"], fb["meta_path"], ev["peak_time_s"],
                fb["fs_raw"], n_chan, int(pinfo["channel_a"]),
                int(pinfo["channel_b"]), fb["gain"],
                fb["notch_60"], fb["notch_120"], fb["bp_ripple"],
            )

    # Save per-pair events and regional events
    out_dir = base_out / f"session_{sn}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for region, df in pair_dfs.items():
        df.to_csv(out_dir / f"session_{sn}_{region}_per_pair_events.csv",
                   index=False)
    for region, reg_evs in regional_events_by_region.items():
        if not reg_evs:
            pd.DataFrame().to_csv(out_dir / f"session_{sn}_{region}_regional_events.csv",
                                     index=False)
            continue
        rows = []
        for i, ev in enumerate(reg_evs):
            rows.append(dict(event_id=i,
                              region=region,
                              peak_time_s=ev["peak_time_s"],
                              n_pairs_active=ev["n_pairs_active"],
                              mean_peak_amplitude_z=ev["mean_peak_z"],
                              mean_duration_ms=ev["mean_duration_ms"],
                              mean_peak_frequency_hz=ev.get("peak_frequency_hz",
                                                              np.nan)))
        pd.DataFrame(rows).to_csv(
            out_dir / f"session_{sn}_{region}_regional_events.csv", index=False)

    # Spike validation
    print(f"  Spike validation...", flush=True)
    validation_summaries = {}
    behav_dfs = {}
    for region in ("ACA", "LHA", "RSP"):
        sp_path = sorted_paths[region]
        reg_evs = regional_events_by_region[region]
        if not sp_path or not Path(sp_path).exists():
            print(f"    {region}: no sorted path; skipping validation", flush=True)
            validation_summaries[region] = dict(n_units=0, n_events=len(reg_evs),
                                                  n_validated=0, p_mw=np.nan)
        elif not reg_evs:
            validation_summaries[region] = dict(n_units=0, n_events=0,
                                                  n_validated=0, p_mw=np.nan)
        else:
            df_val, summary = validate_events_with_spikes(
                reg_evs, sp_path, region, foraging_duration_s, rng)
            df_val["event_id"] = np.arange(len(df_val))
            df_val.to_csv(out_dir / f"session_{sn}_{region}_event_validation.csv",
                            index=False)
            validation_summaries[region] = summary
            print(f"    {region}: {summary['n_units']} good units, "
                  f"{summary['n_validated']}/{summary['n_events']} validated "
                  f"(MW p={summary['p_mw']:.2g})", flush=True)
        # Behavioral context
        df_bc = behavioral_context(reg_evs, binned_npz_path)
        if len(df_bc):
            df_bc["event_id"] = np.arange(len(df_bc))
        df_bc.to_csv(out_dir / f"session_{sn}_{region}_event_behavior.csv",
                       index=False)
        behav_dfs[region] = df_bc

    # Cross-region co-occurrence
    print(f"  Cross-region co-occurrence...", flush=True)
    df_cooc = cross_region_cooccurrence(regional_events_by_region,
                                              foraging_duration_s, rng)
    df_cooc["session"] = sn
    df_cooc.to_csv(out_dir / f"session_{sn}_cross_region_co_occurrence.csv",
                     index=False)
    print(f"    {df_cooc.to_string(index=False)}", flush=True)

    # Example traces: 5 per region (top by amplitude)
    print(f"  Example traces (top 5 per region)...", flush=True)
    fig_dir = base_fig / "example_traces"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for region in ("ACA", "LHA", "RSP"):
        reg_evs = regional_events_by_region[region]
        if not reg_evs:
            continue
        sorted_by_amp = sorted(reg_evs, key=lambda e: -e["mean_peak_z"])
        fb = fbs[region]
        n_chan = int(parse_lf_meta(fb["meta_path"]).get("nSavedChans",
                                                            N_NEURAL_CHANNELS + 1))
        for i, ev in enumerate(sorted_by_amp[:5]):
            if not ev.get("contributing_pairs"):
                continue
            pi, _ = ev["contributing_pairs"][0]
            pinfo = pairs_def[region][0].iloc[pi]
            title = (f"S{sn} {region} event {i} t={ev['peak_time_s']:.2f}s "
                     f"z={ev['mean_peak_z']:.2f} dur={ev['mean_duration_ms']:.1f} ms "
                     f"f={ev.get('peak_frequency_hz', float('nan')):.0f} Hz")
            plot_example_trace(
                fb["bin_path"], fb["meta_path"],
                int(pinfo["channel_a"]), int(pinfo["channel_b"]),
                ev["peak_time_s"], n_chan, fb["gain"],
                fb["notch_60"], fb["notch_120"], fb["bp_ripple"],
                fb["fs_raw"],
                fig_dir / f"session_{sn}_{region}_event_{i}.png",
                title=title,
            )

    print(f"  Done S{sn} [{time.time()-t0:.0f}s]", flush=True)
    return dict(
        sn=sn,
        regional_events=regional_events_by_region,
        validation=validation_summaries,
        behavior=behav_dfs,
        cooc=df_cooc,
        foraging_duration_s=foraging_duration_s,
    )


# ---- Cross-session aggregation & figures ----
def cross_session(per_sess, base_out, base_fig):
    print("\n========== Cross-session aggregation ==========", flush=True)
    # Build aggregate CSVs
    all_rows = []
    for r in per_sess:
        for region, reg_evs in r["regional_events"].items():
            rate = len(reg_evs) / (r["foraging_duration_s"] / 60.0)
            for i, ev in enumerate(reg_evs):
                all_rows.append(dict(
                    session=r["sn"],
                    state=SESSION_STATE[r["sn"]],
                    region=region,
                    event_id=i,
                    peak_time_s=ev["peak_time_s"],
                    n_pairs_active=ev["n_pairs_active"],
                    mean_peak_z=ev["mean_peak_z"],
                    mean_duration_ms=ev["mean_duration_ms"],
                    peak_frequency_hz=ev.get("peak_frequency_hz", np.nan),
                ))
    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(base_out / "all_regional_events.csv", index=False)

    # Per-session event rate
    rate_rows = []
    for r in per_sess:
        for region, reg_evs in r["regional_events"].items():
            rate = len(reg_evs) / (r["foraging_duration_s"] / 60.0)
            rate_rows.append(dict(session=r["sn"], state=SESSION_STATE[r["sn"]],
                                    region=region, n_events=len(reg_evs),
                                    rate_per_min=rate))
    rate_df = pd.DataFrame(rate_rows)
    rate_df.to_csv(base_out / "ripple_rate_per_session.csv", index=False)

    # Validation table
    val_rows = []
    for r in per_sess:
        for region in ("ACA", "LHA", "RSP"):
            v = r["validation"].get(region, {})
            val_rows.append(dict(session=r["sn"], region=region,
                                   n_units=v.get("n_units", 0),
                                   n_events=v.get("n_events", 0),
                                   n_validated=v.get("n_validated", 0),
                                   p_mw=v.get("p_mw", np.nan),
                                   median_event_spikes=v.get("median_event_spikes", np.nan),
                                   median_control_spikes=v.get("median_control_spikes", np.nan)))
    val_df = pd.DataFrame(val_rows)
    val_df.to_csv(base_out / "validation_summary.csv", index=False)

    # Cross-region cooc
    cooc_all = pd.concat([r["cooc"] for r in per_sess], ignore_index=True)
    cooc_all.to_csv(base_out / "cross_region_co_occurrence_all_sessions.csv",
                      index=False)

    # ---- Figures ----
    # Event rate per session
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
    for ax, region in zip(axes, ("ACA", "LHA", "RSP")):
        sub = rate_df[rate_df.region == region]
        colors = ["#1f77b4" if s == "fed" else "#d62728" for s in sub["state"]]
        ax.bar(np.arange(len(sub)), sub["rate_per_min"], color=colors)
        ax.set_xticks(np.arange(len(sub)))
        ax.set_xticklabels([f"S{s}" for s in sub["session"]])
        ax.set_title(region)
        ax.set_ylabel("events/min")
    fig.suptitle("Ripple regional event rate (fed=blue, fasted=red)")
    fig.tight_layout()
    fig.savefig(base_fig / "ripple_rate_per_session.png", dpi=130)
    plt.close(fig)

    # Peak frequency histograms
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, region in zip(axes, ("ACA", "LHA", "RSP")):
        sub = all_df[(all_df.region == region) & all_df.peak_frequency_hz.notna()]
        if not len(sub):
            ax.set_title(f"{region} (n=0)")
            continue
        ax.hist(sub["peak_frequency_hz"], bins=np.arange(100, 260, 10),
                  color="C2", edgecolor="black")
        mode_freq = float(sub["peak_frequency_hz"].median())
        ax.axvline(mode_freq, color="red", ls="--", lw=1)
        ax.set_title(f"{region} (n={len(sub)}, median={mode_freq:.0f} Hz)")
        ax.set_xlabel("Hz")
    axes[0].set_ylabel("count")
    fig.suptitle("Peak frequency distribution per region (100-250 Hz band)")
    fig.tight_layout()
    fig.savefig(base_fig / "peak_frequency_histograms.png", dpi=130)
    plt.close(fig)

    # Duration & amplitude distributions
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for col, region in enumerate(("ACA", "LHA", "RSP")):
        sub = all_df[all_df.region == region]
        axes[0, col].hist(sub["mean_duration_ms"].dropna(), bins=30,
                            color="C0", edgecolor="black")
        axes[0, col].set_title(f"{region} duration (n={len(sub)})")
        axes[0, col].set_xlabel("ms")
        axes[1, col].hist(sub["mean_peak_z"].dropna(), bins=30,
                            color="C4", edgecolor="black")
        axes[1, col].set_title(f"{region} amplitude")
        axes[1, col].set_xlabel("z-score")
    fig.suptitle("Duration & amplitude distributions per region")
    fig.tight_layout()
    fig.savefig(base_fig / "duration_amplitude_distributions.png", dpi=130)
    plt.close(fig)

    # Behavioral context: aggregate per region
    all_bc = []
    for r in per_sess:
        for region, df in r["behavior"].items():
            if df is None or not len(df):
                continue
            d = df.copy(); d["session"] = r["sn"]; d["region"] = region
            all_bc.append(d)
    if all_bc:
        bc_all = pd.concat(all_bc, ignore_index=True)
        bc_all.to_csv(base_out / "event_behavior_all_sessions.csv", index=False)

        ZONES = ["home", "transition", "pot", "pot_zone", "arena", "other"]
        LOCS = ["stationary", "slow_locomotion", "fast_locomotion"]
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        for col, region in enumerate(("ACA", "LHA", "RSP")):
            sub = bc_all[bc_all.region == region]
            # Zone
            counts = sub["zone"].value_counts().reindex(ZONES).fillna(0)
            axes[0, col].bar(ZONES, counts.values, color="C0", edgecolor="black")
            axes[0, col].set_title(f"{region} zone")
            axes[0, col].tick_params(axis="x", rotation=30)
            # Locomotion
            counts = sub["locomotion"].value_counts().reindex(LOCS).fillna(0)
            axes[1, col].bar(LOCS, counts.values, color="C3", edgecolor="black")
            axes[1, col].set_title(f"{region} locomotion")
            axes[1, col].tick_params(axis="x", rotation=30)
        fig.suptitle("Behavioral context per region (event counts)")
        fig.tight_layout()
        fig.savefig(base_fig / "behavioral_context_per_region.png", dpi=130)
        plt.close(fig)

    # Cross-region cooc figure
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    PAIR_LBL = ["ACA-LHA", "ACA-RSP", "LHA-RSP"]
    for ax, pair_name in zip(axes, PAIR_LBL):
        sub = cooc_all[cooc_all.pair == pair_name]
        x = np.arange(len(sub))
        ax.bar(x - 0.2, sub["obs_cooc_rate_A"], 0.4, color="C0", label="observed")
        ax.bar(x + 0.2, sub["shuf_p95"], 0.4, color="C7", label="shuffle p95")
        ax.set_xticks(x); ax.set_xticklabels([f"S{s}" for s in sub["session"]])
        ax.set_title(pair_name)
        ax.set_ylabel("cooc rate (A-events with B within ±50 ms)")
        ax.legend(fontsize=8)
    fig.suptitle("Cross-region ripple co-occurrence per session")
    fig.tight_layout()
    fig.savefig(base_fig / "cross_region_co_occurrence.png", dpi=130)
    plt.close(fig)

    # Print summary
    print("\n========== SUMMARY ==========")
    for region in ("ACA", "LHA", "RSP"):
        sub = all_df[all_df.region == region]
        rate_sub = rate_df[rate_df.region == region]
        val_sub = val_df[val_df.region == region]
        if not len(sub):
            print(f"  {region}: 0 events")
            continue
        modal = float(sub["peak_frequency_hz"].median())
        print(f"  {region}: n={len(sub)} events, "
              f"rate={rate_sub['rate_per_min'].mean():.2f}/min, "
              f"validated={int(val_sub['n_validated'].sum())}/"
              f"{int(val_sub['n_events'].sum())}, "
              f"modal_freq={modal:.0f} Hz, "
              f"mean_dur={sub['mean_duration_ms'].mean():.1f} ms, "
              f"mean_amp_z={sub['mean_peak_z'].mean():.2f}")

    print("\nCross-region cooc (mean observed vs mean shuffle p95):")
    for pair_name in PAIR_LBL:
        sub = cooc_all[cooc_all.pair == pair_name]
        print(f"  {pair_name}: obs={sub['obs_cooc_rate_A'].mean():.3f}, "
              f"shuf_p95={sub['shuf_p95'].mean():.3f}, "
              f"sessions_passing={int(sub['exceeds_p95'].sum())}/{len(sub)}")


# ---- Main ----
def main():
    cfg = load_config()
    base_out, base_fig = out_dirs()

    print("=== Loading geometry ===")
    geom, pairs_def = load_geometry()
    for region in ("ACA", "LHA", "RSP"):
        df, probe = pairs_def[region]
        print(f"  {region}: {len(df)} pairs ({probe})")

    print("\n=== Discovering LFP files ===")
    disc = discover_lfp_files()
    lfp_map = map_session_to_lfp(disc)
    print(f"  Mapped {len(lfp_map)} sessions")

    # Load paths.yaml for sorted paths
    with open(REPO_ROOT / cfg["paths_yaml"]) as f:
        paths_data = yaml.safe_load(f)
    dp_sessions = paths_data["double_probe"]["coordinates_1"]["mouse01"]["sessions"]

    def sorted_paths_for(sn):
        sval = dp_sessions[f"session_{sn}"]
        return dict(
            ACA=sval.get("probe_0_aca", {}).get("sorted"),
            LHA=sval.get("probe_1_lha_rsp", {}).get("sorted"),
            RSP=sval.get("probe_1_lha_rsp", {}).get("sorted"),
        )

    rng = np.random.default_rng(SEED)
    per_sess = []
    for sn in SESSIONS:
        if sn not in lfp_map:
            print(f"  Skipping S{sn} (no LFP)")
            continue
        try:
            r = run_session(sn, lfp_map[sn], sorted_paths_for(sn), cfg,
                              pairs_def, base_out, base_fig, rng, paths_data)
            per_sess.append(r)
        except Exception as e:
            import traceback
            print(f"  ERROR S{sn}: {e}")
            traceback.print_exc()

    cross_session(per_sess, base_out, base_fig)
    print("\nDone.")


if __name__ == "__main__":
    main()
