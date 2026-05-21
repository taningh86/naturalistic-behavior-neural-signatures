"""19b — Re-aggregate SWR per-pair events at a tunable threshold.

The 10% pair threshold used in script 19 was too strict (0-13 regional events
total across 6 sessions). This script reads the saved per-pair event CSVs and
re-runs:

  1. Regional aggregation at a configurable threshold (default 2%).
  2. Peak frequency lookup (re-read raw .lf.bin at event peaks).
  3. Spike validation (good QC units, ±50 ms window vs random controls).
  4. Behavioral context from binned HMM npz.
  5. Cross-region co-occurrence with shuffle null.
  6. Cross-session figures + summary.

Outputs land in `data/HMM/neural_alignment/swr_detection/threshold_{pct}/`.
"""
from pathlib import Path
import argparse
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
from scipy.signal import butter, sosfiltfilt, iirnotch, tf2sos, hilbert
from scipy.stats import mannwhitneyu

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "HMM"))
from _utils import load_config


BASE_OUT_ROOT = REPO_ROOT / "data" / "HMM" / "neural_alignment" / "swr_detection"
BASE_FIG_ROOT = REPO_ROOT / "figures" / "HMM" / "neural_alignment" / "swr_detection"
GEOMETRY_DIR = REPO_ROOT / "data" / "HMM" / "neural_alignment" / "lfp"

SESSIONS = [4, 6, 8, 12, 14, 16]
SESSION_STATE = {4: "fed", 6: "fed", 8: "fed",
                  12: "fasted", 14: "fasted", 16: "fasted"}
SESSION_DATE_MAP = {
    4:  "6_17_25", 6: "6_24_25", 8: "6_30_25",
    12: "7_11_25", 14: "7_17_25", 16: "7_25_25",
}
N_PAIRS_V1 = {"ACA": 370, "LHA": 184, "RSP": 184}
N_PAIRS_V2 = {"ACA": 370, "LHA": 176, "RSP": 176}

# Same constants as script 19
FS_LFP_RAW = 2500.0
FS_ENV_DS = 500.0
RIPPLE_BAND = (100.0, 250.0)
REGIONAL_BIN_MS = 5.0
REGIONAL_COINCIDENCE_WINDOW_MS = 25.0
REFRACTORY_MS = 50.0
SPIKE_WIN_MS = 50.0
N_CONTROLS = 1000
COOC_WINDOW_MS = 50.0
N_COOC_SHUFFLES = 100
N_NEURAL_CHANNELS = 384
FILTER_PAD_SAMPLES = int(0.2 * FS_LFP_RAW)
SEED = 20260512

# QC thresholds
P0_MIN_FR = 0.2
P1_MIN_FR = 0.2
P1_MIN_AMP = 43
LHA_DEPTH_MIN = 0
LHA_DEPTH_MAX = 345
RSP_DEPTH_MIN = 4680
RSP_DEPTH_MAX = 5025

CATGT_ROOT = Path("H:/Neuropixels Data/Cat_GT_Out")


# ---- Geometry ----
def load_geometry(pairs_version="v1"):
    if pairs_version == "v1":
        p0 = pd.read_csv(GEOMETRY_DIR / "bipolar_pairs_imec0.csv")
        p1 = pd.read_csv(GEOMETRY_DIR / "bipolar_pairs_imec1.csv")
    else:
        p0 = pd.read_csv(GEOMETRY_DIR / "bipolar_pairs_imec0_v2.csv")
        p1 = pd.read_csv(GEOMETRY_DIR / "bipolar_pairs_imec1_v2.csv")
    return {
        "ACA": p0,
        "LHA": p1[p1.region == "LHA"].reset_index(drop=True),
        "RSP": p1[p1.region == "RSP"].reset_index(drop=True),
    }


def pair_index_set(geom_pairs_region):
    """Return the set of pair_index values in this region's pair list."""
    return set(geom_pairs_region["pair_index"].astype(int).tolist())


# ---- LFP file discovery ----
def discover_lfp_for_session(sn):
    date_str = SESSION_DATE_MAP[sn]
    cutoff = datetime(2025, 8, 29)
    res = {}
    for folder in sorted(CATGT_ROOT.iterdir()):
        if not folder.name.startswith("catgt_DOUBLE_PROBE_"):
            continue
        if "HFD" in folder.name or "_HOME" in folder.name or "_EXP" in folder.name:
            continue
        if date_str not in folder.name or "_FOR" not in folder.name:
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
            if lf_bin and lf_meta:
                res[f"imec{probe}_bin"] = str(lf_bin[0])
                res[f"imec{probe}_meta"] = str(lf_meta[0])
    return res


def parse_lf_meta(meta_path):
    meta = {}
    with open(meta_path) as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                k, v = line.split("=", 1)
                meta[k] = v
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
    return ((rng_max - rng_min) / (2 * max_int * gain)) * 1e6


def design_notch_sos(f0, q, fs):
    b, a = iirnotch(f0, q, fs=fs)
    return tf2sos(b, a)


def design_bandpass_sos(lo, hi, order, fs):
    return butter(order, [lo / (fs / 2), hi / (fs / 2)],
                   btype="bandpass", output="sos")


# ---- Re-aggregation ----
def aggregate_at_threshold(per_pair_df, n_pairs, threshold_frac, foraging_s):
    """Aggregate per-pair events into regional events at given threshold."""
    bin_samples = max(1, int(REGIONAL_BIN_MS * 1e-3 * FS_ENV_DS))
    coinc_bins = max(1, int(REGIONAL_COINCIDENCE_WINDOW_MS
                              * 1e-3 * FS_ENV_DS / bin_samples))
    n_bins = int(foraging_s * FS_ENV_DS) // bin_samples
    thr = int(np.ceil(threshold_frac * n_pairs))

    # density and contrib per bin
    dens = np.zeros(n_bins, dtype=np.int32)
    contrib = [list() for _ in range(n_bins)]
    for _, r in per_pair_df.iterrows():
        b = int(r["peak_time_s"] * FS_ENV_DS) // bin_samples
        if 0 <= b < n_bins:
            dens[b] += 1
            contrib[b].append(int(r["pair_index"]))

    kernel = np.ones(2 * coinc_bins + 1, dtype=np.int32)
    smoothed = np.convolve(dens, kernel, mode="same")
    in_event = smoothed >= thr
    diff = np.diff(in_event.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)

    events = []
    for s, e in zip(starts, ends):
        peak_bin = s + int(np.argmax(smoothed[s:e]))
        peak_t = peak_bin * bin_samples / FS_ENV_DS
        active = set()
        peak_zs = []
        durs = []
        contributing_pairs = []
        for b in range(max(0, peak_bin - coinc_bins),
                       min(n_bins, peak_bin + coinc_bins + 1)):
            for pi in contrib[b]:
                if pi in active:
                    continue
                active.add(pi)
                contributing_pairs.append(pi)
                # Find the event in per_pair_df
                evs_p = per_pair_df[per_pair_df.pair_index == pi]
                # closest event by peak_time_s
                ix = (evs_p["peak_time_s"] - peak_t).abs().idxmin()
                peak_zs.append(float(evs_p.loc[ix, "peak_amplitude_z"]))
                durs.append(float(evs_p.loc[ix, "duration_ms"]))
        if not active:
            continue
        events.append(dict(
            peak_time_s=float(peak_t),
            n_pairs_active=int(len(active)),
            mean_peak_amplitude_z=float(np.mean(peak_zs)),
            mean_duration_ms=float(np.mean(durs)),
            contributing_pairs=contributing_pairs,
        ))

    events.sort(key=lambda d: d["peak_time_s"])
    merged = []
    refr_s = REFRACTORY_MS * 1e-3
    for ev in events:
        if merged and ev["peak_time_s"] - merged[-1]["peak_time_s"] < refr_s:
            if ev["n_pairs_active"] > merged[-1]["n_pairs_active"]:
                merged[-1] = ev
            continue
        merged.append(ev)
    return merged


# ---- Peak frequency ----
def peak_frequency_from_raw(bin_path, meta_path, peak_time_s, fs_raw, n_chan,
                              ch_a, ch_b, gain, notch_60, notch_120, bp_ripple,
                              window_ms=50.0):
    win_samples = int(window_ms * 1e-3 * fs_raw)
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
    s_off = center - pad_s - win_samples
    seg = filt[s_off:s_off + 2 * win_samples]
    if len(seg) < 32:
        return np.nan
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
    freqs = np.fft.rfftfreq(len(seg), d=1 / fs_raw)
    mask = (freqs >= RIPPLE_BAND[0]) & (freqs <= RIPPLE_BAND[1])
    return float(freqs[mask][np.argmax(spec[mask])]) if mask.any() else np.nan


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
    sorted_path = Path(sorted_path)
    st_path = sorted_path / "spike_times.npy"
    sc_path = sorted_path / "spike_clusters.npy"
    if not st_path.exists() or not sc_path.exists():
        return np.array([])
    st = np.load(st_path).astype(np.int64).ravel()
    sc = np.load(sc_path).astype(np.int64).ravel()
    keep = np.isin(sc, cluster_ids)
    return st[keep] / fs_ap


def validate_events(regional_events, sorted_path, region, foraging_s, rng):
    cluster_ids = load_good_units(sorted_path, region)
    if not len(cluster_ids) or not regional_events:
        return pd.DataFrame(), dict(p_mw=np.nan,
                                       n_units=int(len(cluster_ids)),
                                       n_events=int(len(regional_events)),
                                       n_validated=0,
                                       median_event_spikes=np.nan,
                                       median_control_spikes=np.nan)
    spikes = load_spike_times(sorted_path, cluster_ids)
    if not len(spikes):
        return pd.DataFrame(), dict(p_mw=np.nan,
                                       n_units=int(len(cluster_ids)),
                                       n_events=int(len(regional_events)),
                                       n_validated=0,
                                       median_event_spikes=np.nan,
                                       median_control_spikes=np.nan)
    win_s = SPIKE_WIN_MS * 1e-3

    event_times = np.array([ev["peak_time_s"] for ev in regional_events])
    event_counts = []
    for t in event_times:
        event_counts.append(int(np.sum((spikes >= t - win_s) & (spikes < t + win_s))))

    exclude_pad_s = 0.2
    control_counts = []
    rng_local = np.random.default_rng(rng.integers(1, 1_000_000_000))
    n_tried = 0; n_kept = 0
    while n_kept < N_CONTROLS and n_tried < 10 * N_CONTROLS:
        n_tried += 1
        t = float(rng_local.uniform(exclude_pad_s, foraging_s - exclude_pad_s))
        if len(event_times):
            if np.min(np.abs(event_times - t)) < exclude_pad_s:
                continue
        control_counts.append(int(np.sum((spikes >= t - win_s) & (spikes < t + win_s))))
        n_kept += 1
    control_counts = np.array(control_counts)
    p95 = float(np.percentile(control_counts, 95)) if len(control_counts) else np.nan

    rows = []
    for ev, c in zip(regional_events, event_counts):
        rows.append(dict(peak_time_s=ev["peak_time_s"],
                          event_locked_spikes=int(c),
                          control_p95=p95,
                          validated_flag=bool(c > p95)))
    df = pd.DataFrame(rows)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            _, p_mw = mannwhitneyu(event_counts, control_counts,
                                      alternative="greater")
        except Exception:
            p_mw = np.nan
    summary = dict(
        p_mw=float(p_mw) if np.isfinite(p_mw) else np.nan,
        n_units=int(len(cluster_ids)),
        n_events=int(len(event_counts)),
        n_validated=int(df["validated_flag"].sum()),
        median_event_spikes=float(np.median(event_counts)) if event_counts else np.nan,
        median_control_spikes=float(np.median(control_counts)) if len(control_counts) else np.nan,
    )
    return df, summary


# ---- Behavioral context ----
def behavioral_context(regional_events, binned_npz_path):
    if not regional_events or not Path(binned_npz_path).exists():
        return pd.DataFrame()
    d = np.load(binned_npz_path, allow_pickle=True)
    trial_time = np.asarray(d["trial_time"], dtype=np.float64)
    speed = np.asarray(d.get("speed", np.zeros_like(trial_time)))
    zone = np.asarray(d.get("zone_int", np.full_like(trial_time, -1, dtype=int)))
    ZONE_NAMES = {0: "home", 1: "transition", 2: "pot",
                   3: "pot_zone", 4: "arena", 5: "other"}
    dig = np.asarray(d.get("digging_sand", np.zeros_like(trial_time)))
    feed = np.asarray(d.get("feeding", np.zeros_like(trial_time)))
    rear = np.asarray(d.get("rearing", np.zeros_like(trial_time)))

    rows = []
    for ev in regional_events:
        t = ev["peak_time_s"]
        i = int(np.clip(np.searchsorted(trial_time, t) - 1, 0, len(trial_time) - 1))
        sp = float(speed[i])
        loc = ("stationary" if sp < 1.0
               else "slow_locomotion" if sp < 5.0
               else "fast_locomotion")
        rows.append(dict(peak_time_s=t, bin_idx=i, speed=sp, locomotion=loc,
                          zone=ZONE_NAMES.get(int(zone[i]), str(zone[i])),
                          dig=bool(dig[i]), feed=bool(feed[i]),
                          rear=bool(rear[i])))
    return pd.DataFrame(rows)


# ---- Cross-region cooc ----
def cross_region_cooccurrence(events_by_region, foraging_s, rng):
    rows = []
    PAIRS = [("ACA", "LHA"), ("ACA", "RSP"), ("LHA", "RSP")]
    win_s = COOC_WINDOW_MS * 1e-3
    for A, B in PAIRS:
        ta = np.array([e["peak_time_s"] for e in events_by_region.get(A, [])])
        tb = np.array([e["peak_time_s"] for e in events_by_region.get(B, [])])
        if not len(ta) or not len(tb):
            rows.append(dict(pair=f"{A}-{B}", n_A=int(len(ta)), n_B=int(len(tb)),
                              obs_cooc_rate_A=np.nan, obs_cooc_count=0,
                              shuf_mean=np.nan, shuf_p95=np.nan,
                              exceeds_p95=False))
            continue
        cooc = int(sum(np.any(np.abs(tb - tA) <= win_s) for tA in ta))
        obs_rate = cooc / len(ta)
        shuf_rates = []
        for _ in range(N_COOC_SHUFFLES):
            tb_shuf = rng.uniform(0, foraging_s, size=len(tb))
            c = int(sum(np.any(np.abs(tb_shuf - tA) <= win_s) for tA in ta))
            shuf_rates.append(c / len(ta))
        shuf_rates = np.array(shuf_rates)
        rows.append(dict(pair=f"{A}-{B}", n_A=int(len(ta)), n_B=int(len(tb)),
                          obs_cooc_rate_A=float(obs_rate),
                          obs_cooc_count=int(cooc),
                          shuf_mean=float(shuf_rates.mean()),
                          shuf_p95=float(np.percentile(shuf_rates, 95)),
                          exceeds_p95=bool(obs_rate > float(np.percentile(shuf_rates, 95)))))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.02,
                     help="Fraction of pairs required for regional event (default 0.02)")
    ap.add_argument("--pairs-version", choices=("v1", "v2"), default="v1",
                     help="v1: original 184 LHA + 184 RSP pairs; "
                          "v2: refined 176 LHA + 176 RSP pairs (excludes 8 boundary pairs each)")
    args = ap.parse_args()
    threshold_frac = float(args.threshold)
    pct = int(round(threshold_frac * 100))

    print(f"=== Re-aggregating with pairs={args.pairs_version}, "
          f"threshold = {threshold_frac:.3f} ({pct}%) ===")

    if args.pairs_version == "v1":
        base_out = BASE_OUT_ROOT / f"threshold_{pct:02d}pct"
        base_fig = BASE_FIG_ROOT / f"threshold_{pct:02d}pct"
        n_pairs_map = N_PAIRS_V1
    else:
        base_out = (REPO_ROOT / "data" / "HMM" / "neural_alignment"
                    / "swr_detection_v2" / f"threshold_{pct:02d}pct")
        base_fig = (REPO_ROOT / "figures" / "HMM" / "neural_alignment"
                    / "swr_detection_v2" / f"threshold_{pct:02d}pct")
        n_pairs_map = N_PAIRS_V2
    base_out.mkdir(parents=True, exist_ok=True)
    base_fig.mkdir(parents=True, exist_ok=True)
    print(f"Output: {base_out}")

    cfg = load_config()
    with open(REPO_ROOT / cfg["paths_yaml"]) as f:
        paths_data = yaml.safe_load(f)
    dp_sess = paths_data["double_probe"]["coordinates_1"]["mouse01"]["sessions"]

    geom_pairs = load_geometry(args.pairs_version)
    # Build pair_index → row lookup (pair_index ≠ iloc for LHA/RSP regional subsets)
    geom_lookup = {r: df.set_index("pair_index", drop=False)
                   for r, df in geom_pairs.items()}
    # pair_index sets for filtering per-pair CSVs to the v2 subset
    pair_index_sets = {r: pair_index_set(df) for r, df in geom_pairs.items()}
    print("Pair counts per region:")
    for r, df in geom_pairs.items():
        print(f"  {r}: {len(df)} pairs")
    rng = np.random.default_rng(SEED)

    per_sess = []
    for sn in SESSIONS:
        print(f"\n========== S{sn} ({SESSION_STATE[sn]}) ==========")
        t0 = time.time()

        binned_npz_path = (REPO_ROOT / cfg["out_dirs"]["binned"]
                             / f"session_{sn}.npz")
        binned = np.load(binned_npz_path, allow_pickle=True)
        trial_time = np.asarray(binned["trial_time"], dtype=np.float64)
        foraging_s = float(trial_time[-1] + 0.480)

        lfp_paths = discover_lfp_for_session(sn)
        sval = dp_sess[f"session_{sn}"]
        sorted_paths = dict(
            ACA=sval.get("probe_0_aca", {}).get("sorted"),
            LHA=sval.get("probe_1_lha_rsp", {}).get("sorted"),
            RSP=sval.get("probe_1_lha_rsp", {}).get("sorted"),
        )

        # Filter banks per probe
        fb = {}
        for probe_label, probe_key in [("imec0", "imec0_meta"), ("imec1", "imec1_meta")]:
            meta = parse_lf_meta(lfp_paths[probe_key])
            fs_raw = float(meta.get("imSampRate", FS_LFP_RAW))
            gain = lfp_gain_uV(meta)
            n_chan = int(meta.get("nSavedChans", N_NEURAL_CHANNELS + 1))
            fb[probe_label] = dict(
                fs_raw=fs_raw, gain=gain, n_chan=n_chan,
                bin_path=lfp_paths[probe_label + "_bin"],
                meta_path=lfp_paths[probe_label + "_meta"],
                notch_60=design_notch_sos(60.0, 30, fs_raw),
                notch_120=design_notch_sos(120.0, 30, fs_raw),
                bp_ripple=design_bandpass_sos(*RIPPLE_BAND, 4, fs_raw),
            )

        # Re-aggregate each region
        regional_events_by_region = {}
        for region in ("ACA", "LHA", "RSP"):
            csv = (BASE_OUT_ROOT / f"session_{sn}"
                   / f"session_{sn}_{region}_per_pair_events.csv")
            if not csv.exists():
                regional_events_by_region[region] = []
                continue
            df = pd.read_csv(csv)
            if not len(df):
                regional_events_by_region[region] = []
                continue
            # Filter per-pair events to only those pairs in this version's pair set
            n_pre = len(df)
            df = df[df["pair_index"].isin(pair_index_sets[region])].reset_index(drop=True)
            n_post = len(df)
            if n_pre != n_post:
                print(f"  {region}: filtered {n_pre} → {n_post} per-pair events "
                      f"({100*n_post/n_pre:.0f}% retained at v2 pair filter)")
            evs = aggregate_at_threshold(df, n_pairs_map[region], threshold_frac,
                                            foraging_s)
            print(f"  {region}: {len(evs)} events (threshold "
                  f"≥{int(np.ceil(threshold_frac * n_pairs_map[region]))} pairs)")

            # Peak frequency
            probe = "imec0" if region == "ACA" else "imec1"
            for ev in evs:
                if not ev["contributing_pairs"]:
                    ev["peak_frequency_hz"] = np.nan
                    continue
                pi = ev["contributing_pairs"][0]
                if pi not in geom_lookup[region].index:
                    ev["peak_frequency_hz"] = np.nan
                    continue
                pinfo = geom_lookup[region].loc[pi]
                ev["peak_frequency_hz"] = peak_frequency_from_raw(
                    fb[probe]["bin_path"], fb[probe]["meta_path"],
                    ev["peak_time_s"], fb[probe]["fs_raw"],
                    fb[probe]["n_chan"],
                    int(pinfo["channel_a"]), int(pinfo["channel_b"]),
                    fb[probe]["gain"], fb[probe]["notch_60"],
                    fb[probe]["notch_120"], fb[probe]["bp_ripple"],
                )
            regional_events_by_region[region] = evs

        # Save regional events
        out_dir = base_out / f"session_{sn}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for region, evs in regional_events_by_region.items():
            rows = []
            for i, ev in enumerate(evs):
                rows.append(dict(event_id=i, region=region,
                                  peak_time_s=ev["peak_time_s"],
                                  n_pairs_active=ev["n_pairs_active"],
                                  mean_peak_amplitude_z=ev["mean_peak_amplitude_z"],
                                  mean_duration_ms=ev["mean_duration_ms"],
                                  mean_peak_frequency_hz=ev.get("peak_frequency_hz",
                                                                  np.nan)))
            pd.DataFrame(rows).to_csv(
                out_dir / f"session_{sn}_{region}_regional_events.csv",
                index=False)

        # Spike validation + behavior + cooc
        val_summaries = {}
        behav_dfs = {}
        for region in ("ACA", "LHA", "RSP"):
            sp_path = sorted_paths[region]
            evs = regional_events_by_region[region]
            if sp_path and Path(sp_path).exists() and evs:
                df_val, summary = validate_events(evs, sp_path, region,
                                                      foraging_s, rng)
                df_val["event_id"] = np.arange(len(df_val))
                df_val.to_csv(out_dir / f"session_{sn}_{region}_event_validation.csv",
                                index=False)
                val_summaries[region] = summary
                print(f"    {region} validation: {summary['n_units']} good units, "
                      f"{summary['n_validated']}/{summary['n_events']} validated "
                      f"(MW p={summary['p_mw']:.2g}, med event={summary['median_event_spikes']:.0f} "
                      f"vs control={summary['median_control_spikes']:.1f})")
            else:
                val_summaries[region] = dict(n_units=0, n_events=len(evs),
                                              n_validated=0, p_mw=np.nan,
                                              median_event_spikes=np.nan,
                                              median_control_spikes=np.nan)
            df_bc = behavioral_context(evs, binned_npz_path)
            if len(df_bc):
                df_bc["event_id"] = np.arange(len(df_bc))
            df_bc.to_csv(out_dir / f"session_{sn}_{region}_event_behavior.csv",
                          index=False)
            behav_dfs[region] = df_bc

        df_cooc = cross_region_cooccurrence(regional_events_by_region,
                                                  foraging_s, rng)
        df_cooc["session"] = sn
        df_cooc.to_csv(out_dir / f"session_{sn}_cross_region_co_occurrence.csv",
                         index=False)

        per_sess.append(dict(sn=sn, regional_events=regional_events_by_region,
                              validation=val_summaries, behavior=behav_dfs,
                              cooc=df_cooc, foraging_s=foraging_s))
        print(f"  S{sn} done [{time.time()-t0:.0f}s]")

    # ---- Cross-session aggregation ----
    print("\n========== Cross-session aggregation ==========")
    all_rows = []
    rate_rows = []
    for r in per_sess:
        for region, evs in r["regional_events"].items():
            rate = len(evs) / (r["foraging_s"] / 60.0)
            rate_rows.append(dict(session=r["sn"], state=SESSION_STATE[r["sn"]],
                                    region=region, n_events=len(evs),
                                    rate_per_min=rate))
            for i, ev in enumerate(evs):
                all_rows.append(dict(session=r["sn"], state=SESSION_STATE[r["sn"]],
                                       region=region, event_id=i,
                                       peak_time_s=ev["peak_time_s"],
                                       n_pairs_active=ev["n_pairs_active"],
                                       mean_peak_z=ev["mean_peak_amplitude_z"],
                                       mean_duration_ms=ev["mean_duration_ms"],
                                       peak_frequency_hz=ev.get("peak_frequency_hz",
                                                                  np.nan)))
    all_df = pd.DataFrame(all_rows)
    rate_df = pd.DataFrame(rate_rows)
    all_df.to_csv(base_out / "all_regional_events.csv", index=False)
    rate_df.to_csv(base_out / "ripple_rate_per_session.csv", index=False)

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
                                   median_control_spikes=v.get("median_control_spikes",
                                                                  np.nan)))
    val_df = pd.DataFrame(val_rows)
    val_df.to_csv(base_out / "validation_summary.csv", index=False)

    cooc_all = pd.concat([r["cooc"] for r in per_sess], ignore_index=True)
    cooc_all.to_csv(base_out / "cross_region_co_occurrence_all_sessions.csv",
                      index=False)

    # ---- Figures ----
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, region in zip(axes, ("ACA", "LHA", "RSP")):
        sub = rate_df[rate_df.region == region]
        colors = ["#1f77b4" if s == "fed" else "#d62728" for s in sub["state"]]
        ax.bar(np.arange(len(sub)), sub["rate_per_min"], color=colors)
        ax.set_xticks(np.arange(len(sub)))
        ax.set_xticklabels([f"S{s}" for s in sub["session"]])
        ax.set_title(f"{region} ({sub['n_events'].sum()} total)")
        ax.set_ylabel("events/min")
    fig.suptitle(f"Ripple event rate (threshold={pct}%, fed=blue, fasted=red)")
    fig.tight_layout()
    fig.savefig(base_fig / "ripple_rate_per_session.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, region in zip(axes, ("ACA", "LHA", "RSP")):
        sub = all_df[(all_df.region == region) & all_df.peak_frequency_hz.notna()]
        if not len(sub):
            ax.set_title(f"{region} (n=0)")
            continue
        ax.hist(sub["peak_frequency_hz"], bins=np.arange(100, 260, 10),
                  color="C2", edgecolor="black")
        med = float(sub["peak_frequency_hz"].median())
        ax.axvline(med, color="red", ls="--")
        ax.set_title(f"{region} n={len(sub)}, median={med:.0f} Hz")
        ax.set_xlabel("Hz")
    fig.suptitle(f"Peak frequency distribution (threshold={pct}%)")
    fig.tight_layout()
    fig.savefig(base_fig / "peak_frequency_histograms.png", dpi=130)
    plt.close(fig)

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
    fig.suptitle(f"Duration & amplitude (threshold={pct}%)")
    fig.tight_layout()
    fig.savefig(base_fig / "duration_amplitude_distributions.png", dpi=130)
    plt.close(fig)

    # Behavioral context
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
            counts = sub["zone"].value_counts().reindex(ZONES).fillna(0)
            axes[0, col].bar(ZONES, counts.values, color="C0", edgecolor="black")
            axes[0, col].set_title(f"{region} zone (n={len(sub)})")
            axes[0, col].tick_params(axis="x", rotation=30)
            counts = sub["locomotion"].value_counts().reindex(LOCS).fillna(0)
            axes[1, col].bar(LOCS, counts.values, color="C3", edgecolor="black")
            axes[1, col].set_title(f"{region} locomotion")
            axes[1, col].tick_params(axis="x", rotation=30)
        fig.suptitle(f"Behavioral context per region (threshold={pct}%)")
        fig.tight_layout()
        fig.savefig(base_fig / "behavioral_context_per_region.png", dpi=130)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    PAIR_LBL = ["ACA-LHA", "ACA-RSP", "LHA-RSP"]
    for ax, pair_name in zip(axes, PAIR_LBL):
        sub = cooc_all[cooc_all.pair == pair_name]
        x = np.arange(len(sub))
        ax.bar(x - 0.2, sub["obs_cooc_rate_A"].fillna(0), 0.4, color="C0",
                label="observed")
        ax.bar(x + 0.2, sub["shuf_p95"].fillna(0), 0.4, color="C7",
                label="shuffle p95")
        ax.set_xticks(x); ax.set_xticklabels([f"S{s}" for s in sub["session"]])
        ax.set_title(pair_name)
        ax.set_ylabel("cooc rate")
        ax.legend(fontsize=8)
    fig.suptitle(f"Cross-region co-occurrence (threshold={pct}%)")
    fig.tight_layout()
    fig.savefig(base_fig / "cross_region_co_occurrence.png", dpi=130)
    plt.close(fig)

    # Summary
    print("\n========== SUMMARY ==========")
    for region in ("ACA", "LHA", "RSP"):
        sub = all_df[all_df.region == region]
        rate_sub = rate_df[rate_df.region == region]
        val_sub = val_df[val_df.region == region]
        if not len(sub):
            print(f"  {region}: 0 events")
            continue
        modal = float(sub["peak_frequency_hz"].median()) if sub["peak_frequency_hz"].notna().any() else np.nan
        print(f"  {region}: n={len(sub)} events, "
              f"rate={rate_sub['rate_per_min'].mean():.2f}/min, "
              f"validated={int(val_sub['n_validated'].sum())}/"
              f"{int(val_sub['n_events'].sum())}, "
              f"modal_freq={modal:.0f} Hz, "
              f"dur={sub['mean_duration_ms'].mean():.1f} ms, "
              f"amp_z={sub['mean_peak_z'].mean():.2f}")
    print("\nCross-region cooc (mean obs vs shuffle p95, n sessions passing):")
    for pair_name in PAIR_LBL:
        sub = cooc_all[cooc_all.pair == pair_name]
        print(f"  {pair_name}: obs={sub['obs_cooc_rate_A'].mean():.3f}, "
              f"shuf_p95={sub['shuf_p95'].mean():.3f}, "
              f"passing={int(sub['exceeds_p95'].sum())}/{len(sub)}")
    print("Done.")


if __name__ == "__main__":
    main()
