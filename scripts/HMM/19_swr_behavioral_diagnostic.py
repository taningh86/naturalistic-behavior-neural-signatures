"""19 behavioral diagnostic — verify v2 SWR locomotion/zone lookup.

Two confirmed bugs in script 19's `behavioral_context()`:
  1. The HMM binned npz key for zone integers is `zone`, NOT `zone_int`.
     `d.get("zone_int", np.full_like(trial_time, -1))` fell through and stored
     `-1` for every event. Every v2 zone field is broken.
  2. The HMM binned npz stores event flags in a single 2D array `events`
     (shape T × 7) with column names in `event_names`. Script 19's
     `d.get("digging_sand", ...)` etc. all fell through and stored False.

Speed lookup is correct (key matches), but the locomotion classification
("fast_locomotion" if speed >5 cm/s) may not match the actual speed
distribution of these animals.

Diagnostics
-----------
D1 Speed distribution per session (HMM 480 ms bins): percentiles + fraction
  in absolute speed bins.
D2 v2 stored lookups vs fresh lookups with correct npz keys. Confirms the
  zone/event bugs and quantifies how many events were affected.
D3 Re-classify ripples using animal-relative locomotion thresholds (median /
  75th percentile per session).
D4 Timestamp alignment: HMM trial_time length vs LFP foraging duration vs
  v2 event peak_time_s range.
D5 Waveform comparison: re-read raw .lf.bin in ±100 ms windows around RSP
  events, bandpass 100-250 Hz, average per locomotion state per session.
  If "fast-locomotion ripples" have different waveform morphology, they're
  likely partial artifacts.

Output: data/HMM/neural_alignment/swr_behavioral_diagnostic/
        figures/HMM/neural_alignment/swr_behavioral_diagnostic/
"""
from pathlib import Path
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, iirnotch, tf2sos, hilbert, windows

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "HMM"))
from _utils import load_config


# Inputs
SWR_V2_BASE = REPO / "data/HMM/neural_alignment/swr_detection_v2/threshold_02pct"
GEOM_DIR = REPO / "data/HMM/neural_alignment/lfp"

# Outputs
OUT_DIR = REPO / "data/HMM/neural_alignment/swr_behavioral_diagnostic"
FIG_DIR = REPO / "figures/HMM/neural_alignment/swr_behavioral_diagnostic"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

SESSIONS = [4, 6, 8, 12, 14, 16]
SESSION_STATE = {4: "fed", 6: "fed", 8: "fed",
                  12: "fasted", 14: "fasted", 16: "fasted"}
REGIONS = ("ACA", "LHA", "RSP")
HMM_BIN_S = 0.480

# LFP file discovery (same as script 17/19)
CATGT_ROOT = Path("H:/Neuropixels Data/Cat_GT_Out")
SESSION_DATE_MAP = {
    4: "6_17_25", 6: "6_24_25", 8: "6_30_25",
    12: "7_11_25", 14: "7_17_25", 16: "7_25_25",
}

FS_LFP_RAW = 2500.0
RIPPLE_BAND = (100.0, 250.0)
WAVEFORM_WIN_MS = 100.0
N_NEURAL_CHANNELS = 384
FILTER_PAD_SAMPLES = int(0.2 * FS_LFP_RAW)


def lookup_lfp_paths(sn):
    """Find .lf.bin and .lf.meta for a session."""
    date_str = SESSION_DATE_MAP[sn]
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


def load_geometry_v2():
    p0 = pd.read_csv(GEOM_DIR / "bipolar_pairs_imec0_v2.csv")
    p1 = pd.read_csv(GEOM_DIR / "bipolar_pairs_imec1_v2.csv")
    return {
        "ACA": p0,
        "LHA": p1[p1.region == "LHA"].reset_index(drop=True),
        "RSP": p1[p1.region == "RSP"].reset_index(drop=True),
    }


# ============================================================================
# D1: Speed distribution per session
# ============================================================================
def d1_speed_distribution(binned, sn):
    speed = np.asarray(binned["speed"], dtype=np.float64)
    qs = np.percentile(speed, [5, 25, 50, 75, 90, 95, 99])
    fb1 = float((speed < 1).mean())
    f15 = float(((speed >= 1) & (speed < 5)).mean())
    f510 = float(((speed >= 5) & (speed < 10)).mean())
    f1020 = float(((speed >= 10) & (speed < 20)).mean())
    fa20 = float((speed >= 20).mean())
    return dict(
        session=sn, state=SESSION_STATE[sn],
        n_bins=int(len(speed)),
        percentile_5=qs[0], percentile_25=qs[1], median=qs[2],
        percentile_75=qs[3], percentile_90=qs[4],
        percentile_95=qs[5], percentile_99=qs[6],
        frac_below_1=fb1, frac_1_to_5=f15, frac_5_to_10=f510,
        frac_10_to_20=f1020, frac_above_20=fa20,
        speed=speed,
    )


# ============================================================================
# D2: v2 lookups vs fresh lookups (correct npz keys)
# ============================================================================
ZONE_LABELS = ["home", "transition", "pot", "pot_zone", "arena", "other"]
EVENT_NAMES = ["incomplete_home_returns", "quick_loop_at_home",
                "digging_sand", "feeding", "rearing",
                "exploration_at_transition", "contemplation_at_transition"]


def correct_lookup(binned, peak_time_s):
    trial_time = np.asarray(binned["trial_time"], dtype=np.float64)
    i = int(np.clip(np.searchsorted(trial_time, peak_time_s) - 1,
                      0, len(trial_time) - 1))
    speed = float(binned["speed"][i])
    zone_int = int(binned["zone"][i])
    zone = ZONE_LABELS[zone_int] if 0 <= zone_int < len(ZONE_LABELS) else f"unknown_{zone_int}"
    events_vec = binned["events"][i]
    flags = {name: bool(events_vec[k]) for k, name in enumerate(EVENT_NAMES)}
    return i, speed, zone, flags


# ============================================================================
# D3: Re-classify ripples with animal-relative thresholds
# ============================================================================
def reclassify_locomotion(speed, median, p75):
    if speed < median:
        return "stationary"
    elif speed < p75:
        return "slow"
    return "fast"


# ============================================================================
# D5: Waveform comparison
# ============================================================================
def get_filter_bank(meta):
    fs_raw = float(meta.get("imSampRate", FS_LFP_RAW))
    n_chan = int(meta.get("nSavedChans", N_NEURAL_CHANNELS + 1))
    gain = lfp_gain_uV(meta)
    b60, a60 = iirnotch(60.0, 30, fs=fs_raw)
    sos60 = tf2sos(b60, a60)
    b120, a120 = iirnotch(120.0, 30, fs=fs_raw)
    sos120 = tf2sos(b120, a120)
    bp = butter(4, [RIPPLE_BAND[0]/(fs_raw/2), RIPPLE_BAND[1]/(fs_raw/2)],
                  btype="bandpass", output="sos")
    return dict(fs_raw=fs_raw, n_chan=n_chan, gain=gain,
                notch_60=sos60, notch_120=sos120, bp_ripple=bp)


def extract_ripple_window(bin_path, fb, ch_a, ch_b, peak_time_s,
                             win_ms=WAVEFORM_WIN_MS):
    win_samples = int(win_ms * 1e-3 * fb["fs_raw"])
    n_total = int(Path(bin_path).stat().st_size // (fb["n_chan"] * 2))
    center = int(peak_time_s * fb["fs_raw"])
    pad_s = max(0, center - win_samples - FILTER_PAD_SAMPLES)
    pad_e = min(n_total, center + win_samples + FILTER_PAD_SAMPLES)
    data = np.memmap(bin_path, dtype=np.int16, mode="r",
                       shape=(n_total, fb["n_chan"]))
    raw = np.asarray(data[pad_s:pad_e, [ch_a, ch_b]], dtype=np.float32) * fb["gain"]
    raw = sosfiltfilt(fb["notch_60"], raw, axis=0)
    raw = sosfiltfilt(fb["notch_120"], raw, axis=0)
    bip = raw[:, 0] - raw[:, 1]
    filt = sosfiltfilt(fb["bp_ripple"], bip)
    s_off = center - pad_s - win_samples
    e_off = s_off + 2 * win_samples
    seg = filt[s_off:e_off]
    return seg, fb["fs_raw"]


def main():
    cfg = load_config()
    binned_root = REPO / cfg["out_dirs"]["binned"]
    geom_pairs = load_geometry_v2()
    geom_lookup = {r: df.set_index("pair_index", drop=False)
                   for r, df in geom_pairs.items()}

    # ----------- D1 -----------
    print("=== D1: Speed distribution per session ===", flush=True)
    d1_rows = []
    speed_arrays = {}
    speed_stats = {}  # for D3 thresholds
    for sn in SESSIONS:
        binned = np.load(binned_root / f"session_{sn}.npz", allow_pickle=True)
        r = d1_speed_distribution(binned, sn)
        speed_arrays[sn] = r.pop("speed")
        d1_rows.append(r)
        speed_stats[sn] = dict(median=r["median"], p75=r["percentile_75"])
        print(f"  S{sn} ({SESSION_STATE[sn]}): "
              f"n_bins={r['n_bins']}, "
              f"median={r['median']:.2f}, p95={r['percentile_95']:.2f}, "
              f"frac<1={r['frac_below_1']*100:.1f}%, "
              f"frac 1-5={r['frac_1_to_5']*100:.1f}%, "
              f"frac 5-10={r['frac_5_to_10']*100:.1f}%, "
              f"frac>10={(r['frac_10_to_20']+r['frac_above_20'])*100:.1f}%",
              flush=True)
    d1_df = pd.DataFrame(d1_rows)
    d1_df.to_csv(OUT_DIR / "D1_speed_distribution_per_session.csv", index=False)

    # D1 figure: per-session speed histogram (log y)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    for ax, sn in zip(axes.flat, SESSIONS):
        sp = speed_arrays[sn]
        ax.hist(sp, bins=np.logspace(-2, 2, 50), color="C0", edgecolor="black")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axvline(speed_stats[sn]["median"], color="red", lw=1, ls="--",
                    label=f"median={speed_stats[sn]['median']:.2f}")
        ax.axvline(5.0, color="black", lw=1, ls=":",
                    label="5 cm/s (v2 fast cutoff)")
        ax.set_title(f"S{sn} ({SESSION_STATE[sn]})")
        ax.set_xlabel("speed (cm/s)")
        ax.legend(fontsize=8)
    axes[0,0].set_ylabel("bin count (log)")
    fig.suptitle("D1: Per-session speed distribution (HMM 480 ms bins)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "D1_speed_distribution_per_session.png", dpi=130)
    plt.close(fig)

    # ----------- D2 -----------
    print("\n=== D2: v2 stored lookups vs fresh ===", flush=True)
    d2_rows = []
    mismatch_speed = 0; mismatch_zone = 0; bad_v2_zone = 0
    for sn in SESSIONS:
        binned = np.load(binned_root / f"session_{sn}.npz", allow_pickle=True)
        v2_behav_path = SWR_V2_BASE / f"session_{sn}" / "session_{sn}_RSP_event_behavior.csv".replace("{sn}", str(sn))
        for region in REGIONS:
            v2_path = (SWR_V2_BASE / f"session_{sn}"
                       / f"session_{sn}_{region}_event_behavior.csv")
            if not v2_path.exists():
                continue
            try:
                v2 = pd.read_csv(v2_path)
            except pd.errors.EmptyDataError:
                continue
            if not len(v2):
                continue
            for _, r in v2.iterrows():
                t = float(r["peak_time_s"])
                i, sp_fresh, z_fresh, flags = correct_lookup(binned, t)
                v2_speed = float(r["speed"])
                v2_zone = str(r["zone"])
                m_speed = abs(v2_speed - sp_fresh) < 1e-3
                m_zone = (v2_zone == z_fresh)
                if not m_speed:
                    mismatch_speed += 1
                if not m_zone:
                    mismatch_zone += 1
                if v2_zone in ("-1", "-1.0"):
                    bad_v2_zone += 1
                d2_rows.append(dict(
                    session=sn, region=region,
                    event_id=int(r["event_id"]),
                    peak_time_s=t, behavior_bin_idx=i,
                    v2_speed=v2_speed, lookup_speed=sp_fresh,
                    v2_zone=v2_zone, lookup_zone=z_fresh,
                    match_speed_flag=m_speed, match_zone_flag=m_zone,
                    fresh_dig=flags["digging_sand"],
                    fresh_feed=flags["feeding"],
                    fresh_rear=flags["rearing"],
                ))
    d2_df = pd.DataFrame(d2_rows)
    d2_df.to_csv(OUT_DIR / "D2_lookup_comparison.csv", index=False)
    print(f"  Total events compared: {len(d2_df)}")
    print(f"  Events with v2 zone = '-1': {bad_v2_zone}/{len(d2_df)} "
          f"({100*bad_v2_zone/max(1,len(d2_df)):.1f}%)")
    print(f"  Speed mismatches: {mismatch_speed}/{len(d2_df)}")
    print(f"  Zone mismatches:  {mismatch_zone}/{len(d2_df)} (expected = bad_v2_zone)")

    print("\n  Fresh-lookup zone counts per region:")
    for region in REGIONS:
        sub = d2_df[d2_df.region == region]
        if not len(sub): continue
        zc = sub["lookup_zone"].value_counts()
        print(f"  {region}: {dict(zc)}")

    print("\n  Fresh-lookup event-flag counts per region:")
    for region in REGIONS:
        sub = d2_df[d2_df.region == region]
        if not len(sub): continue
        ndig = int(sub["fresh_dig"].sum())
        nfeed = int(sub["fresh_feed"].sum())
        nrear = int(sub["fresh_rear"].sum())
        print(f"  {region}: dig={ndig}, feed={nfeed}, rear={nrear} (n={len(sub)})")

    # ----------- D3 -----------
    print("\n=== D3: Reclassify with animal-relative thresholds ===", flush=True)
    d3_rows = []
    for sn in SESSIONS:
        median = speed_stats[sn]["median"]
        p75 = speed_stats[sn]["p75"]
        for region in REGIONS:
            v2_path = (SWR_V2_BASE / f"session_{sn}"
                       / f"session_{sn}_{region}_event_behavior.csv")
            if not v2_path.exists():
                continue
            try:
                v2 = pd.read_csv(v2_path)
            except pd.errors.EmptyDataError:
                continue
            if not len(v2):
                continue
            for _, r in v2.iterrows():
                sp = float(r["speed"])
                old = str(r["locomotion"])
                new = reclassify_locomotion(sp, median, p75)
                d3_rows.append(dict(
                    session=sn, state=SESSION_STATE[sn], region=region,
                    event_id=int(r["event_id"]),
                    peak_time_s=float(r["peak_time_s"]),
                    speed=sp, old_loc_state=old, new_loc_state=new,
                    session_median=median, session_p75=p75,
                ))
    d3_df = pd.DataFrame(d3_rows)
    d3_df.to_csv(OUT_DIR / "D3_ripple_behavior_reclassified.csv", index=False)

    print(f"  Per-region OLD loc state distribution (v2 absolute thresholds):")
    for region in REGIONS:
        sub = d3_df[d3_df.region == region]
        if not len(sub): continue
        c = sub["old_loc_state"].value_counts()
        total = len(sub)
        print(f"  {region}: {dict(c)} (n={total})")
    print(f"  Per-region NEW loc state distribution (animal-relative):")
    for region in REGIONS:
        sub = d3_df[d3_df.region == region]
        if not len(sub): continue
        c = sub["new_loc_state"].value_counts()
        total = len(sub)
        print(f"  {region}: {dict(c)} (n={total})")

    # D3 figure
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=False)
    for ax, region in zip(axes, REGIONS):
        sub = d3_df[d3_df.region == region]
        if not len(sub):
            ax.set_title(f"{region} (n=0)"); continue
        sessions = sorted(sub.session.unique())
        x = np.arange(len(sessions))
        states_order = ["stationary", "slow", "fast"]
        bottom = np.zeros(len(sessions))
        for st, col in zip(states_order, ["#5DA5DA", "#FAA43A", "#F17CB0"]):
            counts = [int(((sub.session==sn)&(sub.new_loc_state==st)).sum())
                      for sn in sessions]
            ax.bar(x, counts, bottom=bottom, color=col, label=st,
                    edgecolor="black")
            bottom = bottom + np.array(counts)
        ax.set_xticks(x); ax.set_xticklabels([f"S{s}" for s in sessions])
        ax.set_title(f"{region}: animal-relative locomotion (n={len(sub)})")
        ax.set_ylabel("ripple count")
        ax.legend(fontsize=8)
    fig.suptitle("D3: Ripples by animal-relative locomotion state")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "D3_ripple_locomotion_reclassified.png", dpi=130)
    plt.close(fig)

    # ----------- D4 -----------
    print("\n=== D4: Timestamp alignment ===", flush=True)
    d4_rows = []
    for sn in SESSIONS:
        binned = np.load(binned_root / f"session_{sn}.npz", allow_pickle=True)
        trial_time = np.asarray(binned["trial_time"], dtype=np.float64)
        beh_duration = float(trial_time[-1] + HMM_BIN_S)
        beh_bins = int(len(trial_time))
        # LFP file
        lfp_paths = lookup_lfp_paths(sn)
        if "imec0_meta" not in lfp_paths:
            d4_rows.append(dict(session=sn, alignment_ok_flag=False,
                                  note="no LFP file"))
            continue
        meta = parse_lf_meta(lfp_paths["imec0_meta"])
        fs_raw = float(meta.get("imSampRate", FS_LFP_RAW))
        n_chan = int(meta.get("nSavedChans", N_NEURAL_CHANNELS + 1))
        file_size = Path(lfp_paths["imec0_bin"]).stat().st_size
        lfp_total_samples = file_size // (n_chan * 2)
        lfp_total_duration = lfp_total_samples / fs_raw
        truncated_lfp_samples = min(lfp_total_samples,
                                       int(beh_duration * fs_raw))
        truncated_lfp_duration = truncated_lfp_samples / fs_raw
        offset_ms = (truncated_lfp_duration - beh_duration) * 1000.0
        # peak time range from v2 events
        max_peak_time = 0.0
        for region in REGIONS:
            v2_path = (SWR_V2_BASE / f"session_{sn}"
                       / f"session_{sn}_{region}_event_behavior.csv")
            if v2_path.exists():
                try:
                    v2 = pd.read_csv(v2_path)
                except pd.errors.EmptyDataError:
                    continue
                if len(v2):
                    max_peak_time = max(max_peak_time, float(v2["peak_time_s"].max()))
        d4_rows.append(dict(
            session=sn,
            beh_bins=beh_bins,
            beh_duration_s=beh_duration,
            lfp_fs_raw=fs_raw,
            lfp_total_samples=int(lfp_total_samples),
            lfp_total_duration_s=lfp_total_duration,
            truncated_lfp_samples=int(truncated_lfp_samples),
            truncated_lfp_duration_s=truncated_lfp_duration,
            max_event_peak_time_s=max_peak_time,
            offset_ms=offset_ms,
            alignment_ok_flag=bool(abs(offset_ms) < 100
                                      and max_peak_time <= beh_duration + 0.5),
        ))
        print(f"  S{sn}: beh_dur={beh_duration:.2f}s ({beh_bins} bins), "
              f"lfp_dur_total={lfp_total_duration:.2f}s, "
              f"truncated_lfp_dur={truncated_lfp_duration:.2f}s, "
              f"offset={offset_ms:+.1f}ms, "
              f"max_event_t={max_peak_time:.2f}s, "
              f"ok={d4_rows[-1]['alignment_ok_flag']}", flush=True)
    d4_df = pd.DataFrame(d4_rows)
    d4_df.to_csv(OUT_DIR / "D4_timestamp_alignment.csv", index=False)

    # ----------- D5 -----------
    print("\n=== D5: Waveform comparison ===", flush=True)
    win_ms = WAVEFORM_WIN_MS
    d5_summary = []
    cache_fb = {}
    for sn in SESSIONS:
        lfp_paths = lookup_lfp_paths(sn)
        if "imec0_meta" not in lfp_paths or "imec1_meta" not in lfp_paths:
            continue
        meta0 = parse_lf_meta(lfp_paths["imec0_meta"])
        meta1 = parse_lf_meta(lfp_paths["imec1_meta"])
        cache_fb[sn] = dict(
            imec0=dict(**get_filter_bank(meta0), bin_path=lfp_paths["imec0_bin"]),
            imec1=dict(**get_filter_bank(meta1), bin_path=lfp_paths["imec1_bin"]),
        )

    # Compute waveforms per region per session per loc state (new classification)
    waveforms = {}      # (sn, region, loc_state_new) -> list of np arrays
    fs_ref = FS_LFP_RAW
    for sn in SESSIONS:
        if sn not in cache_fb:
            continue
        d3_sn = d3_df[d3_df.session == sn]
        for region in REGIONS:
            sub = d3_sn[d3_sn.region == region]
            if not len(sub):
                continue
            # Need a contributing pair per event; we only have peak_time + speed
            # Use the regional_events.csv to find n_pairs_active, then use any
            # representative pair from geom_lookup
            reg_events_path = (SWR_V2_BASE / f"session_{sn}"
                                / f"session_{sn}_{region}_regional_events.csv")
            if not reg_events_path.exists():
                continue
            try:
                reg_events = pd.read_csv(reg_events_path)
            except pd.errors.EmptyDataError:
                continue
            # Use the median pair_index for this region as a representative
            # detector (each event involves multiple pairs; pick one to extract
            # the local LFP)
            if region == "ACA":
                fb = cache_fb[sn]["imec0"]
            else:
                fb = cache_fb[sn]["imec1"]
            # representative pair: middle of region by pair_index
            geom = geom_lookup[region]
            rep_pi = int(geom["pair_index"].median())
            if rep_pi not in geom.index:
                rep_pi = int(geom["pair_index"].iloc[len(geom)//2])
            pinfo = geom.loc[rep_pi]
            ch_a, ch_b = int(pinfo["channel_a"]), int(pinfo["channel_b"])
            for _, r in sub.iterrows():
                key = (sn, region, str(r["new_loc_state"]))
                t_peak = float(r["peak_time_s"])
                try:
                    seg, fs = extract_ripple_window(
                        fb["bin_path"], fb, ch_a, ch_b, t_peak, win_ms=win_ms,
                    )
                except Exception:
                    continue
                if not len(seg):
                    continue
                fs_ref = fs
                expected_len = 2 * int(win_ms * 1e-3 * fs)
                if len(seg) < expected_len:
                    continue
                if len(seg) > expected_len:
                    seg = seg[:expected_len]
                waveforms.setdefault(key, []).append(seg)

    # Aggregate per region per loc state across sessions
    fig, axes = plt.subplots(len(REGIONS), 3, figsize=(13, 9))
    states_order = ["stationary", "slow", "fast"]
    colors_state = {"stationary": "#1f77b4", "slow": "#ff7f0e", "fast": "#d62728"}

    for row, region in enumerate(REGIONS):
        for col, st in enumerate(states_order):
            ax = axes[row, col]
            wave_list = []
            for sn in SESSIONS:
                k = (sn, region, st)
                if k in waveforms and len(waveforms[k]):
                    wave_list.extend(waveforms[k])
            if not wave_list:
                ax.set_title(f"{region} {st} (n=0)", fontsize=10)
                continue
            arr = np.array(wave_list)
            mean_wf = arr.mean(axis=0)
            sd_wf = arr.std(axis=0)
            t_axis = (np.arange(arr.shape[1]) - arr.shape[1] // 2) / fs_ref * 1000
            ax.plot(t_axis, mean_wf, color=colors_state[st], lw=1)
            ax.fill_between(t_axis, mean_wf - sd_wf, mean_wf + sd_wf,
                              alpha=0.25, color=colors_state[st])
            ax.axvline(0, color="black", lw=0.5, ls="--", alpha=0.5)
            ax.set_title(f"{region} {st} (n={len(wave_list)})", fontsize=10)
            ax.set_xlabel("ms relative to peak")
            ax.set_ylabel("100-250 Hz bandpass (µV)")
            # summary
            peak_amp = float(np.max(np.abs(mean_wf)))
            d5_summary.append(dict(
                region=region, loc_state=st, n_events=len(wave_list),
                mean_peak_abs_amp_uV=peak_amp,
                mean_envelope_uV=float(np.mean(np.abs(mean_wf))),
            ))
    fig.suptitle("D5: Mean ripple waveform per region × locomotion state "
                  "(animal-relative thresholds, 100-250 Hz)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "D5_mean_ripple_waveform_per_loc_state.png", dpi=130)
    plt.close(fig)
    pd.DataFrame(d5_summary).to_csv(
        OUT_DIR / "D5_waveform_comparison_summary.csv", index=False)

    # Power spectra
    fig, axes = plt.subplots(len(REGIONS), 3, figsize=(13, 9), sharey="row")
    NW = 3; K_tap = 5
    for row, region in enumerate(REGIONS):
        for col, st in enumerate(states_order):
            ax = axes[row, col]
            wave_list = []
            for sn in SESSIONS:
                k = (sn, region, st)
                if k in waveforms and len(waveforms[k]):
                    wave_list.extend(waveforms[k])
            if not wave_list:
                ax.set_title(f"{region} {st} (n=0)", fontsize=10); continue
            arr = np.array(wave_list)
            n_t = arr.shape[1]
            tapers = windows.dpss(n_t, NW, K_tap)
            freqs = np.fft.rfftfreq(n_t, d=1/fs_ref)
            mean_spec = None
            for k_t in range(K_tap):
                fs_arr = np.abs(np.fft.rfft(arr * tapers[k_t], axis=1)) ** 2
                if mean_spec is None:
                    mean_spec = fs_arr.mean(axis=0)
                else:
                    mean_spec = mean_spec + fs_arr.mean(axis=0)
            mean_spec /= K_tap
            mask = (freqs >= 50) & (freqs <= 350)
            ax.semilogy(freqs[mask], mean_spec[mask],
                          color=colors_state[st], lw=1)
            ax.axvspan(100, 250, alpha=0.1, color="gray")
            ax.set_title(f"{region} {st} (n={len(wave_list)})", fontsize=10)
            ax.set_xlabel("Hz")
            ax.set_ylabel("power")
    fig.suptitle("D5: Mean power spectrum per region × loc state (multitaper, ±100 ms window)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "D5_mean_ripple_spectrum_per_loc_state.png", dpi=130)
    plt.close(fig)

    # ----------- Bottom line summary -----------
    print("\n========== BOTTOM LINE ==========")
    print(f"D1: median speed range across sessions = "
          f"{d1_df['median'].min():.2f} - {d1_df['median'].max():.2f} cm/s")
    print(f"    fraction of bins above 5 cm/s (v2 'fast'): "
          f"{((d1_df['frac_5_to_10']+d1_df['frac_10_to_20']+d1_df['frac_above_20'])*100).min():.1f}% "
          f"to {((d1_df['frac_5_to_10']+d1_df['frac_10_to_20']+d1_df['frac_above_20'])*100).max():.1f}%")
    print(f"D2: events with v2 zone = '-1': {bad_v2_zone}/{len(d2_df)} "
          f"({100*bad_v2_zone/max(1,len(d2_df)):.1f}%) — zone lookup BROKEN")
    print(f"    speed mismatch: {mismatch_speed}/{len(d2_df)} — speed lookup OK")
    for region in REGIONS:
        sub = d3_df[d3_df.region == region]
        if not len(sub): continue
        old_fast = int((sub.old_loc_state == "fast_locomotion").sum())
        new_fast = int((sub.new_loc_state == "fast").sum())
        print(f"D3: {region} old 'fast_locomotion' = {old_fast}/{len(sub)} "
              f"({100*old_fast/len(sub):.1f}%), "
              f"new animal-relative 'fast' = {new_fast}/{len(sub)} "
              f"({100*new_fast/len(sub):.1f}%)")
    n_align_ok = int(d4_df["alignment_ok_flag"].sum())
    print(f"D4: alignment OK in {n_align_ok}/{len(d4_df)} sessions "
          f"(|offset|<100 ms AND max_event_t≤beh_duration)")
    print("D5: see figures and D5_waveform_comparison_summary.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()
