"""
Shared helpers for Stage 1 dynamics pipeline.

Conventions
-----------
- Neural data: 50ms bins, Gaussian sigma=1 smoothing, z-scored per unit
  (handled inside dp_cycles_lib.load_neural).
- Speed: ||N(t+1) - N(t)|| in full neural state space, then Gaussian smoothed
  with sigma=SPEED_SIGMA=3 bins (150ms). Rationale: 3 bins is a midrange of the
  plan's 2-4 suggestion; 2 leaves visible single-bin spikes, 4 starts smearing
  sub-second transitions.
- Curvature: 1 - cos(theta) between consecutive velocity vectors, same smoothing.
- Entropy: 60s sliding window, 10s step (matches existing infrastructure).
- Inflections: scipy.signal.find_peaks on Gaussian-smoothed entropy
  (sigma=ENTROPY_SMOOTH_SIGMA=3 entropy steps = 30s). Prominence=0.3 (matches
  prior MIN_AMPLITUDE), distance=6 entropy steps = 60s.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from collections import Counter
from scipy.stats import entropy as sp_entropy
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis" / "cycles_partD"))

from dp_cycles_lib import load_neural, load_behavior, K_PCS

# ---- Parameters (single source of truth across all 24 sessions) ----
BIN_S = 0.05
SPEED_SIGMA = 3            # bins, 150 ms
CURV_SIGMA = 3             # bins, 150 ms
ENTROPY_WINDOW_SEC = 60
ENTROPY_STEP_SEC = 10
ENTROPY_SMOOTH_SIGMA = 3   # entropy steps (30 s)
INFLECTION_PROMINENCE = 0.3
INFLECTION_DISTANCE_SEC = 60.0
PEAK_WIN_BINS = 5          # +/- 5 neural bins (250 ms) around entropy inflection

# Fixed cross-condition behavioral repertoire. Any behavior not present in a
# given session (column missing from xlsx, or <50 events at load-time threshold)
# is reported as 0.0 fraction so the per-session csv has the same columns.
TARGET_BINARY_BEHAVIORS = [
    'feeding',
    'digging_sand',
    'incomplete_home_returns',
    'quick_one_loop_at_home',
    'transition_wall_exploration',
]


def filter_behavior(behav):
    """Drop lever-zone columns and any other paradigm-specific covariates.

    Lever-zone scored data is excluded for HFD (and uniformly applied to all
    sessions to keep behavior keys aligned). The columns are present in every
    session's xlsx because they are part of the EthoVision template, not
    paradigm-specific markers.
    """
    return {k: v for k, v in behav.items() if 'lever' not in k.lower()}


# ---- Session metadata ----
def list_sessions():
    """Return sorted list of session_num values that have ACA + LHA + behavior."""
    with open(REPO / "paths.yaml") as f:
        cfg = yaml.safe_load(f)
    sessions = cfg["double_probe"]["coordinates_1"]["mouse01"]["sessions"]
    out = []
    for k, v in sessions.items():
        snum = int(k.split('_')[1])
        p0 = v.get('probe_0_aca', {}).get('sorted')
        p1 = v.get('probe_1_lha_rsp', {}).get('sorted')
        bh = v.get('behavior')
        if not p0 or not p1 or not bh:
            continue
        if not Path(p0).exists() or not Path(bh).exists():
            continue
        out.append(dict(session=snum,
                        state=v['state'],
                        phase=v['phase'],
                        behavior=bh))
    out.sort(key=lambda r: r['session'])
    return out


# ---- Behavioral entropy ----
def _zone_priority():
    return [
        'Home corner left', 'Home corner right', 'Central Arena Zone',
        'Foraging arena', 'Home', 'ladder to Arena', 'Transition Zone',
        'Pot-1 zone', 'Pot-2 Zone', 'Pot-3 zone', 'Pot-4 zone',
        'Pot-1', 'Pot-2', 'Pot-3', 'Pot-4',
    ]


def load_zones_and_velocity(behavior_xlsx):
    """Read the dual-probe behavior xlsx, return (time_s, zones_str, velocity_cm_s)."""
    df = pd.read_excel(behavior_xlsx, header=None)
    col_names = df.iloc[34].tolist()
    data = df.iloc[36:].reset_index(drop=True)
    data.columns = col_names

    time_vals = pd.to_numeric(data['Recording time'], errors='coerce').values
    vel = pd.to_numeric(data['Velocity(Center-point)'], errors='coerce').values
    vel = np.nan_to_num(vel, nan=0.0)
    vel = np.clip(vel, 0, None)

    zones = np.full(len(time_vals), 'O', dtype=object)
    for zname in _zone_priority():
        col_match = [c for c in col_names if isinstance(c, str)
                     and c.startswith('Zone(') and zname in c]
        if col_match:
            mask = pd.to_numeric(data[col_match[0]], errors='coerce').values == 1
            zones[mask] = zname
    return time_vals.astype(float), zones, vel


def compute_entropy_series(zones, time_vals, vel,
                           window_sec=ENTROPY_WINDOW_SEC,
                           step_sec=ENTROPY_STEP_SEC):
    """Causal entropy assigned to end of window. Returns (ent_t, ent_v, vel_means)."""
    dt = np.median(np.diff(time_vals))
    window_bins = int(window_sec / dt)
    step_bins = int(step_sec / dt)
    ent_t, ent_v, vel_m = [], [], []
    for s in range(0, len(zones) - window_bins, step_bins):
        wz = zones[s:s + window_bins]
        transitions = [f"{wz[j-1]}->{wz[j]}" for j in range(1, len(wz))
                       if wz[j] != wz[j - 1]]
        if len(transitions) < 3:
            continue
        counts = Counter(transitions)
        probs = np.array(list(counts.values()), dtype=float)
        probs /= probs.sum()
        ent_t.append(time_vals[s + window_bins - 1])
        ent_v.append(sp_entropy(probs, base=2))
        vel_m.append(np.nanmean(vel[s:s + window_bins]))
    return np.array(ent_t), np.array(ent_v), np.array(vel_m)


def detect_inflections(ent_t, ent_v,
                       smooth_sigma=ENTROPY_SMOOTH_SIGMA,
                       prominence=INFLECTION_PROMINENCE,
                       distance_sec=INFLECTION_DISTANCE_SEC,
                       step_sec=ENTROPY_STEP_SEC):
    """Find peaks/troughs on smoothed entropy. Returns (peak_idx, trough_idx, smoothed)."""
    smoothed = gaussian_filter1d(ent_v, sigma=smooth_sigma)
    distance_bins = max(1, int(distance_sec / step_sec))
    peaks, _ = find_peaks(smoothed, prominence=prominence, distance=distance_bins)
    troughs, _ = find_peaks(-smoothed, prominence=prominence, distance=distance_bins)
    return peaks, troughs, smoothed


# ---- Trajectory dynamics ----
def compute_speed(matrix, sigma=SPEED_SIGMA):
    """Speed[t] = ||N(t+1) - N(t)|| in full neural state space, then smoothed.
    Returns array of length T-1.
    """
    diff = np.diff(matrix, axis=0)
    speed = np.linalg.norm(diff, axis=1)
    return gaussian_filter1d(speed, sigma=sigma)


def compute_curvature(matrix, sigma=CURV_SIGMA, eps=1e-9):
    """Curvature[t] = 1 - cos(theta_t) between v_t = N(t+1)-N(t) and v_{t+1}.
    Returns array of length T-2.
    """
    diff = np.diff(matrix, axis=0)
    norms = np.linalg.norm(diff, axis=1)
    v_a = diff[:-1]
    v_b = diff[1:]
    n_a = norms[:-1]
    n_b = norms[1:]
    cos_t = np.einsum('ij,ij->i', v_a, v_b) / np.maximum(n_a * n_b, eps)
    cos_t = np.clip(cos_t, -1.0, 1.0)
    curv = 1.0 - cos_t
    return gaussian_filter1d(curv, sigma=sigma)


# ---- Phase definition ----
def define_phases(ent_t, peak_idx, trough_idx, n_total_bins, bin_s=BIN_S,
                  peak_win_bins=PEAK_WIN_BINS):
    """Given entropy inflection indices, return ordered list of phase dicts.

    Each rising/falling phase spans the time between consecutive inflections.
    Each peak/trough is a window of +/-peak_win_bins NEURAL bins around the
    inflection time.

    All bin indices below are in NEURAL bin (50ms) space, clipped to
    [0, n_total_bins).
    """
    inflections = []
    for i in peak_idx:
        inflections.append(('peak', float(ent_t[i])))
    for i in trough_idx:
        inflections.append(('trough', float(ent_t[i])))
    inflections.sort(key=lambda x: x[1])

    phases = []
    for i, (typ, t) in enumerate(inflections):
        nb = int(round(t / bin_s))
        sb = max(0, nb - peak_win_bins)
        eb = min(n_total_bins - 1, nb + peak_win_bins)
        phases.append(dict(
            phase_type=typ,
            start_bin=sb, end_bin=eb,
            inflection_t=t, inflection_bin=nb,
            duration_s=float((eb - sb) * bin_s),
        ))

    for i in range(len(inflections) - 1):
        t_start, t_end = inflections[i][1], inflections[i + 1][1]
        b_start = int(round(t_start / bin_s))
        b_end = int(round(t_end / bin_s))
        b_start = max(0, b_start)
        b_end = min(n_total_bins - 1, b_end)
        if b_end <= b_start:
            continue
        if inflections[i][0] == 'trough' and inflections[i + 1][0] == 'peak':
            ptype = 'rising'
        elif inflections[i][0] == 'peak' and inflections[i + 1][0] == 'trough':
            ptype = 'falling'
        else:
            ptype = 'mixed'
        phases.append(dict(
            phase_type=ptype,
            start_bin=b_start, end_bin=b_end,
            inflection_t=None, inflection_bin=None,
            duration_s=float((b_end - b_start) * bin_s),
        ))
    phases.sort(key=lambda p: p['start_bin'])
    for k, p in enumerate(phases):
        p['phase_id'] = k
    return phases


# ---- Phase summary ----
def fraction_in_class(values_in_phase, classes):
    out = {}
    n = len(values_in_phase)
    if n == 0:
        return {c: 0.0 for c in classes}
    for c in classes:
        out[c] = float(np.mean(values_in_phase == c))
    return out


def transitions_count(values_in_phase):
    return int(np.sum(values_in_phase[1:] != values_in_phase[:-1]))


def phase_summary_row(phase, signals, behav, region_speeds, region_curvs,
                       region_fr, region_pc1, entropy_interp):
    sb, eb = phase['start_bin'], phase['end_bin']
    dur = eb - sb
    if dur < 1:
        return None

    row = dict(
        phase_id=phase['phase_id'],
        phase_type=phase['phase_type'],
        start_bin=sb, end_bin=eb, duration_s=phase['duration_s'],
    )
    # speed/curvature (length T-1, T-2): clamp slice safely
    for region, sp in region_speeds.items():
        seg = sp[sb:min(eb, len(sp))]
        if seg.size:
            row[f'mean_speed_{region}'] = float(np.mean(seg))
            row[f'peak_speed_{region}'] = float(np.max(seg))
            row[f'time_to_peak_speed_{region}_norm'] = float(np.argmax(seg) / max(len(seg) - 1, 1))
        else:
            row[f'mean_speed_{region}'] = np.nan
            row[f'peak_speed_{region}'] = np.nan
            row[f'time_to_peak_speed_{region}_norm'] = np.nan
    for region, cv in region_curvs.items():
        seg = cv[sb:min(eb, len(cv))]
        if seg.size:
            row[f'mean_curv_{region}'] = float(np.mean(seg))
            row[f'peak_curv_{region}'] = float(np.max(seg))
            row[f'time_to_peak_curv_{region}_norm'] = float(np.argmax(seg) / max(len(seg) - 1, 1))
        else:
            row[f'mean_curv_{region}'] = np.nan
            row[f'peak_curv_{region}'] = np.nan
            row[f'time_to_peak_curv_{region}_norm'] = np.nan
    for region, fr in region_fr.items():
        seg = fr[sb:eb]
        row[f'mean_fr_{region}'] = float(np.mean(seg)) if seg.size else np.nan
    for region, pc in region_pc1.items():
        seg = pc[sb:eb]
        row[f'mean_pc1_{region}'] = float(np.mean(seg)) if seg.size else np.nan
    seg_e = entropy_interp[sb:eb]
    row['mean_entropy'] = float(np.mean(seg_e)) if seg_e.size else np.nan

    # Behavioral covariates
    comp = behav['compartment']['values'][sb:eb]
    row['compart_transitions'] = transitions_count(comp)
    comp_frac = fraction_in_class(comp, behav['compartment']['classes'])
    for c, v in comp_frac.items():
        row[f'frac_{c}'] = v
    row['dominant_compartment'] = max(comp_frac, key=comp_frac.get)

    # Continuous behavioral means
    if 'velocity' in behav:
        v = behav['velocity']['values'][sb:eb]
        row['mean_velocity'] = float(np.nanmean(v)) if v.size else np.nan
    pot_keys = [k for k in behav.keys() if k.startswith('dist_pot-') and 'zone' not in k]
    if pot_keys:
        stack = np.stack([behav[k]['values'][sb:eb].astype(float) for k in pot_keys])
        nearest = np.nanmin(stack, axis=0)
        row['mean_dist_nearest_pot'] = float(np.nanmean(nearest)) if nearest.size else np.nan

    # Fixed-set binary behaviors. Absent behaviors (column missing in this session)
    # report frac=0.0 so the csv has the same columns across all 20 sessions.
    best = ('none', 0.0)
    for k in TARGET_BINARY_BEHAVIORS:
        if k in behav and behav[k].get('type') == 'categorical':
            f = float(np.mean(behav[k]['values'][sb:eb] == '1'))
        else:
            f = 0.0
        row[f'frac_{k}'] = f
        if f > best[1]:
            best = (k, f)
    row['dominant_action'] = best[0] if best[1] > 0.05 else 'none'
    row['dominant_action_frac'] = best[1]
    return row


# ---- Helpers ----
def interp_entropy_to_bins(ent_t, smoothed, bin_centers):
    """Linear-interp entropy to neural bin centers; extrapolate edges with nearest."""
    if len(ent_t) < 2:
        return np.full_like(bin_centers, fill_value=np.nan, dtype=float)
    out = np.interp(bin_centers, ent_t, smoothed,
                    left=smoothed[0], right=smoothed[-1])
    return out
