"""Single-probe Mouse01-Coordinates1 library — port of dp_cycles_lib + stage1_lib.

Same neural preprocessing (50 ms bins, sigma=1 Gaussian smoothing, per-unit z-score).
LHA and RSP are split by depth on a single Neuropixels 2.0 probe:
    LHA: depth < 1300 um, RSP: depth >= 1300 um.
Filter: KSLabel='good' AND fr > 0.3 Hz AND Amplitude > 48 uV.

Behavior CSV is wide-format (variables in column 0, 100 ms time bins in columns 1+).
Zone names differ from the dual-probe paradigm — see _zone_priority below.
"""
from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import gaussian_filter1d
from scipy.stats import entropy as sp_entropy
from scipy.signal import find_peaks
import spikeinterface.extractors as se

REPO = Path(__file__).resolve().parent.parent.parent

# ---- Neural preprocessing (mirrors dp_cycles_lib) ----
BIN_MS = 50.0
SMOOTH_SIGMA = 1.0
LHA_DEPTH_MAX = 1300   # um, LHA is below this
RSP_DEPTH_MIN = 1300   # um, RSP is at or above this
MIN_FR = 0.3           # Hz
MIN_AMP = 48.0         # uV (Amplitude column in cluster_info.tsv)

# ---- Stage 1 dynamics constants (mirrors stage1_lib) ----
BIN_S = BIN_MS / 1000.0
SPEED_SIGMA = 3            # bins, 150 ms
CURV_SIGMA = 3
ENTROPY_WINDOW_SEC = 60
ENTROPY_STEP_SEC = 10
ENTROPY_SMOOTH_SIGMA = 3   # entropy steps (30 s)
INFLECTION_PROMINENCE = 0.3
INFLECTION_DISTANCE_SEC = 60.0
PEAK_WIN_BINS = 5

K_PCS = {'LHA': 5, 'RSP': 10}   # default subspace sizes; LHA smaller (matches dual-probe LHA)

# Fixed binary-behavior repertoire across single-probe sessions. Names use the
# Mouse01-Coor1 EthoVision template wording (different from dual-probe).
TARGET_BINARY_BEHAVIORS = [
    'feeding',
    'digging',
    'incomplete_home_return',
    'quick_one_loop_at_home',
    'transition_wall_exploration',
]


# ============================================================================
# Session metadata
# ============================================================================
def _sessions_cfg():
    with open(REPO / "paths.yaml") as f:
        cfg = yaml.safe_load(f)
    return cfg["single_probe"]["coordinates_1"]["mouse01"]["sessions"]


def list_sessions():
    """Return sorted list of session dicts that have sorted output + behavior CSV."""
    sessions = _sessions_cfg()
    out = []
    for k, v in sessions.items():
        snum = int(k.split('_')[1])
        sp = v.get('sorted')
        bh = v.get('behavior')
        if not sp or not bh:
            continue
        if not Path(sp).exists() or not Path(bh).exists():
            continue
        out.append(dict(session=snum,
                        state=v['state'],
                        phase=v['phase'],
                        sorted=sp,
                        behavior=bh))
    out.sort(key=lambda r: r['session'])
    return out


# ============================================================================
# Neural loading (50 ms binned, sigma=1, z-scored)
# ============================================================================
def _good_units_by_region(sorted_path: Path):
    """Return (lha_ids, rsp_ids) from cluster_info.tsv.

    Filter: KSLabel='good' AND fr > MIN_FR AND Amplitude > MIN_AMP.
    Depth split: <LHA_DEPTH_MAX -> LHA, >=RSP_DEPTH_MIN -> RSP.
    """
    ci = sorted_path / "cluster_info.tsv"
    if not ci.exists():
        raise FileNotFoundError(f"Missing cluster_info.tsv at {sorted_path}")
    df = pd.read_csv(ci, sep='\t')
    if 'depth' not in df.columns:
        raise ValueError(f"depth column missing in {ci}")
    label_col = 'group' if 'group' in df.columns and df['group'].eq('good').any() else 'KSLabel'
    keep = df[label_col] == 'good'
    if 'fr' in df.columns:
        keep &= df['fr'] > MIN_FR
    if 'Amplitude' in df.columns:
        keep &= df['Amplitude'] > MIN_AMP
    good = df[keep]
    lha_ids = good[good['depth'] < LHA_DEPTH_MAX]['cluster_id'].astype(int).values
    rsp_ids = good[good['depth'] >= RSP_DEPTH_MIN]['cluster_id'].astype(int).values
    return lha_ids, rsp_ids


def load_neural(session_num: int, region: str):
    """50 ms bins, Gaussian sigma=1, per-unit z-scored matrix.

    Returns (matrix [T, U], bin_centers [T], n_units).
    """
    sval = _sessions_cfg()[f"session_{session_num}"]
    sp = Path(sval['sorted'])
    lha_ids, rsp_ids = _good_units_by_region(sp)
    if region == 'LHA':
        unit_ids = lha_ids
    elif region == 'RSP':
        unit_ids = rsp_ids
    else:
        raise ValueError(f"region must be LHA or RSP, got {region}")
    if len(unit_ids) == 0:
        raise ValueError(f"S{session_num} {region}: no good units after filtering")

    sorting = se.read_kilosort(sp)
    fs = float(sorting.get_sampling_frequency())
    avail = set(sorting.get_unit_ids())
    unit_ids = np.array([u for u in unit_ids if u in avail])
    if len(unit_ids) == 0:
        raise ValueError(f"S{session_num} {region}: filtered units not in sorting")

    spike_times = {}
    all_max = 0.0
    for uid in unit_ids:
        st_samples = sorting.get_unit_spike_train(int(uid))
        st_sec = st_samples.astype(float) / fs
        spike_times[int(uid)] = st_sec
        if len(st_sec):
            all_max = max(all_max, float(st_sec.max()))
    dur = all_max + 1.0

    dt = BIN_S
    n_bins = int(dur / dt)
    bin_edges = np.arange(0, n_bins + 1) * dt
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    sorted_uids = sorted(spike_times.keys())
    matrix = np.zeros((n_bins, len(sorted_uids)))
    for j, uid in enumerate(sorted_uids):
        counts, _ = np.histogram(spike_times[uid], bins=bin_edges)
        matrix[:, j] = gaussian_filter1d(counts.astype(float), sigma=SMOOTH_SIGMA)
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    stds[stds == 0] = 1.0
    matrix = (matrix - means) / stds
    return matrix, bin_centers, len(sorted_uids)


# ============================================================================
# Behavior loading (wide-format 100 ms binned CSV)
# ============================================================================
def _read_wide_csv(path: Path) -> pd.DataFrame:
    """Read the 100ms-binned wide CSV: row index = variable name, columns = time bins.

    Returns a DataFrame indexed by variable name, columns are 0-indexed time bins.
    """
    raw = pd.read_csv(path, header=None, low_memory=False)
    var_names = raw.iloc[:, 0].values
    body = raw.iloc[:, 1:].copy()
    body.index = var_names
    body = body.apply(lambda s: pd.to_numeric(s, errors='coerce'), axis=1)
    return body


# Canonical zone priority (single-probe Mouse01-Coor1 EthoVision template).
# Used to assign a single zone label to each behavioral time-bin for entropy.
_SP_ZONE_PRIORITY = [
    'Pot-1 zone', 'Pot-2 zone', 'Pot-3 zone', 'Pot-4 zone',
    'Home', 'Ladder',
    'Transition zone',
    'Right corner', 'Left corner',
    'Arna center',
    'Foraging arena',
]


def load_zones_and_velocity(behavior_csv):
    """Read the single-probe behavior CSV and return (time_s, zones_str, velocity).

    Zones use a hard priority assignment in _SP_ZONE_PRIORITY order so a single
    label is emitted per time-bin, matching the dual-probe entropy pipeline.
    """
    df = _read_wide_csv(Path(behavior_csv))
    if 'Recording time' not in df.index:
        raise KeyError(f"'Recording time' row missing in {behavior_csv}")
    if 'Velocity' not in df.index:
        raise KeyError(f"'Velocity' row missing in {behavior_csv}")
    time_vals = df.loc['Recording time'].values.astype(float)
    vel = df.loc['Velocity'].values.astype(float)
    vel = np.nan_to_num(vel, nan=0.0)
    vel = np.clip(vel, 0, None)
    zones = np.full(len(time_vals), 'O', dtype=object)
    for zname in _SP_ZONE_PRIORITY:
        if zname in df.index:
            mask = df.loc[zname].values == 1
            zones[mask] = zname
    return time_vals, zones, vel


def load_behavior(session_num: int, bin_centers: np.ndarray) -> dict:
    """Return dict of behavioral variables aligned to neural bin_centers.

    Each entry: {'values': array (len T), 'type': 'continuous'|'categorical',
                 'classes': [...] (categorical only)}
    """
    sval = _sessions_cfg()[f"session_{session_num}"]
    df = _read_wide_csv(Path(sval['behavior']))

    behav_times = df.loc['Recording time'].values.astype(float)
    indices = np.searchsorted(behav_times, bin_centers, side='left')
    indices = np.clip(indices, 0, len(behav_times) - 1)
    prev = np.clip(indices - 1, 0, len(behav_times) - 1)
    use_prev = np.abs(behav_times[prev] - bin_centers) < np.abs(behav_times[indices] - bin_centers)
    indices[use_prev] = prev[use_prev]

    variables: dict = {}
    if 'Velocity' in df.index:
        v = df.loc['Velocity'].values.astype(float)
        variables['velocity'] = {'values': v[indices], 'type': 'continuous'}

    distance_rows = ['Distance to Pot-2', 'Distance to Pot-4',
                     'Distance to Transition zone', 'Distance to Home',
                     'Distance to Foraging arena']
    for r in distance_rows:
        if r in df.index:
            clean = r.lower().replace(' ', '_').replace('-', '_')
            clean = clean.replace('distance_to_', 'dist_')
            variables[clean] = {
                'values': df.loc[r].values.astype(float)[indices],
                'type': 'continuous',
            }

    pot_zones = [r for r in df.index if isinstance(r, str)
                 and r.endswith(' zone') and r.startswith('Pot-')]
    home_row = 'Home' if 'Home' in df.index else None
    ladder_row = 'Ladder' if 'Ladder' in df.index else None

    compartment = np.full(len(indices), 'Arena', dtype=object)
    if home_row is not None:
        compartment[df.loc[home_row].values[indices] == 1] = 'Home'
    if ladder_row is not None:
        compartment[df.loc[ladder_row].values[indices] == 1] = 'Ladder'
    if pot_zones:
        at_pot = np.zeros(len(indices), dtype=bool)
        for r in pot_zones:
            at_pot |= (df.loc[r].values[indices] == 1)
        compartment[at_pot] = 'AtPot'
    variables['compartment'] = {
        'values': compartment, 'type': 'categorical',
        'classes': ['Home', 'Ladder', 'Arena', 'AtPot'],
    }

    skip_exact = {
        'Recording time', 'Areachange', 'Distance moved', 'Velocity',
        'Movement(Moving / Center-point)', 'High acceleration', 'Low acceleration',
        'Pot-1 zone', 'Pot-2 zone', 'Pot-3 zone', 'Pot-4 zone',
        'Transition zone', 'Ladder', 'Home', 'Right corner', 'Left corner',
        'Arna center', 'Foraging arena',
        'Pot-1', 'Pot-2', 'Pot-3', 'Pot-4',
        'Distance to Pot-2', 'Distance to Pot-4',
        'Distance to Transition zone', 'Distance to Home',
        'Distance to Foraging arena',
        'Meander',
    }
    MIN_EVENTS = 50
    for r in df.index:
        if not isinstance(r, str) or r in skip_exact:
            continue
        vals_full = df.loc[r].values
        if (vals_full == 1).sum() < MIN_EVENTS:
            continue
        vals_binned = vals_full[indices].astype(float)
        clean = r.strip().lower().replace(' ', '_').replace('-', '_')
        variables[clean] = {
            'values': vals_binned.astype(int).astype(str),
            'type': 'categorical',
            'classes': ['0', '1'],
        }
    return variables


def filter_behavior(behav: dict) -> dict:
    """No lever-zone columns exist in single-probe Mouse01-Coor1, but keep the
    same hook signature as stage1_lib so downstream code can stay symmetric."""
    return {k: v for k, v in behav.items() if 'lever' not in k.lower()}


# ============================================================================
# Entropy + inflection detection (identical to stage1_lib)
# ============================================================================
def compute_entropy_series(zones, time_vals, vel,
                           window_sec=ENTROPY_WINDOW_SEC,
                           step_sec=ENTROPY_STEP_SEC):
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
    smoothed = gaussian_filter1d(ent_v, sigma=smooth_sigma)
    distance_bins = max(1, int(distance_sec / step_sec))
    peaks, _ = find_peaks(smoothed, prominence=prominence, distance=distance_bins)
    troughs, _ = find_peaks(-smoothed, prominence=prominence, distance=distance_bins)
    return peaks, troughs, smoothed


# ============================================================================
# Trajectory dynamics (identical to stage1_lib)
# ============================================================================
def compute_speed(matrix, sigma=SPEED_SIGMA):
    diff = np.diff(matrix, axis=0)
    speed = np.linalg.norm(diff, axis=1)
    return gaussian_filter1d(speed, sigma=sigma)


def compute_curvature(matrix, sigma=CURV_SIGMA, eps=1e-9):
    diff = np.diff(matrix, axis=0)
    norms = np.linalg.norm(diff, axis=1)
    v_a = diff[:-1]; v_b = diff[1:]
    n_a = norms[:-1]; n_b = norms[1:]
    cos_t = np.einsum('ij,ij->i', v_a, v_b) / np.maximum(n_a * n_b, eps)
    cos_t = np.clip(cos_t, -1.0, 1.0)
    return gaussian_filter1d(1.0 - cos_t, sigma=sigma)


# ============================================================================
# Phase definition and summaries (identical to stage1_lib)
# ============================================================================
def define_phases(ent_t, peak_idx, trough_idx, n_total_bins, bin_s=BIN_S,
                  peak_win_bins=PEAK_WIN_BINS):
    inflections = []
    for i in peak_idx:
        inflections.append(('peak', float(ent_t[i])))
    for i in trough_idx:
        inflections.append(('trough', float(ent_t[i])))
    inflections.sort(key=lambda x: x[1])
    phases = []
    for typ, t in inflections:
        nb = int(round(t / bin_s))
        sb = max(0, nb - peak_win_bins)
        eb = min(n_total_bins - 1, nb + peak_win_bins)
        phases.append(dict(
            phase_type=typ, start_bin=sb, end_bin=eb,
            inflection_t=t, inflection_bin=nb,
            duration_s=float((eb - sb) * bin_s),
        ))
    for i in range(len(inflections) - 1):
        t_start, t_end = inflections[i][1], inflections[i + 1][1]
        b_start = max(0, int(round(t_start / bin_s)))
        b_end = min(n_total_bins - 1, int(round(t_end / bin_s)))
        if b_end <= b_start:
            continue
        if inflections[i][0] == 'trough' and inflections[i + 1][0] == 'peak':
            ptype = 'rising'
        elif inflections[i][0] == 'peak' and inflections[i + 1][0] == 'trough':
            ptype = 'falling'
        else:
            ptype = 'mixed'
        phases.append(dict(
            phase_type=ptype, start_bin=b_start, end_bin=b_end,
            inflection_t=None, inflection_bin=None,
            duration_s=float((b_end - b_start) * bin_s),
        ))
    phases.sort(key=lambda p: p['start_bin'])
    for k, p in enumerate(phases):
        p['phase_id'] = k
    return phases


def fraction_in_class(values_in_phase, classes):
    if len(values_in_phase) == 0:
        return {c: 0.0 for c in classes}
    return {c: float(np.mean(values_in_phase == c)) for c in classes}


def transitions_count(values_in_phase):
    return int(np.sum(values_in_phase[1:] != values_in_phase[:-1]))


def phase_summary_row(phase, signals, behav, region_speeds, region_curvs,
                      region_fr, region_pc1, entropy_interp):
    sb, eb = phase['start_bin'], phase['end_bin']
    if (eb - sb) < 1:
        return None
    row = dict(
        phase_id=phase['phase_id'], phase_type=phase['phase_type'],
        start_bin=sb, end_bin=eb, duration_s=phase['duration_s'],
    )
    for region, sp in region_speeds.items():
        seg = sp[sb:min(eb, len(sp))]
        if seg.size:
            row[f'mean_speed_{region}'] = float(np.mean(seg))
            row[f'peak_speed_{region}'] = float(np.max(seg))
            row[f'time_to_peak_speed_{region}_norm'] = float(
                np.argmax(seg) / max(len(seg) - 1, 1))
        else:
            row[f'mean_speed_{region}'] = np.nan
            row[f'peak_speed_{region}'] = np.nan
            row[f'time_to_peak_speed_{region}_norm'] = np.nan
    for region, cv in region_curvs.items():
        seg = cv[sb:min(eb, len(cv))]
        if seg.size:
            row[f'mean_curv_{region}'] = float(np.mean(seg))
            row[f'peak_curv_{region}'] = float(np.max(seg))
            row[f'time_to_peak_curv_{region}_norm'] = float(
                np.argmax(seg) / max(len(seg) - 1, 1))
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

    comp = behav['compartment']['values'][sb:eb]
    row['compart_transitions'] = transitions_count(comp)
    comp_frac = fraction_in_class(comp, behav['compartment']['classes'])
    for c, v in comp_frac.items():
        row[f'frac_{c}'] = v
    row['dominant_compartment'] = max(comp_frac, key=comp_frac.get)

    if 'velocity' in behav:
        v = behav['velocity']['values'][sb:eb]
        row['mean_velocity'] = float(np.nanmean(v)) if v.size else np.nan
    pot_keys = [k for k in behav.keys() if k.startswith('dist_pot') and 'zone' not in k]
    if pot_keys:
        stack = np.stack([behav[k]['values'][sb:eb].astype(float) for k in pot_keys])
        nearest = np.nanmin(stack, axis=0)
        row['mean_dist_nearest_pot'] = float(np.nanmean(nearest)) if nearest.size else np.nan

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


def interp_entropy_to_bins(ent_t, smoothed, bin_centers):
    if len(ent_t) < 2:
        return np.full_like(bin_centers, fill_value=np.nan, dtype=float)
    return np.interp(bin_centers, ent_t, smoothed,
                     left=smoothed[0], right=smoothed[-1])
