"""
Dual-Probe: Attractor States at Entropy Peaks & Troughs
========================================================
Tests whether neural population states show attractor dynamics at
behavioral entropy extrema.

Three analyses:
1. State-space clustering — population FR vectors at peaks vs troughs,
   PCA-projected, pairwise distances within/between groups
2. Trajectory speed — ||dPC/dt|| approaching vs departing peaks/troughs
3. Recurrence — do troughs revisit the same neural state more than peaks?

Uses GPU (CuPy) for distance matrices and PCA where possible.
Separate analyses for ACA and LHA populations.
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import entropy as sp_entropy, mannwhitneyu, wilcoxon, spearmanr
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema
from sklearn.decomposition import PCA
from collections import Counter
import spikeinterface.extractors as se
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import warnings
import time as timer

warnings.filterwarnings('ignore')

# ---- GPU setup ----
try:
    import cupy as cp
    GPU = True
    print("GPU available (CuPy)")
except ImportError:
    GPU = False
    print("No GPU — using NumPy")

with open("paths.yaml") as f:
    cfg = yaml.safe_load(f)

# ---- Constants ----
FS = 30000
LHA_DEPTH_MIN = 0
LHA_DEPTH_MAX = 345
P0_MIN_FR = 0.2
P1_MIN_FR = 0.2
P1_MIN_AMP = 43

ENTROPY_WINDOW_SEC = 60
ENTROPY_STEP_SEC = 10
SMOOTH_SIGMA = 3
MIN_AMPLITUDE = 0.3

# Bin size for population FR vectors (seconds)
FR_BIN_SEC = 1.0
# Smoothing for FR vectors before PCA
FR_SMOOTH_SIGMA = 10  # bins (=10s at 1s bins)

# Peri-inflection window in entropy steps (each step = 10s)
PERI_WINDOW = 12  # +/-120s

# Top PCs for state-space analysis
N_PCS = 5

# Local contraction rate parameters
CONTRACTION_K = 15         # k nearest neighbors
CONTRACTION_EXCL = 5       # temporal exclusion zone (steps) around reference point
CONTRACTION_FWD = 8        # forward tracking steps (80s at 10s/step)
CONTRACTION_METRIC = 'pc'  # use PC space for neighbor search

SKIP_SESSIONS = {23, 24}

sessions_cfg = cfg["double_probe"]["coordinates_1"]["mouse01"]["sessions"]

# ---- Helper functions ----

zone_priority = [
    'Home corner left', 'Home corner right', 'Central Arena Zone',
    'Foraging arena', 'Home', 'ladder to Arena', 'Transition Zone',
    'Pot-1 zone', 'Pot-2 Zone', 'Pot-3 zone', 'Pot-4 zone',
    'Pot-1', 'Pot-2', 'Pot-3', 'Pot-4',
]
zone_short = {
    'Home': 'H', 'ladder to Arena': 'L', 'Transition Zone': 'T',
    'Foraging arena': 'FA', 'Central Arena Zone': 'CA',
    'Pot-1': 'P1', 'Pot-2': 'P2', 'Pot-3': 'P3', 'Pot-4': 'P4',
    'Pot-1 zone': 'P1z', 'Pot-2 Zone': 'P2z', 'Pot-3 zone': 'P3z', 'Pot-4 zone': 'P4z',
    'Home corner left': 'HCL', 'Home corner right': 'HCR',
    'Lever choice zone': 'LCZ', 'Lever Zone': 'LZ',
    'Pot choice zone': 'PCZ', 'Sand Pots Zone': 'SPZ',
    'Lever1 food zone': 'L1F', 'Lever2 food zone': 'L2F',
}


def load_behavior_xlsx(path):
    df = pd.read_excel(path, header=None)
    col_names = df.iloc[34].tolist()
    data = df.iloc[36:].reset_index(drop=True)
    data.columns = col_names
    time_vals = pd.to_numeric(data['Recording time'], errors='coerce').values
    vel = pd.to_numeric(data['Velocity(Center-point)'], errors='coerce').values
    vel = np.nan_to_num(vel, nan=0.0)
    vel = np.clip(vel, 0, None)
    zones = np.full(len(time_vals), 'O', dtype=object)
    for zname in zone_priority:
        col_match = [c for c in col_names if isinstance(c, str) and
                     c.startswith('Zone(') and zname in c]
        if col_match:
            vals = pd.to_numeric(data[col_match[0]], errors='coerce').values
            mask = vals > 0.5
            short = zone_short.get(zname, zname[:3])
            zones[mask] = short
    return time_vals, vel, zones


def compute_entropy_causal(zones, time_vals, vel, window_sec, step_sec):
    dt = np.median(np.diff(time_vals))
    window_bins = int(window_sec / dt)
    step_bins = int(step_sec / dt)
    ent_times, ent_vals, vel_means = [], [], []
    for start_idx in range(0, len(zones) - window_bins, step_bins):
        wz = zones[start_idx:start_idx + window_bins]
        transitions = []
        for j in range(1, len(wz)):
            if wz[j] != wz[j - 1]:
                transitions.append(f"{wz[j-1]}->{wz[j]}")
        if len(transitions) < 3:
            continue
        counts = Counter(transitions)
        probs = np.array(list(counts.values()), dtype=float)
        probs /= probs.sum()
        h = sp_entropy(probs, base=2)
        ent_times.append(time_vals[start_idx + window_bins - 1])
        ent_vals.append(h)
        vel_means.append(np.nanmean(vel[start_idx:start_idx + window_bins]))
    return np.array(ent_times), np.array(ent_vals), np.array(vel_means)


def get_good_units_p0(sorted_path):
    ci = Path(sorted_path) / "cluster_info.tsv"
    if not ci.exists():
        return np.array([])
    df = pd.read_csv(ci, sep='\t')
    label_col = 'group' if ('group' in df.columns and df['group'].eq('good').any()) else 'KSLabel'
    if label_col not in df.columns:
        return np.array([])
    good = df[(df[label_col] == 'good') & (df['fr'] > P0_MIN_FR)]
    return good['cluster_id'].values


def get_good_units_p1_lha(sorted_path):
    ci = Path(sorted_path) / "cluster_info.tsv"
    if not ci.exists():
        return np.array([])
    df = pd.read_csv(ci, sep='\t')
    label_col = 'group' if ('group' in df.columns and df['group'].eq('good').any()) else 'KSLabel'
    if label_col not in df.columns or 'depth' not in df.columns:
        return np.array([])
    good = df[(df[label_col] == 'good') &
              (df['fr'] > P1_MIN_FR) &
              (df['amp'] > P1_MIN_AMP) &
              (df['depth'] >= LHA_DEPTH_MIN) &
              (df['depth'] <= LHA_DEPTH_MAX)]
    return good['cluster_id'].values


def find_inflections(values, smooth_sigma, min_amp, order=3):
    smoothed = gaussian_filter1d(values, smooth_sigma)
    peaks = list(argrelextrema(smoothed, np.greater, order=order)[0])
    troughs = list(argrelextrema(smoothed, np.less, order=order)[0])
    all_ext = sorted([(p, 'peak', smoothed[p]) for p in peaks] +
                     [(t, 'trough', smoothed[t]) for t in troughs],
                     key=lambda x: x[0])
    if len(all_ext) < 2:
        return [], [], smoothed
    filtered = [all_ext[0]]
    for i in range(1, len(all_ext)):
        if all_ext[i][1] == filtered[-1][1]:
            if all_ext[i][1] == 'peak':
                if all_ext[i][2] > filtered[-1][2]:
                    filtered[-1] = all_ext[i]
            else:
                if all_ext[i][2] < filtered[-1][2]:
                    filtered[-1] = all_ext[i]
        else:
            amp = abs(all_ext[i][2] - filtered[-1][2])
            if amp >= min_amp:
                filtered.append(all_ext[i])
    final_peaks = [f[0] for f in filtered if f[1] == 'peak']
    final_troughs = [f[0] for f in filtered if f[1] == 'trough']
    return final_peaks, final_troughs, smoothed


def compute_local_contraction(pc_at_ent, ref_idx, k, excl, fwd):
    """Compute local contraction rate at a reference time point.

    For the state at ref_idx, find k nearest neighbors in PC space
    (excluding a temporal buffer of +/-excl steps), then track how
    the mean distance to those neighbors evolves over `fwd` forward steps.

    Returns: log_dist_ratio array of length fwd+1 (starting at 0).
             log_dist_ratio[t] = log(mean_d(t) / mean_d(0))
             Negative = contraction (attractor), Positive = divergence.
    """
    n_ent, n_pcs = pc_at_ent.shape
    ref_state = pc_at_ent[ref_idx]  # (n_pcs,)

    # Can't track forward if too close to end
    if ref_idx + fwd >= n_ent:
        return None

    # Distances from ref_state to all other time points
    diffs = pc_at_ent - ref_state  # (n_ent, n_pcs)
    dists = np.sqrt(np.sum(diffs ** 2, axis=1))  # (n_ent,)

    # Exclude temporal buffer around ref_idx
    eligible = np.ones(n_ent, dtype=bool)
    lo = max(0, ref_idx - excl)
    hi = min(n_ent, ref_idx + excl + 1)
    eligible[lo:hi] = False
    # Also exclude points too close to end (can't track forward)
    eligible[n_ent - fwd:] = False

    eligible_idx = np.where(eligible)[0]
    if len(eligible_idx) < k:
        return None

    # Find k nearest neighbors
    eligible_dists = dists[eligible_idx]
    topk_order = np.argsort(eligible_dists)[:k]
    neighbor_idx = eligible_idx[topk_order]

    # Track distance evolution: at each forward step t, compute mean distance
    # between the reference trajectory and each neighbor trajectory
    d0 = dists[neighbor_idx]
    mean_d0 = np.mean(d0)
    if mean_d0 < 1e-10:
        return None

    log_ratios = np.zeros(fwd + 1)
    for t in range(fwd + 1):
        ref_t = pc_at_ent[ref_idx + t]
        neigh_t = pc_at_ent[neighbor_idx + t]
        dt = np.sqrt(np.sum((neigh_t - ref_t) ** 2, axis=1))
        mean_dt = np.mean(dt)
        log_ratios[t] = np.log(max(mean_dt, 1e-10) / mean_d0)

    return log_ratios


def pairwise_distances_gpu(X):
    """Pairwise Euclidean distances using CuPy (GPU). X: (n_points, n_dims)."""
    if GPU:
        X_g = cp.asarray(X, dtype=cp.float32)
        sq = cp.sum(X_g ** 2, axis=1, keepdims=True)
        D2 = sq + sq.T - 2 * cp.dot(X_g, X_g.T)
        D2 = cp.maximum(D2, 0)
        D = cp.sqrt(D2)
        return cp.asnumpy(D)
    else:
        from scipy.spatial.distance import cdist
        return cdist(X, X, metric='euclidean')


def cross_distances_gpu(X, Y):
    """Cross pairwise distances between X (n, d) and Y (m, d) on GPU."""
    if GPU:
        X_g = cp.asarray(X, dtype=cp.float32)
        Y_g = cp.asarray(Y, dtype=cp.float32)
        sqX = cp.sum(X_g ** 2, axis=1, keepdims=True)
        sqY = cp.sum(Y_g ** 2, axis=1, keepdims=True)
        D2 = sqX + sqY.T - 2 * cp.dot(X_g, Y_g.T)
        D2 = cp.maximum(D2, 0)
        D = cp.sqrt(D2)
        return cp.asnumpy(D)
    else:
        from scipy.spatial.distance import cdist
        return cdist(X, Y, metric='euclidean')


# ========================================================================
# BUILD SESSION DATA
# ========================================================================
print("\n" + "=" * 110)
print("DUAL-PROBE: ATTRACTOR STATES AT ENTROPY PEAKS & TROUGHS")
print("Population FR vectors -> PCA -> distance/speed/recurrence")
print("=" * 110)

session_meta = {}
for skey, sval in sessions_cfg.items():
    snum = int(skey.split('_')[1])
    if snum in SKIP_SESSIONS:
        continue
    p0_sorted = sval.get('probe_0_aca', {}).get('sorted')
    p1_sorted = sval.get('probe_1_lha_rsp', {}).get('sorted')
    behav = sval.get('behavior')
    if not p0_sorted or not p1_sorted or not behav:
        continue
    if not Path(p0_sorted).exists() or not Path(behav).exists():
        continue
    session_meta[snum] = {
        'state': sval['state'], 'phase': sval['phase'],
        'p0_sorted': p0_sorted, 'p1_sorted': p1_sorted,
        'behavior': behav,
    }

print(f"Found {len(session_meta)} sessions")

STATE_ORDER = ['fed', 'fasted', 'fed-HFD']
STATE_LABELS = {'fed': 'Fed', 'fasted': 'Fasted', 'fed-HFD': 'HFD'}
STATE_COLORS = {'fed': 'tab:blue', 'fasted': 'tab:red', 'fed-HFD': 'tab:purple'}

# Storage for pooled analysis — per region
# For each region, collect: PC state at peak, PC state at trough,
# peri-inflection PC trajectories, trajectory speeds
regions = ['ACA', 'LHA']

# Pooled storage
pooled_peak_states = {r: [] for r in regions}     # (N_peaks, N_PCS) each
pooled_trough_states = {r: [] for r in regions}    # (N_troughs, N_PCS) each
pooled_peak_speeds = {r: [] for r in regions}      # peri-inflection speed traces
pooled_trough_speeds = {r: [] for r in regions}
pooled_peak_peri_pc = {r: [] for r in regions}     # peri-inflection PC1-PC3 trajectories
pooled_trough_peri_pc = {r: [] for r in regions}
# Local contraction rate: log(d(t)/d(0)) curves per inflection event
pooled_peak_contraction = {r: [] for r in regions}
pooled_trough_contraction = {r: [] for r in regions}
# Control: contraction at random time points
pooled_random_contraction = {r: [] for r in regions}
peak_meta = []   # (session, state, phase) per peak
trough_meta = []

# Per-session results
session_results = []
contraction_results = []

for snum in sorted(session_meta.keys()):
    t0 = timer.time()
    meta = session_meta[snum]
    state, phase = meta['state'], meta['phase']
    print(f"\n  S{snum} ({state}/{phase}):", end='', flush=True)

    # ---- Load behavior & entropy ----
    time_vals, vel, zones = load_behavior_xlsx(meta['behavior'])
    ent_times, ent_vals, vel_means = compute_entropy_causal(
        zones, time_vals, vel, ENTROPY_WINDOW_SEC, ENTROPY_STEP_SEC)

    if len(ent_vals) < 20:
        print(f" too few entropy pts ({len(ent_vals)}), skip")
        continue

    # ---- Find peaks & troughs ----
    peaks, troughs, smoothed = find_inflections(ent_vals, SMOOTH_SIGMA, MIN_AMPLITUDE)
    peaks = [p for p in peaks if PERI_WINDOW <= p < len(ent_vals) - PERI_WINDOW]
    troughs = [t for t in troughs if PERI_WINDOW <= t < len(ent_vals) - PERI_WINDOW]

    if len(peaks) < 1 and len(troughs) < 1:
        print(f" no valid inflections, skip")
        continue

    print(f" entropy={len(ent_vals)}, {len(peaks)}P/{len(troughs)}T", end='')

    # ---- Load neural data ----
    p0_path = Path(meta['p0_sorted'])
    p1_path = Path(meta['p1_sorted'])

    region_data = {}
    for region, sorted_path, get_ids_fn in [
        ('ACA', p0_path, get_good_units_p0),
        ('LHA', p1_path, get_good_units_p1_lha),
    ]:
        unit_ids = get_ids_fn(sorted_path)
        try:
            sorting = se.read_kilosort(sorted_path)
            avail = set(sorting.get_unit_ids())
            unit_ids = np.array([u for u in unit_ids if u in avail])
        except Exception:
            unit_ids = np.array([])
            sorting = None

        if len(unit_ids) < 3:
            region_data[region] = None
            continue

        # Build population FR matrix: (n_units, n_bins)
        bin_edges = np.arange(0, time_vals[-1] + 2, FR_BIN_SEC)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        n_bins = len(bin_centers)
        n_units = len(unit_ids)

        fr_matrix = np.zeros((n_units, n_bins), dtype=np.float32)
        for ui, uid in enumerate(unit_ids):
            st = sorting.get_unit_spike_train(uid) / FS
            fr_matrix[ui] = np.histogram(st, bins=bin_edges)[0].astype(np.float32)

        # Z-score each unit
        for ui in range(n_units):
            mu = fr_matrix[ui].mean()
            sd = fr_matrix[ui].std()
            if sd > 1e-6:
                fr_matrix[ui] = (fr_matrix[ui] - mu) / sd
            else:
                fr_matrix[ui] = 0

        # Smooth each unit
        for ui in range(n_units):
            fr_matrix[ui] = gaussian_filter1d(fr_matrix[ui], FR_SMOOTH_SIGMA)

        # PCA on the full session
        n_pcs = min(N_PCS, n_units - 1)
        pca = PCA(n_components=n_pcs)
        pc_scores = pca.fit_transform(fr_matrix.T)  # (n_bins, n_pcs)

        # Interpolate PC scores to entropy time points
        pc_at_ent = np.zeros((len(ent_times), n_pcs))
        for pi in range(n_pcs):
            pc_at_ent[:, pi] = np.interp(ent_times, bin_centers, pc_scores[:, pi])

        # Also interpolate full FR vector to entropy times for recurrence
        # Use GPU for this if large
        fr_at_ent = np.zeros((len(ent_times), n_units), dtype=np.float32)
        for ui in range(n_units):
            fr_at_ent[:, ui] = np.interp(ent_times, bin_centers, fr_matrix[ui])

        region_data[region] = {
            'unit_ids': unit_ids,
            'n_units': n_units,
            'n_pcs': n_pcs,
            'pca': pca,
            'pc_at_ent': pc_at_ent,        # (n_ent, n_pcs)
            'fr_at_ent': fr_at_ent,         # (n_ent, n_units)
            'var_explained': pca.explained_variance_ratio_,
        }

    aca_ok = region_data.get('ACA') is not None
    lha_ok = region_data.get('LHA') is not None
    print(f", ACA={'%d units %d PCs' % (region_data['ACA']['n_units'], region_data['ACA']['n_pcs']) if aca_ok else 'skip'}"
          f", LHA={'%d units %d PCs' % (region_data['LHA']['n_units'], region_data['LHA']['n_pcs']) if lha_ok else 'skip'}", end='')

    # ---- Extract states at peaks & troughs, compute speed ----
    for region in regions:
        rd = region_data.get(region)
        if rd is None:
            continue

        pc_at_ent = rd['pc_at_ent']
        n_pcs = rd['n_pcs']

        # Compute trajectory speed: ||dPC/dt|| at each entropy step
        dpc = np.diff(pc_at_ent, axis=0)  # (n_ent-1, n_pcs)
        speed = np.sqrt(np.sum(dpc ** 2, axis=1))  # (n_ent-1,)
        # Pad to match length
        speed = np.concatenate([[speed[0]], speed])  # (n_ent,)

        for idx in peaks:
            pooled_peak_states[region].append(pc_at_ent[idx])
            # Peri-inflection speed trace
            speed_window = speed[idx - PERI_WINDOW: idx + PERI_WINDOW + 1]
            pooled_peak_speeds[region].append(speed_window)
            # Peri-inflection PC trajectory (first 3 PCs)
            pc_window = pc_at_ent[idx - PERI_WINDOW: idx + PERI_WINDOW + 1, :min(3, n_pcs)]
            pooled_peak_peri_pc[region].append(pc_window)

        for idx in troughs:
            pooled_trough_states[region].append(pc_at_ent[idx])
            speed_window = speed[idx - PERI_WINDOW: idx + PERI_WINDOW + 1]
            pooled_trough_speeds[region].append(speed_window)
            pc_window = pc_at_ent[idx - PERI_WINDOW: idx + PERI_WINDOW + 1, :min(3, n_pcs)]
            pooled_trough_peri_pc[region].append(pc_window)

    # Track metadata
    for _ in peaks:
        peak_meta.append({'session': snum, 'state': state, 'phase': phase})
    for _ in troughs:
        trough_meta.append({'session': snum, 'state': state, 'phase': phase})

    # ---- Per-session recurrence analysis ----
    for region in regions:
        rd = region_data.get(region)
        if rd is None:
            continue
        fr_at_ent = rd['fr_at_ent']

        if len(peaks) >= 2 and len(troughs) >= 2:
            peak_vecs = fr_at_ent[peaks]
            trough_vecs = fr_at_ent[troughs]

            D_pp = pairwise_distances_gpu(peak_vecs)
            D_tt = pairwise_distances_gpu(trough_vecs)
            D_pt = cross_distances_gpu(peak_vecs, trough_vecs)

            # Extract upper triangle (unique pairs)
            pp_dists = D_pp[np.triu_indices(len(peaks), k=1)]
            tt_dists = D_tt[np.triu_indices(len(troughs), k=1)]
            pt_dists = D_pt.ravel()

            session_results.append({
                'session': snum, 'state': state, 'phase': phase,
                'region': region,
                'n_peaks': len(peaks), 'n_troughs': len(troughs),
                'n_units': rd['n_units'],
                'mean_peak_peak_dist': np.mean(pp_dists),
                'mean_trough_trough_dist': np.mean(tt_dists),
                'mean_peak_trough_dist': np.mean(pt_dists),
                'median_peak_peak_dist': np.median(pp_dists),
                'median_trough_trough_dist': np.median(tt_dists),
                'median_peak_trough_dist': np.median(pt_dists),
            })

    # ---- Per-session local contraction rate ----
    for region in regions:
        rd = region_data.get(region)
        if rd is None:
            continue
        pc_at_ent = rd['pc_at_ent']

        pk_rates, tr_rates, rand_rates = [], [], []

        for idx in peaks:
            lr = compute_local_contraction(pc_at_ent, idx, CONTRACTION_K,
                                           CONTRACTION_EXCL, CONTRACTION_FWD)
            if lr is not None:
                pk_rates.append(lr)
                pooled_peak_contraction[region].append(lr)

        for idx in troughs:
            lr = compute_local_contraction(pc_at_ent, idx, CONTRACTION_K,
                                           CONTRACTION_EXCL, CONTRACTION_FWD)
            if lr is not None:
                tr_rates.append(lr)
                pooled_trough_contraction[region].append(lr)

        # Random control: same number of points from middle of session
        n_rand = len(peaks) + len(troughs)
        rand_margin = PERI_WINDOW + CONTRACTION_FWD
        valid_range = np.arange(rand_margin, len(ent_vals) - rand_margin)
        # Exclude actual peaks/troughs
        excl_set = set(peaks) | set(troughs)
        valid_range = np.array([v for v in valid_range if v not in excl_set])
        if len(valid_range) >= n_rand:
            rng = np.random.RandomState(snum)
            rand_idx = rng.choice(valid_range, size=n_rand, replace=False)
            for idx in rand_idx:
                lr = compute_local_contraction(pc_at_ent, idx, CONTRACTION_K,
                                               CONTRACTION_EXCL, CONTRACTION_FWD)
                if lr is not None:
                    rand_rates.append(lr)
                    pooled_random_contraction[region].append(lr)

        if pk_rates and tr_rates:
            pk_final = np.mean([r[-1] for r in pk_rates])
            tr_final = np.mean([r[-1] for r in tr_rates])
            rand_final = np.mean([r[-1] for r in rand_rates]) if rand_rates else np.nan
            contraction_results.append({
                'session': snum, 'state': state, 'phase': phase,
                'region': region,
                'n_peaks': len(pk_rates), 'n_troughs': len(tr_rates),
                'n_random': len(rand_rates),
                'peak_final_log_ratio': pk_final,
                'trough_final_log_ratio': tr_final,
                'random_final_log_ratio': rand_final,
            })

    elapsed = timer.time() - t0
    print(f" [{elapsed:.1f}s]")

# Convert to arrays
for region in regions:
    pooled_peak_states[region] = np.array(pooled_peak_states[region]) if pooled_peak_states[region] else np.array([])
    pooled_trough_states[region] = np.array(pooled_trough_states[region]) if pooled_trough_states[region] else np.array([])
    pooled_peak_speeds[region] = np.array(pooled_peak_speeds[region]) if pooled_peak_speeds[region] else np.array([])
    pooled_trough_speeds[region] = np.array(pooled_trough_speeds[region]) if pooled_trough_speeds[region] else np.array([])

peak_meta_df = pd.DataFrame(peak_meta)
trough_meta_df = pd.DataFrame(trough_meta)

n_peaks_total = len(peak_meta)
n_troughs_total = len(trough_meta)
print(f"\nPooled: {n_peaks_total} peaks, {n_troughs_total} troughs across all sessions")

# ========================================================================
# ANALYSIS 1: STATE-SPACE CLUSTERING
# ========================================================================
print("\n" + "=" * 110)
print("ANALYSIS 1: STATE-SPACE CLUSTERING (PC distances at peaks vs troughs)")
print("=" * 110)

# For pooled analysis, we need within-session z-scored PCs to be comparable
# Since PCA is fit per-session, raw PC scores aren't directly comparable.
# Instead we use per-session recurrence results, then pool.

df_sess = pd.DataFrame(session_results)
if len(df_sess) > 0:
    for region in regions:
        rdf = df_sess[df_sess['region'] == region]
        if len(rdf) < 3:
            print(f"\n  {region}: too few sessions ({len(rdf)})")
            continue

        print(f"\n  {region} ({len(rdf)} sessions):")
        print(f"    {'Session':<20s}  {'PP dist':>10s}  {'TT dist':>10s}  {'PT dist':>10s}  {'TT < PP?':>10s}")

        tt_less_pp = 0
        for _, row in rdf.iterrows():
            is_less = row['mean_trough_trough_dist'] < row['mean_peak_peak_dist']
            tt_less_pp += int(is_less)
            print(f"    S{row['session']:<2.0f} ({row['state']:<7s}/{row['phase']:<11s})  "
                  f"{row['mean_peak_peak_dist']:10.2f}  {row['mean_trough_trough_dist']:10.2f}  "
                  f"{row['mean_peak_trough_dist']:10.2f}  {'YES' if is_less else 'no':>10s}")

        pp_means = rdf['mean_peak_peak_dist'].values
        tt_means = rdf['mean_trough_trough_dist'].values
        pt_means = rdf['mean_peak_trough_dist'].values

        print(f"\n    Grand mean: PP={np.mean(pp_means):.2f}, TT={np.mean(tt_means):.2f}, PT={np.mean(pt_means):.2f}")
        print(f"    TT < PP in {tt_less_pp}/{len(rdf)} sessions")

        if len(pp_means) >= 3 and len(tt_means) >= 3:
            try:
                stat_w, p_w = wilcoxon(tt_means, pp_means)
                print(f"    Wilcoxon (TT vs PP): p={p_w:.4f}{'*' if p_w < 0.05 else ''}")
            except ValueError:
                print(f"    Wilcoxon: N/A (too few pairs)")

            _, p_mwu = mannwhitneyu(tt_means, pp_means, alternative='less')
            print(f"    MWU (TT < PP): p={p_mwu:.4f}{'*' if p_mwu < 0.05 else ''}")

        # Per-state breakdown
        for st in STATE_ORDER:
            st_df = rdf[rdf['state'] == st]
            if len(st_df) < 2:
                continue
            print(f"    {STATE_LABELS[st]}: PP={st_df['mean_peak_peak_dist'].mean():.2f}, "
                  f"TT={st_df['mean_trough_trough_dist'].mean():.2f}, "
                  f"PT={st_df['mean_peak_trough_dist'].mean():.2f} (n={len(st_df)})")

# ========================================================================
# ANALYSIS 2: TRAJECTORY SPEED
# ========================================================================
print("\n" + "=" * 110)
print("ANALYSIS 2: TRAJECTORY SPEED (||dPC/dt|| around peaks vs troughs)")
print("=" * 110)

time_axis = np.arange(-PERI_WINDOW, PERI_WINDOW + 1) * ENTROPY_STEP_SEC

for region in regions:
    pk_speeds = pooled_peak_speeds[region]
    tr_speeds = pooled_trough_speeds[region]

    if len(pk_speeds) < 4 or len(tr_speeds) < 4:
        print(f"\n  {region}: too few events (peaks={len(pk_speeds)}, troughs={len(tr_speeds)})")
        continue

    print(f"\n  {region} ({len(pk_speeds)} peaks, {len(tr_speeds)} troughs):")

    # Speed at the inflection point (t=0)
    pk_at = pk_speeds[:, PERI_WINDOW]
    tr_at = tr_speeds[:, PERI_WINDOW]
    _, p_at = mannwhitneyu(pk_at, tr_at, alternative='two-sided')
    print(f"    Speed at inflection: peak={np.mean(pk_at):.3f}, trough={np.mean(tr_at):.3f}, MWU p={p_at:.4f}{'*' if p_at < 0.05 else ''}")

    # Speed approach (pre: -60s to -10s, i.e. steps -6 to -1) vs departure (+10s to +60s, steps +1 to +6)
    pre_sl = slice(PERI_WINDOW - 6, PERI_WINDOW)
    post_sl = slice(PERI_WINDOW + 1, PERI_WINDOW + 7)

    for label, speeds, n_ev in [('Peaks', pk_speeds, len(pk_speeds)),
                                  ('Troughs', tr_speeds, len(tr_speeds))]:
        pre_speed = np.mean(speeds[:, pre_sl], axis=1)
        at_speed = speeds[:, PERI_WINDOW]
        post_speed = np.mean(speeds[:, post_sl], axis=1)

        # Deceleration approaching = pre > at
        decel = np.mean(pre_speed) - np.mean(at_speed)
        # Acceleration departing = post > at
        accel = np.mean(post_speed) - np.mean(at_speed)

        try:
            _, p_decel = wilcoxon(pre_speed, at_speed)
        except ValueError:
            p_decel = 1.0
        try:
            _, p_accel = wilcoxon(post_speed, at_speed)
        except ValueError:
            p_accel = 1.0

        print(f"    {label} (n={n_ev}):")
        print(f"      Pre={np.mean(pre_speed):.3f}, At={np.mean(at_speed):.3f}, Post={np.mean(post_speed):.3f}")
        print(f"      Deceleration (pre-at): {decel:+.3f}, Wilcox p={p_decel:.4f}{'*' if p_decel < 0.05 else ''}")
        print(f"      Acceleration (post-at): {accel:+.3f}, Wilcox p={p_accel:.4f}{'*' if p_accel < 0.05 else ''}")

    # Asymmetry: approach vs departure speed ratio
    for label, speeds in [('Peaks', pk_speeds), ('Troughs', tr_speeds)]:
        pre_speed = np.mean(speeds[:, pre_sl], axis=1)
        post_speed = np.mean(speeds[:, post_sl], axis=1)
        ratio = post_speed / (pre_speed + 1e-10)
        print(f"    {label} departure/approach ratio: {np.mean(ratio):.3f} (>1 = faster exit)")

# ========================================================================
# ANALYSIS 3: RECURRENCE
# ========================================================================
print("\n" + "=" * 110)
print("ANALYSIS 3: RECURRENCE (across-session consistency)")
print("=" * 110)

# Since PCA is per-session and not directly comparable across sessions,
# the per-session recurrence analysis (already computed above) is the proper test.
# Here we summarize and add: within-session, do trough states recur more tightly?

if len(df_sess) > 0:
    for region in regions:
        rdf = df_sess[df_sess['region'] == region]
        if len(rdf) < 3:
            continue

        # Recurrence ratio: TT/PP distance — <1 means troughs are more recurrent
        rdf = rdf.copy()
        rdf['recurrence_ratio'] = rdf['mean_trough_trough_dist'] / rdf['mean_peak_peak_dist']

        print(f"\n  {region} — Recurrence ratio (TT dist / PP dist):")
        print(f"    <1 means trough states cluster more tightly (more attractor-like)")
        for _, row in rdf.iterrows():
            print(f"    S{row['session']:<2.0f}: ratio={row['recurrence_ratio']:.3f}")
        print(f"    Mean ratio: {rdf['recurrence_ratio'].mean():.3f}")
        print(f"    Median ratio: {rdf['recurrence_ratio'].median():.3f}")

        # Sign test: how many sessions have ratio < 1?
        n_below = (rdf['recurrence_ratio'] < 1).sum()
        n_total = len(rdf)
        from scipy.stats import binomtest
        binom_p = binomtest(n_below, n_total, 0.5, alternative='greater').pvalue
        print(f"    Ratio < 1 in {n_below}/{n_total} sessions, binomial p={binom_p:.4f}{'*' if binom_p < 0.05 else ''}")

        # Separation: PT dist should be > max(PP, TT) if states are distinct
        rdf['separation'] = rdf['mean_peak_trough_dist'] / ((rdf['mean_peak_peak_dist'] + rdf['mean_trough_trough_dist']) / 2)
        print(f"    Separation index (PT / avg(PP,TT)): mean={rdf['separation'].mean():.3f}")
        print(f"    >1 means peaks and troughs occupy distinct regions")

# ========================================================================
# ANALYSIS 4: LOCAL CONTRACTION RATE
# ========================================================================
print("\n" + "=" * 110)
print("ANALYSIS 4: LOCAL CONTRACTION RATE (neighbor divergence around peaks vs troughs)")
print(f"  k={CONTRACTION_K} neighbors, excl={CONTRACTION_EXCL} steps, forward={CONTRACTION_FWD} steps ({CONTRACTION_FWD*ENTROPY_STEP_SEC}s)")
print("  Negative log-ratio = contraction (attractor), Positive = divergence (repeller)")
print("=" * 110)

# Convert contraction lists to arrays
for region in regions:
    pooled_peak_contraction[region] = np.array(pooled_peak_contraction[region]) if pooled_peak_contraction[region] else np.array([]).reshape(0, CONTRACTION_FWD + 1)
    pooled_trough_contraction[region] = np.array(pooled_trough_contraction[region]) if pooled_trough_contraction[region] else np.array([]).reshape(0, CONTRACTION_FWD + 1)
    pooled_random_contraction[region] = np.array(pooled_random_contraction[region]) if pooled_random_contraction[region] else np.array([]).reshape(0, CONTRACTION_FWD + 1)

contraction_time = np.arange(CONTRACTION_FWD + 1) * ENTROPY_STEP_SEC

for region in regions:
    pk_c = pooled_peak_contraction[region]
    tr_c = pooled_trough_contraction[region]
    rd_c = pooled_random_contraction[region]

    if len(pk_c) < 4 or len(tr_c) < 4:
        print(f"\n  {region}: too few events (peaks={len(pk_c)}, troughs={len(tr_c)})")
        continue

    print(f"\n  {region} ({len(pk_c)} peaks, {len(tr_c)} troughs, {len(rd_c)} random):")

    # Final log-ratio (at max forward step)
    pk_final = pk_c[:, -1]
    tr_final = tr_c[:, -1]
    rd_final = rd_c[:, -1] if len(rd_c) >= 4 else np.array([])

    print(f"    Final log-ratio at +{CONTRACTION_FWD*ENTROPY_STEP_SEC}s:")
    print(f"      Peaks:   mean={np.mean(pk_final):+.4f}, median={np.median(pk_final):+.4f}")
    print(f"      Troughs: mean={np.mean(tr_final):+.4f}, median={np.median(tr_final):+.4f}")
    if len(rd_final) >= 4:
        print(f"      Random:  mean={np.mean(rd_final):+.4f}, median={np.median(rd_final):+.4f}")

    # Statistical tests
    _, p_pk_tr = mannwhitneyu(pk_final, tr_final, alternative='two-sided')
    print(f"    Peak vs Trough: MWU p={p_pk_tr:.4f}{'*' if p_pk_tr < 0.05 else ''}")

    if len(rd_final) >= 4:
        _, p_pk_rd = mannwhitneyu(pk_final, rd_final, alternative='two-sided')
        _, p_tr_rd = mannwhitneyu(tr_final, rd_final, alternative='two-sided')
        print(f"    Peak vs Random: MWU p={p_pk_rd:.4f}{'*' if p_pk_rd < 0.05 else ''}")
        print(f"    Trough vs Random: MWU p={p_tr_rd:.4f}{'*' if p_tr_rd < 0.05 else ''}")

    # One-sample test: is contraction significantly negative (i.e., is it an attractor)?
    from scipy.stats import wilcoxon as wilcoxon_1s
    try:
        _, p_pk_neg = wilcoxon_1s(pk_final, alternative='less')
        print(f"    Peaks < 0 (contraction): Wilcox p={p_pk_neg:.4f}{'*' if p_pk_neg < 0.05 else ''}")
    except ValueError:
        pass
    try:
        _, p_tr_neg = wilcoxon_1s(tr_final, alternative='less')
        print(f"    Troughs < 0 (contraction): Wilcox p={p_tr_neg:.4f}{'*' if p_tr_neg < 0.05 else ''}")
    except ValueError:
        pass

    # Fraction negative (contracting)
    pk_neg_frac = np.mean(pk_final < 0)
    tr_neg_frac = np.mean(tr_final < 0)
    print(f"    Fraction contracting: peaks={pk_neg_frac:.2%}, troughs={tr_neg_frac:.2%}")

    # Per-state breakdown
    pk_states = np.array([m['state'] for m in peak_meta])
    tr_states = np.array([m['state'] for m in trough_meta])
    # Match lengths to contraction arrays
    n_pk = min(len(pk_states), len(pk_c))
    n_tr = min(len(tr_states), len(tr_c))

    for st in STATE_ORDER:
        pk_mask = pk_states[:n_pk] == st
        tr_mask = tr_states[:n_tr] == st
        if pk_mask.sum() < 2 or tr_mask.sum() < 2:
            continue
        pk_st = pk_c[pk_mask, -1]
        tr_st = tr_c[tr_mask, -1]
        _, p_st = mannwhitneyu(pk_st, tr_st, alternative='two-sided')
        print(f"    {STATE_LABELS[st]}: peak={np.mean(pk_st):+.4f} (n={len(pk_st)}), "
              f"trough={np.mean(tr_st):+.4f} (n={len(tr_st)}), MWU p={p_st:.4f}{'*' if p_st < 0.05 else ''}")

# Per-session contraction summary
df_contr = pd.DataFrame(contraction_results)
if len(df_contr) > 0:
    print(f"\n  Per-session contraction summary:")
    for region in regions:
        rdf = df_contr[df_contr['region'] == region]
        if len(rdf) < 3:
            continue
        print(f"\n  {region}:")
        print(f"    {'Session':<20s}  {'Peak logR':>10s}  {'Trough logR':>10s}  {'Random logR':>12s}  {'Tr < Pk?':>10s}")
        tr_less_pk = 0
        for _, row in rdf.iterrows():
            is_less = row['trough_final_log_ratio'] < row['peak_final_log_ratio']
            tr_less_pk += int(is_less)
            print(f"    S{row['session']:<2.0f} ({row['state']:<7s}/{row['phase']:<11s})  "
                  f"{row['peak_final_log_ratio']:+10.4f}  {row['trough_final_log_ratio']:+10.4f}  "
                  f"{row['random_final_log_ratio']:+12.4f}  {'YES' if is_less else 'no':>10s}")
        print(f"    Trough more contracting in {tr_less_pk}/{len(rdf)} sessions")

        # Paired test across sessions
        if len(rdf) >= 5:
            try:
                _, p_paired = wilcoxon(rdf['trough_final_log_ratio'].values,
                                       rdf['peak_final_log_ratio'].values)
                print(f"    Paired Wilcoxon (trough vs peak): p={p_paired:.4f}{'*' if p_paired < 0.05 else ''}")
            except ValueError:
                pass

# ========================================================================
# SAVE DATA
# ========================================================================
df_sess.to_csv("data/dp_entropy_attractor_sessions.csv", index=False)
print(f"\nSaved data/dp_entropy_attractor_sessions.csv ({len(df_sess)} rows)")

df_contr.to_csv("data/dp_entropy_attractor_contraction.csv", index=False)
print(f"Saved data/dp_entropy_attractor_contraction.csv ({len(df_contr)} rows)")

# ========================================================================
# FIGURES
# ========================================================================
outdir = Path("figures")
outdir.mkdir(exist_ok=True)

# Figure 1: Trajectory speed around peaks vs troughs (per region)
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ri, region in enumerate(regions):
    ax = axes[ri]
    pk_speeds = pooled_peak_speeds[region]
    tr_speeds = pooled_trough_speeds[region]

    if len(pk_speeds) < 4 or len(tr_speeds) < 4:
        ax.text(0.5, 0.5, f'{region}\nInsufficient data', transform=ax.transAxes, ha='center')
        continue

    pk_mean = np.mean(pk_speeds, axis=0)
    pk_sem = np.std(pk_speeds, axis=0) / np.sqrt(len(pk_speeds))
    tr_mean = np.mean(tr_speeds, axis=0)
    tr_sem = np.std(tr_speeds, axis=0) / np.sqrt(len(tr_speeds))

    ax.fill_between(time_axis, pk_mean - pk_sem, pk_mean + pk_sem, alpha=0.2, color='tab:red')
    ax.plot(time_axis, pk_mean, color='tab:red', linewidth=2, label=f'Peaks (n={len(pk_speeds)})')
    ax.fill_between(time_axis, tr_mean - tr_sem, tr_mean + tr_sem, alpha=0.2, color='tab:blue')
    ax.plot(time_axis, tr_mean, color='tab:blue', linewidth=2, label=f'Troughs (n={len(tr_speeds)})')

    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time from inflection (s)', fontsize=13)
    ax.set_ylabel('Trajectory speed (||dPC/dt||)', fontsize=13) if ri == 0 else None
    ax.set_title(f'{region}', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.tick_params(labelsize=11)

plt.suptitle('Neural Trajectory Speed Around Entropy Inflections', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(outdir / "dp_entropy_attractor_speed.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/dp_entropy_attractor_speed.png")

# Figure 2: Trajectory speed by state
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey='row')
for ri, region in enumerate(regions):
    pk_speeds_all = pooled_peak_speeds[region]
    tr_speeds_all = pooled_trough_speeds[region]
    if len(pk_speeds_all) < 4:
        continue

    pk_states = np.array([m['state'] for m in peak_meta])
    tr_states = np.array([m['state'] for m in trough_meta])

    # Trim to match region (some sessions may have skipped a region)
    # Use min of meta length and speed array length
    n_pk = min(len(pk_states), len(pk_speeds_all))
    n_tr = min(len(tr_states), len(tr_speeds_all))

    for col, (label, speeds, states, n_ev) in enumerate([
        ('Peaks', pk_speeds_all[:n_pk], pk_states[:n_pk], n_pk),
        ('Troughs', tr_speeds_all[:n_tr], tr_states[:n_tr], n_tr),
    ]):
        ax = axes[ri, col]
        for st in STATE_ORDER:
            mask = states == st
            if mask.sum() < 2:
                continue
            sp = speeds[mask]
            mean_sp = np.mean(sp, axis=0)
            sem_sp = np.std(sp, axis=0) / np.sqrt(len(sp))
            ax.fill_between(time_axis, mean_sp - sem_sp, mean_sp + sem_sp,
                            alpha=0.15, color=STATE_COLORS[st])
            ax.plot(time_axis, mean_sp, color=STATE_COLORS[st], linewidth=2,
                    label=f'{STATE_LABELS[st]} (n={mask.sum()})')

        ax.axvline(0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time from inflection (s)', fontsize=12)
        if col == 0:
            ax.set_ylabel(f'{region} speed', fontsize=13)
        ax.set_title(f'{region} — {label}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.tick_params(labelsize=10)

plt.suptitle('Trajectory Speed by Metabolic State', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(outdir / "dp_entropy_attractor_speed_by_state.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/dp_entropy_attractor_speed_by_state.png")

# Figure 3: Recurrence — bar chart of PP, TT, PT distances per session
if len(df_sess) > 0:
    for region in regions:
        rdf = df_sess[df_sess['region'] == region].copy()
        if len(rdf) < 3:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Panel A: distances per session
        ax = axes[0]
        x = np.arange(len(rdf))
        w = 0.25
        bars_pp = ax.bar(x - w, rdf['mean_peak_peak_dist'].values, w, label='Peak-Peak',
                         color='tab:red', alpha=0.7, edgecolor='black', linewidth=0.5)
        bars_tt = ax.bar(x, rdf['mean_trough_trough_dist'].values, w, label='Trough-Trough',
                         color='tab:blue', alpha=0.7, edgecolor='black', linewidth=0.5)
        bars_pt = ax.bar(x + w, rdf['mean_peak_trough_dist'].values, w, label='Peak-Trough',
                         color='tab:gray', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"S{int(r['session'])}\n{r['state'][:3]}" for _, r in rdf.iterrows()],
                           fontsize=9)
        ax.set_ylabel('Mean pairwise distance\n(population FR space)', fontsize=12)
        ax.set_title(f'{region} — Within-Session Distances', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.tick_params(labelsize=10)

        # Panel B: recurrence ratio
        ax = axes[1]
        rdf_plot = rdf.copy()
        rdf_plot['recurrence_ratio'] = rdf_plot['mean_trough_trough_dist'] / rdf_plot['mean_peak_peak_dist']
        colors = [STATE_COLORS.get(r['state'], 'gray') for _, r in rdf_plot.iterrows()]
        bars = ax.bar(x, rdf_plot['recurrence_ratio'].values, color=colors, alpha=0.7,
                      edgecolor='black', linewidth=0.5)
        ax.axhline(1.0, color='black', linestyle='--', linewidth=1, label='ratio=1 (no difference)')
        ax.set_xticks(x)
        ax.set_xticklabels([f"S{int(r['session'])}" for _, r in rdf_plot.iterrows()], fontsize=10)
        ax.set_ylabel('Recurrence ratio\n(TT dist / PP dist)', fontsize=12)
        ax.set_title(f'{region} — Recurrence Ratio (<1 = trough attractor)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.tick_params(labelsize=10)

        # State legend
        handles = [Patch(facecolor=STATE_COLORS[st], label=STATE_LABELS[st])
                   for st in STATE_ORDER if st in rdf['state'].values]
        ax.legend(handles=handles, fontsize=10, loc='upper right')

        plt.suptitle(f'{region}: Attractor Recurrence Analysis', fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(outdir / f"dp_entropy_attractor_recurrence_{region.lower()}.png",
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved figures/dp_entropy_attractor_recurrence_{region.lower()}.png")

# Figure 4: 2D PC trajectory snippets around a representative peak & trough
# Show the peri-inflection trajectory in PC1-PC2 space, color-coded by time
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ri, region in enumerate(regions):
    pk_peri = pooled_peak_peri_pc[region]
    tr_peri = pooled_trough_peri_pc[region]

    if len(pk_peri) < 1 or len(tr_peri) < 1:
        continue

    # Plot all trajectories overlaid, color = time (blue->red)
    cmap = plt.cm.coolwarm
    n_steps = 2 * PERI_WINDOW + 1
    colors_t = cmap(np.linspace(0, 1, n_steps))

    for col, (label, peri_list) in enumerate([('Peaks', pk_peri), ('Troughs', tr_peri)]):
        ax = axes[ri, col]
        for traj in peri_list:
            if traj.shape[1] < 2:
                continue
            # Thin line for each trajectory
            for t in range(len(traj) - 1):
                ax.plot(traj[t:t+2, 0], traj[t:t+2, 1],
                        color=colors_t[t], alpha=0.15, linewidth=0.8)
            # Mark the inflection point
            ax.plot(traj[PERI_WINDOW, 0], traj[PERI_WINDOW, 1],
                    'k.', markersize=2, alpha=0.3)

        # Mean trajectory
        mean_traj = np.mean(peri_list, axis=0)
        if mean_traj.shape[1] >= 2:
            for t in range(len(mean_traj) - 1):
                ax.plot(mean_traj[t:t+2, 0], mean_traj[t:t+2, 1],
                        color=colors_t[t], linewidth=3, alpha=0.9)
            ax.plot(mean_traj[PERI_WINDOW, 0], mean_traj[PERI_WINDOW, 1],
                    'k*', markersize=12, zorder=10, label='Inflection')

        ax.set_xlabel('PC1 (a.u.)', fontsize=12)
        ax.set_ylabel('PC2 (a.u.)', fontsize=12)
        ax.set_title(f'{region} — {label} (n={len(peri_list)})', fontsize=13, fontweight='bold')
        ax.tick_params(labelsize=10)
        if col == 0 and ri == 0:
            ax.legend(fontsize=9)

# Add colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-120, 120))
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, shrink=0.6, label='Time from inflection (s)')

plt.suptitle('PC Trajectories Around Entropy Inflections\n(blue=approach, red=departure)',
             fontsize=15, fontweight='bold')
plt.tight_layout(rect=[0, 0, 0.92, 0.95])
plt.savefig(outdir / "dp_entropy_attractor_trajectories.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/dp_entropy_attractor_trajectories.png")

# Figure 5: Speed at inflection — boxplot peaks vs troughs, per region
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ri, region in enumerate(regions):
    ax = axes[ri]
    pk_speeds = pooled_peak_speeds[region]
    tr_speeds = pooled_trough_speeds[region]
    if len(pk_speeds) < 4 or len(tr_speeds) < 4:
        ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes, ha='center')
        continue

    pk_at = pk_speeds[:, PERI_WINDOW]
    tr_at = tr_speeds[:, PERI_WINDOW]

    bp = ax.boxplot([pk_at, tr_at], labels=['Peaks', 'Troughs'],
                    patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('tab:red')
    bp['boxes'][0].set_alpha(0.5)
    bp['boxes'][1].set_facecolor('tab:blue')
    bp['boxes'][1].set_alpha(0.5)

    # Overlay individual points
    for xi, data in enumerate([pk_at, tr_at], 1):
        jitter = np.random.normal(0, 0.04, len(data))
        ax.scatter(np.full(len(data), xi) + jitter, data, alpha=0.3, s=15,
                   color='black', zorder=3)

    _, p_val = mannwhitneyu(pk_at, tr_at, alternative='two-sided')
    sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
    ymax = max(np.percentile(pk_at, 95), np.percentile(tr_at, 95))
    ax.text(1.5, ymax * 1.1, f'p={p_val:.4f} {sig}', ha='center', fontsize=12, fontweight='bold')

    ax.set_ylabel('Trajectory speed at inflection', fontsize=13)
    ax.set_title(f'{region}', fontsize=14, fontweight='bold')
    ax.tick_params(labelsize=11)

plt.suptitle('Neural Trajectory Speed: Peaks vs Troughs', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(outdir / "dp_entropy_attractor_speed_boxplot.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/dp_entropy_attractor_speed_boxplot.png")

# Figure 6: Local contraction rate curves — peaks vs troughs vs random
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ri, region in enumerate(regions):
    ax = axes[ri]
    pk_c = pooled_peak_contraction[region]
    tr_c = pooled_trough_contraction[region]
    rd_c = pooled_random_contraction[region]

    if len(pk_c) < 4 or len(tr_c) < 4:
        ax.text(0.5, 0.5, f'{region}\nInsufficient data', transform=ax.transAxes, ha='center')
        continue

    for data, color, label in [
        (pk_c, 'tab:red', f'Peaks (n={len(pk_c)})'),
        (tr_c, 'tab:blue', f'Troughs (n={len(tr_c)})'),
        (rd_c, 'gray', f'Random (n={len(rd_c)})'),
    ]:
        if len(data) < 4:
            continue
        mean_c = np.mean(data, axis=0)
        sem_c = np.std(data, axis=0) / np.sqrt(len(data))
        ax.fill_between(contraction_time, mean_c - sem_c, mean_c + sem_c,
                        alpha=0.2, color=color)
        ax.plot(contraction_time, mean_c, color=color, linewidth=2, label=label)

    ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Forward time (s)', fontsize=13)
    ax.set_ylabel('log(d(t)/d(0))', fontsize=13) if ri == 0 else None
    ax.set_title(f'{region}', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=11)
    ax.text(0.02, 0.02, 'contraction', transform=ax.transAxes, fontsize=9,
            color='green', va='bottom')
    ax.text(0.02, 0.98, 'divergence', transform=ax.transAxes, fontsize=9,
            color='red', va='top')

plt.suptitle(f'Local Contraction Rate (k={CONTRACTION_K} neighbors)\n'
             'Negative = neighbors converge (attractor), Positive = neighbors diverge',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(outdir / "dp_entropy_attractor_contraction.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/dp_entropy_attractor_contraction.png")

# Figure 7: Contraction rate by state
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
pk_states_arr = np.array([m['state'] for m in peak_meta])
tr_states_arr = np.array([m['state'] for m in trough_meta])

for ri, region in enumerate(regions):
    pk_c = pooled_peak_contraction[region]
    tr_c = pooled_trough_contraction[region]

    if len(pk_c) < 4 or len(tr_c) < 4:
        continue

    n_pk = min(len(pk_states_arr), len(pk_c))
    n_tr = min(len(tr_states_arr), len(tr_c))

    for col, (label, data, states, n_ev) in enumerate([
        ('Peaks', pk_c[:n_pk], pk_states_arr[:n_pk], n_pk),
        ('Troughs', tr_c[:n_tr], tr_states_arr[:n_tr], n_tr),
    ]):
        ax = axes[ri, col]
        for st in STATE_ORDER:
            mask = states == st
            if mask.sum() < 2:
                continue
            d = data[mask]
            mean_d = np.mean(d, axis=0)
            sem_d = np.std(d, axis=0) / np.sqrt(len(d))
            ax.fill_between(contraction_time, mean_d - sem_d, mean_d + sem_d,
                            alpha=0.15, color=STATE_COLORS[st])
            ax.plot(contraction_time, mean_d, color=STATE_COLORS[st], linewidth=2,
                    label=f'{STATE_LABELS[st]} (n={mask.sum()})')

        ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xlabel('Forward time (s)', fontsize=12)
        if col == 0:
            ax.set_ylabel(f'{region} log(d(t)/d(0))', fontsize=13)
        ax.set_title(f'{region} — {label}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.tick_params(labelsize=10)

plt.suptitle('Local Contraction Rate by Metabolic State', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(outdir / "dp_entropy_attractor_contraction_by_state.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/dp_entropy_attractor_contraction_by_state.png")

# Figure 8: Final contraction — boxplot peaks vs troughs vs random
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ri, region in enumerate(regions):
    ax = axes[ri]
    pk_c = pooled_peak_contraction[region]
    tr_c = pooled_trough_contraction[region]
    rd_c = pooled_random_contraction[region]

    if len(pk_c) < 4 or len(tr_c) < 4:
        ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes, ha='center')
        continue

    pk_final = pk_c[:, -1]
    tr_final = tr_c[:, -1]
    box_data = [pk_final, tr_final]
    box_labels = ['Peaks', 'Troughs']
    box_colors = ['tab:red', 'tab:blue']
    if len(rd_c) >= 4:
        rd_final = rd_c[:, -1]
        box_data.append(rd_final)
        box_labels.append('Random')
        box_colors.append('gray')

    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, widths=0.5)
    for i, color in enumerate(box_colors):
        bp['boxes'][i].set_facecolor(color)
        bp['boxes'][i].set_alpha(0.5)

    for xi, data in enumerate(box_data, 1):
        jitter = np.random.normal(0, 0.04, len(data))
        ax.scatter(np.full(len(data), xi) + jitter, data, alpha=0.3, s=15,
                   color='black', zorder=3)

    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    _, p_val = mannwhitneyu(pk_final, tr_final, alternative='two-sided')
    sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
    ymax = max(np.percentile(pk_final, 95), np.percentile(tr_final, 95))
    ax.text(1.5, ymax * 1.15, f'p={p_val:.4f} {sig}', ha='center', fontsize=11, fontweight='bold')

    ax.set_ylabel(f'Final log(d/d0) at +{CONTRACTION_FWD*ENTROPY_STEP_SEC}s', fontsize=12)
    ax.set_title(f'{region}', fontsize=14, fontweight='bold')
    ax.tick_params(labelsize=11)

plt.suptitle(f'Local Contraction at +{CONTRACTION_FWD*ENTROPY_STEP_SEC}s: Peaks vs Troughs vs Random',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(outdir / "dp_entropy_attractor_contraction_boxplot.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/dp_entropy_attractor_contraction_boxplot.png")

print("\nDone.")
