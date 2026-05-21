"""
Shared helpers for cycle identification (persistent cohomology circular coordinates).

Implements:
- load_neural(session_num, region)         : same preprocessing as dp_manifold_layer1b
- load_behavior(session_num, bin_centers)  : aligned behavioral variables at 50ms bins
- circ_mean(phi)                           : mean resultant direction
- circ_linear_corr(phi, x)                 : circular-linear correlation (closed form)
- circ_circ_corr(phi1, phi2)               : Fisher-Lee circular-circular correlation
- rayleigh_test(phi)                       : Rayleigh test for non-uniformity
- watson_williams(phi, labels)             : circular ANOVA across groups
- permute_pvalue(stat_fn, a, b, n=1000)    : permutation null
- bh_correct(pvals)                        : Benjamini-Hochberg FDR correction
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy import stats
from scipy.ndimage import gaussian_filter1d
import pycircstat as pc

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dp_avalanche_criticality import (
    get_good_units_p0, get_good_units_p1_lha,
    load_spike_times_for_region,
)
import spikeinterface.extractors as se

BIN_MS = 50.0
SMOOTH_SIGMA = 1.0
K_PCS = {'ACA': 10, 'LHA': 5}


def _sessions_cfg():
    with open(REPO_ROOT / "paths.yaml") as f:
        cfg = yaml.safe_load(f)
    return cfg["double_probe"]["coordinates_1"]["mouse01"]["sessions"]


def load_neural(session_num, region):
    """50ms bins, Gaussian sigma=1, z-scored. Returns (matrix [N,U], bin_centers, n_units)."""
    sessions_cfg = _sessions_cfg()
    sval = sessions_cfg[f"session_{session_num}"]
    if region == 'ACA':
        sp = Path(sval['probe_0_aca']['sorted'])
        uids = get_good_units_p0(sp)
    else:
        sp = Path(sval['probe_1_lha_rsp']['sorted'])
        uids = get_good_units_p1_lha(sp)
    sorting = se.read_kilosort(sp)
    avail = set(sorting.get_unit_ids())
    uids = np.array([u for u in uids if u in avail])
    spike_dict = load_spike_times_for_region(sorting, uids)
    all_sp = np.concatenate(list(spike_dict.values()))
    dur = float(all_sp.max()) + 1.0

    dt = BIN_MS / 1000.0
    n_bins = int(dur / dt)
    bin_edges = np.arange(0, n_bins + 1) * dt
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    unit_ids = sorted(spike_dict.keys())
    matrix = np.zeros((n_bins, len(unit_ids)))
    for j, uid in enumerate(unit_ids):
        counts, _ = np.histogram(spike_dict[uid], bins=bin_edges)
        matrix[:, j] = gaussian_filter1d(counts.astype(float), sigma=SMOOTH_SIGMA)
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    stds[stds == 0] = 1.0
    matrix = (matrix - means) / stds
    return matrix, bin_centers, len(unit_ids)


def load_behavior(session_num, bin_centers):
    """Return dict of behavioral variables aligned to bin_centers.

    Each variable: {'values': array of length len(bin_centers),
                    'type': 'continuous'|'categorical',
                    'classes': list (categorical only)}
    """
    sessions_cfg = _sessions_cfg()
    sval = sessions_cfg[f"session_{session_num}"]
    raw = pd.read_excel(sval['behavior'], header=None)
    col_names = list(raw.iloc[34].values)
    data = raw.iloc[36:].copy()
    data.columns = col_names
    data = data.reset_index(drop=True)
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    behav_times = data['Trial time'].values.astype(float)
    indices = np.searchsorted(behav_times, bin_centers, side='left')
    indices = np.clip(indices, 0, len(behav_times) - 1)
    prev = np.clip(indices - 1, 0, len(behav_times) - 1)
    use_prev = np.abs(behav_times[prev] - bin_centers) < np.abs(behav_times[indices] - bin_centers)
    indices[use_prev] = prev[use_prev]

    variables = {}
    variables['velocity'] = {
        'values': data['Velocity(Center-point)'].values[indices].astype(float),
        'type': 'continuous',
    }
    direction_rad = np.deg2rad(data['Direction'].values[indices].astype(float))
    variables['heading_sin'] = {'values': np.sin(direction_rad), 'type': 'continuous'}
    variables['heading_cos'] = {'values': np.cos(direction_rad), 'type': 'continuous'}

    # distance variables (continuous)
    for col in data.columns:
        if col is None:
            continue
        cs = str(col)
        if cs.startswith('Distance to zone('):
            clean = cs.replace('Distance to zone(', 'dist_').replace(')', '').strip().replace(' ', '_').lower()
            variables[clean] = {
                'values': data[col].values[indices].astype(float),
                'type': 'continuous',
            }

    # compartment
    pot_zone_cols = [c for c in data.columns
                     if 'Zone(Pot-' in str(c) and ' zone' not in c
                     and 'Distance' not in str(c)]
    home_col = [c for c in data.columns if 'Zone(Home' in str(c)
                and 'corner' not in c and 'Distance' not in str(c)]
    ladder_col = [c for c in data.columns if 'Zone(ladder' in str(c)
                  and 'Distance' not in str(c)]

    compartment = np.full(len(indices), 'Arena', dtype=object)
    if home_col:
        compartment[data[home_col[0]].values[indices] == 1] = 'Home'
    if ladder_col:
        compartment[data[ladder_col[0]].values[indices] == 1] = 'Ladder'
    if pot_zone_cols:
        at_pot = np.zeros(len(indices), dtype=bool)
        for c in pot_zone_cols:
            at_pot |= (data[c].values[indices] == 1)
        compartment[at_pot] = 'AtPot'
    variables['compartment'] = {
        'values': compartment, 'type': 'categorical',
        'classes': ['Home', 'Ladder', 'Arena', 'AtPot']
    }

    # scored binary behaviors with enough events
    skip_prefixes = ['Trial time', 'Recording', 'X ', 'Y ', 'Area', 'Elongation',
                     'Direction', 'Distance', 'Velocity', 'Zone(', 'Result']
    MIN_EVENTS = 50
    for col in data.columns:
        if col is None or str(col) == 'nan':
            continue
        cs = str(col)
        if any(cs.startswith(p) for p in skip_prefixes):
            continue
        vals_full = data[col].values
        if (vals_full == 1).sum() < MIN_EVENTS:
            continue
        vals_binned = vals_full[indices].astype(float)
        clean = cs.strip().replace(' ', '_').lower()
        variables[clean] = {
            'values': vals_binned.astype(int).astype(str),
            'type': 'categorical',
            'classes': ['0', '1'],
        }

    return variables


# ---------- Circular statistics ----------

def circ_mean(phi):
    """Mean resultant direction (radians, in [-pi, pi])."""
    return np.arctan2(np.sin(phi).mean(), np.cos(phi).mean())


def circ_var(phi):
    """Circular variance in [0, 1]. 0 = concentrated, 1 = uniform."""
    R = np.hypot(np.sin(phi).mean(), np.cos(phi).mean())
    return 1.0 - R


def rayleigh_test(phi):
    """Return (p, Z) for H0 uniform."""
    p, z = pc.rayleigh(phi)
    return float(p), float(z)


def circ_linear_corr(phi, x):
    """Circular-linear correlation (Mardia / Jammalamadaka)."""
    valid = np.isfinite(x) & np.isfinite(phi)
    if valid.sum() < 20:
        return np.nan, np.nan
    r = pc.corrcl(phi[valid], x[valid])
    # Permutation p-value
    rng = np.random.default_rng(0)
    n_perm = 500
    null = np.empty(n_perm)
    x_v = x[valid]
    phi_v = phi[valid]
    for i in range(n_perm):
        null[i] = pc.corrcl(phi_v, rng.permutation(x_v))
    p = (null >= r).mean()
    return float(r), float(p)


def circ_circ_corr(phi1, phi2):
    """Fisher-Lee circular-circular correlation with permutation p."""
    valid = np.isfinite(phi1) & np.isfinite(phi2)
    if valid.sum() < 20:
        return np.nan, np.nan
    a = phi1[valid]
    b = phi2[valid]
    r = pc.corrcc(a, b)
    rng = np.random.default_rng(0)
    n_perm = 500
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = pc.corrcc(a, rng.permutation(b))
    p = (np.abs(null) >= abs(r)).mean()
    return float(r), float(p)


def watson_williams_groups(phi, labels):
    """Watson-Williams one-way ANOVA across groups.
    Returns (F, p, group_means_dict, n_per_group)."""
    valid = np.isfinite(phi) & (labels != None) & (labels != 'nan')
    phi_v = phi[valid]
    lab_v = np.asarray(labels)[valid]
    groups = {}
    for g in np.unique(lab_v):
        pts = phi_v[lab_v == g]
        if len(pts) >= 10:
            groups[g] = pts
    if len(groups) < 2:
        return np.nan, np.nan, {}, {}
    try:
        p, table = pc.watson_williams(*groups.values())
        F = float(table.loc['Columns', 'F']) if 'Columns' in table.index else np.nan
    except Exception:
        F, p = np.nan, np.nan
    means = {g: float(circ_mean(v)) for g, v in groups.items()}
    n_per = {g: int(len(v)) for g, v in groups.items()}
    return float(F) if np.isfinite(F) else np.nan, float(p) if np.isfinite(p) else np.nan, means, n_per


def bh_correct(pvals):
    """Benjamini-Hochberg FDR-adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1)
    # enforce monotonicity
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty_like(adj)
    out[order] = adj
    return out


def unwrap_phase(phi):
    """Unwrap circular time series to continuous for differentiation."""
    return np.unwrap(phi)
