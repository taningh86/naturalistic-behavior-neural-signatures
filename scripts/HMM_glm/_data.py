"""Data loading for the GLM-HMM pipeline.

Loads spike trains from the dual-probe dataset, applies QC, bins at 50 ms (configurable),
and concatenates sessions across metabolic states with per-bin one-hot input covariate.

Probe 0 = ACA (imec0), Probe 1 = LHA (imec1, depth filter 0-345 µm).

QC criteria (same as the wider pipeline):
  - probe 0 (ACA): KSLabel='good' + fr > 0.2 Hz
  - probe 1 (LHA): KSLabel='good' + fr > 0.2 Hz + amp > 43 µV + depth in [0, 345] µm

HFD sessions S20-S24 lack cluster_info.tsv → fall back to template-derived depths
using `templates.npy` + `channel_positions.npy`, with labels from `cluster_group.tsv`.
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import yaml


REPO = Path(__file__).resolve().parents[2]
PATHS_YAML = REPO / "paths.yaml"

LHA_DEPTH_MIN = 0
LHA_DEPTH_MAX = 345
P0_MIN_FR = 0.2
P1_MIN_FR = 0.2
P1_MIN_AMP = 43

FS_AP = 30000.0  # action-potential sample rate
BIN_S_DEFAULT = 0.050

# Sessions per phase × metabolic state (dual-probe coordinates_1 mouse01)
# Memory: S2 null. S10 = no food (kept by request). S16 P1 missing cluster_info.
SESSIONS_BY_PHASE_STATE = {
    "exploration": {
        "fed":    [3, 5, 7, 9],
        "fasted": [11, 13, 15],
        "HFD":    [19, 21, 23],
    },
    "foraging": {
        "fed":    [4, 6, 8, 10],
        "fasted": [12, 14, 16],
        "HFD":    [20, 22, 24],
    },
}
METABOLIC_STATES = ["fed", "fasted", "HFD"]   # one-hot order


def load_paths():
    with open(PATHS_YAML) as f:
        return yaml.safe_load(f)


def session_sorted_path(paths_data, sn, region):
    """Return Path to the sorted directory for (sn, region)."""
    s = paths_data["double_probe"]["coordinates_1"]["mouse01"]["sessions"]
    key = f"session_{sn}"
    if key not in s:
        return None
    sval = s[key]
    if region == "ACA":
        return Path(sval.get("probe_0_aca", {}).get("sorted") or "")
    elif region in ("LHA", "RSP"):
        return Path(sval.get("probe_1_lha_rsp", {}).get("sorted") or "")
    raise ValueError(region)


# ============================================================================
# Cluster depth fallback (HFD sessions without cluster_info.tsv)
# ============================================================================
def _compute_cluster_depths_fallback(sorted_path: Path) -> dict[int, float]:
    """Compute peak-channel y position per cluster from KS output."""
    templates_path = sorted_path / "templates.npy"
    chan_pos_path = sorted_path / "channel_positions.npy"
    if not templates_path.exists() or not chan_pos_path.exists():
        return {}
    templates = np.load(templates_path)           # (n_templates, T, n_channels)
    channel_positions = np.load(chan_pos_path)    # (n_channels, 2)
    peak_channels = np.argmax(np.max(np.abs(templates), axis=1), axis=1)
    depths = channel_positions[peak_channels, 1]
    spike_clusters_path = sorted_path / "spike_clusters.npy"
    spike_templates_path = sorted_path / "spike_templates.npy"
    if spike_clusters_path.exists() and spike_templates_path.exists():
        spike_clusters = np.load(spike_clusters_path).flatten()
        spike_templates = np.load(spike_templates_path).flatten()
        unique_clusters = np.unique(spike_clusters)
        out = {}
        for cid in unique_clusters:
            mask = spike_clusters == cid
            templates_for_cluster = spike_templates[mask]
            if len(templates_for_cluster) > 0:
                most_common_template = np.bincount(templates_for_cluster).argmax()
                if most_common_template < len(depths):
                    out[int(cid)] = float(depths[most_common_template])
        return out
    return {int(i): float(depths[i]) for i in range(len(depths))}


def _load_cluster_labels_fallback(sorted_path: Path) -> dict[int, str]:
    """Return cluster_id → group label ('good', 'mua', etc.) from cluster_group.tsv."""
    for fname in ("cluster_group.tsv", "cluster_KSLabel.tsv"):
        p = sorted_path / fname
        if p.exists():
            df = pd.read_csv(p, sep="\t")
            id_col = df.columns[0]
            label_col = df.columns[1]
            return {int(r[id_col]): str(r[label_col]).strip() for _, r in df.iterrows()}
    return {}


# ============================================================================
# Good-unit selection
# ============================================================================
def get_good_units(sorted_path: Path, region: str) -> np.ndarray:
    """Return cluster IDs passing the region's QC."""
    sorted_path = Path(sorted_path)
    ci = sorted_path / "cluster_info.tsv"
    if ci.exists():
        try:
            df = pd.read_csv(ci, sep="\t")
        except Exception:
            df = None
        if df is not None:
            label_col = ("group" if ("group" in df.columns
                                       and df["group"].eq("good").any())
                         else ("KSLabel" if "KSLabel" in df.columns else None))
            if label_col is not None:
                if region == "ACA":
                    m = (df[label_col] == "good") & (df.get("fr", 0) > P0_MIN_FR)
                elif region == "LHA":
                    m = ((df[label_col] == "good")
                          & (df.get("fr", 0) > P1_MIN_FR)
                          & (df.get("amp", 0) > P1_MIN_AMP)
                          & (df.get("depth", -1) >= LHA_DEPTH_MIN)
                          & (df.get("depth", -1) <= LHA_DEPTH_MAX))
                else:
                    return np.array([], dtype=int)
                return df.loc[m, "cluster_id"].values.astype(int)

    # Fallback for HFD sessions w/o cluster_info.tsv: cluster_group + template depths
    labels = _load_cluster_labels_fallback(sorted_path)
    if not labels:
        return np.array([], dtype=int)
    good_ids = [cid for cid, lab in labels.items() if lab == "good"]
    if not good_ids:
        return np.array([], dtype=int)
    if region == "ACA":
        # No depth filter, no amp filter applied — best-effort. fr unknown.
        return np.array(good_ids, dtype=int)
    depths = _compute_cluster_depths_fallback(sorted_path)
    if not depths:
        return np.array([], dtype=int)
    out = [cid for cid in good_ids
           if cid in depths and LHA_DEPTH_MIN <= depths[cid] <= LHA_DEPTH_MAX]
    return np.array(out, dtype=int)


# ============================================================================
# Spike binning
# ============================================================================
def bin_spikes(sorted_path: Path, cluster_ids: np.ndarray,
                duration_s: float, bin_s: float = BIN_S_DEFAULT) -> np.ndarray:
    """Bin spike trains into (n_bins, n_units) int counts."""
    sorted_path = Path(sorted_path)
    st_path = sorted_path / "spike_times.npy"
    sc_path = sorted_path / "spike_clusters.npy"
    if not st_path.exists() or not sc_path.exists() or not len(cluster_ids):
        return np.zeros((int(duration_s / bin_s), 0), dtype=np.int32)
    st = np.load(st_path).astype(np.int64).ravel() / FS_AP        # in seconds
    sc = np.load(sc_path).astype(np.int64).ravel()
    keep = np.isin(sc, cluster_ids)
    st = st[keep]; sc = sc[keep]
    # Map cluster_id → column index
    cid_to_col = {int(c): i for i, c in enumerate(cluster_ids)}
    n_bins = int(np.ceil(duration_s / bin_s))
    bin_idx = (st / bin_s).astype(np.int64)
    valid = (bin_idx >= 0) & (bin_idx < n_bins)
    st_b = bin_idx[valid]
    sc_b = sc[valid]
    counts = np.zeros((n_bins, len(cluster_ids)), dtype=np.int32)
    for tb, c in zip(st_b, sc_b):
        col = cid_to_col.get(int(c))
        if col is not None:
            counts[tb, col] += 1
    return counts


def foraging_duration_s(cfg, sn, paths_data=None) -> Optional[float]:
    """Best-effort foraging-phase duration. Tries the HMM-binned npz first
    (most accurate, fed+fasted only). Falls back to max spike time on ACA
    (used for HFD sessions which lack the HMM binned npz)."""
    binned_root = REPO / cfg["out_dirs"]["binned"]
    f = binned_root / f"session_{sn}.npz"
    if f.exists():
        d = np.load(f, allow_pickle=True)
        tt = np.asarray(d["trial_time"], dtype=np.float64)
        HMM_BIN = 0.480
        return float(tt[-1] + HMM_BIN)
    if paths_data is None:
        return None
    return spike_max_time_s(paths_data, sn, "ACA")


def exploration_duration_s(sn, paths_data) -> Optional[float]:
    """Exploration-phase duration from max spike time on ACA."""
    return spike_max_time_s(paths_data, sn, "ACA")


def spike_max_time_s(paths_data, sn, region) -> Optional[float]:
    sp = session_sorted_path(paths_data, sn, region)
    if sp is None or not sp.exists():
        return None
    st_path = sp / "spike_times.npy"
    if not st_path.exists():
        return None
    st = np.load(st_path).astype(np.int64).ravel()
    return float(st.max() / FS_AP) if len(st) else None


# ============================================================================
# Build concatenated dataset for a (region, phase)
# ============================================================================
@dataclass
class SessionSequence:
    session_num: int
    metabolic_state: str
    duration_s: float
    counts: np.ndarray            # (T, n_units_session)
    cluster_ids: np.ndarray
    n_bins: int

@dataclass
class GroupedData:
    region: str
    phase: str
    bin_s: float
    sequences: list[SessionSequence]
    unit_ids_per_session: list[np.ndarray]
    metabolic_states_per_session: list[str]
    metabolic_state_order: list[str]


def load_grouped(region: str, phase: str, cfg, bin_s: float = BIN_S_DEFAULT,
                  verbose: bool = True) -> GroupedData:
    """Load all eligible sessions for a (region, phase), bin at bin_s, return."""
    assert region in ("ACA", "LHA"), region
    assert phase in ("exploration", "foraging"), phase
    paths_data = load_paths()
    sessions_by_state = SESSIONS_BY_PHASE_STATE[phase]

    seqs = []
    for state in METABOLIC_STATES:
        for sn in sessions_by_state[state]:
            sp = session_sorted_path(paths_data, sn, region)
            if sp is None or not sp.exists():
                if verbose:
                    print(f"  [skip] S{sn} {region}: sorted path missing", flush=True)
                continue
            uids = get_good_units(sp, region)
            if not len(uids):
                if verbose:
                    print(f"  [skip] S{sn} {region}: 0 good units", flush=True)
                continue
            # duration
            if phase == "foraging":
                dur = foraging_duration_s(cfg, sn, paths_data=paths_data)
            else:
                dur = exploration_duration_s(sn, paths_data)
            if dur is None or dur < 60:
                if verbose:
                    print(f"  [skip] S{sn} {region}: bad duration ({dur})", flush=True)
                continue
            counts = bin_spikes(sp, uids, dur, bin_s=bin_s)
            if counts.shape[0] < 100:
                if verbose:
                    print(f"  [skip] S{sn} {region}: too few bins", flush=True)
                continue
            if verbose:
                print(f"  S{sn:>2} ({state:>6}) {region}: dur={dur:.1f}s, "
                      f"n_units={counts.shape[1]}, n_bins={counts.shape[0]}",
                      flush=True)
            seqs.append(SessionSequence(
                session_num=sn, metabolic_state=state,
                duration_s=float(dur),
                counts=counts, cluster_ids=uids,
                n_bins=counts.shape[0],
            ))
    grouped = GroupedData(
        region=region, phase=phase, bin_s=bin_s,
        sequences=seqs,
        unit_ids_per_session=[s.cluster_ids for s in seqs],
        metabolic_states_per_session=[s.metabolic_state for s in seqs],
        metabolic_state_order=METABOLIC_STATES,
    )
    return grouped


def session_one_hot_input(metabolic_state: str, T: int) -> np.ndarray:
    """Build (T, 3) one-hot input matrix indexing [fed, fasted, HFD]."""
    u = np.zeros((T, len(METABOLIC_STATES)), dtype=np.float32)
    idx = METABOLIC_STATES.index(metabolic_state)
    u[:, idx] = 1.0
    return u


def pool_input_space(grouped: GroupedData) -> tuple[np.ndarray, dict[int, list[int]]]:
    """Construct the union of unit IDs across sessions for this region.
    For independent Poisson emissions, the "input space" must be the same set
    of units across all sequences passed to ssm. We pool unit IDs and align
    each session's count matrix to that pooled column index.

    Returns:
      pooled_ids: array of unique unit IDs (cluster_id) across sessions
      session_to_pooled_cols: dict session_num → list of column indices in pooled
    """
    # Cluster IDs from different sessions are NOT comparable (independent sortings).
    # So the pooled "input space" is actually a stacked set: per (session, unit)
    # = a unique column. We treat each session's units as distinct units in the
    # pooled emission model, with zeros in the columns for other sessions' units.
    # This loses cross-session unit identity but is the simplest joint fit.
    return _stack_session_units(grouped)


def _stack_session_units(grouped: GroupedData):
    """Create a global column ordering where each (session, unit) pair has a
    unique column. Returns the per-sequence count matrices padded with zeros to
    the global width."""
    seq_widths = [s.counts.shape[1] for s in grouped.sequences]
    n_units_total = sum(seq_widths)
    pooled_ids = []
    session_to_pooled_cols = {}
    cursor = 0
    for s in grouped.sequences:
        cols = list(range(cursor, cursor + s.counts.shape[1]))
        session_to_pooled_cols[s.session_num] = cols
        for cid in s.cluster_ids:
            pooled_ids.append((s.session_num, int(cid)))
        cursor += s.counts.shape[1]
    pooled_ids = np.array(pooled_ids, dtype=np.int64)
    return pooled_ids, session_to_pooled_cols


def prepare_for_fit(grouped: GroupedData):
    """Prepare per-session (counts, input) arrays sized to the pooled unit space.

    Returns:
      datasets: list of dicts with 'counts' (T, D_total) and 'input' (T, M) per session
      D_total: total pooled unit count
      M: input dimensionality (= len(METABOLIC_STATES))
      pooled_ids: (D_total, 2) array of (session_num, cluster_id) per column
      session_to_pooled_cols: dict
    """
    pooled_ids, session_to_pooled_cols = _stack_session_units(grouped)
    D_total = pooled_ids.shape[0]
    M = len(METABOLIC_STATES)
    datasets = []
    for s in grouped.sequences:
        T = s.counts.shape[0]
        full_counts = np.zeros((T, D_total), dtype=np.int32)
        cols = session_to_pooled_cols[s.session_num]
        full_counts[:, cols] = s.counts
        inp = session_one_hot_input(s.metabolic_state, T)
        datasets.append(dict(
            session_num=s.session_num,
            metabolic_state=s.metabolic_state,
            duration_s=s.duration_s,
            counts=full_counts,
            input=inp,
            session_cols=cols,
        ))
    return datasets, D_total, M, pooled_ids, session_to_pooled_cols
