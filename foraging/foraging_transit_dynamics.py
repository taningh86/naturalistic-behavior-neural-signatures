"""
Transit dynamics analysis: neural trajectories during pot-to-pot transitions.

Analyses:
1. Full trajectory extraction (pot entry → departure → transit → arrival at next pot)
2. Rate of change: peaks in |d(metric)/dt| = candidate decision points
3. Gate dynamics pattern during transit
4. Destination-dependent divergence (P2→P4 vs P4→P2)
5. Pre-departure shift (last 2s vs earlier)
6. Change-point detection (CUSUM + variance)
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import spikeinterface.extractors as se
import torch
import torch.nn as nn
from torchdiffeq import odeint
from sklearn.decomposition import PCA
from scipy import stats as sp_stats
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
import warnings, sys

warnings.filterwarnings('ignore')

# =============================================================================
# CONSTANTS
# =============================================================================
FS = 30000
HIDDEN_SIZE = 32
ODE_GATE_HIDDEN = 64
D_SHARED = 32
ODE_DT = 1.0
PRED_BINS = 10
LHA_DEPTH_MAX = 1300
RSP_DEPTH_MIN = 1300
FR_BIN_MS = 100
FR_SAMP = int(FR_BIN_MS * FS / 1000)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RESAMPLE_HZ = 10
PAD_BEFORE = 3.0   # seconds before departure pot entry
PAD_AFTER = 3.0    # seconds after arrival pot exit
PREDEP_WINDOW = 2.0  # last N seconds for pre-departure analysis
DERIV_SMOOTH_WIN = 5  # samples (0.5s at 10Hz)
CP_MIN_SIDE = 5       # min samples on each side for change-point
MAX_TRANSIT_S = 60.0   # exclude transits longer than this

SESSIONS = [(2, 'fed'), (4, 'fed'), (6, 'fasted'), (8, 'fasted')]

with open("paths.yaml") as f:
    paths_config = yaml.safe_load(f)


# =============================================================================
# MODEL
# =============================================================================
class GRUODEFunc(nn.Module):
    def __init__(self, hidden_size, gate_hidden=64):
        super().__init__()
        self.update_gate = nn.Sequential(
            nn.Linear(hidden_size, gate_hidden), nn.Tanh(),
            nn.Linear(gate_hidden, hidden_size), nn.Sigmoid(),
        )
        self.candidate = nn.Sequential(
            nn.Linear(hidden_size, gate_hidden), nn.Tanh(),
            nn.Linear(gate_hidden, hidden_size), nn.Tanh(),
        )
    def forward(self, t, h):
        z = self.update_gate(h)
        n = self.candidate(h)
        return (1 - z) * (n - h)


class PooledGRUODE(nn.Module):
    def __init__(self, session_neuron_counts, d_shared, hidden_size,
                 gate_hidden=64, pred_steps=10):
        super().__init__()
        self.d_shared = d_shared
        self.hidden_size = hidden_size
        self.pred_steps = pred_steps
        self.input_projections = nn.ModuleDict()
        for sn, n_neurons in session_neuron_counts.items():
            self.input_projections[str(sn)] = nn.Linear(n_neurons, d_shared)
        self.ode_func = GRUODEFunc(hidden_size, gate_hidden)
        self.obs_cell = nn.GRUCell(input_size=d_shared, hidden_size=hidden_size)
        self.fc_shared = nn.Linear(hidden_size, d_shared)
        self.output_projections = nn.ModuleDict()
        for sn, n_neurons in session_neuron_counts.items():
            self.output_projections[str(sn)] = nn.Linear(d_shared, n_neurons)
        self.register_buffer('t_span', torch.tensor([0.0, ODE_DT]))


def load_model(region, state):
    model_path = Path("data") / f"gru_ode_10ms_poisson_{region}_{state}_model.pt"
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model = PooledGRUODE(
        checkpoint['neuron_counts'],
        checkpoint['config']['d_shared'],
        checkpoint['config']['hidden_size'],
        checkpoint['config'].get('gate_hidden', ODE_GATE_HIDDEN),
        checkpoint['config'].get('pred_steps', PRED_BINS),
    )
    model.to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def evaluate_flow(ode_func, points):
    h = torch.tensor(points, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        return ode_func(0.0, h).cpu().numpy()


# =============================================================================
# DATA LOADING (from foraging_transition_quantification.py)
# =============================================================================
def load_behavior_data(session_num):
    sp = paths_config['single_probe']['coordinates_1']['mouse01']['sessions']
    sc = sp[f'session_{session_num}']
    bp = sc.get('behavior')
    if not bp or not Path(bp).exists():
        return None, 0
    df = pd.read_csv(bp, header=None)
    n_bins = df.shape[1] - 1
    result = {}
    for i in range(df.shape[0]):
        name = str(df.iloc[i, 0]).strip()
        if name and name != 'nan':
            vals = pd.to_numeric(df.iloc[i, 1:], errors='coerce').values
            result[name] = vals
    return result, n_bins


def get_pot_signal(behav, pot_name, n_bins):
    signal = np.zeros(n_bins)
    if pot_name in behav:
        p = behav[pot_name]
        signal = np.where(~np.isnan(p) & (p > 0), 1, signal)
    zone_name = f'{pot_name} zone'
    if zone_name in behav:
        pz = behav[zone_name]
        signal = np.where(~np.isnan(pz) & (pz > 0), 1, signal)
    return signal


def find_dwell_events(signal, min_bins=10):
    events = []
    in_event = False
    start = 0
    for i in range(len(signal)):
        if not np.isnan(signal[i]) and signal[i] > 0:
            if not in_event:
                in_event = True
                start = i
        else:
            if in_event:
                dur = i - start
                if dur >= min_bins:
                    events.append((start, i - 1, dur))
                in_event = False
    if in_event:
        dur = len(signal) - start
        if dur >= min_bins:
            events.append((start, len(signal) - 1, dur))
    return events


def load_session(sess_num, state):
    print(f"\n  Loading S{sess_num} ({state})...")
    behav, n_bins = load_behavior_data(sess_num)
    exc_df = pd.read_csv("data/excursion_features_all_sessions.csv")
    exc_df = exc_df[exc_df["session"] == sess_num].copy()

    pot_signals = {}
    pot_dwells = {}
    for pot in ['Pot-1', 'Pot-2', 'Pot-3', 'Pot-4']:
        pot_signals[pot] = get_pot_signal(behav, pot, n_bins)
        pot_dwells[pot] = find_dwell_events(pot_signals[pot], min_bins=10)

    feeding = behav.get('Feeding', np.zeros(n_bins))
    feeding = np.where(np.isnan(feeding), 0, feeding)
    feed_onset_bin = np.argmax(feeding > 0) if np.any(feeding > 0) else n_bins
    discovery_time = feed_onset_bin * 0.1

    pot_visit_rows = []
    for pot_name, dwells in pot_dwells.items():
        for dw_start, dw_end, dw_dur in dwells:
            t_start = dw_start * 0.1
            t_end = dw_end * 0.1
            parent_exc = None
            for _, row in exc_df.iterrows():
                if row['start_time'] <= t_start <= row['end_time']:
                    parent_exc = int(row['excursion_idx'])
                    break
            pot_visit_rows.append({
                'pot': pot_name, 'start_s': t_start, 'end_s': t_end,
                'dwell_s': dw_dur * 0.1, 'excursion_idx': parent_exc,
                'pre_discovery': t_start < discovery_time,
            })
    pv_df = pd.DataFrame(pot_visit_rows).sort_values('start_s').reset_index(drop=True)

    # Neural data
    sp = paths_config['single_probe']['coordinates_1']['mouse01']['sessions']
    sorted_path = Path(sp[f'session_{sess_num}']['sorted'])
    sorting = se.read_kilosort(sorted_path)
    ci = pd.read_csv(sorted_path / "cluster_info.tsv", sep="\t")
    lc = "group" if "group" in ci.columns and ci["group"].eq("good").any() else "KSLabel"
    good = ci[ci[lc] == "good"]
    lha_ids = good[good["depth"] < LHA_DEPTH_MAX]["cluster_id"].values
    rsp_ids = good[good["depth"] >= RSP_DEPTH_MIN]["cluster_id"].values

    def get_fr(uids):
        amin = np.inf
        amax = 0
        sts = {}
        for u in uids:
            st = sorting.get_unit_spike_train(u)
            sts[u] = st
            if len(st) > 0:
                amin = min(amin, np.min(st))
                amax = max(amax, np.max(st))
        nb = int((amax - amin) / FR_SAMP) + 1
        d = np.zeros((nb, len(uids)), dtype=np.float32)
        for i, u in enumerate(uids):
            st = sts[u]
            if len(st) > 0:
                b = ((st - amin) // FR_SAMP).astype(int)
                b = b[(b >= 0) & (b < nb)]
                np.add.at(d[:, i], b, 1)
        return d / (FR_BIN_MS / 1000.0), amin

    region_data = {}
    for region, uids in [('LHA', lha_ids), ('RSP', rsp_ids)]:
        fr_data, fr_amin = get_fr(uids)
        h_states = np.load(f"data/gru_ode_10ms_hidden_{region.lower()}_s{sess_num}.npy")
        model = load_model(region.lower(), state)
        ode_func = model.ode_func
        pca = PCA(n_components=3).fit(h_states)
        h_pcs = pca.transform(h_states)
        region_data[region] = {
            'h_states': h_states, 'h_pcs': h_pcs,
            'ode_func': ode_func, 'pca': pca,
            'pop_fr': fr_data.mean(axis=1),
            'offset_10ms': fr_amin / FS,  # simplified: use same offset
            'offset_100ms': fr_amin / FS,
            'n_units': len(uids),
        }

    print(f"    {len(lha_ids)} LHA, {len(rsp_ids)} RSP good units")
    print(f"    Discovery: {discovery_time:.1f}s")
    return pv_df, region_data, discovery_time


# =============================================================================
# METRIC EXTRACTION AT SINGLE TIMEPOINT
# =============================================================================
def get_metrics_at_time(region_data, region, t_sec):
    """Return (FR, PC1, Flow, Gate) at a single timepoint."""
    rd = region_data[region]

    # FR: single 100ms bin
    fr = rd['pop_fr']
    fr_idx = int((t_sec - rd['offset_100ms']) / 0.1)
    fr_idx = np.clip(fr_idx, 0, len(fr) - 1)
    fr_val = fr[fr_idx]

    # PC1: single 10ms bin
    pcs = rd['h_pcs']
    h_idx = int((t_sec - rd['offset_10ms']) / 0.01)
    h_idx = np.clip(h_idx, 0, len(pcs) - 1)
    pc1_val = pcs[h_idx, 0]

    # Flow & Gate from hidden state
    h_states = rd['h_states']
    h_idx2 = np.clip(h_idx, 0, len(h_states) - 1)
    h_pt = h_states[h_idx2:h_idx2+1]
    dhdt = evaluate_flow(rd['ode_func'], h_pt)
    flow_val = np.linalg.norm(dhdt)
    h_t = torch.tensor(h_pt, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        gate_val = rd['ode_func'].update_gate(h_t).cpu().numpy().mean()

    return fr_val, pc1_val, flow_val, gate_val


def extract_trajectory(region_data, region, t_start, t_end, hz=RESAMPLE_HZ):
    """Extract FR, PC1, Flow, Gate at hz resolution over [t_start, t_end]."""
    n_pts = max(2, int((t_end - t_start) * hz))
    t_axis = np.linspace(t_start, t_end, n_pts)
    fr, pc1, flow, gate = [], [], [], []
    for t in t_axis:
        f, p, fl, g = get_metrics_at_time(region_data, region, t)
        fr.append(f)
        pc1.append(p)
        flow.append(fl)
        gate.append(g)
    return {
        't': t_axis, 'fr': np.array(fr), 'pc1': np.array(pc1),
        'flow': np.array(flow), 'gate': np.array(gate),
    }


# =============================================================================
# FIND TRANSIT SEGMENTS
# =============================================================================
def find_transit_segments(pv_df, discovery_time):
    """Find post-discovery P2<->P4 transitions with full timing info."""
    post = pv_df[~pv_df['pre_discovery']].copy()
    transits = []

    for exc_idx in post['excursion_idx'].dropna().unique():
        exc_visits = post[post['excursion_idx'] == exc_idx].sort_values('start_s')
        pots = exc_visits['pot'].values
        starts = exc_visits['start_s'].values
        ends = exc_visits['end_s'].values

        for i in range(len(pots) - 1):
            p1, p2 = pots[i], pots[i+1]
            if {p1, p2} == {'Pot-2', 'Pot-4'}:
                dep_end = ends[i]
                arr_start = starts[i+1]
                transit_dur = arr_start - dep_end
                if transit_dur > MAX_TRANSIT_S:
                    continue
                direction = f'{p1[-1]}->{p2[-1]}'
                transits.append({
                    'exc_idx': int(exc_idx),
                    'dep_pot': p1, 'arr_pot': p2,
                    'dep_start': starts[i], 'dep_end': dep_end,
                    'arr_start': arr_start, 'arr_end': ends[i+1],
                    'transit_dur': transit_dur,
                    'direction': direction,
                })
    return transits


# =============================================================================
# CHANGE-POINT DETECTION
# =============================================================================
def cusum_changepoint(trace, min_side=CP_MIN_SIDE):
    """CUSUM: find point of max cumulative deviation from mean."""
    if len(trace) < 2 * min_side:
        return len(trace) // 2, 0.0
    mean_val = np.mean(trace)
    cusum = np.cumsum(trace - mean_val)
    # Only consider points with enough data on each side
    valid = cusum[min_side:-min_side] if len(cusum) > 2 * min_side else cusum
    if len(valid) == 0:
        return len(trace) // 2, 0.0
    cp_local = np.argmax(np.abs(valid))
    cp_idx = cp_local + min_side
    score = np.abs(cusum[cp_idx]) / (np.std(trace) * np.sqrt(len(trace)) + 1e-10)
    return cp_idx, score


def variance_changepoint(trace, min_side=CP_MIN_SIDE):
    """Find point where variance changes most."""
    n = len(trace)
    if n < 2 * min_side:
        return n // 2, 0.0
    best_idx, best_score = n // 2, 0.0
    for k in range(min_side, n - min_side):
        v_before = np.var(trace[:k])
        v_after = np.var(trace[k:])
        score = abs(v_after - v_before) / (v_before + v_after + 1e-10)
        if score > best_score:
            best_score = score
            best_idx = k
    return best_idx, best_score


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("  TRANSIT DYNAMICS: Pot-to-Pot Neural Trajectories")
    print("=" * 70)

    all_sessions = {}
    all_transits = []
    regions = ['LHA', 'RSP']
    metrics = ['fr', 'pc1', 'flow', 'gate']
    metric_labels = {'fr': 'Pop FR (Hz)', 'pc1': 'PC1', 'flow': 'Flow speed',
                     'gate': 'Gate value'}

    # --- Load all sessions ---
    for sess, state in SESSIONS:
        pv_df, region_data, disc_t = load_session(sess, state)
        transits = find_transit_segments(pv_df, disc_t)
        print(f"    Post-disc transits: {len(transits)}")
        for tr in transits:
            tr['session'] = sess
            tr['state'] = state
            tr['region_data'] = region_data
            tr['pv_df'] = pv_df
        all_transits.extend(transits)
        all_sessions[sess] = {'region_data': region_data, 'pv_df': pv_df,
                              'discovery_time': disc_t, 'state': state}

    print(f"\n  Total transits: {len(all_transits)}")
    n_2to4 = sum(1 for t in all_transits if t['direction'] == '2->4')
    n_4to2 = sum(1 for t in all_transits if t['direction'] == '4->2')
    print(f"    P2->P4: {n_2to4}, P4->P2: {n_4to2}")

    # --- Extract all trajectories ---
    print(f"\n  Extracting trajectories...")
    for tr in all_transits:
        t_start = tr['dep_start'] - PAD_BEFORE
        t_end = tr['arr_end'] + PAD_AFTER
        tr['trajectories'] = {}
        for region in regions:
            tr['trajectories'][region] = extract_trajectory(
                tr['region_data'], region, t_start, t_end)
        dep_dur = tr['dep_end'] - tr['dep_start']
        arr_dur = tr['arr_end'] - tr['arr_start']
        print(f"    S{tr['session']} Exc {tr['exc_idx']} {tr['direction']}: "
              f"dep={dep_dur:.1f}s, transit={tr['transit_dur']:.1f}s, "
              f"arr={arr_dur:.1f}s")

    # =========================================================================
    # ANALYSIS 1: Trajectory visualization (per-session)
    # =========================================================================
    print(f"\n{'='*70}")
    print("  ANALYSIS 1: Trajectory visualization")
    print(f"{'='*70}")

    pot_colors = {'Pot-1': 'orange', 'Pot-2': 'red', 'Pot-3': 'blue',
                  'Pot-4': 'green'}
    from matplotlib.patches import Patch

    for sess, state in SESSIONS:
        sess_trans = [t for t in all_transits if t['session'] == sess]
        if not sess_trans:
            continue
        for region in regions:
            n_tr = len(sess_trans)
            fig, axes = plt.subplots(4, n_tr, figsize=(5 * n_tr, 12),
                                     squeeze=False)
            fig.suptitle(f'{region} -- S{sess} ({state}) Transit Trajectories\n'
                         f'Dashed=departure, dotted=arrival',
                         fontsize=13, fontweight='bold')
            pot_legend = [Patch(facecolor=c, alpha=0.3, label=p)
                          for p, c in pot_colors.items()]
            fig.legend(handles=pot_legend, loc='upper right', fontsize=9,
                       ncol=4, framealpha=0.8)

            for ci, tr in enumerate(sess_trans):
                traj = tr['trajectories'][region]
                t = traj['t']
                dep_end = tr['dep_end']
                arr_start = tr['arr_start']

                # Shade pot visits in this excursion
                exc_visits = tr['pv_df'][tr['pv_df']['excursion_idx'] == tr['exc_idx']]

                for mi, metric in enumerate(metrics):
                    ax = axes[mi, ci]
                    trace = traj[metric]
                    sm = uniform_filter1d(trace, max(3, DERIV_SMOOTH_WIN))
                    ax.plot(t, sm, 'k-', linewidth=1.5)

                    # Shade pot visits
                    for _, pv in exc_visits.iterrows():
                        if pv['end_s'] >= t[0] and pv['start_s'] <= t[-1]:
                            ax.axvspan(max(pv['start_s'], t[0]),
                                       min(pv['end_s'], t[-1]),
                                       alpha=0.2,
                                       color=pot_colors.get(pv['pot'], 'gray'))

                    ax.axvline(dep_end, color='black', ls='--', lw=1.5,
                               alpha=0.7)
                    ax.axvline(arr_start, color='black', ls=':', lw=1.5,
                               alpha=0.7)

                    if mi == 0:
                        ax.set_title(f"Exc {tr['exc_idx']}: {tr['direction']}\n"
                                     f"transit={tr['transit_dur']:.1f}s",
                                     fontsize=10, fontweight='bold')
                    if mi == 3:
                        ax.set_xlabel('Time (s)')
                    if ci == 0:
                        ax.set_ylabel(metric_labels[metric])

            plt.tight_layout(rect=[0, 0, 1, 0.93])
            fig_path = f'figures/foraging_transit_traj_s{sess}_{region.lower()}.png'
            plt.savefig(fig_path, dpi=100, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {fig_path}")

    # =========================================================================
    # ANALYSIS 2: Rate of change — departure-aligned
    # =========================================================================
    print(f"\n{'='*70}")
    print("  ANALYSIS 2: Rate of change (decision point candidates)")
    print(f"{'='*70}")

    roc_rows = []

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Rate of Change |d(metric)/dt| Aligned to Departure (t=0)\n'
                 'Each line = one transition | Dots = peaks',
                 fontsize=13, fontweight='bold')

    session_colors = {2: 'royalblue', 4: 'cornflowerblue',
                      6: 'indianred', 8: 'lightsalmon'}

    for ri, region in enumerate(regions):
        for mi, metric in enumerate(metrics):
            ax = axes[ri, mi]
            all_peak_times = []

            for tr in all_transits:
                traj = tr['trajectories'][region]
                t = traj['t']
                trace = traj[metric]
                sm = uniform_filter1d(trace, max(3, DERIV_SMOOTH_WIN))
                deriv = np.abs(np.diff(sm) * RESAMPLE_HZ)
                t_deriv = (t[:-1] + t[1:]) / 2 - tr['dep_end']  # align to departure

                color = session_colors[tr['session']]
                ax.plot(t_deriv, deriv, color=color, alpha=0.4, linewidth=0.8)

                # Find peaks
                if len(deriv) > 5:
                    prom = np.std(deriv) * 0.5
                    peaks, props = find_peaks(deriv, prominence=max(prom, 0.01))
                    for pk in peaks:
                        pk_time = t_deriv[pk]
                        # Focus on transit period: -2s to +transit+2s
                        if -2 < pk_time < tr['transit_dur'] + 2:
                            all_peak_times.append(pk_time)
                            ax.plot(pk_time, deriv[pk], 'o', color=color,
                                    markersize=3, alpha=0.6)
                            roc_rows.append({
                                'session': tr['session'], 'state': tr['state'],
                                'exc_idx': tr['exc_idx'],
                                'direction': tr['direction'],
                                'region': region, 'metric': metric,
                                'peak_time_rel_dep': pk_time,
                                'peak_magnitude': deriv[pk],
                            })

            ax.axvline(0, color='black', ls='--', lw=2, alpha=0.7,
                       label='Departure')
            median_transit = np.median([t['transit_dur'] for t in all_transits])
            ax.axvline(median_transit, color='green', ls=':', lw=2, alpha=0.7,
                       label=f'Med. arrival ({median_transit:.1f}s)')
            ax.set_title(f'{region} {metric_labels[metric]}', fontsize=11)
            if ri == 1:
                ax.set_xlabel('Time from departure (s)')
            if mi == 0:
                ax.set_ylabel('|d/dt|')
            if ri == 0 and mi == 0:
                ax.legend(fontsize=8)

            # Inset histogram of peak times
            if all_peak_times:
                ins = ax.inset_axes([0.6, 0.6, 0.38, 0.35])
                ins.hist(all_peak_times, bins=15, color='gray', alpha=0.7,
                         edgecolor='black', linewidth=0.3)
                ins.axvline(0, color='black', ls='--', lw=1)
                ins.set_xlabel('Peak time (s)', fontsize=6)
                ins.set_ylabel('Count', fontsize=6)
                ins.tick_params(labelsize=5)

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    plt.savefig('figures/foraging_transit_rateofchange.png', dpi=100,
                bbox_inches='tight')
    plt.close()
    print(f"  Saved: figures/foraging_transit_rateofchange.png")

    roc_df = pd.DataFrame(roc_rows)
    if len(roc_df) > 0:
        print(f"  Total peaks: {len(roc_df)}")
        for region in regions:
            for metric in metrics:
                sub = roc_df[(roc_df['region'] == region) &
                             (roc_df['metric'] == metric)]
                if len(sub) > 0:
                    med = np.median(sub['peak_time_rel_dep'])
                    print(f"    {region} {metric}: {len(sub)} peaks, "
                          f"median={med:+.2f}s from departure")

    # =========================================================================
    # ANALYSIS 3: Gate dynamics during transit
    # =========================================================================
    print(f"\n{'='*70}")
    print("  ANALYSIS 3: Gate dynamics during transit")
    print(f"{'='*70}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Gate Dynamics During Transit\n'
                 'Left: aligned to departure | Right: aligned to arrival',
                 fontsize=13, fontweight='bold')
    dir_colors = {'2->4': 'green', '4->2': 'red'}

    for ri, region in enumerate(regions):
        # Departure-aligned
        ax = axes[ri, 0]
        for tr in all_transits:
            traj = tr['trajectories'][region]
            t_rel = traj['t'] - tr['dep_end']
            gate = uniform_filter1d(traj['gate'], max(3, DERIV_SMOOTH_WIN))
            color = dir_colors[tr['direction']]
            ax.plot(t_rel, gate, color=color, alpha=0.4, linewidth=1)
        ax.axvline(0, color='black', ls='--', lw=2, label='Departure')
        ax.set_xlabel('Time from departure (s)')
        ax.set_ylabel(f'{region} Gate')
        ax.set_title(f'{region} — Departure-aligned')
        ax.set_xlim(-PAD_BEFORE, 10)
        if ri == 0:
            legend_handles = [
                Line2D([0], [0], color='green', label='P2->P4'),
                Line2D([0], [0], color='red', label='P4->P2'),
                Line2D([0], [0], color='black', ls='--', label='Departure'),
            ]
            ax.legend(handles=legend_handles, fontsize=8)

        # Arrival-aligned
        ax = axes[ri, 1]
        for tr in all_transits:
            traj = tr['trajectories'][region]
            t_rel = traj['t'] - tr['arr_start']
            gate = uniform_filter1d(traj['gate'], max(3, DERIV_SMOOTH_WIN))
            color = dir_colors[tr['direction']]
            ax.plot(t_rel, gate, color=color, alpha=0.4, linewidth=1)
        ax.axvline(0, color='black', ls=':', lw=2, label='Arrival')
        ax.set_xlabel('Time from arrival (s)')
        ax.set_ylabel(f'{region} Gate')
        ax.set_title(f'{region} — Arrival-aligned')
        ax.set_xlim(-10, PAD_AFTER)

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    plt.savefig('figures/foraging_transit_gate_pattern.png', dpi=100,
                bbox_inches='tight')
    plt.close()
    print(f"  Saved: figures/foraging_transit_gate_pattern.png")

    # =========================================================================
    # ANALYSIS 4: Destination-dependent divergence
    # =========================================================================
    print(f"\n{'='*70}")
    print("  ANALYSIS 4: Destination-dependent divergence")
    print(f"{'='*70}")

    g_2to4 = [t for t in all_transits if t['direction'] == '2->4']
    g_4to2 = [t for t in all_transits if t['direction'] == '4->2']

    # Common time axis relative to departure
    max_dur = 8.0  # seconds after departure
    t_common = np.arange(-PAD_BEFORE, max_dur, 1.0 / RESAMPLE_HZ)

    for region in regions:
        fig, axes = plt.subplots(5, 1, figsize=(12, 16), sharex=True)
        fig.suptitle(f'{region} — Destination-Dependent Divergence\n'
                     f'Green=P2->P4 (n={n_2to4}), Red=P4->P2 (n={n_4to2}) '
                     f'| Aligned to departure (t=0)',
                     fontsize=13, fontweight='bold')

        for mi, metric in enumerate(metrics):
            ax = axes[mi]
            traces_2to4 = np.full((len(g_2to4), len(t_common)), np.nan)
            traces_4to2 = np.full((len(g_4to2), len(t_common)), np.nan)

            for gi, tr in enumerate(g_2to4):
                traj = tr['trajectories'][region]
                t_rel = traj['t'] - tr['dep_end']
                trace = uniform_filter1d(traj[metric], max(3, DERIV_SMOOTH_WIN))
                for ti, tc in enumerate(t_common):
                    idx = np.argmin(np.abs(t_rel - tc))
                    if abs(t_rel[idx] - tc) < 0.15:
                        traces_2to4[gi, ti] = trace[idx]

            for gi, tr in enumerate(g_4to2):
                traj = tr['trajectories'][region]
                t_rel = traj['t'] - tr['dep_end']
                trace = uniform_filter1d(traj[metric], max(3, DERIV_SMOOTH_WIN))
                for ti, tc in enumerate(t_common):
                    idx = np.argmin(np.abs(t_rel - tc))
                    if abs(t_rel[idx] - tc) < 0.15:
                        traces_4to2[gi, ti] = trace[idx]

            # Mean +/- SEM where n >= 2
            for traces, color, label in [
                (traces_2to4, 'green', 'P2->P4'),
                (traces_4to2, 'red', 'P4->P2'),
            ]:
                n_valid = np.sum(~np.isnan(traces), axis=0)
                mask = n_valid >= 2
                mean_tr = np.nanmean(traces, axis=0)
                sem_tr = np.nanstd(traces, axis=0) / np.sqrt(n_valid + 1e-10)
                ax.plot(t_common[mask], mean_tr[mask], color=color,
                        linewidth=2, label=label)
                ax.fill_between(t_common[mask],
                                mean_tr[mask] - sem_tr[mask],
                                mean_tr[mask] + sem_tr[mask],
                                color=color, alpha=0.15)

            ax.axvline(0, color='black', ls='--', lw=2, alpha=0.7)
            ax.set_ylabel(metric_labels[metric])
            if mi == 0:
                ax.legend(fontsize=9)

        # Bottom panel: running Cohen's d
        ax = axes[4]
        for mi, metric in enumerate(metrics):
            traces_2to4 = np.full((len(g_2to4), len(t_common)), np.nan)
            traces_4to2 = np.full((len(g_4to2), len(t_common)), np.nan)
            for gi, tr in enumerate(g_2to4):
                traj = tr['trajectories'][region]
                t_rel = traj['t'] - tr['dep_end']
                trace = uniform_filter1d(traj[metric], max(3, DERIV_SMOOTH_WIN))
                for ti, tc in enumerate(t_common):
                    idx = np.argmin(np.abs(t_rel - tc))
                    if abs(t_rel[idx] - tc) < 0.15:
                        traces_2to4[gi, ti] = trace[idx]
            for gi, tr in enumerate(g_4to2):
                traj = tr['trajectories'][region]
                t_rel = traj['t'] - tr['dep_end']
                trace = uniform_filter1d(traj[metric], max(3, DERIV_SMOOTH_WIN))
                for ti, tc in enumerate(t_common):
                    idx = np.argmin(np.abs(t_rel - tc))
                    if abs(t_rel[idx] - tc) < 0.15:
                        traces_4to2[gi, ti] = trace[idx]

            cohens_d = np.full(len(t_common), np.nan)
            for ti in range(len(t_common)):
                v1 = traces_2to4[:, ti][~np.isnan(traces_2to4[:, ti])]
                v2 = traces_4to2[:, ti][~np.isnan(traces_4to2[:, ti])]
                if len(v1) >= 2 and len(v2) >= 2:
                    pooled_std = np.sqrt((np.var(v1) + np.var(v2)) / 2 + 1e-10)
                    cohens_d[ti] = (np.mean(v1) - np.mean(v2)) / pooled_std
            ax.plot(t_common, cohens_d, label=metric, linewidth=1.5)

        ax.axvline(0, color='black', ls='--', lw=2, alpha=0.7)
        ax.axhline(0.8, color='gray', ls=':', alpha=0.5, label='|d|=0.8')
        ax.axhline(-0.8, color='gray', ls=':', alpha=0.5)
        ax.set_ylabel("Cohen's d")
        ax.set_xlabel('Time from departure (s)')
        ax.legend(fontsize=8, ncol=5)

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        fig_path = f'figures/foraging_transit_divergence_{region.lower()}.png'
        plt.savefig(fig_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fig_path}")

    # =========================================================================
    # ANALYSIS 5: Pre-departure shift
    # =========================================================================
    print(f"\n{'='*70}")
    print("  ANALYSIS 5: Pre-departure shift (last 2s vs earlier)")
    print(f"{'='*70}")

    predep_rows = []
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Pre-Departure Neural Shift: Early vs Last 2s at Departure Pot\n'
                 'Lines connect early and late values within same transition',
                 fontsize=13, fontweight='bold')

    for ri, region in enumerate(regions):
        for mi, metric in enumerate(metrics):
            ax = axes[ri, mi]
            early_vals, late_vals = [], []

            for tr in all_transits:
                dep_start = tr['dep_start']
                dep_end = tr['dep_end']
                dep_dur = dep_end - dep_start

                if dep_dur < PREDEP_WINDOW + 0.5:
                    continue  # need at least 2.5s dwell

                # Early: from start to (end - PREDEP_WINDOW)
                traj_early = extract_trajectory(
                    tr['region_data'], region,
                    dep_start, dep_end - PREDEP_WINDOW, hz=RESAMPLE_HZ)
                # Late: last PREDEP_WINDOW seconds
                traj_late = extract_trajectory(
                    tr['region_data'], region,
                    dep_end - PREDEP_WINDOW, dep_end, hz=RESAMPLE_HZ)

                early_val = np.mean(traj_early[metric])
                late_val = np.mean(traj_late[metric])
                early_vals.append(early_val)
                late_vals.append(late_val)

                color = session_colors[tr['session']]
                ax.plot([0, 1], [early_val, late_val], '-o', color=color,
                        alpha=0.5, markersize=5)

                predep_rows.append({
                    'session': tr['session'], 'state': tr['state'],
                    'exc_idx': tr['exc_idx'], 'direction': tr['direction'],
                    'region': region, 'metric': metric,
                    'early': early_val, 'late': late_val,
                    'delta': late_val - early_val,
                })

            # Mean
            if early_vals:
                ax.plot([0, 1], [np.mean(early_vals), np.mean(late_vals)],
                        'k-o', linewidth=3, markersize=8, zorder=10)

                # Wilcoxon
                if len(early_vals) >= 3:
                    deltas = np.array(late_vals) - np.array(early_vals)
                    _, p = sp_stats.wilcoxon(deltas)
                    sig = '*' if p < 0.05 else 'ns'
                    ax.set_title(f'{region} {metric_labels[metric]}\n'
                                 f'p={p:.3f} ({sig})', fontsize=10)
                    if p < 0.05:
                        print(f"    {region} {metric}: late-early "
                              f"delta={np.mean(deltas):+.3f}, p={p:.4f}*")
                else:
                    ax.set_title(f'{region} {metric_labels[metric]}\nn<3',
                                 fontsize=10)

            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Early', 'Last 2s'])
            if mi == 0:
                ax.set_ylabel(region)

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    plt.savefig('figures/foraging_transit_predeparture.png', dpi=100,
                bbox_inches='tight')
    plt.close()
    print(f"  Saved: figures/foraging_transit_predeparture.png")

    predep_df = pd.DataFrame(predep_rows)
    predep_df.to_csv('data/foraging_transit_predeparture.csv', index=False)

    # =========================================================================
    # ANALYSIS 6: Change-point detection
    # =========================================================================
    print(f"\n{'='*70}")
    print("  ANALYSIS 6: Change-point detection (CUSUM)")
    print(f"{'='*70}")

    cp_rows = []

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Change-Point Timing Relative to Departure (CUSUM)\n'
                 'Histogram of detected change-points across all transitions',
                 fontsize=13, fontweight='bold')

    for ri, region in enumerate(regions):
        for mi, metric in enumerate(metrics):
            ax = axes[ri, mi]
            cp_times = []

            for tr in all_transits:
                traj = tr['trajectories'][region]
                trace = uniform_filter1d(traj[metric], max(3, DERIV_SMOOTH_WIN))
                t_axis = traj['t']

                # CUSUM
                cp_idx, cp_score = cusum_changepoint(trace)
                cp_time_abs = t_axis[min(cp_idx, len(t_axis)-1)]
                cp_time_rel = cp_time_abs - tr['dep_end']

                # Variance
                vcp_idx, vcp_score = variance_changepoint(trace)
                vcp_time_abs = t_axis[min(vcp_idx, len(t_axis)-1)]
                vcp_time_rel = vcp_time_abs - tr['dep_end']

                cp_times.append(cp_time_rel)

                cp_rows.append({
                    'session': tr['session'], 'state': tr['state'],
                    'exc_idx': tr['exc_idx'], 'direction': tr['direction'],
                    'region': region, 'metric': metric,
                    'cusum_cp_time_rel': cp_time_rel,
                    'cusum_score': cp_score,
                    'variance_cp_time_rel': vcp_time_rel,
                    'variance_score': vcp_score,
                })

            ax.hist(cp_times, bins=15, color='steelblue', alpha=0.7,
                    edgecolor='black', linewidth=0.3)
            ax.axvline(0, color='black', ls='--', lw=2, label='Departure')
            median_transit = np.median([t['transit_dur'] for t in all_transits])
            ax.axvline(median_transit, color='green', ls=':', lw=2,
                       label=f'Med. arrival')
            med_cp = np.median(cp_times)
            ax.axvline(med_cp, color='red', ls='-', lw=2,
                       label=f'Med. CP ({med_cp:+.1f}s)')
            ax.set_title(f'{region} {metric_labels[metric]}', fontsize=10)
            if ri == 1:
                ax.set_xlabel('Time from departure (s)')
            if mi == 0:
                ax.set_ylabel('Count')
            if ri == 0 and mi == 0:
                ax.legend(fontsize=7)

            print(f"    {region} {metric}: median CP = {med_cp:+.2f}s "
                  f"from departure")

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    plt.savefig('figures/foraging_transit_cp_summary.png', dpi=100,
                bbox_inches='tight')
    plt.close()
    print(f"  Saved: figures/foraging_transit_cp_summary.png")

    cp_df = pd.DataFrame(cp_rows)
    cp_df.to_csv('data/foraging_transit_changepoints.csv', index=False)
    print(f"  Saved: data/foraging_transit_changepoints.csv")

    # =========================================================================
    # MASTER CSV
    # =========================================================================
    master_rows = []
    for tr in all_transits:
        row = {
            'session': tr['session'], 'state': tr['state'],
            'exc_idx': tr['exc_idx'], 'direction': tr['direction'],
            'dep_pot': tr['dep_pot'], 'arr_pot': tr['arr_pot'],
            'dep_start': tr['dep_start'], 'dep_end': tr['dep_end'],
            'arr_start': tr['arr_start'], 'arr_end': tr['arr_end'],
            'transit_dur': tr['transit_dur'],
        }
        master_rows.append(row)
    master_df = pd.DataFrame(master_rows)
    master_df.to_csv('data/foraging_transit_dynamics.csv', index=False)
    print(f"  Saved: data/foraging_transit_dynamics.csv")

    print("\nDone!")


if __name__ == '__main__':
    main()
