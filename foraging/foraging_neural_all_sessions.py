"""
Foraging neural signatures -- All sessions (S2, S4, S6, S8).

Generalized version of foraging_neural_s2.py. Runs all 10 analyses per
session with graceful degradation for short pre-discovery windows (fasted
sessions). Saves per-session figures/CSVs plus a combined metrics CSV.

Sessions: S2 (Fed), S4 (Fed), S6 (Fasted), S8 (Fasted)
Models: fed GRU-ODE for S2/S4, fasted for S6/S8
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import spikeinterface.extractors as se
import torch
import torch.nn as nn
from torchdiffeq import odeint
from sklearn.decomposition import PCA
from scipy import stats as sp_stats
from scipy.ndimage import uniform_filter1d
import warnings, sys

warnings.filterwarnings('ignore')

# =============================================================================
# CONSTANTS
# =============================================================================
FS = 30000
BIN_MS = 10
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

# Analysis thresholds
MIN_VISITS_FOR_TREND = 3
MIN_BOTH_POT_EXC = 1
MIN_PRIOR_P4_EXC = 2

SESSION_CONFIG = {
    2: {'state': 'fed', 'phase': 'Foraging'},
    4: {'state': 'fed', 'phase': 'Foraging'},
    6: {'state': 'fasted', 'phase': 'Foraging'},
    8: {'state': 'fasted', 'phase': 'Foraging'},
}

with open("paths.yaml") as f:
    paths_config = yaml.safe_load(f)


# =============================================================================
# MODEL DEFINITION
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


def compute_local_divergence(ode_func, points, eps=0.01):
    divs = np.zeros(len(points))
    for d in range(HIDDEN_SIZE):
        perturb = np.zeros(HIDDEN_SIZE)
        perturb[d] = eps
        f_plus = evaluate_flow(ode_func, points + perturb)
        f_minus = evaluate_flow(ode_func, points - perturb)
        divs += (f_plus[:, d] - f_minus[:, d]) / (2 * eps)
    return divs


# =============================================================================
# BEHAVIOR HELPERS
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


# =============================================================================
# SESSION DATA LOADER
# =============================================================================
def load_session_data(sess_num, state):
    """Load all behavioral + neural data for one session."""
    print(f"\n{'='*70}")
    print(f"  Loading session {sess_num} data ({state})...")
    print(f"{'='*70}")
    sys.stdout.flush()

    sd = {}  # session data dict
    sd['sess'] = sess_num
    sd['state'] = state

    # --- Behavior ---
    behav, n_bins = load_behavior_data(sess_num)
    sd['behav'] = behav
    sd['n_bins'] = n_bins
    sd['session_length_s'] = n_bins * 0.1
    print(f"  Behavior: {n_bins} bins ({sd['session_length_s']:.1f}s)")

    # --- Excursions ---
    exc_df = pd.read_csv("data/excursion_features_all_sessions.csv")
    sd['exc_df'] = exc_df[exc_df["session"] == sess_num].copy()
    print(f"  Excursions: {len(sd['exc_df'])}")

    # --- Pot signals + dwells ---
    pot_signals = {}
    pot_dwells = {}
    for pot in ['Pot-1', 'Pot-2', 'Pot-3', 'Pot-4']:
        pot_signals[pot] = get_pot_signal(behav, pot, n_bins)
        pot_dwells[pot] = find_dwell_events(pot_signals[pot], min_bins=10)
    sd['pot_signals'] = pot_signals
    sd['pot_dwells'] = pot_dwells

    # --- Feeding / Digging ---
    feeding = behav.get('Feeding', np.zeros(n_bins))
    digging = behav.get('Digging', np.zeros(n_bins))
    feeding = np.where(np.isnan(feeding), 0, feeding)
    digging = np.where(np.isnan(digging), 0, digging)
    sd['feeding'] = feeding
    sd['digging'] = digging

    # Discovery time
    feed_onset_bin = np.argmax(feeding > 0) if np.any(feeding > 0) else n_bins
    sd['discovery_time'] = feed_onset_bin * 0.1
    dig_onset_bin = np.argmax(digging > 0) if np.any(digging > 0) else n_bins
    sd['first_dig_time'] = dig_onset_bin * 0.1
    print(f"  Discovery (first feed): {sd['discovery_time']:.1f}s")
    print(f"  First dig: {sd['first_dig_time']:.1f}s")

    # --- Build pot visit DataFrame ---
    pot_visit_rows = []
    for pot_name, dwells in pot_dwells.items():
        for dw_start, dw_end, dw_dur in dwells:
            t_start = dw_start * 0.1
            t_end = dw_end * 0.1
            parent_exc = None
            for _, row in sd['exc_df'].iterrows():
                if row['start_time'] <= t_start <= row['end_time']:
                    parent_exc = int(row['excursion_idx'])
                    break
            feed_bins = int(np.sum(feeding[dw_start:dw_end+1] > 0))
            dig_bins = int(np.sum(digging[dw_start:dw_end+1] > 0))
            pot_visit_rows.append({
                'pot': pot_name, 'start_s': t_start, 'end_s': t_end,
                'dwell_s': dw_dur * 0.1, 'excursion_idx': parent_exc,
                'feed_bins': feed_bins, 'dig_bins': dig_bins,
                'pre_discovery': t_start < sd['discovery_time'],
            })
    pv_df = pd.DataFrame(pot_visit_rows).sort_values('start_s').reset_index(drop=True)
    sd['pv_df'] = pv_df

    # --- Neural data ---
    sp = paths_config['single_probe']['coordinates_1']['mouse01']['sessions']
    sorted_path = Path(sp[f'session_{sess_num}']['sorted'])
    sorting = se.read_kilosort(sorted_path)
    ci = pd.read_csv(sorted_path / "cluster_info.tsv", sep="\t")
    lc = "group" if "group" in ci.columns and ci["group"].eq("good").any() else "KSLabel"
    good = ci[ci[lc] == "good"]
    lha_ids = good[good["depth"] < LHA_DEPTH_MAX]["cluster_id"].values
    rsp_ids = good[good["depth"] >= RSP_DEPTH_MIN]["cluster_id"].values
    print(f"  Good units: {len(lha_ids)} LHA, {len(rsp_ids)} RSP")

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

        allmin = np.inf
        for u in uids:
            st = sorting.get_unit_spike_train(u)
            if len(st) > 0:
                allmin = min(allmin, np.min(st))

        region_data[region] = {
            'h_states': h_states, 'h_pcs': h_pcs,
            'ode_func': ode_func, 'pca': pca,
            'pop_fr': fr_data.mean(axis=1),
            'offset_10ms': allmin / FS,
            'offset_100ms': fr_amin / FS,
            'n_units': len(uids),
        }

    sd['region_data'] = region_data
    print(f"  LHA FR: {len(region_data['LHA']['pop_fr'])} bins, "
          f"RSP FR: {len(region_data['RSP']['pop_fr'])} bins")
    print("  Neural data loaded.")
    sys.stdout.flush()

    return sd


# =============================================================================
# HELPER CLOSURES FOR SESSION DATA
# =============================================================================
def make_helpers(sd):
    """Create helper functions bound to a session's data."""
    rd = sd['region_data']

    def get_h_idx(region, t_sec):
        return max(0, int((t_sec - rd[region]['offset_10ms']) / 0.01))

    def get_fr_idx(region, t_sec):
        return max(0, int((t_sec - rd[region]['offset_100ms']) / 0.1))

    def get_peri_event_fr(region, center_sec, window_before=5.0, window_after=5.0):
        fr = rd[region]['pop_fr']
        offset = rd[region]['offset_100ms']
        c_idx = int((center_sec - offset) / 0.1)
        b_idx = max(0, c_idx - int(window_before / 0.1))
        a_idx = min(len(fr), c_idx + int(window_after / 0.1))
        t = (np.arange(b_idx, a_idx) - c_idx) * 0.1
        return t, fr[b_idx:a_idx]

    def get_peri_event_latent(region, center_sec, window_before=5.0, window_after=5.0):
        h_pcs = rd[region]['h_pcs']
        offset = rd[region]['offset_10ms']
        c_idx = int((center_sec - offset) / 0.01)
        b_idx = max(0, c_idx - int(window_before / 0.01))
        a_idx = min(len(h_pcs), c_idx + int(window_after / 0.01))
        t = (np.arange(b_idx, a_idx) - c_idx) * 0.01
        return t, h_pcs[b_idx:a_idx, 0], h_pcs[b_idx:a_idx, 1], h_pcs[b_idx:a_idx, 2]

    def get_flow_metrics(region, t_sec):
        h = rd[region]['h_states']
        ode_func = rd[region]['ode_func']
        idx = get_h_idx(region, t_sec)
        idx = min(idx, len(h) - 1)
        pt = h[idx:idx+1]
        dhdt = evaluate_flow(ode_func, pt)
        speed = np.linalg.norm(dhdt)
        ht = torch.tensor(pt, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            gate = ode_func.update_gate(ht).cpu().numpy().mean()
        return speed, gate

    def get_metrics_at_time(region, t):
        """Return FR, PC1, flow, gate at a time point."""
        fr_idx = get_fr_idx(region, t)
        fr = rd[region]['pop_fr']
        fw = fr[fr_idx:min(fr_idx+10, len(fr))]
        fr_val = np.mean(fw) if len(fw) > 0 else np.nan

        h_idx = get_h_idx(region, t)
        pcs = rd[region]['h_pcs']
        pw = pcs[h_idx:min(h_idx+100, len(pcs)), 0]
        pc1_val = np.mean(pw) if len(pw) > 0 else np.nan

        speed, gate = get_flow_metrics(region, t)
        return fr_val, pc1_val, speed, gate

    return {
        'get_h_idx': get_h_idx, 'get_fr_idx': get_fr_idx,
        'get_peri_event_fr': get_peri_event_fr,
        'get_peri_event_latent': get_peri_event_latent,
        'get_flow_metrics': get_flow_metrics,
        'get_metrics_at_time': get_metrics_at_time,
    }


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analysis_1_behavioral_sequence(sd, sess):
    """Save pot visit CSV and print sequence. Always runs."""
    pv_df = sd['pv_df']
    out_path = f'data/foraging_excursion_potvisits_s{sess}.csv'
    pv_df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path} ({len(pv_df)} pot visits)")

    pre_disc = pv_df[pv_df['pre_discovery']].copy()
    post_disc = pv_df[~pv_df['pre_discovery']].copy()
    n_p2_pre = len(pre_disc[pre_disc['pot'] == 'Pot-2'])
    n_p4_pre = len(pre_disc[pre_disc['pot'] == 'Pot-4'])
    n_p2_post = len(post_disc[post_disc['pot'] == 'Pot-2'])
    n_p4_post = len(post_disc[post_disc['pot'] == 'Pot-4'])

    print(f"  Pre-disc: {len(pre_disc)} visits (P2={n_p2_pre}, P4={n_p4_pre})")
    print(f"  Post-disc: {len(post_disc)} visits (P2={n_p2_post}, P4={n_p4_post})")

    # Print pre-disc sequences
    if len(pre_disc) > 0:
        for exc_idx, grp in pre_disc.groupby('excursion_idx'):
            grp_s = grp.sort_values('start_s')
            seq = ' -> '.join([f"{r['pot']}({r['start_s']:.0f}s)"
                               for _, r in grp_s.iterrows()])
            print(f"    Exc {exc_idx}: {seq}")

    return {
        'n_pre_disc': len(pre_disc), 'n_post_disc': len(post_disc),
        'n_p2_pre': n_p2_pre, 'n_p4_pre': n_p4_pre,
        'n_p2_post': n_p2_post, 'n_p4_post': n_p4_post,
        'discovery_time': sd['discovery_time'],
        'first_dig_time': sd['first_dig_time'],
    }


def analysis_2_within_excursion(sd, sess, helpers):
    """Within-excursion Pot-2 vs Pot-4 comparison."""
    pv_df = sd['pv_df']
    rd = sd['region_data']
    pre_disc = pv_df[pv_df['pre_discovery']].copy()

    # Find excursions with both Pot-2 and Pot-4
    both_excs = []
    if len(pre_disc) > 0:
        for exc_idx, grp in pre_disc.groupby('excursion_idx'):
            pots = set(grp['pot'].values)
            if 'Pot-2' in pots and 'Pot-4' in pots:
                both_excs.append(exc_idx)

    if len(both_excs) < MIN_BOTH_POT_EXC:
        print(f"  SKIP: Only {len(both_excs)} excursions with both P2+P4 "
              f"(need >= {MIN_BOTH_POT_EXC})")
        return None

    n_exc = len(both_excs)
    fig, axes = plt.subplots(n_exc, 4, figsize=(16, 3.5 * n_exc), squeeze=False)
    fig.suptitle(f'S{sess}: Within-Excursion Pot-2 vs Pot-4\n'
                 f'Same excursion controls for time/arousal', fontsize=13, fontweight='bold')

    get_peri_event_fr = helpers['get_peri_event_fr']
    get_peri_event_latent = helpers['get_peri_event_latent']

    for ei, exc_idx in enumerate(both_excs):
        grp = pre_disc[pre_disc['excursion_idx'] == exc_idx].sort_values('start_s')
        p2_visits = grp[grp['pot'] == 'Pot-2']
        p4_visits = grp[grp['pot'] == 'Pot-4']

        for ci, region in enumerate(['LHA', 'RSP']):
            # FR panel
            ax = axes[ei, ci * 2]
            for _, v in p2_visits.iterrows():
                t, fr = get_peri_event_fr(region, v['start_s'], 3, 5)
                ax.plot(t, fr, color='red', alpha=0.7, linewidth=1)
            for _, v in p4_visits.iterrows():
                t, fr = get_peri_event_fr(region, v['start_s'], 3, 5)
                ax.plot(t, fr, color='green', alpha=0.7, linewidth=1)
            ax.set_title(f'Exc {exc_idx} -- {region} FR')
            ax.set_xlabel('Time from pot arrival (s)')
            ax.set_ylabel('Pop FR (Hz)')
            if ei == 0:
                ax.legend(['Pot-2', 'Pot-4'], fontsize=7)

            # PC1 panel
            ax = axes[ei, ci * 2 + 1]
            for _, v in p2_visits.iterrows():
                t, pc1, _, _ = get_peri_event_latent(region, v['start_s'], 3, 5)
                ax.plot(t[::10], pc1[::10], color='red', alpha=0.7, linewidth=1)
            for _, v in p4_visits.iterrows():
                t, pc1, _, _ = get_peri_event_latent(region, v['start_s'], 3, 5)
                ax.plot(t[::10], pc1[::10], color='green', alpha=0.7, linewidth=1)
            ax.set_title(f'Exc {exc_idx} -- {region} PC1')
            ax.set_xlabel('Time from pot arrival (s)')
            ax.set_ylabel('PC1')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig_path = f'figures/foraging_neural_within_excursion_s{sess}.png'
    plt.savefig(fig_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")
    return {'n_both_pot_exc': len(both_excs)}


def analysis_34_learning_curves(sd, sess, helpers):
    """Across-excursion FR, PC1, flow, gate evolution at pot arrival."""
    pv_df = sd['pv_df']
    rd = sd['region_data']
    pre_disc = pv_df[pv_df['pre_discovery']].copy()
    get_metrics = helpers['get_metrics_at_time']

    metric_labels = ['Pop FR (Hz)', 'PC1', 'Flow speed', 'Gate value']
    fig, axes = plt.subplots(4, 4, figsize=(22, 16))
    fig.suptitle(f'S{sess}: Across-Excursion Neural State at Pot Arrival\n'
                 f'Pre-discovery visits only | Each dot = one pot visit',
                 fontsize=14, fontweight='bold')

    results = {}
    for pi, pot in enumerate(['Pot-2', 'Pot-4']):
        color = '#E53935' if pot == 'Pot-2' else '#43A047'

        for ri, region in enumerate(['LHA', 'RSP']):
            pot_visits = pre_disc[pre_disc['pot'] == pot].sort_values('start_s')
            n_visits = len(pot_visits)
            col = pi * 2 + ri

            if n_visits == 0:
                for mi in range(4):
                    axes[mi, col].set_title(f'{pot} -- {region} (no visits)')
                    axes[mi, col].text(0.5, 0.5, 'No data', ha='center',
                                       va='center', transform=axes[mi, col].transAxes)
                continue

            visit_times = pot_visits['start_s'].values
            visit_nums = np.arange(1, n_visits + 1)

            fr_vals, pc1_vals, flow_vals, gate_vals = [], [], [], []
            for t in visit_times:
                fr_v, pc1_v, flow_v, gate_v = get_metrics(region, t)
                fr_vals.append(fr_v)
                pc1_vals.append(pc1_v)
                flow_vals.append(flow_v)
                gate_vals.append(gate_v)

            all_metrics = [np.array(fr_vals), np.array(pc1_vals),
                           np.array(flow_vals), np.array(gate_vals)]

            for mi, (metric, label) in enumerate(zip(all_metrics, metric_labels)):
                ax = axes[mi, col]
                ax.scatter(visit_nums, metric, color=color, s=60, zorder=5,
                           edgecolors='black', linewidths=0.5)

                key_prefix = f'{region}_{pot.replace("-","")}'
                r_val, p_val = np.nan, np.nan
                if n_visits >= MIN_VISITS_FOR_TREND:
                    z = np.polyfit(visit_nums, metric, 1)
                    ax.plot(visit_nums, np.polyval(z, visit_nums), color=color,
                            linewidth=1.5, linestyle='--', alpha=0.6)
                    r_val, p_val = sp_stats.pearsonr(visit_nums, metric)
                    ax.text(0.05, 0.95, f'r={r_val:.2f}, p={p_val:.3f}',
                            transform=ax.transAxes, fontsize=9, va='top')

                results[f'{key_prefix}_{label.split()[0].lower()}_r'] = r_val
                results[f'{key_prefix}_{label.split()[0].lower()}_p'] = p_val

                if mi == 0:
                    ax.set_title(f'{pot} -- {region} (n={n_visits})', fontweight='bold')
                if mi == 3:
                    ax.set_xlabel('Visit number')
                ax.set_ylabel(label)

            # Print stats
            for mi, label in enumerate(['FR', 'PC1', 'Flow', 'Gate']):
                key_r = f'{region}_{pot.replace("-","")}_' + label.lower().split()[0] + '_r'
                key_p = f'{region}_{pot.replace("-","")}_' + label.lower().split()[0] + '_p'
                if not np.isnan(results.get(key_r, np.nan)):
                    pass  # printed below in batch

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig_path = f'figures/foraging_neural_across_visits_s{sess}.png'
    plt.savefig(fig_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")

    # Print notable trends
    for key, val in results.items():
        if key.endswith('_p') and not np.isnan(val) and val < 0.1:
            r_key = key[:-2] + '_r'
            if r_key in results:
                print(f"    {key}: r={results[r_key]:.3f}, p={val:.4f}")

    return results


def analysis_57_commitment(sd, sess, helpers):
    """Commitment excursion vs prior Pot-4 excursions."""
    pv_df = sd['pv_df']
    rd = sd['region_data']
    pre_disc = pv_df[pv_df['pre_discovery']].copy()
    exc_df = sd['exc_df']
    get_peri_event_fr = helpers['get_peri_event_fr']
    get_peri_event_latent = helpers['get_peri_event_latent']
    get_metrics = helpers['get_metrics_at_time']

    # Find commitment excursion
    commit_exc = None
    for _, row in exc_df.iterrows():
        if row['start_time'] <= sd['first_dig_time'] <= row['end_time']:
            commit_exc = row
            break

    if commit_exc is None:
        print("  SKIP: No commitment excursion found")
        return None

    # Prior Pot-4 excursions
    p4_pre_disc = pre_disc[pre_disc['pot'] == 'Pot-4'].sort_values('start_s')
    prior_p4_exc_idxs = p4_pre_disc['excursion_idx'].unique()
    prior_p4_exc_idxs = [e for e in prior_p4_exc_idxs
                         if e != commit_exc['excursion_idx']]

    if len(prior_p4_exc_idxs) < MIN_PRIOR_P4_EXC:
        print(f"  SKIP: Only {len(prior_p4_exc_idxs)} prior P4 excursions "
              f"(need >= {MIN_PRIOR_P4_EXC})")
        return None

    print(f"  Commitment: Exc {int(commit_exc.excursion_idx)}, "
          f"{commit_exc.start_time:.1f}-{commit_exc.end_time:.1f}s")
    print(f"  Prior Pot-4 excursions: {len(prior_p4_exc_idxs)}")

    # Find first Pot-4 arrival in commitment excursion
    commit_p4 = pv_df[(pv_df['excursion_idx'] == commit_exc['excursion_idx']) &
                       (pv_df['pot'] == 'Pot-4')].sort_values('start_s')
    if len(commit_p4) == 0:
        # Commitment dig might be at a different pot
        commit_any = pv_df[(pv_df['excursion_idx'] == commit_exc['excursion_idx']) &
                           (pv_df['dig_bins'] > 0)].sort_values('start_s')
        if len(commit_any) > 0:
            commit_arrival = commit_any.iloc[0]['start_s']
        else:
            commit_arrival = commit_exc['start_time'] + 2.0
    else:
        commit_arrival = commit_p4.iloc[0]['start_s']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'S{sess}: Commitment Excursion vs Prior Pot-4\n'
                 f'Commit: Exc {int(commit_exc.excursion_idx)} | '
                 f'Prior: {len(prior_p4_exc_idxs)} excursions',
                 fontsize=13, fontweight='bold')

    results = {}
    for ri, region in enumerate(['LHA', 'RSP']):
        # Commitment peri-event
        t_c, fr_c = get_peri_event_fr(region, commit_arrival, 5, 3)
        t_cl, pc1_c, _, _ = get_peri_event_latent(region, commit_arrival, 5, 3)

        # Prior peri-events
        prior_frs = []
        prior_pc1s = []
        prior_pre_frs = []
        for exc_idx in prior_p4_exc_idxs:
            exc_p4 = p4_pre_disc[p4_pre_disc['excursion_idx'] == exc_idx]
            if len(exc_p4) == 0:
                continue
            first_p4_t = exc_p4.iloc[0]['start_s']
            t_p, fr_p = get_peri_event_fr(region, first_p4_t, 5, 3)
            prior_frs.append(fr_p)
            t_pl, pc1_p, _, _ = get_peri_event_latent(region, first_p4_t, 5, 3)
            prior_pc1s.append(pc1_p)
            # Pre-arrival mean FR (-5 to -1s)
            fr_full = rd[region]['pop_fr']
            fr_idx_start = helpers['get_fr_idx'](region, first_p4_t - 5)
            fr_idx_end = helpers['get_fr_idx'](region, first_p4_t - 1)
            if fr_idx_end > fr_idx_start:
                prior_pre_frs.append(np.mean(fr_full[fr_idx_start:fr_idx_end]))

        # Commitment pre-arrival FR
        fr_full = rd[region]['pop_fr']
        fr_idx_start = helpers['get_fr_idx'](region, commit_arrival - 5)
        fr_idx_end = helpers['get_fr_idx'](region, commit_arrival - 1)
        commit_pre_fr = np.mean(fr_full[fr_idx_start:fr_idx_end]) if fr_idx_end > fr_idx_start else np.nan

        # FR traces
        ax = axes[ri, 0]
        for pfr in prior_frs:
            min_len = min(len(t_c), len(pfr))
            ax.plot(t_c[:min_len], pfr[:min_len], color='gray', alpha=0.3, linewidth=0.5)
        if len(prior_frs) > 0:
            max_len = max(len(pf) for pf in prior_frs)
            prior_mean = np.nanmean([np.pad(pf, (0, max_len-len(pf)),
                                            constant_values=np.nan)
                                     for pf in prior_frs], axis=0)
            ax.plot(t_c[:len(prior_mean)], prior_mean[:len(t_c)],
                    color='blue', linewidth=2, label=f'Prior mean (n={len(prior_frs)})')
        ax.plot(t_c[:len(fr_c)], fr_c[:len(t_c)], color='red', linewidth=2,
                label='Commitment')
        ax.set_title(f'{region} -- FR around Pot-4 arrival')
        ax.set_xlabel('Time from Pot-4 arrival (s)')
        ax.set_ylabel('Pop FR (Hz)')
        ax.legend(fontsize=8)

        # Pre-arrival bar
        ax = axes[ri, 1]
        prior_arr = np.array(prior_pre_frs)
        bars_x = ['Prior\n(non-commit)', 'Commitment']
        bars_y = [np.mean(prior_arr), commit_pre_fr]
        bars_err = [sp_stats.sem(prior_arr) if len(prior_arr) > 1 else 0, 0]
        ax.bar(bars_x, bars_y, yerr=bars_err, capsize=5,
               color=['cornflowerblue', 'salmon'], edgecolor='black')
        ax.scatter(np.zeros(len(prior_arr)), prior_arr, color='navy',
                   s=20, zorder=5, alpha=0.5)
        ax.set_ylabel(f'{region} mean FR (-5,-1s)')
        ax.set_title(f'{region} -- Pre-arrival FR comparison')

        results[f'{region}_commit_pre_fr'] = commit_pre_fr
        results[f'{region}_prior_mean_fr'] = np.mean(prior_arr) if len(prior_arr) > 0 else np.nan

        # PC1 traces
        ax = axes[ri, 2]
        for ppc in prior_pc1s:
            ax.plot(t_cl[:len(ppc):10], ppc[:len(t_cl):10],
                    color='gray', alpha=0.3, linewidth=0.5)
        ax.plot(t_cl[::10], pc1_c[::10], color='red', linewidth=2, label='Commitment')
        ax.set_title(f'{region} -- PC1 around Pot-4 arrival')
        ax.set_xlabel('Time from Pot-4 arrival (s)')
        ax.set_ylabel(f'{region} PC1')

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    fig_path = f'figures/foraging_commitment_s{sess}.png'
    plt.savefig(fig_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")
    return results


def analysis_6_continuous(sd, sess, helpers):
    """Continuous flow/gate/PC1/FR tracking. Always runs."""
    rd = sd['region_data']
    disc_t = sd['discovery_time']
    MAX_TIME = min(disc_t + 200, sd['session_length_s'])

    for region in ['LHA', 'RSP']:
        r = rd[region]
        h = r['h_states']
        offset = r['offset_10ms']

        # Subsample every 1s = 100 bins
        end_h_idx = min(int((MAX_TIME - offset) / 0.01), len(h) - 1)
        sub_idx = np.arange(0, end_h_idx, 100)
        sub_t = offset + sub_idx * 0.01
        sub_h = h[sub_idx]

        # Flow metrics
        dhdt = evaluate_flow(r['ode_func'], sub_h)
        flow_speed = np.linalg.norm(dhdt, axis=1)
        ht = torch.tensor(sub_h, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            gate = r['ode_func'].update_gate(ht).cpu().numpy().mean(axis=1)

        pc1 = r['h_pcs'][sub_idx, 0]

        # FR
        fr = r['pop_fr']
        fr_offset = r['offset_100ms']
        fr_end = min(int((MAX_TIME - fr_offset) / 0.1), len(fr) - 1)
        fr_t = fr_offset + np.arange(fr_end) * 0.1
        fr_vals = fr[:fr_end]

        # Smooth
        sm = max(3, int(15))  # 15-point smoothing
        flow_sm = uniform_filter1d(flow_speed, sm)
        gate_sm = uniform_filter1d(gate, sm)
        pc1_sm = uniform_filter1d(pc1, sm)

        fr_sm_win = max(3, int(15))
        fr_sm = uniform_filter1d(fr_vals, fr_sm_win)

        # Plot
        fig, axes = plt.subplots(5, 1, figsize=(16, 14), sharex=True)
        fig.suptitle(f'{region} -- Continuous Neural State (S{sess}, '
                     f'{sd["state"].capitalize()}, Foraging)\n'
                     f'Discovery at {disc_t:.0f}s | Pot visits marked',
                     fontsize=13, fontweight='bold')

        traces = [
            (sub_t, flow_sm, 'Flow speed', 'purple'),
            (sub_t, gate_sm, 'Gate value (z)', 'brown'),
            (sub_t, pc1_sm, 'PC1 position', 'navy'),
            (fr_t, fr_sm, f'Pop FR ({r["n_units"]} units, Hz)', 'black'),
        ]

        pot_colors = {'Pot-1': 'orange', 'Pot-2': 'red',
                      'Pot-3': 'blue', 'Pot-4': 'green'}

        from matplotlib.lines import Line2D
        for ai, (t_data, y_data, ylabel, color) in enumerate(traces):
            ax = axes[ai]
            ax.plot(t_data, y_data, color=color, linewidth=1)
            ax.set_ylabel(ylabel)
            ax.axvline(disc_t, color='red', linewidth=2, linestyle='--', alpha=0.7)
            if sd['first_dig_time'] < MAX_TIME:
                ax.axvline(sd['first_dig_time'], color='red', linewidth=1,
                           linestyle=':', alpha=0.5)
            # Shade pot visits
            for _, pv in sd['pv_df'].iterrows():
                if pv['start_s'] < MAX_TIME:
                    ax.axvspan(pv['start_s'], min(pv['end_s'], MAX_TIME),
                               alpha=0.15, color=pot_colors.get(pv['pot'], 'gray'))
            if ai == 0:
                legend_handles = [
                    Line2D([0], [0], color='red', linewidth=2, linestyle='--',
                           label=f'First feed ({disc_t:.0f}s)'),
                    Line2D([0], [0], color='red', linewidth=1, linestyle=':',
                           alpha=0.5, label=f'First dig ({sd["first_dig_time"]:.0f}s)'),
                ]
                ax.legend(handles=legend_handles, fontsize=8)

        # Behavior raster
        ax = axes[4]
        behav_rows = {'Feed': 5, 'Dig': 4, 'P4': 3, 'P3': 2, 'P2': 1, 'P1': 0}
        for _, pv in sd['pv_df'].iterrows():
            if pv['start_s'] < MAX_TIME:
                pot_key = pv['pot'].replace('ot-', '')
                y = behav_rows.get(pot_key, 0)
                c = pot_colors.get(pv['pot'], 'gray')
                ax.barh(y, pv['dwell_s'], left=pv['start_s'], height=0.6,
                        color=c, alpha=0.6)
        # Draw dig/feed bars from raw signals (not pot visit windows)
        max_bin = int(MAX_TIME / 0.1)
        for sig, row_name, bar_color in [
            (sd['feeding'], 'Feed', 'green'),
            (sd['digging'], 'Dig', 'brown'),
        ]:
            sig_clipped = sig[:max_bin]
            in_event = False
            ev_start = 0
            for bi in range(len(sig_clipped)):
                if sig_clipped[bi] > 0:
                    if not in_event:
                        in_event = True
                        ev_start = bi
                else:
                    if in_event:
                        t0 = ev_start * 0.1
                        dur = (bi - ev_start) * 0.1
                        ax.barh(behav_rows[row_name], dur, left=t0,
                                height=0.6, color=bar_color, alpha=0.8)
                        in_event = False
            if in_event:
                t0 = ev_start * 0.1
                dur = (len(sig_clipped) - ev_start) * 0.1
                ax.barh(behav_rows[row_name], dur, left=t0,
                        height=0.6, color=bar_color, alpha=0.8)
        ax.set_yticks(list(behav_rows.values()))
        ax.set_yticklabels(list(behav_rows.keys()))
        ax.set_ylabel('Behavior')
        ax.set_xlabel('Time in session (s)')
        ax.axvline(disc_t, color='red', linewidth=2, linestyle='--', alpha=0.7)

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        fig_path = f'figures/foraging_continuous_s{sess}_{region.lower()}.png'
        plt.savefig(fig_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fig_path}")

    return {'max_time': MAX_TIME}


def analysis_8_reward(sd, sess, helpers):
    """Peri-reward neural response. Always runs."""
    rd = sd['region_data']
    disc_t = sd['discovery_time']
    dig_t = sd['first_dig_time']

    # Adaptive pre-window
    min_offset = min(rd['LHA']['offset_10ms'], rd['RSP']['offset_10ms'])
    window_before = min(10.0, disc_t - min_offset - 1.0)
    window_before = max(2.0, window_before)
    window_after = 15.0

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle(f'S{sess}: Reward Onset -- Neural Response at First Feeding ({disc_t:.0f}s)\n'
                 f'Window: -{window_before:.0f}s to +{window_after:.0f}s',
                 fontsize=13, fontweight='bold')

    results = {}
    for ri, region in enumerate(['LHA', 'RSP']):
        # FR
        t_fr, fr = helpers['get_peri_event_fr'](region, disc_t, window_before, window_after)
        ax = axes[ri, 0]
        ax.plot(t_fr, fr, color='gray', alpha=0.5, linewidth=0.5)
        if len(fr) > 20:
            fr_sm = uniform_filter1d(fr, 20)
            ax.plot(t_fr, fr_sm, color='red', linewidth=2, label='2s smoothed')
        ax.axvline(0, color='green', linewidth=2, linestyle='--')
        ax.axvline(dig_t - disc_t, color='brown', linewidth=1, linestyle=':')
        ax.set_xlabel('Time from first feed (s)')
        ax.set_ylabel(f'{region} Pop FR (Hz)')
        ax.set_title(f'{region} -- Firing Rate')
        ax.legend(fontsize=8)

        # PC1-3
        t_l, pc1, pc2, pc3 = helpers['get_peri_event_latent'](
            region, disc_t, window_before, window_after)
        ss = 10
        ax = axes[ri, 1]
        ax.plot(t_l[::ss], pc1[::ss], color='navy', linewidth=1.5, label='PC1')
        ax.plot(t_l[::ss], pc2[::ss], color='teal', linewidth=1.5, label='PC2')
        ax.plot(t_l[::ss], pc3[::ss], color='orange', linewidth=1.5, label='PC3')
        ax.axvline(0, color='green', linewidth=2, linestyle='--')
        ax.set_xlabel('Time from first feed (s)')
        ax.set_ylabel(f'{region} PC score')
        ax.set_title(f'{region} -- Latent Trajectory')
        ax.legend(fontsize=8)

        # Flow + gate
        n_pts = int((window_before + window_after) * 10)
        rew_times = np.linspace(disc_t - window_before, disc_t + window_after, n_pts)
        speeds, gates = [], []
        for t in rew_times:
            s, g = helpers['get_flow_metrics'](region, t)
            speeds.append(s)
            gates.append(g)

        ax = axes[ri, 2]
        t_rel = rew_times - disc_t
        ax.plot(t_rel, speeds, color='purple', linewidth=1.5, label='Flow speed')
        ax2 = ax.twinx()
        ax2.plot(t_rel, gates, color='brown', linewidth=1.5, label='Gate', alpha=0.7)
        ax.axvline(0, color='green', linewidth=2, linestyle='--')
        ax.set_xlabel('Time from first feed (s)')
        ax.set_ylabel('Flow speed', color='purple')
        ax2.set_ylabel('Gate value', color='brown')
        ax.set_title(f'{region} -- Flow & Gate')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

        # Save traces for cross-session comparison
        results[f'{region}_reward_fr_t'] = t_fr
        results[f'{region}_reward_fr'] = fr
        results[f'{region}_reward_pc1_t'] = t_l
        results[f'{region}_reward_pc1'] = pc1

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    fig_path = f'figures/foraging_reward_s{sess}.png'
    plt.savefig(fig_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")

    # Save traces for cross-session script
    np.savez(f'data/foraging_reward_traces_s{sess}.npz',
             **{k: v for k, v in results.items() if isinstance(v, np.ndarray)})
    print(f"  Saved: data/foraging_reward_traces_s{sess}.npz")

    return {}


def analysis_9_pre_vs_post(sd, sess, helpers):
    """Pre vs post-feed pot visit comparison."""
    pv_df = sd['pv_df']
    rd = sd['region_data']
    get_metrics = helpers['get_metrics_at_time']

    pre_visits = pv_df[pv_df['pre_discovery'] == True].copy()
    post_visits = pv_df[(pv_df['pre_discovery'] == False) &
                        (pv_df['feed_bins'] == 0)].copy()

    print(f"  Pre-disc: {len(pre_visits)}, Post-disc (no feed): {len(post_visits)}")
    for pot in ['Pot-2', 'Pot-4']:
        n_pre = len(pre_visits[pre_visits['pot'] == pot])
        n_post = len(post_visits[post_visits['pot'] == pot])
        print(f"    {pot}: {n_pre} pre, {n_post} post")

    metric_labels = ['Pop FR (Hz)', 'PC1', 'Flow speed', 'Gate value']

    # --- Extended learning curves ---
    fig, axes = plt.subplots(4, 4, figsize=(22, 16))
    fig.suptitle(f'S{sess}: Pre vs Post-Feed Neural State at Pot Arrival\n'
                 f'Vertical line = discovery ({sd["discovery_time"]:.0f}s)',
                 fontsize=14, fontweight='bold')

    for pi, pot in enumerate(['Pot-2', 'Pot-4']):
        color = '#E53935' if pot == 'Pot-2' else '#43A047'

        for ri, region in enumerate(['LHA', 'RSP']):
            pot_pre = pre_visits[pre_visits['pot'] == pot].sort_values('start_s')
            pot_post = post_visits[post_visits['pot'] == pot].sort_values('start_s')
            all_pot = pd.concat([pot_pre, pot_post]).sort_values('start_s')
            col = pi * 2 + ri

            if len(all_pot) == 0:
                for mi in range(4):
                    axes[mi, col].text(0.5, 0.5, 'No data', ha='center',
                                       va='center', transform=axes[mi, col].transAxes)
                continue

            visit_times = all_pot['start_s'].values
            is_pre = all_pot['pre_discovery'].values.astype(bool)
            visit_nums = np.arange(1, len(visit_times) + 1)
            n_pre_pot = int(is_pre.sum())

            fr_vals, pc1_vals, flow_vals, gate_vals = [], [], [], []
            for t in visit_times:
                f, p, fl, g = get_metrics(region, t)
                fr_vals.append(f)
                pc1_vals.append(p)
                flow_vals.append(fl)
                gate_vals.append(g)

            all_m = [np.array(fr_vals), np.array(pc1_vals),
                     np.array(flow_vals), np.array(gate_vals)]

            for mi, (metric, label) in enumerate(zip(all_m, metric_labels)):
                ax = axes[mi, col]
                pre_mask = is_pre
                post_mask = ~is_pre

                ax.scatter(visit_nums[pre_mask], metric[pre_mask], color=color,
                           s=60, zorder=5, edgecolors='black', linewidths=0.5,
                           marker='o', label='Pre-feed')
                ax.scatter(visit_nums[post_mask], metric[post_mask], color=color,
                           s=60, zorder=5, edgecolors='black', linewidths=0.5,
                           marker='s', alpha=0.5, label='Post-feed')

                if n_pre_pot > 0 and n_pre_pot < len(visit_nums):
                    ax.axvline(n_pre_pot + 0.5, color='red', linewidth=2,
                               linestyle='--', alpha=0.7)

                if pre_mask.sum() >= 3:
                    x_p = visit_nums[pre_mask]
                    z = np.polyfit(x_p, metric[pre_mask], 1)
                    ax.plot(x_p, np.polyval(z, x_p), color=color,
                            linewidth=1.5, linestyle='--', alpha=0.6)
                if post_mask.sum() >= 3:
                    x_po = visit_nums[post_mask]
                    z = np.polyfit(x_po, metric[post_mask], 1)
                    ax.plot(x_po, np.polyval(z, x_po), color=color,
                            linewidth=1.5, linestyle=':', alpha=0.6)

                if mi == 0:
                    ax.set_title(f'{pot} -- {region}', fontweight='bold')
                if mi == 3:
                    ax.set_xlabel('Visit number')
                ax.set_ylabel(label)
                if mi == 0 and pi == 0 and ri == 0:
                    ax.legend(fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig_path = f'figures/foraging_pre_vs_post_visits_s{sess}.png'
    plt.savefig(fig_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")

    # --- Stats summary ---
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    fig.suptitle(f'S{sess}: Pre vs Post-Feed Mean Neural State\nMann-Whitney U',
                 fontsize=14, fontweight='bold')

    results = {}
    for pi, pot in enumerate(['Pot-2', 'Pot-4']):
        pot_pre = pre_visits[pre_visits['pot'] == pot].sort_values('start_s')
        pot_post = post_visits[post_visits['pot'] == pot].sort_values('start_s')

        for ri, region in enumerate(['LHA', 'RSP']):
            col_bar = pi * 2 + ri
            col_txt = col_bar  # use same axes for text overlay

            if len(pot_pre) < 2 or len(pot_post) < 2:
                axes[ri, col_bar].text(0.5, 0.5,
                    f'{pot} {region}\nPre={len(pot_pre)}, Post={len(pot_post)}\nInsufficient data',
                    ha='center', va='center', transform=axes[ri, col_bar].transAxes,
                    fontsize=10)
                axes[ri, col_bar].set_title(f'{pot} -- {region}')
                continue

            metrics_pre = {'FR': [], 'PC1': [], 'Flow': [], 'Gate': []}
            metrics_post = {'FR': [], 'PC1': [], 'Flow': [], 'Gate': []}

            for visits, store in [(pot_pre, metrics_pre), (pot_post, metrics_post)]:
                for t in visits['start_s'].values:
                    f, p, fl, g = get_metrics(region, t)
                    store['FR'].append(f)
                    store['PC1'].append(p)
                    store['Flow'].append(fl)
                    store['Gate'].append(g)

            metric_names = ['FR', 'PC1', 'Flow', 'Gate']
            pre_means = [np.nanmean(metrics_pre[m]) for m in metric_names]
            post_means = [np.nanmean(metrics_post[m]) for m in metric_names]
            pre_sems = [sp_stats.sem([v for v in metrics_pre[m] if not np.isnan(v)])
                        for m in metric_names]
            post_sems = [sp_stats.sem([v for v in metrics_post[m] if not np.isnan(v)])
                         for m in metric_names]

            ax = axes[ri, col_bar]
            x = np.arange(len(metric_names))
            w = 0.35
            ax.bar(x - w/2, pre_means, w, yerr=pre_sems, capsize=4,
                   color='cornflowerblue', edgecolor='black', label='Pre')
            ax.bar(x + w/2, post_means, w, yerr=post_sems, capsize=4,
                   color='salmon', edgecolor='black', label='Post')
            ax.set_xticks(x)
            ax.set_xticklabels(metric_names)
            ax.set_title(f'{pot} -- {region} (n={len(pot_pre)}/{len(pot_post)})',
                         fontweight='bold')
            if pi == 0 and ri == 0:
                ax.legend(fontsize=8)

            # Stats
            key_prefix = f'{region}_{pot.replace("-","")}'
            for m in metric_names:
                pre_arr = np.array(metrics_pre[m])
                post_arr = np.array(metrics_post[m])
                pre_arr = pre_arr[~np.isnan(pre_arr)]
                post_arr = post_arr[~np.isnan(post_arr)]
                if len(pre_arr) >= 2 and len(post_arr) >= 2:
                    u, p_val = sp_stats.mannwhitneyu(pre_arr, post_arr,
                                                     alternative='two-sided')
                    results[f'{key_prefix}_{m}_U'] = u
                    results[f'{key_prefix}_{m}_p'] = p_val
                    results[f'{key_prefix}_{m}_pre_med'] = np.median(pre_arr)
                    results[f'{key_prefix}_{m}_post_med'] = np.median(post_arr)
                    sig = '*' if p_val < 0.05 else ''
                    print(f"    {pot} {region} {m}: pre={np.median(pre_arr):.3f}, "
                          f"post={np.median(post_arr):.3f}, p={p_val:.4f} {sig}")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig_path = f'figures/foraging_pre_vs_post_stats_s{sess}.png'
    plt.savefig(fig_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")
    return results


def analysis_10_transitions(sd, sess, helpers):
    """Pot-2 <-> Pot-4 transition dynamics."""
    pv_df = sd['pv_df']
    rd = sd['region_data']
    disc_t = sd['discovery_time']

    # Find all excursions with both P2 and P4
    transition_excursions = []
    for exc_idx, grp in pv_df.groupby('excursion_idx'):
        pots = set(grp['pot'].values)
        if 'Pot-2' not in pots or 'Pot-4' not in pots:
            continue
        visits = list(grp.sort_values('start_s').itertuples())
        for i in range(len(visits) - 1):
            for j in range(i + 1, len(visits)):
                v1, v2 = visits[i], visits[j]
                if v1.pot == 'Pot-2' and v2.pot == 'Pot-4':
                    transition_excursions.append({
                        'exc_idx': exc_idx, 'direction': 'P2->P4',
                        'pot1': 'Pot-2', 'pot2': 'Pot-4',
                        't1_start': v1.start_s, 't1_end': v1.end_s,
                        't2_start': v2.start_s, 't2_end': v2.end_s,
                        'pre_discovery': v1.start_s < disc_t,
                        'gap_s': v2.start_s - v1.end_s,
                    })
                    break
                elif v1.pot == 'Pot-4' and v2.pot == 'Pot-2':
                    transition_excursions.append({
                        'exc_idx': exc_idx, 'direction': 'P4->P2',
                        'pot1': 'Pot-4', 'pot2': 'Pot-2',
                        't1_start': v1.start_s, 't1_end': v1.end_s,
                        't2_start': v2.start_s, 't2_end': v2.end_s,
                        'pre_discovery': v1.start_s < disc_t,
                        'gap_s': v2.start_s - v1.end_s,
                    })
                    break

    trans_df = pd.DataFrame(transition_excursions)
    if len(trans_df) > 0:
        trans_df = trans_df.drop_duplicates(
            subset=['exc_idx', 'direction']).reset_index(drop=True)

    n_trans = len(trans_df)
    print(f"  Found {n_trans} P2<->P4 transitions")

    if n_trans == 0:
        print("  SKIP: No P2<->P4 transitions found")
        return {'n_transitions': 0}

    for _, tr in trans_df.iterrows():
        phase = 'PRE' if tr['pre_discovery'] else 'POST'
        print(f"    Exc {tr['exc_idx']:.0f}: {tr['direction']} "
              f"({tr['t1_start']:.0f}s->{tr['t2_start']:.0f}s) [{phase}]")

    PAD_BEFORE = 3.0
    PAD_AFTER = 3.0
    RESAMPLE_HZ = 10
    pot_colors = {'Pot-1': 'orange', 'Pot-2': 'red',
                  'Pot-3': 'blue', 'Pot-4': 'green'}

    for region in ['LHA', 'RSP']:
        fig, axes = plt.subplots(5, n_trans, figsize=(5 * n_trans, 18),
                                 squeeze=False)
        fig.suptitle(f'{region} -- S{sess} Pot-2<->Pot-4 Transition Dynamics',
                     fontsize=14, fontweight='bold')

        from matplotlib.patches import Patch
        pot_legend = [Patch(facecolor=c, alpha=0.3, label=p)
                      for p, c in pot_colors.items()]
        fig.legend(handles=pot_legend, loc='upper right', fontsize=9,
                   ncol=4, framealpha=0.8)

        row_labels = ['Pop FR (Hz)', 'PC1', 'Flow speed', 'Gate value',
                      'Divergence']

        for ci, (_, tr) in enumerate(trans_df.iterrows()):
            t_start = tr['t1_start'] - PAD_BEFORE
            t_end = tr['t2_end'] + PAD_AFTER
            phase_label = 'PRE' if tr['pre_discovery'] else 'POST'
            title = (f"Exc {tr['exc_idx']:.0f}: {tr['direction']}\n"
                     f"{tr['t1_start']:.0f}->{tr['t2_start']:.0f}s [{phase_label}]")

            n_pts = int((t_end - t_start) * RESAMPLE_HZ)
            t_axis = np.linspace(t_start, t_end, n_pts)

            fr_trace, pc1_trace, flow_trace, gate_trace = [], [], [], []
            for t in t_axis:
                f, p, fl, g = helpers['get_metrics_at_time'](region, t)
                fr_trace.append(f)
                pc1_trace.append(p)
                flow_trace.append(fl)
                gate_trace.append(g)

            # Divergence
            h_indices = [min(helpers['get_h_idx'](region, t),
                            len(rd[region]['h_states']) - 1) for t in t_axis]
            h_pts = rd[region]['h_states'][h_indices]
            div_trace = compute_local_divergence(rd[region]['ode_func'], h_pts)

            traces = [np.array(fr_trace), np.array(pc1_trace),
                      np.array(flow_trace), np.array(gate_trace),
                      np.array(div_trace)]

            sm_win = max(3, int(1.0 * RESAMPLE_HZ))
            traces_sm = [uniform_filter1d(tr_data, sm_win) for tr_data in traces]

            for ri_row, (trace, label) in enumerate(zip(traces_sm, row_labels)):
                ax = axes[ri_row, ci]
                ax.plot(t_axis, trace, color='black', linewidth=1.5)

                exc_visits = pv_df[pv_df['excursion_idx'] == tr['exc_idx']]
                for _, pv in exc_visits.iterrows():
                    if pv['end_s'] >= t_start and pv['start_s'] <= t_end:
                        ax.axvspan(pv['start_s'], pv['end_s'], alpha=0.2,
                                   color=pot_colors.get(pv['pot'], 'gray'))
                        if ri_row == 0:
                            ax.text((pv['start_s'] + pv['end_s']) / 2,
                                    ax.get_ylim()[1], pv['pot'][-1],
                                    ha='center', va='bottom', fontsize=8,
                                    fontweight='bold',
                                    color=pot_colors.get(pv['pot'], 'gray'))

                if ri_row == 0:
                    ax.set_title(title, fontsize=10, fontweight='bold')
                if ri_row == 4:
                    ax.set_xlabel('Time in session (s)')
                if ci == 0:
                    ax.set_ylabel(label)

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        fig_path = f'figures/foraging_transition_dynamics_s{sess}_{region.lower()}.png'
        plt.savefig(fig_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fig_path}")

    n_pre = len(trans_df[trans_df['pre_discovery']])
    n_post = len(trans_df[~trans_df['pre_discovery']])

    # Print arrival metrics
    for region in ['LHA', 'RSP']:
        print(f"\n  {region} transition arrival metrics:")
        for _, tr in trans_df.iterrows():
            phase = 'PRE' if tr['pre_discovery'] else 'POST'
            f1, p1, fl1, g1 = helpers['get_metrics_at_time'](region, tr['t1_start'])
            f2, p2, fl2, g2 = helpers['get_metrics_at_time'](region, tr['t2_start'])
            print(f"    Exc {tr['exc_idx']:.0f} {tr['direction']} [{phase}]: "
                  f"FR {f1:.2f}->{f2:.2f}, PC1 {p1:.2f}->{p2:.2f}, "
                  f"Flow {fl1:.2f}->{fl2:.2f}, Gate {g1:.3f}->{g2:.3f}")

    return {'n_transitions': n_trans, 'n_pre': n_pre, 'n_post': n_post}


# =============================================================================
# MAIN
# =============================================================================
def main():
    all_metrics = []

    for sess_num, config in SESSION_CONFIG.items():
        print(f"\n{'#'*80}")
        print(f"  SESSION {sess_num} ({config['state'].upper()}, {config['phase']})")
        print(f"{'#'*80}")
        sys.stdout.flush()

        sd = load_session_data(sess_num, config['state'])
        helpers = make_helpers(sd)

        metrics = {'session': sess_num, 'state': config['state']}

        # Analysis 1
        print(f"\n{'='*70}")
        print(f"  ANALYSIS 1: Behavioral Sequence")
        print(f"{'='*70}")
        sys.stdout.flush()
        m1 = analysis_1_behavioral_sequence(sd, sess_num)
        metrics.update(m1)

        # Analysis 6 (continuous -- before others since it's always useful)
        print(f"\n{'='*70}")
        print(f"  ANALYSIS 6: Continuous Neural Tracking")
        print(f"{'='*70}")
        sys.stdout.flush()
        m6 = analysis_6_continuous(sd, sess_num, helpers)
        if m6:
            metrics.update(m6)

        # Analysis 2 (within-excursion)
        print(f"\n{'='*70}")
        print(f"  ANALYSIS 2: Within-Excursion Pot-2 vs Pot-4")
        print(f"{'='*70}")
        sys.stdout.flush()
        m2 = analysis_2_within_excursion(sd, sess_num, helpers)
        if m2:
            metrics.update(m2)

        # Analyses 3-4 (learning curves)
        print(f"\n{'='*70}")
        print(f"  ANALYSES 3-4: Across-Excursion Learning Curves")
        print(f"{'='*70}")
        sys.stdout.flush()
        m34 = analysis_34_learning_curves(sd, sess_num, helpers)
        if m34:
            metrics.update(m34)

        # Analyses 5/7 (commitment)
        print(f"\n{'='*70}")
        print(f"  ANALYSES 5/7: Commitment vs Non-Commitment")
        print(f"{'='*70}")
        sys.stdout.flush()
        m57 = analysis_57_commitment(sd, sess_num, helpers)
        if m57:
            metrics.update(m57)

        # Analysis 8 (reward)
        print(f"\n{'='*70}")
        print(f"  ANALYSIS 8: Reward Onset")
        print(f"{'='*70}")
        sys.stdout.flush()
        m8 = analysis_8_reward(sd, sess_num, helpers)
        if m8:
            metrics.update(m8)

        # Analysis 9 (pre vs post)
        print(f"\n{'='*70}")
        print(f"  ANALYSIS 9: Pre vs Post-Feed")
        print(f"{'='*70}")
        sys.stdout.flush()
        m9 = analysis_9_pre_vs_post(sd, sess_num, helpers)
        if m9:
            metrics.update(m9)

        # Analysis 10 (transitions)
        print(f"\n{'='*70}")
        print(f"  ANALYSIS 10: P2<->P4 Transition Dynamics")
        print(f"{'='*70}")
        sys.stdout.flush()
        m10 = analysis_10_transitions(sd, sess_num, helpers)
        if m10:
            metrics.update(m10)

        all_metrics.append(metrics)

        print(f"\n  Session {sess_num} complete.")
        sys.stdout.flush()

    # Save combined metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv('data/foraging_neural_all_sessions_metrics.csv', index=False)
    print(f"\n{'#'*80}")
    print(f"  ALL SESSIONS COMPLETE")
    print(f"  Saved: data/foraging_neural_all_sessions_metrics.csv")
    print(f"{'#'*80}")


if __name__ == "__main__":
    main()
