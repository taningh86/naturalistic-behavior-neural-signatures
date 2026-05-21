"""
Foraging neural signatures — Session 2 (Fed, Foraging).

Full analysis plan:
1. Behavioral sequence mapping (excursion × pot visit)
2. Within-excursion Pot-2 vs Pot-4 neural comparison
3. Across-excursion Pot-4 response evolution
4. Across-excursion Pot-2 response evolution
5. Commitment excursion vs non-commitment
6. Pre-dig continuous neural tracking
7. Pre-dig neural signature (5s before dig vs 5s before earlier visits)
8. Reward onset neural response
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
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

SESS = 2
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


def load_model(region):
    model_path = Path("data") / f"gru_ode_10ms_poisson_{region}_fed_model.pt"
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model = PooledGRUODE(
        checkpoint['neuron_counts'],
        checkpoint['config']['d_shared'],
        checkpoint['config']['hidden_size'],
        checkpoint['config'].get('gate_hidden', ODE_GATE_HIDDEN),
        checkpoint['config'].get('pred_bins', PRED_BINS),
    ).to(DEVICE)
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
# BEHAVIOR DATA
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
    """Combined pot + pot zone signal."""
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
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("  FORAGING NEURAL SIGNATURES — Session 2 (Fed, Foraging)")
    print("=" * 70)
    sys.stdout.flush()

    # --- Load behavior ---
    behav, n_behav_bins = load_behavior_data(SESS)
    print(f"  Behavior: {n_behav_bins} bins ({n_behav_bins * 0.1:.1f}s)")

    # --- Load excursions ---
    exc_df = pd.read_csv("data/excursion_features_all_sessions.csv")
    s2_exc = exc_df[exc_df["session"] == SESS].copy()
    print(f"  Excursions: {len(s2_exc)}")

    # --- Build pot signals ---
    pot_signals = {}
    for pot in ['Pot-1', 'Pot-2', 'Pot-3', 'Pot-4']:
        pot_signals[pot] = get_pot_signal(behav, pot, n_behav_bins)

    # --- Get dwell events per pot ---
    pot_dwells = {}
    for pot in ['Pot-1', 'Pot-2', 'Pot-3', 'Pot-4']:
        pot_dwells[pot] = find_dwell_events(pot_signals[pot], min_bins=10)

    # --- Behavior signals ---
    feeding = behav.get('Feeding', np.zeros(n_behav_bins))
    digging = behav.get('Digging', np.zeros(n_behav_bins))
    feeding = np.where(np.isnan(feeding), 0, feeding)
    digging = np.where(np.isnan(digging), 0, digging)

    # Discovery time: first feeding onset
    feed_onset_bin = np.argmax(feeding > 0) if np.any(feeding > 0) else n_behav_bins
    discovery_time = feed_onset_bin * 0.1
    print(f"  Discovery time (first feed): {discovery_time:.1f}s")

    # First dig
    dig_onset_bin = np.argmax(digging > 0) if np.any(digging > 0) else n_behav_bins
    first_dig_time = dig_onset_bin * 0.1
    print(f"  First dig: {first_dig_time:.1f}s")

    # =====================================================================
    # ANALYSIS 1: Behavioral Sequence Mapping
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"  ANALYSIS 1: Excursion × Pot Visit Sequence")
    print(f"{'='*70}")
    sys.stdout.flush()

    # Map all pot dwells to excursions
    pot_visit_rows = []
    for pot_name, dwells in pot_dwells.items():
        for dw_start, dw_end, dw_dur in dwells:
            t_start = dw_start * 0.1
            t_end = dw_end * 0.1
            # Find parent excursion
            parent_exc = None
            for _, row in s2_exc.iterrows():
                if row['start_time'] <= t_start <= row['end_time']:
                    parent_exc = int(row['excursion_idx'])
                    break
            # Check feeding/digging during dwell
            feed_bins = np.sum(feeding[dw_start:dw_end+1] > 0)
            dig_bins = np.sum(digging[dw_start:dw_end+1] > 0)
            pot_visit_rows.append({
                'pot': pot_name, 'start_s': t_start, 'end_s': t_end,
                'dwell_s': dw_dur * 0.1, 'excursion_idx': parent_exc,
                'feed_bins': int(feed_bins), 'dig_bins': int(dig_bins),
                'pre_discovery': t_start < discovery_time,
            })

    pv_df = pd.DataFrame(pot_visit_rows).sort_values('start_s').reset_index(drop=True)
    pv_df.to_csv('data/foraging_excursion_potvisits_s2.csv', index=False)
    print(f"  Saved: data/foraging_excursion_potvisits_s2.csv ({len(pv_df)} pot visits)")

    # Print pre-discovery excursion sequences
    pre_disc = pv_df[pv_df['pre_discovery']].copy()
    print(f"\n  Pre-discovery pot visits: {len(pre_disc)}")
    exc_groups = pre_disc.groupby('excursion_idx')
    for exc_idx, grp in exc_groups:
        grp_sorted = grp.sort_values('start_s')
        seq = ' -> '.join([f"{r['pot']}({r['start_s']:.0f}s,{r['dwell_s']:.1f}s)"
                           for _, r in grp_sorted.iterrows()])
        print(f"    Exc {exc_idx}: {seq}")

    # =====================================================================
    # LOAD NEURAL DATA
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"  Loading neural data...")
    print(f"{'='*70}")
    sys.stdout.flush()

    sp = paths_config['single_probe']['coordinates_1']['mouse01']['sessions']
    sorted_path = Path(sp[f'session_{SESS}']['sorted'])
    sorting = se.read_kilosort(sorted_path)
    ci = pd.read_csv(sorted_path / "cluster_info.tsv", sep="\t")
    lc = "group" if "group" in ci.columns and ci["group"].eq("good").any() else "KSLabel"
    good = ci[ci[lc] == "good"]
    lha_ids = good[good["depth"] < LHA_DEPTH_MAX]["cluster_id"].values
    rsp_ids = good[good["depth"] >= RSP_DEPTH_MIN]["cluster_id"].values
    print(f"  Good units: {len(lha_ids)} LHA, {len(rsp_ids)} RSP")

    # Population FR at 100ms
    def get_fr(uids):
        amin = np.inf; amax = 0; sts = {}
        for u in uids:
            st = sorting.get_unit_spike_train(u); sts[u] = st
            if len(st) > 0: amin = min(amin, np.min(st)); amax = max(amax, np.max(st))
        nb = int((amax - amin) / FR_SAMP) + 1
        d = np.zeros((nb, len(uids)), dtype=np.float32)
        for i, u in enumerate(uids):
            st = sts[u]
            if len(st) > 0:
                b = ((st - amin) // FR_SAMP).astype(int)
                b = b[(b >= 0) & (b < nb)]
                np.add.at(d[:, i], b, 1)
        return d / (FR_BIN_MS / 1000.0), amin

    lha_fr, lha_fr_amin = get_fr(lha_ids)
    rsp_fr, rsp_fr_amin = get_fr(rsp_ids)
    lha_pop_fr = lha_fr.mean(axis=1)
    rsp_pop_fr = rsp_fr.mean(axis=1)
    print(f"  LHA FR: {len(lha_pop_fr)} bins, RSP FR: {len(rsp_pop_fr)} bins")

    # GRU-ODE hidden states + models
    region_data = {}
    for region, uids, fr_data, fr_amin in [
            ('LHA', lha_ids, lha_fr, lha_fr_amin),
            ('RSP', rsp_ids, rsp_fr, rsp_fr_amin)]:
        h_states = np.load(f"data/gru_ode_10ms_hidden_{region.lower()}_s{SESS}.npy")
        model = load_model(region.lower())
        ode_func = model.ode_func
        pca = PCA(n_components=3).fit(h_states)
        h_pcs = pca.transform(h_states)

        # Time offset
        allmin = np.inf
        for u in uids:
            st = sorting.get_unit_spike_train(u)
            if len(st) > 0:
                allmin = min(allmin, np.min(st))
        offset_10ms = allmin / FS
        offset_100ms = fr_amin / FS

        region_data[region] = {
            'h_states': h_states, 'h_pcs': h_pcs,
            'ode_func': ode_func, 'pca': pca,
            'pop_fr': fr_data.mean(axis=1),
            'offset_10ms': offset_10ms, 'offset_100ms': offset_100ms,
            'n_units': len(uids),
        }
    print("  Neural data loaded.")
    sys.stdout.flush()

    # Helper: get time indices
    def get_h_idx(region, t_sec):
        return max(0, int((t_sec - region_data[region]['offset_10ms']) / 0.01))

    def get_fr_idx(region, t_sec):
        return max(0, int((t_sec - region_data[region]['offset_100ms']) / 0.1))

    # Helper: extract peri-event window
    def get_peri_event_fr(region, center_sec, window_before=5.0, window_after=5.0):
        """Return (time_axis, fr_values) for a peri-event window."""
        fr = region_data[region]['pop_fr']
        offset = region_data[region]['offset_100ms']
        c_idx = int((center_sec - offset) / 0.1)
        b_idx = max(0, c_idx - int(window_before / 0.1))
        a_idx = min(len(fr), c_idx + int(window_after / 0.1))
        t = (np.arange(b_idx, a_idx) - c_idx) * 0.1
        return t, fr[b_idx:a_idx]

    def get_peri_event_latent(region, center_sec, window_before=5.0, window_after=5.0):
        """Return (time_axis, pc1, pc2, pc3) for peri-event window."""
        h_pcs = region_data[region]['h_pcs']
        offset = region_data[region]['offset_10ms']
        c_idx = int((center_sec - offset) / 0.01)
        b_idx = max(0, c_idx - int(window_before / 0.01))
        a_idx = min(len(h_pcs), c_idx + int(window_after / 0.01))
        t = (np.arange(b_idx, a_idx) - c_idx) * 0.01
        return t, h_pcs[b_idx:a_idx, 0], h_pcs[b_idx:a_idx, 1], h_pcs[b_idx:a_idx, 2]

    def get_flow_metrics(region, t_sec):
        """Compute flow speed and gate at a single time point."""
        h = region_data[region]['h_states']
        ode_func = region_data[region]['ode_func']
        idx = get_h_idx(region, t_sec)
        idx = min(idx, len(h) - 1)
        pt = h[idx:idx+1]
        dhdt = evaluate_flow(ode_func, pt)
        speed = np.linalg.norm(dhdt)
        ht = torch.tensor(pt, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            gate = ode_func.update_gate(ht).cpu().numpy().mean()
        return speed, gate

    # =====================================================================
    # ANALYSIS 6: Continuous Neural Tracking (0 to 900s)
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"  ANALYSIS 6: Continuous Pre-Discovery Neural Tracking")
    print(f"{'='*70}")
    sys.stdout.flush()

    MAX_TIME = 900  # show a bit beyond discovery

    for region in ['LHA', 'RSP']:
        rd = region_data[region]
        h = rd['h_states']
        pcs = rd['h_pcs']
        ode_func = rd['ode_func']
        offset_h = rd['offset_10ms']
        offset_fr = rd['offset_100ms']
        pop_fr = rd['pop_fr']

        # Subsample hidden states (every 1s = 100 bins)
        h_times = offset_h + np.arange(len(h)) * 0.01
        mask = h_times <= MAX_TIME
        sub_every = 100
        sub_idx = np.arange(0, mask.sum(), sub_every)
        sub_h = h[mask][sub_idx]
        sub_t = h_times[mask][sub_idx]

        # Flow speed + gate
        dhdt = evaluate_flow(ode_func, sub_h)
        flow_speed = np.linalg.norm(dhdt, axis=1)
        ht = torch.tensor(sub_h, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            gate = ode_func.update_gate(ht).cpu().numpy().mean(axis=1)

        # PC1
        pc1_sub = pcs[mask][sub_idx, 0]

        # Smooth (30s window)
        sm = 30
        flow_sm = uniform_filter1d(flow_speed, sm)
        gate_sm = uniform_filter1d(gate, sm)
        pc1_sm = uniform_filter1d(pc1_sub, sm)

        # FR
        fr_times = offset_fr + np.arange(len(pop_fr)) * 0.1
        fr_mask = fr_times <= MAX_TIME
        fr_sm = uniform_filter1d(pop_fr[fr_mask], 300)
        fr_t = fr_times[fr_mask]

        # Figure
        fig, axes = plt.subplots(5, 1, figsize=(20, 16), sharex=True)
        fig.suptitle(f'{region} — Continuous Neural State (Session 2, Foraging)\n'
                     f'Discovery at {discovery_time:.0f}s | Pot visits marked',
                     fontsize=14, fontweight='bold')

        pot_colors = {'Pot-1': '#FF9800', 'Pot-2': '#E53935',
                      'Pot-3': '#1E88E5', 'Pot-4': '#43A047'}

        def shade_pot_visits(ax):
            for pot, dwells in pot_dwells.items():
                for s, e, d in dwells:
                    ts = s * 0.1
                    te = e * 0.1
                    if ts <= MAX_TIME:
                        ax.axvspan(ts, min(te, MAX_TIME), color=pot_colors[pot],
                                   alpha=0.2)

        plot_items = [
            (sub_t, flow_sm, 'Flow speed', 'purple'),
            (sub_t, gate_sm, 'Gate value (z)', 'brown'),
            (sub_t, pc1_sm, 'PC1 position', 'navy'),
            (fr_t, fr_sm, f'Pop FR ({rd["n_units"]} units, Hz)', 'black'),
        ]

        for i, (t, vals, ylabel, color) in enumerate(plot_items):
            ax = axes[i]
            shade_pot_visits(ax)
            ax.plot(t, vals, color=color, linewidth=1)
            ax.axvline(discovery_time, color='red', linewidth=2, linestyle='--',
                       alpha=0.7, label=f'Discovery ({discovery_time:.0f}s)')
            ax.axvline(first_dig_time, color='darkred', linewidth=1.5, linestyle=':',
                       alpha=0.7, label=f'First dig ({first_dig_time:.0f}s)')
            ax.set_ylabel(ylabel)
            if i == 0:
                ax.legend(fontsize=8)

        # Panel 5: Pot occupancy ethogram
        ax = axes[4]
        for pot, color in pot_colors.items():
            sig = pot_signals[pot]
            t_behav = np.arange(len(sig)) * 0.1
            mask_b = t_behav <= MAX_TIME
            pot_idx = int(pot[-1]) - 1
            ax.fill_between(t_behav[mask_b], pot_idx, pot_idx + sig[mask_b] * 0.8,
                             color=color, alpha=0.6, step='mid')
        # Overlay dig/feed
        dig_t = np.arange(len(digging)) * 0.1
        feed_t = np.arange(len(feeding)) * 0.1
        ax.fill_between(dig_t[dig_t <= MAX_TIME], 4.2,
                         4.2 + digging[dig_t <= MAX_TIME] * 0.8,
                         color='brown', alpha=0.7, step='mid', label='Digging')
        ax.fill_between(feed_t[feed_t <= MAX_TIME], 5.2,
                         5.2 + feeding[feed_t <= MAX_TIME] * 0.8,
                         color='green', alpha=0.7, step='mid', label='Feeding')
        ax.set_yticks([0.4, 1.4, 2.4, 3.4, 4.6, 5.6])
        ax.set_yticklabels(['P1', 'P2', 'P3', 'P4', 'Dig', 'Feed'])
        ax.set_ylabel('Behavior')
        ax.legend(fontsize=8, loc='upper right')

        axes[-1].set_xlabel('Time in session (s)')
        axes[-1].set_xlim(0, MAX_TIME)

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.savefig(f'figures/foraging_continuous_s2_{region.lower()}.png',
                    dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  Saved: figures/foraging_continuous_s2_{region.lower()}.png")

    # =====================================================================
    # ANALYSIS 2: Within-Excursion Pot-2 vs Pot-4
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"  ANALYSIS 2: Within-Excursion Pot-2 vs Pot-4")
    print(f"{'='*70}")
    sys.stdout.flush()

    # Excursions with both Pot-2 and Pot-4 visits (pre-discovery)
    both_pot_excs = []
    for exc_idx, grp in pre_disc.groupby('excursion_idx'):
        pots_visited = grp['pot'].unique()
        if 'Pot-2' in pots_visited and 'Pot-4' in pots_visited:
            p2 = grp[grp['pot'] == 'Pot-2'].iloc[0]
            p4 = grp[grp['pot'] == 'Pot-4'].iloc[0]
            both_pot_excs.append({'exc_idx': exc_idx,
                                  'p2_start': p2['start_s'], 'p2_end': p2['end_s'],
                                  'p2_dwell': p2['dwell_s'],
                                  'p4_start': p4['start_s'], 'p4_end': p4['end_s'],
                                  'p4_dwell': p4['dwell_s']})

    print(f"  Excursions with both Pot-2 and Pot-4: {len(both_pot_excs)}")

    if len(both_pot_excs) >= 2:
        fig, axes = plt.subplots(len(both_pot_excs), 4, figsize=(20, 4 * len(both_pot_excs)))
        if len(both_pot_excs) == 1:
            axes = axes.reshape(1, -1)
        fig.suptitle('Within-Excursion: Pot-2 vs Pot-4 Neural State\n'
                     'Same excursion controls for time/arousal',
                     fontsize=14, fontweight='bold')

        for ei, exc_info in enumerate(both_pot_excs):
            for ri, region in enumerate(['LHA', 'RSP']):
                # FR around Pot-2 arrival
                t_p2, fr_p2 = get_peri_event_fr(region, exc_info['p2_start'],
                                                  window_before=3, window_after=5)
                t_p4, fr_p4 = get_peri_event_fr(region, exc_info['p4_start'],
                                                  window_before=3, window_after=5)

                ax = axes[ei, ri * 2]
                ax.plot(t_p2, fr_p2, color='#E53935', linewidth=1.5, label='Pot-2')
                ax.plot(t_p4, fr_p4, color='#43A047', linewidth=1.5, label='Pot-4')
                ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
                ax.set_title(f'Exc {exc_info["exc_idx"]} — {region} FR')
                ax.set_xlabel('Time from pot arrival (s)')
                ax.set_ylabel('Pop FR (Hz)')
                if ei == 0:
                    ax.legend(fontsize=8)

                # PC1 around pot arrivals
                t_p2_l, pc1_p2, _, _ = get_peri_event_latent(region, exc_info['p2_start'],
                                                               window_before=3, window_after=5)
                t_p4_l, pc1_p4, _, _ = get_peri_event_latent(region, exc_info['p4_start'],
                                                               window_before=3, window_after=5)
                # Subsample for plotting
                ss = 10
                ax = axes[ei, ri * 2 + 1]
                ax.plot(t_p2_l[::ss], pc1_p2[::ss], color='#E53935', linewidth=1.5, label='Pot-2')
                ax.plot(t_p4_l[::ss], pc1_p4[::ss], color='#43A047', linewidth=1.5, label='Pot-4')
                ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
                ax.set_title(f'Exc {exc_info["exc_idx"]} — {region} PC1')
                ax.set_xlabel('Time from pot arrival (s)')
                ax.set_ylabel('PC1')

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.savefig('figures/foraging_neural_within_excursion_s2.png',
                    dpi=100, bbox_inches='tight')
        plt.close()
        print("  Saved: figures/foraging_neural_within_excursion_s2.png")
    else:
        print("  Not enough excursions with both pots for within-excursion comparison")

    # =====================================================================
    # ANALYSES 3 & 4: Across-Excursion Learning
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"  ANALYSES 3 & 4: Across-Excursion Pot Visit Evolution")
    print(f"{'='*70}")
    sys.stdout.flush()

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle('Across-Excursion Evolution: Neural State at Pot Arrival\n'
                 'Each dot = one pot visit | Pre-discovery visits only',
                 fontsize=14, fontweight='bold')

    for ri, region in enumerate(['LHA', 'RSP']):
        for pi, pot in enumerate(['Pot-2', 'Pot-4']):
            pot_visits = pre_disc[pre_disc['pot'] == pot].sort_values('start_s')
            if len(pot_visits) == 0:
                continue

            visit_times = pot_visits['start_s'].values
            visit_nums = np.arange(1, len(visit_times) + 1)

            # FR at arrival (mean over first 1s = 10 bins of 100ms)
            fr_at_arrival = []
            pc1_at_arrival = []
            flow_at_arrival = []
            gate_at_arrival = []

            for t in visit_times:
                fr_idx = get_fr_idx(region, t)
                fr = region_data[region]['pop_fr']
                fr_window = fr[fr_idx:min(fr_idx+10, len(fr))]
                fr_at_arrival.append(np.mean(fr_window) if len(fr_window) > 0 else np.nan)

                h_idx = get_h_idx(region, t)
                pcs = region_data[region]['h_pcs']
                pc1_window = pcs[h_idx:min(h_idx+100, len(pcs)), 0]
                pc1_at_arrival.append(np.mean(pc1_window) if len(pc1_window) > 0 else np.nan)

                speed, gate_val = get_flow_metrics(region, t)
                flow_at_arrival.append(speed)
                gate_at_arrival.append(gate_val)

            fr_arr = np.array(fr_at_arrival)
            pc1_arr = np.array(pc1_at_arrival)
            flow_arr = np.array(flow_at_arrival)
            gate_arr = np.array(gate_at_arrival)

            color = '#E53935' if pot == 'Pot-2' else '#43A047'
            col_offset = 0 if pot == 'Pot-2' else 2

            # FR evolution
            ax = axes[ri, col_offset]
            ax.scatter(visit_nums, fr_arr, color=color, s=60, zorder=5,
                       edgecolors='black', linewidths=0.5)
            if len(visit_nums) >= 3:
                z = np.polyfit(visit_nums, fr_arr, 1)
                ax.plot(visit_nums, np.polyval(z, visit_nums), color=color,
                        linewidth=1, linestyle='--', alpha=0.5)
                r, p = sp_stats.pearsonr(visit_nums, fr_arr)
                ax.text(0.05, 0.95, f'r={r:.2f}, p={p:.3f}',
                        transform=ax.transAxes, fontsize=9, va='top')
            ax.set_xlabel('Visit number')
            ax.set_ylabel(f'{region} Pop FR (Hz)')
            ax.set_title(f'{pot} — {region} FR at arrival')

            # PC1 evolution
            ax = axes[ri, col_offset + 1]
            ax.scatter(visit_nums, pc1_arr, color=color, s=60, zorder=5,
                       edgecolors='black', linewidths=0.5)
            if len(visit_nums) >= 3:
                z = np.polyfit(visit_nums, pc1_arr, 1)
                ax.plot(visit_nums, np.polyval(z, visit_nums), color=color,
                        linewidth=1, linestyle='--', alpha=0.5)
                r, p = sp_stats.pearsonr(visit_nums, pc1_arr)
                ax.text(0.05, 0.95, f'r={r:.2f}, p={p:.3f}',
                        transform=ax.transAxes, fontsize=9, va='top')
            ax.set_xlabel('Visit number')
            ax.set_ylabel(f'{region} PC1')
            ax.set_title(f'{pot} — {region} PC1 at arrival')

            # Print stats
            print(f"  {region} {pot}: {len(visit_nums)} visits")
            if len(visit_nums) >= 3:
                r_fr, p_fr = sp_stats.pearsonr(visit_nums, fr_arr)
                r_pc, p_pc = sp_stats.pearsonr(visit_nums, pc1_arr)
                print(f"    FR trend: r={r_fr:.3f}, p={p_fr:.3f}")
                print(f"    PC1 trend: r={r_pc:.3f}, p={p_pc:.3f}")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig('figures/foraging_neural_across_visits_s2.png',
                dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/foraging_neural_across_visits_s2.png")

    # =====================================================================
    # ANALYSIS 5 & 7: Commitment Excursion vs Non-Commitment
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"  ANALYSES 5 & 7: Commitment vs Non-Commitment Excursions")
    print(f"{'='*70}")
    sys.stdout.flush()

    # Find the commitment excursion (contains first dig)
    commit_exc = None
    for _, row in s2_exc.iterrows():
        if row['start_time'] <= first_dig_time <= row['end_time']:
            commit_exc = row
            break

    if commit_exc is not None:
        print(f"  Commitment excursion: idx={int(commit_exc.excursion_idx)}, "
              f"{commit_exc.start_time:.1f}-{commit_exc.end_time:.1f}s, "
              f"dur={commit_exc.duration:.1f}s")

        # Prior Pot-4 excursions (non-commitment)
        p4_pre_disc = pre_disc[pre_disc['pot'] == 'Pot-4'].sort_values('start_s')
        prior_p4_exc_idxs = p4_pre_disc['excursion_idx'].unique()
        prior_p4_exc_idxs = [idx for idx in prior_p4_exc_idxs
                              if idx != commit_exc.excursion_idx]

        print(f"  Prior Pot-4 excursions (non-commitment): {len(prior_p4_exc_idxs)}")

        fig, axes = plt.subplots(2, 3, figsize=(21, 12))
        fig.suptitle('Commitment Excursion vs Prior Pot-4 Excursions\n'
                     f'Commit: Exc {int(commit_exc.excursion_idx)} '
                     f'({commit_exc.start_time:.0f}-{commit_exc.end_time:.0f}s) | '
                     f'Prior: {len(prior_p4_exc_idxs)} excursions',
                     fontsize=14, fontweight='bold')

        for ri, region in enumerate(['LHA', 'RSP']):
            # Analysis 7: 5s before Pot-4 contact
            # Commitment: 5s before first dig
            pre_commit_t = first_dig_time
            t_c, fr_c = get_peri_event_fr(region, pre_commit_t,
                                           window_before=5, window_after=3)

            # Prior: 5s before each Pot-4 arrival
            prior_frs = []
            for exc_idx in prior_p4_exc_idxs:
                exc_p4 = p4_pre_disc[p4_pre_disc['excursion_idx'] == exc_idx]
                if len(exc_p4) > 0:
                    p4_time = exc_p4.iloc[0]['start_s']
                    t_p, fr_p = get_peri_event_fr(region, p4_time,
                                                   window_before=5, window_after=3)
                    if len(fr_p) >= 50:
                        prior_frs.append((t_p, fr_p))

            # Plot FR
            ax = axes[ri, 0]
            for t_p, fr_p in prior_frs:
                ax.plot(t_p, fr_p, color='gray', alpha=0.3, linewidth=0.8)
            if prior_frs:
                # Mean of priors (resample to common length)
                min_len = min(len(f) for _, f in prior_frs)
                mean_prior = np.mean([f[:min_len] for _, f in prior_frs], axis=0)
                t_common = prior_frs[0][0][:min_len]
                ax.plot(t_common, mean_prior, color='steelblue', linewidth=2,
                        label=f'Prior mean (n={len(prior_frs)})')
            ax.plot(t_c, fr_c, color='red', linewidth=2, label='Commitment')
            ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
            ax.set_xlabel('Time from Pot-4 arrival (s)')
            ax.set_ylabel(f'{region} Pop FR (Hz)')
            ax.set_title(f'{region} — FR around Pot-4 arrival')
            ax.legend(fontsize=8)

            # Analysis 7: 5s before - compare distributions
            # FR in [-5, -1] window
            pre_commit_fr = []
            t_c5, fr_c5 = get_peri_event_fr(region, pre_commit_t,
                                              window_before=5, window_after=0)
            pre_commit_fr = fr_c5[fr_c5.shape[0]//5:]  # last 4s

            prior_pre_frs = []
            for exc_idx in prior_p4_exc_idxs:
                exc_p4 = p4_pre_disc[p4_pre_disc['excursion_idx'] == exc_idx]
                if len(exc_p4) > 0:
                    p4_time = exc_p4.iloc[0]['start_s']
                    _, fr_p5 = get_peri_event_fr(region, p4_time,
                                                  window_before=5, window_after=0)
                    if len(fr_p5) >= 10:
                        prior_pre_frs.append(np.mean(fr_p5[len(fr_p5)//5:]))

            ax = axes[ri, 1]
            if len(prior_pre_frs) >= 2:
                commit_val = np.mean(pre_commit_fr)
                ax.bar(['Prior\n(non-commit)'], [np.mean(prior_pre_frs)],
                       yerr=[np.std(prior_pre_frs)/np.sqrt(len(prior_pre_frs))],
                       color='steelblue', alpha=0.7, capsize=5)
                ax.bar(['Commitment'], [commit_val], color='red', alpha=0.7)
                # Individual prior values
                ax.scatter(np.zeros(len(prior_pre_frs)), prior_pre_frs,
                           color='steelblue', s=30, alpha=0.5, zorder=5)
                ax.set_ylabel(f'{region} mean FR [-5, -1]s before Pot-4')
                ax.set_title(f'{region} — Pre-arrival FR comparison')

            # PC1 trajectory: commitment vs prior
            ax = axes[ri, 2]
            for exc_idx in prior_p4_exc_idxs:
                exc_p4 = p4_pre_disc[p4_pre_disc['excursion_idx'] == exc_idx]
                if len(exc_p4) > 0:
                    p4_time = exc_p4.iloc[0]['start_s']
                    t_l, pc1_l, _, _ = get_peri_event_latent(region, p4_time,
                                                               window_before=5, window_after=3)
                    ax.plot(t_l[::10], pc1_l[::10], color='gray', alpha=0.3, linewidth=0.8)

            t_lc, pc1_c, _, _ = get_peri_event_latent(region, pre_commit_t,
                                                        window_before=5, window_after=3)
            ax.plot(t_lc[::10], pc1_c[::10], color='red', linewidth=2, label='Commitment')
            ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
            ax.set_xlabel('Time from Pot-4 arrival (s)')
            ax.set_ylabel(f'{region} PC1')
            ax.set_title(f'{region} — PC1 around Pot-4 arrival')
            ax.legend(fontsize=8)

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.savefig('figures/foraging_commitment_s2.png',
                    dpi=100, bbox_inches='tight')
        plt.close()
        print("  Saved: figures/foraging_commitment_s2.png")
    else:
        print("  Could not identify commitment excursion")

    # =====================================================================
    # ANALYSIS 8: Reward Onset
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"  ANALYSIS 8: Reward Onset Neural Response")
    print(f"{'='*70}")
    sys.stdout.flush()

    fig, axes = plt.subplots(2, 3, figsize=(21, 10))
    fig.suptitle(f'Reward Onset: Neural Response at First Feeding ({discovery_time:.0f}s)\n'
                 f'Window: -10s to +15s around first feed onset',
                 fontsize=14, fontweight='bold')

    for ri, region in enumerate(['LHA', 'RSP']):
        # FR
        t_fr, fr_vals = get_peri_event_fr(region, discovery_time,
                                           window_before=10, window_after=15)
        ax = axes[ri, 0]
        ax.plot(t_fr, fr_vals, color='black', linewidth=1)
        # Smooth version
        if len(fr_vals) >= 20:
            fr_sm = uniform_filter1d(fr_vals, 20)
            ax.plot(t_fr, fr_sm, color='red', linewidth=2, label='2s smoothed')
        ax.axvline(0, color='green', linewidth=2, linestyle='--', label='Feed onset')
        ax.axvline(first_dig_time - discovery_time, color='brown', linewidth=1.5,
                   linestyle=':', label='Dig onset')
        ax.set_xlabel('Time from first feed (s)')
        ax.set_ylabel(f'{region} Pop FR (Hz)')
        ax.set_title(f'{region} — Firing Rate')
        ax.legend(fontsize=8)

        # PC1-3
        t_l, pc1, pc2, pc3 = get_peri_event_latent(region, discovery_time,
                                                      window_before=10, window_after=15)
        ss = 10
        ax = axes[ri, 1]
        ax.plot(t_l[::ss], pc1[::ss], color='navy', linewidth=1.5, label='PC1')
        ax.plot(t_l[::ss], pc2[::ss], color='teal', linewidth=1.5, label='PC2')
        ax.plot(t_l[::ss], pc3[::ss], color='orange', linewidth=1.5, label='PC3')
        ax.axvline(0, color='green', linewidth=2, linestyle='--')
        ax.set_xlabel('Time from first feed (s)')
        ax.set_ylabel(f'{region} PC score')
        ax.set_title(f'{region} — Latent Trajectory')
        ax.legend(fontsize=8)

        # Flow speed + gate around reward
        n_pts = 250  # 25s at 0.1s resolution
        reward_times = np.linspace(discovery_time - 10, discovery_time + 15, n_pts)
        speeds = []
        gates = []
        for t in reward_times:
            s, g = get_flow_metrics(region, t)
            speeds.append(s)
            gates.append(g)

        ax = axes[ri, 2]
        t_rel = reward_times - discovery_time
        ax.plot(t_rel, speeds, color='purple', linewidth=1.5, label='Flow speed')
        ax2 = ax.twinx()
        ax2.plot(t_rel, gates, color='brown', linewidth=1.5, label='Gate', alpha=0.7)
        ax.axvline(0, color='green', linewidth=2, linestyle='--')
        ax.set_xlabel('Time from first feed (s)')
        ax.set_ylabel('Flow speed', color='purple')
        ax2.set_ylabel('Gate value', color='brown')
        ax.set_title(f'{region} — Flow & Gate')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    plt.savefig('figures/foraging_reward_s2.png',
                dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/foraging_reward_s2.png")

    # =====================================================================
    # ANALYSIS 9: Pre vs Post-Feed Pot Visit Comparison
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"  ANALYSIS 9: Pre vs Post-Feed Pot Visit Comparison")
    print(f"{'='*70}")
    sys.stdout.flush()

    # Split pot visits into pre and post discovery
    all_visits = pv_df.copy()
    pre_visits = all_visits[all_visits['pre_discovery'] == True]
    post_visits = all_visits[all_visits['pre_discovery'] == False]

    # Exclude the commitment excursion itself (Exc 32 = dig/feed)
    # and any visits with feeding (these are reward, not "visit")
    post_visits = post_visits[(post_visits['feed_bins'] == 0)]

    print(f"  Pre-discovery visits: {len(pre_visits)}")
    print(f"  Post-discovery visits (no feed): {len(post_visits)}")
    for pot in ['Pot-2', 'Pot-4']:
        n_pre = len(pre_visits[pre_visits['pot'] == pot])
        n_post = len(post_visits[post_visits['pot'] == pot])
        print(f"    {pot}: {n_pre} pre, {n_post} post")

    # --- Figure A: Extended learning curves (pre + post, all 4 metrics) ---
    fig, axes = plt.subplots(4, 4, figsize=(22, 16))
    fig.suptitle('Pre vs Post-Feed: Neural State at Pot Arrival\n'
                 'Vertical line = discovery (767s) | Red=Pot-2, Green=Pot-4',
                 fontsize=14, fontweight='bold')

    metric_labels = ['Pop FR (Hz)', 'PC1', 'Flow speed', 'Gate value']

    for pi, pot in enumerate(['Pot-2', 'Pot-4']):
        color = '#E53935' if pot == 'Pot-2' else '#43A047'
        col_offset = 0 if pot == 'Pot-2' else 2

        for ri, region in enumerate(['LHA', 'RSP']):
            # Gather all visits for this pot (pre + post)
            pot_pre = pre_visits[pre_visits['pot'] == pot].sort_values('start_s')
            pot_post = post_visits[post_visits['pot'] == pot].sort_values('start_s')
            all_pot = pd.concat([pot_pre, pot_post]).sort_values('start_s')

            if len(all_pot) == 0:
                continue

            # Compute metrics at each arrival
            visit_times = all_pot['start_s'].values
            is_pre = all_pot['pre_discovery'].values
            visit_nums = np.arange(1, len(visit_times) + 1)
            n_pre_pot = int(is_pre.sum())

            fr_vals, pc1_vals, flow_vals, gate_vals = [], [], [], []
            for t in visit_times:
                # FR
                fr_idx = get_fr_idx(region, t)
                fr = region_data[region]['pop_fr']
                fr_window = fr[fr_idx:min(fr_idx+10, len(fr))]
                fr_vals.append(np.mean(fr_window) if len(fr_window) > 0 else np.nan)
                # PC1
                h_idx = get_h_idx(region, t)
                pcs = region_data[region]['h_pcs']
                pc1_window = pcs[h_idx:min(h_idx+100, len(pcs)), 0]
                pc1_vals.append(np.mean(pc1_window) if len(pc1_window) > 0 else np.nan)
                # Flow + Gate
                speed, gate_val = get_flow_metrics(region, t)
                flow_vals.append(speed)
                gate_vals.append(gate_val)

            all_metrics = [np.array(fr_vals), np.array(pc1_vals),
                           np.array(flow_vals), np.array(gate_vals)]

            for mi, (metric, label) in enumerate(zip(all_metrics, metric_labels)):
                row_idx = mi
                col_idx = pi * 2 + ri
                ax = axes[row_idx, col_idx]

                # Plot pre and post as different markers
                pre_mask = is_pre.astype(bool)
                post_mask = ~pre_mask

                ax.scatter(visit_nums[pre_mask], metric[pre_mask],
                           color=color, s=60, zorder=5, edgecolors='black',
                           linewidths=0.5, marker='o', label='Pre-feed')
                ax.scatter(visit_nums[post_mask], metric[post_mask],
                           color=color, s=60, zorder=5, edgecolors='black',
                           linewidths=0.5, marker='s', alpha=0.5, label='Post-feed')

                # Vertical line at boundary
                if n_pre_pot > 0 and n_pre_pot < len(visit_nums):
                    ax.axvline(n_pre_pot + 0.5, color='red', linewidth=2,
                               linestyle='--', alpha=0.7, label='Discovery')

                # Trend lines: separate for pre and post
                if pre_mask.sum() >= 3:
                    x_pre = visit_nums[pre_mask]
                    y_pre = metric[pre_mask]
                    z = np.polyfit(x_pre, y_pre, 1)
                    ax.plot(x_pre, np.polyval(z, x_pre), color=color,
                            linewidth=1.5, linestyle='--', alpha=0.6)
                if post_mask.sum() >= 3:
                    x_post = visit_nums[post_mask]
                    y_post = metric[post_mask]
                    z = np.polyfit(x_post, y_post, 1)
                    ax.plot(x_post, np.polyval(z, x_post), color=color,
                            linewidth=1.5, linestyle=':', alpha=0.6)

                if mi == 0:
                    ax.set_title(f'{pot} — {region}', fontsize=11, fontweight='bold')
                if mi == 3:
                    ax.set_xlabel('Visit number')
                ax.set_ylabel(label)
                if mi == 0 and pi == 0 and ri == 0:
                    ax.legend(fontsize=7, loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig('figures/foraging_pre_vs_post_visits_s2.png',
                dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/foraging_pre_vs_post_visits_s2.png")

    # --- Figure B: Pre vs Post summary statistics (bar plots) ---
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    fig.suptitle('Pre vs Post-Feed: Mean Neural State at Pot Arrival\n'
                 'Mann-Whitney U test | Session 2',
                 fontsize=14, fontweight='bold')

    for pi, pot in enumerate(['Pot-2', 'Pot-4']):
        pot_pre = pre_visits[pre_visits['pot'] == pot].sort_values('start_s')
        pot_post = post_visits[post_visits['pot'] == pot].sort_values('start_s')

        if len(pot_pre) < 2 or len(pot_post) < 2:
            print(f"  {pot}: not enough visits for stats (pre={len(pot_pre)}, post={len(pot_post)})")
            continue

        for ri, region in enumerate(['LHA', 'RSP']):
            col = pi * 2 + ri

            # Compute metrics
            metrics_pre = {'FR': [], 'PC1': [], 'Flow': [], 'Gate': []}
            metrics_post = {'FR': [], 'PC1': [], 'Flow': [], 'Gate': []}

            for visits, store in [(pot_pre, metrics_pre), (pot_post, metrics_post)]:
                for t in visits['start_s'].values:
                    fr_idx = get_fr_idx(region, t)
                    fr = region_data[region]['pop_fr']
                    fw = fr[fr_idx:min(fr_idx+10, len(fr))]
                    store['FR'].append(np.mean(fw) if len(fw) > 0 else np.nan)

                    h_idx = get_h_idx(region, t)
                    pcs = region_data[region]['h_pcs']
                    pw = pcs[h_idx:min(h_idx+100, len(pcs)), 0]
                    store['PC1'].append(np.mean(pw) if len(pw) > 0 else np.nan)

                    speed, gate_val = get_flow_metrics(region, t)
                    store['Flow'].append(speed)
                    store['Gate'].append(gate_val)

            # Bar plot for each metric
            metric_names = ['FR', 'PC1', 'Flow', 'Gate']
            pre_means = [np.nanmean(metrics_pre[m]) for m in metric_names]
            post_means = [np.nanmean(metrics_post[m]) for m in metric_names]
            pre_sems = [sp_stats.sem([v for v in metrics_pre[m] if not np.isnan(v)])
                        for m in metric_names]
            post_sems = [sp_stats.sem([v for v in metrics_post[m] if not np.isnan(v)])
                         for m in metric_names]

            ax = axes[ri, pi * 2]
            x = np.arange(len(metric_names))
            w = 0.35
            bars1 = ax.bar(x - w/2, pre_means, w, yerr=pre_sems, capsize=4,
                           color='cornflowerblue', edgecolor='black', label='Pre-feed')
            bars2 = ax.bar(x + w/2, post_means, w, yerr=post_sems, capsize=4,
                           color='salmon', edgecolor='black', label='Post-feed')
            ax.set_xticks(x)
            ax.set_xticklabels(metric_names)
            ax.set_title(f'{pot} — {region}', fontweight='bold')
            if pi == 0 and ri == 0:
                ax.legend(fontsize=8)

            # Stats
            ax2 = axes[ri, pi * 2 + 1]
            stats_text = f'{pot} — {region}\n'
            stats_text += f'Pre: n={len(pot_pre)}, Post: n={len(pot_post)}\n\n'
            for m in metric_names:
                pre_arr = np.array(metrics_pre[m])
                post_arr = np.array(metrics_post[m])
                pre_arr = pre_arr[~np.isnan(pre_arr)]
                post_arr = post_arr[~np.isnan(post_arr)]
                if len(pre_arr) >= 2 and len(post_arr) >= 2:
                    u, p = sp_stats.mannwhitneyu(pre_arr, post_arr, alternative='two-sided')
                    sig = '*' if p < 0.05 else ''
                    delta = np.median(post_arr) - np.median(pre_arr)
                    stats_text += (f'{m}: pre={np.median(pre_arr):.3f}, '
                                   f'post={np.median(post_arr):.3f}\n'
                                   f'   d={delta:+.3f}, U={u:.0f}, p={p:.4f} {sig}\n')
                else:
                    stats_text += f'{m}: insufficient data\n'

            ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes,
                     fontsize=9, verticalalignment='top', fontfamily='monospace')
            ax2.axis('off')

            print(f"\n  {pot} — {region}:")
            print(f"    Pre: n={len(pot_pre)}, Post: n={len(pot_post)}")
            for m in metric_names:
                pre_arr = np.array(metrics_pre[m])
                post_arr = np.array(metrics_post[m])
                pre_arr = pre_arr[~np.isnan(pre_arr)]
                post_arr = post_arr[~np.isnan(post_arr)]
                if len(pre_arr) >= 2 and len(post_arr) >= 2:
                    u, p = sp_stats.mannwhitneyu(pre_arr, post_arr, alternative='two-sided')
                    print(f"    {m}: pre_med={np.median(pre_arr):.3f}, "
                          f"post_med={np.median(post_arr):.3f}, "
                          f"U={u:.0f}, p={p:.4f}")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig('figures/foraging_pre_vs_post_stats_s2.png',
                dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/foraging_pre_vs_post_stats_s2.png")

    # =====================================================================
    # ANALYSIS 10: Pot-2 <-> Pot-4 Transition Dynamics
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"  ANALYSIS 10: Pot-2 <-> Pot-4 Transition Dynamics")
    print(f"{'='*70}")
    sys.stdout.flush()

    # Find excursions containing BOTH Pot-2 and Pot-4 visits
    transition_excursions = []
    for exc_idx, grp in pv_df.groupby('excursion_idx'):
        pots_in_exc = set(grp['pot'].values)
        if 'Pot-2' in pots_in_exc and 'Pot-4' in pots_in_exc:
            grp_sorted = grp.sort_values('start_s')
            # Find sequential P2<->P4 pairs (direct or with intervening pots)
            visits = list(grp_sorted.itertuples())
            for i in range(len(visits) - 1):
                for j in range(i + 1, len(visits)):
                    v1, v2 = visits[i], visits[j]
                    if (v1.pot == 'Pot-2' and v2.pot == 'Pot-4'):
                        transition_excursions.append({
                            'exc_idx': exc_idx,
                            'direction': 'P2->P4',
                            'pot1': 'Pot-2', 'pot2': 'Pot-4',
                            't1_start': v1.start_s, 't1_end': v1.end_s,
                            't2_start': v2.start_s, 't2_end': v2.end_s,
                            'pre_discovery': v1.start_s < discovery_time,
                            'gap_s': v2.start_s - v1.end_s,
                        })
                        break  # take first P4 after this P2
                    elif (v1.pot == 'Pot-4' and v2.pot == 'Pot-2'):
                        transition_excursions.append({
                            'exc_idx': exc_idx,
                            'direction': 'P4->P2',
                            'pot1': 'Pot-4', 'pot2': 'Pot-2',
                            't1_start': v1.start_s, 't1_end': v1.end_s,
                            't2_start': v2.start_s, 't2_end': v2.end_s,
                            'pre_discovery': v1.start_s < discovery_time,
                            'gap_s': v2.start_s - v1.end_s,
                        })
                        break

    trans_df = pd.DataFrame(transition_excursions)
    # Deduplicate: keep first transition per excursion per direction
    trans_df = trans_df.drop_duplicates(subset=['exc_idx', 'direction']).reset_index(drop=True)

    print(f"  Found {len(trans_df)} Pot-2<->Pot-4 transitions:")
    for _, tr in trans_df.iterrows():
        phase = 'PRE' if tr['pre_discovery'] else 'POST'
        print(f"    Exc {tr['exc_idx']:.0f}: {tr['direction']} "
              f"({tr['t1_start']:.0f}s->{tr['t2_start']:.0f}s, "
              f"gap={tr['gap_s']:.1f}s) [{phase}]")

    # --- Figure: Time-resolved dynamics during each transition ---
    # For each transition: plot continuous neural state from 3s before pot1 arrival
    # to 3s after pot2 arrival
    PAD_BEFORE = 3.0   # seconds before first pot arrival
    PAD_AFTER = 3.0    # seconds after second pot arrival
    RESAMPLE_HZ = 10   # 100ms resolution for display

    n_trans = len(trans_df)
    if n_trans == 0:
        print("  No transitions found, skipping.")
    else:
        for region in ['LHA', 'RSP']:
            fig, axes = plt.subplots(5, n_trans, figsize=(5 * n_trans, 18),
                                     squeeze=False)
            fig.suptitle(f'{region} — Pot-2<->Pot-4 Transition Dynamics\n'
                         f'Shaded: pot occupancy | Dashed red=discovery (767s)',
                         fontsize=14, fontweight='bold')

            row_labels = ['Pop FR (Hz)', 'PC1', 'Flow speed', 'Gate value',
                          'Divergence']

            for ci, (_, tr) in enumerate(trans_df.iterrows()):
                t_start = tr['t1_start'] - PAD_BEFORE
                t_end = tr['t2_end'] + PAD_AFTER
                phase_label = 'PRE' if tr['pre_discovery'] else 'POST'
                title = (f"Exc {tr['exc_idx']:.0f}: {tr['direction']}\n"
                         f"{tr['t1_start']:.0f}->{tr['t2_start']:.0f}s [{phase_label}]")

                # Time axis at RESAMPLE_HZ
                n_pts = int((t_end - t_start) * RESAMPLE_HZ)
                t_axis = np.linspace(t_start, t_end, n_pts)

                # Compute all metrics along this trajectory
                fr_trace = []
                pc1_trace = []
                flow_trace = []
                gate_trace = []
                div_trace = []

                rd = region_data[region]
                for t in t_axis:
                    # FR
                    fr_idx = get_fr_idx(region, t)
                    fr_idx = min(fr_idx, len(rd['pop_fr']) - 1)
                    fr_trace.append(rd['pop_fr'][fr_idx])
                    # PC1
                    h_idx = get_h_idx(region, t)
                    h_idx = min(h_idx, len(rd['h_pcs']) - 1)
                    pc1_trace.append(rd['h_pcs'][h_idx, 0])
                    # Flow + Gate
                    s, g = get_flow_metrics(region, t)
                    flow_trace.append(s)
                    gate_trace.append(g)

                # Divergence (batch for speed)
                h_indices = [min(get_h_idx(region, t), len(rd['h_states']) - 1)
                             for t in t_axis]
                h_pts = rd['h_states'][h_indices]
                div_vals = compute_local_divergence(rd['ode_func'], h_pts)
                div_trace = div_vals

                traces = [np.array(fr_trace), np.array(pc1_trace),
                          np.array(flow_trace), np.array(gate_trace),
                          np.array(div_trace)]

                # Smooth
                sm_win = max(3, int(1.0 * RESAMPLE_HZ))  # 1s smoothing
                traces_sm = [uniform_filter1d(tr_data, sm_win) for tr_data in traces]

                for ri_row, (trace, label) in enumerate(zip(traces_sm, row_labels)):
                    ax = axes[ri_row, ci]
                    ax.plot(t_axis, trace, color='black', linewidth=1.5)

                    # Shade pot occupancy periods
                    # Get ALL pot visits in this excursion within our window
                    exc_visits = pv_df[pv_df['excursion_idx'] == tr['exc_idx']]
                    pot_colors = {'Pot-1': 'orange', 'Pot-2': 'red',
                                  'Pot-3': 'blue', 'Pot-4': 'green'}
                    for _, pv in exc_visits.iterrows():
                        if pv['end_s'] >= t_start and pv['start_s'] <= t_end:
                            ax.axvspan(pv['start_s'], pv['end_s'],
                                       alpha=0.2, color=pot_colors.get(pv['pot'], 'gray'))
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
            plt.savefig(f'figures/foraging_transition_dynamics_s2_{region.lower()}.png',
                        dpi=100, bbox_inches='tight')
            plt.close()
            print(f"  Saved: figures/foraging_transition_dynamics_s2_{region.lower()}.png")

        # --- Summary comparison: pre vs post transition metrics ---
        print(f"\n  Transition summary (arrival metrics at Pot-2 and Pot-4):")

        for region in ['LHA', 'RSP']:
            print(f"\n  {region}:")
            for _, tr in trans_df.iterrows():
                phase = 'PRE' if tr['pre_discovery'] else 'POST'
                # Metrics at pot1 arrival and pot2 arrival
                m1 = {}
                m2 = {}
                for label, t, store in [('pot1', tr['t1_start'], m1),
                                         ('pot2', tr['t2_start'], m2)]:
                    fr_idx = get_fr_idx(region, t)
                    fr = region_data[region]['pop_fr']
                    fw = fr[fr_idx:min(fr_idx + 10, len(fr))]
                    store['FR'] = np.mean(fw) if len(fw) > 0 else np.nan
                    h_idx = get_h_idx(region, t)
                    pcs = region_data[region]['h_pcs']
                    pw = pcs[h_idx:min(h_idx + 100, len(pcs)), 0]
                    store['PC1'] = np.mean(pw) if len(pw) > 0 else np.nan
                    s, g = get_flow_metrics(region, t)
                    store['Flow'] = s
                    store['Gate'] = g

                print(f"    Exc {tr['exc_idx']:.0f} {tr['direction']} [{phase}]: "
                      f"FR {m1['FR']:.2f}->{m2['FR']:.2f} "
                      f"(d={m2['FR']-m1['FR']:+.2f}), "
                      f"PC1 {m1['PC1']:.2f}->{m2['PC1']:.2f} "
                      f"(d={m2['PC1']-m1['PC1']:+.2f}), "
                      f"Flow {m1['Flow']:.2f}->{m2['Flow']:.2f}, "
                      f"Gate {m1['Gate']:.3f}->{m2['Gate']:.3f}")

    print(f"\n{'='*70}")
    print(f"  DONE — All analyses complete")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
