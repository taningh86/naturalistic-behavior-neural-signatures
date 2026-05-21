"""
Shuffle control for LHA PC1 Pot-2 learning ramp.

Tests whether the observed correlation between LHA PC1 and Pot-2 visit number
could arise from session-wide temporal drift rather than pot-specific learning.

Three controls:
1. LABEL SHUFFLE — randomly assign "Pot-2" labels among all pre-discovery visits,
   keeping the same count. Recompute correlation 10,000 times.
2. ALL-VISITS TIME CONTROL — correlate each metric with time across ALL
   pre-discovery visits (not just Pot-2), to see if there's global drift.
3. POT-4 COMPARISON — same correlation for Pot-4 visits. If drift drives
   everything, Pot-4 should show similar ramps.

Sessions: S2 (Fed, 6 P2 visits), S4 (Fed, 17 P2 visits)
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import spikeinterface.extractors as se
import torch
import torch.nn as nn
from torchdiffeq import odeint
from sklearn.decomposition import PCA
from scipy import stats as sp_stats
import warnings, sys

warnings.filterwarnings('ignore')

# =============================================================================
# CONSTANTS (same as foraging_neural_all_sessions.py)
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
N_SHUFFLES = 10000
MIN_VISITS = 3

with open("paths.yaml") as f:
    paths_config = yaml.safe_load(f)

# =============================================================================
# MODEL (identical to foraging_neural_all_sessions.py)
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
# DATA LOADING
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
    """Load behavioral + neural data for one session."""
    print(f"\n  Loading S{sess_num} ({state})...")
    behav, n_bins = load_behavior_data(sess_num)

    # Excursions
    exc_df = pd.read_csv("data/excursion_features_all_sessions.csv")
    exc_df = exc_df[exc_df["session"] == sess_num].copy()

    # Pot signals
    pot_dwells = {}
    for pot in ['Pot-1', 'Pot-2', 'Pot-3', 'Pot-4']:
        sig = get_pot_signal(behav, pot, n_bins)
        pot_dwells[pot] = find_dwell_events(sig, min_bins=10)

    # Feeding for discovery time
    feeding = behav.get('Feeding', np.zeros(n_bins))
    feeding = np.where(np.isnan(feeding), 0, feeding)
    feed_onset_bin = np.argmax(feeding > 0) if np.any(feeding > 0) else n_bins
    discovery_time = feed_onset_bin * 0.1

    # Build pot visit DataFrame
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

    print(f"    {len(lha_ids)} LHA, {len(rsp_ids)} RSP good units")
    print(f"    Discovery: {discovery_time:.1f}s")
    return pv_df, region_data, discovery_time


# =============================================================================
# METRIC EXTRACTION
# =============================================================================
def get_metrics_at_time(region_data, region, t):
    """Return FR, PC1, flow, gate at a time point."""
    rd = region_data[region]

    # FR
    fr_idx = max(0, int((t - rd['offset_100ms']) / 0.1))
    fr = rd['pop_fr']
    fw = fr[fr_idx:min(fr_idx + 10, len(fr))]
    fr_val = np.mean(fw) if len(fw) > 0 else np.nan

    # PC1
    h_idx = max(0, int((t - rd['offset_10ms']) / 0.01))
    pcs = rd['h_pcs']
    pw = pcs[h_idx:min(h_idx + 100, len(pcs)), 0]
    pc1_val = np.mean(pw) if len(pw) > 0 else np.nan

    # Flow + gate
    h = rd['h_states']
    ode_func = rd['ode_func']
    idx = min(h_idx, len(h) - 1)
    pt = h[idx:idx + 1]
    dhdt = evaluate_flow(ode_func, pt)
    speed = np.linalg.norm(dhdt)
    ht = torch.tensor(pt, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        gate = ode_func.update_gate(ht).cpu().numpy().mean()

    return fr_val, pc1_val, speed, gate


def extract_all_pre_disc_metrics(pv_df, region_data, discovery_time):
    """Extract FR, PC1, flow, gate for every pre-discovery visit."""
    pre = pv_df[pv_df['pre_discovery']].sort_values('start_s').copy()
    rows = []
    for _, visit in pre.iterrows():
        t = visit['start_s']
        pot = visit['pot']
        for region in ['LHA', 'RSP']:
            fr_v, pc1_v, flow_v, gate_v = get_metrics_at_time(region_data, region, t)
            rows.append({
                'pot': pot, 'time_s': t, 'region': region,
                'FR': fr_v, 'PC1': pc1_v, 'Flow': flow_v, 'Gate': gate_v,
            })
    return pd.DataFrame(rows)


# =============================================================================
# SHUFFLE TESTS
# =============================================================================
def run_label_shuffle(metrics_df, target_pot, region, metric_name, n_shuffles=N_SHUFFLES):
    """Shuffle pot labels among all pre-disc visits, recompute correlation."""
    sub = metrics_df[metrics_df['region'] == region].copy()
    all_times = sub['time_s'].values
    all_vals = sub[metric_name].values
    n_target = len(sub[sub['pot'] == target_pot])

    if n_target < MIN_VISITS:
        return None

    # Observed correlation
    target_mask = sub['pot'] == target_pot
    target_vals = all_vals[target_mask]
    target_nums = np.arange(1, n_target + 1)
    r_obs, p_obs = sp_stats.pearsonr(target_nums, target_vals)

    # Shuffle
    rng = np.random.default_rng(42)
    n_total = len(sub)
    shuffle_r = np.zeros(n_shuffles)
    for i in range(n_shuffles):
        idx = rng.choice(n_total, size=n_target, replace=False)
        idx.sort()  # preserve temporal order
        shuf_vals = all_vals[idx]
        shuf_nums = np.arange(1, n_target + 1)
        shuffle_r[i], _ = sp_stats.pearsonr(shuf_nums, shuf_vals)

    # p-value: fraction of shuffles with |r| >= |r_obs|
    p_shuffle = np.mean(np.abs(shuffle_r) >= np.abs(r_obs))

    return {
        'r_obs': r_obs, 'p_obs': p_obs,
        'p_shuffle': p_shuffle,
        'shuffle_r': shuffle_r,
        'n_target': n_target, 'n_total': n_total,
    }


def run_all_visits_time_control(metrics_df, region, metric_name):
    """Correlate metric with time across ALL pre-disc visits."""
    sub = metrics_df[metrics_df['region'] == region].copy()
    times = sub['time_s'].values
    vals = sub[metric_name].values

    if len(vals) < MIN_VISITS:
        return None

    r, p = sp_stats.pearsonr(times, vals)
    return {'r': r, 'p': p, 'n': len(vals)}


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("  SHUFFLE CONTROL: LHA PC1 Pot-2 Learning Ramp")
    print(f"  {N_SHUFFLES} permutations per test")
    print("=" * 70)

    sessions = [(2, 'fed'), (4, 'fed')]
    all_results = []
    metrics_list = ['FR', 'PC1', 'Flow', 'Gate']
    regions = ['LHA', 'RSP']
    pots = ['Pot-2', 'Pot-4']

    session_data = {}
    for sess, state in sessions:
        pv_df, region_data, disc_time = load_session(sess, state)
        mdf = extract_all_pre_disc_metrics(pv_df, region_data, disc_time)
        session_data[sess] = {'pv_df': pv_df, 'region_data': region_data,
                              'disc_time': disc_time, 'metrics_df': mdf}

    # =========================================================================
    # Run all controls
    # =========================================================================
    print(f"\n{'='*70}")
    print("  CONTROL 1: Label Shuffle (10,000 permutations)")
    print(f"{'='*70}")

    shuffle_results = {}
    for sess, state in sessions:
        mdf = session_data[sess]['metrics_df']
        print(f"\n  --- Session {sess} ({state}) ---")
        for region in regions:
            for pot in pots:
                for metric in metrics_list:
                    res = run_label_shuffle(mdf, pot, region, metric)
                    if res is None:
                        continue
                    key = (sess, region, pot, metric)
                    shuffle_results[key] = res
                    sig = "*" if res['p_shuffle'] < 0.05 else "ns"
                    print(f"    {region} {pot} {metric:6s}: "
                          f"r_obs={res['r_obs']:+.3f} p_param={res['p_obs']:.4f} "
                          f"p_shuffle={res['p_shuffle']:.4f} ({sig}) "
                          f"[n={res['n_target']}/{res['n_total']}]")

    print(f"\n{'='*70}")
    print("  CONTROL 2: All-Visits Time Correlation (global drift test)")
    print(f"{'='*70}")

    time_results = {}
    for sess, state in sessions:
        mdf = session_data[sess]['metrics_df']
        print(f"\n  --- Session {sess} ({state}) ---")
        for region in regions:
            for metric in metrics_list:
                res = run_all_visits_time_control(mdf, region, metric)
                if res is None:
                    continue
                key = (sess, region, metric)
                time_results[key] = res
                sig = "*" if res['p'] < 0.05 else "ns"
                print(f"    {region} {metric:6s}: r={res['r']:+.3f} p={res['p']:.4f} "
                      f"({sig}) [n={res['n']}]")

    # =========================================================================
    # Figure: 2 rows (S2, S4) x 3 columns (shuffle hist, pot comparison, time)
    # =========================================================================
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle('Shuffle Controls for LHA PC1 Pot-2 Ramp\n'
                 'Is the trend pot-specific or session-wide drift?',
                 fontsize=14, fontweight='bold')

    for si, (sess, state) in enumerate(sessions):
        mdf = session_data[sess]['metrics_df']

        # --- Column 0: LHA PC1 Pot-2 shuffle distribution ---
        ax = axes[si, 0]
        key_p2 = (sess, 'LHA', 'Pot-2', 'PC1')
        if key_p2 in shuffle_results:
            res = shuffle_results[key_p2]
            ax.hist(res['shuffle_r'], bins=50, color='gray', alpha=0.7,
                    edgecolor='black', linewidth=0.5)
            ax.axvline(res['r_obs'], color='red', linewidth=2, linestyle='--',
                       label=f'Observed r={res["r_obs"]:.2f}')
            ax.axvline(-res['r_obs'], color='red', linewidth=1, linestyle=':',
                       alpha=0.5)
            pctl = np.mean(res['shuffle_r'] <= res['r_obs']) * 100
            ax.set_title(f'S{sess} LHA PC1 @ Pot-2\n'
                         f'p_shuffle={res["p_shuffle"]:.4f} '
                         f'(percentile: {pctl:.1f}%)',
                         fontsize=10)
            ax.set_xlabel('Shuffled r')
            ax.set_ylabel('Count')
            ax.legend(fontsize=8)

        # --- Column 1: LHA PC1 Pot-4 shuffle distribution ---
        ax = axes[si, 1]
        key_p4 = (sess, 'LHA', 'Pot-4', 'PC1')
        if key_p4 in shuffle_results:
            res = shuffle_results[key_p4]
            ax.hist(res['shuffle_r'], bins=50, color='gray', alpha=0.7,
                    edgecolor='black', linewidth=0.5)
            ax.axvline(res['r_obs'], color='green', linewidth=2, linestyle='--',
                       label=f'Observed r={res["r_obs"]:.2f}')
            ax.axvline(-res['r_obs'], color='green', linewidth=1, linestyle=':',
                       alpha=0.5)
            pctl = np.mean(res['shuffle_r'] <= res['r_obs']) * 100
            ax.set_title(f'S{sess} LHA PC1 @ Pot-4\n'
                         f'p_shuffle={res["p_shuffle"]:.4f} '
                         f'(percentile: {pctl:.1f}%)',
                         fontsize=10)
            ax.set_xlabel('Shuffled r')
            ax.legend(fontsize=8)

        # --- Column 2: Pot-2 vs Pot-4 side-by-side scatter ---
        ax = axes[si, 2]
        for pot, color in [('Pot-2', '#E53935'), ('Pot-4', '#43A047')]:
            sub = mdf[(mdf['region'] == 'LHA') & (mdf['pot'] == pot)]
            if len(sub) >= MIN_VISITS:
                vals = sub['PC1'].values
                nums = np.arange(1, len(vals) + 1)
                ax.scatter(nums, vals, color=color, s=50, edgecolors='black',
                           linewidths=0.5, label=f'{pot} (n={len(vals)})', zorder=5)
                z = np.polyfit(nums, vals, 1)
                r, p = sp_stats.pearsonr(nums, vals)
                ax.plot(nums, np.polyval(z, nums), color=color, linewidth=1.5,
                        linestyle='--', alpha=0.6)
                ax.text(0.95, 0.95 - (0.12 if pot == 'Pot-4' else 0),
                        f'{pot}: r={r:.2f}, p={p:.3f}',
                        transform=ax.transAxes, fontsize=8, ha='right', va='top',
                        color=color)
        ax.set_title(f'S{sess} LHA PC1: Pot-2 vs Pot-4', fontsize=10)
        ax.set_xlabel('Visit # (within pot)')
        ax.set_ylabel('PC1')
        ax.legend(fontsize=8)

        # --- Column 3: All-visits time correlation ---
        ax = axes[si, 3]
        sub = mdf[mdf['region'] == 'LHA']
        for pot, color, marker in [('Pot-2', '#E53935', 'o'),
                                    ('Pot-4', '#43A047', 's'),
                                    ('Pot-1', '#FF9800', '^'),
                                    ('Pot-3', '#1E88E5', 'D')]:
            psub = sub[sub['pot'] == pot]
            if len(psub) > 0:
                ax.scatter(psub['time_s'], psub['PC1'], color=color, marker=marker,
                           s=40, edgecolors='black', linewidths=0.5,
                           label=f'{pot} (n={len(psub)})', zorder=5)

        # Global trend line
        key_time = (sess, 'LHA', 'PC1')
        if key_time in time_results:
            tres = time_results[key_time]
            all_t = sub['time_s'].values
            all_v = sub['PC1'].values
            z = np.polyfit(all_t, all_v, 1)
            t_line = np.linspace(all_t.min(), all_t.max(), 100)
            ax.plot(t_line, np.polyval(z, t_line), 'k--', linewidth=1.5, alpha=0.6,
                    label=f'All: r={tres["r"]:.2f}, p={tres["p"]:.3f}')

        ax.set_title(f'S{sess} LHA PC1 vs Time (all visits)', fontsize=10)
        ax.set_xlabel('Time in session (s)')
        ax.set_ylabel('PC1')
        ax.legend(fontsize=7, loc='best')

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    fig_path = 'figures/foraging_shuffle_control_pc1.png'
    plt.savefig(fig_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {fig_path}")

    # =========================================================================
    # Figure 2: Full metric x region shuffle summary
    # =========================================================================
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle('Shuffle Control Summary: All Metrics\n'
                 'Red/green bars = observed r | Gray = shuffle 95% CI',
                 fontsize=14, fontweight='bold')

    for si, (sess, state) in enumerate(sessions):
        for mi, metric in enumerate(metrics_list):
            ax = axes[si, mi]
            x_labels = []
            x_pos = []
            pos = 0
            for region in regions:
                for pot, color in [('Pot-2', '#E53935'), ('Pot-4', '#43A047')]:
                    key = (sess, region, pot, metric)
                    if key in shuffle_results:
                        res = shuffle_results[key]
                        # Shuffle 95% CI
                        ci_lo = np.percentile(res['shuffle_r'], 2.5)
                        ci_hi = np.percentile(res['shuffle_r'], 97.5)
                        ax.barh(pos, ci_hi - ci_lo, left=ci_lo, height=0.6,
                                color='lightgray', edgecolor='gray')
                        # Observed
                        sig = res['p_shuffle'] < 0.05
                        ax.plot(res['r_obs'], pos, 'o', color=color,
                                markersize=10, markeredgecolor='black',
                                markeredgewidth=1.5 if sig else 0.5,
                                zorder=10)
                        if sig:
                            ax.text(res['r_obs'], pos + 0.35,
                                    f'p={res["p_shuffle"]:.3f}*',
                                    fontsize=7, ha='center', fontweight='bold')
                        x_labels.append(f'{region}\n{pot}')
                        x_pos.append(pos)
                        pos += 1

            ax.set_yticks(x_pos)
            ax.set_yticklabels(x_labels, fontsize=8)
            ax.axvline(0, color='black', linewidth=0.5)
            ax.set_xlabel('Pearson r')
            ax.set_title(f'S{sess} {metric}', fontsize=10)
            ax.set_xlim(-1, 1)
            ax.invert_yaxis()

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    fig_path = 'figures/foraging_shuffle_control_all_metrics.png'
    plt.savefig(fig_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")

    # =========================================================================
    # Save results CSV
    # =========================================================================
    rows = []
    for (sess, region, pot, metric), res in shuffle_results.items():
        rows.append({
            'session': sess, 'region': region, 'pot': pot, 'metric': metric,
            'r_obs': res['r_obs'], 'p_parametric': res['p_obs'],
            'p_shuffle': res['p_shuffle'],
            'n_target': res['n_target'], 'n_total': res['n_total'],
            'shuffle_95ci_lo': np.percentile(res['shuffle_r'], 2.5),
            'shuffle_95ci_hi': np.percentile(res['shuffle_r'], 97.5),
        })
    for (sess, region, metric), res in time_results.items():
        rows.append({
            'session': sess, 'region': region, 'pot': 'ALL',
            'metric': metric,
            'r_obs': res['r'], 'p_parametric': res['p'],
            'p_shuffle': np.nan,
            'n_target': res['n'], 'n_total': res['n'],
            'shuffle_95ci_lo': np.nan, 'shuffle_95ci_hi': np.nan,
        })

    out_df = pd.DataFrame(rows).sort_values(
        ['session', 'region', 'pot', 'metric']).reset_index(drop=True)
    csv_path = 'data/foraging_shuffle_control_results.csv'
    out_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*70}")
    print("  INTERPRETATION SUMMARY")
    print(f"{'='*70}")

    for sess, state in sessions:
        print(f"\n  S{sess} ({state}):")
        key_p2 = (sess, 'LHA', 'Pot-2', 'PC1')
        key_p4 = (sess, 'LHA', 'Pot-4', 'PC1')
        key_time = (sess, 'LHA', 'PC1')

        if key_p2 in shuffle_results:
            r2 = shuffle_results[key_p2]
            print(f"    LHA PC1 @ Pot-2: r={r2['r_obs']:+.3f}, "
                  f"p_shuffle={r2['p_shuffle']:.4f}")
        if key_p4 in shuffle_results:
            r4 = shuffle_results[key_p4]
            print(f"    LHA PC1 @ Pot-4: r={r4['r_obs']:+.3f}, "
                  f"p_shuffle={r4['p_shuffle']:.4f}")
        if key_time in time_results:
            rt = time_results[key_time]
            print(f"    All-visits time: r={rt['r']:+.3f}, p={rt['p']:.4f}")

        # Interpretation
        if key_p2 in shuffle_results and key_time in time_results:
            r2 = shuffle_results[key_p2]
            rt = time_results[key_time]
            if r2['p_shuffle'] < 0.05 and rt['p'] >= 0.05:
                print(f"    --> POT-SPECIFIC: Pot-2 ramp survives shuffle, no global drift")
            elif r2['p_shuffle'] < 0.05 and rt['p'] < 0.05:
                print(f"    --> MIXED: Pot-2 ramp survives shuffle but global drift present")
            elif r2['p_shuffle'] >= 0.05 and rt['p'] < 0.05:
                print(f"    --> DRIFT: Pot-2 ramp explained by session-wide drift")
            else:
                print(f"    --> INCONCLUSIVE: Neither pot-specific nor drift significant")

    print("\nDone!")


if __name__ == '__main__':
    main()
