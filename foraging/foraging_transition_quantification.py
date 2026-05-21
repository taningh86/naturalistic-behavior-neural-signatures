"""
Quantify neural state changes during P2<->P4 transitions.

For each post-discovery transition, extract mean FR, PC1, Flow, Gate
during Pot-2 occupancy vs Pot-4 occupancy within the same excursion.
Then test whether the P2->P4 difference is consistent across transitions.

Tests:
1. Paired Wilcoxon signed-rank (within-transition P2 vs P4)
2. One-sample t-test on deltas (is mean delta != 0?)
3. Cross-session consistency (do all sessions show same direction?)
4. Shuffle: randomly swap P2/P4 labels within each transition, 10,000x
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
N_SHUFFLES = 10000

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

    return pv_df, region_data, discovery_time, pot_signals, n_bins


# =============================================================================
# METRIC EXTRACTION OVER TIME WINDOWS
# =============================================================================
def get_mean_metrics_window(region_data, region, t_start, t_end):
    """Return mean FR, PC1, flow, gate over a time window [t_start, t_end]."""
    rd = region_data[region]

    # FR: average over 100ms bins in window
    fr = rd['pop_fr']
    fr_offset = rd['offset_100ms']
    fr_start = max(0, int((t_start - fr_offset) / 0.1))
    fr_end = min(len(fr), int((t_end - fr_offset) / 0.1))
    fr_val = np.mean(fr[fr_start:fr_end]) if fr_end > fr_start else np.nan

    # PC1: average over 10ms bins in window
    h_offset = rd['offset_10ms']
    pcs = rd['h_pcs']
    h_start = max(0, int((t_start - h_offset) / 0.01))
    h_end = min(len(pcs), int((t_end - h_offset) / 0.01))
    pc1_val = np.mean(pcs[h_start:h_end, 0]) if h_end > h_start else np.nan

    # Flow and gate: sample at multiple points and average
    h = rd['h_states']
    ode_func = rd['ode_func']
    n_samples = max(1, min(10, (h_end - h_start) // 100))
    sample_idxs = np.linspace(h_start, max(h_start, h_end - 1), n_samples, dtype=int)
    sample_idxs = np.clip(sample_idxs, 0, len(h) - 1)

    speeds, gates = [], []
    for idx in sample_idxs:
        pt = h[idx:idx + 1]
        dhdt = evaluate_flow(ode_func, pt)
        speeds.append(np.linalg.norm(dhdt))
        ht = torch.tensor(pt, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            g = ode_func.update_gate(ht).cpu().numpy().mean()
        gates.append(g)

    flow_val = np.mean(speeds)
    gate_val = np.mean(gates)

    return fr_val, pc1_val, flow_val, gate_val


# =============================================================================
# FIND P2<->P4 TRANSITIONS
# =============================================================================
def find_transitions(pv_df, discovery_time):
    """Find post-discovery excursions that contain both P2 and P4 visits."""
    post = pv_df[~pv_df['pre_discovery']].copy()
    post = post[post['pot'].isin(['Pot-2', 'Pot-4'])].copy()

    transitions = []
    for exc_idx, grp in post.groupby('excursion_idx'):
        if pd.isna(exc_idx):
            continue
        grp = grp.sort_values('start_s')
        pots_visited = grp['pot'].unique()
        if 'Pot-2' in pots_visited and 'Pot-4' in pots_visited:
            p2_visits = grp[grp['pot'] == 'Pot-2']
            p4_visits = grp[grp['pot'] == 'Pot-4']

            # Determine direction: first pot visited determines direction
            first_pot = grp.iloc[0]['pot']
            if first_pot == 'Pot-2':
                direction = 'P2->P4'
            else:
                direction = 'P4->P2'

            transitions.append({
                'exc_idx': int(exc_idx),
                'direction': direction,
                'p2_visits': p2_visits,
                'p4_visits': p4_visits,
                'all_visits': grp,
            })

    return transitions


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("  P2<->P4 TRANSITION QUANTIFICATION")
    print("  Post-discovery: mean neural state at P2 vs P4 within each transition")
    print("=" * 70)

    sessions = [(2, 'fed'), (4, 'fed'), (6, 'fasted'), (8, 'fasted')]
    regions = ['LHA', 'RSP']
    metrics = ['FR', 'PC1', 'Flow', 'Gate']

    all_transition_rows = []

    for sess, state in sessions:
        pv_df, region_data, disc_time, pot_signals, n_bins = load_session(sess, state)
        transitions = find_transitions(pv_df, disc_time)
        print(f"    Post-disc P2<->P4 transitions: {len(transitions)}")

        for tr in transitions:
            exc_idx = tr['exc_idx']
            direction = tr['direction']
            p2v = tr['p2_visits']
            p4v = tr['p4_visits']

            row = {
                'session': sess, 'state': state,
                'exc_idx': exc_idx, 'direction': direction,
                'n_p2': len(p2v), 'n_p4': len(p4v),
            }

            for region in regions:
                # Mean metrics during P2 occupancy
                p2_metrics = []
                for _, v in p2v.iterrows():
                    m = get_mean_metrics_window(region_data, region,
                                                 v['start_s'], v['end_s'])
                    p2_metrics.append(m)
                p2_mean = np.mean(p2_metrics, axis=0)

                # Mean metrics during P4 occupancy
                p4_metrics = []
                for _, v in p4v.iterrows():
                    m = get_mean_metrics_window(region_data, region,
                                                 v['start_s'], v['end_s'])
                    p4_metrics.append(m)
                p4_mean = np.mean(p4_metrics, axis=0)

                for mi, metric in enumerate(metrics):
                    row[f'{region}_{metric}_P2'] = p2_mean[mi]
                    row[f'{region}_{metric}_P4'] = p4_mean[mi]
                    row[f'{region}_{metric}_delta'] = p4_mean[mi] - p2_mean[mi]

            all_transition_rows.append(row)
            print(f"      Exc {exc_idx} ({direction}): "
                  f"LHA FR delta={row['LHA_FR_delta']:+.3f}, "
                  f"RSP FR delta={row['RSP_FR_delta']:+.3f}")

    df = pd.DataFrame(all_transition_rows)
    df.to_csv('data/foraging_transition_quantification.csv', index=False)
    print(f"\n  Saved: data/foraging_transition_quantification.csv ({len(df)} transitions)")

    # =========================================================================
    # STATISTICS
    # =========================================================================
    print(f"\n{'='*70}")
    print("  PAIRED STATISTICS: P4 vs P2 within each transition")
    print(f"{'='*70}")

    # Pooled across all sessions
    print(f"\n  --- ALL SESSIONS POOLED (n={len(df)}) ---")
    sig_results = {}
    for region in regions:
        for metric in metrics:
            col_p2 = f'{region}_{metric}_P2'
            col_p4 = f'{region}_{metric}_P4'
            col_delta = f'{region}_{metric}_delta'

            p2_vals = df[col_p2].values
            p4_vals = df[col_p4].values
            deltas = df[col_delta].values
            mean_delta = np.mean(deltas)
            n = len(deltas)

            # Wilcoxon signed-rank
            stat, p_wilc = sp_stats.wilcoxon(deltas)
            # One-sample t-test
            t_stat, p_ttest = sp_stats.ttest_1samp(deltas, 0)
            # Sign consistency: how many transitions show same direction?
            n_pos = np.sum(deltas > 0)
            n_neg = np.sum(deltas < 0)
            # Sign test (binomial)
            p_sign = sp_stats.binomtest(n_pos, n, 0.5).pvalue

            sig = "*" if p_wilc < 0.05 else "ns"
            key = (region, metric)
            sig_results[key] = {
                'mean_delta': mean_delta, 'p_wilc': p_wilc,
                'p_ttest': p_ttest, 'p_sign': p_sign,
                'n_pos': n_pos, 'n_neg': n_neg, 'n': n,
            }

            print(f"    {region} {metric:6s}: mean_delta={mean_delta:+.3f} "
                  f"Wilcoxon p={p_wilc:.4f} ({sig}) "
                  f"t-test p={p_ttest:.4f} "
                  f"sign: {n_pos}+/{n_neg}- (p={p_sign:.4f})")

    # Per-session breakdown
    for sess, state in sessions:
        sdf = df[df['session'] == sess]
        n = len(sdf)
        if n < 3:
            print(f"\n  --- S{sess} ({state}): n={n}, too few for stats ---")
            for region in regions:
                for metric in metrics:
                    deltas = sdf[f'{region}_{metric}_delta'].values
                    if len(deltas) > 0:
                        print(f"    {region} {metric:6s}: deltas = "
                              f"{[f'{d:+.3f}' for d in deltas]}")
            continue

        print(f"\n  --- S{sess} ({state}, n={n}) ---")
        for region in regions:
            for metric in metrics:
                deltas = sdf[f'{region}_{metric}_delta'].values
                mean_d = np.mean(deltas)
                if n >= 3:
                    stat, p = sp_stats.wilcoxon(deltas)
                else:
                    p = np.nan
                n_pos = np.sum(deltas > 0)
                n_neg = np.sum(deltas < 0)
                sig = "*" if p < 0.05 else "ns"
                print(f"    {region} {metric:6s}: mean_delta={mean_d:+.3f} "
                      f"p={p:.4f} ({sig}) [{n_pos}+/{n_neg}-]")

    # =========================================================================
    # Fed vs Fasted comparison
    # =========================================================================
    print(f"\n{'='*70}")
    print("  FED vs FASTED: Are transition deltas different by metabolic state?")
    print(f"{'='*70}")

    fed_df = df[df['state'] == 'fed']
    fas_df = df[df['state'] == 'fasted']
    print(f"  Fed: {len(fed_df)} transitions, Fasted: {len(fas_df)} transitions")

    for region in regions:
        for metric in metrics:
            col = f'{region}_{metric}_delta'
            fed_d = fed_df[col].values
            fas_d = fas_df[col].values
            if len(fed_d) >= 2 and len(fas_d) >= 2:
                u_stat, p_mw = sp_stats.mannwhitneyu(fed_d, fas_d, alternative='two-sided')
                sig = "*" if p_mw < 0.05 else "ns"
                print(f"    {region} {metric:6s}: fed={np.mean(fed_d):+.3f} "
                      f"fasted={np.mean(fas_d):+.3f} "
                      f"MWU p={p_mw:.4f} ({sig})")

    # =========================================================================
    # FIGURE 1: Delta distributions (all sessions pooled)
    # =========================================================================
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle('P2<->P4 Transition: Neural State at Pot-4 minus Pot-2\n'
                 'Each dot = one transition | Post-discovery only',
                 fontsize=14, fontweight='bold')

    for ri, region in enumerate(regions):
        for mi, metric in enumerate(metrics):
            ax = axes[ri, mi]
            col_delta = f'{region}_{metric}_delta'

            # Plot by session
            colors = {2: 'royalblue', 4: 'cornflowerblue',
                      6: 'firebrick', 8: 'salmon'}
            labels = {2: 'S2 (Fed)', 4: 'S4 (Fed)',
                      6: 'S6 (Fasted)', 8: 'S8 (Fasted)'}

            x_offset = 0
            all_deltas = []
            for sess, state in sessions:
                sdf = df[df['session'] == sess]
                deltas = sdf[col_delta].values
                all_deltas.extend(deltas)
                x = np.ones(len(deltas)) * x_offset + np.random.normal(0, 0.08, len(deltas))
                ax.scatter(x, deltas, color=colors[sess], s=60,
                           edgecolors='black', linewidths=0.5,
                           label=labels[sess], zorder=5)
                # Mean bar
                if len(deltas) > 0:
                    ax.plot([x_offset - 0.2, x_offset + 0.2],
                            [np.mean(deltas), np.mean(deltas)],
                            color=colors[sess], linewidth=2, zorder=6)
                x_offset += 1

            # Overall mean + stats
            res = sig_results[(region, metric)]
            ax.axhline(0, color='black', linewidth=0.5, linestyle='--')

            # Overall mean
            ax.plot([-0.4, 3.4], [res['mean_delta'], res['mean_delta']],
                    'k-', linewidth=1.5, alpha=0.5)

            title = f'{region} {metric}\n'
            if res['p_wilc'] < 0.001:
                title += f'Wilcoxon p={res["p_wilc"]:.4f}***'
            elif res['p_wilc'] < 0.01:
                title += f'Wilcoxon p={res["p_wilc"]:.4f}**'
            elif res['p_wilc'] < 0.05:
                title += f'Wilcoxon p={res["p_wilc"]:.4f}*'
            else:
                title += f'Wilcoxon p={res["p_wilc"]:.4f} ns'
            title += f' [{res["n_pos"]}+/{res["n_neg"]}-]'
            ax.set_title(title, fontsize=10)
            ax.set_xticks([0, 1, 2, 3])
            ax.set_xticklabels(['S2', 'S4', 'S6', 'S8'])
            ax.set_ylabel(f'Delta (P4 - P2)')
            if ri == 0 and mi == 0:
                ax.legend(fontsize=7, loc='best')

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    plt.savefig('figures/foraging_transition_deltas.png', dpi=100, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: figures/foraging_transition_deltas.png")

    # =========================================================================
    # FIGURE 2: Per-transition paired lines (P2 vs P4)
    # =========================================================================
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle('P2<->P4 Transition: Paired Neural State\n'
                 'Lines connect P2 and P4 values within same excursion',
                 fontsize=14, fontweight='bold')

    for ri, region in enumerate(regions):
        for mi, metric in enumerate(metrics):
            ax = axes[ri, mi]
            col_p2 = f'{region}_{metric}_P2'
            col_p4 = f'{region}_{metric}_P4'

            colors = {2: 'royalblue', 4: 'cornflowerblue',
                      6: 'firebrick', 8: 'salmon'}

            for idx, row in df.iterrows():
                sess = row['session']
                p2_val = row[col_p2]
                p4_val = row[col_p4]
                ax.plot([0, 1], [p2_val, p4_val], '-o', color=colors[sess],
                        alpha=0.6, markersize=5, markeredgecolor='black',
                        markeredgewidth=0.3)

            # Group means
            p2_mean = df[col_p2].mean()
            p4_mean = df[col_p4].mean()
            ax.plot([0, 1], [p2_mean, p4_mean], 'k-o', linewidth=3,
                    markersize=10, markeredgecolor='black', markeredgewidth=1,
                    zorder=10, label=f'Mean')

            res = sig_results[(region, metric)]
            ax.set_title(f'{region} {metric}\np={res["p_wilc"]:.4f}',
                         fontsize=10)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Pot-2', 'Pot-4'])
            ax.set_ylabel(metric)

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    plt.savefig('figures/foraging_transition_paired.png', dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved: figures/foraging_transition_paired.png")

    # =========================================================================
    # FIGURE 3: Per-session breakdown (heatmap of deltas)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('P2->P4 Transition Deltas by Session\n'
                 'Green = P4 higher | Red = P2 higher | Bold = session Wilcoxon p<0.05',
                 fontsize=13, fontweight='bold')

    for ri, region in enumerate(regions):
        ax = axes[ri]
        sess_labels = ['S2\n(Fed)', 'S4\n(Fed)', 'S6\n(Fasted)', 'S8\n(Fasted)', 'ALL']
        data = np.zeros((len(metrics), 5))
        annot = [['' for _ in range(5)] for _ in range(len(metrics))]

        for mi, metric in enumerate(metrics):
            for si, (sess, state) in enumerate(sessions):
                sdf = df[df['session'] == sess]
                deltas = sdf[f'{region}_{metric}_delta'].values
                mean_d = np.mean(deltas) if len(deltas) > 0 else 0
                data[mi, si] = mean_d
                n = len(deltas)
                if n >= 3:
                    _, p = sp_stats.wilcoxon(deltas)
                    sig = '*' if p < 0.05 else ''
                    annot[mi][si] = f'{mean_d:+.2f}{sig}\n(n={n})'
                else:
                    annot[mi][si] = f'{mean_d:+.2f}\n(n={n})'

            # ALL pooled
            res = sig_results[(region, metric)]
            sig = '*' if res['p_wilc'] < 0.05 else ''
            data[mi, 4] = res['mean_delta']
            annot[mi][4] = f'{res["mean_delta"]:+.2f}{sig}\n(n={res["n"]})'

        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
        ax.set_xticks(range(5))
        ax.set_xticklabels(sess_labels)
        ax.set_yticks(range(len(metrics)))
        ax.set_yticklabels(metrics)
        ax.set_title(f'{region}', fontsize=12, fontweight='bold')

        for mi in range(len(metrics)):
            for si in range(5):
                ax.text(si, mi, annot[mi][si], ha='center', va='center',
                        fontsize=8, fontweight='bold' if '*' in annot[mi][si] else 'normal')

    plt.colorbar(im, ax=axes, label='Mean delta (P4 - P2)', shrink=0.8)
    plt.tight_layout(rect=[0, 0, 0.92, 0.90])
    plt.savefig('figures/foraging_transition_heatmap.png', dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved: figures/foraging_transition_heatmap.png")

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*70}")
    print("  SUMMARY: Significant P4-vs-P2 differences (pooled, Wilcoxon p<0.05)")
    print(f"{'='*70}")
    any_sig = False
    for (region, metric), res in sorted(sig_results.items()):
        if res['p_wilc'] < 0.05:
            any_sig = True
            direction = "P4 > P2" if res['mean_delta'] > 0 else "P4 < P2"
            print(f"    {region} {metric}: {direction} "
                  f"(delta={res['mean_delta']:+.3f}, p={res['p_wilc']:.4f}, "
                  f"{res['n_pos']}+/{res['n_neg']}-)")
    if not any_sig:
        print("    None")

    # =========================================================================
    # SHUFFLE CONTROL: Randomly swap P2/P4 labels within each transition
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"  SHUFFLE CONTROL ({N_SHUFFLES} permutations)")
    print(f"  For each permutation: independently flip P2/P4 for each transition,")
    print(f"  recompute Wilcoxon on shuffled deltas. Compare observed stat to null.")
    print(f"{'='*70}")

    rng = np.random.default_rng(42)
    shuffle_results = {}

    for region in regions:
        for metric in metrics:
            col_p2 = f'{region}_{metric}_P2'
            col_p4 = f'{region}_{metric}_P4'

            p2_vals = df[col_p2].values
            p4_vals = df[col_p4].values
            n = len(p2_vals)
            obs_deltas = p4_vals - p2_vals

            # Observed Wilcoxon statistic
            obs_stat, obs_p = sp_stats.wilcoxon(obs_deltas)

            # Shuffle: for each permutation, randomly flip each pair
            null_stats = np.zeros(N_SHUFFLES)
            for si in range(N_SHUFFLES):
                flip = rng.choice([-1, 1], size=n)
                shuf_deltas = obs_deltas * flip
                try:
                    null_stats[si], _ = sp_stats.wilcoxon(shuf_deltas)
                except ValueError:
                    null_stats[si] = 0

            # Two-tailed: proportion of null stats <= observed (smaller = more extreme for Wilcoxon)
            p_shuffle = np.mean(null_stats <= obs_stat)

            key = (region, metric)
            shuffle_results[key] = {
                'obs_stat': obs_stat,
                'obs_p': obs_p,
                'p_shuffle': p_shuffle,
                'null_median': np.median(null_stats),
                'null_95ci_lo': np.percentile(null_stats, 2.5),
                'null_95ci_hi': np.percentile(null_stats, 97.5),
            }

            sig = "*" if p_shuffle < 0.05 else "ns"
            print(f"    {region} {metric:6s}: Wilcoxon stat={obs_stat:.1f}, "
                  f"p_param={obs_p:.4f}, p_shuffle={p_shuffle:.4f} ({sig}) "
                  f"[null 95%CI: {np.percentile(null_stats, 2.5):.1f}-"
                  f"{np.percentile(null_stats, 97.5):.1f}]")

    # Per-session shuffle
    print(f"\n  --- Per-session shuffle ---")
    session_shuffle = {}
    for sess, state in sessions:
        sdf = df[df['session'] == sess]
        n = len(sdf)
        if n < 3:
            print(f"\n  S{sess} ({state}): n={n}, too few for shuffle")
            continue
        print(f"\n  S{sess} ({state}, n={n}):")
        for region in regions:
            for metric in metrics:
                col_p2 = f'{region}_{metric}_P2'
                col_p4 = f'{region}_{metric}_P4'
                p2_vals = sdf[col_p2].values
                p4_vals = sdf[col_p4].values
                obs_deltas = p4_vals - p2_vals
                try:
                    obs_stat, obs_p = sp_stats.wilcoxon(obs_deltas)
                except ValueError:
                    continue

                null_stats = np.zeros(N_SHUFFLES)
                for si in range(N_SHUFFLES):
                    flip = rng.choice([-1, 1], size=n)
                    shuf_deltas = obs_deltas * flip
                    try:
                        null_stats[si], _ = sp_stats.wilcoxon(shuf_deltas)
                    except ValueError:
                        null_stats[si] = 0

                p_shuffle = np.mean(null_stats <= obs_stat)
                sig = "*" if p_shuffle < 0.05 else "ns"
                session_shuffle[(sess, region, metric)] = {
                    'obs_stat': obs_stat, 'p_param': obs_p, 'p_shuffle': p_shuffle,
                    'n': n, 'mean_delta': np.mean(obs_deltas),
                }
                if p_shuffle < 0.1:  # show trending and significant
                    print(f"    {region} {metric:6s}: p_shuffle={p_shuffle:.4f} ({sig}) "
                          f"delta={np.mean(obs_deltas):+.3f}")

    # Save shuffle results
    shuffle_rows = []
    for (region, metric), res in shuffle_results.items():
        shuffle_rows.append({
            'session': 'ALL', 'region': region, 'metric': metric,
            'obs_stat': res['obs_stat'], 'p_param': res['obs_p'],
            'p_shuffle': res['p_shuffle'],
            'null_95ci_lo': res['null_95ci_lo'],
            'null_95ci_hi': res['null_95ci_hi'],
        })
    for (sess, region, metric), res in session_shuffle.items():
        shuffle_rows.append({
            'session': f'S{sess}', 'region': region, 'metric': metric,
            'obs_stat': res['obs_stat'], 'p_param': res['p_param'],
            'p_shuffle': res['p_shuffle'],
            'mean_delta': res['mean_delta'], 'n': res['n'],
        })
    shuf_df = pd.DataFrame(shuffle_rows)
    shuf_df.to_csv('data/foraging_transition_shuffle_results.csv', index=False)
    print(f"\n  Saved: data/foraging_transition_shuffle_results.csv")

    # =========================================================================
    # FIGURE 4: Shuffle null distribution for significant metrics
    # =========================================================================
    sig_keys = [(r, m) for (r, m), res in shuffle_results.items()
                if res['p_shuffle'] < 0.1]
    if sig_keys:
        n_panels = len(sig_keys)
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4),
                                 squeeze=False)
        fig.suptitle('Transition Shuffle Control: Null Distributions\n'
                     'Red line = observed Wilcoxon statistic',
                     fontsize=13, fontweight='bold')
        for pi, (region, metric) in enumerate(sig_keys):
            ax = axes[0, pi]
            res = shuffle_results[(region, metric)]
            # Recompute null for plotting
            p2_vals = df[f'{region}_{metric}_P2'].values
            p4_vals = df[f'{region}_{metric}_P4'].values
            obs_deltas = p4_vals - p2_vals
            null_stats = np.zeros(N_SHUFFLES)
            rng2 = np.random.default_rng(42)
            for si in range(N_SHUFFLES):
                flip = rng2.choice([-1, 1], size=len(obs_deltas))
                shuf_deltas = obs_deltas * flip
                try:
                    null_stats[si], _ = sp_stats.wilcoxon(shuf_deltas)
                except ValueError:
                    null_stats[si] = 0

            ax.hist(null_stats, bins=50, color='gray', alpha=0.7, edgecolor='black',
                    linewidth=0.3)
            ax.axvline(res['obs_stat'], color='red', linewidth=2, linestyle='--')
            sig = "*" if res['p_shuffle'] < 0.05 else "ns"
            ax.set_title(f'{region} {metric}\np_shuffle={res["p_shuffle"]:.4f} ({sig})',
                         fontsize=11)
            ax.set_xlabel('Wilcoxon statistic')
            ax.set_ylabel('Count')

        plt.tight_layout(rect=[0, 0, 1, 0.85])
        plt.savefig('figures/foraging_transition_shuffle.png', dpi=100,
                    bbox_inches='tight')
        plt.close()
        print(f"  Saved: figures/foraging_transition_shuffle.png")

    print("\nDone!")


if __name__ == '__main__':
    main()
