"""
Entropy-Neural Signatures: GRU-ODE Specific Metrics
====================================================

Uses SAVED hidden states (gru_ode_10ms_hidden_{region}_s{1-8}.npy) and
trained models to extract GRU-ODE-unique metrics, then correlates with
behavioral entropy.

Metrics:
1. Flow speed — |dh/dt| from learned ODE at each hidden state
2. Update gate z(h) — mean gate activation (stability)
3. Observation jump — |h[t] - ODE_evolve(h[t-1])| (data correction magnitude)
4. ODE-observation alignment — cosine(ODE direction, actual update)
5. Latent PR — sliding-window participation ratio of hidden states

All computed in batch from saved hidden states — fast.
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchdiffeq import odeint
from scipy.stats import spearmanr, mannwhitneyu
from scipy.ndimage import gaussian_filter1d
from collections import Counter
from scipy.stats import entropy as sp_entropy
import matplotlib.pyplot as plt
import warnings
import time as timer

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG
# =============================================================================
with open("paths.yaml") as f:
    cfg = yaml.safe_load(f)

BIN_SIZE_MS = 10
D_SHARED = 32
HIDDEN_SIZE = 32
ODE_GATE_HIDDEN = 64
ODE_SOLVER = 'rk4'
ODE_STEP_SIZE = 1.0
ODE_DT = 1.0
PRED_STEPS = 10

ENTROPY_WINDOW_SEC = 60
ENTROPY_STEP_SEC = 10

# Batch size for ODE evaluation
BATCH_SIZE = 10000

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sessions_cfg = cfg["single_probe"]["coordinates_1"]["mouse01"]["sessions"]
session_meta = {
    1: ('fed', 'exploration'), 2: ('fed', 'foraging'),
    3: ('fed', 'exploration'), 4: ('fed', 'foraging'),
    5: ('fasted', 'exploration'), 6: ('fasted', 'foraging'),
    7: ('fasted', 'exploration'), 8: ('fasted', 'foraging'),
}

priority_order = [
    'Right corner', 'Left corner', 'Arna center', 'Foraging arena',
    'Home', 'Ladder', 'Transition zone',
    'Pot-1 zone', 'Pot-2 zone', 'Pot-3 zone', 'Pot-4 zone',
    'Pot-1', 'Pot-2', 'Pot-3', 'Pot-4',
]
zone_short = {
    'Home': 'H', 'Ladder': 'L', 'Transition zone': 'T',
    'Foraging arena': 'FA', 'Arna center': 'AC',
    'Pot-1': 'P1', 'Pot-2': 'P2', 'Pot-3': 'P3', 'Pot-4': 'P4',
    'Pot-1 zone': 'P1z', 'Pot-2 zone': 'P2z', 'Pot-3 zone': 'P3z', 'Pot-4 zone': 'P4z',
    'Right corner': 'RC', 'Left corner': 'LC', 'other': 'O',
}


# =============================================================================
# MODEL (must match training script)
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


# =============================================================================
# HELPERS
# =============================================================================
def load_behavior(behav_path):
    df_raw = pd.read_csv(behav_path, header=None)
    var_names = df_raw.iloc[:, 0].values
    time_vals = df_raw.iloc[1, 1:].astype(float).values
    data = df_raw.iloc[:, 1:].values
    behav = {'time': time_vals}
    for i, name in enumerate(var_names):
        if isinstance(name, str):
            behav[name.strip()] = data[i].astype(float)
    return behav


def get_zones(behav):
    n = len(behav['time'])
    zones = np.full(n, 'O', dtype=object)
    for var_name in priority_order:
        if var_name in behav:
            mask = behav[var_name] > 0.5
            zones[mask] = zone_short.get(var_name, var_name)
    return zones


def compute_entropy_trace(zones, time_vals):
    dt = np.median(np.diff(time_vals))
    window_bins = int(ENTROPY_WINDOW_SEC / dt)
    step_bins = int(ENTROPY_STEP_SEC / dt)
    ent_times, ent_vals = [], []
    for start_idx in range(0, len(zones) - window_bins, step_bins):
        end_idx = start_idx + window_bins
        wz = zones[start_idx:end_idx]
        transitions = []
        for j in range(1, len(wz)):
            if wz[j] != wz[j-1]:
                transitions.append(f"{wz[j-1]}->{wz[j]}")
        if len(transitions) < 3:
            continue
        counts = Counter(transitions)
        probs = np.array(list(counts.values()), dtype=float)
        probs /= probs.sum()
        h = sp_entropy(probs, base=2)
        ent_times.append(time_vals[start_idx + window_bins // 2])
        ent_vals.append(h)
    return np.array(ent_times), np.array(ent_vals)


def batch_compute_ode_metrics(model, hidden_states, batch_size=10000):
    """Compute ODE metrics in batches from saved hidden states.

    Args:
        model: trained PooledGRUODE model
        hidden_states: (T, hidden_size) numpy array of saved hidden states

    Returns:
        dict with flow_speed, gate_mean, obs_jump, alignment arrays
    """
    T = len(hidden_states)
    flow_speed = np.zeros(T)
    gate_mean_arr = np.zeros(T)
    obs_jump = np.zeros(T)
    alignment = np.zeros(T)

    t_span = torch.tensor([0.0, ODE_DT]).to(DEVICE)

    with torch.no_grad():
        # 1. Flow speed and gate z(h) — batch evaluate at all hidden states
        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            h_batch = torch.tensor(hidden_states[start:end], dtype=torch.float32).to(DEVICE)

            # ODE velocity: dh/dt = (1-z)*(n-h)
            dhdt = model.ode_func(0, h_batch)
            flow_speed[start:end] = torch.norm(dhdt, dim=1).cpu().numpy()

            # Update gate
            z = model.ode_func.update_gate(h_batch)
            gate_mean_arr[start:end] = z.mean(dim=1).cpu().numpy()

        # 2. Observation jump and alignment — need ODE evolve from h[t-1] to compare with h[t]
        # obs_jump[t] = |h[t] - ode_evolve(h[t-1])|
        # alignment[t] = cosine(dhdt at h[t-1], h[t] - ode_evolve(h[t-1]))
        for start in range(0, T - 1, batch_size):
            end = min(start + batch_size, T - 1)
            h_prev = torch.tensor(hidden_states[start:end], dtype=torch.float32).to(DEVICE)
            h_next = torch.tensor(hidden_states[start+1:end+1], dtype=torch.float32).to(DEVICE)

            # ODE evolve h[t-1] one step
            h_evolved = odeint(
                model.ode_func, h_prev, t_span,
                method=ODE_SOLVER, options={'step_size': ODE_STEP_SIZE},
            )[-1]

            # Observation jump: how far actual h[t] is from where ODE predicted
            obs_update = h_next - h_evolved
            obs_jump[start+1:end+1] = torch.norm(obs_update, dim=1).cpu().numpy()

            # ODE velocity at h[t-1]
            dhdt_prev = model.ode_func(0, h_prev)

            # Alignment: cosine between ODE direction and actual update
            cos_sim = torch.nn.functional.cosine_similarity(dhdt_prev, obs_update, dim=1)
            alignment[start+1:end+1] = cos_sim.cpu().numpy()

    obs_jump[0] = obs_jump[1]  # fill first element
    alignment[0] = alignment[1]

    return {
        'flow_speed': flow_speed,
        'gate_mean': gate_mean_arr,
        'obs_jump': obs_jump,
        'alignment': alignment,
    }


def sliding_window_pr(hidden_states, window_size=6000, step_size=1000):
    """Participation ratio in sliding windows."""
    n_steps = hidden_states.shape[0]
    pr_times, pr_vals = [], []
    for start in range(0, n_steps - window_size, step_size):
        chunk = hidden_states[start:start+window_size]
        cov = np.cov(chunk.T)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = eigvals[eigvals > 0]
        pr = (np.sum(eigvals)**2) / np.sum(eigvals**2)
        center_time = (start + window_size // 2) * BIN_SIZE_MS / 1000.0
        pr_times.append(center_time)
        pr_vals.append(pr)
    return np.array(pr_times), np.array(pr_vals)


# =============================================================================
# MAIN
# =============================================================================
all_stats = []
all_session_data = {}

for region in ['lha', 'rsp']:
    model_path = f"data/gru_ode_10ms_poisson_{region}_combined_model.pt"
    if not Path(model_path).exists():
        print(f"  Model not found: {model_path}")
        continue

    print(f"\n{'='*70}")
    print(f"REGION: {region.upper()}")
    print(f"{'='*70}")

    # Load model
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    neuron_counts = checkpoint['neuron_counts']
    model = PooledGRUODE(
        session_neuron_counts=neuron_counts,
        d_shared=D_SHARED, hidden_size=HIDDEN_SIZE,
        gate_hidden=ODE_GATE_HIDDEN, pred_steps=PRED_STEPS,
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  Loaded model")

    for snum in range(1, 9):
        t0 = timer.time()
        state, phase = session_meta[snum]
        sc = sessions_cfg[f"session_{snum}"]
        behav_path = sc.get('behavior')

        print(f"\n  --- S{snum} ({state}/{phase}) ---")

        if not behav_path or not Path(behav_path).exists():
            print("    SKIP -- no behavior")
            continue

        # Load saved hidden states
        hidden_path = f"data/gru_ode_10ms_hidden_{region}_s{snum}.npy"
        if not Path(hidden_path).exists():
            print(f"    SKIP -- no hidden states at {hidden_path}")
            continue
        hidden_states = np.load(hidden_path)
        print(f"    Hidden states: {hidden_states.shape}")

        # Entropy trace
        behav = load_behavior(behav_path)
        time_vals = behav['time']
        zones = get_zones(behav)
        ent_times, ent_vals = compute_entropy_trace(zones, time_vals)

        # Batch compute ODE metrics
        ode_raw = batch_compute_ode_metrics(model, hidden_states, BATCH_SIZE)

        # Time axis for hidden states (10ms resolution)
        ode_times = np.arange(len(hidden_states)) * BIN_SIZE_MS / 1000.0

        # Smooth to entropy timescale (~60s Gaussian)
        sigma_bins = int(60 * 1000 / BIN_SIZE_MS) / 6  # ~1000 bins sigma
        flow_speed_s = gaussian_filter1d(ode_raw['flow_speed'], sigma_bins)
        gate_mean_s = gaussian_filter1d(ode_raw['gate_mean'], sigma_bins)
        obs_jump_s = gaussian_filter1d(ode_raw['obs_jump'], sigma_bins)
        alignment_s = gaussian_filter1d(ode_raw['alignment'], sigma_bins)

        # Sliding-window PR
        pr_step = int(ENTROPY_STEP_SEC * 1000 / BIN_SIZE_MS)
        pr_window = int(ENTROPY_WINDOW_SEC * 1000 / BIN_SIZE_MS)
        pr_times, pr_vals = sliding_window_pr(hidden_states, pr_window, pr_step)

        # Interpolate at entropy time points
        ode_metrics = {
            'Flow Speed': np.interp(ent_times, ode_times, flow_speed_s),
            'Gate z(h)': np.interp(ent_times, ode_times, gate_mean_s),
            'Obs Jump': np.interp(ent_times, ode_times, obs_jump_s),
            'ODE-Obs Align': np.interp(ent_times, ode_times, alignment_s),
            'Latent PR': np.interp(ent_times, pr_times, pr_vals),
        }

        # Correlations
        print(f"    Spearman correlations (entropy vs GRU-ODE {region.upper()}):")
        for mn, mv in ode_metrics.items():
            rho, p = spearmanr(ent_vals, mv)
            sig = '*' if p < 0.05 else ''
            print(f"      {mn:>15}: rho={rho:+.3f}, p={p:.4f} {sig}")
            all_stats.append({
                'session': snum, 'state': state, 'phase': phase,
                'region': region.upper(), 'metric': mn,
                'analysis': 'spearman', 'rho': rho, 'p': p,
                'significant': p < 0.05,
            })

        # High vs Low entropy
        q25, q75 = np.percentile(ent_vals, 25), np.percentile(ent_vals, 75)
        low_mask = ent_vals <= q25
        high_mask = ent_vals >= q75

        print(f"    High vs Low entropy (Q75={q75:.1f} vs Q25={q25:.1f}):")
        for mn, mv in ode_metrics.items():
            low_v, high_v = mv[low_mask], mv[high_mask]
            if len(low_v) < 3 or len(high_v) < 3:
                continue
            _, p = mannwhitneyu(low_v, high_v, alternative='two-sided')
            sig = '*' if p < 0.05 else ''
            pct = 100 * (np.mean(high_v) - np.mean(low_v)) / (abs(np.mean(low_v)) + 1e-10)
            print(f"      {mn:>15}: low={np.mean(low_v):.4f}, high={np.mean(high_v):.4f}, "
                  f"diff={pct:+.1f}%, p={p:.4f} {sig}")
            all_stats.append({
                'session': snum, 'state': state, 'phase': phase,
                'region': region.upper(), 'metric': mn,
                'analysis': 'high_vs_low', 'low_mean': np.mean(low_v),
                'high_mean': np.mean(high_v), 'pct_diff': pct,
                'p': p, 'significant': p < 0.05,
            })

        # Store for plotting
        all_session_data[(snum, region)] = {
            'state': state, 'phase': phase,
            'ent_times': ent_times, 'ent_vals': ent_vals,
            'ode_metrics': ode_metrics,
        }

        print(f"    Done in {timer.time()-t0:.1f}s")

# =============================================================================
# SAVE
# =============================================================================
stats_df = pd.DataFrame(all_stats)
stats_df.to_csv("data/entropy_gru_ode_stats.csv", index=False)
print(f"\nSaved data/entropy_gru_ode_stats.csv ({len(stats_df)} rows)")


# =============================================================================
# FIGURE 1: Correlation heatmap per region
# =============================================================================
spearman_df = stats_df[stats_df['analysis'] == 'spearman'].copy()
metrics_order = ['Flow Speed', 'Gate z(h)', 'Obs Jump', 'ODE-Obs Align', 'Latent PR']

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle("GRU-ODE Metrics vs Behavioral Entropy", fontsize=14, fontweight='bold')

for ri, region in enumerate(['LHA', 'RSP']):
    ax = axes[ri]
    reg_df = spearman_df[spearman_df['region'] == region]
    present_metrics = [m for m in metrics_order if m in reg_df['metric'].values]

    rho_matrix = np.full((8, len(present_metrics)), np.nan)
    sig_matrix = np.zeros((8, len(present_metrics)), dtype=bool)

    for _, row in reg_df.iterrows():
        si = int(row['session']) - 1
        if row['metric'] in present_metrics:
            mi = present_metrics.index(row['metric'])
            rho_matrix[si, mi] = row['rho']
            sig_matrix[si, mi] = row['significant']

    im = ax.imshow(rho_matrix, cmap='RdBu_r', vmin=-0.6, vmax=0.6, aspect='auto')
    ax.set_xticks(range(len(present_metrics)))
    ax.set_xticklabels(present_metrics, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(8))
    ylabels = [f"S{s+1} ({session_meta[s+1][0][:3]}/{session_meta[s+1][1][:3]})" for s in range(8)]
    ax.set_yticklabels(ylabels, fontsize=9)

    for i in range(8):
        for j in range(len(present_metrics)):
            if not np.isnan(rho_matrix[i, j]):
                if sig_matrix[i, j]:
                    ax.text(j, i, '*', ha='center', va='center', fontsize=14, fontweight='bold')
                ax.text(j, i + 0.3, f"{rho_matrix[i,j]:+.2f}", ha='center', va='center',
                        fontsize=7, color='gray')

    plt.colorbar(im, ax=ax, shrink=0.8, label='Spearman rho')
    ax.set_title(f"{region} (* p<0.05)", fontsize=11)

plt.tight_layout()
plt.savefig("figures/entropy_gru_ode_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/entropy_gru_ode_heatmap.png")


# =============================================================================
# FIGURE 2: Time series for foraging sessions (S2, S4, S6, S8)
# =============================================================================
fig, axes = plt.subplots(5, 4, figsize=(22, 18))
fig.suptitle("GRU-ODE Metrics vs Entropy — Foraging & Exploration Sessions", fontsize=14, fontweight='bold')

plot_sessions = [1, 2, 5, 6]  # fed/exp, fed/for, fas/exp, fas/for
metric_names = ['Flow Speed', 'Gate z(h)', 'Obs Jump', 'ODE-Obs Align', 'Latent PR']
colors = {'lha': '#e74c3c', 'rsp': '#3498db'}

for col_idx, snum in enumerate(plot_sessions):
    state, phase = session_meta[snum]
    axes[0, col_idx].set_title(f"S{snum} ({state}/{phase[:3]})", fontsize=11)

    for row_idx, mn in enumerate(metric_names):
        ax = axes[row_idx, col_idx]

        # Plot entropy on primary axis
        for region in ['lha', 'rsp']:
            key = (snum, region)
            if key not in all_session_data:
                continue
            d = all_session_data[key]
            if row_idx == 0:
                ax.plot(d['ent_times'], d['ent_vals'], color='black', linewidth=1, alpha=0.4)
            ax2 = ax.twinx() if row_idx == 0 else ax
            if row_idx == 0:
                ax2.plot(d['ent_times'], d['ode_metrics'][mn], color=colors[region],
                         linewidth=1.2, label=region.upper(), alpha=0.8)
            else:
                ax.plot(d['ent_times'], d['ode_metrics'][mn], color=colors[region],
                        linewidth=1.2, label=region.upper(), alpha=0.8)

        if col_idx == 0:
            ax.set_ylabel(mn, fontsize=9)
        if row_idx == 4:
            ax.set_xlabel("Time (s)")
        if col_idx == 3 and row_idx == 0:
            ax2.legend(fontsize=8, loc='upper right')
        elif col_idx == 3 and row_idx == 1:
            ax.legend(fontsize=8, loc='upper right')

plt.tight_layout()
plt.savefig("figures/entropy_gru_ode_timeseries.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/entropy_gru_ode_timeseries.png")


# =============================================================================
# FIGURE 3: High vs Low bar chart
# =============================================================================
hl_df = stats_df[stats_df['analysis'] == 'high_vs_low'].copy()

fig, axes = plt.subplots(2, 4, figsize=(22, 8))
fig.suptitle("GRU-ODE Metrics: High vs Low Entropy (difference)", fontsize=13, fontweight='bold')

for idx, snum in enumerate(range(1, 9)):
    ax = axes[idx // 4, idx % 4]
    state, phase = session_meta[snum]
    sess_data = hl_df[hl_df['session'] == snum]

    metrics_plot = ['Flow Speed', 'Gate z(h)', 'Obs Jump', 'ODE-Obs\nAlign', 'Latent PR']
    metrics_keys = ['Flow Speed', 'Gate z(h)', 'Obs Jump', 'ODE-Obs Align', 'Latent PR']
    x_pos = np.arange(len(metrics_plot))
    w = 0.35
    offsets = {'LHA': -w/2, 'RSP': w/2}

    for region, color in [('LHA', '#e74c3c'), ('RSP', '#3498db')]:
        diffs, sigs = [], []
        for mk in metrics_keys:
            row = sess_data[(sess_data['metric'] == mk) & (sess_data['region'] == region)]
            if len(row) > 0:
                diffs.append(row.iloc[0]['high_mean'] - row.iloc[0]['low_mean'])
                sigs.append(row.iloc[0]['significant'])
            else:
                diffs.append(0)
                sigs.append(False)

        ax.bar(x_pos + offsets[region], diffs, w, label=region, color=color, alpha=0.7)
        for i, sf in enumerate(sigs):
            if sf:
                y = diffs[i]
                ax.text(x_pos[i] + offsets[region], y + 0.002 * np.sign(y),
                        '*', ha='center', fontsize=12, fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics_plot, fontsize=7)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.set_title(f"S{snum} ({state}/{phase[:3]})", fontsize=10)
    if idx == 0:
        ax.legend(fontsize=8)
    if idx % 4 == 0:
        ax.set_ylabel("High - Low")

plt.tight_layout()
plt.savefig("figures/entropy_gru_ode_high_vs_low.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/entropy_gru_ode_high_vs_low.png")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY: Significant GRU-ODE Correlations with Entropy (p<0.05)")
print("=" * 70)

sig_df = spearman_df[spearman_df['significant'] == True]
for _, row in sig_df.iterrows():
    print(f"  S{int(row['session'])} ({row['state']}/{row['phase'][:3]}) "
          f"{row['region']} {row['metric']}: rho={row['rho']:+.3f}, p={row['p']:.4f}")

print(f"\nTotal significant: {len(sig_df)} / {len(spearman_df)} tests")

# Count by metric
print("\nBy metric:")
for mn in metrics_order:
    n_sig = len(sig_df[sig_df['metric'] == mn])
    n_tot = len(spearman_df[spearman_df['metric'] == mn])
    print(f"  {mn:>15}: {n_sig}/{n_tot}")
