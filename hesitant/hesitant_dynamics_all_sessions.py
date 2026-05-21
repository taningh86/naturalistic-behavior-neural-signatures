"""
Time-resolved GRU-ODE latent dynamics — ALL sessions (1-8).
Hesitant (<800s) vs Short-committed vs Task-engaged.

For each session x region:
  - Extract trajectories for each group
  - Compute time-resolved flow speed, alignment, distance to FP,
    divergence, gate value, curvature at every time point
  - Resample to normalized time, plot profiles
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spikeinterface.extractors as se
import torch
import torch.nn as nn
from torchdiffeq import odeint
from sklearn.decomposition import PCA
from scipy import stats as sp_stats
import warnings, sys

warnings.filterwarnings('ignore')

BIN_SIZE_MS = 10
FS = 30000
D_SHARED = 32
HIDDEN_SIZE = 32
ODE_GATE_HIDDEN = 64
ODE_DT = 1.0
PRED_BINS = 10
LHA_DEPTH_MAX = 1300
RSP_DEPTH_MIN = 1300
TIME_CUTOFF = 800
N_RESAMPLE = 50

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("paths.yaml") as f:
    paths_config = yaml.safe_load(f)

SESSION_INFO = {
    1: {'state': 'Fed', 'phase': 'Exploration'},
    2: {'state': 'Fed', 'phase': 'Foraging'},
    3: {'state': 'Fed', 'phase': 'Exploration'},
    4: {'state': 'Fed', 'phase': 'Foraging'},
    5: {'state': 'Fasted', 'phase': 'Exploration'},
    6: {'state': 'Fasted', 'phase': 'Foraging'},
    7: {'state': 'Fasted', 'phase': 'Exploration'},
    8: {'state': 'Fasted', 'phase': 'Foraging'},
}


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
    """Load 10ms Poisson GRU-ODE model. Fed model for sessions 1-4, fasted for 5-8."""
    condition = "fed" if state == "Fed" else "fasted"
    model_path = Path("data") / f"gru_ode_10ms_poisson_{region}_{condition}_model.pt"
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


def find_fixed_point(ode_func, hidden_states, n_starts=500, max_steps=5000,
                     speed_thresh=1e-6):
    indices = np.random.choice(len(hidden_states),
                                size=min(n_starts, len(hidden_states)), replace=False)
    starts = hidden_states[indices]
    h = torch.tensor(starts, dtype=torch.float32, device=DEVICE).requires_grad_(True)
    optimizer = torch.optim.Adam([h], lr=0.01)
    for _ in range(max_steps):
        optimizer.zero_grad()
        with torch.enable_grad():
            dhdt = ode_func(0, h)
            loss = (dhdt ** 2).sum(dim=1).sum()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        dhdt_final = ode_func(0, h)
        final_speeds = (dhdt_final ** 2).sum(dim=1).cpu().numpy()
    h_np = h.detach().cpu().numpy()
    mask = final_speeds < speed_thresh
    if mask.sum() == 0:
        return hidden_states.mean(axis=0)
    return h_np[mask].mean(axis=0)


def compute_trajectory_dynamics(ode_func, traj, dom_fp):
    n = len(traj)
    dhdt = evaluate_flow(ode_func, traj)
    flow_speed = np.linalg.norm(dhdt, axis=1)

    traj_vel = np.diff(traj, axis=0)
    traj_speed = np.linalg.norm(traj_vel, axis=1)
    traj_speed = np.append(traj_speed, traj_speed[-1])

    alignment = np.zeros(n)
    for t in range(n - 1):
        fn = np.linalg.norm(dhdt[t])
        vn = np.linalg.norm(traj_vel[t])
        if fn > 1e-8 and vn > 1e-8:
            alignment[t] = np.dot(dhdt[t], traj_vel[t]) / (fn * vn)
    alignment[-1] = alignment[-2] if n > 1 else 0

    dist_to_fp = np.linalg.norm(traj - dom_fp, axis=1)
    ddist = np.diff(dist_to_fp)
    ddist = np.append(ddist, ddist[-1])

    curvature = np.zeros(n)
    for t in range(len(traj_vel) - 1):
        n1 = np.linalg.norm(traj_vel[t])
        n2 = np.linalg.norm(traj_vel[t + 1])
        if n1 > 1e-8 and n2 > 1e-8:
            cos_a = np.clip(np.dot(traj_vel[t], traj_vel[t+1]) / (n1 * n2), -1, 1)
            curvature[t + 1] = np.arccos(cos_a)

    div_indices = np.arange(0, n, 10)
    div_vals = compute_local_divergence(ode_func, traj[div_indices])
    divergence = np.interp(np.arange(n), div_indices, div_vals)

    h_tensor = torch.tensor(traj, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        z_gate = ode_func.update_gate(h_tensor).cpu().numpy()
    mean_gate = z_gate.mean(axis=1)

    return {
        'flow_speed': flow_speed,
        'traj_speed': traj_speed,
        'alignment': alignment,
        'dist_to_fp': dist_to_fp,
        'ddist_fp': ddist,
        'curvature': curvature,
        'divergence': divergence,
        'gate_mean': mean_gate,
    }


def resample(profile, n_out=N_RESAMPLE):
    n_in = len(profile)
    if n_in < 2:
        return np.full(n_out, profile[0] if n_in > 0 else np.nan)
    return np.interp(np.linspace(0, 1, n_out), np.linspace(0, 1, n_in), profile)


def get_good_units(sorted_path_obj):
    ci = sorted_path_obj / "cluster_info.tsv"
    if not ci.exists():
        return np.array([]), np.array([])
    df = pd.read_csv(ci, sep='\t')
    if 'depth' not in df.columns:
        return np.array([]), np.array([])
    lc = None
    if 'group' in df.columns and df['group'].eq('good').any():
        lc = 'group'
    elif 'KSLabel' in df.columns:
        lc = 'KSLabel'
    if lc is None:
        return np.array([]), np.array([])
    good = df[df[lc] == 'good']
    return (good[good['depth'] < LHA_DEPTH_MAX]['cluster_id'].values,
            good[good['depth'] >= RSP_DEPTH_MIN]['cluster_id'].values)


def get_allmin(sorting, unit_ids):
    amin = np.inf
    for u in unit_ids:
        st = sorting.get_unit_spike_train(u)
        if len(st) > 0:
            amin = min(amin, np.min(st))
    return amin


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("  Time-Resolved GRU-ODE Dynamics — All Sessions")
    print(f"  Hesitant cutoff: <{TIME_CUTOFF}s")
    print("=" * 70)
    sys.stdout.flush()

    exc_df = pd.read_csv("data/excursion_features_all_sessions.csv")
    sp = paths_config['single_probe']['coordinates_1']['mouse01']['sessions']

    metrics = ['flow_speed', 'traj_speed', 'alignment', 'dist_to_fp',
               'ddist_fp', 'curvature', 'divergence', 'gate_mean']

    group_names = ["Hesitant", "Short-committed", "Task-engaged"]
    colors = {"Hesitant": "#E53935", "Short-committed": "#1E88E5", "Task-engaged": "#43A047"}

    # Collect results across sessions for summary
    all_session_stats = []

    for region in ["LHA", "RSP"]:
        print(f"\n{'#'*70}")
        print(f"  REGION: {region}")
        print(f"{'#'*70}")

        # One big figure: 8 sessions x 8 metrics
        fig, axes = plt.subplots(8, 8, figsize=(32, 32))
        fig.suptitle(f'Time-Resolved GRU-ODE Dynamics — {region}\n'
                     f'Rows=Sessions 1-8, Cols=Metrics\n'
                     f'Hesitant (<{TIME_CUTOFF}s) vs Short-committed vs Task-engaged',
                     fontsize=16, fontweight='bold', y=0.995)

        metric_titles = ['Flow speed\n|dh/dt|', 'Traj speed\n|dh|',
                         'Alignment\ncos(flow,vel)', 'Dist to FP',
                         'dDist/dt FP', 'Curvature\n(rad)',
                         'Divergence\ntr(J)', 'Gate value\n(z)']

        # Cache models per state
        models_cache = {}

        for snum in range(1, 9):
            info = SESSION_INFO[snum]
            state = info['state']
            phase = info['phase']

            print(f"\n  Session {snum} ({state}, {phase})")
            sys.stdout.flush()

            # Load hidden states
            h_file = f"data/gru_ode_10ms_hidden_{region.lower()}_s{snum}.npy"
            if not Path(h_file).exists():
                print(f"    No hidden states file, skipping")
                continue
            h_states = np.load(h_file)

            # Load model (cached per state x region)
            model_key = f"{region}_{state}"
            if model_key not in models_cache:
                models_cache[model_key] = load_model(region.lower(), state)
            model = models_cache[model_key]
            ode_func = model.ode_func

            # Load sorting for time alignment
            sc = sp[f'session_{snum}']
            sorted_path = Path(sc['sorted'])
            if not sorted_path.exists():
                print(f"    Sorted path not found, skipping")
                continue
            sorting = se.read_kilosort(sorted_path)
            lha_ids, rsp_ids = get_good_units(sorted_path)
            unit_ids = lha_ids if region == "LHA" else rsp_ids
            if len(unit_ids) == 0:
                print(f"    No {region} units, skipping")
                continue
            allmin = get_allmin(sorting, unit_ids)
            offset = allmin / FS

            # Define groups for this session
            s_df = exc_df[exc_df["session"] == snum].copy()
            not_pot = s_df["farthest_zone"] != "Pot"
            all_hes = s_df[(s_df["feeding_bins"] == 0) & (s_df["digging_bins"] == 0) &
                           not_pot & (s_df["reversals"] >= 1) & (s_df["duration"] >= 2.0)]
            hesitant = all_hes[all_hes["start_time"] < TIME_CUTOFF]
            task = s_df[(s_df["feeding_bins"] > 0) | (s_df["digging_bins"] > 0)]
            non_hes_non_task = s_df[~s_df.index.isin(all_hes.index) & ~s_df.index.isin(task.index)]

            if len(hesitant) > 0:
                hes_dur_min = hesitant["duration"].min()
                hes_dur_max = hesitant["duration"].max()
                committed = non_hes_non_task[
                    (non_hes_non_task["reached_arena"] == True) &
                    (non_hes_non_task["duration"] >= hes_dur_min) &
                    (non_hes_non_task["duration"] <= hes_dur_max)
                ]
            else:
                committed = pd.DataFrame()

            groups = {"Hesitant": hesitant, "Short-committed": committed, "Task-engaged": task}

            for gn, gdf in groups.items():
                n = len(gdf)
                t_range = f"t={gdf['start_time'].min():.0f}-{gdf['start_time'].max():.0f}s" if n > 0 else ""
                print(f"    {gn}: {n} bouts {t_range}")

            if len(hesitant) == 0:
                print(f"    No early hesitant bouts, skipping session")
                # Empty row
                for ci in range(8):
                    ax = axes[snum - 1, ci]
                    ax.text(0.5, 0.5, 'No hesitant\nbouts', ha='center', va='center',
                            transform=ax.transAxes, fontsize=9)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if ci == 0:
                        ax.set_ylabel(f'S{snum}\n({state[:3]},{phase[:3]})',
                                      fontsize=9, rotation=0, labelpad=40, va='center')
                continue

            # Find fixed point
            np.random.seed(42)
            dom_fp = find_fixed_point(ode_func, h_states)

            # Compute dynamics for each group
            group_profiles = {gn: {m: [] for m in metrics} for gn in group_names}

            for gn, gdf in groups.items():
                for _, row in gdf.iterrows():
                    gs = max(0, int((row["start_time"] - offset) / 0.01))
                    ge = min(h_states.shape[0], int((row["end_time"] - offset) / 0.01))
                    if ge - gs < 30:
                        continue
                    traj = h_states[gs:ge, :]
                    dyn = compute_trajectory_dynamics(ode_func, traj, dom_fp)
                    for m in metrics:
                        group_profiles[gn][m].append(resample(dyn[m]))

            # Convert to arrays
            for gn in group_names:
                for m in metrics:
                    arr = group_profiles[gn][m]
                    group_profiles[gn][m] = np.array(arr) if len(arr) > 0 else np.array([]).reshape(0, N_RESAMPLE)

            # Collect session-level stats
            for m in metrics:
                h_prof = group_profiles["Hesitant"][m]
                c_prof = group_profiles["Short-committed"][m]
                if len(h_prof) >= 2 and len(c_prof) >= 2:
                    for ti, tl in [(12, '25%'), (25, '50%'), (37, '75%')]:
                        hv = h_prof[:, ti]
                        cv = c_prof[:, ti]
                        stat, pval = sp_stats.mannwhitneyu(hv, cv, alternative='two-sided')
                        all_session_stats.append({
                            'region': region, 'session': snum, 'state': state,
                            'phase': phase, 'metric': m, 'time_pct': tl,
                            'hes_median': np.median(hv), 'com_median': np.median(cv),
                            'n_hes': len(h_prof), 'n_com': len(c_prof),
                            'U': stat, 'p': pval,
                        })

            # Plot row for this session
            norm_time = np.linspace(0, 100, N_RESAMPLE)
            for ci, m in enumerate(metrics):
                ax = axes[snum - 1, ci]
                for gn in group_names:
                    profiles = group_profiles[gn][m]
                    if len(profiles) == 0:
                        continue
                    mean_p = np.mean(profiles, axis=0)
                    sem_p = np.std(profiles, axis=0) / np.sqrt(len(profiles))
                    ax.plot(norm_time, mean_p, color=colors[gn], linewidth=1.5,
                            label=f'{gn} ({len(profiles)})')
                    ax.fill_between(norm_time, mean_p - sem_p, mean_p + sem_p,
                                    color=colors[gn], alpha=0.15)

                if m in ['alignment', 'ddist_fp', 'divergence']:
                    ax.axhline(0, color='gray', linewidth=0.3, linestyle='--')

                ax.tick_params(labelsize=7)
                if snum == 1:
                    ax.set_title(metric_titles[ci], fontsize=9)
                if ci == 0:
                    ax.set_ylabel(f'S{snum}\n({state[:3]},{phase[:3]})',
                                  fontsize=9, rotation=0, labelpad=40, va='center')
                if snum == 8:
                    ax.set_xlabel('Bout %', fontsize=8)
                if snum == 1 and ci == 0:
                    ax.legend(fontsize=6, loc='best')

            sys.stdout.flush()

        plt.tight_layout(rect=[0.03, 0, 1, 0.97])
        plt.savefig(f'figures/hesitant_dynamics_allsessions_{region.lower()}.png',
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Saved: figures/hesitant_dynamics_allsessions_{region.lower()}.png")

    # Save stats
    stats_df = pd.DataFrame(all_session_stats)
    stats_df.to_csv('data/hesitant_dynamics_allsessions_stats.csv', index=False)
    print(f"\nSaved: data/hesitant_dynamics_allsessions_stats.csv")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  Summary: Significant results (p<0.05) across sessions")
    print(f"{'='*70}")
    if len(stats_df) > 0:
        sig = stats_df[stats_df['p'] < 0.05].sort_values(['region', 'metric', 'session'])
        for _, row in sig.iterrows():
            print(f"  {row['region']} S{row['session']} ({row['state'][:3]},{row['phase'][:3]}) "
                  f"{row['metric']:20s} t={row['time_pct']}: "
                  f"Hes={row['hes_median']:.4f} vs Com={row['com_median']:.4f}, "
                  f"p={row['p']:.4f} (n={row['n_hes']}vs{row['n_com']})")

    print("\nDone!")


if __name__ == "__main__":
    main()
