"""
GRU-ODE Manifold Topology
=========================
Uses UMAP (nonlinear embedding) to reveal the geometric shape of the
neural manifold learned by GRU-ODE -- rings, tori, clusters, etc.

For each region (LHA, RSP) and condition (Fed, Fasted):
  - UMAP 3D embedding of hidden states
  - Color by time, session, and flow speed
  - Multiple viewing angles to reveal topology
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import umap
import spikeinterface.extractors as se
import torch
import torch.nn as nn
from torchdiffeq import odeint
from sklearn.decomposition import PCA
import warnings
import time

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

BIN_SIZE_MS = 10
PRED_WINDOW_MS = 100
PRED_BINS = PRED_WINDOW_MS // BIN_SIZE_MS
FS = 30000
BIN_SAMPLES = int(BIN_SIZE_MS * FS / 1000)

SEQ_LEN = 50
STRIDE = 50
D_SHARED = 32
HIDDEN_SIZE = 32
ODE_GATE_HIDDEN = 64

ODE_SOLVER = 'rk4'
ODE_STEP_SIZE = 1.0
ODE_DT = 1.0

LHA_DEPTH_MAX = 1300
RSP_DEPTH_MIN = 1300

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
# MODEL (same as other scripts)
# =============================================================================

class GRUODEFuncTau(nn.Module):
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
        self.log_tau = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, t, h):
        z = self.update_gate(h)
        n = self.candidate(h)
        tau = torch.exp(self.log_tau)
        return (1.0 / tau) * (1 - z) * (n - h)


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
        self.ode_func = GRUODEFuncTau(hidden_size, gate_hidden)
        self.obs_cell = nn.GRUCell(input_size=d_shared, hidden_size=hidden_size)
        self.fc_shared = nn.Linear(hidden_size, d_shared)
        self.output_projections = nn.ModuleDict()
        for sn, n_neurons in session_neuron_counts.items():
            self.output_projections[str(sn)] = nn.Linear(d_shared, n_neurons)
        self.register_buffer('t_span', torch.tensor([0.0, ODE_DT]))

    def _ode_evolve(self, h):
        return odeint(self.ode_func, h, self.t_span,
                      method=ODE_SOLVER, options={'step_size': ODE_STEP_SIZE})[-1]

    def extract_hidden_states(self, x, session_num):
        sn_key = str(session_num)
        batch_size = x.shape[0]
        with torch.no_grad():
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
            hidden_seq = []
            for k in range(x.shape[1]):
                h = self._ode_evolve(h)
                x_proj = self.input_projections[sn_key](x[:, k, :])
                h = self.obs_cell(x_proj, h)
                hidden_seq.append(h.unsqueeze(1))
            return torch.cat(hidden_seq, dim=1)


# =============================================================================
# DATA LOADING
# =============================================================================

def get_good_units_by_region(sorted_path_obj):
    ci = sorted_path_obj / "cluster_info.tsv"
    if not ci.exists():
        return np.array([]), np.array([])
    df = pd.read_csv(ci, sep='\t')
    if 'depth' not in df.columns:
        return np.array([]), np.array([])
    label_col = None
    if 'group' in df.columns and df['group'].eq('good').any():
        label_col = 'group'
    elif 'KSLabel' in df.columns:
        label_col = 'KSLabel'
    if label_col is None:
        return np.array([]), np.array([])
    good = df[df[label_col] == 'good']
    lha_ids = good[good['depth'] < LHA_DEPTH_MAX]['cluster_id'].values
    rsp_ids = good[good['depth'] >= RSP_DEPTH_MIN]['cluster_id'].values
    return lha_ids, rsp_ids


def bin_spike_trains(sorting, unit_ids):
    spike_trains = {}
    all_min, all_max = np.inf, 0
    for uid in unit_ids:
        st = sorting.get_unit_spike_train(uid)
        spike_trains[uid] = st
        if len(st) > 0:
            all_min = min(all_min, np.min(st))
            all_max = max(all_max, np.max(st))
    n_bins = int((all_max - all_min) / BIN_SAMPLES) + 1
    data = np.zeros((n_bins, len(unit_ids)), dtype=np.float32)
    for i, uid in enumerate(unit_ids):
        st = spike_trains[uid]
        if len(st) > 0:
            b = ((st - all_min) // BIN_SAMPLES).astype(int)
            b = b[(b >= 0) & (b < n_bins)]
            np.add.at(data[:, i], b, 1)
    means = data.mean(axis=0, keepdims=True)
    stds = data.std(axis=0, keepdims=True)
    stds[stds < 1e-8] = 1.0
    zscore_data = (data - means) / stds
    return zscore_data, n_bins


def load_sessions(session_nums):
    sessions_data = {}
    sp = paths_config['single_probe']['coordinates_1']['mouse01']['sessions']
    for sess_num in session_nums:
        info = SESSION_INFO[sess_num]
        key = f"session_{sess_num}"
        sc = sp[key]
        sorted_path = Path(sc['sorted'])
        if not sorted_path.exists():
            continue
        sorting = se.read_kilosort(sorted_path)
        lha_ids, rsp_ids = get_good_units_by_region(sorted_path)
        if len(lha_ids) < 3 or len(rsp_ids) < 3:
            continue
        lha_zscore, lha_bins = bin_spike_trains(sorting, lha_ids)
        rsp_zscore, rsp_bins = bin_spike_trains(sorting, rsp_ids)
        sessions_data[sess_num] = {
            'lha': {'zscore': lha_zscore, 'n_neurons': len(lha_ids), 'n_bins': lha_bins},
            'rsp': {'zscore': rsp_zscore, 'n_neurons': len(rsp_ids), 'n_bins': rsp_bins},
            'state': info['state'], 'phase': info['phase'],
        }
        print(f"  S{sess_num} ({info['phase']}): LHA={len(lha_ids)}, RSP={len(rsp_ids)}")
    return sessions_data


def extract_hidden_with_metadata(model, sessions_data, session_nums, region, device):
    """Extract hidden states AND keep track of session/time metadata."""
    all_hidden = []
    all_sessions = []
    all_times = []      # fractional time within session (0 to 1)
    all_phases = []
    all_speeds = []     # ODE flow speed at each state

    model.eval()
    for sn in session_nums:
        zscore = sessions_data[sn][region]['zscore']
        T = len(zscore)
        seqs = []
        time_fracs = []
        for i in range(0, T - SEQ_LEN - PRED_BINS, STRIDE):
            seqs.append(zscore[i:i + SEQ_LEN])
            time_fracs.append((i + SEQ_LEN) / T)  # fractional time

        seqs_t = torch.tensor(np.array(seqs), dtype=torch.float32).to(device)

        chunk_size = 128
        for start in range(0, len(seqs_t), chunk_size):
            chunk = seqs_t[start:start + chunk_size]
            h = model.extract_hidden_states(chunk, sn)
            h_last = h[:, -1, :]  # (batch, hidden_size)

            # Compute flow speed
            with torch.no_grad():
                dhdt = model.ode_func(0, h_last)
                speed = torch.sqrt((dhdt**2).sum(dim=1)).cpu().numpy()

            all_hidden.append(h_last.cpu().numpy())
            all_speeds.append(speed)

            n_batch = len(chunk)
            all_sessions.extend([sn] * n_batch)
            all_times.extend(time_fracs[start:start + n_batch])
            all_phases.extend([sessions_data[sn]['phase']] * n_batch)

    hidden = np.concatenate(all_hidden, axis=0)
    speeds = np.concatenate(all_speeds, axis=0)
    sessions_arr = np.array(all_sessions)
    times_arr = np.array(all_times)
    phases_arr = np.array(all_phases)

    return hidden, sessions_arr, times_arr, phases_arr, speeds


def load_model(region, state, device):
    model_path = Path("data") / f"gru_ode_10ms_tau_{region}_{state.lower()}_model.pt"
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    neuron_counts = checkpoint['neuron_counts']
    model = PooledGRUODE(neuron_counts, D_SHARED, HIDDEN_SIZE, ODE_GATE_HIDDEN,
                         PRED_BINS).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"Device: {DEVICE}")
    print("GRU-ODE Manifold Topology via UMAP\n")

    # Load sessions
    print("Loading sessions...")
    fed_data = load_sessions([1, 2, 3, 4])
    fasted_data = load_sessions([5, 6, 7, 8])
    print()

    for region in ['lha', 'rsp']:
        for state, sess_nums, sess_data in [
            ('Fed', [1,2,3,4], fed_data),
            ('Fasted', [5,6,7,8], fasted_data),
        ]:
            print(f"\n{'='*60}")
            print(f"  {region.upper()} {state} -- UMAP Topology")
            print(f"{'='*60}")

            model = load_model(region, state, DEVICE)

            # Extract hidden states with metadata
            print("  Extracting hidden states...")
            t0 = time.time()
            hidden, sessions, times, phases, speeds = extract_hidden_with_metadata(
                model, sess_data, sess_nums, region, DEVICE)
            print(f"  Got {len(hidden)} states ({time.time()-t0:.1f}s)")

            # Subsample for UMAP (it can be slow on >10k points)
            n_umap = min(8000, len(hidden))
            idx = np.random.choice(len(hidden), n_umap, replace=False)
            hidden_sub = hidden[idx]
            sessions_sub = sessions[idx]
            times_sub = times[idx]
            phases_sub = phases[idx]
            speeds_sub = speeds[idx]

            # UMAP 3D
            print("  Running UMAP 3D...")
            t0 = time.time()
            reducer_3d = umap.UMAP(n_components=3, n_neighbors=30,
                                    min_dist=0.1, metric='euclidean',
                                    random_state=42)
            emb_3d = reducer_3d.fit_transform(hidden_sub)
            print(f"  UMAP 3D done ({time.time()-t0:.1f}s)")

            # UMAP 2D
            print("  Running UMAP 2D...")
            t0 = time.time()
            reducer_2d = umap.UMAP(n_components=2, n_neighbors=30,
                                    min_dist=0.1, metric='euclidean',
                                    random_state=42)
            emb_2d = reducer_2d.fit_transform(hidden_sub)
            print(f"  UMAP 2D done ({time.time()-t0:.1f}s)")

            # =============================================================
            # Figure: 2x3 layout
            # Row 1: 2D UMAP colored by (session, time, flow speed)
            # Row 2: 3D UMAP from 3 viewing angles colored by time
            # =============================================================
            fig = plt.figure(figsize=(18, 11))
            fig.suptitle(f"{region.upper()} {state} -- Manifold Topology (UMAP)\n"
                         f"{n_umap} hidden states, n_neighbors=30, min_dist=0.1",
                         fontsize=14, fontweight='bold')

            # --- Row 1: 2D UMAP ---

            # Panel 1: Colored by session
            ax = fig.add_subplot(2, 3, 1)
            unique_sessions = np.unique(sessions_sub)
            session_cmap = plt.cm.Set1
            for i, sn in enumerate(unique_sessions):
                mask = sessions_sub == sn
                phase = SESSION_INFO[sn]['phase']
                ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                           c=[session_cmap(i / max(len(unique_sessions)-1, 1))],
                           s=3, alpha=0.4, label=f'S{sn} ({phase[:3]})',
                           rasterized=True)
            ax.legend(fontsize=8, markerscale=3, loc='best')
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.set_title('Colored by Session', fontsize=11)

            # Panel 2: Colored by time within session
            ax = fig.add_subplot(2, 3, 2)
            sc = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=times_sub,
                            cmap='viridis', s=3, alpha=0.4, rasterized=True)
            plt.colorbar(sc, ax=ax, label='Time (fraction)', shrink=0.8)
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.set_title('Colored by Time', fontsize=11)

            # Panel 3: Colored by flow speed
            ax = fig.add_subplot(2, 3, 3)
            speed_pct = np.percentile(speeds_sub, [2, 98])
            sc = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=speeds_sub,
                            cmap='hot_r', s=3, alpha=0.4, rasterized=True,
                            vmin=speed_pct[0], vmax=speed_pct[1])
            plt.colorbar(sc, ax=ax, label='Flow speed ||dh/dt||', shrink=0.8)
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.set_title('Colored by Flow Speed', fontsize=11)

            # --- Row 2: 3D UMAP from 3 viewing angles ---
            viewing_angles = [
                (30, 45, 'Front-left'),
                (30, 135, 'Front-right'),
                (80, 45, 'Top-down'),
            ]

            for vi, (elev, azim, view_name) in enumerate(viewing_angles):
                ax = fig.add_subplot(2, 3, 4 + vi, projection='3d')
                sc = ax.scatter(emb_3d[:, 0], emb_3d[:, 1], emb_3d[:, 2],
                                c=times_sub, cmap='viridis', s=2, alpha=0.3,
                                rasterized=True)
                ax.view_init(elev=elev, azim=azim)
                ax.set_xlabel('U1', fontsize=9)
                ax.set_ylabel('U2', fontsize=9)
                ax.set_zlabel('U3', fontsize=9)
                ax.set_title(f'3D: {view_name}', fontsize=10)
                ax.tick_params(labelsize=7)

            plt.tight_layout(rect=[0, 0, 1, 0.93])
            outpath = Path("figures") / f"gru_ode_manifold_topology_{region}_{state.lower()}.png"
            fig.savefig(outpath, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {outpath}")

    # =========================================================================
    # Combined figure: all 4 conditions side by side (2D UMAP colored by time)
    # =========================================================================
    print("\nGenerating combined overview figure...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Neural Manifold Topology -- All Conditions\n'
                 'UMAP 2D, colored by time within session',
                 fontsize=14, fontweight='bold')

    conditions = [
        ('lha', 'Fed', [1,2,3,4], fed_data, axes[0,0]),
        ('lha', 'Fasted', [5,6,7,8], fasted_data, axes[0,1]),
        ('rsp', 'Fed', [1,2,3,4], fed_data, axes[1,0]),
        ('rsp', 'Fasted', [5,6,7,8], fasted_data, axes[1,1]),
    ]

    for region, state, sess_nums, sess_data, ax in conditions:
        model = load_model(region, state, DEVICE)
        hidden, sessions, times, phases, speeds = extract_hidden_with_metadata(
            model, sess_data, sess_nums, region, DEVICE)

        n_sub = min(6000, len(hidden))
        idx = np.random.choice(len(hidden), n_sub, replace=False)

        reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                            random_state=42)
        emb = reducer.fit_transform(hidden[idx])

        sc = ax.scatter(emb[:, 0], emb[:, 1], c=times[idx], cmap='viridis',
                        s=3, alpha=0.4, rasterized=True)
        plt.colorbar(sc, ax=ax, label='Time', shrink=0.8)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title(f'{region.upper()} {state}', fontsize=12, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig("figures/gru_ode_manifold_topology_overview.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: figures/gru_ode_manifold_topology_overview.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
