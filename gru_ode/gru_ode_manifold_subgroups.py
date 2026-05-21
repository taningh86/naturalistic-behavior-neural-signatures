"""
GRU-ODE Manifold Topology -- Neuron Subgroup Analysis
=====================================================
The full-population UMAP shows no clean geometric structure, but
subpopulations might. This script:

1. Clusters neurons into functional subgroups using the model's
   output projection weights (neurons driven by similar hidden dims)
2. For each subgroup, identifies the dominant hidden dimensions
3. Runs UMAP on the subgroup-specific hidden subspace
4. Colors the full manifold by each subgroup's aggregate activity
   to reveal hidden geometric patterns (rings, tori, etc.)
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import umap
import spikeinterface.extractors as se
import torch
import torch.nn as nn
from torchdiffeq import odeint
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
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

N_SUBGROUPS = 4  # Number of neuron subgroups to find

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


def extract_hidden(model, sessions_data, session_nums, region, device):
    """Extract hidden states."""
    all_hidden = []
    model.eval()
    for sn in session_nums:
        zscore = sessions_data[sn][region]['zscore']
        T = len(zscore)
        seqs = []
        for i in range(0, T - SEQ_LEN - PRED_BINS, STRIDE):
            seqs.append(zscore[i:i + SEQ_LEN])

        seqs_t = torch.tensor(np.array(seqs), dtype=torch.float32).to(device)
        chunk_size = 128
        for start in range(0, len(seqs_t), chunk_size):
            chunk = seqs_t[start:start + chunk_size]
            h = model.extract_hidden_states(chunk, sn)
            all_hidden.append(h[:, -1, :].cpu().numpy())

    return np.concatenate(all_hidden, axis=0)


def load_model(region, state, device):
    model_path = Path("data") / f"gru_ode_10ms_tau_{region}_{state.lower()}_model.pt"
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    neuron_counts = checkpoint['neuron_counts']
    model = PooledGRUODE(neuron_counts, D_SHARED, HIDDEN_SIZE, ODE_GATE_HIDDEN,
                         PRED_BINS).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, neuron_counts


# =============================================================================
# SUBGROUP ANALYSIS
# =============================================================================

def get_output_weight_matrix(model, session_nums):
    """Get the combined output weight path: hidden -> shared -> neurons.
    Average across sessions to get a representative weight matrix."""
    fc_w = model.fc_shared.weight.detach().cpu().numpy()  # (d_shared, hidden)

    all_out_w = []
    for sn in session_nums:
        out_w = model.output_projections[str(sn)].weight.detach().cpu().numpy()  # (n_neurons, d_shared)
        # Combined: (n_neurons, hidden) = out_w @ fc_w
        combined = out_w @ fc_w  # (n_neurons, hidden_size)
        all_out_w.append(combined)

    return all_out_w  # list of (n_neurons_i, hidden_size) per session


def cluster_hidden_dims(model):
    """Cluster the 32 hidden dimensions by their co-activation patterns
    using the ODE function weights."""
    # Use the candidate and gate network weights to characterize each dim
    # Candidate: hidden -> gate_hidden -> hidden
    cand_w1 = model.ode_func.candidate[0].weight.detach().cpu().numpy()  # (64, 32)
    cand_w2 = model.ode_func.candidate[2].weight.detach().cpu().numpy()  # (32, 64)
    gate_w1 = model.ode_func.update_gate[0].weight.detach().cpu().numpy()
    gate_w2 = model.ode_func.update_gate[2].weight.detach().cpu().numpy()

    # Feature vector for each hidden dim: how it affects and is affected by others
    # Combine input and output roles
    dim_features = np.concatenate([
        cand_w1.T,   # (32, 64) - how each dim feeds into candidate
        cand_w2,     # (32, 64) - how candidate output maps to each dim
        gate_w1.T,   # (32, 64)
        gate_w2,     # (32, 64)
    ], axis=1)  # (32, 256)

    # Cluster hidden dims
    clustering = AgglomerativeClustering(n_clusters=N_SUBGROUPS, linkage='ward')
    dim_labels = clustering.fit_predict(dim_features)

    return dim_labels, dim_features


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"Device: {DEVICE}")
    print(f"Manifold Subgroup Analysis -- {N_SUBGROUPS} subgroups\n")

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
            print(f"  {region.upper()} {state} -- Subgroup Topology")
            print(f"{'='*60}")

            model, neuron_counts = load_model(region, state, DEVICE)

            # Cluster hidden dimensions into functional subgroups
            dim_labels, dim_features = cluster_hidden_dims(model)
            subgroups = {}
            for g in range(N_SUBGROUPS):
                dims = np.where(dim_labels == g)[0]
                subgroups[g] = dims
                tau_vals = torch.exp(model.ode_func.log_tau).detach().cpu().numpy()
                print(f"  Subgroup {g+1}: dims {dims} "
                      f"(n={len(dims)}, mean tau={tau_vals[dims].mean():.3f})")

            # Extract hidden states
            print("  Extracting hidden states...")
            t0 = time.time()
            hidden = extract_hidden(
                model, sess_data, sess_nums, region, DEVICE)
            print(f"  Got {len(hidden)} states ({time.time()-t0:.1f}s)")

            # Subsample for UMAP
            n_sub = min(6000, len(hidden))
            idx = np.random.choice(len(hidden), n_sub, replace=False)
            hidden_sub = hidden[idx]

            # ================================================================
            # Figure: (N_SUBGROUPS+1) x 2 layout
            # Col 1: UMAP of subgroup-specific hidden dims
            # Col 2: Full UMAP colored by subgroup aggregate activity
            # Row 0: Full population (all 32 dims)
            # Rows 1-N: Each subgroup
            # ================================================================

            n_rows = N_SUBGROUPS + 1
            fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows))
            fig.suptitle(f"{region.upper()} {state} -- Subgroup Manifold Topology\n"
                         f"Hidden dims clustered by ODE weight structure "
                         f"into {N_SUBGROUPS} subgroups",
                         fontsize=13, fontweight='bold')

            # Full-population UMAP (reference)
            print("  UMAP: full population...")
            reducer_full = umap.UMAP(n_components=2, n_neighbors=30,
                                      min_dist=0.1, random_state=42)
            emb_full = reducer_full.fit_transform(hidden_sub)

            # Row 0, Col 0: Full UMAP colored by mean activity
            ax = axes[0, 0]
            mean_act = hidden_sub.mean(axis=1)
            sc = ax.scatter(emb_full[:, 0], emb_full[:, 1], c=mean_act,
                            cmap='viridis', s=3, alpha=0.4, rasterized=True)
            plt.colorbar(sc, ax=ax, label='Mean hidden', shrink=0.8)
            ax.set_title('All 32 dims -- colored by mean h', fontsize=10)
            ax.set_xlabel('UMAP 1', fontsize=9)
            ax.set_ylabel('UMAP 2', fontsize=9)

            # Row 0, Col 1: Full UMAP colored by flow speed
            ax = axes[0, 1]
            with torch.no_grad():
                h_t = torch.tensor(hidden_sub, dtype=torch.float32, device=DEVICE)
                dhdt = model.ode_func(0, h_t)
                speeds = torch.sqrt((dhdt**2).sum(dim=1)).cpu().numpy()
            sp_pct = np.percentile(speeds, [2, 98])
            sc = ax.scatter(emb_full[:, 0], emb_full[:, 1], c=speeds,
                            cmap='hot_r', s=3, alpha=0.4, rasterized=True,
                            vmin=sp_pct[0], vmax=sp_pct[1])
            plt.colorbar(sc, ax=ax, label='Flow speed', shrink=0.8)
            ax.set_title('All 32 dims -- colored by flow speed', fontsize=10)
            ax.set_xlabel('UMAP 1', fontsize=9)
            ax.set_ylabel('UMAP 2', fontsize=9)

            # Each subgroup
            subgroup_colors = ['#E65100', '#1565C0', '#2E7D32', '#7B1FA2',
                               '#C62828', '#00838F']

            for g in range(N_SUBGROUPS):
                dims = subgroups[g]
                row = g + 1

                # Col 0: UMAP on subgroup dims only
                ax = axes[row, 0]
                sub_hidden = hidden_sub[:, dims]

                if len(dims) >= 2:
                    print(f"  UMAP: subgroup {g+1} ({len(dims)} dims)...")
                    reducer_sub = umap.UMAP(n_components=2, n_neighbors=30,
                                             min_dist=0.05, random_state=42)
                    emb_sub = reducer_sub.fit_transform(sub_hidden)

                    # Color by aggregate activity of these dims
                    sub_act = sub_hidden.mean(axis=1)
                    sc = ax.scatter(emb_sub[:, 0], emb_sub[:, 1], c=sub_act,
                                    cmap='viridis', s=3, alpha=0.5, rasterized=True)
                    plt.colorbar(sc, ax=ax, label='Subgroup activity', shrink=0.8)
                else:
                    ax.text(0.5, 0.5, f'Only {len(dims)} dim\n(skip UMAP)',
                            transform=ax.transAxes, ha='center', fontsize=12)

                ax.set_title(f'Subgroup {g+1}: dims {list(dims)}\n'
                             f'({len(dims)} dims, tau={tau_vals[dims].mean():.3f})',
                             fontsize=9, color=subgroup_colors[g % len(subgroup_colors)])
                ax.set_xlabel('UMAP 1', fontsize=9)
                ax.set_ylabel('UMAP 2', fontsize=9)

                # Col 1: Full UMAP colored by this subgroup's activity
                ax = axes[row, 1]
                sub_act_full = hidden_sub[:, dims].mean(axis=1)
                sc = ax.scatter(emb_full[:, 0], emb_full[:, 1], c=sub_act_full,
                                cmap='coolwarm', s=3, alpha=0.4, rasterized=True)
                plt.colorbar(sc, ax=ax, label=f'Subgroup {g+1} activity', shrink=0.8)
                ax.set_title(f'Full UMAP colored by subgroup {g+1}', fontsize=10)
                ax.set_xlabel('UMAP 1', fontsize=9)
                ax.set_ylabel('UMAP 2', fontsize=9)

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            outpath = Path("figures") / f"gru_ode_manifold_subgroups_{region}_{state.lower()}.png"
            fig.savefig(outpath, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {outpath}")

            # =================================================================
            # Also: pairwise hidden dim UMAP -- try all pairs of subgroups
            # to see if any 2-subgroup combo reveals geometry
            # =================================================================
            if N_SUBGROUPS <= 4:
                n_pairs = N_SUBGROUPS * (N_SUBGROUPS - 1) // 2
                fig2, axes2 = plt.subplots(2, max(n_pairs, 1), figsize=(6*max(n_pairs,1), 10))
                if n_pairs == 1:
                    axes2 = axes2.reshape(2, 1)
                fig2.suptitle(f"{region.upper()} {state} -- Pairwise Subgroup Manifolds",
                              fontsize=13, fontweight='bold')

                pair_idx = 0
                for g1 in range(N_SUBGROUPS):
                    for g2 in range(g1+1, N_SUBGROUPS):
                        if pair_idx >= axes2.shape[1]:
                            break
                        dims_pair = np.concatenate([subgroups[g1], subgroups[g2]])

                        print(f"  UMAP: subgroups {g1+1}+{g2+1} ({len(dims_pair)} dims)...")
                        pair_hidden = hidden_sub[:, dims_pair]
                        reducer_pair = umap.UMAP(n_components=2, n_neighbors=30,
                                                  min_dist=0.05, random_state=42)
                        emb_pair = reducer_pair.fit_transform(pair_hidden)

                        # Top: colored by subgroup g1 activity
                        ax = axes2[0, pair_idx]
                        act_g1 = hidden_sub[:, subgroups[g1]].mean(axis=1)
                        sc = ax.scatter(emb_pair[:, 0], emb_pair[:, 1], c=act_g1,
                                        cmap='coolwarm', s=3, alpha=0.4, rasterized=True)
                        plt.colorbar(sc, ax=ax, shrink=0.8)
                        ax.set_title(f'SG{g1+1}+SG{g2+1}, color=SG{g1+1}', fontsize=9)
                        ax.set_xlabel('UMAP 1', fontsize=8)

                        # Bottom: colored by subgroup g2 activity
                        ax = axes2[1, pair_idx]
                        act_g2 = hidden_sub[:, subgroups[g2]].mean(axis=1)
                        sc = ax.scatter(emb_pair[:, 0], emb_pair[:, 1], c=act_g2,
                                        cmap='coolwarm', s=3, alpha=0.4, rasterized=True)
                        plt.colorbar(sc, ax=ax, shrink=0.8)
                        ax.set_title(f'SG{g1+1}+SG{g2+1}, color=SG{g2+1}', fontsize=9)
                        ax.set_xlabel('UMAP 1', fontsize=8)

                        pair_idx += 1

                plt.tight_layout(rect=[0, 0, 1, 0.95])
                outpath2 = Path("figures") / f"gru_ode_manifold_pairs_{region}_{state.lower()}.png"
                fig2.savefig(outpath2, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  Saved: {outpath2}")

    print("\nDone!")


if __name__ == "__main__":
    main()
