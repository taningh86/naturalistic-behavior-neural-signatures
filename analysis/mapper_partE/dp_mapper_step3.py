"""
Step 3: behavioral coloring + statistics on existing PC1 Mapper graphs.

For each (S3/S4 x ACA/LHA): rebuild the graph (same PC1 settings as Step 1),
then compute:
- Moran's I on graph adjacency for continuous variables (velocity, dist_pot-1..4)
- Chi-square (and Cramer's V) for compartment label clustering across nodes
- Render per-variable graph figures

Outputs:
    data/mapper_partE/step3_behavior_stats.csv
    figures/mapper_partE/S{N}_{region}_step3_{var}.png  (one per variable)
"""
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis" / "cycles_partD"))
sys.path.insert(0, str(REPO / "analysis" / "mapper_partE"))

from dp_cycles_lib import load_neural, load_behavior, K_PCS
from dp_mapper_lib import build_mapper, node_mean, morans_I, graph_chi2

SEED = 42
LENS = 'pc1'
SESSIONS = [3, 4]
REGIONS = ['ACA', 'LHA']
CONT_VARS = ['velocity', 'dist_pot-1_/_center-point', 'dist_pot-2_/_center-point',
             'dist_pot-3_/_center-point', 'dist_pot-4_/_center-point']

outdir = REPO / "data" / "mapper_partE"
figdir = REPO / "figures" / "mapper_partE"
outdir.mkdir(parents=True, exist_ok=True)
figdir.mkdir(parents=True, exist_ok=True)


def plot_continuous(G, node_vals, title, out_path, label):
    pos = nx.spring_layout(G, seed=SEED)
    fig, ax = plt.subplots(figsize=(10, 8))
    sizes = [60 + 10 * np.sqrt(len(G.nodes[n]['members'])) for n in G.nodes]
    raw = [node_vals[n] for n in G.nodes]
    vals = np.array([np.nan if v is None else v for v in raw], dtype=float)
    finite = np.isfinite(vals)
    if finite.any():
        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
    else:
        vmin, vmax = 0.0, 1.0
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
    node_list = list(G.nodes)
    nan_nodes = [n for n, ok in zip(node_list, finite) if not ok]
    if nan_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=nan_nodes,
                               node_color=[(0.7, 0.7, 0.7, 1.0)] * len(nan_nodes),
                               node_size=[sizes[i] for i, ok in enumerate(finite) if not ok],
                               edgecolors='black', linewidths=0.5, ax=ax)
    good_nodes = [n for n, ok in zip(node_list, finite) if ok]
    if good_nodes:
        nodes_drawn = nx.draw_networkx_nodes(G, pos, nodelist=good_nodes,
                                             node_color=vals[finite], cmap='magma',
                                             vmin=vmin, vmax=vmax,
                                             node_size=[sizes[i] for i, ok in enumerate(finite) if ok],
                                             edgecolors='black', linewidths=0.5, ax=ax)
        cb = plt.colorbar(nodes_drawn, ax=ax, label=label,
                          shrink=0.6, fraction=0.04, pad=0.02)
        cb.ax.tick_params(labelsize=8)
    ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()


def main():
    rows = []
    for sess in SESSIONS:
        for region in REGIONS:
            print(f"\n--- S{sess} {region} ---")
            matrix, bin_centers, n_units = load_neural(sess, region)
            K = K_PCS[region]
            G, desc, _, _, eps = build_mapper(matrix, K, lens_name=LENS)
            print(f"  graph: nodes={desc['n_nodes']} edges={desc['n_edges']} "
                  f"comps={desc['n_components']} cycles={desc['n_cycles']}")

            behav = load_behavior(sess, bin_centers)

            # compartment chi-square
            comp = behav['compartment']['values']
            classes = behav['compartment']['classes']
            chi2, p_chi, V = graph_chi2(G, comp, classes=classes)
            print(f"  compartment chi-square: chi2={chi2:.1f} p={p_chi:.2e} V={V:.3f}")
            rows.append(dict(session=sess, region=region, variable='compartment',
                             stat='chi2', value=chi2, p=p_chi, effect=V,
                             n_nodes=desc['n_nodes']))

            # continuous variables: Moran's I + figure
            for var in CONT_VARS:
                if var not in behav:
                    continue
                vals = behav[var]['values'].astype(float)
                I, p_I = morans_I(G, vals)
                nm = node_mean(G, vals)
                short_label = var.replace('_/_center-point', '').replace('_', ' ')
                print(f"  {short_label:14s} Moran I={I:.3f}  p_perm={p_I:.3f}")
                rows.append(dict(session=sess, region=region, variable=var,
                                 stat='morans_I', value=I, p=p_I, effect=I,
                                 n_nodes=desc['n_nodes']))
                short_var = var.replace('_/_center-point', '').replace('-', '').replace('_', '')
                fig_path = figdir / f"S{sess}_{region}_step3_{short_var}.png"
                plot_continuous(G, nm,
                                title=f"S{sess} {region} (lens={LENS}) — {short_label}\n"
                                      f"Moran's I={I:.3f} (p_perm={p_I:.3f})",
                                out_path=fig_path,
                                label=f"mean {short_label}")

    df = pd.DataFrame(rows)
    out_csv = outdir / "step3_behavior_stats.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
