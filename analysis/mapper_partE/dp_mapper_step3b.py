"""
Step 3b: binary scored behaviors on the Mapper graph.

Behaviors: feeding, digging_sand, quick_one_loop_at_home, incomplete_home_returns.

For each (S3/S4 x ACA/LHA): per-node fraction of bins in which behavior is on,
Moran's I on that per-node fraction (does behavior-enriched neighborhood cluster
on the graph?), and chi-square on the node x {off,on} contingency table.

Outputs:
    data/mapper_partE/step3b_binary_behaviors.csv
    figures/mapper_partE/S{N}_{region}_step3b_{behavior}.png
"""
import sys
from pathlib import Path
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

LENS = 'pc1'
SEED = 42
SESSIONS = [3, 4]
REGIONS = ['ACA', 'LHA']
BEHAVIORS = ['feeding', 'digging_sand', 'quick_one_loop_at_home',
             'incomplete_home_returns']

outdir = REPO / "data" / "mapper_partE"
figdir = REPO / "figures" / "mapper_partE"
outdir.mkdir(parents=True, exist_ok=True)
figdir.mkdir(parents=True, exist_ok=True)


def plot_fraction(G, node_vals, title, out_path, label):
    pos = nx.spring_layout(G, seed=SEED)
    fig, ax = plt.subplots(figsize=(10, 8))
    sizes = [60 + 10 * np.sqrt(len(G.nodes[n]['members'])) for n in G.nodes]
    raw = [node_vals[n] for n in G.nodes]
    vals = np.array([np.nan if v is None else v for v in raw], dtype=float)
    finite = np.isfinite(vals)
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
                                             node_color=vals[finite], cmap='inferno',
                                             vmin=0.0, vmax=1.0,
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
            print(f"\n=== S{sess} {region} ===")
            matrix, bin_centers, n_units = load_neural(sess, region)
            K = K_PCS[region]
            G, desc, _, _, _ = build_mapper(matrix, K, lens_name=LENS)
            print(f"  graph: nodes={desc['n_nodes']} edges={desc['n_edges']} "
                  f"comps={desc['n_components']} cycles={desc['n_cycles']}")

            behav = load_behavior(sess, bin_centers)

            for bname in BEHAVIORS:
                if bname not in behav:
                    print(f"  [skip] {bname} not present")
                    continue
                # binary -> {0, 1} integer
                vals_str = behav[bname]['values']
                vals = (vals_str == '1').astype(float)
                # per-node fraction
                frac = node_mean(G, vals)
                # Moran's I
                I, p_I = morans_I(G, vals)
                # chi-square (on string labels)
                chi2, p_chi, V = graph_chi2(G, vals_str, classes=['0', '1'])
                base_rate = float(np.mean(vals))
                print(f"  {bname:36s} base={base_rate:.3f}  Moran I={I:.3f} "
                      f"(p={p_I:.3f})  chi2={chi2:.0f} (p={p_chi:.2e}, V={V:.3f})")
                rows.append(dict(session=sess, region=region, behavior=bname,
                                 base_rate=base_rate, n_nodes=desc['n_nodes'],
                                 morans_I=I, p_morans=p_I,
                                 chi2=chi2, p_chi2=p_chi, cramers_V=V))
                fig_path = figdir / f"S{sess}_{region}_step3b_{bname}.png"
                plot_fraction(G, frac,
                              title=f"S{sess} {region} (lens={LENS}) — {bname}\n"
                                    f"base rate={base_rate:.2%}, Moran's I={I:.3f} "
                                    f"(p={p_I:.3f}), V={V:.3f}",
                              out_path=fig_path,
                              label=f"fraction of bins with {bname}=1")

    df = pd.DataFrame(rows)
    out_csv = outdir / "step3b_binary_behaviors.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
