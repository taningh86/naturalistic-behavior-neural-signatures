"""
Mapper Step 1 on a single session/region.

Usage:
    python dp_mapper_step1.py <session_num> <region> [lens]

lens: "pc12" (default, PC1+PC2 2D) or "pc1" (1D)

Settings:
- Cover: 20 intervals, 50% overlap (per dimension)
- Clusterer: DBSCAN with eps from global 5-NN distance, 90th pct
- Min cluster size: 5

Outputs:
    data/mapper_partE/S{N}_{region}_step1_{lens}_graph.json
    data/mapper_partE/S{N}_{region}_step1_{lens}_descriptors.json
    figures/mapper_partE/S{N}_{region}_step1_{lens}_compartment.png
    figures/mapper_partE/S{N}_{region}_step1_{lens}_velocity.png
"""
import sys
import json
import time as timer
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import kmapper as km

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis" / "cycles_partD"))

from dp_cycles_lib import load_neural, load_behavior, K_PCS

N_INTERVALS = 20
OVERLAP = 0.5
MIN_CLUSTER = 5
SEED = 42

outdir = REPO / "data" / "mapper_partE"
figdir = REPO / "figures" / "mapper_partE"
outdir.mkdir(parents=True, exist_ok=True)
figdir.mkdir(parents=True, exist_ok=True)


def estimate_dbscan_eps(X, k=5, percentile=90):
    """Pick DBSCAN eps from the k-NN distance distribution."""
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(X)
    d, _ = nn.kneighbors(X)
    kth = d[:, k]
    return float(np.percentile(kth, percentile))


def graph_from_kmapper(graph_dict):
    """Convert km graph dict to networkx for descriptors."""
    G = nx.Graph()
    for node_id in graph_dict['nodes'].keys():
        G.add_node(node_id, members=graph_dict['nodes'][node_id])
    for src, dsts in graph_dict['links'].items():
        for dst in dsts:
            G.add_edge(src, dst)
    return G


def graph_descriptors(G):
    """Quantitative summary of graph shape."""
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    n_components = nx.number_connected_components(G)
    # Cyclomatic number (independent cycles)
    n_cycles = n_edges - n_nodes + n_components
    degrees = dict(G.degree())
    n_branching = sum(1 for d in degrees.values() if d >= 3)
    n_endpoints = sum(1 for d in degrees.values() if d == 1)
    n_isolated = sum(1 for d in degrees.values() if d == 0)
    # Diameter on largest component
    diameter = None
    if n_nodes > 0:
        comps = list(nx.connected_components(G))
        largest = max(comps, key=len)
        H = G.subgraph(largest)
        if H.number_of_nodes() > 1:
            try:
                diameter = nx.diameter(H)
            except nx.NetworkXError:
                diameter = None
    return dict(
        n_nodes=n_nodes,
        n_edges=n_edges,
        n_components=int(n_components),
        n_cycles=int(n_cycles),
        n_branching=int(n_branching),
        n_endpoints=int(n_endpoints),
        n_isolated=int(n_isolated),
        diameter=int(diameter) if diameter is not None else None,
        median_node_size=float(np.median([len(G.nodes[n]['members']) for n in G.nodes])) if n_nodes else 0,
        max_node_size=int(np.max([len(G.nodes[n]['members']) for n in G.nodes])) if n_nodes else 0,
    )


def compartment_majority(members, compartment_array):
    """Return majority compartment label for cluster members."""
    if len(members) == 0:
        return 'empty'
    vals, counts = np.unique(compartment_array[members], return_counts=True)
    return str(vals[np.argmax(counts)])


def color_by_continuous(members, var_array):
    if len(members) == 0:
        return np.nan
    vals = var_array[members]
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if len(vals) else np.nan


def plot_graph(G, node_labels, node_values, title, out_path,
               cmap_label=None, cmap='viridis'):
    pos = nx.spring_layout(G, seed=SEED, k=None)
    fig, ax = plt.subplots(figsize=(11, 9))
    sizes = [60 + 10 * np.sqrt(len(G.nodes[n]['members'])) for n in G.nodes]
    if isinstance(node_values, dict) and node_values:
        # categorical
        unique = sorted({v for v in node_values.values() if v is not None})
        palette = plt.cm.tab10(np.linspace(0, 1, max(len(unique), 1)))
        color_map = {u: palette[i] for i, u in enumerate(unique)}
        node_colors = [color_map.get(node_values.get(n, 'empty'), (0.7, 0.7, 0.7, 1.0))
                       for n in G.nodes]
        nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=sizes,
                               edgecolors='black', linewidths=0.5, ax=ax)
        # legend
        handles = [plt.scatter([], [], s=80, c=[color_map[u]], label=u) for u in unique]
        ax.legend(handles=handles, title=cmap_label or 'category', loc='best', fontsize=8)
    else:
        # continuous
        raw = [node_values[n] for n in G.nodes]
        vals = np.array([np.nan if v is None else v for v in raw], dtype=float)
        # mask NaN nodes to a neutral grey
        finite = np.isfinite(vals)
        if finite.any():
            vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
        else:
            vmin, vmax = 0.0, 1.0
        nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
        node_list = list(G.nodes)
        # draw NaN nodes first (grey)
        nan_nodes = [n for n, ok in zip(node_list, finite) if not ok]
        if nan_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=nan_nodes,
                                   node_color=[(0.7, 0.7, 0.7, 1.0)] * len(nan_nodes),
                                   node_size=[sizes[i] for i, ok in enumerate(finite) if not ok],
                                   edgecolors='black', linewidths=0.5, ax=ax)
        good_nodes = [n for n, ok in zip(node_list, finite) if ok]
        good_vals = vals[finite]
        good_sizes = [sizes[i] for i, ok in enumerate(finite) if ok]
        nodes_drawn = nx.draw_networkx_nodes(G, pos, nodelist=good_nodes,
                                             node_color=good_vals, cmap=cmap,
                                             vmin=vmin, vmax=vmax,
                                             node_size=good_sizes,
                                             edgecolors='black', linewidths=0.5, ax=ax)
        if nodes_drawn is not None:
            cb = plt.colorbar(nodes_drawn, ax=ax, label=cmap_label or 'value',
                              shrink=0.6, fraction=0.04, pad=0.02)
            cb.ax.tick_params(labelsize=8)
    ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()


def run(session_num, region, lens_name='pc12'):
    print(f"\n{'='*70}")
    print(f"Mapper Step 1: S{session_num} {region} (lens={lens_name})")
    print(f"{'='*70}")
    t0 = timer.time()
    matrix, bin_centers, n_units = load_neural(session_num, region)
    print(f"Neural matrix: {matrix.shape}")

    K = K_PCS[region]
    pca = PCA(n_components=K)
    X_pca = pca.fit_transform(matrix)
    var_expl = sum(pca.explained_variance_ratio_) * 100
    print(f"PCA K={K} ({var_expl:.1f}% variance)")

    if lens_name == 'pc1':
        lens = X_pca[:, :1]
    elif lens_name == 'pc12':
        lens = X_pca[:, :2]
    else:
        raise ValueError(f"Unknown lens {lens_name}; use pc1 or pc12")
    print(f"Lens: {lens_name}, shape={lens.shape}")

    # DBSCAN eps
    eps = estimate_dbscan_eps(X_pca, k=5, percentile=90)
    print(f"DBSCAN eps (k=5, 90th pct of 5-NN distance): {eps:.3f}")
    clusterer = DBSCAN(eps=eps, min_samples=MIN_CLUSTER)

    cover = km.Cover(n_cubes=N_INTERVALS, perc_overlap=OVERLAP)
    mapper = km.KeplerMapper(verbose=0)
    print(f"Building Mapper graph (n_cubes={N_INTERVALS}, overlap={OVERLAP})...")
    tg = timer.time()
    graph = mapper.map(
        lens=lens,
        X=X_pca,             # cluster in full PCA space
        clusterer=clusterer,
        cover=cover,
        remove_duplicate_nodes=True,
    )
    print(f"  done in {timer.time()-tg:.1f}s")

    G = graph_from_kmapper(graph)
    desc = graph_descriptors(G)
    print(f"Graph: nodes={desc['n_nodes']}, edges={desc['n_edges']}, "
          f"components={desc['n_components']}, cycles={desc['n_cycles']}, "
          f"branching={desc['n_branching']}, endpoints={desc['n_endpoints']}")
    print(f"  diameter={desc['diameter']}, median node size={desc['median_node_size']:.1f}")

    # Behavioral coloring
    behav = load_behavior(session_num, bin_centers)
    compartment = behav['compartment']['values']
    velocity = behav['velocity']['values']
    # Find a dist_pot variable if present
    dist_pot_keys = [k for k in behav.keys() if k.startswith('dist_pot')]
    print(f"Available behavioral vars: compartment, velocity, "
          f"dist_pot variants: {dist_pot_keys[:5]}")

    node_compartment = {n: compartment_majority(G.nodes[n]['members'], compartment)
                         for n in G.nodes}
    node_velocity = {n: color_by_continuous(G.nodes[n]['members'], velocity)
                     for n in G.nodes}

    fig_compart = figdir / f"S{session_num}_{region}_step1_{lens_name}_compartment.png"
    plot_graph(G, node_compartment, node_compartment,
               title=f"S{session_num} {region} (lens={lens_name}) — by compartment "
                     f"(nodes={desc['n_nodes']}, edges={desc['n_edges']}, "
                     f"comps={desc['n_components']}, cycles={desc['n_cycles']})",
               out_path=fig_compart,
               cmap_label='compartment')

    fig_vel = figdir / f"S{session_num}_{region}_step1_{lens_name}_velocity.png"
    plot_graph(G, None, node_velocity,
               title=f"S{session_num} {region} (lens={lens_name}) — by mean velocity",
               out_path=fig_vel, cmap_label='mean velocity (cm/s)', cmap='magma')

    # Save graph + descriptors
    g_out = outdir / f"S{session_num}_{region}_step1_{lens_name}_graph.json"
    nodes_export = {str(n): {
        'members': [int(m) for m in G.nodes[n]['members']],
        'majority_compartment': node_compartment[n],
        'mean_velocity': node_velocity[n] if np.isfinite(node_velocity[n]) else None,
    } for n in G.nodes}
    edges_export = [[str(u), str(v)] for u, v in G.edges]
    with open(g_out, 'w') as f:
        json.dump({'nodes': nodes_export, 'edges': edges_export,
                   'eps': eps, 'n_intervals': N_INTERVALS,
                   'overlap': OVERLAP, 'lens': lens_name}, f, indent=2)
    print(f"Saved {g_out}")

    d_out = outdir / f"S{session_num}_{region}_step1_{lens_name}_descriptors.json"
    desc['session'] = session_num
    desc['region'] = region
    desc['n_units'] = n_units
    desc['n_bins'] = matrix.shape[0]
    desc['K_pcs'] = K
    desc['pca_var_pct'] = var_expl
    desc['lens'] = lens_name
    desc['eps'] = eps
    desc['n_intervals'] = N_INTERVALS
    desc['overlap'] = OVERLAP
    with open(d_out, 'w') as f:
        json.dump(desc, f, indent=2)
    print(f"Saved {d_out}")
    print(f"Total time: {(timer.time()-t0)/60:.1f} min")


def main():
    if len(sys.argv) < 3:
        print("Usage: python dp_mapper_step1.py <session_num> <region> [lens]")
        sys.exit(1)
    session_num = int(sys.argv[1])
    region = sys.argv[2]
    lens_name = sys.argv[3] if len(sys.argv) > 3 else 'pc12'
    run(session_num, region, lens_name=lens_name)


if __name__ == '__main__':
    main()
