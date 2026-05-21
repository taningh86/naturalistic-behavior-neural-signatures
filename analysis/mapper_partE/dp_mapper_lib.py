"""
Shared helpers for Mapper Part E (Steps 3, 5, 6).

build_mapper(matrix, K, lens_name, eps_pct=90):
    Run PCA -> lens -> DBSCAN-Mapper, return (G, desc, X_pca, lens, eps).

graph_descriptors(G):
    nodes / edges / components / cycles / branching / endpoints / diameter.

morans_I(G, values):
    Graph-Moran's-I for a continuous variable on Mapper nodes.

graph_chi2(G, labels):
    Chi-square test that nodes are non-uniformly enriched for a categorical label.
"""
import numpy as np
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import kmapper as km

N_INTERVALS = 20
OVERLAP = 0.5
MIN_CLUSTER = 5


def estimate_dbscan_eps(X, k=5, percentile=90):
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(X)
    d, _ = nn.kneighbors(X)
    return float(np.percentile(d[:, k], percentile))


def graph_from_kmapper(graph_dict):
    G = nx.Graph()
    for n, mem in graph_dict['nodes'].items():
        G.add_node(n, members=list(mem))
    for src, dsts in graph_dict['links'].items():
        for dst in dsts:
            G.add_edge(src, dst)
    return G


def graph_descriptors(G):
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    n_components = nx.number_connected_components(G) if n_nodes else 0
    n_cycles = n_edges - n_nodes + n_components
    deg = dict(G.degree())
    n_branching = sum(1 for d in deg.values() if d >= 3)
    n_endpoints = sum(1 for d in deg.values() if d == 1)
    n_isolated = sum(1 for d in deg.values() if d == 0)
    diameter = None
    if n_nodes > 0:
        comps = list(nx.connected_components(G))
        H = G.subgraph(max(comps, key=len))
        if H.number_of_nodes() > 1:
            try:
                diameter = nx.diameter(H)
            except nx.NetworkXError:
                diameter = None
    sizes = [len(G.nodes[n]['members']) for n in G.nodes] if n_nodes else [0]
    return dict(
        n_nodes=n_nodes, n_edges=n_edges, n_components=int(n_components),
        n_cycles=int(n_cycles), n_branching=int(n_branching),
        n_endpoints=int(n_endpoints), n_isolated=int(n_isolated),
        diameter=int(diameter) if diameter is not None else None,
        median_node_size=float(np.median(sizes)),
        max_node_size=int(np.max(sizes)),
    )


def build_mapper(matrix, K, lens_name='pc1', eps_pct=90, n_intervals=N_INTERVALS,
                 overlap=OVERLAP, min_cluster=MIN_CLUSTER, pca_seed=0):
    pca = PCA(n_components=K, random_state=pca_seed)
    X_pca = pca.fit_transform(matrix)
    var_pct = float(np.sum(pca.explained_variance_ratio_) * 100.0)

    if lens_name == 'pc1':
        lens = X_pca[:, :1]
    elif lens_name == 'pc12':
        lens = X_pca[:, :2]
    else:
        raise ValueError(f"unknown lens {lens_name}")

    eps = estimate_dbscan_eps(X_pca, k=5, percentile=eps_pct)
    clusterer = DBSCAN(eps=eps, min_samples=min_cluster)
    cover = km.Cover(n_cubes=n_intervals, perc_overlap=overlap)
    mapper = km.KeplerMapper(verbose=0)
    graph = mapper.map(lens=lens, X=X_pca, clusterer=clusterer, cover=cover,
                       remove_duplicate_nodes=True)
    G = graph_from_kmapper(graph)
    desc = graph_descriptors(G)
    desc['eps'] = eps
    desc['pca_var_pct'] = var_pct
    desc['lens'] = lens_name
    return G, desc, X_pca, lens, eps


def node_mean(G, values):
    """Per-node mean of a continuous variable."""
    out = {}
    for n in G.nodes:
        m = G.nodes[n]['members']
        if not m:
            out[n] = np.nan
            continue
        v = values[m]
        v = v[np.isfinite(v)]
        out[n] = float(np.mean(v)) if len(v) else np.nan
    return out


def morans_I(G, values):
    """Moran's I on the graph adjacency for a continuous variable.

    Larger I = neighboring nodes have similar values (smooth on graph).
    Returns (I, p_z) using the analytical mean/variance under randomization.
    """
    nm = node_mean(G, values)
    nodes = [n for n in G.nodes if np.isfinite(nm[n])]
    if len(nodes) < 5 or G.number_of_edges() == 0:
        return np.nan, np.nan
    idx = {n: i for i, n in enumerate(nodes)}
    x = np.array([nm[n] for n in nodes], dtype=float)
    n = len(x)
    xbar = x.mean()
    dev = x - xbar
    s2 = (dev ** 2).sum()
    if s2 == 0:
        return np.nan, np.nan

    W = 0
    cross = 0.0
    for u, v in G.edges:
        if u in idx and v in idx:
            i, j = idx[u], idx[v]
            cross += dev[i] * dev[j]
            W += 1
    if W == 0:
        return np.nan, np.nan
    cross *= 2  # both directions
    W *= 2

    I = (n / W) * (cross / s2)
    EI = -1.0 / (n - 1)
    # Permutation p-value: shuffle node values, recompute I (cheap, since W fixed)
    rng = np.random.default_rng(0)
    n_perm = 500
    null = np.empty(n_perm)
    edges = list(G.edges)
    for k in range(n_perm):
        xp = rng.permutation(x)
        d = xp - xp.mean()
        c = 0.0
        for u, v in edges:
            if u in idx and v in idx:
                c += d[idx[u]] * d[idx[v]]
        c *= 2
        s = (d ** 2).sum()
        null[k] = (n / W) * (c / s) if s > 0 else 0
    p_perm = (null >= I).mean()
    return float(I), float(p_perm)


def graph_chi2(G, labels, classes=None):
    """Chi-square: are nodes non-uniformly enriched for label classes?

    Builds a [n_nodes x n_classes] table of label counts within each node,
    tests independence (label distribution differs across nodes).
    Skips empty/singleton tables.
    """
    if classes is None:
        classes = sorted({str(v) for v in np.unique(labels) if v is not None})
    if len(classes) < 2 or G.number_of_nodes() < 2:
        return np.nan, np.nan, np.nan
    tab = []
    for n in G.nodes:
        m = G.nodes[n]['members']
        if not m:
            continue
        row = [int(np.sum(labels[m] == c)) for c in classes]
        if sum(row) > 0:
            tab.append(row)
    tab = np.asarray(tab)
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return np.nan, np.nan, np.nan
    # Drop all-zero columns
    keep = tab.sum(axis=0) > 0
    tab = tab[:, keep]
    if tab.shape[1] < 2:
        return np.nan, np.nan, np.nan
    try:
        chi2, p, dof, _ = stats.chi2_contingency(tab)
    except Exception:
        return np.nan, np.nan, np.nan
    # Cramer's V (effect size)
    n = tab.sum()
    r, c = tab.shape
    V = float(np.sqrt(chi2 / (n * (min(r, c) - 1)))) if min(r, c) > 1 and n > 0 else np.nan
    return float(chi2), float(p), V
