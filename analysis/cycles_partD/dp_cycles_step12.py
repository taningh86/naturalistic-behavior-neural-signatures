"""
Step 1-2: Rerun PH with dreimac, extract circular coordinates for top 2 H1 features.

Usage:
    python dp_cycles_step12.py <session_num> <region>

Outputs (per region):
    data/cycles_partD/S{N}_{region}_step12.json   persistence + per-feature metadata
    data/cycles_partD/phase_timeseries/S{N}_{region}_seed42.npz
        keys: phi (F x N), cocycle_indices (F,), persistences (F,)
    figures/cycles_partD/S{N}_{region}_step12_ph.png  persistence diagram + top features
    figures/cycles_partD/S{N}_{region}_step12_phase.png  phase time series

STOP condition: if the phase time series looks degenerate (constant, or circular variance < 0.5).
"""
import sys
import json
import time as timer
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from dreimac import CircularCoords

from dp_cycles_lib import (
    load_neural, K_PCS, BIN_MS, SMOOTH_SIGMA, circ_var,
)

N_LANDMARKS = 1000   # matches dp_manifold_layer1b
SEED_DEFAULT = 42
N_TOP_FEATURES = 2

repo_root = Path(__file__).resolve().parent.parent.parent
outdir = repo_root / "data" / "cycles_partD"
ts_dir = outdir / "phase_timeseries"
figdir = repo_root / "figures" / "cycles_partD"
outdir.mkdir(parents=True, exist_ok=True)
ts_dir.mkdir(parents=True, exist_ok=True)
figdir.mkdir(parents=True, exist_ok=True)


def persistence_lifetimes(dgm):
    """Return (lifetime_sorted_desc, indices_sorted_desc) for a single H-k diagram."""
    finite = np.isfinite(dgm[:, 1])
    life = np.where(finite, dgm[:, 1] - dgm[:, 0], 0.0)
    order = np.argsort(life)[::-1]
    return life[order], order


def run_region(session_num, region, seed=SEED_DEFAULT):
    print(f"\n{'='*70}")
    print(f"Step 1-2: S{session_num} {region} (seed={seed})")
    print(f"{'='*70}")

    t0 = timer.time()
    matrix, bin_centers, n_units = load_neural(session_num, region)
    print(f"Neural matrix: {matrix.shape} ({n_units} units), {len(bin_centers)} bins")

    K = K_PCS[region]
    pca = PCA(n_components=K)
    X_pca = pca.fit_transform(matrix)
    print(f"PCA to K={K} ({sum(pca.explained_variance_ratio_)*100:.1f}% variance)")

    # Shuffle row order with given seed so dreimac's internal MaxMin (which starts from row 0)
    # selects different landmarks. Un-shuffle phi afterwards.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(X_pca.shape[0])
    inv_perm = np.argsort(perm)
    X_perm = X_pca[perm]

    print(f"Fitting dreimac CircularCoords (N={X_perm.shape[0]}, n_landmarks={N_LANDMARKS})...")
    tph = timer.time()
    cc = CircularCoords(X_perm, n_landmarks=N_LANDMARKS, maxdim=1, verbose=False)
    print(f"  persistence done in {timer.time()-tph:.1f}s")

    dgms = cc._dgms
    h1 = dgms[1]
    lifetimes_desc, order_desc = persistence_lifetimes(h1)
    print(f"H1 features (n={len(h1)}). Top lifetimes: "
          + ", ".join(f"{v:.3f}" for v in lifetimes_desc[:5]))

    # Compare to prior PH run stored in data/manifold/S{N}_{region}_layer1b.json if present
    prior_path = repo_root / "data" / "manifold" / f"S{session_num}_{region}_layer1b.json"
    prior = None
    if prior_path.exists():
        with open(prior_path) as f:
            prior = json.load(f)
        prior_max = prior['stats']['H1_max_persistence']
        prior_top3 = prior['stats']['H1_top3']
        print(f"Prior PH (layer1b): H1_max={prior_max:.3f}, top3={prior_top3}")
        print(f"Current H1 top3: {lifetimes_desc[:3].tolist()}")
    else:
        print(f"  no prior layer1b for this region (expected at {prior_path.name})")

    # Extract phase for top N features that clear >3SD above null
    # If prior is available, use its null mean/SD; otherwise take top N_TOP_FEATURES unconditionally
    threshold = 0.0
    if prior is not None:
        null_mean = prior['null_comparison']['H1']['max_pers_null_mean']
        null_std = prior['null_comparison']['H1']['max_pers_null_std']
        threshold = null_mean + 3 * null_std
        print(f"Significance threshold (prior null mean + 3 SD): {threshold:.3f}")

    keep_idx = []
    for rank, li in enumerate(lifetimes_desc[:N_TOP_FEATURES]):
        if li >= threshold:
            keep_idx.append(int(order_desc[rank]))
        else:
            print(f"  feature rank {rank} lifetime={li:.3f} below threshold; excluding")
    if len(keep_idx) == 0:
        print(f"STOP: no features clear threshold. Not extracting coordinates.")
        summary = {
            'session': session_num, 'region': region, 'seed': seed,
            'n_units': n_units, 'n_bins': matrix.shape[0],
            'K': K, 'n_landmarks': N_LANDMARKS,
            'h1_lifetimes_top5': lifetimes_desc[:5].tolist(),
            'threshold': threshold,
            'n_features_extracted': 0,
            'stopped': True,
            'stop_reason': 'no_features_above_threshold',
        }
        out = outdir / f"S{session_num}_{region}_step12.json"
        with open(out, 'w') as f:
            json.dump(summary, f, indent=2)
        return

    print(f"\nExtracting circular coordinates for {len(keep_idx)} feature(s)")
    phis_perm = []  # each is length N in permuted order
    persistences = []
    cocycle_indices = []
    for rank, ci in enumerate(keep_idx):
        # dreimac's `cocycle_idx` expects index in the persistence diagram (after internal sort).
        # CircularCoords indexes cocycles in DECREASING persistence order; rank is the index.
        # But `keep_idx` holds original H1-diagram indices. Map rank -> dreimac's rank.
        dreimac_idx = rank  # top-N already sorted descending
        # standard_range=False because with neural data the cocycle birth is often
        # larger than the "standard" cover-radius cutoff (dreimac raises "class too short"
        # otherwise). This passes non-standard to the EM coordinate construction.
        phi = cc.get_coordinates(perc=0.5, cocycle_idx=dreimac_idx,
                                 standard_range=False)
        phi = np.asarray(phi)
        if phi.shape[0] != X_perm.shape[0]:
            raise RuntimeError(f"phi length {phi.shape[0]} != N {X_perm.shape[0]}")
        phis_perm.append(phi)
        persistences.append(float(lifetimes_desc[rank]))
        cocycle_indices.append(dreimac_idx)
        cv = circ_var(phi)
        print(f"  feature rank {rank} (persistence={lifetimes_desc[rank]:.3f}): "
              f"phi range [{phi.min():.3f}, {phi.max():.3f}], circ_var={cv:.3f}")
        if cv < 0.1:
            print(f"    WARNING: circ_var very low — phase is nearly constant")

    # Un-permute to original time order
    phis = [phi[inv_perm] for phi in phis_perm]
    phi_stack = np.asarray(phis)  # (F, N)

    # Save
    npz_path = ts_dir / f"S{session_num}_{region}_seed{seed}.npz"
    np.savez_compressed(
        npz_path,
        phi=phi_stack,
        persistences=np.asarray(persistences),
        cocycle_indices=np.asarray(cocycle_indices, dtype=int),
        bin_centers=bin_centers,
    )
    print(f"Saved {npz_path}")

    # Persistence diagram figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for k, dgm in enumerate(dgms):
        finite = dgm[np.isfinite(dgm[:, 1])]
        if len(finite) > 0:
            ax.scatter(finite[:, 0], finite[:, 1], s=18, alpha=0.5, c=colors[k],
                       label=f'H{k} ({len(finite)})')
    # Highlight extracted features in H1 diagram
    h1_finite = dgms[1][np.isfinite(dgms[1][:, 1])]
    # Map rank -> diagram index
    life_h1 = h1_finite[:, 1] - h1_finite[:, 0]
    order_h1 = np.argsort(life_h1)[::-1]
    for r in range(len(keep_idx)):
        i = order_h1[r]
        ax.scatter(h1_finite[i, 0], h1_finite[i, 1], s=180, facecolor='none',
                   edgecolor='red', linewidth=2, zorder=5,
                   label=f'extracted rank {r}' if r == 0 else None)
    max_val = max(finite.max() for finite in [dgm[np.isfinite(dgm[:, 1])] for dgm in dgms] if len(finite) > 0)
    ax.plot([0, max_val * 1.1], [0, max_val * 1.1], 'k--', alpha=0.3, lw=1)
    ax.set_xlabel('Birth')
    ax.set_ylabel('Death')
    ax.set_title(f'S{session_num} {region} persistence (seed={seed})')
    ax.legend(fontsize=8)

    # Lifetime rank plot
    ax = axes[1]
    for k in range(len(dgms)):
        dgm = dgms[k]
        finite = dgm[np.isfinite(dgm[:, 1])]
        if len(finite) == 0:
            continue
        life = finite[:, 1] - finite[:, 0]
        life_sorted = np.sort(life)[::-1]
        ax.plot(np.arange(len(life_sorted)), life_sorted, '-o',
                ms=3, color=colors[k], label=f'H{k}')
    ax.set_xlabel('Feature rank')
    ax.set_ylabel('Persistence')
    ax.set_title('Lifetime by rank')
    ax.legend()
    ax.set_xlim(-0.5, 30)
    plt.tight_layout()
    ph_fig = figdir / f"S{session_num}_{region}_step12_ph.png"
    plt.savefig(ph_fig, dpi=120)
    plt.close()
    print(f"Saved {ph_fig}")

    # Phase time series figure
    n_show = min(len(bin_centers), 3000)
    t_sec = bin_centers[:n_show]
    fig, axes = plt.subplots(len(phis), 1, figsize=(14, 2.2 * len(phis)),
                             sharex=True, squeeze=False)
    for r, phi in enumerate(phis):
        ax = axes[r, 0]
        ax.plot(t_sec, phi[:n_show], lw=0.5)
        ax.set_ylabel(f'phi_{r} (rad)')
        ax.set_title(f'Feature {r} (persistence={persistences[r]:.3f})')
        ax.set_ylim(-0.2, 2 * np.pi + 0.2)
    axes[-1, 0].set_xlabel('Time (s)')
    plt.tight_layout()
    ts_fig = figdir / f"S{session_num}_{region}_step12_phase.png"
    plt.savefig(ts_fig, dpi=120)
    plt.close()
    print(f"Saved {ts_fig}")

    # Summary JSON
    summary = {
        'session': session_num, 'region': region, 'seed': seed,
        'n_units': n_units, 'n_bins': matrix.shape[0], 'K': K,
        'n_landmarks': N_LANDMARKS,
        'h1_lifetimes_top5': lifetimes_desc[:5].tolist(),
        'prior_threshold': threshold,
        'n_features_extracted': len(keep_idx),
        'extracted_persistences': persistences,
        'extracted_circ_var': [float(circ_var(phi)) for phi in phis],
        'stopped': False,
    }
    out = outdir / f"S{session_num}_{region}_step12.json"
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {out}")
    print(f"Region total time: {(timer.time()-t0)/60:.1f} min")


def main():
    if len(sys.argv) < 3:
        print("Usage: python dp_cycles_step12.py <session_num> <region>")
        sys.exit(1)
    session_num = int(sys.argv[1])
    region = sys.argv[2]
    run_region(session_num, region)


if __name__ == '__main__':
    main()
