"""
Part F Step 1: extract cycle (cocycle) representatives for the top H1 features.

Inputs: neural matrix (50ms, smoothed, z-scored) -> PCA -> ripser(do_cocycles=True,
n_perm=3000). Per the ripser docs, cocycle vertex indices reference the ORIGINAL
point cloud (i.e., they are already time bin indices in the 36019-bin session).

Outputs (per session/region):
    data/h1_behavior_partF/{region}_{session}_step1.json
        feature_id, persistence, birth, death, n_simplices,
        landmark_bins (sorted unique time-bin indices participating in the cocycle),
        n_landmark_bins
    data/h1_behavior_partF/{region}_{session}_dgm1.npy   (full H1 diagram)

Usage: python partF_step1_extract_cycles.py <session_num> <region>
"""
import sys
from pathlib import Path
import json
import time
import numpy as np
from sklearn.decomposition import PCA
from ripser import ripser

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis" / "cycles_partD"))

from dp_cycles_lib import load_neural, K_PCS

N_LANDMARKS = 3000
N_TOP = 5
SEED = 42

outdir = REPO / "data" / "h1_behavior_partF"
outdir.mkdir(parents=True, exist_ok=True)


def run(session_num, region):
    print(f"\n{'='*70}")
    print(f"Part F Step 1: S{session_num} {region}")
    print(f"{'='*70}")
    t0 = time.time()
    matrix, bin_centers, n_units = load_neural(session_num, region)
    print(f"Neural matrix: {matrix.shape}")

    K = K_PCS[region]
    pca = PCA(n_components=K)
    X = pca.fit_transform(matrix)
    var_pct = float(np.sum(pca.explained_variance_ratio_) * 100.0)
    print(f"PCA K={K} ({var_pct:.1f}% variance). X shape: {X.shape}")

    print(f"Running ripser (maxdim=1, n_perm={N_LANDMARKS}, do_cocycles=True)...")
    tr = time.time()
    res = ripser(X, maxdim=1, n_perm=N_LANDMARKS, do_cocycles=True)
    print(f"  done in {time.time()-tr:.1f}s")

    dgm1 = res['dgms'][1]
    cocycles1 = res['cocycles'][1]
    print(f"H1 features: {len(dgm1)} (some may be infinite)")
    finite_mask = np.isfinite(dgm1[:, 1])
    finite = dgm1[finite_mask]
    finite_cocycles = [cocycles1[i] for i in range(len(dgm1)) if finite_mask[i]]
    persistence = finite[:, 1] - finite[:, 0]
    order = np.argsort(persistence)[::-1]
    top_idx = order[:N_TOP]
    print(f"\nTop {N_TOP} H1 features by persistence:")
    print("  rank  persistence  birth     death     n_simplices  n_unique_bins")
    feats = []
    for rk, idx in enumerate(top_idx):
        bd = finite[idx]
        coc = finite_cocycles[idx]
        # cocycle is (k, 3) -- [vert_i, vert_j, coeff]
        # vertex indices are in original point-cloud space (per ripser docs)
        if coc.shape[0] == 0:
            print(f"  {rk}     EMPTY COCYCLE")
            feats.append(dict(feature_id=int(rk), persistence=float(persistence[idx]),
                              birth=float(bd[0]), death=float(bd[1]),
                              n_simplices=0, landmark_bins=[]))
            continue
        verts = np.unique(coc[:, :2].astype(int).ravel())
        verts = sorted(int(v) for v in verts)
        print(f"  {rk}     {persistence[idx]:.3f}        {bd[0]:.3f}    "
              f"{bd[1]:.3f}    {coc.shape[0]:5d}        {len(verts):5d}")
        feats.append(dict(
            feature_id=int(rk),
            persistence=float(persistence[idx]),
            birth=float(bd[0]),
            death=float(bd[1]),
            n_simplices=int(coc.shape[0]),
            landmark_bins=verts,
            n_landmark_bins=len(verts),
        ))

    # also save persistence stats overall
    out = {
        'session': session_num,
        'region': region,
        'n_units': int(n_units),
        'n_bins': int(matrix.shape[0]),
        'pca_K': K,
        'pca_var_pct': var_pct,
        'n_landmarks': N_LANDMARKS,
        'n_h1_finite': int(len(finite)),
        'top_persistences': [float(persistence[i]) for i in top_idx],
        'features': feats,
    }
    out_json = outdir / f"S{session_num}_{region}_step1.json"
    with open(out_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_json}")

    # save full H1 diagram for reference
    np.save(outdir / f"S{session_num}_{region}_dgm1.npy", dgm1)
    print(f"Saved diagram: {outdir / f'S{session_num}_{region}_dgm1.npy'}")
    print(f"Total time: {(time.time()-t0)/60:.1f} min")


def main():
    if len(sys.argv) < 3:
        print("Usage: python partF_step1_extract_cycles.py <session_num> <region>")
        sys.exit(1)
    session_num = int(sys.argv[1])
    region = sys.argv[2]
    run(session_num, region)


if __name__ == '__main__':
    main()
