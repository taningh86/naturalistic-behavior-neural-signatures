"""
Test: run dreimac without my row permutation, on both ACA and LHA.
Also test with more landmarks (2000) to see if that helps coverage.
"""
import sys
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from dreimac import CircularCoords

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis" / "cycles_partD"))
from dp_cycles_lib import load_neural, K_PCS


def describe(phi):
    phi = np.asarray(phi)
    R = np.hypot(np.sin(phi).mean(), np.cos(phi).mean())
    bins = np.linspace(0, 2 * np.pi, 9)
    h, _ = np.histogram(np.mod(phi, 2 * np.pi), bins=bins)
    return 1 - R, h


def run_case(region, n_landmarks, use_permutation):
    matrix, bc, nu = load_neural(3, region)
    X = PCA(n_components=K_PCS[region]).fit_transform(matrix)
    if use_permutation:
        rng = np.random.default_rng(42)
        X = X[rng.permutation(X.shape[0])]
    cc = CircularCoords(X, n_landmarks=n_landmarks, maxdim=1, verbose=False)
    finite = cc._dgms[1][np.isfinite(cc._dgms[1][:, 1])]
    life = finite[:, 1] - finite[:, 0]
    order = np.argsort(life)[::-1]
    print(f"\n{region} n_land={n_landmarks} perm={use_permutation}: "
          f"top H1 persistences {[f'{life[order[r]]:.3f}' for r in range(5)]}")
    for r in range(2):
        try:
            phi = cc.get_coordinates(perc=0.5, cocycle_idx=r, standard_range=False)
            cv, h = describe(phi)
            print(f"  rank {r}: circ_var={cv:.4f}, hist8={h.tolist()}")
        except Exception as e:
            print(f"  rank {r}: EXCEPTION {type(e).__name__}: {str(e)[:80]}")


def main():
    for region in ['ACA', 'LHA']:
        run_case(region, 1000, use_permutation=False)
        run_case(region, 2000, use_permutation=False)


if __name__ == '__main__':
    main()
