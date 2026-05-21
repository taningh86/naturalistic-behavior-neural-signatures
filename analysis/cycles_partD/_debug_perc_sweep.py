"""
Debug: sweep perc (coverage) values to see if phase distribution recovers.
"""
import sys
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from dreimac import CircularCoords

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis" / "cycles_partD"))
from dp_cycles_lib import load_neural, K_PCS, circ_var


def describe(phi):
    R = np.hypot(np.sin(phi).mean(), np.cos(phi).mean())
    cv = 1 - R
    # Fraction in each pi/4 bin
    bins = np.linspace(0, 2*np.pi, 9)
    h, _ = np.histogram(np.mod(phi, 2*np.pi), bins=bins)
    return cv, h


def main():
    matrix, bc, nu = load_neural(3, 'ACA')
    X = PCA(n_components=K_PCS['ACA']).fit_transform(matrix)
    rng = np.random.default_rng(42)
    perm = rng.permutation(X.shape[0])
    X_perm = X[perm]

    cc = CircularCoords(X_perm, n_landmarks=1000, maxdim=1, verbose=False)
    # Print top H1 features (birth, death)
    dgm_h1 = cc._dgms[1]
    finite = dgm_h1[np.isfinite(dgm_h1[:, 1])]
    life = finite[:, 1] - finite[:, 0]
    order = np.argsort(life)[::-1]
    print("Top 5 H1 features (birth, death, persistence):")
    for r in range(5):
        b, d = finite[order[r]]
        print(f"  rank {r}: birth={b:.3f} death={d:.3f} pers={d-b:.3f}")

    # Test perc values
    print("\nSweeping perc for rank-0 cocycle:")
    for perc in [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
        try:
            phi = cc.get_coordinates(perc=perc, cocycle_idx=0,
                                      standard_range=False)
            cv, h = describe(np.asarray(phi))
            print(f"  perc={perc}: circ_var={cv:.4f}  hist_8bin={h.tolist()}")
        except Exception as e:
            print(f"  perc={perc}: EXCEPTION {type(e).__name__}: {str(e)[:80]}")


if __name__ == '__main__':
    main()
