"""
Part F shuffle null: per-unit circular-shift of the neural matrix, then re-run
the full pipeline (PCA -> ripser -> cocycle -> k-NN expansion -> temporal +
behavioral tests). Behavior is held fixed (real EthoVision data); only neural
data is shuffled.

If the pipeline is sound, shuffles should produce few/no behaviorally-enriched
features (most should be Class C). If shuffles consistently produce Class A
features with significant behaviors, the pipeline has a leak (likely neural
autocorrelation -> spurious temporal clustering of supporting bins).

Usage: python partF_shuffle_null.py <session_num> <region> [n_shuffles]
"""
import sys
from pathlib import Path
import json
import time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from ripser import ripser
from scipy import stats

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis" / "cycles_partD"))

from dp_cycles_lib import load_neural, load_behavior, K_PCS, bh_correct

N_LANDMARKS = 3000
N_TOP = 5
K_NN = 20
WIN_SEC = 60.0
BIN_S = 0.05
N_PERM = 1000
MIN_BINS = 20

datdir = REPO / "data" / "h1_behavior_partF"
datdir.mkdir(parents=True, exist_ok=True)


def circ_shift_per_unit(matrix, rng):
    out = np.empty_like(matrix)
    n = matrix.shape[0]
    for j in range(matrix.shape[1]):
        k = int(rng.integers(low=n // 10, high=n - n // 10))
        out[:, j] = np.roll(matrix[:, j], k)
    return out


def temporal_test(supporting_bins, n_total_bins):
    bins_per_win = int(WIN_SEC / BIN_S)
    n_win = int(np.ceil(n_total_bins / bins_per_win))
    if n_win < 2:
        return np.nan, np.nan, np.nan
    obs = np.zeros(n_win, dtype=int)
    win_idx = np.clip(supporting_bins // bins_per_win, 0, n_win - 1)
    for w in win_idx:
        obs[w] += 1
    expected = np.full(n_win, fill_value=len(supporting_bins) / n_win)
    chi2 = float(np.sum((obs - expected) ** 2 / np.maximum(expected, 1e-9)))
    p = float(stats.chi2.sf(chi2, df=n_win - 1))
    concentration = float(obs.max() / max(expected[0], 1e-9))
    return chi2, p, concentration


def cat_chi2(values, supporting_mask, classes):
    out = {}
    n_total = len(values)
    n_sup = supporting_mask.sum()
    n_rest = n_total - n_sup
    for c in classes:
        is_c = (values == c)
        a = int((is_c & supporting_mask).sum())
        b = int((is_c & ~supporting_mask).sum())
        ctab = np.array([[a, n_sup - a], [b, n_rest - b]])
        if ctab.min() < 0:
            continue
        try:
            chi2, p, _, _ = stats.chi2_contingency(ctab)
        except Exception:
            chi2, p = np.nan, np.nan
        lfc = float(np.log2((a + 1) / (n_sup + 1) /
                             max((b + 1) / (n_rest + 1), 1e-12)))
        out[str(c)] = dict(p=float(p), log2_fc=lfc)
    return out


def cont_perm_test(values, supporting_mask, n_perm=N_PERM, seed=0):
    v = values[np.isfinite(values)]
    sup_mask_v = supporting_mask[np.isfinite(values)]
    if v.size < 10 or sup_mask_v.sum() < 2:
        return np.nan, np.nan
    m_sup = float(v[sup_mask_v].mean())
    m_rest = float(v[~sup_mask_v].mean())
    pooled_sd = float(v.std())
    d = (m_sup - m_rest) / pooled_sd if pooled_sd else np.nan
    obs = abs(m_sup - m_rest)
    rng = np.random.default_rng(seed)
    n_sup = int(sup_mask_v.sum())
    null = np.empty(n_perm)
    for i in range(n_perm):
        idx = rng.choice(v.size, size=n_sup, replace=False)
        rest_idx = np.setdiff1d(np.arange(v.size), idx, assume_unique=False)
        null[i] = abs(v[idx].mean() - v[rest_idx].mean())
    p = float((null >= obs).mean())
    return float(d), p


def run_one(matrix, K, behav, compartment_classes, binary_names, cont_vars,
            label='real'):
    pca = PCA(n_components=K)
    X = pca.fit_transform(matrix)
    res = ripser(X, maxdim=1, n_perm=N_LANDMARKS, do_cocycles=True)
    dgm = res['dgms'][1]
    cocs = res['cocycles'][1]
    finite = np.isfinite(dgm[:, 1])
    dgm_f = dgm[finite]
    cocs_f = [cocs[i] for i in range(len(dgm)) if finite[i]]
    persistence = dgm_f[:, 1] - dgm_f[:, 0]
    order = np.argsort(persistence)[::-1]
    top = order[:N_TOP]

    nn = NearestNeighbors(n_neighbors=K_NN + 1)
    nn.fit(X)
    n_total = matrix.shape[0]

    feat_results = []
    for rk_idx, idx in enumerate(top):
        coc = cocs_f[idx]
        if coc.shape[0] == 0:
            continue
        verts = np.unique(coc[:, :2].astype(int).ravel())
        _, knn_idx = nn.kneighbors(X[verts])
        supporting = np.unique(knn_idx.ravel())
        n_sup = len(supporting)
        sup_mask = np.zeros(n_total, dtype=bool)
        sup_mask[supporting] = True

        tchi2, tp, tconc = temporal_test(supporting, n_total)

        # Behavioral tests
        p_pool = []
        details = []
        if n_sup >= MIN_BINS:
            comp_results = cat_chi2(behav['compartment']['values'], sup_mask,
                                     compartment_classes)
            for c, info in comp_results.items():
                if np.isfinite(info['p']):
                    p_pool.append(info['p'])
                    details.append(('compartment', c, info['log2_fc']))
            for bn in binary_names:
                vals_str = behav[bn]['values']
                bin_results = cat_chi2(vals_str, sup_mask, ['1'])
                for c, info in bin_results.items():
                    if np.isfinite(info['p']):
                        p_pool.append(info['p'])
                        details.append((bn, c, info['log2_fc']))
            for cv in cont_vars:
                vals = behav[cv]['values'].astype(float)
                d, p = cont_perm_test(vals, sup_mask, seed=rk_idx)
                if np.isfinite(p):
                    p_pool.append(p)
                    details.append((cv, None, d))

        n_sig = 0
        min_q = np.nan
        if p_pool:
            qs = bh_correct(p_pool)
            n_sig = int((qs < 0.05).sum())
            min_q = float(qs.min())

        feat_results.append(dict(
            label=label, feature_rank=rk_idx,
            persistence=float(persistence[idx]),
            n_supporting=n_sup,
            temporal_p=float(tp), temporal_concentration=float(tconc),
            n_behaviors_q05=n_sig,
            min_q=min_q,
        ))
    return feat_results


def main():
    if len(sys.argv) < 3:
        print("Usage: python partF_shuffle_null.py <session_num> <region> [n_shuffles]")
        sys.exit(1)
    session_num = int(sys.argv[1])
    region = sys.argv[2]
    n_shuffles = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    print(f"\n=== Part F shuffle null: S{session_num} {region}, n={n_shuffles} ===")
    t0 = time.time()
    matrix, bin_centers, n_units = load_neural(session_num, region)
    K = K_PCS[region]

    behav = load_behavior(session_num, bin_centers)
    compartment_classes = behav['compartment']['classes']
    binary_names = [k for k, v in behav.items()
                    if v.get('type') == 'categorical' and k != 'compartment']
    cont_vars = ['velocity']
    cont_vars += [k for k in behav.keys() if k.startswith('dist_pot')]

    all_results = []

    # Real (re-run for direct comparison)
    print("\n[real]")
    real = run_one(matrix, K, behav, compartment_classes, binary_names,
                   cont_vars, label='real')
    for r in real:
        print(f"  feat#{r['feature_rank']}: pers={r['persistence']:.3f} "
              f"n_sup={r['n_supporting']} temp_p={r['temporal_p']:.2g} "
              f"n_q<0.05={r['n_behaviors_q05']} min_q={r['min_q']:.2g}")
    all_results.extend(real)

    # Shuffles
    for s in range(n_shuffles):
        rng = np.random.default_rng(2000 + s)
        shuf = circ_shift_per_unit(matrix, rng)
        label = f'null{s}'
        print(f"\n[{label}]")
        ts = time.time()
        nulls = run_one(shuf, K, behav, compartment_classes, binary_names,
                        cont_vars, label=label)
        for r in nulls:
            print(f"  feat#{r['feature_rank']}: pers={r['persistence']:.3f} "
                  f"n_sup={r['n_supporting']} temp_p={r['temporal_p']:.2g} "
                  f"n_q<0.05={r['n_behaviors_q05']} min_q={r['min_q']:.2g}")
        all_results.extend(nulls)
        print(f"  shuffle time: {(time.time()-ts):.1f}s")

    # Save
    out_json = datdir / f"S{session_num}_{region}_shuffle_null.json"
    with open(out_json, 'w') as f:
        json.dump(dict(session=session_num, region=region,
                       n_shuffles=n_shuffles,
                       results=all_results), f, indent=2)
    print(f"\nSaved {out_json}")
    print(f"Total time: {(time.time()-t0)/60:.1f} min")

    # Summary: real vs shuffle
    print("\nSummary (real top features vs shuffle top features):")
    print(f"  N real features: {len([r for r in all_results if r['label']=='real'])}")
    print(f"  N null features: {len([r for r in all_results if r['label']!='real'])}")
    real_pers = [r['persistence'] for r in all_results if r['label'] == 'real']
    null_pers = [r['persistence'] for r in all_results if r['label'] != 'real']
    real_nsig = [r['n_behaviors_q05'] for r in all_results if r['label'] == 'real']
    null_nsig = [r['n_behaviors_q05'] for r in all_results if r['label'] != 'real']
    print(f"  persistence: real {np.mean(real_pers):.3f}±{np.std(real_pers):.3f} "
          f"vs null {np.mean(null_pers):.3f}±{np.std(null_pers):.3f}")
    print(f"  n_behaviors with q<0.05: real {np.mean(real_nsig):.1f}±{np.std(real_nsig):.1f} "
          f"vs null {np.mean(null_nsig):.1f}±{np.std(null_nsig):.1f}")


if __name__ == '__main__':
    main()
