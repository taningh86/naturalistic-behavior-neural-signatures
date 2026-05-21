"""
Part F Steps 2-5: from cocycle representatives -> behavioral mapping + classification.

Step 2: For each top-N H1 feature, expand the cocycle landmark set by k-NN in PCA space
        (k=K_NN per landmark, default 20). Union the neighborhoods to form the supporting
        time-bin set per feature.
Step 3: Temporal structure analysis on supporting bins.
        Bin session into WIN_SEC-second windows (default 60s). Count supporting bins per
        window. Test against uniform via chi-square. Compute Gini-like temporal
        concentration: max-window-count / (n_bins * frac_uniform).
Step 4: Behavioral content analysis.
        For categorical (compartment, scored binaries): chi-square 2x2 [supporting vs rest]
        x [class==c vs other]. Effect: log-fold-change ratio (supporting-rate / baseline).
        For continuous (velocity, dist_pots): permutation test on mean.
        Within each feature, BH-correct across all behavioral tests.
Step 5: Per-feature classification:
        A: temporal_p < 0.05 AND any behavior_q < 0.05  (clustered + coherent)
        B: temporal_p >= 0.05 AND any behavior_q < 0.05 (scattered + coherent)
        C: no behavior_q < 0.05                          (no coherence)

Outputs (per region/session):
    data/h1_behavior_partF/S{N}_{region}_step25.json
    data/h1_behavior_partF/S{N}_{region}_step25_summary.md
    figures/h1_behavior_partF/S{N}_{region}_feature{rk}_temporal.png
    figures/h1_behavior_partF/S{N}_{region}_feature{rk}_behavior.png

Usage: python partF_step25.py <session_num> <region> [<step1_json>]
"""
import sys
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy import stats

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis" / "cycles_partD"))

from dp_cycles_lib import load_neural, load_behavior, K_PCS, bh_correct

K_NN = 20
WIN_SEC = 60.0
BIN_S = 0.05
N_PERM = 1000
MIN_BINS = 20
SEED = 42

datdir = REPO / "data" / "h1_behavior_partF"
figdir = REPO / "figures" / "h1_behavior_partF"
datdir.mkdir(parents=True, exist_ok=True)
figdir.mkdir(parents=True, exist_ok=True)


def expand_neighborhood(X, landmark_idx, k=K_NN):
    """Union k-NN neighborhoods (in PCA space) for the given landmark indices."""
    if len(landmark_idx) == 0:
        return np.array([], dtype=int)
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(X)
    pts = X[landmark_idx]
    _, idx = nn.kneighbors(pts)
    return np.unique(idx.ravel())


def temporal_test(supporting_bins, n_total_bins):
    """Chi-square: supporting bins distributed uniformly over WIN_SEC windows?"""
    bins_per_win = int(WIN_SEC / BIN_S)
    n_win = int(np.ceil(n_total_bins / bins_per_win))
    if n_win < 2:
        return np.nan, np.nan, np.nan
    obs = np.zeros(n_win, dtype=int)
    win_idx = np.clip(supporting_bins // bins_per_win, 0, n_win - 1)
    for w in win_idx:
        obs[w] += 1
    expected = np.full(n_win, fill_value=len(supporting_bins) / n_win)
    if np.any(expected < 1):
        # combine sparse tails by using G-test approximation; expected too low
        pass
    chi2 = float(np.sum((obs - expected) ** 2 / np.maximum(expected, 1e-9)))
    p = float(stats.chi2.sf(chi2, df=n_win - 1))
    # Concentration: max window count vs uniform expectation
    concentration = float(obs.max() / max(expected[0], 1e-9))
    return chi2, p, concentration


def cat_chi2_logfc(values, supporting_mask, classes):
    """Chi-square 2x2 [supporting vs rest] x [class==c vs other], for each class.
    Returns dict of {class -> (chi2, p, log2_fold_change)}.
    """
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
        rate_sup = a / n_sup if n_sup else 0
        rate_rest = b / n_rest if n_rest else 0
        # log2 fold change with small-sample bias correction
        lfc = float(np.log2((a + 1) / (n_sup + 1) / max((b + 1) / (n_rest + 1), 1e-12)))
        out[str(c)] = dict(chi2=float(chi2), p=float(p),
                            rate_sup=float(rate_sup),
                            rate_rest=float(rate_rest),
                            log2_fc=lfc, n_in_sup=a, n_in_rest=b)
    return out


def cont_perm_test(values, supporting_mask, n_perm=N_PERM):
    """Permutation test: mean(supporting) vs mean(rest). Returns (mean_sup, mean_rest, d, p)."""
    v = values[np.isfinite(values)]
    sup_mask_v = supporting_mask[np.isfinite(values)]
    if v.size < 10 or sup_mask_v.sum() < 2:
        return np.nan, np.nan, np.nan, np.nan
    m_sup = float(v[sup_mask_v].mean())
    m_rest = float(v[~sup_mask_v].mean())
    pooled_sd = float(v.std())
    d = (m_sup - m_rest) / pooled_sd if pooled_sd else np.nan
    obs = abs(m_sup - m_rest)
    rng = np.random.default_rng(SEED)
    n_sup = int(sup_mask_v.sum())
    null = np.empty(n_perm)
    for i in range(n_perm):
        idx = rng.choice(v.size, size=n_sup, replace=False)
        m_a = v[idx].mean()
        rest_idx = np.setdiff1d(np.arange(v.size), idx, assume_unique=False)
        m_b = v[rest_idx].mean()
        null[i] = abs(m_a - m_b)
    p = float((null >= obs).mean())
    return m_sup, m_rest, float(d), p


def plot_temporal(supporting_bins, n_total_bins, title, out_path):
    bins_per_win = int(WIN_SEC / BIN_S)
    n_win = int(np.ceil(n_total_bins / bins_per_win))
    obs = np.zeros(n_win, dtype=int)
    win_idx = np.clip(supporting_bins // bins_per_win, 0, n_win - 1)
    for w in win_idx:
        obs[w] += 1
    expected = len(supporting_bins) / n_win
    fig, ax = plt.subplots(figsize=(9, 3))
    win_t = np.arange(n_win) * WIN_SEC / 60.0
    ax.bar(win_t, obs, width=WIN_SEC / 60.0 * 0.95, color='steelblue',
           edgecolor='navy', linewidth=0.5)
    ax.axhline(expected, color='red', linestyle='--', linewidth=1,
               label=f'uniform expectation ({expected:.1f})')
    ax.set_xlabel('Session time (min)')
    ax.set_ylabel(f'#supporting bins per {int(WIN_SEC)}s window')
    ax.set_title(title)
    ax.legend(fontsize=8, loc='upper right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_behavior(rows, title, out_path):
    """rows: list of (label, value_or_lfc, q, kind). kind in {'lfc', 'd'}."""
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(9, max(2.5, 0.35 * len(rows))))
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    qs = [r[2] for r in rows]
    colors = ['firebrick' if q < 0.05 else 'lightgrey' for q in qs]
    bars = ax.barh(np.arange(len(rows)), vals, color=colors,
                   edgecolor='black', linewidth=0.4)
    for i, (v, q) in enumerate(zip(vals, qs)):
        ax.text(v + (0.02 * np.sign(v) if v != 0 else 0.02), i,
                f"q={q:.2g}", va='center', fontsize=8)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color='black', linewidth=0.4)
    ax.set_xlabel('log2 FC (categorical) or Cohen d (continuous)')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def run(session_num, region, step1_json=None):
    print(f"\n{'='*70}")
    print(f"Part F Steps 2-5: S{session_num} {region}")
    print(f"{'='*70}")
    if step1_json is None:
        step1_json = datdir / f"S{session_num}_{region}_step1.json"
    with open(step1_json) as f:
        s1 = json.load(f)
    feats = s1['features']
    K = s1['pca_K']
    print(f"Loaded Step 1: {len(feats)} features, top persistences "
          f"{[round(p, 3) for p in s1['top_persistences']]}")

    matrix, bin_centers, n_units = load_neural(session_num, region)
    pca = PCA(n_components=K)
    X = pca.fit_transform(matrix)
    n_total = matrix.shape[0]
    print(f"PCA shape {X.shape}; {n_total} total bins")

    # Behaviors
    behav = load_behavior(session_num, bin_centers)
    compartment = behav['compartment']['values']
    compartment_classes = behav['compartment']['classes']
    binary_names = [k for k, v in behav.items()
                    if v.get('type') == 'categorical' and k != 'compartment']
    cont_vars = ['velocity']
    cont_vars += [k for k in behav.keys() if k.startswith('dist_pot')]
    cont_vars += [k for k in ('dist_home', 'dist_ladder') if k in behav]

    nn = NearestNeighbors(n_neighbors=K_NN + 1)
    nn.fit(X)

    feature_results = []
    for feat in feats:
        rk = feat['feature_id']
        landmark_bins = np.array(feat['landmark_bins'], dtype=int)
        n_lm = len(landmark_bins)
        if n_lm < 1:
            print(f"  Feature {rk}: empty landmarks; skipping")
            continue
        _, knn_idx = nn.kneighbors(X[landmark_bins])
        supporting = np.unique(knn_idx.ravel())
        n_sup = len(supporting)
        print(f"\nFeature {rk}: pers={feat['persistence']:.3f}, "
              f"n_landmarks={n_lm} -> n_supporting={n_sup}")
        if n_sup < MIN_BINS:
            print(f"  TOO FEW supporting bins (<{MIN_BINS}); skipping behavior tests")
        sup_mask = np.zeros(n_total, dtype=bool)
        sup_mask[supporting] = True

        # Step 3: temporal
        tchi2, tp, tconc = temporal_test(supporting, n_total)
        print(f"  temporal: chi2={tchi2:.1f} p={tp:.3g} "
              f"max_window/uniform={tconc:.1f}x")

        # Step 4: behavioral tests
        cat_results = {}  # categorical
        cont_results = {}  # continuous
        if n_sup >= MIN_BINS:
            # compartment
            cat_results['compartment'] = cat_chi2_logfc(compartment, sup_mask, compartment_classes)
            # binary scored
            for bn in binary_names:
                vals_str = behav[bn]['values']
                cat_results[bn] = cat_chi2_logfc(vals_str, sup_mask, ['1'])
            # continuous
            for cv in cont_vars:
                vals = behav[cv]['values'].astype(float)
                m_s, m_r, d, p = cont_perm_test(vals, sup_mask)
                cont_results[cv] = dict(mean_sup=m_s, mean_rest=m_r,
                                         cohens_d=d, p=p)

            # Collect all p-values for BH correction
            p_pool = []
            label_pool = []
            for bn, classes in cat_results.items():
                for c, info in classes.items():
                    if np.isfinite(info['p']):
                        label_pool.append((bn, c, 'cat'))
                        p_pool.append(info['p'])
            for cv, info in cont_results.items():
                if np.isfinite(info['p']):
                    label_pool.append((cv, None, 'cont'))
                    p_pool.append(info['p'])
            if p_pool:
                qs = bh_correct(p_pool)
                for (bn, c, kind), q in zip(label_pool, qs):
                    if kind == 'cat':
                        cat_results[bn][c]['q'] = float(q)
                    else:
                        cont_results[bn]['q'] = float(q)

        # Step 5: classification
        any_q_sig = False
        for bn, classes in cat_results.items():
            for c, info in classes.items():
                if info.get('q', 1.0) < 0.05:
                    any_q_sig = True
        for cv, info in cont_results.items():
            if info.get('q', 1.0) < 0.05:
                any_q_sig = True
        if any_q_sig and tp < 0.05:
            cls = 'A'
        elif any_q_sig and tp >= 0.05:
            cls = 'B'
        else:
            cls = 'C'

        feature_results.append(dict(
            feature_id=rk,
            persistence=feat['persistence'],
            birth=feat['birth'],
            death=feat['death'],
            n_landmarks=n_lm,
            n_supporting=n_sup,
            temporal_chi2=tchi2,
            temporal_p=tp,
            temporal_concentration=tconc,
            categorical=cat_results,
            continuous=cont_results,
            classification=cls,
        ))

        # Plots (only when sup is large enough)
        if n_sup >= MIN_BINS:
            ftemp = figdir / f"S{session_num}_{region}_feature{rk}_temporal.png"
            plot_temporal(supporting, n_total,
                          title=f"S{session_num} {region} feat#{rk} pers={feat['persistence']:.2f} "
                                f"({n_sup} bins, p={tp:.2g}, conc={tconc:.1f}x)",
                          out_path=ftemp)

            # Build top-significant rows for behavior plot (max 12 by |effect|)
            rows = []
            for bn, classes in cat_results.items():
                for c, info in classes.items():
                    if np.isfinite(info.get('q', np.nan)):
                        rows.append((f"{bn}={c}", info['log2_fc'], info['q']))
            for cv, info in cont_results.items():
                if np.isfinite(info.get('q', np.nan)):
                    rows.append((cv, info['cohens_d'], info['q']))
            rows.sort(key=lambda r: -abs(r[1]))
            rows = rows[:12]
            fbeh = figdir / f"S{session_num}_{region}_feature{rk}_behavior.png"
            plot_behavior(rows,
                          title=f"S{session_num} {region} feat#{rk} (class {cls}) — top behavioral effects",
                          out_path=fbeh)

    out = dict(
        session=session_num, region=region,
        K_NN=K_NN, win_sec=WIN_SEC, n_perm=N_PERM, min_bins=MIN_BINS,
        n_total_bins=n_total, n_units=n_units,
        features=feature_results,
    )
    out_json = datdir / f"S{session_num}_{region}_step25.json"
    with open(out_json, 'w') as f:
        json.dump(out, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))
    print(f"\nSaved {out_json}")

    # Markdown summary
    md_path = datdir / f"S{session_num}_{region}_step25_summary.md"
    with open(md_path, 'w') as f:
        f.write(f"# S{session_num} {region} — H1 behavioral mapping\n\n")
        f.write(f"Top {len(feature_results)} H1 features. K_NN={K_NN}, MIN_BINS={MIN_BINS}, "
                f"win={WIN_SEC}s.\n\n")
        f.write("| feat | pers | n_landmarks | n_supporting | temporal_p | concentration | class | top behavioral effects (q<0.05) |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for fr in feature_results:
            sig_rows = []
            for bn, classes in fr['categorical'].items():
                for c, info in classes.items():
                    if info.get('q', 1.0) < 0.05:
                        sig_rows.append((f"{bn}={c}", info['log2_fc'], info['q']))
            for cv, info in fr['continuous'].items():
                if info.get('q', 1.0) < 0.05:
                    sig_rows.append((cv, info['cohens_d'], info['q']))
            sig_rows.sort(key=lambda r: -abs(r[1]))
            sig_str = '; '.join(f"{r[0]} (eff={r[1]:.2f}, q={r[2]:.2g})" for r in sig_rows[:6])
            if not sig_str:
                sig_str = '(none)'
            f.write(f"| {fr['feature_id']} | {fr['persistence']:.3f} | "
                    f"{fr['n_landmarks']} | {fr['n_supporting']} | "
                    f"{fr['temporal_p']:.3g} | {fr['temporal_concentration']:.2f}x | "
                    f"{fr['classification']} | {sig_str} |\n")
        f.write("\n## Honest assessment\n\n")
        cls_counts = {'A': 0, 'B': 0, 'C': 0}
        for fr in feature_results:
            cls_counts[fr['classification']] += 1
        f.write(f"- Class A (clustered + coherent): {cls_counts['A']}\n")
        f.write(f"- Class B (scattered + coherent): {cls_counts['B']}\n")
        f.write(f"- Class C (no behavioral coherence): {cls_counts['C']}\n")
        too_small = [fr for fr in feature_results if fr['n_supporting'] < MIN_BINS]
        if too_small:
            f.write(f"\nFeatures skipped due to <{MIN_BINS} supporting bins: "
                    f"{[fr['feature_id'] for fr in too_small]}\n")
    print(f"Saved {md_path}")
    return out


def main():
    if len(sys.argv) < 3:
        print("Usage: python partF_step25.py <session_num> <region> [step1_json]")
        sys.exit(1)
    session_num = int(sys.argv[1])
    region = sys.argv[2]
    step1_json = sys.argv[3] if len(sys.argv) > 3 else None
    run(session_num, region, step1_json)


if __name__ == '__main__':
    main()
