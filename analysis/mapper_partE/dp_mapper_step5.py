"""
Step 5: subsample robustness.

For each (S3/S4 x ACA/LHA) re-run the full pipeline (PCA -> PC1 lens -> Mapper)
on N_SUB random 80% subsamples of time bins. Report mean +/- std of graph
descriptors and compare to the primary run.

Why subsample (not seed): PCA + DBSCAN + km.Cover are deterministic given the
data; the only thing that perturbs the graph in a meaningful way is which bins
are present. 80% is enough to perturb while keeping comparable density.

Outputs:
    data/mapper_partE/step5_robustness.csv
    figures/mapper_partE/step5_descriptor_stability.png
"""
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis" / "cycles_partD"))
sys.path.insert(0, str(REPO / "analysis" / "mapper_partE"))

from dp_cycles_lib import load_neural, K_PCS
from dp_mapper_lib import build_mapper

LENS = 'pc1'
N_SUB = 5
FRAC = 0.8
SESSIONS = [3, 4]
REGIONS = ['ACA', 'LHA']

outdir = REPO / "data" / "mapper_partE"
figdir = REPO / "figures" / "mapper_partE"
outdir.mkdir(parents=True, exist_ok=True)
figdir.mkdir(parents=True, exist_ok=True)

DESCRIPTOR_KEYS = ['n_nodes', 'n_edges', 'n_components', 'n_cycles',
                   'n_branching', 'n_endpoints', 'diameter', 'max_node_size']


def main():
    rows = []
    for sess in SESSIONS:
        for region in REGIONS:
            print(f"\n=== S{sess} {region} ===")
            matrix, _, _ = load_neural(sess, region)
            K = K_PCS[region]

            # primary
            G0, d0, _, _, _ = build_mapper(matrix, K, lens_name=LENS)
            d0['session'] = sess
            d0['region'] = region
            d0['run'] = 'primary'
            print(f"  primary: nodes={d0['n_nodes']} edges={d0['n_edges']} "
                  f"comps={d0['n_components']} cycles={d0['n_cycles']}")
            rows.append(d0)

            # subsamples
            n_total = matrix.shape[0]
            n_keep = int(n_total * FRAC)
            for s in range(N_SUB):
                rng = np.random.default_rng(100 + s)
                idx = np.sort(rng.choice(n_total, size=n_keep, replace=False))
                sub = matrix[idx]
                G, d, _, _, _ = build_mapper(sub, K, lens_name=LENS)
                d['session'] = sess
                d['region'] = region
                d['run'] = f'sub{s}'
                print(f"  sub{s}: nodes={d['n_nodes']} edges={d['n_edges']} "
                      f"comps={d['n_components']} cycles={d['n_cycles']}")
                rows.append(d)

    df = pd.DataFrame(rows)
    out_csv = outdir / "step5_robustness.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")

    # summary: mean +/- std per (session, region) over the 5 subsamples
    sub = df[df['run'].str.startswith('sub')]
    summary = sub.groupby(['session', 'region'])[DESCRIPTOR_KEYS].agg(['mean', 'std'])
    print("\nSubsample mean +/- std (n=5):")
    print(summary)
    primary = df[df['run'] == 'primary'].set_index(['session', 'region'])[DESCRIPTOR_KEYS]
    print("\nPrimary:")
    print(primary)

    # Plot
    fig, axs = plt.subplots(2, 4, figsize=(16, 7), sharey=False)
    keys_to_plot = ['n_nodes', 'n_edges', 'n_components', 'n_cycles',
                    'n_branching', 'n_endpoints', 'diameter', 'max_node_size']
    cond = [(s, r) for s in SESSIONS for r in REGIONS]
    cond_labels = [f"S{s}-{r}" for s, r in cond]
    x = np.arange(len(cond))
    for ax, key in zip(axs.flat, keys_to_plot):
        means = []
        stds = []
        prims = []
        for s, r in cond:
            sub_vals = sub[(sub.session == s) & (sub.region == r)][key].values
            means.append(np.nanmean(sub_vals))
            stds.append(np.nanstd(sub_vals))
            prims.append(primary.loc[(s, r), key])
        ax.bar(x, means, yerr=stds, capsize=4, color='lightsteelblue',
               edgecolor='steelblue', label='subsamples')
        ax.scatter(x, prims, color='red', zorder=5, label='primary', marker='D', s=40)
        ax.set_xticks(x)
        ax.set_xticklabels(cond_labels, rotation=30)
        ax.set_title(key, fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    axs.flat[0].legend(loc='upper right', fontsize=8)
    plt.suptitle(f"Step 5 robustness: {N_SUB} subsamples at {int(FRAC*100)}% retention", y=1.02)
    plt.tight_layout()
    fig_path = figdir / "step5_descriptor_stability.png"
    plt.savefig(fig_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved {fig_path}")


if __name__ == '__main__':
    main()
