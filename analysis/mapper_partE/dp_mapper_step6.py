"""
Step 6: shuffle null.

For each (S3/S4 x ACA/LHA), generate N_NULL surrogate datasets by independently
circular-shifting each unit's z-scored activity, then run the full pipeline
(PCA -> PC1 lens -> Mapper). This destroys cross-unit correlations while
preserving each unit's marginal distribution and autocorrelation.

Compares descriptors of the real graph against the shuffle distribution. Reports
z-score and one-sided p (real >= null) for n_cycles, n_branching, n_endpoints,
n_nodes, diameter.

Outputs:
    data/mapper_partE/step6_shuffle_null.csv
    figures/mapper_partE/step6_shuffle_null.png
"""
import sys
from pathlib import Path
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
N_NULL = 5
SESSIONS = [3, 4]
REGIONS = ['ACA', 'LHA']
DESCRIPTOR_KEYS = ['n_nodes', 'n_edges', 'n_components', 'n_cycles',
                   'n_branching', 'n_endpoints', 'diameter', 'max_node_size']

outdir = REPO / "data" / "mapper_partE"
figdir = REPO / "figures" / "mapper_partE"
outdir.mkdir(parents=True, exist_ok=True)
figdir.mkdir(parents=True, exist_ok=True)


def circ_shift_per_unit(matrix, rng):
    """Independently circular-shift each column (unit) by a random offset."""
    out = np.empty_like(matrix)
    n = matrix.shape[0]
    for j in range(matrix.shape[1]):
        k = int(rng.integers(low=n // 10, high=n - n // 10))
        out[:, j] = np.roll(matrix[:, j], k)
    return out


def main():
    rows = []
    for sess in SESSIONS:
        for region in REGIONS:
            print(f"\n=== S{sess} {region} ===")
            matrix, _, _ = load_neural(sess, region)
            K = K_PCS[region]

            # real
            G0, d0, _, _, _ = build_mapper(matrix, K, lens_name=LENS)
            d0['session'] = sess
            d0['region'] = region
            d0['run'] = 'real'
            print(f"  real: nodes={d0['n_nodes']} edges={d0['n_edges']} "
                  f"comps={d0['n_components']} cycles={d0['n_cycles']} "
                  f"branch={d0['n_branching']}")
            rows.append(d0)

            # nulls
            for s in range(N_NULL):
                rng = np.random.default_rng(2000 + s)
                shuf = circ_shift_per_unit(matrix, rng)
                _, d, _, _, _ = build_mapper(shuf, K, lens_name=LENS)
                d['session'] = sess
                d['region'] = region
                d['run'] = f'null{s}'
                print(f"  null{s}: nodes={d['n_nodes']} edges={d['n_edges']} "
                      f"comps={d['n_components']} cycles={d['n_cycles']} "
                      f"branch={d['n_branching']}")
                rows.append(d)

    df = pd.DataFrame(rows)
    out_csv = outdir / "step6_shuffle_null.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")

    # Stats: per (session, region, descriptor) compute z-score and p(real >= null)
    summary_rows = []
    null = df[df['run'].str.startswith('null')]
    real = df[df['run'] == 'real'].set_index(['session', 'region'])
    for (s, r), grp in null.groupby(['session', 'region']):
        for key in DESCRIPTOR_KEYS:
            vals = grp[key].astype(float).values
            real_val = float(real.loc[(s, r), key])
            mu, sd = float(np.nanmean(vals)), float(np.nanstd(vals))
            z = (real_val - mu) / sd if sd > 0 else np.nan
            p_high = (np.sum(vals >= real_val) + 1) / (len(vals) + 1)
            p_low = (np.sum(vals <= real_val) + 1) / (len(vals) + 1)
            summary_rows.append(dict(session=s, region=r, descriptor=key,
                                     real=real_val, null_mean=mu, null_std=sd,
                                     z=z, p_real_ge_null=p_high,
                                     p_real_le_null=p_low))
    summary = pd.DataFrame(summary_rows)
    out_summary = outdir / "step6_shuffle_null_summary.csv"
    summary.to_csv(out_summary, index=False)
    print(f"Saved {out_summary}")
    print("\nReal vs null (key descriptors):")
    pivot = summary.pivot_table(
        index=['session', 'region'],
        columns='descriptor',
        values=['real', 'null_mean', 'z']
    )
    print(pivot[['real', 'null_mean', 'z']].to_string())

    # Plot bar charts: real (red) vs mean+/-std null (blue) per condition
    keys_to_plot = ['n_nodes', 'n_edges', 'n_components', 'n_cycles',
                    'n_branching', 'n_endpoints', 'diameter', 'max_node_size']
    fig, axs = plt.subplots(2, 4, figsize=(16, 7))
    cond = [(s, r) for s in SESSIONS for r in REGIONS]
    cond_labels = [f"S{s}-{r}" for s, r in cond]
    x = np.arange(len(cond))
    for ax, key in zip(axs.flat, keys_to_plot):
        null_means = []
        null_stds = []
        reals = []
        for s, r in cond:
            row = summary[(summary.session == s) & (summary.region == r)
                          & (summary.descriptor == key)].iloc[0]
            null_means.append(row['null_mean'])
            null_stds.append(row['null_std'])
            reals.append(row['real'])
        ax.bar(x - 0.2, null_means, width=0.4, yerr=null_stds, capsize=4,
               color='lightsteelblue', edgecolor='steelblue', label='null')
        ax.bar(x + 0.2, reals, width=0.4, color='salmon',
               edgecolor='firebrick', label='real')
        ax.set_xticks(x)
        ax.set_xticklabels(cond_labels, rotation=30)
        ax.set_title(key, fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    axs.flat[0].legend(loc='upper right', fontsize=8)
    plt.suptitle(f"Step 6: real vs per-unit circular-shift null (n={N_NULL})", y=1.02)
    plt.tight_layout()
    fig_path = figdir / "step6_shuffle_null.png"
    plt.savefig(fig_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved {fig_path}")


if __name__ == '__main__':
    main()
