"""
Generate a consolidated key findings table from all foraging neural analyses.
Reads from data/foraging_neural_all_sessions_metrics.csv to include ALL sessions.
Saves as CSV and as a formatted figure.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

print("=" * 70)
print("  Consolidating Key Findings (All Sessions)")
print("=" * 70)

# =========================================================================
# Load the metrics CSV (produced by foraging_neural_all_sessions.py)
# =========================================================================
metrics = pd.read_csv('data/foraging_neural_all_sessions_metrics.csv')
metrics = metrics.set_index('session')

# =========================================================================
# 1. Behavioral findings table (hardcoded -- comes from behavioral extraction)
# =========================================================================
behav_rows = [
    {'Session': 'S2', 'State': 'Fed', 'Discovery (s)': 767.2, 'First Dig (s)': 767.5,
     'Dig Pot': 'Pot-4', 'Pre-disc visits': 32, 'Pre P2': 6, 'Pre P4': 16,
     'Post P2': 9, 'Post P4': 20, 'P2-P4 trans': 5, 'Return to P4 (s)': 60.0},
    {'Session': 'S4', 'State': 'Fed', 'Discovery (s)': 936.8, 'First Dig (s)': 782.8,
     'Dig Pot': 'Pot-2!', 'Pre-disc visits': 39, 'Pre P2': 17, 'Pre P4': 6,
     'Post P2': 17, 'Post P4': 13, 'P2-P4 trans': 12, 'Return to P4 (s)': 31.2},
    {'Session': 'S6', 'State': 'Fasted', 'Discovery (s)': 33.7, 'First Dig (s)': 33.1,
     'Dig Pot': 'Pot-4', 'Pre-disc visits': 3, 'Pre P2': 0, 'Pre P4': 2,
     'Post P2': 6, 'Post P4': 23, 'P2-P4 trans': 5, 'Return to P4 (s)': 590.3},
    {'Session': 'S8', 'State': 'Fasted', 'Discovery (s)': 30.2, 'First Dig (s)': 17.0,
     'Dig Pot': 'Pot-4', 'Pre-disc visits': 2, 'Pre P2': 0, 'Pre P4': 1,
     'Post P2': 8, 'Post P4': 56, 'P2-P4 trans': 4, 'Return to P4 (s)': 321.3},
]
behav_df = pd.DataFrame(behav_rows)

# =========================================================================
# 2. Learning curve findings (Analyses 3-4) -- read from metrics CSV
# =========================================================================
def sig_label(p):
    if pd.isna(p): return '--'
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    if p < 0.1: return '(*)'
    return 'ns'

def direction_label(r):
    if pd.isna(r): return '--'
    if abs(r) < 0.15: return '--'
    return 'Increasing' if r > 0 else 'Decreasing'

learning_rows = []
sessions_info = [
    (2, 'Fed', 6, 16),   # (session_num, state, n_p2_pre, n_p4_pre)
    (4, 'Fed', 17, 6),
    (6, 'Fasted', 0, 2),
    (8, 'Fasted', 0, 1),
]

for snum, state, n_p2, n_p4 in sessions_info:
    row = metrics.loc[snum] if snum in metrics.index else None

    for region in ['LHA', 'RSP']:
        for pot, n_visits in [('Pot2', n_p2), ('Pot4', n_p4)]:
            pot_label = pot.replace('Pot', 'Pot-')
            for metric_key, metric_label in [('pop', 'FR'), ('pc1', 'PC1'),
                                              ('flow', 'Flow'), ('gate', 'Gate')]:
                r_col = f'{region}_{pot}_{metric_key}_r'
                p_col = f'{region}_{pot}_{metric_key}_p'

                if n_visits < 3:
                    # Not enough visits for correlation
                    learning_rows.append({
                        'Session': f'S{snum}', 'State': state, 'Region': region,
                        'Pot': pot_label, 'Metric': metric_label,
                        'n': n_visits, 'r': '--', 'p': '--',
                        'Direction': f'n<3', 'Sig': '--'
                    })
                elif row is not None and r_col in row.index and not pd.isna(row[r_col]):
                    r_val = row[r_col]
                    p_val = row[p_col]
                    learning_rows.append({
                        'Session': f'S{snum}', 'State': state, 'Region': region,
                        'Pot': pot_label, 'Metric': metric_label,
                        'n': n_visits, 'r': f'{r_val:.3f}', 'p': f'{p_val:.4f}',
                        'Direction': direction_label(r_val), 'Sig': sig_label(p_val)
                    })
                else:
                    learning_rows.append({
                        'Session': f'S{snum}', 'State': state, 'Region': region,
                        'Pot': pot_label, 'Metric': metric_label,
                        'n': n_visits, 'r': '--', 'p': '--',
                        'Direction': 'no data', 'Sig': '--'
                    })

learning_df = pd.DataFrame(learning_rows)

# =========================================================================
# 3. Pre vs Post-Feed findings (Analysis 9) -- read from metrics CSV
# =========================================================================
prepost_rows = []

for snum, state, n_p2, n_p4 in sessions_info:
    row = metrics.loc[snum] if snum in metrics.index else None
    n_p2_post = behav_rows[[i for i, b in enumerate(behav_rows) if b['Session'] == f'S{snum}'][0]]['Post P2']
    n_p4_post = behav_rows[[i for i, b in enumerate(behav_rows) if b['Session'] == f'S{snum}'][0]]['Post P4']

    for region in ['LHA', 'RSP']:
        for pot, n_pre, n_post in [('Pot2', n_p2, n_p2_post), ('Pot4', n_p4, n_p4_post)]:
            pot_label = pot.replace('Pot', 'Pot-')
            for metric_key, metric_label in [('FR', 'FR'), ('PC1', 'PC1'),
                                              ('Flow', 'Flow'), ('Gate', 'Gate')]:
                u_col = f'{region}_{pot}_{metric_key}_U'
                p_col = f'{region}_{pot}_{metric_key}_p'
                pre_col = f'{region}_{pot}_{metric_key}_pre_med'
                post_col = f'{region}_{pot}_{metric_key}_post_med'

                if n_pre < 1 or n_post < 1:
                    prepost_rows.append({
                        'Session': f'S{snum}', 'State': state, 'Region': region,
                        'Pot': pot_label, 'Metric': metric_label,
                        'n_pre': n_pre, 'n_post': n_post,
                        'Pre median': '--', 'Post median': '--',
                        'p': '--', 'Sig': '--'
                    })
                elif row is not None and p_col in row.index and not pd.isna(row[p_col]):
                    p_val = row[p_col]
                    pre_med = row[pre_col]
                    post_med = row[post_col]
                    prepost_rows.append({
                        'Session': f'S{snum}', 'State': state, 'Region': region,
                        'Pot': pot_label, 'Metric': metric_label,
                        'n_pre': n_pre, 'n_post': n_post,
                        'Pre median': f'{pre_med:.3f}',
                        'Post median': f'{post_med:.3f}',
                        'p': f'{p_val:.4f}', 'Sig': sig_label(p_val)
                    })
                else:
                    prepost_rows.append({
                        'Session': f'S{snum}', 'State': state, 'Region': region,
                        'Pot': pot_label, 'Metric': metric_label,
                        'n_pre': n_pre, 'n_post': n_post,
                        'Pre median': '--', 'Post median': '--',
                        'p': '--', 'Sig': '--'
                    })

prepost_df = pd.DataFrame(prepost_rows)

# =========================================================================
# Save CSVs
# =========================================================================
behav_df.to_csv('data/foraging_key_findings_behavioral.csv', index=False)
learning_df.to_csv('data/foraging_key_findings_learning_curves.csv', index=False)
prepost_df.to_csv('data/foraging_key_findings_pre_vs_post.csv', index=False)
print("  Saved: data/foraging_key_findings_behavioral.csv")
print("  Saved: data/foraging_key_findings_learning_curves.csv")
print("  Saved: data/foraging_key_findings_pre_vs_post.csv")

# =========================================================================
# Helper: filter to show significant + interesting rows for compact display
# =========================================================================
def filter_for_display(df, is_learning=True):
    """Keep significant rows, plus one summary row per session/pot combo that has no data."""
    sig_rows = df[df['Sig'].isin(['*', '**', '***', '(*)'])].copy()
    # Add 'n<3' summary rows (one per session/pot instead of all metric combos)
    nodata_rows = df[df['Sig'] == '--'].copy()
    if len(nodata_rows) > 0:
        # Keep one row per session/pot to show it was tested but insufficient
        summary_nodata = nodata_rows.groupby(['Session', 'Pot']).first().reset_index()
        if is_learning:
            summary_nodata['Metric'] = 'all'
            summary_nodata['Region'] = '--'
        sig_rows = pd.concat([sig_rows, summary_nodata], ignore_index=True)
    # Also add notable ns results from fed sessions for completeness
    ns_fed = df[(df['Sig'] == 'ns') & (df['State'] == 'Fed')].copy()
    sig_rows = pd.concat([sig_rows, ns_fed], ignore_index=True)
    sig_rows = sig_rows.drop_duplicates()
    # Sort by session then significance
    sig_order = {'***': 0, '**': 1, '*': 2, '(*)': 3, 'ns': 4, '--': 5}
    sig_rows['_sort'] = sig_rows['Sig'].map(sig_order).fillna(6)
    sig_rows = sig_rows.sort_values(['Session', '_sort']).drop(columns=['_sort'])
    return sig_rows


# =========================================================================
# Figure: Consolidated tables
# =========================================================================
fig = plt.figure(figsize=(24, 32))
fig.suptitle('Foraging Neural Signatures: Key Findings\nAll Sessions (S2, S4 Fed | S6, S8 Fasted)',
             fontsize=16, fontweight='bold', y=0.99)

# --- Table A: Behavioral Summary ---
ax1 = fig.add_axes([0.02, 0.88, 0.96, 0.09])
ax1.axis('off')
ax1.set_title('A. Behavioral Summary', fontsize=13, fontweight='bold', loc='left', pad=10)
t1 = ax1.table(cellText=behav_df.values, colLabels=behav_df.columns,
               cellLoc='center', loc='center')
t1.auto_set_font_size(False)
t1.set_fontsize(8.5)
t1.auto_set_column_width(list(range(len(behav_df.columns))))
for i in range(len(behav_df)):
    color = '#D6EAF8' if behav_rows[i]['State'] == 'Fed' else '#FADBD8'
    for j in range(len(behav_df.columns)):
        t1[i + 1, j].set_facecolor(color)
for j in range(len(behav_df.columns)):
    t1[0, j].set_facecolor('#D5D8DC')
    t1[0, j].set_text_props(fontweight='bold')

# --- Table B: Learning Curves (ALL sessions) ---
# Show significant + notable results plus n<3 summary rows
learn_display = filter_for_display(learning_df, is_learning=True)
learn_show = learn_display[['Session', 'State', 'Region', 'Pot', 'Metric', 'n', 'r', 'p', 'Direction', 'Sig']].copy()

n_learn = len(learn_show)
learn_height = max(0.04, min(0.30, 0.015 * (n_learn + 1)))
ax2 = fig.add_axes([0.02, 0.88 - learn_height - 0.03, 0.96, learn_height])
ax2.axis('off')
ax2.set_title('B. Across-Excursion Learning Curves (Pre-Discovery Pot Visits) -- All Sessions',
              fontsize=13, fontweight='bold', loc='left', pad=10)

t2 = ax2.table(cellText=learn_show.values, colLabels=learn_show.columns,
               cellLoc='center', loc='center')
t2.auto_set_font_size(False)
t2.set_fontsize(8)
t2.auto_set_column_width(list(range(len(learn_show.columns))))
for j in range(len(learn_show.columns)):
    t2[0, j].set_facecolor('#D5D8DC')
    t2[0, j].set_text_props(fontweight='bold')
for i in range(n_learn):
    sig = learn_show.iloc[i]['Sig']
    state = learn_show.iloc[i]['State']
    if sig in ['*', '**', '***']:
        color = '#ABEBC6'  # green
    elif sig == '(*)':
        color = '#F9E79F'  # yellow
    elif sig == '--':
        color = '#E8DAEF' if state == 'Fasted' else '#F2F3F4'  # light purple for fasted no-data
    else:
        color = 'white'
    for j in range(len(learn_show.columns)):
        t2[i + 1, j].set_facecolor(color)

# --- Table C: Pre vs Post Feed (ALL sessions) ---
# Filter: show significant, trending, and summary rows for no-data
pp_display = prepost_df.copy()
# For display: collapse no-data rows to one per session/pot
pp_sig = pp_display[pp_display['Sig'].isin(['*', '**', '***', '(*)'])].copy()
pp_ns_fed = pp_display[(pp_display['Sig'] == 'ns') & (pp_display['State'] == 'Fed')].copy()
pp_nodata = pp_display[pp_display['Sig'] == '--'].copy()
pp_nodata_summary = pp_nodata.groupby(['Session', 'Pot']).first().reset_index()
pp_nodata_summary['Metric'] = 'all'
pp_nodata_summary['Region'] = '--'
# S6 has actual results for Pot-4 (n_pre=2) -- keep all those
pp_s6_actual = pp_display[(pp_display['Session'] == 'S6') & (pp_display['Sig'] == 'ns')].copy()
pp_combined = pd.concat([pp_sig, pp_ns_fed, pp_s6_actual, pp_nodata_summary], ignore_index=True)
pp_combined = pp_combined.drop_duplicates()
sig_order = {'***': 0, '**': 1, '*': 2, '(*)': 3, 'ns': 4, '--': 5}
pp_combined['_sort'] = pp_combined['Sig'].map(sig_order).fillna(6)
pp_combined = pp_combined.sort_values(['Session', '_sort']).drop(columns=['_sort'])

pp_show = pp_combined[['Session', 'State', 'Region', 'Pot', 'Metric', 'n_pre', 'n_post',
                        'Pre median', 'Post median', 'p', 'Sig']].copy()

n_pp = len(pp_show)
pp_height = max(0.04, min(0.30, 0.015 * (n_pp + 1)))
pp_bottom = 0.88 - learn_height - 0.03 - pp_height - 0.03
ax3 = fig.add_axes([0.02, pp_bottom, 0.96, pp_height])
ax3.axis('off')
ax3.set_title('C. Pre vs Post-Feed Comparison (Mann-Whitney U) -- All Sessions',
              fontsize=13, fontweight='bold', loc='left', pad=10)

t3 = ax3.table(cellText=pp_show.values, colLabels=pp_show.columns,
               cellLoc='center', loc='center')
t3.auto_set_font_size(False)
t3.set_fontsize(8)
t3.auto_set_column_width(list(range(len(pp_show.columns))))
for j in range(len(pp_show.columns)):
    t3[0, j].set_facecolor('#D5D8DC')
    t3[0, j].set_text_props(fontweight='bold')
for i in range(n_pp):
    sig = pp_show.iloc[i]['Sig']
    state = pp_show.iloc[i]['State']
    if sig == '*':
        color = '#ABEBC6'
    elif sig == '(*)':
        color = '#F9E79F'
    elif sig == '--':
        color = '#E8DAEF' if state == 'Fasted' else '#F2F3F4'
    else:
        color = 'white'
    for j in range(len(pp_show.columns)):
        t3[i + 1, j].set_facecolor(color)

# --- Table D: Key Biological Findings ---
findings = [
    ['1', 'LHA PC1 ramps at Pot-2', 'S2 (r=0.80, p=0.056), S4 (r=0.76, p=0.0004)',
     'Replicates', 'Progressive latent shift with repeated empty-pot visits'],
    ['2', 'LHA gate decreases at Pot-2', 'S2 (r=-0.82, p=0.044)',
     'S2 only (S4 not tested)', 'Dynamics activate at devalued pot'],
    ['3', 'LHA gate increases at Pot-4', 'S2 (r=0.64, p=0.008)',
     'S2 only', 'Dynamics freeze at food pot over learning'],
    ['4', 'LHA flow decreases at Pot-4', 'S2 (r=-0.53, p=0.037)',
     'S2 only', 'Slowing dynamics = increasing certainty?'],
    ['5', 'RSP PC1 declines at Pot-4', 'S2 (r=-0.57, p=0.021)',
     'S2 only', 'Spatial/familiarity encoding update'],
    ['6', 'LHA FR drops at Pot-2 post-feed', 'S4 (p=0.030), S2 (p=0.066)',
     'Consistent', 'Value update: Pot-2 devalued after discovery'],
    ['7', 'Fasted mice skip learning', 'S6: disc=33.7s, S8: disc=30.2s; 0 P2 visits',
     'Consistent', 'No learning curve possible (n<3 pre-disc visits)'],
    ['8', 'S6 Pot-4 pre vs post: all ns', 'n_pre=2, all p>=0.53',
     'Consistent w/ floor', 'Rapid discovery leaves little to update'],
    ['9', 'S8: insufficient pre-disc data', 'n_P2=0, n_P4=1',
     'N/A', 'Cannot test pre vs post (n<2)'],
    ['10', 'Fasted delay return to P4', 'S6: 590s, S8: 321s vs Fed: 31-60s',
     'Consistent', 'Opposite of expected: fasted slower to re-exploit'],
    ['11', 'RSP freezes at known food pot', 'S2 Exc 78, S8 all transitions',
     'Observed', 'Post-disc P2->P4: RSP gate spikes, flow collapses'],
    ['12', 'Reward onset transients', 'All sessions',
     'Consistent', 'FR dip at dig, rise at feed; PC1 transients; gate drops'],
]

findings_cols = ['#', 'Finding', 'Evidence', 'Replication', 'Interpretation']
n_findings = len(findings)
find_height = max(0.04, min(0.25, 0.015 * (n_findings + 1)))
find_bottom = pp_bottom - find_height - 0.03
ax4 = fig.add_axes([0.02, find_bottom, 0.96, find_height])
ax4.axis('off')
ax4.set_title('D. Key Biological Findings', fontsize=13, fontweight='bold', loc='left', pad=10)

t4 = ax4.table(cellText=findings, colLabels=findings_cols,
               cellLoc='left', loc='center',
               colWidths=[0.03, 0.2, 0.25, 0.15, 0.37])
t4.auto_set_font_size(False)
t4.set_fontsize(8)
for j in range(len(findings_cols)):
    t4[0, j].set_facecolor('#D5D8DC')
    t4[0, j].set_text_props(fontweight='bold')
for i, row in enumerate(findings):
    rep = row[3]
    if 'Replicates' in rep or 'Consistent' in rep:
        color = '#ABEBC6'
    elif 'Observed' in rep:
        color = '#F9E79F'
    elif 'only' in rep:
        color = '#FAD7A0'
    elif 'N/A' in rep:
        color = '#E8DAEF'
    else:
        color = 'white'
    for j in range(len(findings_cols)):
        t4[i + 1, j].set_facecolor(color)

plt.savefig('figures/foraging_key_findings_table.png', dpi=100, bbox_inches='tight')
plt.close()
print("  Saved: figures/foraging_key_findings_table.png")

# =========================================================================
# Print summary counts
# =========================================================================
print(f"\n  Learning curves table: {len(learning_df)} total rows "
      f"({len(learning_df[learning_df['Sig'].isin(['*','**','***'])])} significant, "
      f"{len(learning_df[learning_df['Sig']=='(*)'])} trending, "
      f"{len(learning_df[learning_df['Sig']=='--'])} insufficient data)")
print(f"  Pre vs post table: {len(prepost_df)} total rows "
      f"({len(prepost_df[prepost_df['Sig'].isin(['*','**','***'])])} significant, "
      f"{len(prepost_df[prepost_df['Sig']=='(*)'])} trending, "
      f"{len(prepost_df[prepost_df['Sig']=='--'])} insufficient data)")

# Session breakdown
for snum in [2, 4, 6, 8]:
    sname = f'S{snum}'
    lc = learning_df[learning_df['Session'] == sname]
    pp = prepost_df[prepost_df['Session'] == sname]
    lc_sig = len(lc[lc['Sig'].isin(['*', '**', '***', '(*)'])])
    pp_sig = len(pp[pp['Sig'].isin(['*', '**', '***', '(*)'])])
    lc_nd = len(lc[lc['Sig'] == '--'])
    pp_nd = len(pp[pp['Sig'] == '--'])
    print(f"  {sname}: learning {lc_sig} sig/trend + {lc_nd} no-data | "
          f"pre-post {pp_sig} sig/trend + {pp_nd} no-data")

print(f"\n{'='*70}")
print("  DONE")
print(f"{'='*70}")
