"""Merge S20's per-session csv into all_sessions_summary.csv."""
import pandas as pd
from pathlib import Path

OUT = Path('data/dynamics_stage1')
agg_path = OUT / 'all_sessions_summary.csv'
s20_path = OUT / 'session_20_summary.csv'

agg = pd.read_csv(agg_path)
s20 = pd.read_csv(s20_path)

print(f'before: {len(agg)} rows, {agg.session.nunique()} sessions')
agg_no_s20 = agg[agg.session != 20]
combo = pd.concat([agg_no_s20, s20], ignore_index=True)
combo = combo.sort_values(['session', 'start_bin']).reset_index(drop=True)
combo.to_csv(agg_path, index=False)
print(f'after:  {len(combo)} rows, {combo.session.nunique()} sessions')
print(f'S20 rows in merged: {(combo.session == 20).sum()}')

print(f'\nphases per session (final):')
print(combo.groupby(['session', 'state', 'exp_phase']).size().to_string())
