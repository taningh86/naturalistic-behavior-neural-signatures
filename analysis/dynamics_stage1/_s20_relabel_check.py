"""Recheck S20 LHA after cluster_info.tsv curation."""
import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from dp_avalanche_criticality import (
    get_good_units_p1_lha, P1_MIN_FR, P1_MIN_AMP,
    LHA_DEPTH_MIN, LHA_DEPTH_MAX,
)

with open("paths.yaml") as f:
    cfg = yaml.safe_load(f)
sval = cfg["double_probe"]["coordinates_1"]["mouse01"]["sessions"]["session_20"]
sp = Path(sval['probe_1_lha_rsp']['sorted'])
ci = sp / "cluster_info.tsv"

df = pd.read_csv(ci, sep='\t')
print(f'cluster_info.tsv rows: {len(df)}')
print(f'columns: {list(df.columns)}')
label_col = 'group' if ('group' in df.columns and df['group'].eq('good').any()) else 'KSLabel'
print(f'label_col chosen: {label_col}')
print(f'\nlabel counts ({label_col}):')
print(df[label_col].value_counts().to_string())

print(f'\nApplying probe-1 LHA filters: KSLabel=good, FR>{P1_MIN_FR}, AMP>{P1_MIN_AMP}, depth in [{LHA_DEPTH_MIN}, {LHA_DEPTH_MAX}] um')
g_label = df[df[label_col] == 'good']
print(f'  after label filter:        {len(g_label)}')
g_fr = g_label[g_label['fr'] > P1_MIN_FR]
print(f'  + fr filter:               {len(g_fr)}')
g_amp = g_fr[g_fr['amp'] > P1_MIN_AMP]
print(f'  + amp filter:              {len(g_amp)}')
g_depth = g_amp[(g_amp['depth'] >= LHA_DEPTH_MIN) & (g_amp['depth'] <= LHA_DEPTH_MAX)]
print(f'  + depth filter (final):    {len(g_depth)}')

print(f'\nfinal IDs (first 10): {list(g_depth["cluster_id"].values[:10])}')

# Confirm via the actual loader
final = get_good_units_p1_lha(sp)
print(f'\nget_good_units_p1_lha() returns: {len(final)} units')
