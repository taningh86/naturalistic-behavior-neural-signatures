"""Compare behavior keys across pilot sessions, with lever-zone columns flagged."""
import sys
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "analysis" / "cycles_partD"))
sys.path.insert(0, str(REPO / "analysis" / "dynamics_stage1"))
from stage1_lib import list_sessions
from dp_cycles_lib import load_neural, load_behavior

PILOT_SESSIONS = [3, 4, 11, 19]

session_keys = {}
for snum in PILOT_SESSIONS:
    print(f'\n=== S{snum} ===')
    _, bin_centers, _ = load_neural(snum, 'ACA')
    behav = load_behavior(snum, bin_centers)
    keys = sorted(behav.keys())
    session_keys[snum] = set(keys)
    cont = [k for k, v in behav.items() if v.get('type') == 'continuous']
    cat = [k for k, v in behav.items() if v.get('type') == 'categorical']
    print(f'  total: {len(keys)}  continuous: {len(cont)}  categorical: {len(cat)}')
    lever = [k for k in keys if 'lever' in k.lower()]
    print(f'  lever-related: {lever}')
    print(f'  binary categorical (excl compartment):')
    for k in cat:
        if k == 'compartment':
            continue
        print(f'    - {k}')

print('\n\n=== set comparisons ===')
fed_exp = session_keys[3]
fed_for = session_keys[4]
fas_exp = session_keys[11]
hfd_exp = session_keys[19]

print(f'\nIn fed-exp (S3) but NOT in HFD-exp (S19):')
for k in sorted(fed_exp - hfd_exp):
    print(f'  - {k}')

print(f'\nIn HFD-exp (S19) but NOT in fed-exp (S3):')
for k in sorted(hfd_exp - fed_exp):
    print(f'  - {k}')

print(f'\nIn fasted-exp (S11) but NOT in HFD-exp (S19):')
for k in sorted(fas_exp - hfd_exp):
    print(f'  - {k}')

print(f'\nIn HFD-exp (S19) but NOT in fasted-exp (S11):')
for k in sorted(hfd_exp - fas_exp):
    print(f'  - {k}')

print(f'\nCommon to all 4 pilots:')
common = fed_exp & fed_for & fas_exp & hfd_exp
print(f'  count: {len(common)}')
