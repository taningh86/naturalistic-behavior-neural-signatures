import yaml
from pathlib import Path

with open('paths.yaml') as f:
    cfg = yaml.safe_load(f)
sessions = cfg['double_probe']['coordinates_1']['mouse01']['sessions']
print(f'Total session keys in paths.yaml: {len(sessions)}')
print()
for k in sorted(sessions.keys(), key=lambda x: int(x.split('_')[1])):
    v = sessions[k]
    p0 = v.get('probe_0_aca', {}).get('sorted')
    p1 = v.get('probe_1_lha_rsp', {}).get('sorted')
    bh = v.get('behavior')
    p0_ok = (p0 is not None) and Path(p0).exists()
    p1_ok = (p1 is not None) and Path(p1).exists()
    bh_ok = (bh is not None) and Path(bh).exists()
    state = v.get('state', '?')
    phase = v.get('phase', '?')
    flags = []
    if not p0: flags.append('no-p0-path')
    elif not p0_ok: flags.append('p0-missing')
    if not p1: flags.append('no-p1-path')
    elif not p1_ok: flags.append('p1-missing')
    if not bh: flags.append('no-bh-path')
    elif not bh_ok: flags.append('bh-missing')
    valid = (p0_ok and p1_ok and bh_ok)
    mark = 'OK' if valid else 'SKIP'
    print(f'  {k:14s} state={state:8s} phase={phase:12s} {mark} {flags}')
