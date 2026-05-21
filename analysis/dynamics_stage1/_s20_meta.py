"""Print full metadata for session_20."""
import yaml
from pathlib import Path

with open("paths.yaml") as f:
    cfg = yaml.safe_load(f)

s = cfg["double_probe"]["coordinates_1"]["mouse01"]["sessions"]["session_20"]
for k, v in s.items():
    print(f'{k}: {v}')
