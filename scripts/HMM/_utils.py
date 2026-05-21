"""Shared utilities for the behavioral HMM pipeline."""
from pathlib import Path
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_paths_yaml(cfg):
    paths_path = REPO_ROOT / cfg["paths_yaml"]
    with open(paths_path) as f:
        return yaml.safe_load(f)


def session_xlsx(paths_data, session_num):
    """Return (xlsx_path, state) for a dual-probe session."""
    sessions = paths_data["double_probe"]["coordinates_1"]["mouse01"]["sessions"]
    s = sessions[f"session_{session_num}"]
    return s["behavior"], s["state"]


def session_list(cfg):
    """Return [(session_num, state), ...] for all configured foraging sessions."""
    out = []
    for s in cfg["sessions"]["fed"]:
        out.append((s, "fed"))
    for s in cfg["sessions"]["fasted"]:
        out.append((s, "fasted"))
    return out


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)
    return Path(p)
