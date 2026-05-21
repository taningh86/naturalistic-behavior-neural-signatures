"""Methodology figure: what is behavioral entropy and how is it calculated.

Uses a real foraging session to illustrate the sliding-window Shannon entropy
calculation on zone-transition sequences.

Output: figures/entropy_methodology.png
"""
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import entropy as sp_entropy


REPO = Path(__file__).resolve().parent
FIG = REPO / "figures" / "entropy_methodology.png"

WINDOW_SEC = 60.0
STEP_SEC = 10.0

# Use session 6 (fed foraging) — typical example
SESSION = 6
EXAMPLE_CHUNK_S = (300, 900)        # 10 minutes for clarity
ZOOM_WINDOW_START_S = 500.0          # 60 s window to highlight in middle panel

# Zone name lookup (xlsx columns → short)
ZONE_PRIORITY = [
    "Home corner left", "Home corner right", "Central Arena Zone",
    "Foraging arena", "Home", "ladder to Arena", "Transition Zone",
    "Pot-1 zone", "Pot-2 Zone", "Pot-3 zone", "Pot-4 zone",
    "Pot-1", "Pot-2", "Pot-3", "Pot-4",
]
ZONE_SHORT = {
    "Home": "H", "ladder to Arena": "L", "Transition Zone": "T",
    "Foraging arena": "FA", "Central Arena Zone": "CA",
    "Pot-1": "P1", "Pot-2": "P2", "Pot-3": "P3", "Pot-4": "P4",
    "Pot-1 zone": "P1z", "Pot-2 Zone": "P2z", "Pot-3 zone": "P3z",
    "Pot-4 zone": "P4z",
    "Home corner left": "HCL", "Home corner right": "HCR",
}
ZONE_COLOR = {
    "H": "#3498db", "L": "#9b59b6", "T": "#e67e22",
    "FA": "#95a5a6", "CA": "#bdc3c7",
    "HCL": "#34495e", "HCR": "#34495e",
    "P1": "#e74c3c", "P2": "#c0392b", "P3": "#d35400", "P4": "#a93226",
    "P1z": "#f1948a", "P2z": "#e6b0aa", "P3z": "#edbb99", "P4z": "#d98880",
    "O": "#cccccc",
}


def load_zones(snum):
    paths = yaml.safe_load(open(REPO / "paths.yaml"))
    sess = paths["double_probe"]["coordinates_1"]["mouse01"]["sessions"]
    xlsx_path = sess[f"session_{snum}"]["behavior"]
    df_raw = pd.read_excel(xlsx_path, header=None)
    col_names = df_raw.iloc[34].tolist()
    data = df_raw.iloc[36:].reset_index(drop=True)
    data.columns = col_names
    time_vals = pd.to_numeric(data["Recording time"], errors="coerce").values
    zones = np.full(len(time_vals), "O", dtype=object)
    for zname in ZONE_PRIORITY:
        col_match = [c for c in col_names
                     if isinstance(c, str) and c.startswith("Zone(") and zname in c]
        if col_match:
            vals = pd.to_numeric(data[col_match[0]], errors="coerce").values
            mask = vals > 0.5
            short = ZONE_SHORT.get(zname, zname[:3])
            zones[mask] = short
    return time_vals, zones


def compute_entropy_window(zones, time_vals, window_sec, step_sec):
    dt = np.median(np.diff(time_vals))
    window_bins = int(window_sec / dt)
    step_bins = int(step_sec / dt)
    ent_times, ent_vals = [], []
    for start_idx in range(0, len(zones) - window_bins, step_bins):
        wz = zones[start_idx:start_idx + window_bins]
        transitions = []
        for j in range(1, len(wz)):
            if wz[j] != wz[j - 1]:
                transitions.append(f"{wz[j-1]}->{wz[j]}")
        if len(transitions) < 3:
            continue
        counts = Counter(transitions)
        probs = np.array(list(counts.values()), dtype=float)
        probs /= probs.sum()
        h = sp_entropy(probs, base=2)
        ent_times.append(time_vals[start_idx + window_bins - 1])
        ent_vals.append(h)
    return np.array(ent_times), np.array(ent_vals)


def main():
    print(f"Loading session {SESSION}…")
    time_vals, zones = load_zones(SESSION)
    ent_times, ent_vals = compute_entropy_window(zones, time_vals,
                                                       WINDOW_SEC, STEP_SEC)
    print(f"Total bins: {len(zones)}, dt = {np.median(np.diff(time_vals)):.3f} s")
    print(f"Entropy points: {len(ent_vals)}")

    # Filter to chunk
    t0, t1 = EXAMPLE_CHUNK_S
    chunk_mask = (time_vals >= t0) & (time_vals <= t1)
    t_chunk = time_vals[chunk_mask]
    z_chunk = zones[chunk_mask]
    ent_mask = (ent_times >= t0) & (ent_times <= t1)
    et_chunk = ent_times[ent_mask]
    ev_chunk = ent_vals[ent_mask]

    # Find transitions in zoom window
    zw_start = ZOOM_WINDOW_START_S
    zw_end = zw_start + WINDOW_SEC
    zw_mask = (time_vals >= zw_start) & (time_vals <= zw_end)
    zw_zones = zones[zw_mask]
    zw_transitions = []
    for j in range(1, len(zw_zones)):
        if zw_zones[j] != zw_zones[j-1]:
            zw_transitions.append(f"{zw_zones[j-1]}→{zw_zones[j]}")
    zw_counter = Counter(zw_transitions)
    probs = np.array(list(zw_counter.values()), dtype=float)
    probs /= probs.sum()
    h_demo = sp_entropy(probs, base=2)

    # ===== FIGURE =====
    fig = plt.figure(figsize=(17, 14))
    gs = fig.add_gridspec(4, 6, height_ratios=[1.0, 2.2, 3.0, 1.4],
                            hspace=0.8, wspace=0.55)

    # ----- Title block -----
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.text(0.5, 0.8, "Behavioral entropy: how it's calculated",
                    ha="center", va="top", fontsize=18, fontweight="bold",
                    transform=ax_title.transAxes)
    ax_title.text(0.5, 0.45,
        "Per-window Shannon entropy over the distribution of zone→zone\n"
        "transitions. High entropy = many different transition types, "
        "uniformly used (varied/exploratory behavior).\n"
        "Low entropy = few transition types or one dominates "
        "(stereotyped behavior, e.g., repeatedly going Home↔Pot).",
        ha="center", va="top", fontsize=11,
        transform=ax_title.transAxes)

    # ----- Top: zone timeline (chunk) -----
    ax_zone = fig.add_subplot(gs[1, :])
    unique_zones = np.unique(z_chunk)
    zone_to_y = {z: i for i, z in enumerate(sorted(unique_zones))}
    for j, t in enumerate(t_chunk):
        z = z_chunk[j]
        ax_zone.plot(t, zone_to_y[z], ".", color=ZONE_COLOR.get(z, "k"),
                       markersize=2)
    ax_zone.set_yticks(list(zone_to_y.values()))
    ax_zone.set_yticklabels(list(zone_to_y.keys()), fontsize=9)
    ax_zone.set_xlabel("Time (s)", fontsize=10)
    ax_zone.set_ylabel("Zone", fontsize=10)
    ax_zone.set_xlim(t0, t1)
    ax_zone.set_title(f"Step 1: Track zone occupancy over time "
                        f"(S{SESSION} fed foraging, {t0}–{t1} s)",
                        fontsize=11, fontweight="bold")
    # Highlight the 60 s window we will analyze
    rect = Rectangle((zw_start, -0.3), WINDOW_SEC, len(unique_zones) - 0.4,
                      linewidth=2, edgecolor="red", facecolor="red", alpha=0.15)
    ax_zone.add_patch(rect)
    # Place label INSIDE the highlighted window at the top so it doesn't collide
    # with the suptitle of the panel
    ax_zone.text(zw_start + WINDOW_SEC/2, len(unique_zones) - 0.6,
                   "60 s window",
                   ha="center", va="top", color="red", fontsize=10,
                   fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.25",
                              facecolor="white", edgecolor="red", alpha=0.85))
    ax_zone.grid(True, alpha=0.3, axis="x")

    # ----- Middle: transitions within the highlighted window -----
    ax_trans = fig.add_subplot(gs[2, :3])
    # Build a stacked bar showing the transition types and their counts
    if zw_counter:
        labels = list(zw_counter.keys())
        counts = list(zw_counter.values())
        colors = plt.cm.tab20(np.linspace(0, 1, len(labels)))
        y_pos = np.arange(len(labels))
        ax_trans.barh(y_pos, counts, color=colors, edgecolor="black")
        ax_trans.set_yticks(y_pos)
        ax_trans.set_yticklabels(labels, fontsize=9)
        ax_trans.invert_yaxis()
        for i, c in enumerate(counts):
            ax_trans.text(c + 0.05, i,
                            f"p={c/sum(counts):.2f}",
                            va="center", fontsize=8)
        ax_trans.set_xlabel("Count within 60 s window", fontsize=10)
        ax_trans.set_title(f"Step 2: Enumerate zone→zone transitions in the window\n"
                              f"({sum(counts)} transitions total, "
                              f"{len(labels)} unique types)",
                              fontsize=11, fontweight="bold")
        ax_trans.grid(True, alpha=0.3, axis="x")
    else:
        ax_trans.text(0.5, 0.5, "(window had <3 transitions; skipped)",
                        ha="center", va="center", transform=ax_trans.transAxes)

    # ----- Middle right: the entropy formula and result (clean layout) -----
    ax_form = fig.add_subplot(gs[2, 3:])
    ax_form.axis("off")

    # Title
    ax_form.text(0.5, 0.95,
        "Step 3: Shannon entropy of the transition distribution",
        ha="center", va="top", fontsize=12, fontweight="bold",
        transform=ax_form.transAxes)

    # Formula with "(bits)" inline so the subscript doesn't crash into a label
    ax_form.text(0.5, 0.74,
        r"$H \;=\; -\sum_i \, p_i \, \log_2 p_i \quad$ (bits)",
        ha="center", va="center", fontsize=24,
        transform=ax_form.transAxes)

    # Variable definition (well below formula's subscript)
    ax_form.text(0.5, 0.55,
        r"where  $p_i \;=\;$  count(transition$_i$) $/$ total transitions",
        ha="center", va="center", fontsize=11,
        transform=ax_form.transAxes)

    # For this window — header + three lines, well spaced
    ax_form.text(0.5, 0.40, "For the highlighted window:",
                   ha="center", va="center", fontsize=11, fontweight="bold",
                   transform=ax_form.transAxes)
    ax_form.text(0.5, 0.30,
        f"total transitions  =  {sum(zw_counter.values())}",
        ha="center", va="center", fontsize=11,
        family="monospace", transform=ax_form.transAxes)
    ax_form.text(0.5, 0.22,
        f"unique types       =  {len(zw_counter)}",
        ha="center", va="center", fontsize=11,
        family="monospace", transform=ax_form.transAxes)
    ax_form.text(0.5, 0.14,
        f"H  =  {h_demo:.3f}  bits",
        ha="center", va="center", fontsize=13, fontweight="bold",
        family="monospace", color="#27ae60", transform=ax_form.transAxes)

    # Window parameters (bottom, italic)
    ax_form.text(0.5, 0.03,
        "Window: 60 s  ·  Step: 10 s (overlapping)  ·  "
        "End-aligned (causal)  ·  skip <3 transitions",
        ha="center", va="center", fontsize=9, style="italic",
        transform=ax_form.transAxes)

    # ----- Bottom: entropy time series across the chunk -----
    ax_ent = fig.add_subplot(gs[3, :])
    ax_ent.plot(et_chunk, ev_chunk, color="#27ae60", lw=1.4)
    ax_ent.fill_between(et_chunk, 0, ev_chunk, color="#27ae60", alpha=0.25)
    ax_ent.axvline(zw_end, color="red", lw=1.2, ls="--", alpha=0.8)
    # mark the demo window's entropy
    ax_ent.scatter([zw_end], [h_demo], color="red", s=60, zorder=5,
                     label=f"window above → H = {h_demo:.3f} bits")
    ax_ent.set_xlim(t0, t1)
    ax_ent.set_xlabel("Time (s) (= end of 60 s window)", fontsize=10)
    ax_ent.set_ylabel("H (bits)", fontsize=10)
    ax_ent.set_title("Step 4: Slide window in 10 s steps → entropy time series",
                       fontsize=11, fontweight="bold")
    ax_ent.legend(fontsize=9, loc="lower right")
    ax_ent.grid(True, alpha=0.3)

    fig.savefig(FIG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIG}")


if __name__ == "__main__":
    main()
