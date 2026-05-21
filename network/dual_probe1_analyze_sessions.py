"""
Dual-Probe Probe-1 (LHA+RSP) Analysis: Load all double-probe sessions for probe-1 (imec1),
compute QC metrics, and assign regions by depth (LHA < 1100 µm, RSP >= 1100 µm).

Fed: sessions 1, 3-10 (session 2 null)
Fasted: sessions 11-16 (sessions 17-18 null)
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spikeinterface.extractors as se
import warnings

warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

with open("paths.yaml") as f:
    paths_config = yaml.safe_load(f)

# Depth boundaries — same as single-probe
LHA_DEPTH_MAX = 1100  # 0-1100 um = LHA

def assign_region(depth):
    if depth < LHA_DEPTH_MAX:
        return "LHA"
    else:
        return "RSP"

def load_cluster_depths(sorted_path_obj):
    cluster_info_path = sorted_path_obj / "cluster_info.tsv"
    if not cluster_info_path.exists():
        print(f"[WARNING] cluster_info.tsv not found at {cluster_info_path}")
        return None
    try:
        cluster_df = pd.read_csv(cluster_info_path, sep='\t')
        if 'depth' not in cluster_df.columns:
            print(f"[WARNING] 'depth' column not found in cluster_info.tsv")
            return None
        return dict(zip(cluster_df['cluster_id'], cluster_df['depth']))
    except Exception as e:
        print(f"[ERROR] Could not load cluster_info.tsv: {e}")
        return None

def compute_session_metrics(mouse_key, sorted_path, session_key, session_name):
    print(f"\n{'='*70}")
    print(f"Processing: {session_name}")
    print(f"State: {session_key['state']}, Phase: {session_key['phase']}")
    print(f"{'='*70}")

    sorted_path_obj = Path(sorted_path)

    try:
        sorting = se.read_kilosort(sorted_path_obj)
        print(f"[OK] Loaded {sorting.get_num_units()} units")
    except Exception as e:
        print(f"[ERROR] Could not load sorting: {e}")
        return None

    depth_dict = load_cluster_depths(sorted_path_obj)
    if depth_dict is None:
        print(f"[WARNING] No depth information available, using index as proxy")
        unit_ids = sorting.get_unit_ids()
        depth_dict = {uid: uid * 20 for uid in unit_ids}

    unit_ids = sorting.get_unit_ids()
    metrics_list = []

    all_spikes = []
    for uid in unit_ids:
        all_spikes.extend(sorting.get_unit_spike_train(uid))

    if len(all_spikes) > 0:
        duration_s = np.max(all_spikes) / sorting.get_sampling_frequency()
    else:
        duration_s = 1.0

    for unit_id in unit_ids:
        spike_times = sorting.get_unit_spike_train(unit_id)
        firing_rate = len(spike_times) / duration_s if duration_s > 0 else 0

        if len(spike_times) > 1:
            isi = np.diff(spike_times) / sorting.get_sampling_frequency()
            isi_violations = np.sum(isi < 0.0015) / len(isi)
        else:
            isi_violations = np.nan

        bin_size = 60
        n_bins = int(np.ceil(duration_s / bin_size))
        if n_bins > 0:
            bins = np.linspace(0, duration_s * sorting.get_sampling_frequency(), n_bins + 1)
            spike_counts, _ = np.histogram(spike_times, bins=bins)
            presence_ratio = np.sum(spike_counts > 0) / n_bins
        else:
            presence_ratio = 0

        depth = depth_dict.get(unit_id, np.nan)
        region = assign_region(depth)

        passes_qc = (
            firing_rate >= 0.5 and
            isi_violations <= 0.05 and
            presence_ratio >= 0.8
        )

        metrics_list.append({
            'session': session_name,
            'mouse': mouse_key,
            'state': session_key['state'],
            'phase': session_key['phase'],
            'unit_id': unit_id,
            'depth_um': depth,
            'region': region,
            'firing_rate_hz': firing_rate,
            'isi_violations': isi_violations,
            'presence_ratio': presence_ratio,
            'spike_count': len(spike_times),
            'passes_qc': passes_qc
        })

    metrics_df = pd.DataFrame(metrics_list)
    n_lha = (metrics_df['region'] == 'LHA').sum()
    n_rsp = (metrics_df['region'] == 'RSP').sum()
    n_qc = metrics_df['passes_qc'].sum()
    print(f"[OK] Computed metrics for {len(metrics_df)} units")
    print(f"     LHA: {n_lha}, RSP: {n_rsp}")
    print(f"     QC Pass: {n_qc}/{len(metrics_df)}")

    return metrics_df


# =============================================================================
# MAIN: Process all probe-1 sessions (fed + fasted)
# =============================================================================

all_metrics = []
dp_sessions = paths_config["double_probe"]["coordinates_1"]["mouse01"]["sessions"]

print(f"\n{'#'*70}")
print(f"DOUBLE PROBE — PROBE-1 (LHA+RSP) — ALL SESSIONS")
print(f"{'#'*70}")

for session_key in sorted(dp_sessions.keys(), key=lambda x: int(x.split('_')[1])):
    session_data = dp_sessions[session_key]
    session_name = f"mouse01_double_probe_coor1_{session_key}"

    p1 = session_data.get("probe_1_lha_rsp", {})
    sorted_path = p1.get("sorted") if p1 else None

    if sorted_path is None:
        print(f"\n[SKIP] {session_name}: No sorted data for probe-1")
        continue

    try:
        metrics_df = compute_session_metrics("mouse01", sorted_path, session_data, session_name)
        if metrics_df is not None:
            all_metrics.append(metrics_df)
    except Exception as e:
        print(f"[ERROR] Failed to process {session_name}: {e}")
        continue

# Combine
if len(all_metrics) > 0:
    combined_df = pd.concat(all_metrics, ignore_index=True)
    print(f"\n{'='*70}")
    print(f"COMBINED RESULTS: {len(combined_df)} total units (probe-1)")
    print(f"{'='*70}\n")

    # Summary by region
    print("SUMMARY BY REGION:")
    print("-" * 70)
    for region in ['LHA', 'RSP']:
        rd = combined_df[combined_df['region'] == region]
        qc = rd[rd['passes_qc']]
        print(f"\n{region}:")
        print(f"  Total units: {len(rd)}")
        print(f"  QC Pass: {len(qc)} ({len(qc)/len(rd)*100:.1f}%)" if len(rd) > 0 else "  QC Pass: 0")
        if len(qc) > 0:
            print(f"  Firing rate (QC pass): {qc['firing_rate_hz'].mean():.2f} +/- {qc['firing_rate_hz'].std():.2f} Hz")

    # Summary by state
    print("\nSUMMARY BY STATE:")
    print("-" * 70)
    for state in ['fed', 'fasted']:
        sd = combined_df[combined_df['state'] == state]
        print(f"\n{state.upper()}:")
        print(f"  Total units: {len(sd)}")
        print(f"  QC Pass: {sd['passes_qc'].sum()} ({sd['passes_qc'].sum()/len(sd)*100:.1f}%)")
        for region in ['LHA', 'RSP']:
            rsd = sd[sd['region'] == region]
            qc = rsd[rsd['passes_qc']]
            print(f"    {region}: {len(rsd)} total, {len(qc)} QC pass")

    # Summary by state and phase
    print("\nSUMMARY BY STATE x PHASE x REGION (QC pass only):")
    print("-" * 70)
    for state in ['fed', 'fasted']:
        for phase in ['exploration', 'foraging']:
            subset = combined_df[(combined_df['state'] == state) & (combined_df['phase'] == phase) & (combined_df['passes_qc'])]
            n_lha = (subset['region'] == 'LHA').sum()
            n_rsp = (subset['region'] == 'RSP').sum()
            print(f"  {state}_{phase}: LHA={n_lha}, RSP={n_rsp}")

    # Save
    output_file = Path("data/double_probe_probe1_unit_metrics.csv")
    output_file.parent.mkdir(exist_ok=True, parents=True)
    combined_df.to_csv(output_file, index=False)
    print(f"\n[OK] Saved combined metrics to: {output_file}")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Double Probe — Probe-1 (LHA+RSP) — Unit Quality Metrics', fontsize=14, fontweight='bold')

    # Depth distribution
    ax = axes[0, 0]
    lha_depths = combined_df[combined_df['region'] == 'LHA']['depth_um'].dropna()
    rsp_depths = combined_df[combined_df['region'] == 'RSP']['depth_um'].dropna()
    ax.hist(lha_depths, bins=30, alpha=0.6, label=f'LHA (n={len(lha_depths)})', color='blue')
    ax.hist(rsp_depths, bins=30, alpha=0.6, label=f'RSP (n={len(rsp_depths)})', color='red')
    ax.axvline(LHA_DEPTH_MAX, color='black', linestyle='--', linewidth=2, label='Boundary (1100 um)')
    ax.set_xlabel('Depth (micrometers)')
    ax.set_ylabel('Unit Count')
    ax.set_title('Depth Distribution by Region')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Firing rate by region and state
    ax = axes[0, 1]
    qc_data = combined_df[combined_df['passes_qc']]
    if len(qc_data) > 0:
        sns.boxplot(data=qc_data, x='region', y='firing_rate_hz', hue='state', ax=ax, palette=['#3498db', '#e74c3c'])
    ax.set_ylabel('Firing Rate (Hz)')
    ax.set_title('Firing Rate by Region & State (QC Pass)')
    ax.grid(True, alpha=0.3, axis='y')

    # QC pass rate by state and region
    ax = axes[1, 0]
    qc_summary = combined_df.groupby(['state', 'region'])['passes_qc'].agg(['sum', 'count'])
    qc_summary['pass_rate'] = qc_summary['sum'] / qc_summary['count']
    qc_pivot = qc_summary['pass_rate'].unstack()
    qc_pivot.plot(kind='bar', ax=ax, color=['blue', 'red'])
    ax.set_ylabel('QC Pass Rate')
    ax.set_xlabel('Metabolic State')
    ax.set_title('QC Pass Rate by State and Region')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(title='Region')
    ax.grid(True, alpha=0.3, axis='y')

    # Unit count by session
    ax = axes[1, 1]
    session_counts = combined_df[combined_df['passes_qc']].groupby(['session', 'region']).size().unstack(fill_value=0)
    labels = [s.replace('mouse01_double_probe_coor1_', '') for s in session_counts.index]
    session_counts.plot(kind='bar', ax=ax, color=['blue', 'red'], alpha=0.7)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('QC Pass Units')
    ax.set_title('Good Units per Session by Region')
    ax.legend(title='Region')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig_file = Path("figures/double_probe_probe1_analysis_overview.png")
    plt.savefig(fig_file, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved figure to: {fig_file}")
    plt.close()

    print("\n[DONE] Probe-1 QC analysis complete!")
else:
    print("[ERROR] No metrics data generated.")
