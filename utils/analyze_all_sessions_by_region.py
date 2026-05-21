"""
Comprehensive analysis: Load all single-probe sessions, compute QC metrics,
separate units by anatomical region (LHA vs RSP) using depth, and visualize.
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

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Load config
with open("paths.yaml") as f:
    paths_config = yaml.safe_load(f)

# Depth boundaries (micrometers)
LHA_DEPTH_MAX = 1100  # 0-1100 um = LHA
RSP_DEPTH_MIN = 1100  # >1100 um = RSP

def assign_region(depth):
    """Assign brain region based on depth."""
    if depth < LHA_DEPTH_MAX:
        return "LHA"
    else:
        return "RSP"

def load_cluster_depths(sorted_path_obj):
    """Load unit depths from cluster_info.tsv in sorter_output."""
    cluster_info_path = sorted_path_obj / "cluster_info.tsv"

    if not cluster_info_path.exists():
        print(f"[WARNING] cluster_info.tsv not found at {cluster_info_path}")
        return None

    try:
        cluster_df = pd.read_csv(cluster_info_path, sep='\t')
        # Expect columns: cluster_id, depth, etc.
        if 'depth' not in cluster_df.columns:
            print(f"[WARNING] 'depth' column not found in cluster_info.tsv")
            return None

        depth_dict = dict(zip(cluster_df['cluster_id'], cluster_df['depth']))
        return depth_dict
    except Exception as e:
        print(f"[ERROR] Could not load cluster_info.tsv: {e}")
        return None

def compute_session_metrics(session_path, sorted_path, session_key, session_name):
    """Compute QC metrics for a single session."""

    print(f"\n{'='*70}")
    print(f"Processing: {session_name}")
    print(f"State: {session_key['state']}, Phase: {session_key['phase']}")
    print(f"{'='*70}")

    sorted_path_obj = Path(sorted_path)

    # Load sorting
    try:
        sorting = se.read_kilosort(sorted_path_obj)
        print(f"[OK] Loaded {sorting.get_num_units()} units")
    except Exception as e:
        print(f"[ERROR] Could not load sorting: {e}")
        return None

    # Load depths from cluster_info.tsv
    depth_dict = load_cluster_depths(sorted_path_obj)
    if depth_dict is None:
        print(f"[WARNING] No depth information available, using index as proxy")
        # Fallback: use unit IDs as depth proxy (not ideal)
        unit_ids = sorting.get_unit_ids()
        depth_dict = {uid: uid * 20 for uid in unit_ids}  # Rough approximation

    # Compute metrics
    unit_ids = sorting.get_unit_ids()
    metrics_list = []

    # Get total duration
    all_spikes = []
    for uid in unit_ids:
        all_spikes.extend(sorting.get_unit_spike_train(uid))

    if len(all_spikes) > 0:
        max_spike = np.max(all_spikes)
        duration_s = max_spike / sorting.get_sampling_frequency()
    else:
        duration_s = 1.0

    for unit_id in unit_ids:
        spike_times = sorting.get_unit_spike_train(unit_id)

        # Firing rate
        firing_rate = len(spike_times) / duration_s if duration_s > 0 else 0

        # ISI violations
        if len(spike_times) > 1:
            isi = np.diff(spike_times) / sorting.get_sampling_frequency()
            isi_violations = np.sum(isi < 0.0015) / len(isi)
        else:
            isi_violations = np.nan

        # Presence ratio
        bin_size = 60
        n_bins = int(np.ceil(duration_s / bin_size))
        if n_bins > 0:
            bins = np.linspace(0, duration_s * sorting.get_sampling_frequency(), n_bins + 1)
            spike_counts, _ = np.histogram(spike_times, bins=bins)
            presence_ratio = np.sum(spike_counts > 0) / n_bins
        else:
            presence_ratio = 0

        # Get depth
        depth = depth_dict.get(unit_id, np.nan)
        region = assign_region(depth)

        # QC pass/fail
        passes_qc = (
            firing_rate >= 0.5 and
            isi_violations <= 0.05 and
            presence_ratio >= 0.8
        )

        metrics_list.append({
            'session': session_name,
            'mouse': session_path,
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
    print(f"[OK] Computed metrics for {len(metrics_df)} units")
    print(f"     LHA: {(metrics_df['region'] == 'LHA').sum()}, RSP: {(metrics_df['region'] == 'RSP').sum()}")
    print(f"     QC Pass: {metrics_df['passes_qc'].sum()}/{len(metrics_df)}")

    return metrics_df

# Main loop: process all single-probe sessions
all_metrics = []

for coordinates_key in ['coordinates_1', 'coordinates_2']:
    if coordinates_key not in paths_config["single_probe"]:
        continue

    coor_data = paths_config["single_probe"][coordinates_key]

    for mouse_key, mouse_data in coor_data.items():
        print(f"\n{'#'*70}")
        print(f"MOUSE: {mouse_key.upper()} ({coordinates_key.upper()})")
        print(f"{'#'*70}")

        sessions = mouse_data["sessions"]

        for session_key, session_data in sessions.items():
            session_name = f"{mouse_key}_{coordinates_key}_{session_key}"
            sorted_path = session_data["sorted"]

            try:
                metrics_df = compute_session_metrics(
                    mouse_key, sorted_path, session_data, session_name
                )
                if metrics_df is not None:
                    all_metrics.append(metrics_df)
            except Exception as e:
                print(f"[ERROR] Failed to process {session_name}: {e}")
                continue

# Combine all metrics
if len(all_metrics) > 0:
    combined_df = pd.concat(all_metrics, ignore_index=True)
    print(f"\n{'='*70}")
    print(f"COMBINED RESULTS: {len(combined_df)} total units across all sessions")
    print(f"{'='*70}\n")

    # Summary statistics
    print("SUMMARY BY REGION:")
    print("-" * 70)
    for region in ['LHA', 'RSP']:
        region_data = combined_df[combined_df['region'] == region]
        print(f"\n{region}:")
        print(f"  Total units: {len(region_data)}")
        print(f"  QC Pass: {region_data['passes_qc'].sum()} ({region_data['passes_qc'].sum()/len(region_data)*100:.1f}%)")
        print(f"  Firing rate: {region_data['firing_rate_hz'].mean():.2f} +/- {region_data['firing_rate_hz'].std():.2f} Hz")
        print(f"  ISI violations: {region_data['isi_violations'].mean()*100:.2f} +/- {region_data['isi_violations'].std()*100:.2f}%")
        print(f"  Presence ratio: {region_data['presence_ratio'].mean():.2f} +/- {region_data['presence_ratio'].std():.2f}")

    print("\nSUMMARY BY STATE:")
    print("-" * 70)
    for state in ['fed', 'fasted']:
        state_data = combined_df[combined_df['state'] == state]
        print(f"\n{state.upper()}:")
        print(f"  Total units: {len(state_data)}")
        print(f"  QC Pass: {state_data['passes_qc'].sum()} ({state_data['passes_qc'].sum()/len(state_data)*100:.1f}%)")
        for region in ['LHA', 'RSP']:
            region_state_data = state_data[state_data['region'] == region]
            if len(region_state_data) > 0:
                print(f"  {region}: {len(region_state_data)} units, {region_state_data['passes_qc'].sum()} QC pass")

    # Save combined results
    output_file = Path("data/all_sessions_unit_metrics_by_region.csv")
    output_file.parent.mkdir(exist_ok=True, parents=True)
    combined_df.to_csv(output_file, index=False)
    print(f"\n[OK] Saved combined metrics to: {output_file}")

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Depth distribution by region
    ax = axes[0, 0]
    lha_depths = combined_df[combined_df['region'] == 'LHA']['depth_um']
    rsp_depths = combined_df[combined_df['region'] == 'RSP']['depth_um']
    ax.hist(lha_depths, bins=30, alpha=0.6, label=f'LHA (n={len(lha_depths)})', color='blue')
    ax.hist(rsp_depths, bins=30, alpha=0.6, label=f'RSP (n={len(rsp_depths)})', color='red')
    ax.axvline(LHA_DEPTH_MAX, color='black', linestyle='--', linewidth=2, label='Boundary (1100 um)')
    ax.set_xlabel('Depth (micrometers)')
    ax.set_ylabel('Unit Count')
    ax.set_title('Anatomical Distribution of Units')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Firing rate by region
    ax = axes[0, 1]
    sns.boxplot(data=combined_df[combined_df['passes_qc']], x='region', y='firing_rate_hz', ax=ax)
    ax.set_ylabel('Firing Rate (Hz)')
    ax.set_title('Firing Rate by Region (QC Pass Only)')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: ISI violations by region
    ax = axes[1, 0]
    sns.boxplot(data=combined_df[combined_df['passes_qc']], x='region', y='isi_violations', ax=ax)
    ax.set_ylabel('ISI Violations (fraction)')
    ax.set_title('ISI Violations by Region (QC Pass Only)')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: QC pass rate by state and region
    ax = axes[1, 1]
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

    plt.tight_layout()
    fig_file = Path("figures/region_analysis_overview.png")
    plt.savefig(fig_file, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved figure to: {fig_file}")
    plt.close()

    print("\n[DONE] Analysis complete!")
else:
    print("[ERROR] No metrics data generated.")
