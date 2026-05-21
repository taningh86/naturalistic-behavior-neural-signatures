"""
Dual-Probe Probe-0 (ACA) Analysis: Load all double-probe sessions for probe-0 (imec0),
compute QC metrics, and label all units as ACA.

All double-probe sessions are currently fed only.
Session 2 has null sorted paths and is skipped.
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


def load_cluster_depths(sorted_path_obj):
    """Load unit depths from cluster_info.tsv in sorter_output."""
    cluster_info_path = sorted_path_obj / "cluster_info.tsv"

    if not cluster_info_path.exists():
        print(f"[WARNING] cluster_info.tsv not found at {cluster_info_path}")
        return None

    try:
        cluster_df = pd.read_csv(cluster_info_path, sep='\t')
        if 'depth' not in cluster_df.columns:
            print(f"[WARNING] 'depth' column not found in cluster_info.tsv")
            return None

        depth_dict = dict(zip(cluster_df['cluster_id'], cluster_df['depth']))
        return depth_dict
    except Exception as e:
        print(f"[ERROR] Could not load cluster_info.tsv: {e}")
        return None


def compute_session_metrics(mouse_key, sorted_path, session_key, session_name):
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
        unit_ids = sorting.get_unit_ids()
        depth_dict = {uid: uid * 20 for uid in unit_ids}

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

        # Get depth — all units are ACA for probe-0
        depth = depth_dict.get(unit_id, np.nan)
        region = "ACA"

        # QC pass/fail
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
    print(f"[OK] Computed metrics for {len(metrics_df)} units")
    print(f"     ACA: {(metrics_df['region'] == 'ACA').sum()}")
    print(f"     QC Pass: {metrics_df['passes_qc'].sum()}/{len(metrics_df)}")

    return metrics_df


# =============================================================================
# MAIN: Process all double-probe probe-0 sessions
# =============================================================================

all_metrics = []

dp_data = paths_config["double_probe"]["coordinates_1"]["mouse01"]
sessions = dp_data["sessions"]

print(f"\n{'#'*70}")
print(f"DOUBLE PROBE — PROBE-0 (ACA) — MOUSE01 COORDINATES-1")
print(f"{'#'*70}")

for session_key, session_data in sessions.items():
    session_name = f"mouse01_double_probe_coor1_{session_key}"

    # Get probe-0 sorted path
    probe0_data = session_data.get("probe_0_aca", {})
    sorted_path = probe0_data.get("sorted") if probe0_data else None

    if sorted_path is None:
        print(f"\n[SKIP] {session_name}: No sorted data available for probe-0")
        continue

    try:
        metrics_df = compute_session_metrics(
            "mouse01", sorted_path, session_data, session_name
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
    print(f"COMBINED RESULTS: {len(combined_df)} total ACA units across all sessions")
    print(f"{'='*70}\n")

    # Summary statistics
    print("SUMMARY (ACA — Probe-0):")
    print("-" * 70)
    print(f"  Total units: {len(combined_df)}")
    print(f"  QC Pass: {combined_df['passes_qc'].sum()} ({combined_df['passes_qc'].sum()/len(combined_df)*100:.1f}%)")
    print(f"  Firing rate: {combined_df['firing_rate_hz'].mean():.2f} +/- {combined_df['firing_rate_hz'].std():.2f} Hz")
    print(f"  ISI violations: {combined_df['isi_violations'].mean()*100:.2f} +/- {combined_df['isi_violations'].std()*100:.2f}%")
    print(f"  Presence ratio: {combined_df['presence_ratio'].mean():.2f} +/- {combined_df['presence_ratio'].std():.2f}")

    print("\nSUMMARY BY PHASE:")
    print("-" * 70)
    for phase in ['exploration', 'foraging']:
        phase_data = combined_df[combined_df['phase'] == phase]
        print(f"\n{phase.upper()}:")
        print(f"  Total units: {len(phase_data)}")
        print(f"  QC Pass: {phase_data['passes_qc'].sum()} ({phase_data['passes_qc'].sum()/len(phase_data)*100:.1f}%)")
        qc_pass = phase_data[phase_data['passes_qc']]
        if len(qc_pass) > 0:
            print(f"  Firing rate (QC pass): {qc_pass['firing_rate_hz'].mean():.2f} +/- {qc_pass['firing_rate_hz'].std():.2f} Hz")

    # Save combined results
    output_file = Path("data/double_probe_probe0_unit_metrics.csv")
    output_file.parent.mkdir(exist_ok=True, parents=True)
    combined_df.to_csv(output_file, index=False)
    print(f"\n[OK] Saved combined metrics to: {output_file}")

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Double Probe — Probe-0 (ACA) — Unit Quality Metrics', fontsize=14, fontweight='bold')

    # Plot 1: Depth distribution
    ax = axes[0, 0]
    aca_depths = combined_df['depth_um'].dropna()
    ax.hist(aca_depths, bins=30, alpha=0.7, color='green', label=f'ACA (n={len(aca_depths)})')
    ax.set_xlabel('Depth (micrometers)')
    ax.set_ylabel('Unit Count')
    ax.set_title('Depth Distribution of ACA Units')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Firing rate by phase
    ax = axes[0, 1]
    sns.boxplot(data=combined_df[combined_df['passes_qc']], x='phase', y='firing_rate_hz', ax=ax,
                palette='Set2')
    ax.set_ylabel('Firing Rate (Hz)')
    ax.set_title('Firing Rate by Phase (QC Pass Only)')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: ISI violations distribution
    ax = axes[1, 0]
    sns.boxplot(data=combined_df[combined_df['passes_qc']], x='phase', y='isi_violations', ax=ax,
                palette='Set2')
    ax.set_ylabel('ISI Violations (fraction)')
    ax.set_title('ISI Violations by Phase (QC Pass Only)')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: QC pass rate by session
    ax = axes[1, 1]
    qc_by_session = combined_df.groupby('session')['passes_qc'].agg(['sum', 'count'])
    qc_by_session['pass_rate'] = qc_by_session['sum'] / qc_by_session['count']
    # Shorten labels for readability
    labels = [s.replace('mouse01_double_probe_coor1_', 'S') for s in qc_by_session.index]
    ax.bar(range(len(qc_by_session)), qc_by_session['pass_rate'], color='green', alpha=0.7)
    ax.set_xticks(range(len(qc_by_session)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('QC Pass Rate')
    ax.set_title('QC Pass Rate by Session')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)

    plt.tight_layout()
    fig_file = Path("figures/double_probe_probe0_analysis_overview.png")
    plt.savefig(fig_file, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved figure to: {fig_file}")
    plt.close()

    print("\n[DONE] Dual-probe probe-0 (ACA) QC analysis complete!")
else:
    print("[ERROR] No metrics data generated.")
