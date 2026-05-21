"""
Compute spike sorting quality metrics using SpikeInterface.
Identifies good units from Kilosort3 output.
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import spikeinterface as si
import spikeinterface.core as sicore
import spikeinterface.extractors as se
import warnings

warnings.filterwarnings('ignore')

# Load config
with open("paths.yaml") as f:
    paths_config = yaml.safe_load(f)

# Target session
session_key = paths_config["single_probe"]["coordinates_1"]["mouse01"]["sessions"]["session_1"]
sorted_path = Path(session_key["sorted"])
raw_path = Path(session_key["raw"])

print(f"Loading sorted data from: {sorted_path}")
print(f"Loading raw data from: {raw_path}\n")

# Load the sorting object
try:
    sorting = se.read_kilosort(sorted_path)
    print(f"[OK] Loaded {sorting.get_num_units()} units")
    print(f"[OK] Sampling rate: {sorting.get_sampling_frequency()} Hz")
except Exception as e:
    print(f"[ERROR] Error loading sorting: {e}")
    exit(1)

# Load recording and register it with sorting
try:
    recording = se.read_spikeglx(raw_path)
    sorting = sorting.register_recording(recording)
    print(f"[OK] Loaded recording: {recording.get_num_frames()} frames, {recording.get_num_channels()} channels\n")
except Exception as e:
    print(f"[WARNING] Could not load recording: {e}")
    print(f"Computing duration from spike data only\n")

# Get unit IDs
unit_ids = sorting.get_unit_ids()
print(f"Unit IDs: {unit_ids}\n")

# Compute basic spike train metrics
print("Computing quality metrics...")
metrics_dict = {}

for unit_id in unit_ids:
    spike_times = sorting.get_unit_spike_train(unit_id)

    # Firing rate - get duration in seconds
    try:
        duration_s = sorting.get_total_duration()
    except:
        # If recording not loaded, estimate from max spike time
        all_spikes = []
        for uid in unit_ids:
            all_spikes.extend(sorting.get_unit_spike_train(uid))
        max_spike = np.max(all_spikes) if len(all_spikes) > 0 else 1
        duration_s = max_spike / sorting.get_sampling_frequency()

    firing_rate = len(spike_times) / duration_s if duration_s > 0 else 0

    # ISI violations (refractory period)
    if len(spike_times) > 1:
        isi = np.diff(spike_times) / sorting.get_sampling_frequency()  # in seconds
        isi_violations = np.sum(isi < 0.0015) / len(isi)  # 1.5ms threshold
    else:
        isi_violations = np.nan

    # Presence ratio
    bin_size = 60  # 60 second bins
    n_bins = int(np.ceil(duration_s / bin_size))
    bins = np.linspace(0, duration_s * sorting.get_sampling_frequency(), n_bins + 1)
    spike_counts, _ = np.histogram(spike_times, bins=bins)
    presence_ratio = np.sum(spike_counts > 0) / n_bins

    metrics_dict[unit_id] = {
        'firing_rate_hz': firing_rate,
        'isi_violations': isi_violations,
        'presence_ratio': presence_ratio,
        'spike_count': len(spike_times)
    }

# Convert to DataFrame
metrics_df = pd.DataFrame(metrics_dict).T
metrics_df.index.name = 'unit_id'

print("\n" + "="*70)
print("SPIKE TRAIN METRICS")
print("="*70)
print(metrics_df.to_string())

# Apply quality thresholds
print("\n" + "="*70)
print("QUALITY CONTROL THRESHOLDS")
print("="*70)

thresholds = {
    'firing_rate_min': 0.5,      # Hz
    'isi_violations_max': 0.05,  # 5%
    'presence_ratio_min': 0.8,   # 80% of session
}

print(f"Firing rate minimum: {thresholds['firing_rate_min']} Hz")
print(f"ISI violations maximum: {thresholds['isi_violations_max']*100}%")
print(f"Presence ratio minimum: {thresholds['presence_ratio_min']}\n")

# Quality flag
metrics_df['passes_qc'] = (
    (metrics_df['firing_rate_hz'] >= thresholds['firing_rate_min']) &
    (metrics_df['isi_violations'] <= thresholds['isi_violations_max']) &
    (metrics_df['presence_ratio'] >= thresholds['presence_ratio_min'])
)

print("UNIT QC STATUS:")
print("-" * 70)
for unit_id, row in metrics_df.iterrows():
    status = "PASS" if row['passes_qc'] else "FAIL"
    print(f"Unit {unit_id}: {status}")
    print(f"  Firing rate: {row['firing_rate_hz']:.2f} Hz", end="")
    if row['firing_rate_hz'] < thresholds['firing_rate_min']:
        print(" [TOO LOW]")
    else:
        print()

    print(f"  ISI violations: {row['isi_violations']*100:.1f}%", end="")
    if row['isi_violations'] > thresholds['isi_violations_max']:
        print(" [TOO HIGH - contamination]")
    else:
        print()

    print(f"  Presence ratio: {row['presence_ratio']:.2f}", end="")
    if row['presence_ratio'] < thresholds['presence_ratio_min']:
        print(" [TOO LOW - inconsistent activity]")
    else:
        print()
    print()

# Summary
n_pass = metrics_df['passes_qc'].sum()
n_total = len(metrics_df)
print("="*70)
print(f"SUMMARY: {n_pass}/{n_total} units pass QC ({n_pass/n_total*100:.1f}%)")
print("="*70)

# Save to CSV
output_file = Path("data/unit_qc_metrics_session1_fed_exp.csv")
output_file.parent.mkdir(exist_ok=True, parents=True)
metrics_df.to_csv(output_file)
print(f"\n[OK] Saved metrics to: {output_file}")

print("\nNotes:")
print("- Firing rate < 0.5 Hz: likely noise or dead channel")
print("- ISI violations > 5%: unit may be contaminated by other neurons")
print("- Presence ratio < 0.8: unit is inconsistently active")
print("\nCompare these results with your phy manual curation!")
