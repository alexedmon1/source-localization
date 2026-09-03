# Validation Module

**Created:** 2025-12-01
**Updated:** 2026-01-30
**Version:** 0.4.0

## Overview

The `source_localization.validation` module provides comprehensive tools for validating source localization accuracy through dipole simulation and quantitative metrics. This is critical for assessing the reliability of source localization results before applying them to real experimental data.

## Validation Strategy

Based on the literature review in `documentation/VALIDATION_METHODS.md`, this module implements a **three-tier validation approach**:

### Tier 1: Computational Validation (Implemented)
- ✅ Simulated dipole validation with known ground truth
- ✅ Point Spread Function (PSF) analysis
- ✅ Crosstalk Function (CTF) analysis
- ✅ Localization error metrics
- ✅ ROI classification accuracy

### Tier 2: Sensitivity Analysis (Robustness Testing)
- ✅ SNR sensitivity testing (-20 to +40 dB range)
- ✅ Noise level sensitivity (0.01 to 10000 µV²)
- ✅ Dipole amplitude sensitivity (5 to 500 nAm)
- ✅ Multi-configuration comparison plots
- ⏭️ Conductivity sensitivity (±50% skull conductivity)
- ⏭️ Depth weighting parameter sweep

### Tier 3: Empirical Validation (Coming Soon)
- ⏭️ Electrode-level vs source-level comparison
- ⏭️ Multiple inverse method comparison

## Quick Start

### Batch Validation (Recommended)

Run validation across multiple presets and inverse methods with a single command:

```bash
# Run all presets with multiple methods
source-localization validate --batch \
    --all-presets \
    --methods sLORETA MNE eLORETA \
    --eeg data.set \
    --output ./validation_results

# Run specific presets
source-localization validate --batch \
    --presets shell_ellipsoid ellipsoid_cartesian \
    --methods sLORETA \
    --eeg data.set

# Compare existing results
source-localization validate --compare \
    results/config1/validation_results.json \
    results/config2/validation_results.json
```

### Python API for Batch Validation

```python
from source_localization.validation import BatchValidationRunner

# Initialize batch runner
runner = BatchValidationRunner(
    presets=['shell_ellipsoid', 'ellipsoid_cartesian', 'shell_sphere'],
    methods=['sLORETA', 'MNE', 'eLORETA'],
    eeg_file='data.set',
    output_dir='./validation_results',
    n_trials=10,
    snr_db=10.0,
    test_all_sources=True  # Test all sources (not a subset)
)

# Run validation for all configurations
results = runner.run_all(verbose=True)

# Generate comparison report
runner.generate_comparison_report()

# Access results
for config_name, result in results.items():
    print(f"{config_name}:")
    print(f"  Median error: {result.median_error_mm:.2f} mm")
    print(f"  ROI accuracy: {result.roi_accuracy:.1%}")
    print(f"  Error by depth: {result.error_by_depth}")
```

### Standardized Output Schema

All validation results follow a standardized JSON schema:

```json
{
  "preset": "shell_ellipsoid",
  "method": "sLORETA",
  "n_sources": 145,
  "n_simulations": 1450,
  "snr_db": 10.0,
  "n_trials": 10,
  "mean_error_mm": 2.68,
  "median_error_mm": 1.78,
  "std_error_mm": 2.45,
  "percentile_95_mm": 6.23,
  "roi_accuracy": 0.514,
  "roi_n_valid": 1420,
  "error_by_depth": {
    "0-1mm": {"mean": 0.5, "median": 0.3, "std": 0.4, "n": 50},
    "1-2mm": {"mean": 0.8, "median": 0.5, "std": 0.6, "n": 120},
    "2-3mm": {"mean": 1.2, "median": 0.8, "std": 0.9, "n": 200},
    "3-4mm": {"mean": 2.5, "median": 1.9, "std": 1.8, "n": 180},
    "4-5mm": {"mean": 3.5, "median": 3.0, "std": 2.2, "n": 150},
    "5+mm": {"mean": 4.2, "median": 3.9, "std": 2.5, "n": 200}
  },
  "timestamp": "2026-01-28T10:30:00",
  "duration_seconds": 245.3,
  "version": "1.0.0"
}
```

### Robustness Testing (Physics Validation)

Robustness testing validates that the source localization pipeline exhibits physically correct behavior: higher noise should produce worse localization, and higher signal should improve it.

#### Quick Start

```bash
# Run robustness test on a single pipeline
python -c "
from source_localization.validation import RobustnessTest

# Load from pipeline directory
test = RobustnessTest.from_pipeline_dir('./results/ellipsoid_roi_based')

# Run SNR sensitivity test
test.run_snr_test(snr_range=(-15, 30), n_positions=3, n_trials=8)

# Run noise level test
test.run_noise_test(noise_values=[0.1, 1.0, 10.0, 100.0, 1000.0], n_positions=3, n_trials=8)

# Save results and generate plots
test.save_results('./results/robustness/robustness_results.json')
test.generate_report('./results/robustness/robustness_report.txt')
test.plot_results('./results/robustness/')
"
```

#### Python API

```python
from source_localization.validation import RobustnessTest

# Initialize from pipeline output directory
test = RobustnessTest.from_pipeline_dir(
    pipeline_dir='./results/ellipsoid_roi_based',
    inverse_method='sLORETA',  # 'sLORETA', 'MNE', or 'dSPM'
    inverse_snr=3.0
)

# Run individual tests
snr_results = test.run_snr_test(
    snr_range=(-20, 40),    # dB range
    snr_step=5,              # Step size in dB
    n_positions=4,           # Test positions at different depths
    n_trials=10,             # Trials per position per SNR level
    amplitude_nAm=50.0       # Fixed dipole amplitude
)

noise_results = test.run_noise_test(
    noise_range=(0.01, 10000),  # µV² range (log-spaced)
    n_levels=8,                  # Number of noise levels
    n_positions=4,
    n_trials=10,
    amplitude_nAm=50.0           # Fixed amplitude
)

amplitude_results = test.run_amplitude_test(
    amplitude_range=(5, 500),    # nAm range (log-spaced)
    n_levels=8,
    n_positions=4,
    n_trials=10,
    noise_variance_uV2=1.0       # Fixed noise
)

# Or run all tests at once
all_results = test.run_all_tests()

# Check physics validation
summary = test.get_summary()
for test_name, stats in summary['tests'].items():
    status = "✓ PASSED" if stats['physics_valid'] else "✗ FAILED"
    print(f"{test_name}: r={stats['correlation']:.3f} {status}")

# Generate outputs
test.save_results('robustness_results.json')
test.generate_report('robustness_report.txt', config_name='ellipsoid_roi_based')
test.plot_results('./figures/', config_name='ROI-based')
```

#### Multi-Configuration Comparison

Compare robustness across multiple pipeline configurations:

```python
from source_localization.validation import RobustnessTest

# Load results from multiple configs
config_results = {
    'ROI-based': RobustnessTest.load_results('roi/robustness_results.json'),
    'Cartesian': RobustnessTest.load_results('cart/robustness_results.json'),
    'Shell': RobustnessTest.load_results('shell/robustness_results.json'),
}

# Generate 2x2 comparison plot
colors = {
    'ROI-based': '#2ecc71',
    'Cartesian': '#3498db',
    'Shell': '#e74c3c',
}
RobustnessTest.plot_multi_config_summary(config_results, './figures/', colors=colors)
```

This generates `robustness_summary_2x2.png` with:
- **A)** SNR vs Error (all configs overlaid)
- **B)** Noise Level vs Error (all configs overlaid)
- **C)** SNR-Error correlations bar chart
- **D)** Error at different SNR levels (grouped bars)

#### Expected Physics Behavior

| Test | Expected Correlation | Physics Validation |
|------|---------------------|-------------------|
| **SNR** | r < -0.5 (negative) | Higher SNR → Lower error |
| **Noise** | r > +0.5 (positive, log scale) | Higher noise → Higher error |
| **Amplitude** | r < -0.3 (negative, log scale) | Higher amplitude → Lower error |

#### Output Files

```
robustness/
├── robustness_results.json       # Full data (errors, correlations, metadata)
├── robustness_report.txt         # Human-readable summary
├── robustness_snr_detailed.png   # SNR test with scatter + regression
├── robustness_noise_detailed.png # Noise test with scatter + regression
├── robustness_snr_comparison.png # Bar chart of low/mid/high SNR errors
└── robustness_summary.png        # Combined overview (if multiple tests)
```

#### Example Results

From validation on ellipsoid BEM configurations (January 2026):

| Configuration | SNR Correlation | Noise Correlation | Physics Valid |
|---------------|-----------------|-------------------|---------------|
| ROI-based (230 sources) | r = -0.67 | r = +0.69 | ✓ |
| Cartesian (1847 sources) | r = -0.85 | r = +0.87 | ✓ |
| Shell (145 sources) | r = -0.83 | r = +0.87 | ✓ |

All configurations show correct physics: localization error decreases with better signal quality.

### Depth-Stratified Metrics

Compute localization error stratified by distance from electrodes:

```python
from source_localization.validation import (
    compute_source_depths,
    compute_depth_stratified_error,
    DEFAULT_DEPTH_BINS_MM
)

# Compute depths for all sources
depths = compute_source_depths(
    source_coords_mm=source_positions,
    electrode_coords_mm=electrode_positions,
    method='min'  # Minimum distance to any electrode
)

# After running validation, stratify errors by depth
results = [
    {'depth': depths[i], 'localization_error': errors[i]}
    for i in range(len(depths))
]

error_by_depth = compute_depth_stratified_error(
    results,
    depth_bins=DEFAULT_DEPTH_BINS_MM  # [(0,1,'0-1mm'), (1,2,'1-2mm'), ...]
)

for bin_name, stats in error_by_depth.items():
    print(f"{bin_name}: mean={stats['mean']:.2f}mm, n={stats['n']}")
```

### Basic Usage

```python
from source_localization import Pipeline
from source_localization.validation import (
    DipoleSimulator,
    compute_localization_error,
    compute_roi_classification_accuracy,
    create_validation_report
)

# Run a pipeline
pipeline = Pipeline.from_preset('sphere_volumetric')
results = pipeline.run(eeg_file='data.set')

# Initialize dipole simulator
simulator = DipoleSimulator(
    forward_model=results['forward_solution']['forward'],
    info=results['eeg_data']['info'],
    source_space=results['source_space']['source_space']
)

# Simulate dipole at known location
eeg_data, metadata = simulator.simulate_dipole(
    position_mm=[0, 0, 5],  # Target position in mm
    amplitude_nAm=50.0,      # Dipole amplitude
    snr_db=10.0              # Signal-to-noise ratio
)

print(f"Simulated dipole:")
print(f"  Target: {metadata['dipole_position_mm']} mm")
print(f"  Actual: {metadata['actual_position_mm']} mm")
print(f"  Error: {metadata['position_error_mm']:.3f} mm")
```

### Running Validation Tests

```python
# Simulate dipoles at all ROI centroids
from source_localization.validation import load_atlas_roi_centroids, load_roi_source_mapping

# Load ROI centroids
roi_centroids, roi_names = load_atlas_roi_centroids(
    atlas_file='data/atlas/Atlas_3DRois.nii',
    labels_file='data/atlas/roi_mapping.json'
)

# Map sources to ROIs
source_positions = results['source_space']['positions_mm']
roi_mapping = load_roi_source_mapping(source_positions, 'data/atlas/Atlas_3DRois.nii')

# Run validation for each ROI
validation_results = []

for roi_id, centroid in roi_centroids.items():
    # Simulate dipole
    eeg_data, sim_meta = simulator.simulate_dipole(
        position_mm=centroid,
        amplitude_nAm=50.0,
        snr_db=3.0
    )

    # Apply inverse solution (your inverse solution code here)
    # ...
    # stc = apply_inverse_solution(eeg_data)

    # Compute metrics
    error = compute_localization_error(
        true_position_mm=sim_meta['actual_position_mm'],
        estimated_position_mm=peak_source_position
    )

    roi_correct = compute_roi_classification_accuracy(
        true_roi=roi_id,
        estimated_roi=estimated_roi_id
    )

    validation_results.append({
        'roi_id': roi_id,
        'roi_name': roi_names[roi_id],
        'localization_error_mm': error,
        'roi_correct': roi_correct
    })

# Generate report
report_path = create_validation_report(
    results=validation_results,
    pipeline_name='sphere_volumetric',
    output_dir='validation_results/',
    roi_names=roi_names
)

print(f"Validation report generated: {report_path}")
```

## Validation Metrics

### 1. Localization Error

**Euclidean distance** between true and estimated source positions.

```python
from source_localization.validation import compute_localization_error

error_mm = compute_localization_error(
    true_position_mm=[0, 0, 5],
    estimated_position_mm=[0.2, -0.1, 5.3]
)
print(f"Error: {error_mm:.2f} mm")
```

**Interpretation:**
- <1 mm: Excellent (Lee et al. 2013 achieved <1.2 mm in mice)
- 1-3 mm: Good (acceptable for mouse brain scale ~8mm radius)
- 3-5 mm: Fair (may confuse adjacent ROIs)
- >5 mm: Poor (likely misidentifies brain regions)

### 2. ROI Classification Accuracy

**Binary classification:** Does the estimated peak source fall in the correct ROI?

```python
from source_localization.validation import compute_roi_classification_accuracy

correct = compute_roi_classification_accuracy(
    true_roi=5,         # Thalamus_L
    estimated_roi=5     # Also Thalamus_L
)
print(f"Correct: {correct}")  # True
```

### 3. Hierarchical ROI Accuracy

**Fuzzy matching** with spatial proximity between ROI centroids.

```python
from source_localization.validation import compute_hierarchical_roi_accuracy

accuracy = compute_hierarchical_roi_accuracy(
    true_roi=5,              # Thalamus_L
    estimated_roi=29,        # Hypothalamus_L (nearby)
    roi_distance_matrix=distances,
    roi_ids=roi_ids
)

print(accuracy)
# {'exact': False, 'adjacent': False, 'nearby': True, 'close': True,
#  'same_hemisphere': True, 'distance_mm': 2.2}
```

**Hierarchy levels:**
1. Exact: Same ROI (0 mm)
2. Adjacent: ≤2 mm between centroids
3. Nearby: ≤3 mm between centroids
4. Close: ≤5 mm between centroids
5. Same hemisphere: Left vs right

### 4. Point Spread Function (PSF)

**Spatial resolution** metrics quantifying source localization spread.

```python
from source_localization.validation import compute_point_spread_function

psf_metrics = compute_point_spread_function(
    source_power=source_power_avg,      # Time-averaged power
    source_positions_mm=source_positions,
    peak_idx=peak_source_index,
    threshold_fraction=0.5               # FWHM
)

print(psf_metrics)
# {'fwhm_mm': 2.5,
#  'n_sources_active': 15,
#  'spread_volume_mm3': 65.4,
#  'centroid_shift_mm': 0.3}
```

### 5. Crosstalk Function (CTF)

**Power leakage** from true source ROI to incorrect ROIs.

```python
from source_localization.validation import compute_crosstalk_function

ctf_metrics = compute_crosstalk_function(
    source_power=source_power_avg,
    roi_mapping=roi_mapping,
    true_roi=5,
    top_n=5
)

print(ctf_metrics)
# {'true_roi_power_fraction': 0.75,    # 75% of power in correct ROI
#  'top_rois': [(5, 0.75), (6, 0.12), (29, 0.08), ...],
#  'n_rois_active': 8,
#  'ctf_ratio': 6.25}                  # 6.25× stronger than next ROI
```

## Advanced Usage

### Two-Dipole Resolution Test

Test ability to resolve two nearby sources:

```python
# Simulate two dipoles
eeg_data, metadata = simulator.simulate_two_dipoles(
    position1_mm=[0, 0, 5],
    position2_mm=[2, 0, 5],  # 2mm apart
    amplitude1_nAm=50.0,
    amplitude2_nAm=50.0,
    snr_db=10.0
)

print(f"Dipole separation: {metadata['separation_mm']:.2f} mm")

# After inverse solution, check if both peaks were recovered
from source_localization.validation import compute_two_dipole_resolution

resolution = compute_two_dipole_resolution(
    source_power=source_power_avg,
    source_positions_mm=source_positions,
    true_pos1_mm=metadata['dipole1']['actual_position_mm'],
    true_pos2_mm=metadata['dipole2']['actual_position_mm'],
    min_separation_mm=2.0
)

print(resolution)
# {'resolved': True, 'n_peaks': 2,
#  'peak1_error_mm': 0.5, 'peak2_error_mm': 0.8,
#  'estimated_separation_mm': 2.1}
```

### ROI Dipole Simulation

Simulate dipole at ROI centroid:

```python
# Get sources for specific ROI
roi_sources = np.where(roi_mapping == 15)[0]  # Primary Visual Cortex

# Simulate dipole at ROI center
eeg_data, metadata = simulator.simulate_roi_dipole(
    roi_sources=roi_sources,
    roi_label="Primary Visual Cortex",
    amplitude_nAm=60.0,
    snr_db=12.0
)

print(f"Simulated {metadata['roi_label']} with {metadata['roi_n_sources']} sources")
print(f"ROI centroid: {metadata['roi_centroid_mm']} mm")
```

## Validation Reports

The `create_validation_report()` function generates comprehensive markdown reports with:

### Report Sections

1. **Summary Statistics**
   - Mean/median/std localization error
   - 95th percentile error
   - ROI classification accuracy

2. **Validation Figures**
   - Localization error distribution (histogram + boxplot)
   - Per-ROI classification accuracy (bar plot)
   - PSF analysis (FWHM distribution)
   - Error vs depth (scatter + trend line)

3. **Per-ROI Results Table**
   - ROI ID, name, mean error, accuracy, n_tests

### Example Report

```python
from source_localization.validation import create_validation_report

report_path = create_validation_report(
    results=validation_results,        # List of dicts with metrics
    pipeline_name='sphere_volumetric',
    output_dir='validation_results/',
    roi_names=roi_names
)

# Output:
# validation_results/
# ├── sphere_volumetric_validation_report.md
# └── figures/
#     ├── sphere_volumetric_localization_error.png
#     ├── sphere_volumetric_roi_accuracy.png
#     ├── sphere_volumetric_psf_analysis.png
#     └── sphere_volumetric_error_vs_depth.png
```

## Expected Performance

Based on Lee et al. (2013) - the gold standard mouse validation study using optogenetic ground truth:

| Metric | Target | Lee et al. (2013) |
|--------|--------|-------------------|
| **Localization Error** | | |
| Mean | <3 mm | <1.2 mm |
| 95th percentile | <5 mm | - |
| **ROI Classification** | | |
| Accuracy | >80% | - |
| **PSF (MNE method)** | | |
| ROC AUC | >0.95 | >0.99 |
| False Positive Rate @ 90% sensitivity | <10% | <1.3% |

## Critical: Depth-Dependent Performance

**⚠️ Validation metrics MUST be stratified by source depth from electrodes.**

Our validation (January 2026) revealed that **localization accuracy degrades significantly with depth**:

### Empirical Results (Ellipsoid BEM, Volumetric Sources, SNR=10dB)

| Depth from Electrodes | sLORETA Error | sLORETA ROI Acc | MNE Error | MNE ROI Acc |
|-----------------------|---------------|-----------------|-----------|-------------|
| **0-3 mm** (superficial) | **0.97 mm** | **65%** | 1.26 mm | 63% |
| 3-4 mm | 3.58 mm | 47% | 4.38 mm | 0% |
| 4-5 mm | 3.91 mm | 20% | 5.44 mm | 0% |
| **5-6 mm** (deep) | 4.39 mm | 10% | 6.42 mm | 0% |

### Key Implications

1. **Report depth-stratified metrics**: A single mean localization error is misleading. Always report by depth bin.

2. **Superficial sources are ~4× more accurate**: Sources within 3mm of electrodes have <1mm error; deep sources have >4mm error.

3. **MNE has superficial bias**: ROI accuracy drops to 0% for depths >3mm with MNE. sLORETA maintains some accuracy even at depth.

4. **Mouse brain depth range**: With electrodes at Z≈3mm and brain extending to Z≈-4mm, most brain volume is 3-7mm from electrodes.

### Computing Source Depth

```python
def compute_depth_from_electrodes(source_position_mm, electrode_positions_mm):
    """Depth = minimum distance from source to any electrode."""
    distances = np.linalg.norm(electrode_positions_mm - source_position_mm, axis=1)
    return float(np.min(distances))
```

### Recommended Depth Bins

For mouse EEG with dorsal electrode arrays:
- **Superficial (0-3mm)**: Cortical surface, well-localized
- **Mid-depth (3-5mm)**: Subcortical, moderate accuracy
- **Deep (>5mm)**: Thalamus/hypothalamus, poor accuracy

## ROI Classification: Beyond Binary Accuracy

Simple "correct ROI or not" is insufficient because:
1. ROIs vary in size (65-2000+ voxels)
2. Adjacent ROIs may have overlapping neural activity
3. Deep sources have large spatial uncertainty

### Proposed: Confidence-Weighted ROI Classification

Instead of binary ROI assignment, compute a **probability distribution over ROIs**:

```python
def compute_roi_confidence_map(source_power, roi_mapping, source_positions_mm):
    """
    Returns confidence (0-1) for each ROI based on power distribution.

    Parameters
    ----------
    source_power : array (n_sources,)
        Time-averaged power at each source
    roi_mapping : array (n_sources,)
        ROI label for each source
    source_positions_mm : array (n_sources, 3)
        Source coordinates

    Returns
    -------
    roi_confidence : dict
        {roi_id: confidence} where confidence = fraction of total power
    """
    total_power = np.sum(source_power)
    roi_confidence = {}

    for roi_id in np.unique(roi_mapping):
        if roi_id == 0:  # Skip background
            continue
        mask = roi_mapping == roi_id
        roi_power = np.sum(source_power[mask])
        roi_confidence[roi_id] = roi_power / total_power

    return roi_confidence
```

### Visualization: False-Colored Confidence Grid

Generate a volumetric confidence map showing localization uncertainty:

```python
def create_confidence_volume(source_power, source_positions_mm,
                              volume_shape, voxel_size_mm):
    """
    Create 3D volume with source power interpolated to voxel grid.

    Returns
    -------
    confidence_volume : array (X, Y, Z)
        Power normalized to 0-1 range at each voxel
    """
    from scipy.interpolate import griddata

    # Create voxel grid
    x = np.arange(0, volume_shape[0]) * voxel_size_mm[0]
    y = np.arange(0, volume_shape[1]) * voxel_size_mm[1]
    z = np.arange(0, volume_shape[2]) * voxel_size_mm[2]
    grid = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.stack([g.ravel() for g in grid], axis=1)

    # Interpolate source power to voxel grid
    confidence = griddata(source_positions_mm, source_power,
                          grid_points, method='linear', fill_value=0)
    confidence = confidence.reshape(volume_shape)

    # Normalize to 0-1
    confidence /= np.max(confidence)

    return confidence
```

### Proposed Metrics for Comprehensive ROI Classification

| Metric | Description | Use Case |
|--------|-------------|----------|
| **Top-1 Accuracy** | Is highest-power ROI correct? | Current binary metric |
| **Top-3 Accuracy** | Is true ROI in top 3? | Accounts for uncertainty |
| **Confidence Ratio** | Power in true ROI / power in top incorrect ROI | Discrimination strength |
| **Entropy** | -Σ p·log(p) over ROI confidences | Localization certainty |
| **Depth-Weighted Accuracy** | Accuracy weighted by 1/depth | Fair comparison across depths |

### Example Output Format

```json
{
  "source_localization": {
    "peak_position_mm": [1.2, 0.5, 2.1],
    "depth_from_electrodes_mm": 2.3,
    "confidence_category": "high",

    "roi_confidences": {
      "Primary_Motor_Cortex_L": 0.42,
      "Primary_Somatosensory_Cortex_L": 0.28,
      "Secondary_Motor_Cortex_L": 0.15,
      "other": 0.15
    },

    "metrics": {
      "top1_roi": "Primary_Motor_Cortex_L",
      "top1_confidence": 0.42,
      "confidence_ratio": 1.5,
      "entropy": 1.82,
      "localization_error_mm": 1.1
    }
  }
}
```

### Confidence Categories

Based on depth and power distribution:

| Category | Depth | Top-1 Confidence | Expected Error |
|----------|-------|------------------|----------------|
| **High** | <3mm | >40% | <1.5 mm |
| **Medium** | 3-5mm | 20-40% | 1.5-4 mm |
| **Low** | >5mm | <20% | >4 mm |

## API Reference

### Classes

#### `DipoleSimulator`
```python
DipoleSimulator(forward_model, info, source_space, verbose=True)
```

**Methods:**
- `simulate_dipole(position_mm, orientation, amplitude_nAm, snr_db, ...)`
- `simulate_roi_dipole(roi_sources, roi_label, **kwargs)`
- `simulate_two_dipoles(position1_mm, position2_mm, ...)`
- `create_mne_raw(eeg_data, sfreq)`

### Functions

#### Metrics
- `compute_localization_error(true_position_mm, estimated_position_mm)`
- `compute_roi_classification_accuracy(true_roi, estimated_roi)`
- `compute_hierarchical_roi_accuracy(true_roi, estimated_roi, roi_distance_matrix, roi_ids)`
- `compute_point_spread_function(source_power, source_positions_mm, peak_idx, threshold_fraction)`
- `compute_crosstalk_function(source_power, roi_mapping, true_roi, top_n)`
- `compute_amplitude_recovery(true_amplitude_nAm, estimated_peak_power, estimated_total_power)`
- `compute_depth_bias(localization_errors_mm, source_depths_mm)`
- `compute_two_dipole_resolution(source_power, source_positions_mm, true_pos1_mm, true_pos2_mm)`
- `summarize_validation_results(results)`
- `compute_source_depths(source_coords_mm, electrode_coords_mm, method='min')`
- `compute_depth_stratified_error(results, depth_bins=None)`

#### Batch Validation
- `BatchValidationRunner(presets, methods, eeg_file, output_dir, n_trials, snr_db)`
- `BatchValidationRunner.run_all(verbose=True)` → Dict[str, ValidationOutputSchema]
- `BatchValidationRunner.generate_comparison_report()` → Path
- `BatchValidationRunner.compare_results(result_paths)` → Dict (static method)
- `ValidationOutputSchema.to_json(path)` - Save results to JSON
- `ValidationOutputSchema.from_json(path)` - Load results from JSON

#### Robustness Testing
- `RobustnessTest(fwd, src, info, inverse_method, inverse_snr, verbose)` - Initialize from MNE objects
- `RobustnessTest.from_pipeline_dir(pipeline_dir, inverse_method, ...)` - Initialize from pipeline output
- `RobustnessTest.run_snr_test(snr_range, snr_step, n_positions, n_trials, ...)` → RobustnessResults
- `RobustnessTest.run_noise_test(noise_range, n_levels, n_positions, n_trials, ...)` → RobustnessResults
- `RobustnessTest.run_amplitude_test(amplitude_range, n_levels, n_positions, n_trials, ...)` → RobustnessResults
- `RobustnessTest.run_all_tests(...)` → Dict[str, RobustnessResults]
- `RobustnessTest.get_summary()` → dict with correlation and physics validation status
- `RobustnessTest.plot_results(output_dir, config_name, ...)` - Generate individual test plots
- `RobustnessTest.plot_multi_config_summary(config_results, output_dir, colors)` - Static method for 2x2 comparison
- `RobustnessTest.save_results(output_path)` - Save to JSON
- `RobustnessTest.load_results(input_path)` - Load from JSON (static)
- `RobustnessTest.generate_report(output_path, config_name)` → str - Generate text report
- `RobustnessResults` - Dataclass with `correlation`, `means`, `stds`, `medians` properties

#### Visualization
- `generate_localization_error_map(pipeline_dir, output_dir)` - Estimated errors from curves
- `generate_validated_error_map(source_coords, errors, electrode_coords, ...)` - Measured errors
- `create_localization_error_figure(source_coords, errors, electrode_coords, ...)`
- `create_error_colormap()` - White-to-red colormap for error visualization

#### Utilities
- `load_atlas_roi_centroids(atlas_file, labels_file)`
- `load_roi_source_mapping(source_positions_mm, atlas_file, verbose, radius_mm)`
- `get_roi_sources(roi_id, roi_mapping)`
- `create_validation_report(results, pipeline_name, output_dir, roi_names)`

## Implementation Notes

### Antwerp Atlas 10× Voxel Correction

The Antwerp Mouse Brain Atlas has voxel sizes that are **10× larger** than reality in the NIfTI header. The validation module automatically applies this correction:

```python
# In load_atlas_roi_centroids() and load_roi_source_mapping()
affine_corrected = atlas_nii.affine.copy()
affine_corrected[:3, :3] /= 10.0  # Scale voxel sizes
affine_corrected[:3, 3] /= 10.0   # Scale translation
```

This ensures correct spatial coordinates for:
- ROI centroid positions
- Source-to-ROI mapping
- Distance calculations

### Proximity-Based ROI Mapping

Sources are mapped to ROIs using KD-tree proximity search within 1.0 mm radius (configurable). This is more robust than exact voxel lookup for:
- Surface sources (may fall between voxels)
- Volumetric sources at atlas boundaries
- Handling numerical precision issues

## References

1. **Lee et al. (2013)** - "Dipole Source Localization of Mouse Electroencephalogram Using the FieldTrip Toolbox." *PLoS ONE* 8(11): e79442.
   - Gold standard mouse validation using optogenetic ground truth
   - Achieved <1.2 mm localization accuracy
   - ROC AUC >0.99 for MNE method

2. **Documentation/VALIDATION_METHODS.md**
   - Comprehensive literature review
   - Validation strategy and metrics
   - Expected accuracy targets

3. **MNE-Python Resolution Metrics**
   - PSF/CTF analysis methods
   - Point localization error (PLE)
   - Spatial deviation (SD)

## Related Modules

- `source_localization.steps.inverse_solution` - Inverse solution methods (MNE, dSPM, sLORETA, eLORETA, LCMV, DICS)
- `source_localization.utils.atlas` - Atlas utilities
- `source_localization.utils.roi` - ROI extraction
- `adv_test/validation/` - Original reference implementation

## Implementation Status

### Completed (v0.4.0, January 2026)

#### Robustness Testing (NEW)
- [x] `RobustnessTest` class for physics validation
- [x] SNR sensitivity testing (-20 to +40 dB range)
- [x] Noise level sensitivity testing (0.01 to 10000 µV², log-spaced)
- [x] Dipole amplitude sensitivity testing (5 to 500 nAm, log-spaced)
- [x] Automatic physics validation (correlation thresholds)
- [x] JSON export with `save_results()` / `load_results()`
- [x] Text report generation with `generate_report()`
- [x] Individual test plots with scatter, mean lines, and regression
- [x] Multi-configuration comparison with `plot_multi_config_summary()` (2x2 layout)

#### Depth-Stratified Metrics
- [x] `compute_source_depths()` - Compute distance from sources to electrode array
- [x] `compute_depth_stratified_error()` - Bin errors by depth with standardized bins
- [x] `DEFAULT_DEPTH_BINS_MM` - Standard depth bins: 0-1mm, 1-2mm, 2-3mm, 3-4mm, 4-5mm, 5+mm

#### Batch Validation
- [x] `BatchValidationRunner` class for multi-preset, multi-method testing
- [x] `ValidationOutputSchema` - Standardized JSON output format
- [x] Automatic comparison report generation
- [x] CLI integration: `--batch`, `--all-presets`, `--methods`, `--compare`

#### Visualization
- [x] Localization error maps with white-to-red colormap
- [x] `generate_validated_error_map()` for actual measured errors
- [x] Three-view brain visualization (top, sagittal, coronal)
- [x] Depth vs error scatter plots with trend lines

### Future Work

#### Medium Priority
- [ ] CLI integration for robustness testing: `source-localization validate --robustness`
- [ ] Implement `compute_roi_confidence_map()` function
- [ ] Add confidence-weighted ROI classification metrics
- [ ] Create 3D confidence volume visualization
- [ ] Add false-colored uncertainty overlay for brain atlas

#### Lower Priority
- [ ] Conductivity sensitivity testing (±50% skull conductivity)
- [ ] Depth weighting parameter sweep
- [ ] Write comprehensive unit tests
- [ ] Add integration tests with real pipeline outputs
