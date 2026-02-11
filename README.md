# Mouse EEG Source Localization Package

**Created:** 2025-11-26
**Last Updated:** 2026-01-30
**Version:** 1.4.1
**Status:** Production Ready

A complete, validated Python package for mouse EEG source localization using the Antwerp Mouse Brain Atlas. Includes multi-subject batch processing, MNE-based spectral/connectivity analysis, and publication-quality visualizations.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Multi-Subject Study Processing](#multi-subject-study-processing)
- [Spectral and Connectivity Analysis](#spectral-and-connectivity-analysis)
- [Configuration Presets](#configuration-presets)
- [Usage](#usage)
  - [Command Line Interface](#command-line-interface)
  - [Python API](#python-api)
- [Publication-Quality Visualizations](#publication-quality-visualizations)
- [Validation](#validation)
- [Pipeline Architecture](#pipeline-architecture)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

---

## Overview

This package provides a complete, validated pipeline for performing source localization on mouse EEG data. It implements multiple head models (BEM), source space configurations, and inverse methods to estimate brain activity from scalp EEG recordings.

### What is Source Localization?

Source localization solves the EEG inverse problem: given electrode measurements on the scalp, estimate the locations and strengths of neural activity sources within the brain. This allows researchers to:

- Identify which brain regions are active during specific tasks or conditions
- Compute region-of-interest (ROI) level metrics for statistical analysis
- Compare activity patterns across experimental groups
- Validate electrode-level findings with anatomical specificity

---

## Features

### Core Pipeline
- **8 validated pipeline presets** (4 source types × 2 BEM types)
- **2 BEM types**: Sphere (analytical, fast) and Ellipsoid (numerical, accurate)
- **4 source space types**:
  - **Surface**: Icosphere mesh on brain surface
  - **ROI-based**: Sources at atlas ROI centroids
  - **Cartesian**: 3D volumetric grid
  - **Shell**: Concentric geometry-matched shells (best conditioning)
- **5 inverse methods**: MNE, dSPM, sLORETA, LCMV, DICS beamformers
- **Packaged Antwerp Mouse Brain Atlas** (47 ROIs, plus 22-ROI coarse atlas)
- **32-channel electrode array** coordinates included

### Multi-Subject Study Processing (NEW in v1.3.0)
- **BIDS-inspired folder hierarchy** for organized data management
- **Batch processing** with parallel job support
- **Study configuration** via YAML files
- **Progress tracking** and status reporting
- **Group-level result collection**

### MNE-Based Analysis (NEW in v1.3.0)
- **Band power analysis** using MNE's optimized Welch PSD
- **Connectivity analysis** via MNE-Connectivity (coherence, PLV, wPLI, imcoh)
- **Automatic epoching** of continuous data for connectivity
- **Results saved** to study folder hierarchy
- **Group-level aggregation** of results

### Depth-Weighted ROI Extraction
- **Empirically-validated depth weighting**: 0-1mm: 77%, 1-2mm: 36%, 2-3mm: 4%, >3mm: ~0%
- **ROI time series** weighted by localization accuracy at each depth
- **MNE-compatible output** (.set files loadable in MNE/EEGLAB)

### Publication-Quality Visualizations
- **Smooth interpolated heatmaps** (like fMRI activation maps)
- **ROI parcellation overlays** with boundaries and labels
- **Connectivity visualizations**: matrices, chord diagrams, brain networks
- **Custom neuroimaging colormaps**: `hot_black`, `diverging_bwr`
- **Publication presets**: 300 DPI SVG/PDF output

### Best Performing Configuration

**Recommended:** `roi_based_ellipsoid` preset
- **ROI classification accuracy:** 76.9% (validated on dipole simulations)
- **Mean localization error:** 1.67 mm
- **Anatomically accurate** ellipsoidal head model
- **Optimized for statistical modeling** (low inter-ROI collinearity)

---

## Installation

### Prerequisites

- Python >= 3.8
- Virtual environment manager (uv recommended)

### Using uv (Recommended)

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/drpedapati/AlexProjects.git
cd AlexProjects/mouse-eeg-source-localization/source_localization
uv venv && source .venv/bin/activate
uv pip install -e .

# For connectivity analysis (optional)
uv pip install -e ".[connectivity]"
```

### Verify Installation

```bash
source-localization --help
source-localization study --help
```

---

## Quick Start

### Single Subject

```bash
# Run source localization on one EEG file
source-localization run --preset roi_based_ellipsoid --eeg /path/to/data.set --output ./results

# View results
open results/pipeline_report.html
```

### Multi-Subject Study

```bash
# 1. Initialize study from a folder of EEG files
source-localization study init /path/to/eeg_data --name "MyStudy"

# 2. Process all subjects (4 parallel jobs)
source-localization study run study_config.yaml --jobs 4

# 3. Run spectral and connectivity analysis
source-localization study analyze study_config.yaml --verbose

# 4. Check status
source-localization study status study_config.yaml
```

---

## Multi-Subject Study Processing

The study module provides a BIDS-inspired framework for organizing and processing multi-subject EEG studies.

### Folder Hierarchy

```
study_folder/
├── study_config.yaml           # Study configuration
├── sourcedata/                 # Raw EEG files (or symlinks)
│   ├── sub-001.set
│   └── sub-002.set
├── participants.csv            # Subject metadata (optional)
└── derivatives/
    └── source_localization/
        ├── sub-001/
        │   ├── pipeline/       # Source localization outputs
        │   ├── roi_timeseries/ # ROI time series (.set files)
        │   └── analysis/       # Band power, connectivity
        └── group/
            ├── group_band_power.csv
            └── connectivity_*.csv
```

### CLI Commands

```bash
# Initialize study from folder
source-localization study init /path/to/data --name "MyStudy" --preset roi_based_ellipsoid

# Run source localization pipeline
source-localization study run study_config.yaml --jobs 4 --verbose

# Run spectral/connectivity analysis (uses MNE)
source-localization study analyze study_config.yaml --bands delta theta alpha beta gamma --connectivity coherence

# Check processing status
source-localization study status study_config.yaml

# Collect group results
source-localization study collect study_config.yaml
```

### Python API

```python
from source_localization.study import (
    StudyConfig,
    process_study,
    create_study_from_folder,
    analyze_study
)

# Create study from folder
config = create_study_from_folder(
    folder='/path/to/eeg_data',
    name='MyStudy',
    preset='roi_based_ellipsoid'
)

# Process all subjects
result = process_study(config, n_jobs=4)
print(f"Processed {result.n_completed} subjects, {result.n_failed} failed")

# Run analysis
df = analyze_study(
    config,
    bands={'gamma': (30, 80), 'theta': (4, 8)},
    connectivity_methods=['coherence', 'plv'],
    n_jobs=4
)
```

---

## Spectral and Connectivity Analysis

The analysis module provides wrapper functions around MNE-Python and MNE-Connectivity for batch analysis of processed subjects.

### Why Use MNE Wrappers?

- **Optimized implementations**: MNE's algorithms are well-tested and performant
- **Standardized methods**: Same algorithms used in human neuroimaging
- **Organized outputs**: Results saved to study folder hierarchy
- **Group aggregation**: Automatic collection of subject-level results

### Band Power Analysis

Computes power spectral density per ROI using Welch's method:

```python
from source_localization.study import analyze_subject, DEFAULT_BANDS

# Analyze single subject
result = analyze_subject(
    subject_dir='derivatives/source_localization/sub-001',
    bands=DEFAULT_BANDS,  # delta, theta, alpha, beta, gamma
    overwrite=False
)

# Output: analysis/band_power.csv
# Columns: roi, band, fmin, fmax, power, power_db
```

### Connectivity Analysis

Computes ROI-to-ROI connectivity using MNE-Connectivity:

```python
# Requires: pip install mne-connectivity
result = analyze_subject(
    subject_dir='derivatives/source_localization/sub-001',
    connectivity_methods=['coherence', 'plv', 'wpli'],
    connectivity_bands=['gamma', 'theta'],
    epoch_length=2.0  # For continuous data, create 2s epochs
)

# Output: analysis/connectivity_coherence_gamma.csv (46x46 matrix)
```

### Available Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `coherence` | Magnitude-squared coherence | Default, linear relationships |
| `plv` | Phase-locking value | Phase synchronization |
| `wpli` | Weighted phase-lag index | Volume conduction robust |
| `imcoh` | Imaginary coherence | Zero-lag artifact removal |

### CLI Usage

```bash
# Run analysis on all subjects
source-localization study analyze study_config.yaml \
    --bands delta theta alpha beta gamma \
    --connectivity coherence plv \
    --epoch-length 2.0 \
    --jobs 4 \
    --verbose
```

---

## Configuration Presets

### Available Presets (8 total)

| Preset | BEM | Source Type | Sources | Use Case |
|--------|-----|-------------|---------|----------|
| **`roi_based_ellipsoid`** | Ellipsoid | ROI-based | ~200 | **Statistical modeling (LMMs)** |
| `roi_based_sphere` | Sphere | ROI-based | ~200 | Fast ROI-based |
| `ellipsoid_surface` | Ellipsoid | Surface | 73 | Best spatial accuracy |
| `sphere_surface` | Sphere | Surface | 73 | Fast prototyping |
| `ellipsoid_cartesian` | Ellipsoid | Cartesian | ~200 | Dense volumetric grid |
| `sphere_cartesian` | Sphere | Cartesian | ~500 | Maximum volumetric sources |
| **`shell_ellipsoid`** | Ellipsoid | Shell | ~400 | **Best conditioning, whole-brain** |
| `shell_sphere` | Sphere | Shell | ~400 | Fast shell-based |

### Source Space Types

| Type | Description | Conditioning | Best For |
|------|-------------|--------------|----------|
| **Surface** | Icosphere mesh on brain surface | Excellent (20) | Cortical activity, best localization |
| **ROI-based** | Sources at atlas ROI centroids | Good (44) | ROI-level statistics, LMMs |
| **Cartesian** | 3D volumetric grid | Poor (93) | Dense whole-brain coverage |
| **Shell** | Concentric geometry-matched shells | **Best (23)** | Parametric mapping, depth analysis |

### Which Preset Should I Use?

- **Statistical analysis (LMMs):** `roi_based_ellipsoid` - Best ROI accuracy
- **Best spatial localization:** `ellipsoid_surface` - Lowest localization error
- **Whole-brain parametric mapping:** `shell_ellipsoid` - Best conditioning, depth-stratified
- **Fast iteration:** `sphere_surface` - Quick analytical BEM
- **Dense coverage:** `ellipsoid_cartesian` - Maximum volumetric sources

### Alternative Atlases

The package includes two atlas versions that can be specified in custom configurations:
- **47-ROI (default):** Full Antwerp Mouse Brain Atlas
- **22-ROI coarse:** Bilateral ROIs merged for connectivity analysis with fewer regions

---

## Usage

### Command Line Interface

```bash
# Run pipeline with preset
source-localization run --preset roi_based_ellipsoid --eeg data.set --output ./results

# Override parameters
source-localization run --preset roi_based_ellipsoid --eeg data.set \
    --snr 5.0 --method sLORETA --output ./results

# Include optional post-processing
source-localization run --preset roi_based_ellipsoid --eeg data.set \
    --spectral --visualize --output ./results
```

### Python API

```python
from source_localization import Pipeline

# Create and run pipeline
pipeline = Pipeline.from_preset('roi_based_ellipsoid')
results = pipeline.run(eeg_file='data.set', output_dir='./results')

# Access outputs
stc = results['inverse_solution']['stc']
roi_timeseries = results['roi_extraction']['roi_stcs_signed']

# With parameter overrides
pipeline = Pipeline.from_preset(
    'roi_based_ellipsoid',
    **{'inverse.snr': 5.0, 'inverse.method': 'sLORETA'}
)
```

### Output Files

The pipeline produces MNE/EEGLAB-compatible .set files:

```
results/
├── data/
│   ├── roi_timeseries_signed.set     # ROI time series (for connectivity)
│   ├── roi_timeseries_magnitude.set  # Absolute values (for power)
│   └── *.pkl                         # Intermediate results
├── figures/
│   └── *.png                         # Visualizations
└── pipeline_report.html              # Summary report
```

**Load in MNE:**
```python
import mne
epochs = mne.io.read_epochs_eeglab('results/data/roi_timeseries_signed.set')
# Or for continuous data:
raw = mne.io.read_raw_eeglab('results/data/roi_timeseries_signed.set')
```

---

## Publication-Quality Visualizations

### Source Map Visualization

```python
from source_localization.source_analysis import (
    SourceMapVisualizer, PRESETS, apply_style
)

# Apply publication style
apply_style('publication')

# Create visualizer
viz = SourceMapVisualizer(source_coords, brain_surface)

# Smooth interpolated heatmap (like fMRI)
fig = viz.plot_surface_heatmap_smooth(
    gamma_power,
    view='dorsal',
    cmap='hot_black',
    show_all_sources=True
)
```

### Connectivity Visualization

```python
from source_localization.source_analysis import (
    ConnectivityVisualizer, extract_mne_connectivity
)

# After computing connectivity with MNE
from mne_connectivity import spectral_connectivity_epochs
conn = spectral_connectivity_epochs(epochs, method='coh', fmin=30, fmax=80)

# Extract matrix and visualize
matrix = extract_mne_connectivity(conn, freq_band=(30, 80))
viz = ConnectivityVisualizer(roi_labels=epochs.ch_names)

# Connectivity matrix with clustering
fig = viz.plot_connectivity_matrix(matrix, cluster_order=True)

# Chord diagram
fig = viz.plot_chord_diagram(matrix, threshold_percentile=90)
```

### ROI Visualization

```python
from source_localization.source_analysis import ROIVisualizer

roi_viz = ROIVisualizer(atlas_path, roi_mapping_path)
fig = roi_viz.plot_roi_map(roi_values, show_boundaries=True, show_labels=True)
```

---

## Validation

The package includes a comprehensive dipole simulation framework for validating source localization accuracy. This allows you to test different pipeline configurations without requiring actual EEG data.

### How Validation Works

Validation uses **forward-inverse testing**: a known dipole is placed at a specific location, its scalp EEG is simulated using the forward model, then the inverse solution attempts to recover the original location. Metrics include:

- **Localization error (mm)**: Euclidean distance between true and estimated positions
- **ROI accuracy (%)**: Whether the estimated source is in the correct brain region
- **Depth-stratified analysis**: Performance by source depth from electrodes

### Setting Up a Validation Study

Create a validation directory with your pipeline configurations:

```
my_validation/
├── configs/                          # Your validation configs
│   ├── ellipsoid_shell_sLORETA.yaml
│   ├── sphere_cartesian_MNE.yaml
│   └── ...
└── results/                          # Created automatically
    └── ellipsoid_shell_sLORETA/
        ├── metrics.json
        ├── validation_report.html
        └── figures/
```

### Validation Config Format

Validation configs are standard pipeline configs with optional `validation` section:

```yaml
# Required: Pipeline configuration
pipeline:
  name: ellipsoid_shell_sLORETA
  bem_type: ellipsoid       # sphere or ellipsoid
  source_type: shell        # shell, cartesian, surface, or roi_based

# Required: Input files (relative to package data/)
inputs:
  brain_volume: data/atlas/Atlas_3DRois.nii
  brain_labels: data/atlas/Atlas_3DRoisLeftRight.Labels.nii
  roi_mapping: data/atlas/roi_mapping.json
  electrodes_csv: data/electrodes/mouse_array_coords.csv
  eeg_file: null  # Not needed for validation

# Required: Inverse method settings
inverse:
  method: sLORETA           # MNE, sLORETA, dSPM, eLORETA
  snr: 3.0
  depth_weighting: 0.8

# Required: BEM configuration (match bem_type)
bem:
  ellipsoid:
    n_layers: 3
    conductivities: [0.33, 0.0042, 0.33]  # brain, skull, scalp
    radii_ratios: [0.87, 0.92, 1.0]
    ellipsoid_method: axis_aligned
    ellipsoid_margin: 1.23
    use_cache: true

# Required: Source space configuration (match source_type)
source_space:
  shell:
    n_shells: 3
    min_points_per_shell: 20
    max_points_per_shell: 100
    distribution: fibonacci
    filter_exterior: true

# Optional: Validation-specific settings
validation:
  snr_levels: [10]          # SNR levels to test (dB)
  n_trials: 25              # Trials per test position
  test_mode: combined       # combined (recommended), roi_centroids, or uniform_grid
  grid_spacing_mm: 1.0      # For uniform_grid/combined mode
  grid_margin_mm: 0.2       # For uniform_grid/combined mode
  scale_factor: 1.0         # For brain size scaling tests

  # Dipole simulation parameters
  dipole:
    amplitude_nAm: 50.0     # Dipole amplitude
    duration_s: 1.0         # Simulation duration
    sfreq: 500.0            # Sampling frequency

# Optional: Output settings
outputs:
  dir: null                 # Auto-set by validation runner
  save_intermediate: true
  figure_format: png
  figure_dpi: 100
```

### Running Validation via CLI

```bash
# List available configs
source-localization validate --test-dir ./my_validation --config configs/ --list

# Run all configs
source-localization validate --test-dir ./my_validation --config configs/ --all

# Run specific config
source-localization validate --test-dir ./my_validation --config configs/ellipsoid_shell_sLORETA.yaml

# Quick test mode (5 ROIs, 1 trial, SNR=10)
source-localization validate --test-dir ./my_validation --config configs/ --all --quick

# Override SNR and trials
source-localization validate --test-dir ./my_validation --config configs/ --all \
    --snr 5 10 20 --trials 50

# Use combined test mode (recommended - ROI accuracy from centroids, localization from grid)
source-localization validate --test-dir ./my_validation --config configs/ --all \
    --test-mode combined

# Use uniform grid only (position-independent localization metrics)
source-localization validate --test-dir ./my_validation --config configs/ --all \
    --test-mode uniform_grid

# Use coarse 22-ROI atlas
source-localization validate --test-dir ./my_validation --config configs/ --all \
    --atlas coarse_22roi

# Summarize existing results
source-localization validate --summarize ./my_validation/results/

# Compare multiple configs
source-localization validate --compare \
    ./my_validation/results/config1/ \
    ./my_validation/results/config2/
```

### Test Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `roi_centroids` | Test at ROI centroid positions (default) | Best for ROI accuracy (tests at ROI centers) |
| `uniform_grid` | Test on uniform 3D grid across brain | Best for localization error, depth analysis |
| `combined` | **Recommended**: Run both modes, report ROI accuracy from centroids + localization error from grid | Comprehensive validation with appropriate metrics from each mode |

**Why use combined mode?**
- **ROI accuracy** is best measured at ROI centroid positions (fair test of ROI classification)
- **Localization error** is best measured at uniform grid positions (position-independent spatial accuracy)
- Combined mode runs both and reports the appropriate metric from each

```bash
# Run combined mode validation (recommended)
source-localization validate --test-dir ./my_validation --config configs/ --all \
    --test-mode combined --snr 10.0 --trials 25
```

### Understanding Results

Each validation run produces:

```
results/config_name/
├── metrics.json              # Full metrics (JSON)
├── validation_report.html    # Interactive HTML report
├── data/                     # Intermediate pipeline outputs
└── figures/
    ├── localization_error_map_snr10.png
    └── depth_error_snr10.png
```

**metrics.json structure:**
```json
{
  "config_name": "ellipsoid_shell_sLORETA",
  "n_sources": 185,
  "n_test_positions": 46,
  "snr_results": {
    "10": {
      "localization_error": {
        "mean": 2.14,
        "median": 1.34,
        "std": 2.86
      },
      "roi_accuracy": {
        "exact": 0.80,
        "n_correct": 37,
        "n_total": 46
      },
      "depth_stratified": {
        "1-2mm": {"n_trials": 10, "localization_error_mean": 0.5},
        "2-3mm": {"n_trials": 15, "localization_error_mean": 1.2},
        ...
      }
    }
  }
}
```

### Validation Results Summary

From extensive validation (257,000+ simulations across 81 configurations):

- **sLORETA** consistently outperforms other methods
- **Best config:** ellipsoid + ROI-based + sLORETA (76.9% ROI accuracy, 1.67mm error)
- **Depth-accuracy relationship:** 0-1mm: 77%, 1-2mm: 36%, 2-3mm: 4%, >3mm: ~0%

### Python API for Validation

```python
from source_localization.validation import ValidationRunner, run_validation
from pathlib import Path

# Option 1: Run multiple configs
results = run_validation(
    test_dir='/path/to/my_validation',
    config_files=[Path('configs/config1.yaml'), Path('configs/config2.yaml')],
    snr_levels=[5, 10, 20],
    n_trials=25,
    test_mode='uniform_grid',
    verbose=True
)

# Option 2: Run single config with more control
runner = ValidationRunner(
    config_path='configs/ellipsoid_shell_sLORETA.yaml',
    output_dir='results/ellipsoid_shell_sLORETA',
    verbose=True
)
runner.setup()
metrics = runner.run(snr_levels=[10], n_trials=25)
runner.save_results(metrics)
```

---

## Pipeline Architecture

```
1. Electrode Registration  → MNE Info with 32 channel positions
2. EEG Data Loading       → EEGLAB .set file → epochs
3. BEM Model              → Sphere (analytical) or Ellipsoid (numerical)
4. Source Space           → Surface / ROI-based / Cartesian / Shell
5. Forward Solution       → Leadfield matrix G
6. Inverse Solution       → MNE / dSPM / sLORETA / LCMV / DICS
7. ROI Extraction         → 47 ROI time series with depth weighting

Optional (--spectral, --visualize):
8. Spectral Analysis      → Band power per ROI
9. Visualization          → Figures and HTML report
```

---

## Troubleshooting

### Common Issues

**"Electrodes inside head model"**
- Use `bem.sphere.fit_to_electrodes: false` in config

**"mne-connectivity not found"**
- Install with: `pip install mne-connectivity`

**"Connectivity requires epoched data"**
- The analysis module automatically creates epochs from continuous data
- Adjust epoch length with `--epoch-length` CLI option

**"Module not found"**
```bash
source .venv/bin/activate
uv pip install -e .
```

---

## Citation

```bibtex
@software{mouse_eeg_source_localization,
  author = {Lexy, Alex and Pedapati, Ernest},
  title = {Mouse EEG Source Localization Package},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/drpedapati/AlexProjects}
}
```

---

## Package Structure

```
source_localization/
├── src/source_localization/
│   ├── pipeline.py              # Main orchestrator
│   ├── cli.py                   # CLI entry point
│   ├── config.py                # Configuration management
│   ├── steps/                   # Pipeline step implementations
│   ├── bem/                     # BEM models (sphere, ellipsoid)
│   ├── source_space/            # Source space types
│   ├── inverse/                 # Inverse methods
│   ├── study/                   # Multi-subject processing (NEW)
│   │   ├── config.py            # StudyConfig class
│   │   ├── batch.py             # Batch processing
│   │   └── analysis.py          # MNE wrapper analysis
│   ├── source_analysis/         # Visualization and ROI extraction
│   │   ├── roi_analysis.py      # Depth-weighted ROI extraction
│   │   ├── visualization*.py    # Publication-quality figures
│   │   └── ...
│   ├── validation/              # Dipole simulation validation
│   ├── config/presets/          # 8 validated YAML presets
│   └── data/                    # Atlas and electrode files
└── pyproject.toml
```

---

## Changelog

### Version 1.4.1 (2026-01-30)

**Source Space Standardization & Validation Documentation**

- **Standardized source placement** across all source types
  - Shell: Changed default scales from 0.3-0.9 to 0.4-0.8 (matches surface)
  - Cartesian: Added `inset_factor` parameter (default 0.80)
  - All source types now place outermost sources ~2mm from electrodes
  - Removed aggressive `filter_above_electrodes` from shell source space

- **Comprehensive validation documentation**
  - Full guide for setting up validation studies
  - Config format reference with all validation parameters
  - CLI command examples for all validation modes
  - Results format explanation

- **Bug fixes**
  - Fixed NoneType error in validation figure generation for sparse depth bins

### Version 1.4.0 (2026-01-28)

**Shell-Based Source Space & Preset Reorganization**

- **New shell source space type** for whole-brain parametric mapping
  - Concentric geometry-matched shells (Fibonacci spiral distribution)
  - Best forward matrix conditioning (22.7 vs 93 for Cartesian)
  - Explicit depth stratification for depth-resolved analysis
  - MRI space mapping utilities for parametric visualization

- **Reorganized presets** (8 total, down from 12)
  - Renamed `volumetric` → `cartesian` for clarity
  - Renamed `shell_based` → `shell`
  - Removed coarse22/cortex preset variants (atlases still included)
  - 4 source types × 2 BEM types = 8 presets

- **New validation metrics**
  - Localization error by electrode distance analysis
  - Forward matrix conditioning comparison across source types

### Version 1.3.0 (2026-01-26)

**Multi-Subject Study Processing & MNE Analysis Wrappers**

- **Study module** for batch processing multi-subject studies
  - BIDS-inspired folder hierarchy
  - Parallel job support
  - Study configuration via YAML
  - CLI commands: `study init`, `study run`, `study analyze`, `study status`

- **MNE-based analysis** for spectral and connectivity
  - Band power via MNE's Welch PSD
  - Connectivity via MNE-Connectivity (coherence, PLV, wPLI, imcoh)
  - Automatic epoching of continuous data
  - Group-level result aggregation

- **Simplified ROI extraction**
  - Depth-weighted averaging (our unique contribution)
  - MNE-compatible .set output files
  - Use MNE for all downstream analysis

### Version 1.2.0 (2026-01-26)

- Publication-quality visualizations
- Smooth interpolated heatmaps
- Connectivity visualizations (matrices, chord diagrams)
- Custom neuroimaging colormaps

### Version 1.1.0 (2026-01-26)

- Source-level analysis module
- Depth-restricted source spaces
- Atlas lookup functionality

### Version 1.0.0 (2025-12-01)

- Production release
- 5 validated pipeline presets
- Comprehensive validation framework

---

## License

MIT License

## Authors

**Alex Lexy** - Primary Developer
**Ernest Pedapati, MD** - Principal Investigator

Cincinnati Children's Hospital Medical Center
