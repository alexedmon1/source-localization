# Mouse EEG Source Localization Package

**Created:** 2025-11-26
**Last Updated:** 2026-09-03
**Version:** 0.5.0
**Status:** Alpha

A Python pipeline for mouse EEG source localization: a 30-channel scalp
recording in, per-ROI source time series and whole-brain source estimates out.
Ships with several registered brain parcellations (Antwerp and Allen CCFv3
derived), multi-subject batch processing, MNE-based spectral and connectivity
analysis, a dipole-simulation validation framework, and figure utilities.

The version and status above come from `pyproject.toml` and are checked by
`tests/test_readme_consistency.py`, as are the atlas and preset tables below.
If this file and the code disagree, the code is right and the test is broken.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Multi-Subject Study Processing](#multi-subject-study-processing)
- [Spectral and Connectivity Analysis](#spectral-and-connectivity-analysis)
- [Configuration Presets](#configuration-presets)
- [Atlas Selection](#atlas-selection)
- [Usage](#usage)
- [What the outputs are](#what-the-outputs-are)
- [Visualizations](#visualizations)
- [Validation](#validation)
- [Pipeline Architecture](#pipeline-architecture)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [Package Structure](#package-structure)
- [History](#history)

---

## Overview

> **📖 Documentation site:** [`docs/`](docs/index.md) — guides (adding an atlas,
> electrode setup, Monte Carlo source sampling) and **known issues** you should
> read before interpreting output. Preview locally with
> `uv run --no-project --with "mkdocs-material>=9.5,<10" mkdocs serve`.

Source localization solves the EEG inverse problem: given electrode
measurements on the scalp, estimate the locations and strengths of neural
activity inside the brain. This package does that for mouse recordings from a
fixed 30-channel array, and reports the result over an atlas parcellation so
that it can feed ROI-level statistics.

It is a research codebase under active development. The inverse problem is
ill-posed and 30 sensors cap the effective rank of any solution at 30 whatever
the source count, so read [Validation](#validation) and the
[known issues](docs/index.md#known-issues) before treating a per-ROI number as a
claim about anatomy.

---

## Features

### Core Pipeline
- **11 pipeline presets** covering 2 BEM types and 4 source-space families
- **2 BEM types**: sphere (analytical, fast) and ellipsoid (numerical, fitted to the brain mask)
- **4 source-space families**:
  - **Surface**: anatomical cortical mid-ribbon cut from the Allen32 parcellation
    (default), or a geometric icosphere
  - **ROI-based**: sources placed inside each atlas parcel
  - **Cartesian**: 3D volumetric grid
  - **Shell**: concentric geometry-matched shells
- **Inverse methods**: custom MNE, dSPM, sLORETA and eLORETA with mouse-scale
  regularization; stock MNE-Python LCMV and DICS beamformers (see
  [known issues](docs/known-issues/REGULARIZATION_SCALING_ISSUE.md) for the
  beamformer caveat)
- **Orientation constraint**: fixed to the cortical normal on the anatomical
  surface, free elsewhere
- **Registered atlases** selectable with `--atlas` (see [Atlas Selection](#atlas-selection))
- **30-channel electrode array** coordinates included

### Multi-Subject Study Processing
- BIDS-inspired folder hierarchy, YAML study configuration
- Batch processing with parallel jobs, status reporting, QC
- Group-level result collection

### MNE-Based Analysis
- Band power per ROI using MNE's Welch PSD
- Connectivity via MNE-Connectivity (coherence, PLV, wPLI, imcoh)
- Automatic epoching of continuous data for connectivity

### ROI Extraction
- The pipeline's ROI step assigns each source to a parcel (by the source
  space's own parcel labels where it has them, otherwise nearest labeled voxel)
  and takes the **unweighted mean** of the sources in each parcel.
- Two variants are written: **magnitude** (always positive) and **signed**. Under
  the fixed-orientation anatomical surface the signed value is the component
  along the cortical normal. Under free orientation it is the component with the
  largest variance within each epoch. Neither is an SVD.
- Depth-weighted extraction exists in `source_analysis.roi_analysis` but is
  **not** part of the pipeline; the earlier claim that it was has been removed.

### Visualizations
- Per-step QC figures and an HTML report from every run
- Interpolated heatmaps, ROI parcellation overlays, connectivity matrices and
  chord diagrams in `source_analysis`

---

## Installation

### Prerequisites

- Python >= 3.9 and < 3.14
- Virtual environment manager (uv recommended)

### Using uv (Recommended)

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/alexedmon1/source-localization.git
cd source-localization
uv venv && source .venv/bin/activate
uv pip install -e .

# For connectivity analysis (optional)
uv pip install -e ".[connectivity]"
```

### Reproducing a published analysis

A plain install resolves whatever versions are current, which is **not** what
produced any published result — MNE in particular resolves to 1.12.1 rather than
the 1.11.0 used throughout. Install against a lockfile instead:

```bash
uv venv .venv
uv pip install --python .venv -r <lockfile>
uv pip install --python .venv --no-deps "source-localization @ git+https://github.com/alexedmon1/source-localization.git@<tag>"
```

`--no-deps` is deliberate: the lockfile is the authority on versions, and letting
the package re-resolve its own dependencies is what drifts the environment.

Release tags used by published work:

| tag | what it is |
|---|---|
| `v0.2.1` | the reconstruction cited by the Fmr1 KO vs WT manuscript |
| `v0.2.3` | **v0.2.1 plus one line of packaging metadata** — declares `scikit-learn`, which `source_space/roi_based.py` imports but v0.2.1 never declared. `git diff v0.2.1 v0.2.3` touches `pyproject.toml` and nothing else, so every analysis path is byte-identical. Install this; cite v0.2.1 |
| `v0.4.1` | corrected brain mask — sources are placed inside the brain rather than the meningeal rim |
| `v0.4.2` | v0.4.1 plus the same packaging fix. Install this; cite v0.4.1 |

Note that versions before 0.5.0 did not reorder EEG channels to match the
electrode registration. Recordings stored in a channel order other than
E1..E30 were silently misaligned with the leadfield. Check
`epochs.ch_names` on your input files before relying on an older tag.

### Verify Installation

```bash
source-localization --help
source-localization study --help
python -c "import source_localization; print(source_localization.__version__)"
```

---

## Quick Start

### Single Subject

```bash
# Anatomical surface, ellipsoid BEM, Allen32 parcels (the preset carries its atlas)
source-localization run --preset ellipsoid_surface --eeg /path/to/data.set --output ./results

# ROI-based sources on the Allen32 atlas (what the FORGE study used)
source-localization run --preset roi_based_ellipsoid --atlas allen32 --eeg /path/to/data.set --output ./results

# View results
open results/pipeline_report.html
```

Presets other than the two anatomical-surface ones default to the Antwerp
atlas when `--atlas` is omitted. Prefer `--atlas allen32` for anything that
ends in a per-ROI claim; see [Atlas Selection](#atlas-selection).

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
            ├── subjects.csv            # written by `study collect`
            ├── group_band_power.csv    # written by `study analyze`
            └── connectivity_*.csv
```

### CLI Commands

```bash
# Initialize study from folder (default preset: roi_based_ellipsoid)
source-localization study init /path/to/data --name "MyStudy" --preset roi_based_ellipsoid

# Run source localization pipeline
source-localization study run study_config.yaml --jobs 4 --verbose

# Run spectral/connectivity analysis (uses MNE)
source-localization study analyze study_config.yaml --bands delta theta alpha beta low_gamma high_gamma --connectivity coherence

# Check processing status
source-localization study status study_config.yaml

# Collect the per-subject table into group/subjects.csv
source-localization study collect study_config.yaml

# Quality control report
source-localization study qc study_config.yaml
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
print(f"Processed {result.n_successful} subjects, {result.n_failed} failed")

# Run analysis
df = analyze_study(
    config,
    bands={'low_gamma': (30, 55), 'theta': (4, 10)},
    connectivity_methods=['coherence', 'plv'],
    n_jobs=4
)
```

---

## Spectral and Connectivity Analysis

The analysis module wraps MNE-Python and MNE-Connectivity for batch analysis of processed subjects.

### Band Power Analysis

Computes power spectral density per ROI using Welch's method:

```python
from source_localization.study import analyze_subject, DEFAULT_BANDS

# DEFAULT_BANDS: delta, theta, alpha, beta, low_gamma, high_gamma
result = analyze_subject(
    subject_dir='derivatives/source_localization/sub-001',
    bands=DEFAULT_BANDS,
    overwrite=False
)

# Output: analysis/band_power.csv
# Columns: roi, band, fmin, fmax, power, power_db
```

There is no band called `gamma`. The defaults split it into `low_gamma`
(30-55 Hz) and `high_gamma` (65-100 Hz). Unknown band names are rejected.

### Connectivity Analysis

Computes ROI-to-ROI connectivity using MNE-Connectivity:

```python
# Requires: pip install mne-connectivity
result = analyze_subject(
    subject_dir='derivatives/source_localization/sub-001',
    connectivity_methods=['coherence', 'plv', 'wpli'],
    connectivity_bands=['low_gamma', 'theta'],
    epoch_length=2.0  # For continuous data, create 2s epochs
)

# Output: analysis/connectivity_coherence_low_gamma.csv (n_roi x n_roi matrix)
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
source-localization study analyze study_config.yaml \
    --bands delta theta alpha beta low_gamma high_gamma \
    --connectivity coherence plv \
    --epoch-length 2.0 \
    --jobs 4 \
    --verbose
```

---

## Configuration Presets

### Available Presets (11 total)

Source counts were measured on 2026-09-03 with the Allen32 atlas. ROI-based
counts depend on the atlas (e.g. `roi_based_sphere` places 184 sources on
Antwerp). Presets in `src/source_localization/config/presets/` are the source
of truth for every setting; the CLI lists them under `--preset`.

| Preset | BEM | Source space | Sources | Notes |
|--------|-----|--------------|---------|-------|
| **`ellipsoid_surface`** | Ellipsoid (extended) | Anatomical surface, 0.5 mm | 1310 | Cortical mid-ribbon, fixed orientation. Carries the Allen32 atlas |
| `ellipsoid_surface_anatomical` | Ellipsoid (extended) | Anatomical surface, 0.5 mm | 1310 | Same configuration as `ellipsoid_surface`, kept under its old name |
| `sphere_surface` | Sphere | Icosphere, ico 3 | 305 | Geometric surface, free orientation. Fast |
| **`roi_based_ellipsoid`** | Ellipsoid | ROI-based | 204 | Sources inside each parcel. Used by the FORGE study with `--atlas allen32` |
| `roi_based_sphere` | Sphere | ROI-based | 196 | Fast ROI-based. **The sphere does not contain all ROI sources**: 8 of 204 dropped on allen32, 22 of 206 on antwerp (`tests/test_preset_containment.py`) |
| `shell_ellipsoid` | Ellipsoid | Shell | 187 | Concentric shells; see [shell coverage known issue](docs/known-issues/SHELL_ROI_COVERAGE.md) |
| `shell_ellipsoid_extended` | Ellipsoid (extended) | Shell | 115 | Conductor extended over the olfactory bulbs |
| `shell_sphere` | Sphere | Shell | 91 | Fast shell-based |
| `ellipsoid_cartesian` | Ellipsoid | Cartesian grid | 141 | Volumetric grid, auto spacing |
| `ellipsoid_cartesian_extended` | Ellipsoid (extended) | Cartesian grid | 150 | Conductor extended over the olfactory bulbs |
| `sphere_cartesian` | Sphere | Cartesian grid | 80 | Volumetric grid, auto spacing |

"Extended" ellipsoids shift the centre anteriorly and lengthen the Y semi-axis
so the olfactory bulbs fall inside the conductor. Without that, MNE drops
bulb sources from the forward. Since 0.5.0 the forward step reports any dropped
sources and restricts every per-source array to the survivors, so a drop can no
longer misalign downstream indexing, but it still discards anatomy.

### Source Space Types

| Type | Description | Orientation | Best for |
|------|-------------|-------------|----------|
| **Surface (anatomical)** | Cortical mid-ribbon from Allen32, sources carry their parcel | Fixed to cortical normal | Cortical activity, signed measures |
| Surface (icosphere) | Geometric mesh inset from the brain surface | Free | Baseline comparisons |
| **ROI-based** | Sources placed inside each atlas parcel | Free | ROI-level statistics, mixed models |
| Cartesian | 3D volumetric grid | Free | Whole-brain coverage |
| Shell | Concentric geometry-matched shells | Free | Depth-stratified parametric mapping |

### Which Preset Should I Use?

- **ROI statistics on an existing design:** `roi_based_ellipsoid --atlas allen32`.
- **Cortical, signed, orientation-aware analysis:** `ellipsoid_surface`.
- **Whole-brain parametric mapping:** `shell_ellipsoid`, with the coverage caveat linked above.
- **Fast iteration:** `sphere_surface`.

All inverse methods can be set per preset or with `--method`. sLORETA is the
default in every preset and the most robust to mouse-scale leadfields (see the
[regularization known issue](docs/known-issues/REGULARIZATION_SCALING_ISSUE.md)).

---

## Atlas Selection

Atlases are declared in `src/source_localization/data/atlas/registry.yaml` and
selected with `--atlas` on the CLI or `atlas=` in the Python API. All of them
share one voxel grid and one coordinate space, so BEM geometry and electrode
positions are identical across atlases; only the label volume and the ROI
mapping change.

| Atlas | Parcels | Description |
|-------|---------|-------------|
| `antwerp` | 46 | UAntwerpen C57BL/6 MRI atlas: 46 named structures plus background. Labels only ~31% of the brain mask, so nearest-label assignment can move a source up to 2.6 mm. Default for presets that do not carry their own atlas |
| `allen32` | 32 | Allen CCFv3 registered into Antwerp space, 16 parcels per hemisphere (ids 1-16 left, 17-32 right). Tiles grey matter. Each parcel carries a `tier` and `category`. **Prefer this for per-ROI claims** |
| `allen` | 32 | Alias of `allen32` (identical files) |
| `allen64` | 64 | Finer Allen parcellation, 32 per hemisphere. No `roi_categories` file |
| `allen26` | 26 | Allen32 with six bilateral pairs merged that the 30-channel array cannot separate |
| `coarse22` | 22 | The 46 Antwerp structures regrouped into 22 regions. Inherits Antwerp's sparse coverage. Accepted as `coarse_22roi` by the validation CLI |

```bash
# CLI
source-localization run --preset shell_ellipsoid --atlas allen32 --eeg data.set --output results/

# Python API
pipeline = Pipeline.from_preset('shell_ellipsoid', atlas='allen32')
```

The anatomical-surface presets (`ellipsoid_surface`,
`ellipsoid_surface_anatomical`) are cut from the Allen32 parcellation and carry
Allen32 parcel ids on every source. They run with `allen32`, `allen`, or `allen64`;
asking for `antwerp` with them raises a clear error rather than mislabelling
the sources.

Every other preset works with every registered atlas.

---

## Usage

### Command Line Interface

```bash
# Run pipeline with preset
source-localization run --preset roi_based_ellipsoid --atlas allen32 --eeg data.set --output ./results

# Override parameters
source-localization run --preset roi_based_ellipsoid --atlas allen32 --eeg data.set \
    --snr 5.0 --method sLORETA --output ./results

# Include optional post-processing
source-localization run --preset roi_based_ellipsoid --atlas allen32 --eeg data.set \
    --spectral --visualize --output ./results
```

`--method` on the CLI accepts `MNE`, `dSPM`, and `sLORETA`. `eLORETA`, `LCMV`,
and `DICS` can be set through `inverse.method` in a config file or an API
override.

### Python API

```python
from source_localization import Pipeline

# Create and run pipeline
pipeline = Pipeline.from_preset('roi_based_ellipsoid', atlas='allen32')
results = pipeline.run(eeg_file='data.set', output_dir='./results')

# Access outputs
stc = results['inverse_solution']['stc_signed']          # mne.VolSourceEstimate
roi_timeseries = results['roi_extraction']['roi_stcs_signed']  # dict: roi name -> array

# With parameter overrides (dotted keys)
pipeline = Pipeline.from_preset(
    'roi_based_ellipsoid', atlas='allen32',
    **{'inverse.snr': 5.0, 'inverse.method': 'sLORETA'}
)
```

### Input requirements

The EEG file must be an EEGLAB `.set` (epoched or continuous) whose channel
names are the electrode labels in `data/electrodes/mouse_array_coords.csv`
(`E1`..`E30`). Channel **order** in the file does not matter: the EEG step
reorders channels to match the electrode registration and refuses a file that
lacks a registered electrode. Channels not in the registration are dropped with
a message.

### Output Files

```
results/
├── data/
│   ├── roi_timeseries_signed.set     # ROI time series (for connectivity)
│   ├── roi_timeseries_magnitude.set  # Absolute values (for power)
│   ├── source_timeseries_*.set       # Per-source time series
│   └── step*_*.pkl                   # Intermediate results
├── figures/
│   └── *.png                         # Per-step QC figures
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

## What the outputs are

- **Magnitude** is the L2 norm across orientation components per source (or the
  absolute value under fixed orientation), averaged over the sources in each
  parcel. Use it for power.
- **Signed** keeps polarity. Under the anatomical surface it is the projection
  onto the cortical normal, so sign is anatomically meaningful. Under free
  orientation it is the max-variance component per epoch, whose sign is
  arbitrary between epochs and sources; use it for within-epoch connectivity,
  not for polarity claims.
- **DICS** produces a single band-power value per source, not a time series.
  The pipeline exports it as a one-sample estimate.
- **ROIs with no assigned source are omitted** from the outputs rather than
  written as zeros. The run log lists them.

---

## Visualizations

### Source Map Visualization

```python
from source_localization.source_analysis import (
    SourceMapVisualizer, PRESETS, apply_style
)

apply_style('publication')
viz = SourceMapVisualizer(source_coords, brain_surface)
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
from mne_connectivity import spectral_connectivity_epochs

conn = spectral_connectivity_epochs(epochs, method='coh', fmin=30, fmax=55)
matrix = extract_mne_connectivity(conn, freq_band=(30, 55))
viz = ConnectivityVisualizer(roi_labels=epochs.ch_names)
fig = viz.plot_connectivity_matrix(matrix, cluster_order=True)
fig = viz.plot_chord_diagram(matrix, threshold_percentile=90)
```

### ROI Visualization

```python
from source_localization.source_analysis import ROIVisualizer

roi_viz = ROIVisualizer(atlas_path, roi_mapping_path)
fig = roi_viz.plot_roi_map(roi_values, show_boundaries=True, show_labels=True)
```

`ROIVisualizer`, `AtlasLookup`, and `CorticalSourceSpace` read label volumes
through `utils.atlas.get_true_affine()`, which detects whether a file's header
is 10x inflated. Do not scale affines by hand; see CLAUDE.md for which bundled
files are in which convention.

---

## Validation

The package includes a dipole simulation framework for validating source
localization accuracy without EEG data.

### How Validation Works

Validation uses **forward-inverse testing**: a known dipole is placed at a
specific location, its scalp EEG is simulated using the forward model, then
the inverse solution attempts to recover the original location. Metrics include:

- **Localization error (mm)**: Euclidean distance between true and estimated positions
- **ROI accuracy (%)**: whether the estimated source is in the correct parcel
- **Depth-stratified analysis**: performance by source depth from electrodes

By default the same forward model is used for simulation and inversion (an
"inverse crime") with white noise, so the reported numbers are best-case. The
runner supports mismatched conductivities and colored noise for a harder test.

### Setting Up a Validation Study

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

Validation configs are standard pipeline configs with an optional `validation` section:

```yaml
pipeline:
  name: ellipsoid_shell_sLORETA
  bem_type: ellipsoid       # sphere or ellipsoid
  source_type: shell        # shell, cartesian, surface, or roi_based

inputs:
  brain_volume: data/atlas/Atlas_3DRois.nii
  brain_labels: data/atlas/allen/allen_labels.nii.gz
  roi_mapping: data/atlas/allen/roi_mapping.json
  electrodes_csv: data/electrodes/mouse_array_coords.csv
  eeg_file: null  # Not needed for validation

inverse:
  method: sLORETA           # MNE, sLORETA, dSPM, eLORETA, LCMV, DICS
  snr: 3.0
  depth_weighting: 0.8

bem:
  ellipsoid:
    n_layers: 3
    conductivities: [0.33, 0.0042, 0.33]  # brain, skull, scalp
    radii_ratios: [0.87, 0.92, 1.0]
    ellipsoid_method: axis_aligned
    ellipsoid_margin: 1.23
    use_cache: true

source_space:
  shell:
    n_shells: 3
    min_points_per_shell: 20
    max_points_per_shell: 100
    distribution: fibonacci
    filter_exterior: true

validation:
  snr_levels: [10]          # SNR levels to test (dB)
  n_trials: 25              # Trials per test position
  test_mode: combined       # combined (recommended), roi_centroids, or uniform_grid
  grid_spacing_mm: 1.0
  grid_margin_mm: 0.2
  scale_factor: 1.0
  dipole:
    amplitude_nAm: 50.0
    duration_s: 1.0
    sfreq: 500.0

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

# Choose the test mode
source-localization validate --test-dir ./my_validation --config configs/ --all \
    --test-mode combined

# Choose the atlas (registry names; `full` and `coarse_22roi` are accepted legacy aliases)
source-localization validate --test-dir ./my_validation --config configs/ --all \
    --atlas allen32

# Summarize existing results
source-localization validate --summarize ./my_validation/results/

# Compare multiple configs
source-localization validate --compare \
    ./my_validation/results/config1/ \
    ./my_validation/results/config2/
```

The validation CLI defaults to the Antwerp atlas. Selecting anything else
suffixes the output directory with the atlas name.

### Test Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `roi_centroids` | Test at parcel centroid positions | ROI accuracy |
| `uniform_grid` | Test on a uniform 3D grid across the brain | Localization error, depth analysis |
| `combined` | Both: ROI accuracy from centroids, localization error from the grid | Recommended |

### Understanding Results

```
results/config_name/
├── metrics.json              # Full metrics (JSON)
├── validation_report.html    # Interactive HTML report
├── data/                     # Intermediate pipeline outputs
└── figures/
    ├── localization_error_map_snr10.png
    └── depth_error_snr10.png
```

`metrics.json` holds, per SNR level, localization error statistics, ROI
accuracy (exact hits over test positions), and depth-stratified error.

### What the validation numbers mean

Earlier versions of this file quoted a single headline accuracy for the
recommended preset. Those figures came from one configuration on the Antwerp
atlas under the default inverse crime, and the shallowest depth bin was being
reported as if it were the whole brain. Accuracy depends strongly on depth,
atlas, and source space, so run the validation for the configuration you
intend to use and report the depth-stratified result. The
[Monte Carlo sampling guide](docs/guides/monte_carlo_sampling.md) describes
how to integrate over source-grid placement.

### Python API for Validation

```python
from source_localization.validation import ValidationRunner, run_validation
from pathlib import Path

# Run multiple configs
results = run_validation(
    test_dir='/path/to/my_validation',
    config_files=[Path('configs/config1.yaml'), Path('configs/config2.yaml')],
    snr_levels=[5, 10, 20],
    n_trials=25,
    test_mode='uniform_grid',
    verbose=True
)

# Run a single config with more control
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
1. Electrode Registration  → MNE Info with 30 channel positions
2. EEG Data Loading       → EEGLAB .set → epochs, channels reordered to the registration
3. BEM Model              → Sphere (analytical) or Ellipsoid (numerical)
4. Source Space           → Surface / ROI-based / Cartesian / Shell
5. Forward Solution       → Leadfield G; per-source arrays restricted to kept sources
6. Inverse Solution       → MNE / dSPM / sLORETA / eLORETA / LCMV / DICS
7. ROI Extraction         → One time series per parcel with at least one source

Optional (--spectral, --visualize):
8. Spectral Analysis      → Band power per ROI
9. Visualization          → Figures and HTML report
```

---

## Troubleshooting

**"recording lacks N registered electrode(s)"**
- The `.set` file is missing channels that the electrode CSV registers. The
  forward is built for all 30, so the data cannot be aligned. Check the file's
  channel names.

**"The anatomical surface is built from a parcellation whose roi_mapping.json defines categories ..."**
- You asked for `ellipsoid_surface` with a non-Allen atlas. Use `--atlas allen32`
  or a non-surface preset.

**"BEM CONTAINMENT: N sources fall outside the inner skull"**
- The conductor is smaller than the source space. Downstream indexing stays
  correct, but those sources are gone. Use an "extended" preset or a larger
  ellipsoid margin.

**"Electrodes inside head model"**
- Use `bem.sphere.fit_to_electrodes: false` in config

**"mne-connectivity not found"**
- Install with: `pip install mne-connectivity`

**"Module not found"**
```bash
source .venv/bin/activate
uv pip install -e .
```

---

## Citation

```bibtex
@software{mouse_eeg_source_localization,
  author = {Edmondson, Alex and Pedapati, Ernest},
  title = {Mouse EEG Source Localization Package},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/alexedmon1/source-localization}
}
```

---

## Package Structure

```
source_localization/
├── src/source_localization/
│   ├── pipeline.py              # Main orchestrator
│   ├── cli.py                   # CLI entry point
│   ├── config.py                # Configuration and atlas registry
│   ├── steps/                   # Pipeline step implementations
│   ├── bem/                     # BEM models (sphere, ellipsoid)
│   ├── source_space/            # Source space types
│   ├── study/                   # Multi-subject processing
│   │   ├── config.py            # StudyConfig class
│   │   ├── batch.py             # Batch processing
│   │   └── analysis.py          # MNE wrapper analysis
│   ├── source_analysis/         # Figures, atlas lookup, depth-weighted ROI utilities
│   ├── validation/              # Dipole simulation validation
│   ├── utils/atlas.py           # Affine-convention detection; always go through this
│   ├── config/presets/          # 11 YAML presets
│   └── data/                    # Atlas and electrode files
└── pyproject.toml
```

---

## History

The package version is `0.5.0`. Earlier revisions of this README carried a
separate 1.x numbering that never corresponded to a package release; it has
been dropped. Release tags are listed under
[Reproducing a published analysis](#reproducing-a-published-analysis), and
`git log` is the changelog.

Notable changes in 0.5.0:

- Anatomical cortical mid-ribbon surface source space with fixed orientation
- Atlas registry (`registry.yaml`) driving both CLIs; `allen26` added
- EEG channels reordered to the electrode registration; the inverse refuses a
  mismatched order
- Forward step restricts per-source arrays to the sources MNE kept
- Anatomical-surface presets carry the Allen32 atlas
- Monte Carlo ROI operator

---

## License

MIT License

## Authors

**Alex Edmondson** - Primary Developer
**Ernest Pedapati, MD** - Principal Investigator

Cincinnati Children's Hospital Medical Center
