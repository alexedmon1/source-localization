# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Overview

This is the **Mouse EEG Source Localization** package - a production-ready Python pipeline for solving the EEG inverse problem on mouse brain data. It estimates neural source activity from 30-channel scalp EEG recordings using the Antwerp Mouse Brain Atlas (47 ROIs).

## Development Commands

### Setup
```bash
cd source_localization
uv venv && source .venv/bin/activate
uv pip install -e .
```

### Run Pipeline
```bash
# Using preset (recommended)
python -m source_localization.cli --preset ellipsoid_surface --eeg /path/to/data.set --output ./results

# All presets
python -m source_localization.cli --preset ellipsoid_surface --eeg data.set --output results/ellipsoid_surface
```

### Run Tests
```bash
# PSD sanity test (requires EEG file and prior pipeline run)
pytest tests/test_psd_sanity.py --eeg-file /path/to/file.set -v

# Direct execution
python tests/test_psd_sanity.py /path/to/file.set
```

### Verify Installation
```bash
source-localization --help
python -c "import source_localization; print(source_localization.__version__)"
```

## Architecture

### Pipeline Flow (7 Core Steps)
```
Pipeline.run()
    ├── 1. electrode_registration  → MNE Info with channel positions
    ├── 2. eeg_data               → Load EEGLAB .set, create epochs
    ├── 3. bem_model              → Build head model (sphere or ellipsoid)
    ├── 4. source_space           → Create source grid (volumetric/surface/ROI-based)
    ├── 5. forward_solution       → Compute leadfield matrix G
    ├── 6. inverse_solution       → Apply MNE/dSPM/sLORETA
    └── 7. roi_extraction         → Map sources to ROIs, export .set files

Output: roi_timeseries_magnitude.set, roi_timeseries_signed.set (MNE/EEGLAB compatible)

Optional post-processing (--spectral, --visualize flags):
    ├── spectral_analysis         → Compute band power (theta/alpha/beta/gamma)
    └── visualization             → Generate plots and HTML report
```

### Key Module Organization
```
src/source_localization/
├── pipeline.py           # Main orchestrator (Pipeline class)
├── cli.py                # CLI entry point
├── config.py             # Configuration management
├── steps/                # Pipeline step implementations
│   ├── bem_model.py
│   ├── source_space.py
│   ├── forward_solution.py
│   ├── inverse_solution.py
│   └── ...
├── bem/                  # BEM model types
│   ├── sphere.py         # Analytical 3-layer sphere
│   └── ellipsoid.py      # Numerical ellipsoid BEM
├── source_space/         # Source space types
│   ├── volumetric.py     # 3D grid sources
│   ├── surface.py        # Cortical surface mesh
│   └── roi_based.py      # Sources per atlas ROI
└── config/presets/       # 8 validated YAML configs
```

### Configuration System
- Presets in `src/source_localization/config/presets/*.yaml`
- Available presets: `ellipsoid_surface` (best), `sphere_surface`, `roi_based_sphere`, `ellipsoid_volumetric`, `sphere_volumetric`
- Override via CLI: `--snr 5.0 --method MNE`
- Or Python API: `Pipeline.from_preset('ellipsoid_surface', **{'inverse.snr': 5.0})`

## Critical Implementation Details

### Atlas Voxel Size Scaling (CRITICAL)
Most atlas NIfTI files have voxel sizes 10× larger than reality, and need correcting:
```python
# Header: [2.03, 0.80, 2.0] mm → Actual: [0.203, 0.080, 0.2] mm
from source_localization.utils.atlas import get_true_affine
affine = get_true_affine(nifti_img)   # ALWAYS use this, never scale by hand
```

**The rule is file-specific, not universal.** Some bundled files are already
stored in true units, and correcting those scales them down a further 10× —
silently, because the affine stays well-formed and coordinates simply land
outside the volume:

| File | Header zooms | Needs 10× correction? |
|---|---|---|
| `Atlas_3DRois.nii`, `Atlas_3DRois_brain.nii.gz` | 2.03, 0.80, 2.00 | **Yes** |
| `*.pre_symm.*`, `*.preregfix.*`, `*.ORIGINAL_SWAPPED.*` | 2.03, 0.80, 2.00 | **Yes** |
| `Atlas_3DRoisLeftRight.Labels.nii` (+ `.preregfix`) | 0.203, 0.080, 0.200 | **No — already true** |

Both conventions describe the same geometry: `get_true_affine()` on the inflated
brain volume reproduces the Labels file's native affine to within 4e-5 mm.

`get_true_affine()` / `get_true_voxel_sizes()` **detect** which convention a file
uses (`header_is_inflated()`), so they are safe to call on any atlas file. This
is why `steps/roi_extraction.py` correctly uses the Labels file's affine as-is.

Two failure modes, both silent — always go through `utils/atlas.py`:
- Correcting an already-true file → coordinates 10× too small.
- Scaling only `affine[:3, :3]` and not the translation → origin 10× off, so the
  volume lands nowhere near the source coordinates.

### BEM Geometry
- Brain radius: ~6.4mm (from 95th percentile of brain voxels)
- **NEVER** set `fit_to_electrodes: true` - this breaks geometry
- Sphere BEM uses analytical solution (fast)
- Ellipsoid BEM uses numerical solution (more accurate, 1.5mm localization error)

### Inverse Methods
- `dSPM`: Noise-normalized (default, recommended)
- `MNE`: Raw minimum norm
- `sLORETA`: Standardized low resolution
- Default SNR=3.0, λ²=1/9

## Validation

The package includes dipole simulation validation:
```python
from source_localization.validation import validate_pipeline
results = validate_pipeline(pipeline_dir='./results/ellipsoid_surface', n_rois=46)
# Returns: mean_error_mm (~1.5mm), roi_accuracy (~13%)
```

## Python API Usage

```python
from source_localization import Pipeline

# From preset
pipeline = Pipeline.from_preset('ellipsoid_surface')
results = pipeline.run(eeg_file='data.set', output_dir='./results')

# Access outputs
stc = results['inverse_solution']['stc']  # Source time courses
roi_stcs = results['roi_extraction']['roi_stcs_signed']  # Signed ROI time series

# Optional: run spectral analysis
pipeline.run_spectral_analysis()
roi_power = pipeline.step_outputs['spectral_analysis']['roi_band_power']['gamma']
```

## Data Files

Bundled with package in `src/source_localization/data/`:
- `atlas/Atlas_3DRois.nii` - Brain volume with 47 ROI labels
- `atlas/roi_mapping.json` - ROI names and metadata
- `electrodes/mouse_array_coords.csv` - 30-channel NeuroNexus array positions (E1–E30, plus a Bregma fiducial)

## Output Structure

Each pipeline run creates:
```
output_dir/
├── data/
│   ├── roi_timeseries_magnitude.set  # MNE/EEGLAB compatible (always positive)
│   ├── roi_timeseries_signed.set     # MNE/EEGLAB compatible (signed, for connectivity)
│   └── step*_*.pkl                   # Intermediate pickle files
├── figures/              # PNG visualizations per step
├── bem_cache/            # Cached BEM models for reuse
└── pipeline_report.html  # Interactive summary report
```

**Primary outputs** are the `.set` files - load in MNE or EEGLAB for downstream analysis.

## Troubleshooting

- **"Electrodes inside head model"**: Set `bem.sphere.fit_to_electrodes: false`
- **"Brain radius too small"**: Verify using skull-stripped `Atlas_3DRois_brain.nii.gz`
- **"Forward matrix singular"**: Check electrodes are outside scalp, try different source spacing
- **ImportError**: Activate venv and reinstall with `uv pip install -e .`
