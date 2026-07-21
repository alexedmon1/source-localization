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
    ├── 4. source_space           → Create source grid (cartesian/shell/surface/roi_based)
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
│   ├── volumetric.py     # 3D Cartesian grid  (source_type: cartesian)
│   ├── shell_based.py    # Concentric geometry-matched shells (source_type: shell)
│   ├── surface.py        # Cortical surface mesh (source_type: surface)
│   └── roi_based.py      # Sources per atlas ROI (source_type: roi_based)
└── config/presets/       # 10 YAML configs
```

### Configuration System
- Presets in `src/source_localization/config/presets/*.yaml`
- Override via CLI: `--snr 5.0 --method MNE`
- Or Python API: `Pipeline.from_preset('ellipsoid_surface', **{'inverse.snr': 5.0})`

**Source types** are `cartesian`, `shell`, `surface`, `roi_based`. Note the
module name does not match the config key: `source_type: cartesian` is
implemented by `source_space/volumetric.py`. `volumetric` is still accepted as a
legacy alias for `cartesian`.

**The 10 presets** (`{bem}_{source}`), all of which exist on disk — check the
directory rather than trusting this list, since presets are added over time:

| BEM \ source | cartesian | shell | surface | roi_based |
|---|---|---|---|---|
| ellipsoid | `ellipsoid_cartesian`, `ellipsoid_cartesian_extended` | `shell_ellipsoid`, `shell_ellipsoid_extended` | `ellipsoid_surface` | `roi_based_ellipsoid` |
| sphere | `sphere_cartesian` | `shell_sphere` | `sphere_surface` | `roi_based_sphere` |

## Critical Implementation Details

### Atlas Voxel Size Scaling (CRITICAL)
Most atlas NIfTI files have voxel sizes 10× larger than reality, and need correcting:
```python
# Header: [2.03, 0.80, 2.0] mm → Actual: [0.203, 0.080, 0.2] mm
from source_localization.utils.atlas import get_true_affine
affine = get_true_affine(nifti_img)   # scales the 3×3 block AND the translation
```

**The rule is file-specific.** Some bundled files are already stored in true
units, and `get_true_affine()` scales unconditionally — so applying it to those
shrinks the geometry a further 10×, silently, because the affine stays
well-formed and coordinates simply land outside the volume:

| File | Header zooms | Pass to `get_true_affine()`? |
|---|---|---|
| `Atlas_3DRois.nii`, `Atlas_3DRois_brain.nii.gz` | 2.03, 0.80, 2.00 | **Yes** |
| `*.pre_symm.*`, `*.preregfix.*`, `*.ORIGINAL_SWAPPED.*` | 2.03, 0.80, 2.00 | **Yes** |
| `Atlas_3DRoisLeftRight.Labels.nii` (+ `.preregfix`) | 0.203, 0.080, 0.200 | **No — use its affine as-is** |

Both conventions describe the same geometry: `get_true_affine()` on the inflated
brain volume reproduces the Labels file's native affine to within 4e-5 mm. This
is why `steps/roi_extraction.py` correctly uses the Labels affine unmodified —
that is deliberate, not a bug.

Two silent failure modes, both of which produce a valid-looking affine:
- Correcting an already-true file → coordinates land 10× too small.
- Scaling only `affine[:3, :3]` and not the translation → the origin is left 10×
  out, so the volume sits nowhere near the source coordinates.

Before adding a correction, check `nifti_img.header.get_zooms()`: a mouse-brain
voxel is a few hundred micrometres, so any value ≥ 0.5 mm means the header is
inflated.

### BEM Geometry
- Brain radius: ~6.4mm (from 95th percentile of brain voxels)
- **NEVER** set `fit_to_electrodes: true` - this breaks geometry
- Sphere BEM uses analytical solution (fast)
- Ellipsoid BEM uses numerical solution (more accurate; the previously quoted
  "1.5 mm localization error" came from a single-dipole test under favourable
  assumptions and should not be quoted as a general accuracy — see Validation)

### Inverse Methods
- `dSPM`: Noise-normalized (default, recommended)
- `MNE`: Raw minimum norm
- `sLORETA`: Standardized low resolution
- Default SNR=3.0, λ²=1/9

## Validation

The package includes dipole simulation validation. **There is no
`validate_pipeline()` function** — that API was documented here but never
existed. The real entry points are:

```bash
source-localization validate --help          # CLI
```

```python
from source_localization.validation import ValidationRunner
runner = ValidationRunner(config_path='...', output_dir='./validation')
runner.setup()
results = runner.run()
```

Other useful pieces live in `source_localization.validation`: `DipoleSimulator`
(synthesize EEG from a known dipole), `RobustnessTest` (sweeps over SNR, noise
structure, amplitude), and `BatchValidationRunner`.

**Do not quote accuracy figures from memory.** The previously documented
"~1.5 mm error, ~13% ROI accuracy" describes a single-dipole test with a
matched forward model and white noise — a best case, not expected performance.
Localization accuracy on this array depends strongly on source depth and on SNR,
and two simultaneous sources are much harder than one. Run the validation for
the configuration you actually care about rather than citing a headline number.

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
- `electrodes/mouse_array_coords.csv` - 30-channel NeuroNexus array positions (E1–E30, plus a Bregma fiducial row)

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
