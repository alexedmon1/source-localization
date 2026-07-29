# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Overview

This is the **Mouse EEG Source Localization** package - a production-ready Python pipeline for solving the EEG inverse problem on mouse brain data. It estimates neural source activity from 30-channel scalp EEG recordings, and reports it over an atlas parcellation.

**The atlas is a choice, not a fixed property of the package.** Four parcellations
are registered (`antwerp`, `allen`/`allen32`, `allen64`) and the source geometry
is identical across all of them — only the ROI label lookup changes. Antwerp is
the historical default; **`allen32` is the one to prefer for ROI-level work**.
See "Atlas Selection" below before assuming which atlas is in play.

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

### Atlas Selection

Atlases are declared in `src/source_localization/data/atlas/registry.yaml`, loaded
into `config.ATLAS_DEFINITIONS`, and applied by `Config.apply_atlas(name)`, which
overwrites the `inputs:` paths. **Presets hardcode the Antwerp paths**, so an
atlas is only used if it is asked for explicitly.

```bash
python -m source_localization.cli --preset shell_ellipsoid --atlas allen ...
```
```python
Pipeline.from_preset('shell_ellipsoid', atlas='allen32')   # API reaches all keys
```

| Registry key | Labels file | Parcels | Notes |
|---|---|---|---|
| `antwerp` | `Atlas_3DRoisLeftRight.Labels.nii` | 46 + background | Default. Named structures only |
| `allen` | `allen/allen_labels.nii.gz` | 32 + background | **Identical to `allen32`** — same files |
| `allen32` | `allen/allen_labels.nii.gz` | 32 + background | 16 per hemisphere, `1-16 = L`, `17-32 = R` |
| `allen64` | `allen/allen_labels_allen64.nii.gz` | 64 + background | 32 per hemisphere |

Each registry entry is `inputs:` (the file paths, copied into the pipeline
config) plus `meta:` (parcel count, coverage, tier scheme — descriptive only,
never copied). Read metadata with `Config.atlas_meta(name)`. `--atlas` choices
and help text are both generated from the registry, so **adding an atlas to
`registry.yaml` is the only step needed** to make it selectable.
`tests/test_atlas_registry.py` asserts the declared parcel counts and coverage
percentages against the shipped NIfTI files, so the registry cannot drift from
what it describes.

**Both CLIs share the registry.** `cli.py` and `validation/cli.py` used to have
disjoint atlas vocabularies overlapping on one name, with the validation side
hardcoding paths in `if/elif` chains. Names now resolve through
`config.resolve_atlas_name()`, and the validation CLI additionally accepts the
two legacy names its scripts pass:

| Legacy name | Canonical |
|---|---|
| `full` | `antwerp` |
| `coarse_22roi` | `coarse22` |

The validation CLI still **defaults to `full`** (i.e. Antwerp), and selecting
anything else suffixes the output directory with the atlas name.

### Label volumes have different affine conventions — always detect

`coarse_22roi_atlas.nii` is the one label volume with a **10×-inflated header**;
Antwerp's and both Allen files are already in true units. `roi_extraction.py`
used the raw affine, which is right for the latter and puts coarse22's labels at
**±49 mm while sources sit at ±5 mm**. Nothing raised — the nearest-labeled-voxel
fallback always finds *a* voxel — so all 215 shell sources silently collapsed
onto **2 of 22** ROIs. Fixed by routing through `get_true_affine()`, which
detects the convention and is a verified no-op for the true-unit files.

After the fix, ROIs receiving at least one of the 215 shell sources:
`antwerp` 37/46, `allen32` 31/32, `allen64` 59/64, `coarse22` 22/22.

**Any code reading a label volume must go through `utils/atlas.py`.** Do not use
`nii.affine` directly, and do not assume the file you were handed matches the one
you tested against.

**Coverage matters for anything that maps coordinates to ROIs.** Measured against
the 0.5 mm validation source grid (4251 positions inside the brain mask):

| | Antwerp | Allen32 |
|---|---|---|
| positions strictly inside a label | 31.4% | **69.3%** |
| within 0.5 mm of a label | 73.2% | **98.9%** |
| within 1.0 mm of a label | 90.0% | **100%** |
| worst-case nearest-label distance | 2.56 mm | **0.98 mm** |
| smallest parcel (grid positions) | **1** | **18** |

Antwerp leaves large unlabeled interior regions, so nearest-labeled-voxel
assignment (`steps/roi_extraction.py`) can move a source up to 2.6 mm to reach a
label, and ten of its parcels have ≤4 candidate positions. Allen32 is
grey-matter-tiling, so the same assignment is a sub-voxel nudge. Prefer Allen32
whenever the result is a per-ROI claim.

Allen32 also carries metadata the Antwerp mapping does not: a `tier` per parcel
(1 = dorsal cortex near the electrodes, 2 = central subcortical, 3 =
lateral/ventral far from them) and `category` (cortical, hippocampal, thalamic,
subcortical, hypothalamic, olfactory, cerebellum, brainstem). The tiers encode
how well EEG can reach each parcel and are the natural stratifier for
depth-dependent results.

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
| `Atlas_3DRois.nii` (+ `.pre_symm`, `.preregfix`) | 2.03, 0.80, 2.00 | **Yes** |
| `Atlas_3DRois_brain.nii.gz` (+ `.pre_symm`, `.preregfix`) | 2.03, 0.80, 2.00 | **Yes** |
| `Atlas_3DRoisLeftRight.Labels.ORIGINAL_SWAPPED.*` | 2.03, 0.80, 2.00 | **Yes** |
| `allen/allen_labels_v1_49roi.*` (superseded) | 2.03, 0.80, 2.00 | **Yes** |
| `coarse_parcellation/coarse_22roi_atlas.*` | 2.03, 0.80, 2.00 | **Yes** |
| `Atlas_3DRoisLeftRight.Labels.nii` (+ `.preregfix`) | 0.203, 0.080, 0.200 | **No — already true** |
| `allen/allen_labels.nii.gz` (+ `_allen64`, `.preregfix`) | 0.203, 0.080, 0.200 | **No — already true** |
| `allen/allen_annotation_in_antwerp.*` | 0.203, 0.080, 0.200 | **No — already true** |

**The filename suffix does not tell you the convention.** `.preregfix` appears on
both sides of this table — `Atlas_3DRois.preregfix.nii` is inflated while
`Atlas_3DRoisLeftRight.Labels.preregfix.nii` is not. Only the header zooms
decide. All current Allen label files are already in true units.

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

Bundled with package in `src/source_localization/data/`.

**Do not confuse the template with the labels.** `Atlas_3DRois.nii` is the MRI
*intensity* template — it holds ~808,000 distinct grey values, not ROI IDs.
Every parcellation is a separate integer label volume. All files below share one
voxel grid (64 × 256 × 50) and one coordinate space, so they are interchangeable
as label sources.

| File | Role | Contents |
|---|---|---|
| `atlas/Atlas_3DRois.nii` | `brain_volume` | **Intensity template.** Continuous MRI values. **Not labels** |
| `atlas/Atlas_3DRois_brain.nii.gz` | `brain_mask` | Skull-stripped binary mask, 563.7 mm³. Defines the source volume |
| `atlas/Atlas_3DRoisLeftRight.Labels.nii` | `brain_labels` (antwerp) | Integer IDs 0–46; 46 named structures + background |
| `atlas/roi_mapping.json` | `roi_mapping` (antwerp) | Names, abbreviations, colors, categories for the above |
| `atlas/allen/allen_labels.nii.gz` | `brain_labels` (allen/allen32) | Integer IDs 0–32; 16 parcels per hemisphere |
| `atlas/allen/roi_mapping.json` | `roi_mapping` (allen/allen32) | Names, `tier`, `category`, `hemisphere`, source Allen structure IDs |
| `atlas/allen/allen_labels_allen64.nii.gz` | `brain_labels` (allen64) | Integer IDs 0–64; 32 parcels per hemisphere |
| `atlas/registry.yaml` | — | Maps atlas name → the file set above |
| `electrodes/mouse_array_coords.csv` | — | 30-channel NeuroNexus array positions (E1–E30, plus a Bregma fiducial) |

Also present but **not registered, do not use as label sources**:
`allen/allen_labels_v1_49roi.*` (superseded by allen32/allen64),
`allen/allen_annotation_in_antwerp.*` (raw warped Allen CCFv3 annotation, 553
distinct structures, before parcel grouping — note `allen/METHODS.md` says "661
Allen structures preserved", which does not match the shipped file),
`coarse_parcellation/coarse_22roi_atlas.*`
(a separate 22-ROI Antwerp regrouping), and every `.pre_symm` / `.preregfix` /
`.ORIGINAL_SWAPPED` variant (earlier registration states, kept for provenance).

`atlas/allen/METHODS.md` documents the ANTs registration of Allen CCFv3 into
Antwerp space and how the parcels were built.

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
