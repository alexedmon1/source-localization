# TODO: shell-pipeline ROI coverage at 215 vertices

The shell pipeline (preset `shell_ellipsoid`, output `localization/rest_shell/derivatives/`) samples 215 vertices on the brain surface mesh at ~1.7 mm spacing. After the post-fix atlas (commits `9fecf77` + `67680f9`) made the source-space symmetric in L/R, thin lateral cortical ROIs (Auditory ~1.5 mm thick at ±4.5 mm from midline) still get dropped from `step6_roi_timeseries.pkl` because the sampling grid lands in adjacent ROIs (e.g., Somatosensory_R) instead of inside the thin Auditory_R voxel set.

Symptom on FORGE sub-801 post-fix smoke test (2026-04-28):
- ROI-based pipeline: 32/32 ROIs present, every L/R pair mirrored within 0.1 mm.
- Shell pipeline: 31/32 ROIs present; `Auditory_R` excluded ("with no sources"). 1 vertex landed in Auditory_R's bbox at X=+3.55, but its specific (X,Y,Z) voxel was labeled Somatosensory_R, not Auditory_R.

This is a sampling-resolution issue, not a registration bug, and it does not currently affect any FORGE manuscript analysis (vertex-level analyses operate on raw vertex maps without ROI assignment; ROI analyses use `rest_roi/`). But the shell ROI extraction step is misleading — it claims "ROI not found" when it should say "below sampling resolution."

## Options

1. **Bump default vertex count** to ~500 for the shell preset. ~1.0 mm spacing should cover Auditory_R reliably.
2. **ROI-aware shell sampling** — guarantee N vertices per ROI on the mesh, similar to ROI-based pipeline. Requires post-hoc redistribution of mesh vertices.
3. **Per-subject coverage report** — emit per-ROI vertex count for the shell pipeline so downstream code can detect missing ROIs without parsing logs.
4. **Documentation only** — flag in README and CLI help that shell ROI extraction at default resolution does not guarantee per-ROI coverage; recommend ROI-based pipeline for any per-ROI analysis.

## Verification after a fix

Run sub-801 through the shell pipeline; confirm `step6_roi_timeseries.pkl` has all 32 expected ROIs, including Auditory_R. Run a few more subjects (e.g. sub-832, sub-914 — small-eyed mice known to have anatomical variation) and confirm 32/32 holds.

Filed: 2026-04-28 by FORGE/Edmondson lab during regfix re-run.
