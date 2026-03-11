# Allen Full-Brain Atlas: Methods

## Overview

This atlas provides an **anatomically constrained, depth-adaptive** whole-brain parcellation derived from the Allen Mouse Brain Common Coordinate Framework v3 (CCFv3), registered into the Antwerp (UAnterwerpen C57BL/6 MRI) coordinate space. It is designed for mouse EEG source localization with a 32-channel electrode array, where spatial resolution degrades with depth from the cortical surface.

**61 parcels** cover the grey matter brain volume, with finer parcellation near the electrodes (cortical surface) and coarser parcellation at depth, reflecting the physical limits of EEG spatial resolution. Parcellation respects major anatomical division boundaries (e.g., thalamus, hippocampus, cortex) — structures are never merged across divisions.

## Source Data

- **Allen CCFv3 template**: `average_template_25.nrrd` — 25 µm isotropic Nissl-derived intensity average (528 × 320 × 456 voxels, PIR orientation)
- **Allen CCFv3 annotation**: `annotation_25.nrrd` — integer label volume with 672 unique structure IDs
- **Allen structure ontology**: `structure_graph.json` — hierarchical tree of 1327 brain structures with ancestry, naming, and color metadata
- **Antwerp atlas**: `Atlas_3DRois_brain.nii.gz` — skull-stripped MRI template (64 × 256 × 50 voxels, RAS orientation, header voxel sizes 10× inflated)

All Allen data downloaded from the Allen Institute API (`http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/`).

## Registration: Allen CCFv3 to Antwerp Space

### Coordinate Conversion (NRRD to NIfTI)

The Allen CCFv3 NRRD files use a PIR (Posterior-Inferior-Right) axis convention with voxel spacing in **microns** (25 µm), stored under the `left-posterior-superior` space tag. We constructed a RAS-compatible NIfTI affine directly from the known PIR convention:

- Axis 0 (528 voxels, 13.2 mm): Anterior → Posterior → mapped to A decreasing
- Axis 1 (320 voxels, 8.0 mm): Dorsal → Ventral → mapped to S decreasing
- Axis 2 (456 voxels, 11.4 mm): Left → Right → mapped to R increasing

Micron values were divided by 1000 to convert to mm. The resulting NIfTI was reoriented to RAS using `nibabel.orientations`.

### Antwerp Header Correction

The Antwerp atlas NIfTI header contains voxel sizes that are 10× larger than the true physical dimensions. The affine was corrected by dividing spatial components by 10:

- Header: 2.03 × 0.80 × 2.0 mm → True: 0.203 × 0.080 × 0.200 mm
- Physical extent: 13.0 × 20.5 × 10.0 mm

### ANTs Registration

Cross-modality registration (Nissl histology → MRI) was performed using ANTs (`antsRegistration`) with three stages:

1. **Rigid** (6 DOF): MI metric, 64 bins, 0.5 sampling, 2000×1000×500×250 iterations, shrink 12×8×4×2
2. **Affine** (12 DOF): MI metric, 64 bins, step size 0.05, same convergence schedule
3. **SyN** (deformable): CC metric, radius 4, 200×150×100×50 iterations, shrink 8×4×2×1

Key parameters:
- **Fixed image**: Antwerp brain (corrected header)
- **Moving image**: Allen template (RAS, mm)
- **Masking**: Fixed mask only (Antwerp brain mask); no moving mask
- **Histogram matching**: Enabled (cross-modality)
- **Center-of-mass initialization**: Enabled

Registration quality: spatial correlation r = 0.572, 80.8% coverage of Antwerp brain voxels, 661 Allen structures preserved.

### Transform Application

The Allen annotation volume was warped to Antwerp space using `antsApplyTransforms` with `GenericLabel` interpolation (nearest-neighbor with label voting).

### Pre-registration Smoothing

To reduce aliasing artifacts from the large resolution mismatch (25 µm → ~200 µm), a Gaussian probabilistic label smoothing was applied to the Allen annotation **before** warping:

1. For each of the 671 non-background labels, a binary mask was created
2. Each mask was smoothed with a 3D Gaussian filter (σ = 3 voxels = 75 µm)
3. At each voxel, the label with the highest smoothed probability was assigned

This produces smoother boundaries than direct nearest-neighbor resampling, at the cost of losing the smallest structures (553 labels retained from 672).

## Anatomically Constrained, Depth-Adaptive Parcellation

### Rationale

With 32 electrodes on a ~6.4 mm radius mouse head, EEG spatial resolution is approximately 2 mm near the cortical surface but degrades with depth. A uniform parcellation would either over-resolve deep structures or under-resolve cortical ones.

The previous v1 parcellation (49 ROIs) used purely spatial clustering without anatomical constraints, which caused structures like the left thalamus to be absorbed into neighboring parcels (striatum, hippocampus). The v2 parcellation enforces anatomical division boundaries.

### Anatomical Divisions

Each Allen structure is assigned to one of 11 major anatomical divisions based on the Allen ontology hierarchy. Clustering is performed **within** divisions — structures never merge across division boundaries.

| Division | Allen Root ID(s) | Structures | Parcels |
|----------|-----------------|------------|---------|
| Isocortex | 315 | 238 | 21 |
| Hippocampal formation | 1089 | 26 | 6 |
| Olfactory areas | 698 | 18 | 6 |
| Cerebellum | 512 | 21 | 5 |
| Midbrain | 313 | 58 | 5 |
| Cortical subplate (amygdala) | 567 | 11 | 4 |
| Medulla | 354 | 47 | 4 |
| Pons | 771 | 28 | 4 |
| Thalamus | 549 | 51 | 3 |
| Hypothalamus | 1097 | 50 | 3 |
| Cerebral nuclei (striatum) | 623 | 27 | 4 (L=2, R=2) |

**Excluded divisions** (no neural signal expected):
- Fiber tracts (967, 1009): ~77 structures → background (label 0)
- Ventricular systems (73): ~7 structures → background (label 0)

### Depth Zones

Depth from the dorsal brain surface was computed for each voxel by finding the highest brain voxel in each (R, A) column and measuring the distance along the S axis:

| Zone | Depth | Cluster threshold | Parcels |
|------|-------|-------------------|---------|
| Superficial | 0–2 mm | 2 mm | 27 |
| Mid-depth | 2–4 mm | 3 mm | 20 |
| Deep | 4+ mm | 4 mm | 14 |

Each Allen structure is assigned to a depth zone based on the **median depth** of its voxels.

### Hemisphere Split

The brain midline was determined at voxel X = 32 (0.203 mm/voxel). Each structure is assigned to the hemisphere containing the majority of its voxels.

### Spatial Clustering

Within each (division × hemisphere × depth zone) group, Allen structures were clustered by centroid distance (in mm) using **complete-linkage hierarchical clustering**, with the distance threshold set to the zone's resolution.

### Size-Based Merging

Clusters failing the minimum-extent criterion — `min(R_extent, A_extent) < zone_resolution` — were iteratively merged into their nearest valid neighbor (by centroid distance, within the same group). If all clusters in a group are undersized, the largest is kept as-is.

### Result

| Zone | Left | Right | Total |
|------|------|-------|-------|
| Superficial (0–2 mm) | 15 | 12 | 27 |
| Mid-depth (2–4 mm) | 10 | 10 | 20 |
| Deep (4+ mm) | 8 | 6 | 14 |
| **Total** | **33** | **28** | **61** |

### ROI Categories

| Category | Parcels | Description |
|----------|---------|-------------|
| cortical | 21 | Isocortex (primary/secondary motor, somatosensory, visual, auditory, retrosplenial, insular) |
| brainstem | 13 | Midbrain + Pons + Medulla |
| hippocampal | 6 | CA1, CA3, dentate, entorhinal, subiculum |
| olfactory | 6 | Main/accessory olfactory bulb, piriform, olfactory areas |
| cerebellum | 5 | Lobules, vermis, paraflocculus |
| subcortical | 4 | Striatum (caudoputamen, nucleus accumbens), globus pallidus, amygdala |
| thalamic | 3 | Reticular, ventral medial, ventral posteromedial thalamus |
| hypothalamic | 3 | Zona incerta, hypothalamic nuclei |

## Files

| File | Description |
|------|-------------|
| `allen_labels.nii.gz` | Parcellation label volume (61 labels, 1-indexed, in Antwerp coordinate space) |
| `roi_mapping.json` | ROI metadata: names, abbreviations, colors, division, hemisphere, depth zone, Allen structure composition |
| `allen_annotation_in_antwerp.nii.gz` | Full Allen CCFv3 annotation (553 structures) warped to Antwerp space, for reference |
| `METHODS.md` | This file |

## Known Limitations

- **Pixelated boundaries**: Some label boundaries show single-voxel holes and staircase artifacts due to the ~8× resolution mismatch between Allen (25 µm) and Antwerp (200 µm) grids.
- **Registration quality**: The cross-modality registration (r = 0.572) is reasonable but not perfect. Boundary regions, particularly at the edges of the brain and in the olfactory bulb, show the largest misalignment.
- **Bregma-lambda compression**: The transformed bregma-lambda distance is 3.35 mm vs the expected ~4.2 mm, suggesting ~20% AP compression in the warp.
- **Asymmetric L/R cluster counts**: The Allen atlas is not perfectly symmetric after warping, resulting in slight L/R differences (33 left vs 28 right parcels).
- **Excluded white matter**: Fiber tract and ventricular voxels are unlabeled (background), creating gaps in the label volume. This is intentional — EEG does not detect signals from these tissues.

## Version History

- **v2 (2026-03-11)**: Anatomically constrained parcellation (61 ROIs). Added division boundaries (thalamus, hippocampus, cortex, etc. never merge). Excluded fiber tracts and ventricles. 11 anatomical divisions × hemisphere × depth zone clustering.
- **v1 (2026-03-06)**: Initial depth-adaptive parcellation (49 ROIs). Purely spatial clustering without anatomical constraints.

## Citations

- Wang Q, Ding SL, Li Y, et al. (2020) The Allen Mouse Brain Common Coordinate Framework: A 3D Reference Atlas. Cell 181(4):936-953.e20. https://doi.org/10.1016/j.cell.2020.04.007
- Avants BB, Tustison NJ, Song G, et al. (2011) A reproducible evaluation of ANTs similarity metric performance in brain image registration. NeuroImage 54(3):2033-2044.
- Pallast N, Wieters F, Nill M, et al. (2017) Alzheimers Res Ther 9:94 (Antwerp atlas).
