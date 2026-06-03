# Allen Full-Brain Atlas: Methods

## Overview

This atlas provides an **anatomically constrained, depth-adaptive** whole-brain parcellation derived from the Allen Mouse Brain Common Coordinate Framework v3 (CCFv3), registered into the Antwerp (UAnterwerpen C57BL/6 MRI) coordinate space. It is designed for mouse EEG source localization with a 32-channel electrode array, where spatial resolution degrades with depth from the cortical surface.

**64 parcels** (32 per hemisphere) cover the grey matter brain volume, with finer parcellation near the electrodes (cortical surface) and coarser parcellation at depth, reflecting the physical limits of EEG spatial resolution. Parcellation respects major anatomical division boundaries (e.g., thalamus, hippocampus, cortex) — structures are never merged across divisions. Hemispheric symmetry is enforced: clustering is performed on the left hemisphere only, then mirrored to the right.

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
| Superficial | 0–2 mm | 2 mm | 28 |
| Mid-depth | 2–4 mm | 3 mm | 20 |
| Deep | 4+ mm | 4 mm | 16 |

Each Allen structure is assigned to a depth zone based on the **median depth** of its voxels.

### Hemisphere Split and Symmetric Clustering

The brain midline was determined from the **center-of-mass** of the annotation volume (voxel X = 30), not the volume center (X = 32). The Antwerp MRI template brain is offset ~2 voxels (0.4 mm) from the volume midpoint, and using the volume center caused a 20% L/R voxel imbalance.

Each Allen structure's voxels are split into left (X < midline) and right (X ≥ midline) portions. Clustering is performed on the **left hemisphere only**, then cluster assignments are mirrored to the right hemisphere using the same Allen structure IDs. This guarantees every parcel has both a L and R counterpart.

Right-hemisphere structures with no left counterpart (21 structures, mostly midline raphe nuclei and circumventricular organs with 1–75 voxels) are merged into their nearest same-division right-hemisphere parcel.

### Spatial Clustering

Within each (division × depth zone) group — **left hemisphere only** — Allen structures were clustered by centroid distance (in mm) using **complete-linkage hierarchical clustering**, with the distance threshold set to the zone's resolution.

### Size-Based Merging

Clusters failing the minimum-extent criterion — `min(R_extent, A_extent) < zone_resolution` — were iteratively merged into their nearest valid neighbor (by centroid distance, within the same group). If all clusters in a group are undersized, the largest is kept as-is.

### Result

| Zone | Left | Right | Total |
|------|------|-------|-------|
| Superficial (0–2 mm) | 14 | 14 | 28 |
| Mid-depth (2–4 mm) | 10 | 10 | 20 |
| Deep (4+ mm) | 8 | 8 | 16 |
| **Total** | **32** | **32** | **64** |

### ROI Categories

| Category | Parcels | Description |
|----------|---------|-------------|
| cortical | 18 | Isocortex (primary/secondary motor, somatosensory, visual, retrosplenial, insular) |
| brainstem | 14 | Midbrain + Pons + Medulla + superior colliculus |
| hippocampal | 8 | CA1/CA2, CA3, subiculum, hippocampo-amygdalar transition |
| olfactory | 8 | Main olfactory bulb, piriform, olfactory areas |
| cerebellum | 6 | Lobules IV-V, simple lobule, paraflocculus |
| subcortical | 4 | Caudoputamen, nucleus accumbens |
| thalamic | 4 | Reticular nucleus, ventral medial thalamus |
| hypothalamic | 2 | Hypothalamus |

### Composite ROI Naming (v0.2.0)

Four 32-ROI composites in the Allen32 parcellation carry display labels chosen to reflect the structures they actually bundle. Earlier identifiers (`Prefrontal_mPFC`, `Striatum`, `Amygdala`, `Brainstem`) named only the most prominent constituent and were misleading about the breadth of each composite.

| Identifier (v0.2.0) | Legacy alias | Bundled Allen structures | Naming rationale |
|---|---|---|---|
| `Frontal_Anterior_{L,R}` | `Prefrontal_mPFC_{L,R}` | Anterior cingulate (ACA), Prelimbic (PL), Infralimbic (ILA), Orbital (ORB), Frontal pole (FRP) — 38 sub-structures per hemisphere | ORB and FRP are lateral / ventral, not medial — the "mPFC" qualifier was inaccurate for the composite. |
| `Basal_Ganglia_{L,R}` | `Striatum_{L,R}` | Caudoputamen (CP), Nucleus accumbens (ACB), Olfactory tubercle (OT), **Pallidum (PAL)** | Pallidum (globus pallidus) is anatomically distinct from striatum; the broader basal-ganglia label is more accurate. |
| `Amygdalar_Complex_{L,R}` | `Amygdala_{L,R}` | Basolateral (BLA), Central (CEA), and Cortical (COA) amygdalar nuclei + **Claustrum (CLA)** + **Endopiriform nucleus (EP)** | CLA and EP are adjacent to but anatomically distinct from amygdala. |
| `Brainstem_Tectum_{L,R}` | `Brainstem_{L,R}` | Midbrain (MB: PAG, SN, VTA, RN, MRN, **superior + inferior colliculus**), Pons (P), Medulla (MY) | Superior / inferior colliculi are tectum (dorsal midbrain), not brainstem proper. |

The 10-region category `Prefrontal` was renamed to `Frontal-Anterior` for the same reason; the other 10-region categories are unchanged. `Striatum`, `Amygdala`, and `Brainstem` are inside the `Deep Subcortical` umbrella at the 10-region tier, which is unchanged. Region membership and label IDs (1--32) are unchanged — these are nomenclature clarifications only and have no effect on numerical results. Pre-v0.2.0 derivatives use the legacy names; downstream consumers should update their CSVs / configs accordingly.

## Files

| File | Description |
|------|-------------|
| `allen_labels.nii.gz` | Parcellation label volume (64 labels, 1-indexed, in Antwerp coordinate space) |
| `roi_mapping.json` | ROI metadata: names, abbreviations, colors, division, hemisphere, depth zone, Allen structure composition |
| `allen_annotation_in_antwerp.nii.gz` | Full Allen CCFv3 annotation (553 structures) warped to Antwerp space, for reference |
| `roi_categories.yaml` | 10-region grouping for the 32-ROI tier |
| `METHODS.md` | This file |

## Known Limitations

- **Pixelated boundaries**: Some label boundaries show single-voxel holes and staircase artifacts due to the ~8× resolution mismatch between Allen (25 µm) and Antwerp (200 µm) grids.
- **Registration quality**: The cross-modality registration (r = 0.572) is reasonable but not perfect. Boundary regions, particularly at the edges of the brain and in the olfactory bulb, show the largest misalignment.
- **Bregma-lambda compression**: The transformed bregma-lambda distance is 3.35 mm vs the expected ~4.2 mm, suggesting ~20% AP compression in the warp.
- **Right-hemisphere orphans**: 21 Allen structures exist only in the right hemisphere (mostly midline structures), absorbed into nearest same-division parcels.
- **Excluded white matter**: Fiber tract and ventricular voxels are unlabeled (background), creating gaps in the label volume. This is intentional — EEG does not detect signals from these tissues.

## Version History

- **v0.2.0 (2026-04-25)**: Renamed four composite ROI labels for nomenclature accuracy: `Prefrontal_mPFC` → `Frontal_Anterior`, `Striatum` → `Basal_Ganglia` (composite includes pallidum), `Amygdala` → `Amygdalar_Complex` (composite includes claustrum and endopiriform nucleus), `Brainstem` → `Brainstem_Tectum` (composite includes superior and inferior colliculi). 10-region category `Prefrontal` → `Frontal-Anterior`. Region membership and label IDs (1--32) unchanged. Legacy identifiers retained as deprecated aliases in `roi_categories.yaml`. See "Composite ROI Naming" section above.
- **v3 (2026-03-11)**: Hemispheric symmetry enforcement (64 ROIs, 32 per hemisphere). Fixed midline from volume center (X=32) to brain center-of-mass (X=30). Cluster left hemisphere, mirror to right. Absorb right-only orphan structures into nearest same-division parcel.
- **v2 (2026-03-11)**: Anatomically constrained parcellation (61 ROIs). Added division boundaries (thalamus, hippocampus, cortex, etc. never merge). Excluded fiber tracts and ventricles. 11 anatomical divisions × hemisphere × depth zone clustering. Had L/R asymmetry (33L vs 28R) due to wrong midline.
- **v1 (2026-03-06)**: Initial depth-adaptive parcellation (49 ROIs). Purely spatial clustering without anatomical constraints.

## Citations

- Wang Q, Ding SL, Li Y, et al. (2020) The Allen Mouse Brain Common Coordinate Framework: A 3D Reference Atlas. Cell 181(4):936-953.e20. https://doi.org/10.1016/j.cell.2020.04.007
- Avants BB, Tustison NJ, Song G, et al. (2011) A reproducible evaluation of ANTs similarity metric performance in brain image registration. NeuroImage 54(3):2033-2044.
- Pallast N, Wieters F, Nill M, et al. (2017) Alzheimers Res Ther 9:94 (Antwerp atlas).
