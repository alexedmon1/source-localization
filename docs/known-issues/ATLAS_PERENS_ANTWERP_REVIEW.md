# Atlas review: Allen labels in Antwerp space, and a possible Perens-based head model

**Date**: 2026-10-20
**Status**: Two defects identified (not fixed here); a new atlas is available as an option to evaluate.
**Source**: built and measured in neurofaune (`feat/mouse-support`, commits d3ea8a3, ddb9927).
The package and full numbers are in `/mnt/arborea/atlases/Antwerp/README.md` and
`/mnt/arborea/atlases/Antwerp/comparison/`.

## 1. Defects in the bundled Allen atlas (`data/atlas/allen/`)

### 1a. Hippocampus_Ant / Hippocampus_Post are spatially swapped

In `allen_labels.nii.gz` (allen32), IDs 7/23 ("Hippocampus_Ant") occupy y voxels 81–103 (mean 97),
and IDs 8/24 ("Hippocampus_Post") occupy y 104–161 (mean 115). In this volume y increases
rostrally: the olfactory bulb is at mean y 200 and the cerebellum at mean y 66. So "Ant" is the
caudal half.

`allen26_labels.nii.gz` inherits the swap (IDs 4/14 vs 5/15). Any result naming these parcels
refers to the opposite half. Fix by swapping the names in the ROI mappings, not the voxels, so
existing label IDs stay valid. Check `source-analytics` outputs that report them.

### 1b. 19 non-existent CCFv3 structure IDs in `allen_annotation_in_antwerp.nii.gz`

19 IDs, covering 3,443 voxels (~11 mm³), are not in the CCFv3 ontology. That is the signature of
warping raw CCFv3 IDs through ANTs in float32. CCFv3 IDs reach 614,454,277, but float32 holds
integers exactly only up to 2^24, so large layer-subdivision IDs come back as nearby IDs that do
not exist. Those voxels then fall out of every parcel.

Fix: compact the labels to 0..N before warping and map them back afterwards. This is what
`neurofaune.atlas.label_transfer.warp_labels` does.

### 1c. METHODS.md corrections

- **The moving image was `average_template_25.nrrd`**, the CCFv3 serial two-photon tomography
  autofluorescence average. It is not Nissl and not MRI, so the registration was cross-modality
  (r = 0.572).
- **The "~20% AP compression" is not supported.** Rostro-caudal distances between CCFv3 landmark
  centroids (MOB, anterior and posterior commissure, inferior colliculus, cerebellum) in this
  volume are 0.98–1.02× those of the in situ Perens brain. The 3.35 mm bregma–lambda figure must
  come from how bregma and lambda were placed in CCF, not from the warp.

## 2. Preset bregma / lambda vs skull-derived bregma

Perens et al. 2023 provide skull-derived stereotaxic coordinates (micro-CT sutures) for every
voxel of their MRI template. Carried into Antwerp through a T2w→T2w registration:

- **Skull bregma in Antwerp:** world (0.01, 1.47, 2.79) mm, native voxel (29.8, 145.9, 38.5).
  Using only the affine part of the warp moves it 0.15 mm.
- **Preset `bregma_vox` [30, 149, 41]** is **0.57 mm** away: 0.25 mm rostral and **0.5 mm dorsal**
  of skull bregma. All electrodes share Z = 41, so the array sits about 0.5 mm above the skull
  surface by this estimate.
- **Preset `lambda_vox` [30, 97, 41]** is at AP −3.62 mm in the skull frame. The preset
  bregma–lambda gap is 4.16 Antwerp-mm but 3.88 skull-mm.
- **Antwerp is about 3% longer rostro-caudally than skull mm** over the dorsal head, and about 9%
  within ±2 mm of bregma, because the in vivo brain is larger (564 vs 514 mm³). Electrode offsets
  from the H32 spec are in skull mm but are placed 1:1 in Antwerp mm, so electrodes near the
  array edges are off by about 0.1–0.35 mm.

All of this is below the 0.5–1.5 mm source grid, but it is systematic.

## 3. New atlas option: Antwerp + CCFv3 through the Perens T2w bridge

`/mnt/arborea/atlases/Antwerp/` is the same Antwerp grid with:

- **CCFv3 annotation, 665 structures** (fiber tracts and ventricles included, 0 invalid IDs), from
  the Perens T2w template, which already carries CCFv3 labels. The registration is T2w→T2w,
  r = 0.79.
- **allen32 / allen26 / allen64 rebuilt** from the same structure-ID definitions
  (`roi_mapping_*.json`), with the hippocampal halves named by position.
- **An isotropic 0.1 mm hub grid** with the same labels.
- **Skull-frame maps** (`stereotaxic_{ml,ap,dv}_mm.nii.gz`) and `stereotaxic.json`.

**Old vs new, per parcel** (`comparison/allen32_old_vs_new.tsv`):
- Dorsal-cortex parcels move 0.07–0.27 mm (Dice 0.77–0.90).
- Rostral hippocampus moves 0.31–0.42 mm; everything else ≤ 0.27 mm.
- Against Antwerp's own MRI-drawn labels, median Dice is 0.53 new vs 0.49 old. Dorsal cortex and
  cerebellum improve; thalamus, striatum, septum, cingulate and orbital get worse.
- The new labels stop at the brain edge (Dice vs the Antwerp brain mask 0.90 vs 0.96), leaving a
  ~0.1–0.2 mm surface-CSF gap.

**As a drop-in label map:** add it as a new registry entry (e.g. `allen32_v2`, `allen26_v2`) next to
the old one. Do not replace the old one, because existing results were made with it.

## 4. To evaluate: a Perens-based head model

The Perens release (`/mnt/arborea/atlases/perens_2023/Multimodal_mouse_brain_atlas_files/`)
includes:
- an in situ T2w brain template (25 µm)
- full CCFv3 annotation
- tissue, ventricle and hemisphere masks
- skull-derived bregma coordinates
- deformation fields to and from CCFv3 and LSFM

A head model built in Perens space would have:

- **Electrodes placed directly in skull mm** from bregma, with no Antwerp-vs-skull scaling and no
  preset-bregma offset.
- **Source space and labels from CCFv3** natively; `neurofaune.atlas.antwerp.parcellate` rebuilds
  allen32/26/64 on any CCFv3 annotation.
- **An in situ brain shape** (inside an intact skull), arguably closer to the real conductor than
  an in vivo template whose BEM is an ellipsoid fitted to the brain anyway.

Caveats:
- The release has no skull or scalp surfaces (no CT template in this download), so the BEM would
  still be the fitted ellipsoid/sphere, or skull surfaces would come from elsewhere.
- Every downstream result changes coordinates. This would be a new head model version, not a
  patch.
- Validation numbers are self-consistent under either atlas. The gain is anatomical validity and
  a direct MRI↔EEG correspondence, not better simulated localization.

**Suggested next step:** add `allen26_v2` as a registry entry and rerun one real-data ROI
attribution to see whether region assignments actually change before considering a head-model
switch.

## 5. Other atlases to consider

The full list of candidate mouse atlases, anatomical and functional, with status, download
locations, and what is already on `/mnt/arborea/atlases/`, is kept in neurofaune:
`~/sandbox/neurofaune/docs/MOUSE_ATLAS_CANDIDATES.md` (branch `feat/mouse-support`). It covers
RABIES (in use), Grandjean 2020 multi-centre, the Gozzi triple-network, Allen connectivity
modules (Harris 2019), DSURQE/Dorr, BlueBrain CCFv3-augmented, the Duke stereotaxic atlas,
AMBMC, DevCCF and others.

Most relevant here:
- **Allen connectivity modules** as a coarse grouping of allen26. No registration needed.
- **BlueBrain CCFv3-augmented** labels before any new head model.
- **Duke stereotaxic bregma**, as a cross-check on the preset bregma.

