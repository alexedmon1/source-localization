# Electrode Coordinate Setup Guide

**Purpose:** Guide for setting up electrode coordinates for your specific mouse brain atlas and electrode array.

**Created:** 2025-11-26
**Last Updated:** 2025-11-26

---

## Overview

To perform EEG source localization, you need electrode coordinates that are properly registered to your MRI atlas. This guide explains how to:

1. Measure electrode positions from a photograph of your array
2. Calculate MRI voxel coordinates using Bregma as a reference landmark
3. Generate the CSV file required by the pipeline

---

## Prerequisites

### Required Materials

1. **Photograph of electrode array** placed on the mouse skull
   - Clear view of all electrodes
   - Bregma suture visible (or marked)
   - Known pixel dimensions (mm/pixel)

2. **MRI Atlas** (e.g., UAnterwerpen C57BL/6 Atlas)
   - NIfTI file with brain anatomy
   - Known voxel dimensions
   - Manually identified Bregma coordinates in MRI space

3. **Electrode labels** - unique identifier for each electrode (e.g., E1, E2, ...)

---

## Step 1: Identify Bregma in Your MRI Atlas

Bregma is the skull landmark at the intersection of the coronal and sagittal sutures. Since most brain atlases don't include the skull, you must estimate Bregma's position using brain anatomy.

### Method: Manual Alignment to Histological Atlas

1. **Open your MRI atlas** in a NIfTI viewer (e.g., FSLeyes, ITK-SNAP)

2. **Load Paxinos & Franklin Mouse Brain Atlas** (histology reference)
   - Find the Bregma 0.00mm coronal section (typically Figure 31)

3. **Match anatomical features** between histology and MRI:
   - Corpus callosum shape
   - Ventricle size and position
   - Cortical layers
   - Striatum boundaries

4. **Record Bregma coordinates** in 0-indexed voxel space
   - Example for UAnterwerpen atlas: `(30, 149, 41)`
   - This becomes your reference point for all electrodes

### Validation: Bregma-Lambda Distance

Lambda is another skull landmark ~4.2mm posterior to Bregma in C57BL/6 mice.

1. **Identify Lambda** using the same histology matching approach
   - Example for UAnterwerpen: `(30, 97, 41)` (same X and Z as Bregma)

2. **Verify distance** using the atlas voxel sizes:
   ```python
   # UAnterwerpen voxel sizes (corrected): 0.203125 × 0.080078125 × 0.200 mm
   distance_y_voxels = 149 - 97  # = 52 voxels
   distance_mm = 52 × 0.080078125  # = 4.16 mm ≈ 4.2 mm ✓
   ```

3. **Expected distance:** 4.2 ± 0.3 mm
   - Within tolerance? Bregma is likely correct
   - Outside tolerance? Re-check anatomical alignment

---

## Step 2: Measure Electrodes from Photograph

### Photo Calibration

1. **Determine pixel dimensions** (mm/pixel):
   - Method A: Known ruler in photo
   - Method B: Known electrode spacing
   - Method C: Camera specifications + working distance

   Example values:
   ```
   X_pixel = 0.01385 mm/pixel
   Y_pixel = 0.01564 mm/pixel
   ```

2. **Locate Bregma in photo** (pixel coordinates):
   - Mark the suture intersection
   - Record pixel position from image origin

   Example: `Bregma_photo = (336, 182) pixels`

### Electrode Measurements

For each electrode, measure the distance from the image origin (top-left corner):

```
Label    Xmm (mm)    Ymm (mm)
E1       0.55        7.06
E2       2.40        7.04
E3       3.64        7.05
...
```

**Tool:** Any image measurement software (ImageJ/Fiji, GIMP, calibrated ruler)

---

## Step 3: Calculate MRI Voxel Coordinates

Use the provided `recalculate_electrode_coords.py` script to convert photo measurements to MRI coordinates.

### Script Location

```bash
source-localization/scripts/recalculate_electrode_coords.py
```

### Usage

```bash
cd /path/to/source-localization

python scripts/recalculate_electrode_coords.py \
  --input /path/to/electrode_measurements.csv \
  --output /path/to/mouse_array_coords.csv \
  --bregma-mri 30 149 41 \
  --mri-voxel-sizes 0.203125 0.080078125 0.200 \
  --photo-pixel-sizes 0.01385 0.01564 \
  --bregma-photo-pixels 336 182
```

### Parameters

- `--input`: CSV with columns `Label`, `Xmm`, `Ymm` (manual photo measurements)
- `--output`: Output CSV with calculated MRI coordinates
- `--bregma-mri`: Bregma position in MRI voxel space (0-indexed)
- `--mri-voxel-sizes`: MRI voxel dimensions in mm (X, Y, Z)
- `--photo-pixel-sizes`: Photo pixel dimensions in mm (X, Y)
- `--bregma-photo-pixels`: Bregma location in photo pixels (X, Y)

### Input CSV Format

```csv
Label,Xmm,Ymm
E1,0.55,7.06
E2,2.40,7.04
E3,3.64,7.05
...
Bregma,4.66,2.85
```

### Output CSV Format

The script generates a complete CSV with all calculated fields:
```csv
Label,Xmm,Ymm,Xprec,Yprec,X,Y,...,Xbregmm,Ybregmm,...,X-MRI,Y-MRI,Z-MRI
E1,0.55,7.06,39.71,451.41,40,451,...,-4.10,4.21,...,9.82,96.46,41
...
```

Key output columns:
- `X-MRI`, `Y-MRI`, `Z-MRI`: Electrode positions in MRI voxel space (1-indexed for CSV)
- `Xbregmm`, `Ybregmm`: Offset from Bregma in mm (for verification)

---

## Step 4: Configure Pipeline

Once you have the electrode CSV, configure the pipeline to use it:

### Update Config File

Edit `config/default_config.yaml` (or your custom config):

```yaml
inputs:
  electrodes_csv: "/path/to/your/mouse_array_coords.csv"  # Use absolute path for study data

electrode:
  projection_method: "intensity"  # Validated projection method
  skull_offset_mm: 0.0  # Electrodes at skull surface
  bregma_vox: [30, 149, 41]  # Bregma for validation (0-indexed)
  lambda_vox: [30, 97, 41]   # Lambda for validation (0-indexed)
  create_visualization: true  # Generate QC figures
```

### Run Pipeline

```bash
source-localization \
  --preset ellipsoid_volumetric \
  --eeg /path/to/eeg_data.set \
  --output /path/to/output_dir \
  --method dSPM
```

The pipeline will:
1. Load electrodes from your CSV
2. Project them onto the skull surface
3. Validate with Bregma-Lambda distance check
4. Generate visualization for QC

---

## Validation and Quality Control

### Automated Checks

The pipeline performs several automatic validations:

1. **Bregma validation**
   - Checks if Bregma coordinates fall inside brain mask
   - Validates against expected anatomical location

2. **Lambda distance check**
   - If Lambda coordinates provided, validates 4.2 ± 0.3 mm distance
   - Ensures coordinate system consistency

3. **Projection quality**
   - Verifies all electrodes successfully project to skull surface
   - Checks for outliers or unexpected Z-adjustments

### Visual Inspection

The generated `electrode_registration_validation.png` shows:
- **3D view**: Electrodes on brain surface
- **Slice views**: Coronal, sagittal, axial with electrode positions
- **Projection statistics**: Mean Z-adjustment, distance metrics
- **Electrode table**: All coordinates with QC metrics

**Red flags to check:**
- Electrodes outside brain volume
- Extremely large Z-adjustments (>5mm)
- Bregma-Lambda distance outside 3.9-4.5mm range
- Non-uniform electrode distribution

---

## Troubleshooting

### Issue: "Bregma outside brain mask"

**Cause:** Bregma MRI coordinates don't fall inside brain tissue

**Solution:**
1. Re-check Bregma identification in MRI atlas
2. Verify you're using 0-indexed coordinates
3. Try adjacent voxels (±1-2 voxels in Y and Z)

### Issue: "Bregma-Lambda distance out of range"

**Expected:** 4.2 ± 0.3 mm for C57BL/6 mice

**If too short (<3.9mm):**
- Lambda may be too anterior
- Check Paxinos-Franklin alignment

**If too long (>4.5mm):**
- Lambda may be too posterior
- Verify voxel size correction (10× scaling for UAnterwerpen)

### Issue: "Large Z-adjustments during projection"

**Cause:** Flat Z-coordinate from photo doesn't match curved skull surface

**Expected:** Z-adjustments typically 0.2-1.0mm

**If much larger (>2mm):**
- Initial Z-estimate may be incorrect
- Brain surface extraction may have issues
- Consider using `projection_method: 'rbf'` for smoother fit

### Issue: "Electrodes project outside brain"

**Cause:** Electrode array extends beyond brain boundaries

**Solutions:**
1. Verify photo measurements are correct
2. Check Bregma alignment accuracy
3. Confirm electrode array is centered on skull
4. Consider excluding peripheral electrodes

---

## Atlas-Specific Notes

### UAnterwerpen C57BL/6 Atlas

**File:** `Atlas_3DRois.nii`

**Critical voxel scaling issue:**
- NIfTI header reports voxel sizes 10× larger than actual
- **Header:** 2.03125 × 0.80078125 × 2.0 mm
- **Actual:** 0.203125 × 0.080078125 × 0.2 mm
- Pipeline automatically corrects this using `atlas_utils.get_true_affine()`

**Validated landmarks:**
- Bregma: `(30, 149, 41)` in 0-indexed voxels
- Lambda: `(30, 97, 41)` in 0-indexed voxels
- Distance: 4.16 mm (validates anatomical accuracy)

### Custom Atlas

If using a different atlas:

1. **Determine voxel sizes**
   ```python
   import nibabel as nib
   img = nib.load('your_atlas.nii')
   voxel_sizes = img.header.get_zooms()[:3]
   print(f"Voxel sizes: {voxel_sizes} mm")
   ```

2. **Identify Bregma manually**
   - Use histological atlas as reference
   - Match anatomical features
   - Record 0-indexed voxel coordinates

3. **Update config**
   ```yaml
   electrode:
     bregma_vox: [X, Y, Z]  # Your atlas Bregma
   ```

4. **Update script parameters**
   ```bash
   --mri-voxel-sizes X Y Z  # Your atlas voxel sizes
   ```

---

## References

### Anatomical References

- **Paxinos, G. & Franklin, K. B. J.** (2019). *The Mouse Brain in Stereotaxic Coordinates* (5th ed.). Academic Press.
  - Used for Bregma and Lambda landmark identification
  - Coronal sections with stereotaxic coordinates

- **Ma et al. (2008)** "In vivo 3D digital atlas database of the adult C57BL/6J mouse brain by magnetic resonance microscopy"
  - UAnterwerpen atlas source publication
  - 7T MRI acquisition parameters

### Methodological References

- Surface-based skull landmark estimation is commonly used in rodent neuroimaging when CT is unavailable
- Expected accuracy: 0.3-0.8mm (acceptable for source localization at typical source spacings of 1-2mm)
- Cross-modal registration (histology → MRI) inherently has uncertainty due to tissue processing differences

---

## Example Workflow

### Complete Example: New Electrode Array

```bash
# 1. Measure electrodes from photo, create input CSV
cat > electrode_measurements.csv << EOF
Label,Xmm,Ymm
E1,0.55,7.06
E2,2.40,7.04
E3,3.64,7.05
Bregma,4.66,2.85
EOF

# 2. Calculate MRI coordinates
python scripts/recalculate_electrode_coords.py \
  --input electrode_measurements.csv \
  --output mouse_array_coords.csv \
  --bregma-mri 30 149 41 \
  --mri-voxel-sizes 0.203125 0.080078125 0.200 \
  --photo-pixel-sizes 0.01385 0.01564 \
  --bregma-photo-pixels 336 182

# 3. Verify output
head mouse_array_coords.csv

# 4. Run pipeline with electrode validation
source-localization \
  --eeg /path/to/data.set \
  --output /path/to/output \
  --config custom_config.yaml

# 5. Check electrode registration QC
open /path/to/output/electrode_registration/electrode_registration_validation.png
```

---

## Support

For questions or issues with electrode coordinate setup:

1. Check validation outputs in `electrode_registration_validation.png`
2. Review Bregma-Lambda distance validation
3. Consult the example CSV file: `data/electrodes/mouse_array_coords.csv`
4. See the detailed protocol: `ELECTRODE_COORDINATE_CALCULATION.md` (in validation/production docs)

---

## Summary Checklist

- [ ] Identify Bregma in MRI atlas using histology reference
- [ ] Validate Lambda distance (~4.2mm posterior to Bregma)
- [ ] Measure photo pixel dimensions
- [ ] Locate Bregma in photo
- [ ] Measure all electrode positions in photo
- [ ] Run `recalculate_electrode_coords.py` script
- [ ] Inspect generated CSV for reasonable coordinates
- [ ] Update config with electrode CSV path
- [ ] Run pipeline with validation enabled
- [ ] Check QC visualization for any issues
- [ ] Verify Bregma-Lambda distance in output logs

---

**Note:** Electrode coordinate accuracy is critical for source localization quality. Take time to carefully validate each step!
