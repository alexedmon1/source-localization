#!/usr/bin/env python3
"""
Create coarse parcellation NIfTI from the original 47-ROI atlas.

This script remaps the original atlas to a 22-ROI coarse parcellation:
- 14 subcortical ROIs (7 bilateral structures)
- 8 cortical ROIs (4 bilateral regions)

Dropped regions: White matter (Corpus Callosum), Septal Nucleus
"""

import json
import nibabel as nib
import numpy as np
from pathlib import Path


def create_coarse_atlas():
    """Create the coarse parcellation atlas."""

    # Paths
    script_dir = Path(__file__).parent
    atlas_dir = script_dir.parent

    # Load original atlas
    original_nii = nib.load(atlas_dir / "Atlas_3DRoisLeftRight.Labels.nii")
    original_data = original_nii.get_fdata().astype(np.int16)

    # Load coarse mapping
    with open(script_dir / "coarse_22roi_mapping.json", "r") as f:
        coarse_mapping = json.load(f)

    # Build remapping dictionary: original_index -> new_index
    remap = {}
    for new_id_str, roi_info in coarse_mapping["rois"].items():
        new_id = int(new_id_str)
        for orig_idx in roi_info["original_indices"]:
            remap[orig_idx] = new_id

    # All indices not in remap become 0 (background)
    # This includes white matter (1,2,3,24,25,26) and septal nucleus (10,33)

    # Create new atlas
    coarse_data = np.zeros_like(original_data)

    for orig_idx, new_idx in remap.items():
        coarse_data[original_data == orig_idx] = new_idx

    # Verify mapping
    unique_orig = np.unique(original_data)
    unique_coarse = np.unique(coarse_data)

    print(f"Original atlas unique values: {len(unique_orig)} ({unique_orig.min()}-{unique_orig.max()})")
    print(f"Coarse atlas unique values: {len(unique_coarse)} ({unique_coarse.min()}-{unique_coarse.max()})")

    # Check which original indices were dropped (mapped to 0)
    dropped = []
    for idx in unique_orig:
        if idx not in remap and idx != 0:
            dropped.append(idx)
    print(f"Dropped original indices: {dropped}")

    # Count voxels per ROI
    print("\nVoxel counts per coarse ROI:")
    for new_id_str, roi_info in coarse_mapping["rois"].items():
        new_id = int(new_id_str)
        count = np.sum(coarse_data == new_id)
        if count > 0 or new_id == 0:
            print(f"  {new_id:2d}: {roi_info['name']:20s} = {count:6d} voxels")

    # Save coarse atlas
    coarse_nii = nib.Nifti1Image(coarse_data, original_nii.affine, original_nii.header)
    output_path = script_dir / "coarse_22roi_atlas.nii"
    nib.save(coarse_nii, output_path)
    print(f"\nSaved coarse atlas to: {output_path}")

    # Also save compressed version
    output_path_gz = script_dir / "coarse_22roi_atlas.nii.gz"
    nib.save(coarse_nii, output_path_gz)
    print(f"Saved compressed atlas to: {output_path_gz}")

    return coarse_data


if __name__ == "__main__":
    create_coarse_atlas()
