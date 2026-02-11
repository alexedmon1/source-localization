#!/usr/bin/env python3
"""
Atlas Utilities for UAnterwerpen C57BL/6 MRI Atlas

Provides voxel size correction and coordinate conversion functions
for the UAnterwerpen C57BL/6 MRI atlas, which has NIfTI header voxel
sizes scaled 10× larger than actual for software compatibility.

Author: CINCI Brain Lab
Date: 2025-11-13
"""

import numpy as np
import nibabel as nib

# ==============================================================================
# ATLAS VOXEL SIZE CORRECTION
# ==============================================================================
# UAnterwerpen C57BL/6 MRI Atlas has 10× scaled voxel sizes in NIfTI header
# to maintain compatibility with neuroimaging software that requires dims >= 1.
# Actual voxel sizes: 0.08 × 0.08 × 0.2 mm³ (X × Y × Z)
# Header reports: ~0.8 × 0.8 × 2.0 mm³
# ==============================================================================

ATLAS_VOXEL_SCALE_FACTOR = 0.1  # Multiply header values by this


def get_true_voxel_sizes(nifti_img):
    """
    Get actual voxel sizes for UAnterwerpen atlas.

    The atlas NIfTI header contains voxel sizes scaled 10× larger
    for software compatibility. This function returns the true sizes.

    Parameters
    ----------
    nifti_img : nibabel.Nifti1Image
        Loaded NIfTI image

    Returns
    -------
    tuple of float
        True voxel sizes in mm: (x_size, y_size, z_size)
        Expected: (0.08, 0.08, 0.2) for UAnterwerpen C57BL/6 atlas

    Examples
    --------
    >>> import nibabel as nib
    >>> img = nib.load('Atlas_3DRois.nii')
    >>> true_sizes = get_true_voxel_sizes(img)
    >>> print(true_sizes)
    (0.08, 0.08, 0.2)
    """
    header_zooms = nifti_img.header.get_zooms()[:3]  # First 3 dimensions only
    true_zooms = tuple(z * ATLAS_VOXEL_SCALE_FACTOR for z in header_zooms)
    return true_zooms


def get_true_affine(nifti_img):
    """
    Get corrected affine matrix with proper voxel→mm scaling.

    The affine matrix encodes the voxel-to-mm coordinate transformation.
    Since the header voxel sizes are 10× too large, the affine diagonal
    must be corrected by the same factor.

    Parameters
    ----------
    nifti_img : nibabel.Nifti1Image
        Loaded NIfTI image

    Returns
    -------
    np.ndarray, shape (4, 4)
        Corrected affine matrix for true voxel sizes

    Notes
    -----
    This corrects both the diagonal elements (voxel sizes) and any
    off-diagonal elements in the 3×3 rotation/scaling submatrix.
    The translation column (last column) is also scaled.

    Examples
    --------
    >>> import nibabel as nib
    >>> img = nib.load('Atlas_3DRois.nii')
    >>> true_affine = get_true_affine(img)
    >>> # Diagonal should be ~[0.08, 0.08, 0.2] instead of [0.8, 0.8, 2.0]
    """
    affine = nifti_img.affine.copy()
    # Scale the entire 3×3 rotation/scaling submatrix
    affine[:3, :3] *= ATLAS_VOXEL_SCALE_FACTOR
    # Scale the translation vector (origin offset)
    affine[:3, 3] *= ATLAS_VOXEL_SCALE_FACTOR
    return affine


def voxel_to_mm(voxel_coords, nifti_img):
    """
    Convert voxel coordinates to mm using corrected affine.

    Parameters
    ----------
    voxel_coords : np.ndarray, shape (N, 3) or (3,)
        Voxel coordinates (0-indexed)
    nifti_img : nibabel.Nifti1Image
        Loaded NIfTI image

    Returns
    -------
    np.ndarray, shape (N, 3) or (3,)
        Coordinates in mm

    Examples
    --------
    >>> import nibabel as nib
    >>> import numpy as np
    >>> img = nib.load('Atlas_3DRois.nii')
    >>> bregma_vox = np.array([30, 149, 41])
    >>> bregma_mm = voxel_to_mm(bregma_vox, img)
    >>> print(bregma_mm)  # Should be reasonable mm values (e.g., [2.4, 11.92, 8.2])
    """
    affine = get_true_affine(nifti_img)
    return nib.affines.apply_affine(affine, voxel_coords)


def mm_to_voxel(mm_coords, nifti_img):
    """
    Convert mm coordinates to voxel using corrected affine.

    Parameters
    ----------
    mm_coords : np.ndarray, shape (N, 3) or (3,)
        Coordinates in mm
    nifti_img : nibabel.Nifti1Image
        Loaded NIfTI image

    Returns
    -------
    np.ndarray, shape (N, 3) or (3,)
        Voxel coordinates (0-indexed, may be fractional)

    Examples
    --------
    >>> import nibabel as nib
    >>> import numpy as np
    >>> img = nib.load('Atlas_3DRois.nii')
    >>> bregma_mm = np.array([2.4, 11.92, 8.2])
    >>> bregma_vox = mm_to_voxel(bregma_mm, img)
    >>> print(np.round(bregma_vox))  # Should be close to [30, 149, 41]
    """
    affine = get_true_affine(nifti_img)
    inv_affine = np.linalg.inv(affine)
    return nib.affines.apply_affine(inv_affine, mm_coords)


def validate_voxel_sizes(nifti_img, expected=(0.203, 0.080, 0.200), tolerance=0.01):
    """
    Validate that corrected voxel sizes match expected values.

    Parameters
    ----------
    nifti_img : nibabel.Nifti1Image
        Loaded NIfTI image
    expected : tuple of float, optional
        Expected voxel sizes in mm (x, y, z)
        Default: (0.203, 0.080, 0.200) for UAnterwerpen atlas
    tolerance : float, optional
        Maximum allowed difference in mm
        Default: 0.01 mm

    Returns
    -------
    bool
        True if voxel sizes match expected within tolerance

    Raises
    ------
    ValueError
        If voxel sizes do not match expected values

    Examples
    --------
    >>> import nibabel as nib
    >>> img = nib.load('Atlas_3DRois.nii')
    >>> validate_voxel_sizes(img)  # Raises ValueError if incorrect
    True
    """
    true_sizes = get_true_voxel_sizes(nifti_img)
    expected = np.array(expected)
    true_sizes_arr = np.array(true_sizes)

    diff = np.abs(true_sizes_arr - expected)

    if np.any(diff > tolerance):
        raise ValueError(
            f"Voxel sizes do not match expected values!\n"
            f"  Expected: {expected}\n"
            f"  Got:      {true_sizes_arr}\n"
            f"  Diff:     {diff}\n"
            f"  Tolerance: {tolerance} mm\n"
            f"This may indicate a different atlas or incorrect scaling factor."
        )

    return True


def compute_distance_mm(coords1, coords2, nifti_img):
    """
    Compute Euclidean distance between two points in mm.

    Convenience function that handles voxel→mm conversion automatically.

    Parameters
    ----------
    coords1 : np.ndarray, shape (3,)
        First point in voxel coordinates (0-indexed)
    coords2 : np.ndarray, shape (3,)
        Second point in voxel coordinates (0-indexed)
    nifti_img : nibabel.Nifti1Image
        Loaded NIfTI image

    Returns
    -------
    float
        Distance in mm

    Examples
    --------
    >>> import nibabel as nib
    >>> import numpy as np
    >>> img = nib.load('Atlas_3DRois.nii')
    >>> bregma = np.array([30, 149, 41])
    >>> lambda_pt = np.array([30, 97, 41])
    >>> dist = compute_distance_mm(bregma, lambda_pt, img)
    >>> print(f"{dist:.2f} mm")  # Should be ~4.16 mm
    4.16 mm
    """
    coords1_mm = voxel_to_mm(coords1, nifti_img)
    coords2_mm = voxel_to_mm(coords2, nifti_img)
    return float(np.linalg.norm(coords2_mm - coords1_mm))


# ==============================================================================
# Validation and QC Functions
# ==============================================================================

def validate_bregma_lambda_distance(bregma_vox, lambda_vox, nifti_img,
                                     expected=4.2, tolerance=0.3):
    """
    Validate that Bregma-Lambda distance matches expected value.

    Parameters
    ----------
    bregma_vox : array-like, shape (3,)
        Bregma voxel coordinates (0-indexed)
    lambda_vox : array-like, shape (3,)
        Lambda voxel coordinates (0-indexed)
    nifti_img : nibabel.Nifti1Image
        Loaded NIfTI image
    expected : float, optional
        Expected distance in mm (default: 4.2 mm for C57BL/6)
    tolerance : float, optional
        Acceptable deviation in mm (default: 0.3 mm)

    Returns
    -------
    dict
        Dictionary with validation results:
        - 'distance_mm': float, actual distance
        - 'expected_mm': float, expected distance
        - 'deviation_mm': float, absolute deviation
        - 'within_tolerance': bool, whether distance is acceptable
        - 'bregma_mm': array, Bregma in mm
        - 'lambda_mm': array, Lambda in mm

    Examples
    --------
    >>> import nibabel as nib
    >>> import numpy as np
    >>> img = nib.load('Atlas_3DRois.nii')
    >>> bregma = np.array([30, 149, 41])
    >>> lambda_pt = np.array([30, 97, 41])
    >>> result = validate_bregma_lambda_distance(bregma, lambda_pt, img)
    >>> print(f"Distance: {result['distance_mm']:.2f} mm")
    >>> print(f"Valid: {result['within_tolerance']}")
    """
    bregma_vox = np.asarray(bregma_vox, dtype=float)
    lambda_vox = np.asarray(lambda_vox, dtype=float)

    bregma_mm = voxel_to_mm(bregma_vox, nifti_img)
    lambda_mm = voxel_to_mm(lambda_vox, nifti_img)

    distance = float(np.linalg.norm(lambda_mm - bregma_mm))
    deviation = abs(distance - expected)

    return {
        'distance_mm': distance,
        'expected_mm': expected,
        'deviation_mm': deviation,
        'within_tolerance': deviation <= tolerance,
        'bregma_mm': bregma_mm,
        'lambda_mm': lambda_mm,
        'delta_mm': lambda_mm - bregma_mm
    }


def check_midline_alignment(coords1, coords2, nifti_img, tolerance_mm=0.5):
    """
    Check if two points are aligned on the midline (same X coordinate).

    Parameters
    ----------
    coords1 : array-like, shape (3,)
        First point voxel coordinates
    coords2 : array-like, shape (3,)
        Second point voxel coordinates
    nifti_img : nibabel.Nifti1Image
        Loaded NIfTI image
    tolerance_mm : float, optional
        Maximum X-coordinate deviation in mm (default: 0.5 mm)

    Returns
    -------
    dict
        Dictionary with alignment check results:
        - 'aligned': bool, whether points are on midline
        - 'deviation_mm': float, X-coordinate deviation
        - 'tolerance_mm': float, tolerance used

    Examples
    --------
    >>> # Bregma and Lambda should both be on midline
    >>> result = check_midline_alignment(bregma, lambda_pt, img)
    >>> print(f"Midline aligned: {result['aligned']}")
    """
    coords1_mm = voxel_to_mm(np.asarray(coords1, dtype=float), nifti_img)
    coords2_mm = voxel_to_mm(np.asarray(coords2, dtype=float), nifti_img)

    deviation = abs(coords1_mm[0] - coords2_mm[0])

    return {
        'aligned': deviation <= tolerance_mm,
        'deviation_mm': float(deviation),
        'tolerance_mm': tolerance_mm,
        'coords1_x_mm': float(coords1_mm[0]),
        'coords2_x_mm': float(coords2_mm[0])
    }


def check_depth_consistency(coords1, coords2, nifti_img, tolerance_mm=1.0):
    """
    Check if two points have consistent dorsoventral depth (Z coordinate).

    Useful for validating that Bregma and Lambda are on the same
    dorsal surface (level skull).

    Parameters
    ----------
    coords1 : array-like, shape (3,)
        First point voxel coordinates
    coords2 : array-like, shape (3,)
        Second point voxel coordinates
    nifti_img : nibabel.Nifti1Image
        Loaded NIfTI image
    tolerance_mm : float, optional
        Maximum Z-coordinate deviation in mm (default: 1.0 mm)

    Returns
    -------
    dict
        Dictionary with depth consistency check results:
        - 'consistent': bool, whether depths are similar
        - 'deviation_mm': float, Z-coordinate deviation
        - 'tolerance_mm': float, tolerance used

    Examples
    --------
    >>> # Bregma and Lambda should have similar Z (level skull)
    >>> result = check_depth_consistency(bregma, lambda_pt, img)
    >>> print(f"Depth consistent: {result['consistent']}")
    """
    coords1_mm = voxel_to_mm(np.asarray(coords1, dtype=float), nifti_img)
    coords2_mm = voxel_to_mm(np.asarray(coords2, dtype=float), nifti_img)

    deviation = abs(coords1_mm[2] - coords2_mm[2])

    return {
        'consistent': deviation <= tolerance_mm,
        'deviation_mm': float(deviation),
        'tolerance_mm': tolerance_mm,
        'coords1_z_mm': float(coords1_mm[2]),
        'coords2_z_mm': float(coords2_mm[2])
    }


# ==============================================================================
# Main (for testing)
# ==============================================================================

if __name__ == '__main__':
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python atlas_utils.py <path_to_atlas.nii>")
        print("\nExample:")
        print("  python atlas_utils.py inputs/Atlas_3DRois.nii")
        sys.exit(1)

    atlas_path = Path(sys.argv[1])
    if not atlas_path.exists():
        print(f"Error: File not found: {atlas_path}")
        sys.exit(1)

    print("=" * 70)
    print("Atlas Utilities - UAnterwerpen C57BL/6 MRI Atlas")
    print("=" * 70)

    # Load atlas
    print(f"\nLoading atlas: {atlas_path}")
    img = nib.load(str(atlas_path))

    # Check header voxel sizes
    header_sizes = img.header.get_zooms()[:3]
    print(f"\nHeader voxel sizes: {header_sizes} mm")

    # Get true voxel sizes
    true_sizes = get_true_voxel_sizes(img)
    print(f"True voxel sizes:   {true_sizes} mm (× {ATLAS_VOXEL_SCALE_FACTOR})")

    # Validate
    try:
        validate_voxel_sizes(img)
        print("✓ Voxel sizes validated")
    except ValueError as e:
        print(f"✗ Validation failed:\n{e}")

    # Test landmarks
    print("\n" + "=" * 70)
    print("Testing Bregma-Lambda Landmarks")
    print("=" * 70)

    bregma_vox = np.array([30, 149, 41])
    lambda_vox = np.array([30, 97, 41])

    print(f"\nBregma voxel: {bregma_vox}")
    print(f"Lambda voxel: {lambda_vox}")

    # Convert to mm
    bregma_mm = voxel_to_mm(bregma_vox, img)
    lambda_mm = voxel_to_mm(lambda_vox, img)

    print(f"\nBregma mm: [{bregma_mm[0]:.2f}, {bregma_mm[1]:.2f}, {bregma_mm[2]:.2f}]")
    print(f"Lambda mm: [{lambda_mm[0]:.2f}, {lambda_mm[1]:.2f}, {lambda_mm[2]:.2f}]")

    # Distance
    dist_result = validate_bregma_lambda_distance(bregma_vox, lambda_vox, img)
    print(f"\nBregma-Lambda distance: {dist_result['distance_mm']:.2f} mm")
    print(f"Expected: {dist_result['expected_mm']:.2f} ± 0.3 mm")
    print(f"Deviation: {dist_result['deviation_mm']:.2f} mm")
    print(f"✓ Valid" if dist_result['within_tolerance'] else "✗ Outside tolerance")

    # Midline
    midline_result = check_midline_alignment(bregma_vox, lambda_vox, img)
    print(f"\nMidline alignment:")
    print(f"  Bregma X: {midline_result['coords1_x_mm']:.2f} mm")
    print(f"  Lambda X: {midline_result['coords2_x_mm']:.2f} mm")
    print(f"  Deviation: {midline_result['deviation_mm']:.3f} mm")
    print(f"  ✓ Aligned" if midline_result['aligned'] else "  ✗ Not aligned")

    # Depth
    depth_result = check_depth_consistency(bregma_vox, lambda_vox, img)
    print(f"\nDepth consistency:")
    print(f"  Bregma Z: {depth_result['coords1_z_mm']:.2f} mm")
    print(f"  Lambda Z: {depth_result['coords2_z_mm']:.2f} mm")
    print(f"  Deviation: {depth_result['deviation_mm']:.3f} mm")
    print(f"  ✓ Consistent" if depth_result['consistent'] else "  ✗ Not consistent")

    print("\n" + "=" * 70)
    print("Validation complete!")
    print("=" * 70)
