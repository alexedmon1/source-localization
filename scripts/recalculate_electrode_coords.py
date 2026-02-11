#!/usr/bin/env python3
"""
Recalculate electrode MRI coordinates based on updated Bregma position.

This script reproduces the electrode coordinate calculation pipeline:
1. Photo measurements (Xmm, Ymm) → pixel coords → offset from Bregma
2. Convert offsets to mm using photo pixel sizes
3. Convert mm offsets to MRI voxel coordinates using MRI voxel sizes

Usage:
    python recalculate_electrode_coords.py \
        --input inputs/mouse_array_coords.csv \
        --output inputs/mouse_array_coords_UPDATED.csv \
        --bregma-mri 30 149 41

Author: Pipeline validation
Date: 2025-11-13
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def recalculate_mri_coords(
    df: pd.DataFrame,
    bregma_mri: tuple,
    mri_voxel_sizes: tuple = (0.203125, 0.080078125, 0.200),
    photo_pixel_sizes: tuple = (0.01385, 0.01564),
    bregma_photo_pixels: tuple = (336, 182)
) -> pd.DataFrame:
    """
    Recalculate X-MRI, Y-MRI, Z-MRI columns based on updated Bregma MRI coordinates.

    Parameters
    ----------
    df : pd.DataFrame
        Input electrode coordinates dataframe
    bregma_mri : tuple
        (X, Y, Z) Bregma coordinates in MRI voxel space (0-indexed)
    mri_voxel_sizes : tuple
        (X, Y, Z) MRI voxel dimensions in mm
    photo_pixel_sizes : tuple
        (X, Y) Photo pixel dimensions in mm
    bregma_photo_pixels : tuple
        (X, Y) Bregma location in photo pixel coordinates

    Returns
    -------
    pd.DataFrame
        Updated dataframe with recalculated X-MRI, Y-MRI, Z-MRI columns
    """

    df_out = df.copy()

    # Unpack parameters
    bregma_x_mri, bregma_y_mri, bregma_z_mri = bregma_mri
    mri_voxel_x, mri_voxel_y, mri_voxel_z = mri_voxel_sizes
    pixel_x, pixel_y = photo_pixel_sizes
    bregma_pixel_x, bregma_pixel_y = bregma_photo_pixels

    print(f"\nRecalculating electrode MRI coordinates...")
    print(f"  Bregma MRI (voxels): ({bregma_x_mri}, {bregma_y_mri}, {bregma_z_mri})")
    print(f"  MRI voxel sizes (mm): X={mri_voxel_x:.6f}, Y={mri_voxel_y:.6f}, Z={mri_voxel_z:.3f}")
    print(f"  Photo pixel sizes (mm): X={pixel_x:.5f}, Y={pixel_y:.5f}")
    print(f"  Bregma photo pixels: ({bregma_pixel_x}, {bregma_pixel_y})")

    # Process each row (skip Bregma row itself)
    n_updated = 0
    for idx, row in df_out.iterrows():
        if row['Label'] == 'Bregma':
            # Update Bregma row
            df_out.at[idx, 'X-MRI'] = bregma_x_mri
            df_out.at[idx, 'Y-MRI'] = bregma_y_mri
            df_out.at[idx, 'Z-MRI'] = bregma_z_mri
            print(f"\n  {row['Label']}: Updated to MRI coords ({bregma_x_mri}, {bregma_y_mri}, {bregma_z_mri})")
            n_updated += 1
            continue

        # For electrode rows:
        # Use existing Xbregmm, Ybregmm (mm offset from Bregma in photo space)
        x_offset_mm = row['Xbregmm']
        y_offset_mm = row['Ybregmm']

        # Convert mm offset to MRI voxel offset
        x_offset_voxels = x_offset_mm / mri_voxel_x
        y_offset_voxels = y_offset_mm / mri_voxel_y

        # Calculate new MRI coordinates
        x_mri_new = bregma_x_mri + x_offset_voxels
        y_mri_new = bregma_y_mri + y_offset_voxels
        z_mri_new = bregma_z_mri  # All electrodes at same depth

        # Store old values for comparison
        x_mri_old = row['X-MRI']
        y_mri_old = row['Y-MRI']
        z_mri_old = row['Z-MRI']

        # Update dataframe
        df_out.at[idx, 'X-MRI'] = x_mri_new
        df_out.at[idx, 'Y-MRI'] = y_mri_new
        df_out.at[idx, 'Z-MRI'] = z_mri_new

        # Print update for first few electrodes
        if n_updated < 5:
            print(f"  {row['Label']}: "
                  f"Old=({x_mri_old:.2f}, {y_mri_old:.2f}, {z_mri_old:.0f}) → "
                  f"New=({x_mri_new:.2f}, {y_mri_new:.2f}, {z_mri_new:.0f}) "
                  f"[Δ=({x_mri_new-x_mri_old:.2f}, {y_mri_new-y_mri_old:.2f}, {z_mri_new-z_mri_old:.0f})]")

        n_updated += 1

    print(f"\n  Total rows updated: {n_updated}")

    # Calculate summary statistics
    delta_x = df_out['X-MRI'] - df['X-MRI']
    delta_y = df_out['Y-MRI'] - df['Y-MRI']
    delta_z = df_out['Z-MRI'] - df['Z-MRI']

    # Exclude Bregma from delta calculations
    electrode_mask = df_out['Label'] != 'Bregma'

    print(f"\n  Electrode coordinate changes:")
    print(f"    ΔX: mean={delta_x[electrode_mask].mean():.3f}, std={delta_x[electrode_mask].std():.3f}")
    print(f"    ΔY: mean={delta_y[electrode_mask].mean():.3f}, std={delta_y[electrode_mask].std():.3f}")
    print(f"    ΔZ: mean={delta_z[electrode_mask].mean():.3f}, std={delta_z[electrode_mask].std():.3f}")

    return df_out


def main():
    parser = argparse.ArgumentParser(
        description='Recalculate electrode MRI coordinates based on updated Bregma position',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update coordinates with new Bregma position
  python recalculate_electrode_coords.py \\
      --input inputs/mouse_array_coords.csv \\
      --output inputs/mouse_array_coords_UPDATED.csv \\
      --bregma-mri 30 149 41

  # Use custom MRI voxel sizes
  python recalculate_electrode_coords.py \\
      --input inputs/mouse_array_coords.csv \\
      --output inputs/mouse_array_coords_UPDATED.csv \\
      --bregma-mri 30 149 41 \\
      --mri-voxel-sizes 0.203125 0.080078125 0.200
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input CSV file with electrode coordinates'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output CSV file with updated coordinates'
    )

    parser.add_argument(
        '--bregma-mri',
        type=float,
        nargs=3,
        required=True,
        metavar=('X', 'Y', 'Z'),
        help='Bregma coordinates in MRI voxel space (0-indexed)'
    )

    parser.add_argument(
        '--mri-voxel-sizes',
        type=float,
        nargs=3,
        default=(0.203125, 0.080078125, 0.200),
        metavar=('X', 'Y', 'Z'),
        help='MRI voxel dimensions in mm (default: 0.203125 0.080078125 0.200)'
    )

    parser.add_argument(
        '--photo-pixel-sizes',
        type=float,
        nargs=2,
        default=(0.01385, 0.01564),
        metavar=('X', 'Y'),
        help='Photo pixel dimensions in mm (default: 0.01385 0.01564)'
    )

    parser.add_argument(
        '--bregma-photo-pixels',
        type=int,
        nargs=2,
        default=(336, 182),
        metavar=('X', 'Y'),
        help='Bregma location in photo pixel coordinates (default: 336 182)'
    )

    args = parser.parse_args()

    # Validate input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Load input CSV
    print(f"\nLoading: {input_path}")
    df = pd.read_csv(input_path, encoding='utf-8-sig')  # Handle BOM if present
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    # Verify required columns exist
    required_cols = ['Label', 'Xbregmm', 'Ybregmm', 'X-MRI', 'Y-MRI', 'Z-MRI']
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Recalculate coordinates
    df_updated = recalculate_mri_coords(
        df,
        bregma_mri=tuple(args.bregma_mri),
        mri_voxel_sizes=tuple(args.mri_voxel_sizes),
        photo_pixel_sizes=tuple(args.photo_pixel_sizes),
        bregma_photo_pixels=tuple(args.bregma_photo_pixels)
    )

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_updated.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")
    print(f"  Rows: {len(df_updated)}")

    # Create backup of original if overwriting
    if input_path == output_path:
        backup_path = input_path.with_suffix('.csv.bak')
        print(f"  Original backed up to: {backup_path}")


if __name__ == '__main__':
    main()
