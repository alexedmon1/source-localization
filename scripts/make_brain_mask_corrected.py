"""Build a source-placement brain mask with the non-brain surface rim removed.

Why this exists
---------------
`Atlas_3DRois_brain.nii.gz` measures **563.7 mm³**. Published adult C57BL/6
whole-brain volume is ~450–520 mm³, so the shipped mask is over-inclusive. The
excess is a thin surface rim — meninges, CSF and partial-volume edge — and it is
essentially all unlabeled by Allen-32:

    unlabeled voxels: median depth 0.37 mm inside the mask
    labeled   voxels: median depth 1.15 mm

That rim matters because the **shell** source space places its outermost shell at
0.95 of the brain radius: 14.4% of shell sources sit within 0.4 mm of the mask
surface, and 52.8% of its unlabeled sources are there. Those sources are being
placed in tissue that is probably not brain. The Cartesian grid has the same
exposure at 8.8%; ROI-based has none, because it places sources inside atlas
parcels and never consults this mask.

Method, and why not plain erosion
---------------------------------
A uniform morphological erosion would clip thin structures preferentially, and in
a mouse brain the thin structures are exactly the ones we care about keeping —
olfactory bulb, cerebellar edges.

Instead we remove **only unlabeled voxels shallower than a threshold**, and never
remove a labeled voxel at any depth. Every Allen-32 parcel is therefore protected
by construction, including the thin ones.

This is sound because Allen-32 labels reach the pial surface: the minimum labeled
depth is 0.08 mm — a single voxel — and the 1st percentile is 0.30 mm. So the
unlabeled rim is not unlabeled cortex sitting outside the parcellation; it is
tissue beyond where the atlas says brain is.

Threshold
---------
t = 0.4 mm, chosen because:

  t (mm)   volume     unlabeled   connected components
    0.0    563.7 mm³    34.3%      1     (shipped; above literature range)
    0.3    474.6 mm³    22.0%      1
    0.4    463.4 mm³    20.1%      1     <-- selected
    0.5    443.8 mm³    16.5%      1     (below literature range)
    0.6    438.4 mm³    15.5%      3     (fragmenting)
    0.8    427.1 mm³    13.3%      9

0.4 mm lands inside the published 450–520 mm³ range, keeps the mask a single
connected component, and matches the measured median depth of unlabeled voxels
(0.37 mm). Past 0.5 mm the mask begins to fragment, which is the signal that
real tissue is being removed.

Scope — read this before using the output
-----------------------------------------
This mask is for **source placement only**. The ellipsoid BEM is fitted to the
original mask (`bem/ellipsoid.py`), and changing that would alter the forward
model and therefore every result the pipeline has ever produced. The two uses
have different requirements: the BEM needs the outer boundary of the conducting
compartment, where a slightly generous surface is conservative and is smoothed by
the 1.23 ellipsoid margin regardless; the source space needs to know where neural
sources can physically be, which should exclude meninges and CSF.

    uv run --with nibabel --with numpy --with scipy python make_brain_mask_corrected.py
"""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

from source_localization.utils.atlas import get_true_affine

ATLAS = Path(__file__).resolve().parent.parent / "src/source_localization/data/atlas"
MASK_IN = ATLAS / "Atlas_3DRois_brain.nii.gz"
LABELS = ATLAS / "allen/allen_labels.nii.gz"
MASK_OUT = ATLAS / "Atlas_3DRois_brain_srcmask.nii.gz"
REPORT = ATLAS / "Atlas_3DRois_brain_srcmask.json"

THRESHOLD_MM = 0.4
LITERATURE_MM3 = (450.0, 520.0)


def main() -> None:
    mimg = nib.load(MASK_IN)
    limg = nib.load(LABELS)
    B = mimg.get_fdata() > 0
    L = np.round(limg.get_fdata()).astype(int)
    if B.shape != L.shape:
        raise SystemExit(f"shape mismatch: mask {B.shape} vs labels {L.shape}")

    aff = get_true_affine(limg)
    vs = np.abs(np.diag(aff)[:3])
    vv = float(np.prod(vs))

    depth = ndimage.distance_transform_edt(B, sampling=vs)
    labeled = B & (L > 0)
    unlabeled = B & (L == 0)

    # The operation: drop shallow UNLABELED voxels only.
    drop = unlabeled & (depth < THRESHOLD_MM)
    M = B & ~drop

    # Guarantees we assert rather than assume.
    assert not (labeled & ~M).any(), "a labeled voxel was removed"
    n_comp = ndimage.label(M)[1]
    assert n_comp == 1, f"corrected mask is not connected ({n_comp} components)"

    vol_in, vol_out = B.sum() * vv, M.sum() * vv
    in_range = LITERATURE_MM3[0] <= vol_out <= LITERATURE_MM3[1]

    # Per-ROI volumes must be untouched, since no labeled voxel moved.
    per_roi = {}
    for rid in np.unique(L[L > 0]):
        a = int(((L == rid) & B).sum())
        b = int(((L == rid) & M).sum())
        per_roi[int(rid)] = {"before": a, "after": b, "unchanged": a == b}
    assert all(v["unchanged"] for v in per_roi.values()), "a parcel lost voxels"

    out = nib.Nifti1Image(M.astype(np.uint8), mimg.affine, mimg.header)
    out.set_data_dtype(np.uint8)
    nib.save(out, MASK_OUT)

    report = {
        "purpose": "source placement only; the BEM is still fitted to the original mask",
        "input_mask": MASK_IN.name,
        "labels": str(LABELS.relative_to(ATLAS)),
        "output_mask": MASK_OUT.name,
        "method": ("remove unlabeled voxels shallower than the threshold; "
                   "never remove a labeled voxel at any depth"),
        "threshold_mm": THRESHOLD_MM,
        "voxel_size_mm": [round(float(v), 6) for v in vs],
        "voxels_before": int(B.sum()),
        "voxels_after": int(M.sum()),
        "voxels_removed": int(drop.sum()),
        "volume_mm3_before": round(vol_in, 1),
        "volume_mm3_after": round(vol_out, 1),
        "literature_range_mm3": list(LITERATURE_MM3),
        "within_literature_range": bool(in_range),
        "unlabeled_fraction_before": round(float(unlabeled.sum() / B.sum()), 4),
        "unlabeled_fraction_after": round(float((M & (L == 0)).sum() / M.sum()), 4),
        "connected_components": int(n_comp),
        "labeled_voxels_removed": 0,
        "parcels_unchanged": f"{sum(v['unchanged'] for v in per_roi.values())}/{len(per_roi)}",
        "min_labeled_depth_mm": round(float(depth[labeled].min()), 3),
        "median_unlabeled_depth_mm": round(float(np.median(depth[unlabeled])), 3),
        "median_labeled_depth_mm": round(float(np.median(depth[labeled])), 3),
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps(report, indent=2))
    print(f"\nSaved: {MASK_OUT}")
    print(f"Saved: {REPORT}")
    if not in_range:
        print("WARNING: corrected volume is outside the literature range")


if __name__ == "__main__":
    main()
