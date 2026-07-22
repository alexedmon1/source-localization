# Adding an atlas to the verification benchmark

Verification is **atlas-agnostic**: a run is defined by one *matched bundle*, and adding a
new atlas is a data task, not a code change.

## The bundle

A bundle is three files that share **one voxel grid** and **one coordinate frame**:

| file | role |
|------|------|
| `template.nii[.gz]` | anatomical volume — defines geometry and the working affine convention |
| `labels.nii[.gz]`   | integer ROI parcellation, **same voxel grid** as the template |
| `roi_mapping.json`  | `{"rois": {"<id>": {"name": ...}}}` — every label id has a name |

### The one rule

The template and labels must be in the **same affine convention**. This pipeline stores
atlas volumes in a *10× convention* (voxel sizes > 1 mm; the ~1 cm mouse brain is scaled up
so FSL/ANTs behave well), and the code divides back to true mm via `corrected_atlas_affine`.
If your `labels` are exported already in true mm while the `template` is 10×, they no longer
share a frame — and `load_bundle` will reject them. (This exact mismatch silently collapsed
ROI accuracy from ~46 % to ~5 % in July 2026; it is now a hard error.)

## Register and verify

```python
from source_localization.validation import load_bundle

bundle = load_bundle(
    template="data/atlas/Atlas_3DRois.nii",
    labels="data/atlas/my_atlas/labels.nii.gz",
    roi_mapping="data/atlas/my_atlas/roi_mapping.json",
    name="MyAtlas",
)
print(bundle.summary())
```

`load_bundle` asserts, up front and loudly:

1. template and labels share one voxel grid (same shape),
2. they resolve to the same coordinate frame after convention correction,
3. every non-zero label has a `roi_mapping` entry (and reports orphan names),
4. the ROI centroids span a plausible brain (not collapsed to the centre).

Any violation raises `BundleConsistencyError` with a message naming the offending file —
you find out at load time, not from a silently wrong accuracy number.

## Then run the benchmark

Point the validation run at the bundle's `labels`/`roi_mapping` for ROI-classification
accuracy, and at the `template` for the uniform test grid:

```bash
uv run --no-sync python -m source_localization.validation \
  --test-dir <dir> --config config/ --all \
  --test-mode roi_centroids --trials 25 --snr 10   # ROI accuracy on your atlas
```

The bundled Allen-32 atlas (`data/atlas/Atlas_3DRois.nii` + `allen/allen_labels.nii.gz` +
`allen/roi_mapping.json`) is the reference example and is covered by
`tests/test_atlas_bundle.py`.
