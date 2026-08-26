# Monte Carlo source sampling

A source space is a discrete approximation of a continuous current distribution,
and the grid you deploy is one arbitrary placement of it. Monte Carlo sampling
integrates over placement instead of picking one: it draws *K* sparse source
configurations from a dense pool and averages the ROI operators they induce.

**Off by default.** `source_sampling: fixed` is the shipped behaviour and this
page explains both what the alternative does and — importantly — what it was
measured to be good for, which is narrower than it first appears.

## Why placement is arbitrary

The Cartesian grid is built as `np.arange(0, shape[i], spacing_voxels[i])`. Its
phase is anchored to voxel 0 of the volume array — the corner of a bounding box,
with no anatomical meaning. Change the array padding and every source moves.

That matters because which small parcels get sampled depends on it. Sweeping the
offset across a single grid cell on the MEA30 montage:

| | |
|---|---|
| parcels covered, per offset | **29–32** |
| `Auditory_L` sampled | 9 of 10 offsets |
| `Auditory_R` sampled | 9 of 10 offsets — **not the same nine** |
| union over offsets | all 32 |

So a parcel missing from a Cartesian run can be a coin flip that the code always
flips the same way.

## Configuration

```yaml
source_space:
  source_sampling: monte_carlo     # default: fixed
  monte_carlo:
    n_sources: 160                 # sources per draw
    n_draws: 100                   # K
    seed: 20260821
    pool_spacing_mm: 0.20          # surface and cartesian pool density
    pool_n_shells: 24              # shell pool density
    pool_points_per_shell: 1200
```

When enabled, `monte_carlo_roi` replaces `inverse_solution` and `roi_extraction`.
The source space becomes the **pool** draws are taken from rather than the
deployed grid, so each source-space module builds itself dense under this flag;
deployed density becomes a property of the draw.

Every source-space type needs its own pool parameter. Wiring only one leaves the
others drawing from a pool barely larger than a draw — `shell` drew 160 from a
pool of 161 (no gain at all) and `cartesian` failed outright drawing 160 from
150 until their pool settings existed.

`roi_based` is deliberately unsupported. Its sources sit at ROI centroids by PCA
placement, which is the point of that source space; there is no dense pool to
draw from without making it a different method.

## Why it costs almost nothing

The parcel time series is **linear** in the sensor data:

```
y_p = mean_{i in p} (W_norm[i, :] · B) = (mean_{i in p} W_norm[i, :]) · B
```

So averaging ROI time series over K draws is algebraically identical to
averaging the K operators and applying the result once. A draw contributes an
`(n_parcels × n_channels)` matrix; K of them collapse into one operator of a few
kilobytes. Nothing per-draw is materialised, **no vertex-level estimate is ever
built**, and the operator depends only on `(montage, BEM, source pool)` — not on
the recording — so it is computed once and reused for every subject.

| | fixed | monte_carlo |
|---|---|---|
| per-subject output | ~7 MB ROI + 0.5–0.9 GB vertex STC | **~8 MB, ROI only** |
| peak RAM | full vertex estimate | one `30 × 160` matrix |
| 476-run four-arm study | ~278 GB | **~8 GB** |

`tests/test_realizations.py` locks the linearity equivalence. If that test ever
fails, per-draw outputs would have to be materialised and this design collapses.

## ROI-only, by construction

No single grid is ever solved, so there is no vertex set for a vertex-level
analysis to refer to. Combining `monte_carlo` with `skip_roi_extraction` raises
rather than producing something meaningless.

This is worth stating plainly rather than treating as a gap: a vertex-level
result was always a statement about **one arbitrary discretisation**. The
supporting evidence was already there — density buys parcel occupancy rather
than resolution (excess error flat at 0.52–0.67 mm across a 20× range of source
counts), effective rank is ~16 with 1,310 sources, and parcel leakage is flat at
29–30% whether you use 356 sources or 3,480. Monte Carlo does not create that
problem; it makes it impossible to ignore.

## What it was measured to do

Parameters were chosen by sweep, scored against a dense 0.20 mm pool standing in
for continuous truth. At 160 sources and K=100: **median SNR gain 1.14×**,
auditory ~1.20×. K converges by 100 — K=200 and K=400 are identical — and 80
sources per draw is past a cliff where parcels start dropping out of draws.

The mechanism is the triangle inequality: sparse draws steer the operator row in
different directions, so their average shrinks in noise norm while the signal
term, which integrates the same anatomy, does not.

## What it was measured **not** to do

**It does not improve region identification, and slightly degrades it.** Tested
on 119 recordings × 3 paradigms, against a fixed-grid baseline at constant
chance level:

| | separation from chance | top-1 | top-3 | d_z |
|---|---:|---:|---:|---:|
| fixed | **3.7 ranks** | 42% | 73% | +1.10 |
| monte_carlo | 2.9 ranks | 31% | **77%** | +0.96 |

The SNR gain is real, but **SNR against sensor noise is the wrong currency for a
discrimination task**. What a region-identification statistic depends on is the
*contrast between parcels*, and averaging compresses it:

| parcel | fixed self-fraction | monte carlo | change |
|---|---:|---:|---:|
| Somatosensory_L | 62.4% | 55.9% | **−6.5%** |
| Visual_Parietal_L | 63.0% | 56.8% | −6.2% |
| Motor_L | 27.3% | 34.7% | **+7.4%** |
| Auditory_L | 4.0% | 6.5% | +2.5% |

Regression toward the mean: well-resolved parcels lose, poorly-resolved parcels
gain. That also explains a dissociation seen in every measure — top-1 falls
while top-3 rises. The target lands *near* the top more often and *at* the top
less often, because the spread between parcels is squeezed.

## When to reach for it

Compression that lifts poorly-resolved parcels and trims well-resolved ones is
plausibly what a **network or whole-brain analysis** wants, where stability
across many parcels matters more than contrast between two of them. That use is
untested and nothing measured so far argues against it.

For "which region does this response come from", use `fixed`. The measurements
above are why.
