# Single-source localization validation — methods supplement

> **Provenance.** Methods supplement moved into the repository from the external staging area on 2026-10-09, text
> unchanged except paths. Produced July 2026 (v0.4-0.5) with `DipolePosterior` under its stated assumptions: an
> exact forward model and white noise of known size. These are the **fundamental limits under a correct model**,
> the same assumptions the v0.6.0 **legacy** regime makes (see `source_localization.validation.regime`). They
> include no truth head-model perturbation, no recorded backgrounds and no noise-only control. The noise-only
> controls now written by `run_posterior_orbital.py` and `run_roi_certainty.py` are in the
> [validation guide](../README.md), section 6. Reproduction commands use `<pipeline_dir>` (an ellipsoid-BEM shell
> run, see the guide, step 2) and `<output_dir>`.

**Scope.** 30-channel mouse scalp EEG; three-layer ellipsoid boundary-element
(BEM) head model; Antwerp C57BL/6 atlas geometry. All quantities characterize
this array and geometry; electrode count is the dominant lever on every number.

This document quantifies how precisely a **single** source can be localized, and
establishes that the reported credible regions mean what they say (calibration).
Its companions are `TWO_SOURCE_VALIDATION.md` (whether two simultaneous sources
can be detected and told apart) and `ROI_CERTAINTY_VALIDATION.md` (how much to
believe an atlas-parcel attribution, which reuses this posterior and inherits its
calibration). The reasoning history, including approaches that were tried and
retracted, is in `SOURCE_CERTAINTY.md`.

---

## 1. Forward model and geometry

| Component | Specification |
|---|---|
| Sensors | 30 scalp electrodes (NeuroNexus array), dorsal |
| Head model | 3-layer ellipsoid BEM |
| Conductivities (brain, skull, scalp) | 0.33, 0.0042, 0.33 S/m |
| Inner BEM surface | brain ellipsoid inflated by `ellipsoid_margin` = 1.23 |
| Source constraint | atlas brain mask, `inside_frac` = 0.93 of the boundary |
| Brain volume (source space) | 531 mm³ |
| Source-space extent (L-R × A-P × D-V) | ≈ 10 × 14 × 6 mm |

The lead-field operator is rebuilt densely from the cached BEM of a completed
pipeline run, **independent of the pipeline's own 215-point source lattice**, so
the validation characterizes the geometry rather than one particular discretized
inverse. The single-source analysis uses a dense **0.5 mm grid (4,251
positions)**. At each position the 30×3 lead field is reduced by SVD to an
orthonormal column basis `Q_x` (three columns) with singular values `S_x`.

---

## 2. Posterior model

For sensor data `B` (30 channels) and a dipole at position `x` with moment
integrated out under an isotropic Gaussian prior (moment scale `τ`, noise
variance `σ²`), the location posterior is

```
p(x | B) ∝ exp( −½ r(x) / σ² ) · |A_x|^(−T/2),   A_x = σ²I + τ² S_x²
```

where `r(x) = ‖B − Q_x Q_xᵀ B‖²` is the residual of the best-fitting dipole at
`x` and `T` is the number of time samples. The log-determinant term is the Occam
penalty from marginalizing the moment. The result is a normalized probability
density over the source space, so its **credible regions can be tested against
ground truth** (§3).

The moment marginalization is essential. Profiling the moment out instead
(plugging in its maximum-likelihood value) yields an over-confident posterior
whose credible regions under-cover once noise approaches signal — see §3.

---

## 3. Calibration of credible regions

A p% credible region should contain the true source p% of the time. This is the
central property that distinguishes a calibrated posterior from an arbitrary
display threshold, and it is testable.

**Procedure.** For each SNR, 200 dipoles are simulated at continuous, off-grid
positions and random orientations, with white sensor noise at the target SNR.
For each nominal credible level p, the fraction of trials in which the true
source fell inside the p% credible region is recorded (empirical coverage).
Sampling error ≈ ±3 percentage points per cell.

**Result (marginal-moment posterior).** Empirical coverage matches the nominal
level, erring slightly conservative — regions marginally larger than strictly
necessary, which is the safe direction:

| Nominal level | +20 dB | +10 dB | +5 dB | 0 dB | −5 dB |
|---|---|---|---|---|---|
| 50% | 70% | 57% | 50% | 49% | 50% |
| 68% | 83% | 74% | 72% | 73% | 68% |
| 90% | 96% | 94% | 92% | 94% | 90% |
| 95% | 98% | 97% | 96% | 96% | 96% |

Under-coverage (empirical below nominal) would be the dangerous direction: it
would mean the stated confidence is a lie. It is not observed. For comparison,
the **profile-moment** posterior does under-cover — asking for 68% it delivers
only 53% at 0 dB — which is why the marginal posterior is used throughout.

*Figure:* `posterior_calibration.png` (coverage vs nominal level, marginal vs
profile, across SNR).

---

## 4. Localization precision

Median credible-region size, pooled over random position and orientation
(source space = 531 mm³):

| SNR | 50% region | 95% region | 95% equiv. radius | 95% region, % of brain |
|---|---|---|---|---|
| +20 dB | 0.2 mm³ | 0.9 mm³ | 0.59 mm | 0.2% |
| +10 dB | 7.3 mm³ | 97.2 mm³ | 2.85 mm | 18.3% |
| +5 dB | 26.3 mm³ | 152.4 mm³ | 3.31 mm | 28.7% |
| 0 dB | 52.1 mm³ | 250.8 mm³ | 3.91 mm | 47.2% |
| −5 dB | 99.4 mm³ | 352.4 mm³ | 4.38 mm | 66.3% |

The "% of brain" column is a volume ratio (region volume ÷ 531 mm³), not a
confidence — the confidence is fixed at 95% and the volume is what varies. A
rough operational reading:

- **< 1%** — sub-structure resolvable; a claim about a specific small ROI is
  supportable.
- **~10–30%** — coarse regional (lobe-scale) claims only.
- **> 40%** — the honest statement is "somewhere in roughly this half of the
  brain"; no ROI-level claim survives.

This reads ROI-level supportability off the region *volume*;
`ROI_CERTAINTY_VALIDATION.md` measures it directly, as the probability an atlas
parcel attribution is correct.

**Sub-structure is resolvable only at +20 dB.** At 0 dB and below the 95% region
spans roughly half the brain or more.

**Depth dominates the pooled numbers.** Gain falls steeply with depth, so a deep
source is far less constrained than a shallow one at the same SNR. At 0 dB a
dorsal source has a 95% region of ~6 mm³ (0.8% of the brain) versus ~237 mm³
(31.5%) for a ventral one. For deep sources the region is crescent-shaped rather
than a compact ball, so the "equivalent radius" column understates the ambiguity
and the volume is the number to trust.

*Figure:* `posterior_orbital.png` (p(location | data) slices, 50% and 95%
regions, truth marked).

**Where in the brain the deployed estimator is accurate.** The credible-region
sizes above are the *fundamental limit* — what any estimator could achieve under
the model. The pipeline's actual estimator (sLORETA) is separately characterized
by its peak localization error as a function of source depth (distance to the
nearest electrode), from single-dipole simulation:

| depth (mm) | 1.5 | 2.5 | 3.5 | 4.5 | 5.5 |
|---|---|---|---|---|---|
| sLORETA localization error (mm) | 0.46 | 0.74 | 1.71 | 2.67 | 4.31 |

Rendered over the brain — mapping that curve onto each source by its distance to
the nearest electrode — the error is a smooth gradient: sub-millimetre directly
beneath the dorsal electrode array and rising to several millimetres in deep and
ventral tissue. This is the same depth axis that governs two-source resolvability
(`TWO_SOURCE_VALIDATION.md` §4.2), and it makes concrete why depth, not noise, is
the binding constraint: the accuracy of a localization claim is set by how close
the source sits to the array.

*Figure:* `localization_error_brain_overlay.png` (dorsal, sagittal, coronal;
error heatmap over the skull-stripped brain, electrodes as white markers).

---

## 5. Reproducibility

From the repo root with the virtual environment active
(`source_localization` 0.4.0):

```bash
P=<pipeline_dir>
O=<output_dir>

python scripts/run_posterior_orbital.py --pipeline-dir $P --output-dir $O/posterior

# localization-error brain overlay (deployed sLORETA estimator; §4 figure)
python scripts/localization_error_brain_overlay.py \
    --pipeline $P --output $O/docs/localization_error_brain_overlay.png
```

The dense lead field is rebuilt from the cached BEM in seconds. The posterior
driver caches its operator and per-figure data under `<output>/cache/`; rerunning
with `--replot` redraws from cache with no forward solve or simulation. Cache
validity is keyed on the pipeline directory, grid spacing, `inside_frac`, and the
BEM file's size and modification time, so a rebuilt head model invalidates it
automatically.

The brain-overlay script reads the pipeline's source coordinates
(`data/step3_source_coords_mm.npy`) and electrode montage (`data/step1_info.pkl`)
and renders the validated sLORETA depth→error curve (the table in §4) over the
bundled skull-stripped atlas; `--alpha` controls overlay opacity.

---

## 6. Assumptions and limitations

1. **Fundamental limit vs deployed pipeline.** The posterior assumes a correct
   forward model and known noise; it bounds what any estimator could achieve. The
   pipeline's sLORETA output is worse, and that gap is now characterized, not
   outstanding: as peak localization error by depth in §4 above, and as
   ROI-attribution accuracy in `ROI_CERTAINTY_VALIDATION.md`.
2. **White noise — checked separately, in the evoked regime.** Calibration (§3)
   is verified under white noise. Whether it survives realistic
   spatially-correlated / 1/f noise, and whether whitening by the estimated noise
   covariance restores it, is the **noise-model robustness** dimension, scoped to
   **evoked** responses — a time-locked source with a separable baseline
   covariance and trial-averaged SNR, the only regime in which whitening is
   well-posed. It makes no claim about resting-state use, which has no
   signal/noise partition and is neither validated nor invalidated by it.
3. **SNR is not measurable in resting-state recordings.** There is no
   signal/noise partition and no time-locked component to average up
   (evoked/single-trial RMS matches 1/√N), so the SNR axis here is a property of
   the simulation, not of the target recordings. A single dipole explains only
   0.37–0.48 of sensor variance in the available recordings. The defensible
   reformulation — how strong an added source must be relative to ongoing
   activity to be localized to within X mm — is proposed in
   `SOURCE_CERTAINTY.md` §4.2 and not yet built.
4. **Matched moment prior.** The moment prior scale is matched to the simulated
   amplitude; real data would estimate it.
5. **Uniform spatial prior.** Deep sources come out broad because the data are
   genuinely uninformative there, not because of the prior.
6. **Single-dipole model.** Distributed or extended sources, and the
   two-simultaneous-source case, are out of scope here; the latter is covered in
   `TWO_SOURCE_VALIDATION.md`.
