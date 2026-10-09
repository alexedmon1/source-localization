# ROI-identification certainty validation — methods supplement

> **Provenance.** Methods supplement moved into the repository from the external staging area on 2026-10-09, text
> unchanged except paths. Produced July 2026 (v0.4-0.5) with `DipolePosterior` under its stated assumptions: an
> exact forward model and white noise of known size. These are the **fundamental limits under a correct model**,
> the same assumptions the v0.6.0 **legacy** regime makes (see `source_localization.validation.regime`). They
> include no truth head-model perturbation, no recorded backgrounds and no noise-only control. The noise-only
> controls now written by `run_posterior_orbital.py` and `run_roi_certainty.py` are in the
> [validation guide](../README.md), section 6. Reproduction commands use `<pipeline_dir>` (an ellipsoid-BEM shell
> run, see the guide, step 2) and `<output_dir>`.

**Scope.** 30-channel mouse scalp EEG; three-layer ellipsoid boundary-element
(BEM) head model; Antwerp C57BL/6 source geometry. Parcels are the **allen32**
labels (16 per hemisphere) conformed into that Antwerp space — the geometry is a
fixed property of the source space and the atlas is a label lookup on top of it.
All quantities characterize this array and geometry; electrode count is the
dominant lever on every number.

This document quantifies the practical question the pipeline's ROI output
raises: **when activity is attributed to an atlas parcel, how much should you
believe it?** It is the companion to `SINGLE_SOURCE_VALIDATION.md` (how precisely
one source is located, and that the credible regions are calibrated) and
`TWO_SOURCE_VALIDATION.md` (whether two sources can be told apart). It reuses the
single-source posterior and inherits its calibration, so the numbers below rest
on that foundation. The reasoning history is in `SOURCE_CERTAINTY.md`.

---

## 1. Forward model and geometry

The forward model, BEM, montage, and dense **0.5 mm grid (4,251 positions,
531 mm³)** are identical to `SINGLE_SOURCE_VALIDATION.md` §1 — the lead field is
rebuilt densely from the cached BEM, independent of the pipeline's own lattice.
Two source spaces appear here, one per arm (§2):

| | grid | role |
|---|---|---|
| dense 0.5 mm, 4,251 positions | the posterior's own grid | **ceiling** arm |
| 215-point shell (pipeline source space) | the deployed inverse's grid | **deployed** arm |

The parcellation is allen32: 32 lateralized parcels, `1–16 = L`, `17–32 = R`,
tiling grey matter. It is preferred here over Antwerp because it labels ~66 % of
the brain vs ~31 %, and its worst-case nearest-label distance is 0.98 mm vs
2.56 mm — so the nearest-label assignment below is a sub-voxel nudge, not a drag
across unlabeled tissue.

---

## 2. From location to parcel: two arms

The posterior of `SINGLE_SOURCE_VALIDATION.md` is a probability density over
location. Integrating it over a parcel gives a probability **mass** per parcel,

```
P(parcel k | B) = Σ_{x ∈ k}  p(x | B)
```

which is mass-conserving (sums to 1 over parcels). Every grid position is
assigned to a parcel by **nearest labeled voxel**, exactly as
`steps/roi_extraction.py` does, so the numbers describe the deployed convention.
Simulated truth is drawn only from positions **strictly inside** a label, so "the
true parcel" is always defined.

Two arms are evaluated on the **same** simulated sensor data, so the only
difference between them is the estimator:

- **ceiling** — the fundamental limit. The calibrated posterior on the dense
  grid; its headline output is not an accuracy but a probability,
  `P(parcel k | B)`, and its point attribution is that vector's argmax.
- **deployed** — what a user actually runs. sLORETA on the 215-point shell; the
  attributed parcel is the one holding the reconstruction's peak. A point
  estimate only — no probability, so no reliability curve. The sLORETA
  regularization uses the pipeline's fixed default (SNR = 3), independent of the
  true data SNR, because a user cannot know it.

**The gap between the two arms is the total cost of deployment, not an "estimator
gap."** They differ in three ways at once: source space (the ceiling's grid
always contains the true location as a node; the 215-point shell almost never
does), attribution rule (parcel-integrated argmax vs single peak), and operator
(posterior vs sLORETA — the smallest of the three). Read the gap as "fundamental
limit vs deployed pipeline," not "MNE vs sLORETA."

---

## 3. Calibration of the parcel probabilities

This is **not a new calibration claim.** It is the single-source posterior
calibration (`SINGLE_SOURCE_VALIDATION.md` §3) re-expressed on the atlas
partition. That section certifies the posterior by its **credible regions** (a
90 % region contains the truth 90 % of the time); this section certifies the
*same* posterior by its **parcel probabilities**. Both follow from `p(x | B)`
being an honest posterior, but the parcel version is not redundant: credible
regions are highest-density sets growing from the mode, whereas parcels are a
fixed anatomical partition, so region coverage does not by itself certify that
"`P(k) = 0.3`" is honest. This section confirms the calibration **survives
projection onto the parcels** — the only calibration statement that underwrites
the ROI product.

**Procedure (proper multiclass reliability).** Over *every* `(trial, parcel)`
pair, bin by the predicted `P(parcel | B)` and record how often that parcel
really is the true one. On the diagonal, "the pipeline reported `P(k) = p`" means
"the source is in `k` with probability `p`." (Binning the *true* parcel's mass
against the argmax-win rate mixes two different events — a parcel holding 0.3 of
the mass wins far more than 30 % of the time when the rest is fragmented — and is
**not** a calibration test.)

**Result.** The empirical curve tracks the diagonal at every SNR: when the
posterior assigns 0.4 to a parcel, the source is there ~43 %. It carries the same
mild conservatism the credible-region coverage showed (slightly under-confident
at +20 dB) — the same posterior, the same signature, a second lens. The parcel
probabilities are therefore usable as stated: a calibrated "believe it X %."

*Figure:* `roi_reliability.png` (predicted vs empirical, per SNR, with per-bin
counts).

---

## 4. Identification accuracy

### 4.1 Two readouts

`P(parcel | B)` is the honest per-attribution belief; for a whole-pipeline
score we also report the argmax attribution against the truth, two ways:

- **exact-match accuracy** — attributed parcel is *exactly* the true one.
  Balanced across parcels, so it equals macro-averaged recall. Harsh and
  anatomically blind: a contralateral homologue scores the same as the far side
  of the brain.
- **distance-tolerance accuracy `A(r)`** — attributed parcel within `r` mm
  (centroid-to-centroid) of the true parcel. `A(0)` is exact match, `A(∞) = 1`.
  It grades *how anatomically wrong* a miss is. It matters because "wrong
  hemisphere" is not one error size: the L↔R homologue centroid distance ranges
  from **1.19 mm** (Frontal_Anterior, near the midline) to **8.63 mm**
  (Auditory, lateral). A tolerance forgives the first and not the second;
  collapsing hemispheres would forgive both.

*Figures:* `roi_tolerance.png` (`A(r)` curves, pooled and by depth);
`roi_confusion.png` (recall and precision, both arms, parcels depth-ordered).

### 4.2 Depth is the ceiling

Exact-match accuracy, ceiling / deployed, by depth band (distance to nearest
electrode), allen32, 30 sources per parcel:

| depth band | +20 dB | +10 dB | +0 dB |
|---|---|---|---|
| very shallow (1.2–2.3 mm) | 0.99 / 0.45 | 0.80 / 0.41 | 0.41 / 0.32 |
| shallow (2.3–3.5 mm) | 0.91 / 0.26 | 0.57 / 0.21 | 0.16 / 0.10 |
| mid (3.5–4.7 mm) | 0.67 / 0.26 | 0.42 / 0.18 | 0.18 / 0.05 |
| deep (4.7–7.6 mm) | 0.29 / 0.09 | 0.11 / 0.06 | 0.10 / 0.01 |
| **overall** | **0.75 / 0.29** | **0.51 / 0.24** | **0.25 / 0.14** |

Depth is the binding constraint, the same axis that governs single-source
precision and two-source resolvability. This is not a parcel-size artifact: at
fixed depth, larger parcels are modestly easier (partial correlation of recall
with log-volume, depth held fixed, ≈ +0.49), but marginally parcel size barely
predicts accuracy (correlation ≈ +0.10) because allen32's large parcels happen
to be the deep ones. Recall correlates with depth at −0.82. Where a parcel fails,
its posterior mass is genuinely low (deep parcels ~0.05, not a discarded "real"
probability) — the failure is absent evidence, not a winner-take-all artifact.

*Figure:* `roi_belief_by_parcel.png` (per-parcel `P(true | B)` and recall, sorted
shallow → deep).

### 4.3 Two deployed-arm behaviours worth naming

- **Nearly flat vs SNR.** The deployed accuracy barely moves from 0 to 20 dB,
  because sLORETA uses the pipeline's fixed regularization (SNR = 3) regardless
  of the true data SNR — the honest deployed behaviour. The ceiling, which uses
  the correct noise level, rises steeply. Most of the widening gap at high SNR is
  the deployed arm leaving resolvable information on the table.
- **An attractor column.** The deployed confusion matrix concentrates
  mis-attributions into a few parcels (Lateral_Cortex, Cerebellum): the sLORETA
  peak on the shell has preferred basins. A distance tolerance of a few mm
  absorbs much of this (`A(r)` rises fast), but exact-match does not.

---

## 5. Reproducibility

From the repo root with the virtual environment active
(`source_localization` 0.4.0):

```bash
P=<pipeline_dir>
O=<output_dir>

python scripts/run_roi_certainty.py --pipeline-dir $P \
    --output-dir $O/roi_certainty --n-per-parcel 30 --headline-snr 10

# redraw all figures from cache, no forward computation:
python scripts/run_roi_certainty.py --pipeline-dir $P \
    --output-dir $O/roi_certainty --replot
```

The driver runs one pass per `(SNR, trial)`: it simulates once and evaluates both
arms on that same data, computing the posterior once per trial (feeding the
argmax, the parcel mass, and the reliability diagram together). It caches the
operator and per-trial arrays under `<output>/cache/`; `--replot` rebuilds every
figure from the cache. `--n-per-parcel` sets the trials per parcel (balanced
sampling keeps small parcels from vanishing); `--headline-snr` picks the SNR for
the single-condition figures (confusion, tolerance, per-parcel belief).

---

## 6. Assumptions and limitations

1. **Fundamental limit vs deployed pipeline.** The ceiling arm assumes a correct
   forward model and known noise; it bounds what any estimator could achieve. The
   deployed sLORETA arm is characterized alongside it, so the gap is quantified
   here (identification) and in `SINGLE_SOURCE_VALIDATION.md` §4 (position
   error), not left outstanding.
2. **White noise — validated separately, in the evoked regime.** The inherited
   calibration (§3) holds under white noise. Whether it survives realistic
   spatially-correlated / 1/f noise, and whether whitening by the estimated noise
   covariance restores it, is the **noise-model robustness** dimension — scoped
   to **evoked** responses (a time-locked source with a separable baseline
   covariance and trial-averaged SNR), the only regime in which whitening is
   well-posed. That dimension makes **no claim about resting-state use**:
   resting-state output (ROI time series for connectivity / relative analyses)
   has no signal/noise partition, is not single-transient localization, and is
   neither validated nor invalidated by it.
3. **SNR is a simulation axis**, not a measured property of the target
   recordings; see `SOURCE_CERTAINTY.md` §4.
4. **Nearest-label attribution and uniform spatial prior.** Truth is drawn only
   from labeled tissue; precision (`P(true = k | attributed k)`) depends on the
   uniform-over-labeled-tissue prior. allen32 makes the nearest-label step
   sub-voxel; under a sparser atlas (Antwerp) it is a larger reach and the
   numbers would differ.
5. **Single-dipole model.** Distributed or extended sources are out of scope; the
   two-source case is in `TWO_SOURCE_VALIDATION.md`.
