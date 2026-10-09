# Two-source resolvability validation — methods supplement

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

This document quantifies whether **two simultaneous sources** can be detected as
two, how far apart they can be told to be, and where in the brain that is
possible. Its companions are `SINGLE_SOURCE_VALIDATION.md` (how precisely one
source can be located, and that the credible regions are calibrated) and
`ROI_CERTAINTY_VALIDATION.md` (how much to believe an atlas-parcel attribution).
The two-source posterior here inherits the same probability model and the same
white-noise assumption, so that calibration underwrites the numbers below. The
reasoning history, including approaches that were tried and retracted, is in
`SOURCE_CERTAINTY.md`.

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
inverse. The two-source analysis runs on a **1.0 mm grid (522 positions,
135,981 unordered pairs)**; the joint posterior lives on pairs of positions, so
cost is quadratic in position count and a finer grid is not affordable for the
pair scan. At each position the 30×3 lead field is reduced by SVD to an
orthonormal column basis `Q_x` (three columns) with singular values `S_x`.

---

## 2. Posterior model

### 2.1 Single dipole (recap)

For a single dipole at `x` with moment integrated out under an isotropic
Gaussian prior (moment scale `τ`, noise variance `σ²`), the location posterior is
`p(x | B) ∝ exp(−½ r(x)/σ²) · |σ²I + τ²S_x²|^(−T/2)`, where
`r(x) = ‖B − Q_x Q_xᵀ B‖²`. This normalized density is the object whose credible
regions are calibrated against ground truth in `SINGLE_SOURCE_VALIDATION.md` §3.

### 2.2 Two dipoles

The two-dipole marginal likelihood is evaluated per unordered pair of positions.
Moments are integrated out jointly under the same Gaussian prior, giving
`B | x₁,x₂ ~ N(0, σ²I + τ² K Kᵀ)` with `K = [Q_{x₁}S_{x₁}, Q_{x₂}S_{x₂}]`. The
full log-determinant is retained so the two- and one-source evidences share an
absolute scale. The pair residual is computed from 3×3 blocks
(`Q_{x₁}ᵀQ_{x₂}`, `Q_xᵀB`) without ever forming the 30×6 stacked lead field.

The reported density is the symmetrized marginal `p(a source is at x | B)`
(summing the joint over the partner position), because the two sources are
exchangeable — there is no distinguishable "source 1" cloud.

### 2.3 Model comparison

Each model's evidence is its likelihood averaged over its own uniform prior
(positions, or unordered pairs), so the ratio carries an Occam penalty for the
larger model. The log₁₀ Bayes factor (two sources vs one) is the detection
statistic in §3.

---

## 3. Detection and separation

Sources restricted to the dorsal half of the source space, depth-matched within
a pair (gain falls steeply with depth; unmatched pairs fail because the weaker
source vanishes — a gain effect rather than a resolution limit). Random
orientations, 30 trials per cell.

### 3.1 Detection

The log₁₀ Bayes factor is compared to a **matched one-source null** — the same
analysis on single-source data — with a detection called when it exceeds the
null's 90th percentile (a 10% false-alarm rate by construction). The null median
log₁₀BF is negative at every SNR (−2.17, −0.88, −0.25), confirming the comparison
correctly prefers one source when only one is present.

**P(detect two sources), 10% false-alarm floor:**

| SNR | null med. log₁₀BF | 2.0 mm | 2.9 mm | 4.2 mm | 6.0 mm | 8.0 mm |
|---|---|---|---|---|---|---|
| +20 dB | −2.17 | 97% | 100% | 100% | 100% | 100% |
| +10 dB | −0.88 | 73% | 80% | 80% | 90% | 80% |
| 0 dB | −0.25 | 27% | 40% | 50% | 70% | 53% |

SNR, not separation, is the binding constraint: detection is near-certain at
+20 dB down to 2 mm, a coin-flip-to-80% at +10 dB, and near the false-alarm floor
at 0 dB. The curves are nearly flat in separation above ~3 mm.

### 3.2 Separation estimation

Posterior median separation versus truth — a harder problem than detection:

| SNR | true 2.0 | true 2.9 | true 4.2 | true 6.0 | true 8.0 |
|---|---|---|---|---|---|
| +20 dB | 2.8 | 3.2 | 4.2 | 5.8 | 7.8 |
| +10 dB | 4.8 | 4.8 | 4.8 | 5.8 | 6.2 |
| 0 dB | 5.2 | 5.2 | 5.2 | 5.2 | 5.2 |

At +20 dB the estimate tracks truth from ~3 mm up. At +10 dB and below it
saturates near 5–6 mm regardless of the true separation — the prior showing
through. The operational consequence: at +10 dB one may be able to assert *two
sources are present* while being unable to state *how far apart*.

*Figure:* `two_source_posterior.png` (panels A–C: Bayes factor vs null,
detection rate, separation estimate).

---

## 4. Resolvability maps

**"Resolvable" is two questions, and each has its own threshold.**

- *Detection* — are there two sources, or one? — is answered by the two-vs-one
  Bayes factor `B₁₀`, reported as the posterior probability
  `P(two | B) = B₁₀ / (1 + B₁₀)` under equal prior odds.
- *Resolution* — are the two at distinguishable locations, or one merged blob? —
  is answered by whether the two per-source 95% credible regions are **disjoint**.

A pair is taken to be **resolved** when *both* hold: `P(two | B) ≥ 0.9` (strong
evidence, ≈ 10:1 odds, ≈ log₁₀B₁₀ ≥ 1, and close to the calibrated
10%-false-alarm operating point of §3.1) **and** the two 95% credible regions do
not overlap. The spatial test is evaluated per trial, not on the pair-averaged
density: averaging over pairs inflates each source's region (it folds in
pair-to-pair scatter) and would make "disjoint" fail even for cleanly resolved
pairs. Detection is the easier bar; resolution is the harder one, and the two
come apart with depth. §4.1–4.2 show the posterior densities behind these
verdicts; §4.3 reduces them to the verdict itself.

### 4.1 Canonical-frame average

A single simulated pair with a single noise draw is idiosyncratic: depth,
orientation, and gain balance make individual panels at the same separation
disagree (one merged blob versus two crisp maxima). To show the *typical*
posterior rather than one realization, trials are averaged in a canonical frame.
For each (SNR, separation):

1. Draw many depth-matched pairs at that separation (default 40 pairs × 2 noise
   draws).
2. For each trial, split the joint posterior mass between the two true sources
   (assign each candidate pair to sources A/B by the closer pairing — legitimate
   for validation, where truth is known).
3. Rotate the density in the axial plane so the A→B direction lies along +x and
   the pair midpoint sits at the origin; source A therefore lands at (−d/2, 0)
   and B at (+d/2, 0).
4. Accumulate (bilinear) into a common fine grid (0.2 mm) and average.

The result is displacement space (mm from the pair midpoint), not anatomy: depth
is projected out and every pair's orientation is rotated into a common frame, so
idiosyncratic pairs average away. Colour encodes `P(a source within 1 mm of this
point)` — a bounded probability that peaks at the source and merges into a single
region when two sources are unresolvable. The average is computed on the 1 mm
grid; smoothness comes from averaging over pairs, not from grid resolution, so a
finer grid (whose pair scan is quadratic) buys nothing here.

**Reading.** At +20 dB two crisp, equally bright regions appear at every
separation (70–91% of mass within 0.5 mm). At +10 dB they fuse at ~2 mm and
resolve by ~6 mm. At 0 dB they are diffuse and merged at every separation
(Bayes factor ≈ 0).

*Figure:* `two_source_series.png`.

### 4.2 Stratification by depth

The same canonical average, computed within bands of source depth (distance to
the nearest electrode), one figure per SNR. Depth — not SNR — is the binding
constraint on localization: gain falls and the point-spread flattens with depth,
so deep sources reconstruct nearly uniformly regardless of noise. The four bands
are very shallow (1.2–2.3 mm, median 1.8), shallow (2.3–3.5, median 3.0), mid
(3.5–4.7, median 4.1), and deep (4.7–7.6, median 5.5).

**Median log₁₀ Bayes factor at 6 mm separation** (a generous separation, so
these reflect the depth ceiling rather than a too-close-pair effect):

| depth band (median) | +20 dB | +10 dB | 0 dB |
|---|---|---|---|
| very shallow (1.8 mm) | +131.8 | +9.6 | +0.0 |
| shallow (3.0 mm) | +69.9 | +3.9 | −0.2 |
| mid (4.1 mm) | +2.7 | −0.2 | −0.4 |
| deep (5.5 mm) | −0.3 | −0.2 | −0.1 |

**P(a source within 0.5 mm of its true position), at 6 mm separation:**

| depth band (median) | +20 dB | +10 dB | 0 dB |
|---|---|---|---|
| very shallow (1.8 mm) | 94% | 53% | 7% |
| shallow (3.0 mm) | 88% | 32% | 4% |
| mid (4.1 mm) | 20% | 4% | 2% |
| deep (5.5 mm) | 4% | 2% | 2% |

**Depth sets a hard ceiling that SNR cannot lift.** Even at +20 dB, resolvability
collapses from near-certain at the surface to essentially absent in the deep
band: the deep Bayes factor stays at or below zero (the model comparison prefers
*one* source) at every separation out to 8 mm, and localization mass within
0.5 mm falls to ~4%. A deep pair is unresolvable no matter how strong the signal
or how far apart the sources are. Only the very-shallow band survives the drop to
+10 dB; at 0 dB no band resolves.

*Figures:* `two_source_series_depth_p20dB.png`, `two_source_series_depth_p10dB.png`,
`two_source_series_depth_p0dB.png`.

### 4.3 Resolvability summary map

The verdict itself, per depth band × separation × SNR: over many depth-matched
trials, the **detection rate** (fraction with `P(two|B) ≥ 0.9`) and the
**resolution rate** (fraction *resolved* — detected AND the two 95% credible
regions disjoint). A cell is called resolvable when it is resolved on the
majority (≥50%) of trials.

**Resolution rate at 6 mm separation** (a generous separation, so the numbers
reflect the depth/SNR ceiling rather than a too-close-pair effect):

| depth band (median) | +20 dB | +10 dB | 0 dB |
|---|---|---|---|
| very shallow (1.8 mm) | 100% | 78% | 0% |
| shallow (3.0 mm) | 95% | 55% | 0% |
| mid (4.1 mm) | 38% | 0% | 0% |
| deep (5.5 mm) | 0% | 0% | 0% |

**Two sources are reliably resolved only when both sit shallow (≤ ~3.5 mm from
the array) and SNR is ≥ +10 dB, and then only for separations ≳ 3–4 mm.** Deep
pairs are neither detected nor resolved at any SNR or separation. The map also
makes the detection-vs-resolution gap concrete: at +20 dB, mid-depth pairs at
6–8 mm are *detected* as two ~65% of the time (`P(two|B) = 1.0`) but *resolved*
into distinct locations only ~38–50% of the time — the evidence says "two", yet
the two cannot be placed apart. Detection always reaches somewhat deeper than
resolution.

Note the detection rates here use the fixed `P(two|B) ≥ 0.9` cut, whereas the
§3.1 detection table uses the calibrated matched-null operating point (10%
false-alarm), whose threshold sits well below `P = 0.9` at low SNR. The two are
therefore not directly comparable: §3.1 is the false-alarm-controlled operating
point, §4.3 the stricter fixed-evidence view. Both tell the same story — depth
and SNR, not separation, are what bind.

*Figure:* `two_source_resolvability_map.png` (colour = median `P(two|B)`; cyan
dashed = detection threshold; green outline = resolved on ≥50% of trials).

---

## 5. Reproducibility

From the repo root with the virtual environment active
(`source_localization` 0.4.0):

```bash
P=<pipeline_dir>
O=<output_dir>

# detection, separation, canonical-frame series (§3–4.1)
python scripts/run_two_source_posterior.py --pipeline-dir $P \
    --output-dir $O/two_source_posterior

# resolvability stratified by depth, one figure per SNR (§4.2)
python scripts/run_two_source_posterior.py --pipeline-dir $P \
    --output-dir $O/two_source_posterior --depth-series

# resolvability summary map: detection + resolution rate by depth x sep x SNR (§4.3)
python scripts/run_two_source_posterior.py --pipeline-dir $P \
    --output-dir $O/two_source_posterior --resolvability-map --canonical-draws 1
```

The dense lead field is rebuilt from the cached BEM in seconds. The driver
caches its operator and per-figure data under `<output>/cache/`; rerunning with
`--replot` redraws all figures from cache (no forward solve, pair scan, or
simulation). Cache validity is keyed on the pipeline directory, grid spacing,
`inside_frac`, and the BEM file's size and modification time, so a rebuilt head
model invalidates it automatically. The `--canonical-pairs` and
`--canonical-draws` flags control the averaging depth of the resolvability maps.

---

## 6. Assumptions and limitations

1. **Fundamental limit vs deployed pipeline.** The posterior assumes a correct
   forward model and known noise; it bounds what any estimator could achieve. The
   pipeline's sLORETA output is worse; that gap is characterized in
   `SINGLE_SOURCE_VALIDATION.md` §4 (position error) and
   `ROI_CERTAINTY_VALIDATION.md` (ROI attribution), not outstanding.
2. **White noise — checked separately, in the evoked regime.** The credible-region
   calibration (`SINGLE_SOURCE_VALIDATION.md` §3) is verified under white noise.
   Whether it survives realistic structured noise, and whether whitening by the
   estimated noise covariance restores it, is the **noise-model robustness**
   dimension, scoped to **evoked** responses — the only regime in which whitening
   is well-posed. It makes no claim about resting-state use, which has no
   signal/noise partition and is neither validated nor invalidated by it.
3. **SNR is not measurable in resting-state recordings** (no signal/noise
   partition, no time-locked component to average up). The SNR axis here is a
   property of the simulation, not of the target recordings; see
   `SOURCE_CERTAINTY.md` §4.
4. **Matched moment prior**, **uniform spatial prior**, and **one- and
   two-dipole models only** — distributed or extended sources are out of scope.
5. **Depth-matched pairs.** The resolvability numbers hold depth roughly fixed
   within a pair; mixed-depth pairs are dominated by gain imbalance rather than
   the resolution limit under study, and are not characterized here.
