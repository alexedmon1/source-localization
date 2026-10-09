# How certain is a source localization result?

> **Provenance.** The reasoning record behind the three methods supplements, moved into the repository from the external staging area on 2026-10-09, text
> unchanged except paths. Produced July 2026 (v0.4-0.5) with `DipolePosterior` under its stated assumptions: an
> exact forward model and white noise of known size. These are the **fundamental limits under a correct model**,
> the same assumptions the v0.6.0 **legacy** regime makes (see `source_localization.validation.regime`). They
> include no truth head-model perturbation, no recorded backgrounds and no noise-only control. The noise-only
> controls now written by `run_posterior_orbital.py` and `run_roi_certainty.py` are in the
> [validation guide](../README.md), section 6. Reproduction commands use `<pipeline_dir>` (an ellipsoid-BEM shell
> run, see the guide, step 2) and `<output_dir>`.

**Status:** working document, branch `feat/validation-expectation-ranges`.
Covers the single-source case and the two-source case. The ROI-level statement
("is this activity really from ROI X and not elsewhere?") builds directly on
these and is sketched at the end.

**Scope:** 30-channel mouse EEG, ellipsoid 3-layer BEM, Antwerp atlas geometry.
All numbers below are for that array. More electrodes is the lever that would
change them; nothing else here comes close.

---

## Figures

These are the figures cited by this reasoning record:

| Figure | Shows | Section |
|---|---|---|
| `posterior_calibration.png` | Coverage vs nominal level, profile vs marginal, across SNR | §2.1 |
| `posterior_orbital.png` | p(location \| data) slices, 50% / 95% regions, truth marked | §2.2 |
| `two_source_posterior.png` | Bayes factor vs null, detectability, separation estimate, two-lobe density | §3 |

**Resolvability figures (2026-07-22), in `../two_source_posterior/`.** The
per-source resolvability picture is now carried by the canonical-frame average —
`two_source_series.png` (pooled) and `two_source_series_depth_{p20,p10,p0}dB.png`
(stratified by depth, one per SNR). These supersede the single-pair panels D–F of
`two_source_posterior.png` for the "where are they" question and are documented in
`TWO_SOURCE_VALIDATION.md` §4. The two shipping methods supplements
(`SINGLE_SOURCE_VALIDATION.md`, `TWO_SOURCE_VALIDATION.md`) carry the current
numbers in tabular form; this document is the fuller reasoning record behind them.

Figures from earlier generations of this work have been moved out of this
directory deliberately, so nothing here can later be picked up and mistaken for
a current result:

- `../retracted/` — analyses that produced **wrong** numbers. §1.1 and §1.3
  below describe the errors; the artifacts are there if the reasoning needs
  auditing, and its README explains each one.
- `../superseded/` — analyses that were **sound but have been replaced**: the
  resolution maps, the threshold-separability work, the 3D reconstruction
  renders, and the earlier two-dipole sweeps. Its README lists, per directory,
  which findings still stand and which should not be quoted.

**Not yet built:** a 3D transparent-brain render with explicit boundaries and
depth cues, and the ROI-certainty figures of §6.

---

## 1. The question, and why the earlier answers were wrong

We want to say how much to trust a spatial claim: "activity is *here*, not
there". Three earlier attempts each failed for an instructive reason, and the
failures are worth keeping because they are easy to repeat.

### 1.1 Spatial dispersion (RETRACTED)

Spatial dispersion (SD) is the RMS distance of a reconstruction from the true
source, weighted by `d² · psf²`. Reported as "~5 mm blur".

**It is nearly meaningless here.** Because distance enters as `d²`, faint
far-field leakage dominates it. A *completely flat* map — zero information —
scores 5.3–5.7 mm on this geometry. Measured values are 4.5–5.5 mm, i.e.
**78–96% of the ceiling**:

| depth bin | measured SD | flat-map SD | measured ÷ flat-map |
|---|---|---|---|
| very_shallow | 4.49 mm | 5.73 mm | 78% |
| shallow | 4.83 mm | 5.27 mm | 92% |
| mid | 5.17 mm | 5.58 mm | 93% |
| deep | 5.51 mm | 5.74 mm | 96% |

> **The % here is a ratio of two distances, not a probability.** 100% would mean
> the metric scores exactly what a completely uninformative map scores.
>
> **Good = low. Bad = near 100%.** Below ~50% the metric would have real
> headroom and be reporting something about localization. **Observed 78–96% is
> bad** — SD is nearly saturated, so almost all of what it reports is the size
> of the source space. This is a verdict on the *metric*, not on the pipeline.

It has almost no dynamic range, which is why it barely moved with depth or SNR.
It was largely measuring the size of the source space. **Do not quote it.**

### 1.2 The reconstruction is a core on a pedestal

The point-spread function decays with distance and then *plateaus* — it never
reaches zero. The plateau height is the depth-dependent part:

| depth bin | 0–1 mm | 2–3 mm | 4–5 mm | 8–9 mm | half-max radius |
|---|---|---|---|---|---|
| very_shallow | 1.00 | 0.51 | 0.29 | 0.25 | **2.6 mm** |
| shallow | 0.97 | 0.60 | 0.47 | 0.41 | 3.5 mm |
| mid | 0.94 | 0.66 | 0.55 | 0.47 | 8.2 mm |
| deep | 0.99 | 0.81 | 0.64 | 0.69 | never reaches half |

> **Cells are reconstructed amplitude as a fraction of that map's own peak**
> (1.00 = the peak), averaged over sources in the depth bin, at the distance
> from the true source given by the column. They are not probabilities and do
> not sum to anything.
>
> **Good = falls to near 0 within a few mm, and a small half-max radius. Bad =
> levels off well above 0.** A far-field value of 0.1 would mean activity 8 mm
> away is clearly distinguishable from the source; 0.7 means it is nearly
> indistinguishable. **very_shallow (0.25, half-max 2.6 mm) is acceptable; deep
> (0.69, never reaching half-max) is bad** — a deep source's reconstruction is
> barely distinguishable from a uniform map.

That pedestal is what defeats both SD and any threshold-free lobe detector.

### 1.3 Thresholding the reconstruction (SUPERSEDED)

Taking everything above X% of the peak and counting connected blobs is how a
source map is actually read, and it does remove the pedestal. Blob radius has
real dynamic range (very_shallow 4.04 mm at a 15% threshold → 0.90 mm at 80%;
deep 4.11 → 3.84, i.e. never tightens).

> **"T = 25%" here means a contour drawn at 25% of the reconstruction's peak
> amplitude.** It is a display setting. Despite reading like the "25% credible
> region" of §2, the two are unrelated: this one says nothing about how often
> the source is inside it.
>
> There is no good/bad reading available for these numbers, and **that is the
> finding** — a threshold percentage cannot be audited against ground truth, so
> there is no basis on which to call any value right or wrong. Section 2 exists
> to replace it with a quantity that can be.

Nothing guarantees the true source is inside the 25% region 25% of the time — or
any other fraction. This is exactly the objection that motivated the rest of
this document.

It also revealed a squeeze that any threshold method faces: at a low threshold
both sources survive but merge into one blob (90–100% of pairs); at a high
threshold the blobs finally separate but the weaker source has dropped below the
threshold entirely (62–93%). No threshold does both well.

> **Depth-matching matters.** Gain falls steeply with depth, so two
> equal-strength dipoles at different depths reconstruct with very unequal
> amplitude (mean ratio 0.38 unrestricted, 0.71 when matched to ±0.4 mm).
> Unmatched pairs fail mostly because the weaker source vanishes — a gain effect
> masquerading as a resolution limit. Always depth-match when asking a
> resolution question.

---

## 2. What replaces them: a calibrated posterior

Instead of asking what a source *does to* the reconstruction (forward), ask
where the source *is* given the data (inverse):

```
p(x | B) ∝ exp( −r(x) / 2σ² ),    r(x) = ‖B − Q_x Q_xᵀ B‖²
```

with the dipole moment integrated out under a Gaussian prior, which contributes
an Occam term. This is a genuine probability density: it normalizes over the
volume, and its **credible regions can be tested against ground truth**.

Evaluated on a dense grid (0.5 mm, 4,251 positions, 531 mm³) rebuilt from the
same cached BEM — deliberately not the pipeline's 215-point lattice — with truth
placed at continuous off-grid positions.

> **Correction (2026-07-21).** An earlier version of this section built that
> grid from the BEM's *innermost surface*. That surface is not the brain: it is
> an ellipsoid fitted to the brain voxels and then inflated by
> `ellipsoid_margin: 1.23`, so it is ~23% larger on every axis and ~1.9× the
> volume. The consequences were that 49% of candidate positions lay outside real
> tissue, 28 of them sat **above the highest electrode** (impossible source
> locations), and the brain volume was overstated as 753 mm³ against an actual
> 531 mm³ — which halved every "percent of brain" figure. The grid is now
> constrained by the atlas brain mask, the same constraint the pipeline's own
> source space uses, and a check rejects any candidate position above the
> electrode plane. All numbers in §2.2 and §3 are the corrected ones.
>
> Note the pipeline itself was never affected: it places sources at 0.25–0.95 of
> the brain radius and never samples the inflated region. This was a defect in
> the validation code only.
>
> Reading the atlas mask requires `utils.atlas.get_true_affine`, which divides
> the header voxel sizes by 10 **and the translation with them**. Scaling only
> the 3×3 block puts the mask nowhere near the source coordinates.

### 2.1 The percentages are audited

*Figure: `posterior_calibration.png`.*

A p% credible region should contain the true source p% of the time. It does:

| SNR | asked for 50% | asked for 68% | asked for 90% | asked for 95% |
|---|---|---|---|---|
| +20 dB | got 70% | got 83% | got 96% | got 98% |
| +10 dB | got 57% | got 74% | got 94% | got 97% |
| 0 dB | got 49% | got 73% | got 94% | got 96% |
| −5 dB | got 50% | got 68% | got 90% | got 96% |

> **The header and the cells are different quantities, and the table only means
> something because they can disagree.**
> - **Column header** = the *nominal credible level*: the probability mass we
>   asked the region to contain. A design parameter — we choose it.
> - **Cell** = the *empirical coverage*: the fraction of simulation trials in
>   which the true source actually fell inside that region. Measured, over 200
>   trials per cell (sampling error ≈ ±3 percentage points).
>
> **Good = cell matches header. Bad = cell below header.** Below is the
> dangerous direction: it means the region is too small and the stated
> confidence is a lie (asking for 90% but being right 60% of the time). Above
> the header is merely wasteful — the region is larger than it needed to be, so
> claims are weaker than they could be but not wrong.
>
> Agreement is the claim being made. **Observed: good** — every cell is within
> a few points of its header, erring high. For the *profile* likelihood they
> disagreed badly — asked for 68%, got 53% — which is what over-confidence
> looks like and why the marginal version is used instead.

Slightly conservative (regions a little larger than strictly needed), which is
the safe direction. **Marginalizing the dipole moment is what buys this** —
profiling it out instead is over-confident once noise approaches signal (its 68%
region covered only 53% at 0 dB). Residual over-coverage at +20 dB is grid
discretization, not a modelling error.

### 2.2 How precisely can a single source be located?

Pooled over random positions and orientations (brain volume 531 mm³):

| SNR | 50% region | 95% region | 95% equiv. radius | 95% region as % of brain |
|---|---|---|---|---|
| +20 dB | 0.2 mm³ | 0.9 mm³ | 0.59 mm | 0.2% |
| +10 dB | 7.3 mm³ | 97.2 mm³ | 2.85 mm | 18.3% |
| +5 dB | 26.3 mm³ | 152.4 mm³ | 3.31 mm | 28.7% |
| 0 dB | 52.1 mm³ | 250.8 mm³ | 3.91 mm | 47.2% |
| −5 dB | 99.4 mm³ | 352.4 mm³ | 4.38 mm | 66.3% |

> **The % here is a volume ratio** — the 95% credible region's volume divided by
> the whole source space (531 mm³). It is *not* a confidence; the confidence is
> fixed at 95% and the volume is what varies.
>
> **Good = small. Bad = large.** Rough reading, since the useful comparison is
> to whatever anatomy you want to resolve:
> - **<1%** — sub-structure resolvable; a claim about a specific small ROI is
>   supportable.
> - **~10%** — usable for coarse regional claims only (lobe-scale).
> - **>40%** — the honest statement is "somewhere in roughly this half of the
>   brain". No ROI-level claim survives.
>
> **Observed: good only at +20 dB.** At 0 dB and below the region covers 41–63%
> of the brain, which is effectively no spatial information. Note also that the
> "equivalent radius" column assumes the region is a compact ball — for deep
> sources it is a crescent, so that column flatters the result and the volume is
> the number to trust.

**Depth dominates these pooled numbers.** At 0 dB a dorsal source has a 95%
region of 6 mm³ (0.8% of the brain) while a ventral one spans 237 mm³ (31.5%) —
and the ventral region is a crescent, not a ball, so an equivalent radius
understates how ambiguous it is. See `posterior_orbital.png`.

---

## 3. Two sources

From `scripts/run_two_source_posterior.py`; figure `two_source_posterior.png`.
Grid 1.0 mm (522 positions, 135,981 pairs — the pair scan is quadratic), sources
restricted to the **dorsal half** of the brain, random orientations, 30 trials
per cell. The two-dipole posterior separates three questions the earlier
threshold work ran together.

### 3.1 Are there two sources at all?

Bayes factor for a two-source vs one-source model. Crucially this is scored
against a **matched one-source null** — the same analysis run on data from a
single source — so "detection" is called at a fixed 10% false-alarm rate rather
than at an arbitrary threshold. The null behaves correctly: its median log₁₀BF
is negative (−2.17 at +20 dB), i.e. the model comparison actively prefers one
source when there is one.

**P(detect two sources), 10% false-alarm rate:**

| SNR | null median log₁₀BF | 2 mm | 3 mm | 4 mm | 6 mm | 8 mm |
|---|---|---|---|---|---|---|
| +20 dB | −2.17 | 97% | 100% | **100%** | 100% | **100%** |
| +10 dB | −0.88 | 73% | 80% | 80% | 90% | 80% |
| 0 dB | −0.25 | 27% | 40% | 50% | 70% | 53% |

> **The % here is a detection rate** — the fraction of simulation trials (30 per
> cell, so ±9 percentage points) in which the two-vs-one Bayes factor beat the
> null's 90th percentile. It is a frequency, not a confidence attached to any
> single recording.
>
> **The floor is 10%, not 0%.** The threshold is set at the null's 90th
> percentile, so one-source data trips it 10% of the time *by construction*.
> That is the false-alarm rate we chose to accept.
> - **Good = ≥80%**, i.e. two sources are found nearly whenever present.
> - **Marginal = 40–70%** — better than chance, but a coin-flip per recording.
> - **Bad = near 10%** — indistinguishable from the false-alarm floor, meaning
>   the method cannot tell two sources from one at all.
>
> **Observed: +20 dB is good (97–100%), +10 dB is marginal (73–90%), 0 dB is
> poor** (27–70%, never reaching the ≥80% bar).
>
> The **null median log₁₀BF** column is a sanity check on the method, not a
> result: it should be negative, meaning the model comparison correctly prefers
> one source when only one is present. It is (−2.17 to −0.25). Had it been
> positive, the whole table would be untrustworthy.

**SNR, not separation, is the binding constraint.** At +20 dB two sources are
detectable essentially always, down to 2 mm. At +10 dB it is a coin-flip to
90%. At 0 dB detection (27–70%, rising with separation) is unreliable and never
reaches the ≥80% bar.

The curves are nearly flat in separation above ~3 mm at +20 and +10 dB. What
limits detection there is not how close the sources are, but whether the noise
permits the second dipole to be distinguished at all.

### 3.2 How far apart are they?

A different, harder problem — posterior median separation vs truth:

| SNR | true 2 mm | true 3 mm | true 4 mm | true 6 mm | true 8 mm |
|---|---|---|---|---|---|
| +20 dB | 2.8 | 3.2 | 4.2 | 5.8 | 7.8 |
| +10 dB | 4.8 | 4.8 | 4.8 | 5.8 | 6.2 |
| 0 dB | 5.2 | 5.2 | 5.2 | 5.2 | 5.2 |

> **Cells are the posterior's median estimate of the separation, in mm**;
> compare each against its column header, which is the truth.
>
> **Good = cell matches header. Bad = cell constant across the row**, because a
> number that does not move when the truth moves carries no information about
> the truth — it is just the prior showing through.
>
> **Observed: +20 dB is good from ~3 mm up** (4.2/5.8/7.8 against 4/6/8), with
> the true-2 mm pair estimated at 2.8. **+10 dB and 0 dB are bad** — +10 dB
> saturates near 4.8–6 mm and 0 dB is flat at 5.2 mm whether the truth is 2 mm
> or 8 mm. Combined with §3.1 this gives the operationally important case: at
> +10 dB you may be able to say *two sources are present* while being unable to
> say *how far apart they are*.

At +20 dB the estimate tracks truth from ~3 mm up. At +10 dB it saturates near
4.8–6 mm and at 0 dB it is **flat at 5.2 mm regardless of the true separation** —
uninformative. So at working SNRs you may be able to say *there are two sources*
without being able to say *how far apart they are*.

### 3.3 Where are they?

The symmetrized marginal `p(a source is at x | B)` — the density of *a* source,
which is the honest object because the two sources are exchangeable. Panels D–F
of the figure show a representative 4.5 mm pair: at +20 dB two lobes with an
ambiguity bridge (log₁₀BF +21.8, posterior separation 4.2 mm vs 4.5 true); at
+10 dB and 0 dB it collapses to a single blob (log₁₀BF −0.5 and +0.0).

The *typical* version of this picture — averaged over many pairs in a canonical
frame, and stratified by depth — is now the primary resolvability figure; see
`TWO_SOURCE_VALIDATION.md` §4 and the note in the figure inventory above.

### 3.4 What this replaces

This supersedes the earlier prominence-gated two-lobe detector and its
hand-tuned null, and it explains why that work concluded so pessimistically: it
tested at a single moderate SNR, where §3.1 shows detection genuinely is
marginal. The SNR dependence was the missing axis.

---

## 4. Which SNR applies to the real recordings?

**⚠️ The honest answer is that SNR is not measurable in this data, and an earlier
version of this section wrongly claimed it was (≈ −2 to 0 dB). That claim is
retracted.** It is recorded here because the reasoning error is an easy one to
repeat.

### 4.0 Why the question has no answer as posed

SNR requires partitioning the recording into signal and noise. These are
resting-state segments (see §4.1): there is no stimulus, no time-locked
response, and no baseline in which the source of interest is known to be absent.
Every component of the recording is brain activity. There is nothing to call
noise.

What *can* be measured is the fraction of sensor variance a **single dipole**
explains — a model-adequacy statistic, not an SNR:

| recording | epochs | single-dipole explained variance (median) |
|---|---|---|
| D901_FX9_sbpro_0 | 214 | 0.48 |
| DFS9_931_0 | 216 | 0.39 |
| DFS9_937_0 | 218 | 0.37 |
| DFS9_955_0 | 662 | 0.42 |

> **Cells are a fraction of sensor variance (0–1), per time sample, median over
> samples** — how much of the measured scalp field the single best-fitting
> dipole reproduces. This is a *model-fit* statistic. It says nothing on its own
> about noise, because what the dipole misses is mostly other brain activity.
>
> **Good = near 1.0** (the data really do look like one dipole, so the model in
> §2–3 applies). **Bad = near 0.5 or below** (the data are not one-dipole-like
> and the single-dipole analysis is describing something the data do not
> contain).
>
> **Observed 0.37–0.48 is bad in that specific sense** — a single dipole
> accounts for well under half the sensor variance. That is unsurprising for
> ongoing activity with many simultaneous sources, and it is the honest reason
> to be cautious about applying §2–3 to these recordings.

Converting that to `SNR = EV/(1−EV)` — as this document previously did, giving
"−2 to 0 dB" — silently asserts that everything one dipole misses is noise. It
is not: it is mostly other simultaneous brain sources, plus mismatch between a
point dipole and genuinely distributed activity. The conversion is invalid and
those dB figures should not be quoted or compared against the simulation tables.

The one thing the explained-variance numbers do establish: **a single dipole
accounts for well under half the sensor variance**, so the single-dipole model
underlying §2 and §3 is a poor description of resting-state data regardless of
any noise question.

### 4.1 There is no evoked response to average up

Trial-averaging normally buys `10·log10(N)` dB — with ~216 epochs that would be
~23 dB, which would land squarely on the +20 dB row where everything works. It
does not happen here. The ratio of evoked RMS to single-trial RMS is:

| recording | epochs | evoked/single RMS | expected if pure noise (1/√N) |
|---|---|---|---|
| D901_FX9_sbpro_0 | 214 | 0.066 | 0.068 |
| DFS9_931_0 | 216 | 0.068 | 0.068 |
| DFS9_937_0 | 218 | 0.069 | 0.068 |

> **Cells are a ratio of amplitudes (unitless)**: the RMS of the trial-average
> divided by the RMS of a typical single trial. Compare against the last column,
> which is what the ratio would be if the epochs contained nothing time-locked.
>
> **Good (for averaging to help) = ratio near 1.0**, meaning the response
> survives averaging and N trials buy you 10·log₁₀(N) dB. **Bad = ratio near
> 1/√N**, meaning averaging cancels everything and buys nothing.
>
> **Observed: bad, and unambiguously so** — the measured ratio matches 1/√N to
> three decimal places. There is no time-locked component whatsoever. Note this
> is a statement about the *paradigm*, not the recording quality or the
> pipeline: there is simply no evoked response in these epochs to build up.

The ratio matches `1/√N` to three decimals: **there is no time-locked component
at all.** Averaging cancels essentially everything, so the trial-averaged SNR in
the table above is not an evoked signal — it is the residual. These epochs are
effectively resting-state segments.

What this does establish, independent of any SNR question: **averaging cannot
improve matters in this dataset.** Trial-averaging normally buys `10·log10(N)`
dB, which with ~216 epochs would be ~23 dB — enough to move from the bottom row
of §3.1 to the top. There is nothing time-locked for it to build up. That is a
property of the paradigm, and no inverse method can recover what the recording
does not contain.

### 4.2 The defensible reformulation (proposed, not yet built)

The way out is to stop asking for an SNR and instead ask a question that is
well-posed for ongoing data:

> **How strong must a source be, relative to the ongoing activity actually
> present in these recordings, before it can be localized to within X mm?**

That is answerable because the *added* source is under our control even though
the background is not. Concretely:

1. Estimate the sensor covariance **C** of the real recordings — the empirical
   spatial structure of ongoing activity. No signal/noise split required.
2. Whiten the posterior with **C** instead of assuming `σ²I`. This also fixes
   the white-noise assumption flagged below, and connects to the known
   [scaled-identity regularization limitation](../../known-issues/REGULARIZATION_SCALING_ISSUE.md)
   in the pipeline's inverse.
3. Add a simulated test dipole of known amplitude on top of real data segments
   and measure credible-region size and two-source detectability as a function
   of that amplitude, expressed in nAm or as a ratio to ongoing activity power.

This replaces the SNR axis in §2.2 and §3.1 with an axis that means something
for resting-state data, and it grounds the whole characterization in the real
recordings rather than in white-noise simulation.

### 4.3 Calibration is unverified for real data

The residual in real recordings is **spatially structured** (other brain
sources), whereas the posterior in §2 assumes white noise. The calibration
demonstrated in §2.1 was verified under white noise only, so **on real data the
credible regions are likely over-confident** — the true regions would be larger.
Re-running the coverage test with the empirical covariance from §4.2 is the
check that would settle it, and until it is done every number in §2 and §3
should be read as a best case for idealized noise, not a description of these
recordings.

---

## 5. Caveats that apply to everything above

1. **This is the fundamental limit, not the pipeline.** The posterior assumes a
   correct forward model and known white noise. It bounds what *any* estimator
   could achieve. The pipeline's sLORETA output is worse; the gap is the
   estimator's cost, and quantifying it is outstanding work.
2. **The moment prior is matched** to the simulated amplitude. Real data would
   have to estimate it.
3. **Uniform spatial prior.** Deep sources come out broad because the data are
   genuinely uninformative there, not because of the prior.
4. **Single- and two-dipole models only.** Distributed or extended sources are
   not covered.

---

## 5. Toward ROI-level certainty (next)

The practical question is: *if activity is attributed to ROI X, how confident is
that, versus it having come from elsewhere?* The posterior answers this directly
by integration — no new machinery is needed:

```
P(source ∈ ROI_k | B) = Σ_{x ∈ ROI_k} p(x | B)
```

That gives a per-ROI probability for each recording, and because the underlying
density is calibrated, the ROI probabilities inherit the property that can be
audited: when the method says 70%, it should be right 70% of the time (a
reliability diagram over simulated sources). Expected outputs: a per-ROI
certainty table, the confusion structure (which ROIs leak into which), and the
reliability curve.

Note in advance: an ROI smaller than the credible region can never be asserted
confidently at that SNR — the 95% regions in §2.2 span 10–60% of the brain at
10 dB and below, which already bounds what any ROI-level claim can look like.
