# How certain is a source localization result?

**Status:** working document, branch `feat/validation-expectation-ranges`.
Covers the single-source case and the two-source case. The ROI-level statement
("is this activity really from ROI X and not elsewhere?") builds directly on
these and is sketched at the end.

**Scope:** 30-channel mouse EEG, ellipsoid 3-layer BEM, Antwerp atlas geometry.
All numbers below are for that array. More electrodes is the lever that would
change them; nothing else here comes close.

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

| depth bin | measured SD | flat-map SD | % of ceiling |
|---|---|---|---|
| very_shallow | 4.49 mm | 5.73 mm | 78% |
| shallow | 4.83 mm | 5.27 mm | 92% |
| mid | 5.17 mm | 5.58 mm | 93% |
| deep | 5.51 mm | 5.74 mm | 96% |

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

That pedestal is what defeats both SD and any threshold-free lobe detector.

### 1.3 Thresholding the reconstruction (SUPERSEDED)

Taking everything above X% of the peak and counting connected blobs is how a
source map is actually read, and it does remove the pedestal. Blob radius has
real dynamic range (very_shallow 4.04 mm at a 15% threshold → 0.90 mm at 80%;
deep 4.11 → 3.84, i.e. never tightens).

But **"25%" is a display setting, not a probability.** Nothing guarantees the
true source is inside the 25% region 25% of the time — or any other fraction.
The percentages could not be audited, which is exactly the objection that
motivated the rest of this document.

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

Evaluated on a dense grid (0.5 mm, ~6,000 positions) rebuilt from the same
cached BEM — deliberately not the pipeline's 215-point lattice — with truth
placed at continuous off-grid positions.

### 2.1 The percentages are audited

A p% credible region should contain the true source p% of the time. It does:

| SNR | 50% | 68% | 90% | 95% |
|---|---|---|---|---|
| +10 dB | 63% | 74% | 89% | 95% |
| 0 dB | 55% | 73% | 88% | 97% |
| −5 dB | 56% | 73% | 89% | 94% |

Slightly conservative (regions a little larger than strictly needed), which is
the safe direction. **Marginalizing the dipole moment is what buys this** —
profiling it out instead is over-confident once noise approaches signal (its 68%
region covered only 53% at 0 dB). Residual over-coverage at +20 dB is grid
discretization, not a modelling error.

### 2.2 How precisely can a single source be located?

Pooled over random positions and orientations (brain volume ≈ 753 mm³):

| SNR | 50% region | 95% region | 95% equiv. radius | % of brain |
|---|---|---|---|---|
| +20 dB | 0.4 mm³ | 2.4 mm³ | 0.83 mm | 0.3% |
| +10 dB | 8.8 mm³ | 80.3 mm³ | 2.68 mm | 10.7% |
| +5 dB | 26.2 mm³ | 191.4 mm³ | 3.57 mm | 25.4% |
| 0 dB | 58.9 mm³ | 309.5 mm³ | 4.20 mm | 41.1% |
| −5 dB | 132.8 mm³ | 472.2 mm³ | 4.83 mm | 62.7% |

**Depth dominates these pooled numbers.** At 0 dB a dorsal source has a 95%
region of 6 mm³ (0.8% of the brain) while a ventral one spans 237 mm³ (31.5%) —
and the ventral region is a crescent, not a ball, so an equivalent radius
understates how ambiguous it is. See `docs/posterior_orbital.png`.

---

## 3. Two sources

From `scripts/run_two_source_posterior.py`; figure `docs/two_source_posterior.png`.
Grid 1.0 mm (759 positions, 288k pairs — the pair scan is quadratic), sources
restricted to the **dorsal half** of the brain, random orientations, 30 trials
per cell. The two-dipole posterior separates three questions the earlier
threshold work ran together.

### 3.1 Are there two sources at all?

Bayes factor for a two-source vs one-source model. Crucially this is scored
against a **matched one-source null** — the same analysis run on data from a
single source — so "detection" is called at a fixed 10% false-alarm rate rather
than at an arbitrary threshold. The null behaves correctly: its median log₁₀BF
is negative (−1.81 at +20 dB), i.e. the model comparison actively prefers one
source when there is one.

**P(detect two sources), 10% false-alarm rate:**

| SNR | null median log₁₀BF | 2 mm | 3 mm | 4 mm | 6 mm | 8 mm |
|---|---|---|---|---|---|---|
| +20 dB | −1.81 | 83% | 97% | **100%** | 97% | **100%** |
| +10 dB | −0.67 | 53% | 80% | 73% | 73% | 73% |
| 0 dB | −0.18 | 13% | 37% | 30% | 33% | 33% |

**SNR, not separation, is the binding constraint.** At +20 dB two sources are
detectable essentially always, down to 2–3 mm. At +10 dB it is a coin-flip to
80%. At 0 dB detection (13–37%) is barely above the 10% false-alarm floor —
you cannot tell.

Note the curves are nearly flat in separation above ~3 mm. What limits
detection here is not how close the sources are, but whether the noise permits
the second dipole to be distinguished at all.

### 3.2 How far apart are they?

A different, harder problem — posterior median separation vs truth:

| SNR | true 2 mm | true 3 mm | true 4 mm | true 6 mm | true 8 mm |
|---|---|---|---|---|---|
| +20 dB | 3.5 | 3.5 | 4.8 | 6.2 | 7.8 |
| +10 dB | 5.2 | 5.5 | 5.2 | 6.2 | 7.2 |
| 0 dB | 5.2 | 5.8 | 5.5 | 6.2 | 6.2 |

At +20 dB the estimate tracks truth from ~4 mm up (slightly over-estimating
close pairs). At +10 dB and below it **saturates around 5–6 mm regardless of the
true separation** — the estimate is uninformative below ~6 mm. So at working
SNRs you may be able to say *there are two sources* without being able to say
*how far apart they are*.

### 3.3 Where are they?

The symmetrized marginal `p(a source is at x | B)` — the density of *a* source,
which is the honest object because the two sources are exchangeable. Panels D–F
of the figure show a representative 4.1 mm pair: at +20 dB two lobes with an
ambiguity bridge (log₁₀BF +2.0, posterior separation 4.2 mm vs 4.1 true); at
+10 dB and 0 dB it collapses to a single blob (log₁₀BF −0.4).

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
   [scaled-identity regularization limitation](REGULARIZATION_SCALING_ISSUE.md)
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
