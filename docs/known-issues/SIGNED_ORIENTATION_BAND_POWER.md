# Known issue: the per-epoch signed orientation distorts band power

**Affects:** every free-orientation run whose analytics read the signed series (`step6_roi_timeseries_signed.pkl`)
under the default `inverse.signed_orientation: per_epoch_variance`. That includes the roi_based presets, and so
FORGE's `rest_roi_allen26` (v0.5.1, where per-epoch is the only rule) and the source-analytics roi_psd and
roi_aperiodic tables built on it. The magnitude series (L2 norm over the three components) does not choose a
component and is not affected.

**Related:** the phase cost of the same rule (surface-evoked X56; troubleshooting log section 28; commit `bc4c63f`,
which added `inverse.signed_orientation: leadfield` as an option and kept the default). This note is the band-power
cost.

Filed: 2026-10-21, from bayes-sl's end-to-end leakage check (EE2, `reports/2026-10-19_end_to_end_plan.md`).

**Summary.**
- In simulation, a planted band-limited source is suppressed 15-25 dB until it wins its source's broadband variance.
  Then the ROI flips and the reading jumps; near the threshold the response can be non-monotone.
- On FORGE's 60 real recordings the choice rarely flips (97% of source-epochs at the source's usual component, 75%
  x; real gamma barely moves it). The readout is in practice a fixed, mostly-x projection.
- FORGE's ROI-average Low Gamma group contrasts keep their direction under three other readouts (sizes move up to
  0.3 SD). Per-ROI contrasts move by up to 1.4 SD, and under the leadfield option four contrasts above 0.5 SD
  change sign.

## The rule

`apply_inverse_to_epoch` (`steps/inverse_solution.py`) computes, per epoch and per source, the three Cartesian
components of the sLORETA estimate and keeps the one with the **largest variance in that epoch**, over the whole
band the data carry. ROI series are the plain mean of the kept series over the ROI's sources.

The choice is made by whatever dominates a source's broadband variance in that epoch. A band-limited signal at that
source is read only through the kept component: in full if it lies along it, weakly or not at all if it does not.

## What the simulation shows

Set-up (bayes-sl `scripts/ee2_cohort.py`, `scripts/ee2_diag_component_flip*.py`): real FORGE backgrounds (the
`*_data_postcomp.set` epochs, as read) plus one planted Low Gamma (30-55 Hz) dipole, read out through FORGE's own
kernel (sLORETA, snr 3, the run's forward; reproduces `step6_roi_timeseries_signed.pkl` exactly) and measured as
source-analytics measures it (Welch, absolute dB, 50 draws of 80 x 2-s epochs).

- At every source, the background's broadband variance is largest in x (normalised variances x, y, z around
  1 : 0.1-0.5 : 0.1-0.8), so the readout keeps x.
- A dipole at its source's strongest-gain direction (mostly y or z) is nearly invisible in x. Read through the
  background's own choices it is attenuated 15-25 dB relative to its own component.
- Once its variance in its own component exceeds the background's in x, the choice flips, for every epoch of the
  ROI at once, and the reading jumps.

Thalamus, sub-801, plant amplitude swept (plant level relative to the background as read through the background's
choices):

| Plant | Real readout, Thalamus Low Gamma | Background's choices held fixed | Epochs whose Thalamus map changed |
|---|---|---|---|
| -15 dB | +0.3 dB | +0.1 dB | 9% |
| -9 dB | **+15.1 dB** | +0.5 dB | 100% |
| -3 dB | **+23.3 dB** | +1.8 dB | 100% |
| +15 dB | +41.4 dB | +15.2 dB | 100% |

With random dipole orientations (uniform on the sphere) the response is a threshold too, at a level that depends on
the orientation and the background:
- 2 backgrounds x 5 regions, amplitude -30 to +30 dB: nothing moves below about -20 dB; above it, increases of
  10-25 dB over 10-15 dB of amplitude.
- **Non-monotone:** in Frontal_Anterior (sub-801) a stronger plant *lowered* the ROI's Low Gamma by up to 3.5 dB
  over a 20 dB range of amplitudes. Plausibly the flips at neighbouring sources leave the ROI mean averaging
  components that now disagree in sign, which cancels power; not yet isolated.

## Why it matters

- **A source the background does not favour is read through the wrong component.** Below the threshold it is
  attenuated; FORGE's resting data sit there almost always (next section), so the readout is in practice a fixed,
  mostly-x projection that sees little of activity in y and z.
- **Above the threshold, band power is not read linearly.** A band-limited source that does win the variance flips
  the ROI: it reads as a jump, mice near the threshold split, and a larger source can read as less power. In FORGE's
  resting data this is rare; a strong evoked or pathological rhythm could cross it.
- **The choice is made once per epoch for all frequencies,** so a band is read through whichever component the
  dominant band picked.

## What the real data show

bayes-sl `scripts/ee2_flip_real.py`, the 60 analysed FORGE mice (`rest_roi_allen26`, every epoch), outputs in
bayes-sl `scratch/ee2_flip/`. The readout as rebuilt reproduces FORGE's 50-draw Low Gamma values to 5e-8 dB.

**1. On real data the choice rarely flips: it is a near-fixed, mostly-x projection.** Per mouse, over source x
epoch, median [range]:

| | |
|---|---|
| kept component is x | 0.75 [0.66, 0.79] |
| kept component is the source's modal one | 0.97 [0.92, 0.98] |
| kept component is also the Low Gamma-dominant one | 0.95 [0.87, 0.97] |
| kept component is also the Delta-dominant one | 0.73 [0.42, 0.94] |

- Less stable in cortex: modal share 0.80 in Auditory_L, 0.89-0.94 in Somatosensory, Visual_Parietal, Frontal_Anterior,
  Retrosplenial, Motor_R and Olfactory_Bulb; 0.97-1.00 in deep ROIs.
- **Real gamma barely moves it.** Off-modal share in the top vs the bottom quartile of each source's epoch Low Gamma
  power: 0.041 vs 0.033 on average; median difference per source 0.000; higher in high-gamma epochs at 15% of sources.

So the threshold the simulation shows is not crossed often by FORGE's own resting gamma. What the readout does in
practice is read a fixed projection per source, x at three sources in four, and see little of activity in the other
two directions. That is where the risk is: a group difference in another orientation is under-read, and per-ROI
results depend on which projection is read.

**2. FORGE's Low Gamma under four readouts.** signed (FORGE), fixed (each source's modal component, every epoch),
vector (the three Cartesian components' ROI means as separate series, band powers summed: no choice) and leadfield
(this repo's `inverse.signed_orientation: leadfield` directions for FORGE's forward). Group contrasts are plain mean
differences against WT_VEH in each readout's own pooled within-group SD (descriptive; not the Bayes engine).

| | fixed | vector | leadfield |
|---|---|---|---|
| Between-mouse correlation with signed, per ROI: median [min] | 0.983 [0.872] | 0.847 [0.513] | 0.578 [0.210] |
| Per-ROI contrast vs WT_VEH, \|readout - signed\|: median, max (SD) | 0.05, 0.71 | 0.11, 1.36 | 0.30, 0.90 |
| Sign disagreements, of 104 ROI x contrast (of which \|signed\| > 0.5) | 7 (0) | 6 (0) | 11 (4) |
| Correlation of the 104 contrasts with signed | 0.97 | 0.76 | 0.69 |

ROI-average contrast vs WT_VEH (SD):

| | signed | fixed | vector | leadfield |
|---|---|---|---|---|
| KO_VEH | 0.97 | 0.94 | 1.19 | 0.92 |
| KO_HD_IV | 1.13 | 1.05 | 1.43 | 1.09 |
| KO_LD_IV_ICV | 0.42 | 0.41 | 0.52 | 0.34 |
| KO_HD_ICV | 0.33 | 0.32 | 0.35 | 0.18 |

- **ROI-average conclusions hold** in direction under every readout. Sizes move by up to 0.3 SD; the vector readout,
  which sees all three directions, gives the largest (KO_VEH 1.19, KO_HD_IV 1.43).
- **Per-ROI conclusions do not.** Under the vector readout single contrasts move by up to 1.36 SD (Visual_Parietal_L,
  KO_HD_IV: 0.16 signed, 1.52 vector; Frontal_Anterior, KO_VEH: 0.08 vs 1.32). Under leadfield, four contrasts
  larger than 0.5 SD change sign.
- The leadfield directions are not a drop-in replacement for band power: between mice they correlate with signed at
  0.58 (median ROI), because they read a different projection (each source's most array-visible direction, mostly
  not x).
- Not separated here: how much of the vector readout's larger contrasts is signal the x projection misses, and how
  much is a smaller between-mouse SD.

## Options

1. **Vector power for band-power analytics** (proposed): take each ROI's mean of each Cartesian component as its own
   series and sum their band powers. No choice, no sign cancellation between components, linear in power, and every
   orientation counts. Not a single signed series, so not for phase or connectivity. On FORGE it keeps the
   ROI-average contrasts' direction and moves per-ROI contrasts (section above).
2. **`inverse.signed_orientation: leadfield`** (exists since `bc4c63f`): one fixed direction per source (the
   dominant leadfield direction, signs aligned within each parcel), the same for every epoch and recording. Band
   power is then linear in the source amplitude and the phase cost (X56) goes too. But only the projection on that
   direction is seen: a dipole perpendicular to it is invisible at any amplitude. On FORGE it reads a different
   projection from the per-epoch rule (between-mouse r 0.58). Suited to phase and connectivity, where a single
   signed series is needed.
3. **Magnitude series** for power measures: already written; no choice, but the norm is not a band-limited linear
   operation (it mixes bands), so band power of the magnitude is not band power of the source.
4. **Keep per-epoch, and say what it reads:** in practice a fixed, mostly-x projection on FORGE. Per-ROI band power
   claims then need to name the projection, and a planted-source check (bayes-sl EE2) has to use the same readout.

Not decided here. Which readout source-analytics' roi_psd and roi_aperiodic should read is the Conductor's call;
FORGE's published tables were made with per-epoch.

## Verification after a fix

Rerun the EE2 amplitude sweep (bayes-sl `scripts/ee2_diag_component_flip.py`) under the chosen readout: the Low
Gamma increase at the planted ROI should rise smoothly with amplitude (3 dB per doubling of plant power once the
plant dominates, no jumps, no decreases). Then recompute FORGE's roi_psd band power under it and compare the group
contrasts with the published tables.
