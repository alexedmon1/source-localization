# Model catalogue

Every probability model this package uses to say where a signal comes from: the location posteriors, the priors they rest on, the calibrations applied to their maps, the noise whitening they assume, and the definitions of the electrode fields they locate. Each has a stable **id** and a **version**; results name the ids and versions that produced them.

*Generated from `source_localization/methods.py` (`python -m source_localization.methods > docs/methods/CATALOGUE.md`); do not edit by hand. `tests/test_catalogue.py` checks it is current.*

## Location posteriors

| id | version | what it is |
|---|---|---|
| `locate.single_source_exact` | 1 | one dipole, exact head model |
| `locate.single_source` | 1 | one dipole, averaged over a head-model ensemble |
| `locate.mirror_pair` | 1 | a dipole and its mirror image across the midline |
| `locate.field_choice` | 1 | where a field is: the pair if P(pair) reaches the rule, else the single peak |
| `locate.field_model_map` | 1 | located mass of a sample's field over resamples of its subjects |
| `locate.source_set` | 1 | any fixed set of dipoles, with model evidence |

### `locate.single_source_exact` (version 1)

One dipole, exact head model.

- **Model:** Single free-orientation dipole at grid position x, forward model taken as exact, white sensor noise of known SD sigma. Default: the moment is profiled out, log p(B | x) = -||B - Q_x Q_x^T B||^2 / (2 sigma^2). With a moment variance given, it is marginalized instead: B | x ~ N(0, sigma^2 I + tau^2 G_x G_x^T).
- **Rests on:** `prior.position_uniform`, `prior.moment_given_sd`
- **Validity:** Coverage checked in simulation with the true head model (DipolePosterior.coverage_test). The fundamental limit for one dipole, not the pipeline estimator.
- **Caveat:** Assumes the head model is exact; for real data use locate.single_source. Profiling is overconfident once noise is comparable to the signal (at 0 dB its 68% region covered 53% of the time); the marginalized form stays calibrated.
- **Implementation:** `source_localization.validation.posterior:DipolePosterior`
- **Tests:** `tests/test_posterior.py`
- **Reference:** docs/validation/methods/SOURCE_CERTAINTY.md

### `locate.single_source` (version 1)

One dipole, averaged over a head-model ensemble.

- **Model:** Per whitened data column j: y_j | x, k ~ N(0, I + tau^2 G_xk G_xk^T), the moment marginalized. The likelihood is averaged over the K head models (not the posteriors), so the head model is marginalized: p(x | y) is proportional to (1/K) sum_k p(y | x, k). The one-dipole case of locate.source_set.
- **Rests on:** `prior.position_uniform`, `prior.moment_empirical_bayes`, `prior.head_ensemble_pa_phase2`
- **Validity:** Inputs must be whitened (the caller fits the whitener). Raw maps are overconfident; use a calibration from this catalogue inside its domain. Calibrated for single-source truths only.
- **Caveat:** A response with more than one source is read as one confident point between them; compare with locate.mirror_pair or locate.source_set through prior.model_odds_equal.
- **Implementation:** `source_localization.locate.ensemble_posterior:EnsemblePosterior`
- **Tests:** `tests/test_ensemble_posterior.py`
- **Reference:** module docstring of source_localization.locate.ensemble_posterior; bayes-sl reports/TB_map_calibration.md

### `locate.mirror_pair` (version 1)

A dipole and its mirror image across the midline.

- **Model:** locate.source_set with configurations (x, x mirrored across the midline x = midline_x_mm), x on the right at least min_offset_mm (default 0.3 mm) from it, mirrors snapped to the grid. Each member has an independent free-orientation moment. position_map gives each member half the mass of its configuration (a map summing to 2 counts both members).
- **Rests on:** `prior.position_uniform`, `prior.moment_empirical_bayes`, `prior.head_ensemble_pa_phase2`
- **Validity:** As locate.single_source. The single-source calibration carries over to pair maps on MEA30 (90% region: 92-93% coverage for auditory-cortex pairs, 81-83% for random pairs; bayes-sl reports/TB_mirror_pair.md).
- **Caveat:** Pairs very close to the midline look like one central source; at high SNR a single deep source can pass for a near-midline pair (bayes-sl tb16 control).
- **Implementation:** `source_localization.locate.ensemble_posterior:MirrorPairPosterior`
- **Tests:** `tests/test_ensemble_posterior.py`
- **Reference:** bayes-sl reports/TB_mirror_pair.md

### `locate.field_choice` (version 1)

Where a field is: the pair if P(pair) reaches the rule, else the single peak.

- **Model:** For a whitened field column: P(pair) from locate.single_source and locate.mirror_pair (equal prior odds). If P(pair) >= pair_rule (default 0.5), the location is the mirror pair at the pair map's peak (both members); otherwise the single-source map's peak. Both maps are calibrated, and the space must lie in the calibration's domain.
- **Rests on:** `prior.position_uniform`, `prior.moment_empirical_bayes`, `prior.head_ensemble_pa_phase2`, `prior.model_odds_equal`
- **Validity:** The rule of surface-evoked method_crossover.FieldModel.locate (H039 arm C) and bayes-sl L2/L3; with locate.mirror_pair's caveat that a single deep source at high SNR can pass for a near-midline pair.
- **Implementation:** `source_localization.locate.field_model:FieldLocator.locate`
- **Tests:** `tests/test_field_locate.py`
- **Reference:** docs/methods/FIELD_LOCATION.md

### `locate.field_model_map` (version 1)

Located mass of a sample's field over resamples of its subjects.

- **Model:** Per draw (subjects resampled with replacement; record weights w_unit / records of unit): the field is re-derived from the weighted records (evoked: first time course of the mean unit-normalised response; steady: field.phase_free; ongoing: field.csd_leading), the weighted grand-mean column and noise are formed along it, and the column is located by locate.field_choice; each draw puts mass 1/len(points) on its points. The whitener is fitted once per sample on one noise block per unit, through the whole sample's field.
- **Rests on:** `prior.position_uniform`, `prior.moment_empirical_bayes`, `prior.head_ensemble_pa_phase2`, `prior.model_odds_equal`
- **Validity:** Scored as the benchmark's arm FM (bayes-sl reports/2026-11-01_bm5_first_scores.md) and in the location scorecard (reports/2026-11-05_l6_final_scorecard.md).
- **Implementation:** `source_localization.locate.field_model:FieldModel.map`
- **Tests:** `tests/test_field_locate.py`
- **Reference:** surface-evoked method_crossover.FieldModel (H039 arm C); docs/methods/FIELD_LOCATION.md

### `locate.source_set` (version 1)

Any fixed set of dipoles, with model evidence.

- **Model:** A configuration q places d/3 free-orientation dipoles with stacked whitened leadfield G_q. Per column: y_j | q, k ~ N(0, I + tau^2 G_qk G_qk^T), each moment ~ N(0, tau^2 I3) independently, averaged over head models. log_evidence averages the likelihood over the set's configurations, so different sets can be compared.
- **Rests on:** `prior.position_uniform`, `prior.moment_empirical_bayes`, `prior.head_ensemble_pa_phase2`
- **Validity:** As locate.single_source; calibration checked for one source and for mirror pairs only.
- **Implementation:** `source_localization.locate.ensemble_posterior:SourceSetPosterior`
- **Tests:** `tests/test_ensemble_posterior.py`
- **Reference:** module docstring of source_localization.locate.ensemble_posterior

## Priors of the location posteriors

| id | version | what it is |
|---|---|---|
| `prior.position_uniform` | 1 | uniform over candidate positions or configurations |
| `prior.moment_empirical_bayes` | 1 | dipole moments Gaussian, scale estimated per pattern |
| `prior.moment_given_sd` | 1 | dipole moment Gaussian with a given SD, or profiled out |
| `prior.head_ensemble_pa_phase2` | 1 | head-model uncertainty: the 33-model ensemble |
| `prior.model_odds_equal` | 1 | equal prior odds between source sets |

### `prior.position_uniform` (version 1)

Uniform over candidate positions or configurations.

- **Model:** p(x) constant over the grid points inside the brain (or over a source set's configurations).
- **Values:** uniform; the grid spacing sets its resolution (0.5 mm by default)
- **Caveat:** Because gain falls with depth, deep sources get broad posteriors under it; that is a statement about the data, not the prior.
- **Implementation:** `source_localization.locate.ensemble_posterior:SourceSetPosterior`

### `prior.moment_empirical_bayes` (version 1)

Dipole moments Gaussian, scale estimated per pattern.

- **Model:** Each dipole moment ~ N(0, tau^2 I3). tau^2 is an empirical-Bayes estimate per data pattern: max(||y||^2 / c - s m, 1e-3 m) / median_x ||G_x,nominal||_F^2 (c columns, m channels, noise variance s = 1 after whitening). With several dipoles it is a per-source variance.
- **Values:** tau^2 from the data, not a fixed prior; floor 1e-3 m
- **Implementation:** `source_localization.locate.ensemble_posterior:SourceSetPosterior.moment_variance`
- **Reference:** probability-atlas pa.posterior (the reference implementation)

### `prior.moment_given_sd` (version 1)

Dipole moment Gaussian with a given SD, or profiled out.

- **Model:** With moment_std given: the moment ~ N(0, moment_std^2 I3) and is marginalized. Without it: no prior; the moment is profiled out (its maximum-likelihood value is used).
- **Values:** moment_std set by the caller; None profiles the moment out
- **Implementation:** `source_localization.validation.posterior:DipolePosterior.posterior`
- **Tests:** `tests/test_posterior.py`

### `prior.head_ensemble_pa_phase2` (version 1)

Head-model uncertainty: the 33-model ensemble.

- **Model:** The nominal head model plus 32 draws from HeadModelPrior: electrode-registration shift ~ N(0, 0.3 mm) per axis, skull-conductivity factor log-uniform on [0.5, 2.0]. Draws from numpy default_rng([20261005, 1, 0]); leadfields on the 0.5 mm volumetric grid, unreferenced as MNE returns them.
- **Values:** shift SD 0.3 mm (x, y, z), skull factor log-uniform 0.5-2.0, K = 1 nominal + 32 draws, seed [20261005, 1, 0]
- **Validity:** MEA30 on the ellipsoid BEM. Every calibration in this catalogue was measured with this ensemble.
- **Implementation:** `source_localization.locate.head_ensemble:ENSEMBLE_PA_PHASE2`
- **Tests:** `tests/test_locate.py`
- **Reference:** Ported from probability-atlas Phase 2 (pa/ensemble.py, scripts/phase2_simulate.py VARIANTS["primary"], K_ENS 32, rng_for(1, 0)); draws identical, leadfields equal to a relative 2e-16 on the Sentinel grid (field-method plan, F1)

### `prior.model_odds_equal` (version 1)

Equal prior odds between source sets.

- **Model:** P(set | y) from the sets' log evidences (locate.source_set.log_evidence) with equal prior probability for each set compared (for example one source vs a mirror pair).
- **Values:** equal prior probability per set
- **Validity:** P(pair) roughly calibrated, slightly overconfident, on MEA30 (patterns with P(pair) 0.95-1.0 were pairs 94.5% of the time; bayes-sl reports/TB_mirror_pair.md).
- **Implementation:** `source_localization.locate.ensemble_posterior:model_probabilities`

## Map calibrations

| id | version | what it is |
|---|---|---|
| `calibration.none` | 1 | the raw averaged posterior |
| `calibration.volumetric` | 1 | tempered and blurred, for volumetric maps |
| `calibration.surface` | 1 | tempered, for the cortical surface readout |

### `calibration.none` (version 1)

The raw averaged posterior.

- **Model:** Temperature 1, no blur.
- **Values:** temperature 1.0, blur 0 mm
- **Caveat:** Overconfident under head-model error: the 90% region held the truth about 55-65% of the time on MEA30 (bayes-sl reports/TB_map_focality.md).
- **Implementation:** `source_localization.locate.ensemble_posterior:UNCALIBRATED`

### `calibration.volumetric` (version 1)

Tempered and blurred, for volumetric maps.

- **Model:** Normalized log posterior divided by T = 1.5, then an isotropic Gaussian blur of 0.25 mm over the candidate positions (mass-conserving).
- **Values:** temperature 1.5, blur 0.25 mm
- **Validity:** MEA30, prior.head_ensemble_pa_phase2, 0.5 mm volumetric grid, single-source truths, whitened real noise. 90% region held the truth 88-89% of the time at half, primary and double SNR (80%: 79-81%; 95%: 92-94%). Chosen on one simulated set, checked on another.
- **Caveat:** Enforced: the field locator refuses a space outside this domain (array mea30_sentinel = data/calibration/mea30_sentinel.json, electrodes within 0.1 mm; ensemble; 0.5 mm; volumetric).
- **Implementation:** `source_localization.locate.ensemble_posterior:VOLUMETRIC`
- **Tests:** `tests/test_ensemble_posterior.py`
- **Reference:** bayes-sl reports/TB_map_calibration.md

### `calibration.surface` (version 1)

Tempered, for the cortical surface readout.

- **Model:** Normalized log posterior divided by T = 2.5; no blur (blurring made it worse).
- **Values:** temperature 2.5, blur 0 mm
- **Validity:** MEA30, prior.head_ensemble_pa_phase2, cortical/cerebellar/olfactory surface positions. Approximately calibrated: 90% region 88-91% at half and primary SNR, 83-85% at double SNR.
- **Caveat:** Enforced: the field locator refuses a space outside this domain (array mea30_sentinel, electrodes within 0.1 mm; ensemble; 0.5 mm; surface positions).
- **Implementation:** `source_localization.locate.ensemble_posterior:SURFACE`
- **Tests:** `tests/test_ensemble_posterior.py`
- **Reference:** bayes-sl reports/TB_map_calibration.md

## Noise whitening

| id | version | what it is |
|---|---|---|
| `whiten.sample` | 1 | pooled sample covariance shape, per-recording scale |
| `whiten.ledoit_wolf` | 1 | Ledoit-Wolf shrinkage covariance shape, per-recording scale |
| `whiten.oas` | 1 | OAS shrinkage covariance shape, per-recording scale |
| `whiten.diagonal` | 1 | independent electrode noise, then referenced |

### `whiten.sample` (version 1)

Pooled sample covariance shape, per-recording scale.

- **Model:** Data and leadfields are projected onto the n-1 average-reference directions (the reference electrode drops out). Each recording's noise scale is the RMS of its referenced noise samples; the spatial covariance shape C is pooled over recordings from their scale-normalised samples, estimated by the plain sample covariance. Whitener W = C^(-1/2) B'; a recording's pattern y becomes W y / scale. Noise samples are, for trial averages, (odd-trial average - even-trial average) / 2, so their variance is the noise in the full average; complex steady-state noise enters as [Re, Im] columns.
- **Values:** estimator "sample"
- **Validity:** Assumes every recording shares one noise shape and differs only in size. Which estimator to use is chosen per measure by held-out Gaussian log-likelihood among those whose held-out whitened power is within 1 +- 0.15 (choose_whitener; probability-atlas 2026-10-05 rule).
- **Caveat:** Amplifies noise it was not fitted on (held-out whitened power 2.2x for ASSR, 15-24x for chirp peaks in probability-atlas Phase 1); prefer a shrinkage estimator.
- **Implementation:** `source_localization.locate.whitening:NoiseWhitener`
- **Tests:** `tests/test_locate.py`

### `whiten.ledoit_wolf` (version 1)

Ledoit-Wolf shrinkage covariance shape, per-recording scale.

- **Model:** Data and leadfields are projected onto the n-1 average-reference directions (the reference electrode drops out). Each recording's noise scale is the RMS of its referenced noise samples; the spatial covariance shape C is pooled over recordings from their scale-normalised samples, estimated by scikit-learn's Ledoit-Wolf shrinkage (zero-mean samples). Whitener W = C^(-1/2) B'; a recording's pattern y becomes W y / scale. Noise samples are, for trial averages, (odd-trial average - even-trial average) / 2, so their variance is the noise in the full average; complex steady-state noise enters as [Re, Im] columns.
- **Values:** estimator "ledoit_wolf"
- **Validity:** Assumes every recording shares one noise shape and differs only in size. Which estimator to use is chosen per measure by held-out Gaussian log-likelihood among those whose held-out whitened power is within 1 +- 0.15 (choose_whitener; probability-atlas 2026-10-05 rule).
- **Implementation:** `source_localization.locate.whitening:NoiseWhitener`
- **Tests:** `tests/test_locate.py`

### `whiten.oas` (version 1)

OAS shrinkage covariance shape, per-recording scale.

- **Model:** Data and leadfields are projected onto the n-1 average-reference directions (the reference electrode drops out). Each recording's noise scale is the RMS of its referenced noise samples; the spatial covariance shape C is pooled over recordings from their scale-normalised samples, estimated by scikit-learn's oracle approximating shrinkage (OAS; zero-mean samples). Whitener W = C^(-1/2) B'; a recording's pattern y becomes W y / scale. Noise samples are, for trial averages, (odd-trial average - even-trial average) / 2, so their variance is the noise in the full average; complex steady-state noise enters as [Re, Im] columns.
- **Values:** estimator "oas"
- **Validity:** Assumes every recording shares one noise shape and differs only in size. Which estimator to use is chosen per measure by held-out Gaussian log-likelihood among those whose held-out whitened power is within 1 +- 0.15 (choose_whitener; probability-atlas 2026-10-05 rule). Used by the field method for the frozen Sentinel fields (surface-evoked localize_fields_bayes.py).
- **Implementation:** `source_localization.locate.whitening:NoiseWhitener`
- **Tests:** `tests/test_locate.py`

### `whiten.diagonal` (version 1)

Independent electrode noise, then referenced.

- **Model:** Data and leadfields are projected onto the n-1 average-reference directions (the reference electrode drops out). Each recording's noise scale is the RMS of its referenced noise samples; the spatial covariance shape C is pooled over recordings from their scale-normalised samples, estimated by C = B' diag(d) B, d the per-electrode mean square of the samples (independent noise per electrode). Whitener W = C^(-1/2) B'; a recording's pattern y becomes W y / scale. Noise samples are, for trial averages, (odd-trial average - even-trial average) / 2, so their variance is the noise in the full average; complex steady-state noise enters as [Re, Im] columns.
- **Values:** estimator "diagonal"
- **Validity:** Assumes every recording shares one noise shape and differs only in size. Which estimator to use is chosen per measure by held-out Gaussian log-likelihood among those whose held-out whitened power is within 1 +- 0.15 (choose_whitener; probability-atlas 2026-10-05 rule).
- **Implementation:** `source_localization.locate.whitening:NoiseWhitener`
- **Tests:** `tests/test_locate.py`

## Field definitions

| id | version | what it is |
|---|---|---|
| `field.response_svd` | 1 | leading spatial singular vectors of a group-mean evoked response |
| `field.phase_aligned` | 1 | real pattern of a steady-state response, phase-aligned across recordings |
| `field.phase_free` | 1 | leading eigenvector of the phase-free steady-state cross-product |
| `field.csd_leading` | 1 | leading eigenvector of the mean trace-normalised cross-spectral density |

### `field.response_svd` (version 1)

Leading spatial singular vectors of a group-mean evoked response.

- **Model:** Per unit (for example a mouse-day): the mean of its recordings' trial averages, scaled to unit Frobenius norm over the time span, so every unit weighs the same. The fields are the first k left singular vectors of the units' mean over that span (orthogonal by construction).
- **Values:** time span (Sentinel chirp: 5-300 ms), k fields (Sentinel chirp: 2)
- **Validity:** Electrode space only: no head model. A field is the data's pattern, not a generator; where it comes from is answered by locating it (locate.*). Its sign is fixed by the template's sign rule. Reproduces Sentinel chirp_D and chirp_U2 (surface-evoked sentinel_fields_v1.json) to 1e-16 from their defining data.
- **Caveat:** Successive components are orthogonal; physiological fields may be mixtures of them.
- **Implementation:** `source_localization.field.define:response_svd`
- **Tests:** `tests/test_field.py`
- **Reference:** surface-evoked troubleshooting log 67; docs/methods/FIELD_TEMPLATES.md

### `field.phase_aligned` (version 1)

Real pattern of a steady-state response, phase-aligned across recordings.

- **Model:** Complex coefficients at the driving rate per recording. Iterate: normalise the pattern D; rotate each recording onto D's phase and scale it to unit norm; average within units; D = real part of the units' mean. The result depends on the starting pattern only through which phase it aligns to.
- **Values:** starting pattern, iterations (Sentinel ASSR: 3, from log 66's group pattern)
- **Validity:** Electrode space only: no head model. A field is the data's pattern, not a generator; where it comes from is answered by locating it (locate.*). Its sign is fixed by the template's sign rule. Reproduces Sentinel assr40_D and assr80_D exactly from their defining data.
- **Implementation:** `source_localization.field.define:phase_aligned_pattern`
- **Tests:** `tests/test_field.py`
- **Reference:** surface-evoked troubleshooting log 69; docs/methods/FIELD_TEMPLATES.md

### `field.phase_free` (version 1)

Leading eigenvector of the phase-free steady-state cross-product.

- **Model:** The leading eigenvector of sum_r w_r Re(z_r z_r^H) / |z_r|^2 over recordings' complex patterns z_r: no starting pattern and no iteration.
- **Values:** recording weights w (default equal)
- **Validity:** Electrode space only: no head model. A field is the data's pattern, not a generator; where it comes from is answered by locating it (locate.*). Its sign is fixed by the template's sign rule.
- **Implementation:** `source_localization.field.define:phase_free_pattern`
- **Tests:** `tests/test_field.py`
- **Reference:** surface-evoked method_crossover.FieldModel.decompose (ASSR)

### `field.csd_leading` (version 1)

Leading eigenvector of the mean trace-normalised cross-spectral density.

- **Model:** For an ongoing band: each subject's band-integrated cross-spectral matrix (real part) divided by its trace, averaged with weights; the field is the leading eigenvector.
- **Values:** band (bayes-sl L2/L3: 30-55 Hz), subject weights (default equal)
- **Validity:** Electrode space only: no head model. A field is the data's pattern, not a generator; where it comes from is answered by locating it (locate.*). Its sign is fixed by the template's sign rule.
- **Implementation:** `source_localization.field.define:csd_leading_pattern`
- **Tests:** `tests/test_field.py`
- **Reference:** surface-evoked method_crossover.FieldModel.decompose (rest); bayes-sl reports/2026-11-04_l3_plan.md

## Field projections

| id | version | what it is |
|---|---|---|
| `field.model_filter_pair` | 1 | the electrode field of a located mirror pair, as a spatial filter |
| `field.project_inner` | 1 | each field separately: the inner product with the recording |
| `field.project_joint` | 1 | all selected fields at once, by least squares |

### `field.model_filter_pair` (version 1)

The electrode field of a located mirror pair, as a spatial filter.

- **Model:** Members at the grid positions nearest the located right member and its mirror image. Under each head model (average-referenced leadfields) one mirrored moment (left = right mirrored in x) is fitted to the target field by least squares; each model's pattern is normalised and signed to agree with the target; the filter is their mean, normalised. Projected like a template field.
- **Values:** `prior.head_ensemble_pa_phase2`, target field (e.g. chirp_D), located right member (mm)
- **Validity:** Built for Sentinel's chirp pair (surface-evoked H038): cos 0.990 with chirp_D, 0.912 with assr40_D.
- **Implementation:** `source_localization.locate.field_model:model_filter_pair`
- **Tests:** `tests/test_field_locate.py`
- **Reference:** surface-evoked readout_filters.py (templates/h038_model_filter.json); docs/methods/FIELD_LOCATION.md

### `field.project_inner` (version 1)

Each field separately: the inner product with the recording.

- **Model:** Epochs average-referenced over the template's channels, picked by name in the template's order; virtual channel s_k(t) = u_k . x(t) for each field u_k. Residual: x - P x with P the orthogonal projector onto the span of the selected fields, stored as its trial average.
- **Values:** fields: all in the template by default, residual of: the projected fields by default
- **Validity:** Reproduces surface-evoked's per-recording field series (rerun070_fields.py) for the Sentinel held-out roster (field-method plan, F3).
- **Caveat:** Overlapping fields share signal: each series also carries what it shares with the others (Sentinel chirp_D and assr40_D are not orthogonal).
- **Implementation:** `source_localization.field.project:project`
- **Tests:** `tests/test_field.py`
- **Reference:** docs/methods/FIELD_TEMPLATES.md; surface-evoked troubleshooting log 70

### `field.project_joint` (version 1)

All selected fields at once, by least squares.

- **Model:** As field.project_inner, but the series are the least-squares coefficients of the selected fields jointly: s(t) = (M M^T)^-1 M x(t). Equal to the inner projection when the fields are orthogonal. Same residual.
- **Values:** fields: all in the template by default
- **Validity:** Not yet used in a study.
- **Implementation:** `source_localization.field.project:project`
- **Tests:** `tests/test_field.py`
- **Reference:** docs/methods/FIELD_TEMPLATES.md
