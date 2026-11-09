# Model catalogue

Every probability model this package uses to say where a signal comes from: the location posteriors, the priors they rest on, and the calibrations applied to their maps. Each has a stable **id** and a **version**; results name the ids and versions that produced them.

*Generated from `source_localization/methods.py` (`python -m source_localization.methods > docs/methods/CATALOGUE.md`); do not edit by hand. `tests/test_catalogue.py` checks it is current.*

## Location posteriors

| id | version | what it is |
|---|---|---|
| `locate.single_source_exact` | 1 | one dipole, exact head model |
| `locate.single_source` | 1 | one dipole, averaged over a head-model ensemble |
| `locate.mirror_pair` | 1 | a dipole and its mirror image across the midline |
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
- **Implementation:** `source_localization.validation.ensemble_posterior:EnsemblePosterior`
- **Tests:** `tests/test_ensemble_posterior.py`
- **Reference:** module docstring of source_localization.validation.ensemble_posterior; bayes-sl reports/TB_map_calibration.md

### `locate.mirror_pair` (version 1)

A dipole and its mirror image across the midline.

- **Model:** locate.source_set with configurations (x, x mirrored across the midline x = midline_x_mm), x on the right at least min_offset_mm (default 0.3 mm) from it, mirrors snapped to the grid. Each member has an independent free-orientation moment. position_map gives each member half the mass of its configuration (a map summing to 2 counts both members).
- **Rests on:** `prior.position_uniform`, `prior.moment_empirical_bayes`, `prior.head_ensemble_pa_phase2`
- **Validity:** As locate.single_source. The single-source calibration carries over to pair maps on MEA30 (90% region: 92-93% coverage for auditory-cortex pairs, 81-83% for random pairs; bayes-sl reports/TB_mirror_pair.md).
- **Caveat:** Pairs very close to the midline look like one central source; at high SNR a single deep source can pass for a near-midline pair (bayes-sl tb16 control).
- **Implementation:** `source_localization.validation.ensemble_posterior:MirrorPairPosterior`
- **Tests:** `tests/test_ensemble_posterior.py`
- **Reference:** bayes-sl reports/TB_mirror_pair.md

### `locate.source_set` (version 1)

Any fixed set of dipoles, with model evidence.

- **Model:** A configuration q places d/3 free-orientation dipoles with stacked whitened leadfield G_q. Per column: y_j | q, k ~ N(0, I + tau^2 G_qk G_qk^T), each moment ~ N(0, tau^2 I3) independently, averaged over head models. log_evidence averages the likelihood over the set's configurations, so different sets can be compared.
- **Rests on:** `prior.position_uniform`, `prior.moment_empirical_bayes`, `prior.head_ensemble_pa_phase2`
- **Validity:** As locate.single_source; calibration checked for one source and for mirror pairs only.
- **Implementation:** `source_localization.validation.ensemble_posterior:SourceSetPosterior`
- **Tests:** `tests/test_ensemble_posterior.py`
- **Reference:** module docstring of source_localization.validation.ensemble_posterior

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
- **Implementation:** `source_localization.validation.ensemble_posterior:SourceSetPosterior`

### `prior.moment_empirical_bayes` (version 1)

Dipole moments Gaussian, scale estimated per pattern.

- **Model:** Each dipole moment ~ N(0, tau^2 I3). tau^2 is an empirical-Bayes estimate per data pattern: max(||y||^2 / c - s m, 1e-3 m) / median_x ||G_x,nominal||_F^2 (c columns, m channels, noise variance s = 1 after whitening). With several dipoles it is a per-source variance.
- **Values:** tau^2 from the data, not a fixed prior; floor 1e-3 m
- **Implementation:** `source_localization.validation.ensemble_posterior:SourceSetPosterior.moment_variance`
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
- **Implementation:** `external:probability-atlas pa/ensemble.py draw_ensemble, ensemble_leadfields; scripts/phase2_simulate.py VARIANTS["primary"] (moves here in phase F1)`
- **Reference:** source_localization.validation.head_models:HeadModelPrior has the same parameters

### `prior.model_odds_equal` (version 1)

Equal prior odds between source sets.

- **Model:** P(set | y) from the sets' log evidences (locate.source_set.log_evidence) with equal prior probability for each set compared (for example one source vs a mirror pair).
- **Values:** equal prior probability per set
- **Validity:** P(pair) roughly calibrated, slightly overconfident, on MEA30 (patterns with P(pair) 0.95-1.0 were pairs 94.5% of the time; bayes-sl reports/TB_mirror_pair.md).
- **Implementation:** `source_localization.validation.ensemble_posterior:model_probabilities`

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
- **Implementation:** `source_localization.validation.ensemble_posterior:UNCALIBRATED`

### `calibration.volumetric` (version 1)

Tempered and blurred, for volumetric maps.

- **Model:** Normalized log posterior divided by T = 1.5, then an isotropic Gaussian blur of 0.25 mm over the candidate positions (mass-conserving).
- **Values:** temperature 1.5, blur 0.25 mm
- **Validity:** MEA30, prior.head_ensemble_pa_phase2, 0.5 mm volumetric grid, single-source truths, whitened real noise. 90% region held the truth 88-89% of the time at half, primary and double SNR (80%: 79-81%; 95%: 92-94%). Chosen on one simulated set, checked on another.
- **Implementation:** `source_localization.validation.ensemble_posterior:VOLUMETRIC`
- **Tests:** `tests/test_ensemble_posterior.py`
- **Reference:** bayes-sl reports/TB_map_calibration.md

### `calibration.surface` (version 1)

Tempered, for the cortical surface readout.

- **Model:** Normalized log posterior divided by T = 2.5; no blur (blurring made it worse).
- **Values:** temperature 2.5, blur 0 mm
- **Validity:** MEA30, prior.head_ensemble_pa_phase2, cortical/cerebellar/olfactory surface positions. Approximately calibrated: 90% region 88-91% at half and primary SNR, 83-85% at double SNR.
- **Implementation:** `source_localization.validation.ensemble_posterior:SURFACE`
- **Tests:** `tests/test_ensemble_posterior.py`
- **Reference:** bayes-sl reports/TB_map_calibration.md
