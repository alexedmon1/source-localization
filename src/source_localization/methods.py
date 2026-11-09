"""The method vocabulary: what this package can actually be asked to do.

One declaration per choice, so the CLI's ``choices=``, its ``list`` output and
the documentation all read from the same place. ``tests/test_methods.py`` checks
each name against the dispatcher that has to accept it, so a method added to
``steps/inverse_solution.py`` and not here (or the reverse) fails the suite
rather than going unlisted -- ``--method`` offered three of the six inverse
methods for several releases because the list was written out by hand.

A *pipeline* is four independent choices: a head model (BEM), where the sources
go (source space), how densely they are placed (sampling), and how the inverse
is solved. The atlas is a fifth, and lives in ``data/atlas/registry.yaml``.
"""

from __future__ import annotations

from typing import Dict, NamedTuple


class Method(NamedTuple):
    """One selectable option, with what a user needs to choose between them."""

    name: str
    summary: str
    #: Longer note: when to prefer it, or what it is known not to do.
    detail: str = ""
    #: True for the value presets use unless they say otherwise.
    default: bool = False
    #: Set when the option carries a caveat that changes how results are read.
    caveat: str = ""


# --- Head models -----------------------------------------------------------

BEM_TYPES: Dict[str, Method] = {
    'sphere': Method(
        'sphere', '3-layer analytical sphere',
        'Closed-form, so it is fast and never fails to converge. The mouse head '
        'is not spherical, and roi_based source spaces lose sources to it.'),
    'ellipsoid': Method(
        'ellipsoid', 'numerical ellipsoid BEM',
        'Fits the mouse head; ~1.5 mm localization error in simulation. '
        'Slower to build, and cached under bem_cache/.', default=True),
}


# --- Where the sources go --------------------------------------------------

SOURCE_SPACES: Dict[str, Method] = {
    'surface': Method(
        'surface', 'cortical mid-ribbon mesh, or an icosphere',
        "source_space.surface.method: 'anatomical' (default) cuts the ribbon "
        "from Allen32 and carries fixed orientation; 'icosphere' is the older "
        "geometric shell.", default=True),
    'roi_based': Method(
        'roi_based', 'one source per atlas ROI, at its PCA centroid',
        'The conventional comparator. Excluded from Monte Carlo sampling: its '
        'placement is the method, so there is no dense pool to draw from.'),
    'cartesian': Method(
        'cartesian', 'regular 3D grid through the brain volume',
        'Grid phase is anchored to voxel 0 of the volume array, a bounding-box '
        'corner with no anatomical meaning.',
        caveat='Sweeping that offset across one cell moves parcel coverage '
               'between 29 and 32. Prefer monte_carlo sampling here.'),
    'shell': Method(
        'shell', 'concentric geometry-matched shells',
        'Depth-stratified sampling that follows the head shape.'),
    'hybrid': Method(
        'hybrid', 'anatomical surface plus a volume grid for deep structures',
        'Cortex, cerebellum and olfactory bulb on the mid-ribbon at their '
        'normals; thalamus, tectum, hippocampus, basal ganglia, hypothalamus '
        'and amygdala on a Cartesian grid with free orientation. Gives a deep '
        'generator somewhere to go other than the nearest cortex.',
        caveat='Monte Carlo sampling only. On a dorsal array deep and cortical '
               'electrode patterns are near-collinear, so how a response is '
               'split between the two parts is set largely by the inverse.'),
}


# --- How densely, and at which placement -----------------------------------

SAMPLING_MODES: Dict[str, Method] = {
    'fixed': Method(
        'fixed', 'solve one source grid',
        'Produces vertex-level output as well as parcel series. The grid is one '
        'arbitrary placement of a continuous current distribution.',
        default=True),
    'monte_carlo': Method(
        'monte_carlo', 'average the ROI operator over K sparse draws',
        'Integrates over source placement instead of committing to one grid. '
        'Nearly free: the parcel series is linear in the sensor data, so K '
        'operators are averaged and applied once. Works with surface, '
        'cartesian, shell and hybrid.',
        caveat='ROI-only by construction — no single grid is solved, so the run '
               'has no vertex output and skip_roi_extraction raises.'),
    'parcel_subspace': Method(
        'parcel_subspace', "one inverse over each parcel's leadfield patterns",
        'Builds the same dense pool as monte_carlo, represents each parcel by the '
        'top n_patterns patterns of its pool leadfield, and solves one MNE (or '
        'sLORETA) inverse over all of them. Deterministic: no draws, seed or sign '
        'alignment. In simulation it localized about 1 mm closer than the deployed '
        'monte_carlo operator (surface-evoked troubleshooting log, section 24).',
        caveat='ROI-only, like monte_carlo. Which per-parcel component series to '
               'use for evoked measures is provisional (dominant pattern as signed, '
               'root-sum-square as magnitude); all components are saved.'),
}


# --- How the inverse is solved ---------------------------------------------

INVERSE_METHODS: Dict[str, Method] = {
    'sLORETA': Method(
        'sLORETA', 'standardized low-resolution tomography',
        'The default in every preset, and the most robust to mouse-scale '
        'leadfields.', default=True),
    'dSPM': Method(
        'dSPM', 'noise-normalized minimum norm'),
    'MNE': Method(
        'MNE', 'raw minimum norm'),
    'eLORETA': Method(
        'eLORETA', 'exact low-resolution tomography',
        'Custom iterative implementation; slower than the three above.'),
    'LCMV': Method(
        'LCMV', 'linearly constrained minimum variance beamformer (stock MNE)',
        caveat='Not covered by the mouse-scale regularization fix that the four '
               'minimum-norm methods above get.'),
    'DICS': Method(
        'DICS', 'dynamic imaging of coherent sources (stock MNE)',
        'Frequency-domain beamformer.',
        caveat='One power value per source, not a time course: written as CSV, '
               'and it fails on the surface presets. Not covered by the '
               'mouse-scale regularization fix.'),
}


def default_of(options: Dict[str, Method]) -> str | None:
    """The name marked default, or None when the choice has no default."""
    return next((m.name for m in options.values() if m.default), None)


def choices(options: Dict[str, Method]) -> list:
    """Names in declaration order, for ``argparse(choices=...)``."""
    return list(options)


# === The model catalogue ====================================================
#
# Every probability model this package uses to say *where* something is: the
# location posteriors, each prior they rest on, the calibrations applied to
# their maps, and the noise whitening they assume. One entry per model, with a stable identifier and a version, so a
# result can name exactly what produced it and a reader can find its
# assumptions. The rules (bayes-sl reports/2026-11-09_field_method_library_plan.md,
# "Identification and documentation"):
#
# - ``id`` never changes once released; ``version`` goes up whenever what the
#   model computes changes (not for docs or refactors).
# - ``priors`` lists the ids of the priors a posterior rests on; each is its own
#   entry, with its values.
# - ``validity`` says where the model, or its calibration, has been checked. A
#   calibration is valid only inside that domain.
# - ``implementation`` is ``module:object`` importable from this package, or
#   ``external:<where>`` for code that has not moved here yet (with the phase
#   that moves it).
# - ``docs/methods/CATALOGUE.md`` is generated from these entries
#   (``python -m source_localization.methods > docs/methods/CATALOGUE.md``);
#   ``tests/test_catalogue.py`` fails when it is stale, when an id repeats, when
#   a prior id is not in the catalogue, or when an implementation does not import.


class ModelSpec(NamedTuple):
    """One catalogued model: what it computes, what it assumes, where it holds."""

    id: str
    version: int
    summary: str
    #: The model: likelihood, generative statement or procedure.
    model: str
    #: Ids of the priors it rests on (each a catalogue entry), or, for a prior,
    #: its values.
    priors: tuple = ()
    #: Where it has been checked; for a calibration, the only domain it holds in.
    validity: str = ""
    implementation: str = ""
    tests: str = ""
    reference: str = ""
    caveat: str = ""


_EP = 'source_localization.locate.ensemble_posterior'

LOCATION_POSTERIORS: Dict[str, ModelSpec] = {
    'locate.single_source_exact': ModelSpec(
        'locate.single_source_exact', 1,
        'one dipole, exact head model',
        'Single free-orientation dipole at grid position x, forward model taken as '
        'exact, white sensor noise of known SD sigma. Default: the moment is profiled '
        'out, log p(B | x) = -||B - Q_x Q_x^T B||^2 / (2 sigma^2). With a moment '
        'variance given, it is marginalized instead: B | x ~ N(0, sigma^2 I + tau^2 '
        'G_x G_x^T).',
        priors=('prior.position_uniform', 'prior.moment_given_sd'),
        validity='Coverage checked in simulation with the true head model '
                 '(DipolePosterior.coverage_test). The fundamental limit for one '
                 'dipole, not the pipeline estimator.',
        implementation='source_localization.validation.posterior:DipolePosterior',
        tests='tests/test_posterior.py',
        reference='docs/validation/methods/SOURCE_CERTAINTY.md',
        caveat='Assumes the head model is exact; for real data use '
               'locate.single_source. Profiling is overconfident once noise is '
               'comparable to the signal (at 0 dB its 68% region covered 53% of the '
               'time); the marginalized form stays calibrated.'),
    'locate.single_source': ModelSpec(
        'locate.single_source', 1,
        'one dipole, averaged over a head-model ensemble',
        'Per whitened data column j: y_j | x, k ~ N(0, I + tau^2 G_xk G_xk^T), the '
        'moment marginalized. The likelihood is averaged over the K head models '
        '(not the posteriors), so the head model is marginalized: p(x | y) is '
        'proportional to (1/K) sum_k p(y | x, k). The one-dipole case of '
        'locate.source_set.',
        priors=('prior.position_uniform', 'prior.moment_empirical_bayes',
                'prior.head_ensemble_pa_phase2'),
        validity='Inputs must be whitened (the caller fits the whitener). Raw maps '
                 'are overconfident; use a calibration from this catalogue inside its '
                 'domain. Calibrated for single-source truths only.',
        implementation=f'{_EP}:EnsemblePosterior',
        tests='tests/test_ensemble_posterior.py',
        reference=f'module docstring of {_EP}; bayes-sl reports/TB_map_calibration.md',
        caveat='A response with more than one source is read as one confident '
               'point between them; compare with locate.mirror_pair or '
               'locate.source_set through prior.model_odds_equal.'),
    'locate.mirror_pair': ModelSpec(
        'locate.mirror_pair', 1,
        'a dipole and its mirror image across the midline',
        'locate.source_set with configurations (x, x mirrored across the midline '
        'x = midline_x_mm), x on the right at least min_offset_mm (default 0.3 mm) '
        'from it, mirrors snapped to the grid. Each member has an independent '
        'free-orientation moment. position_map gives each member half the mass of '
        'its configuration (a map summing to 2 counts both members).',
        priors=('prior.position_uniform', 'prior.moment_empirical_bayes',
                'prior.head_ensemble_pa_phase2'),
        validity='As locate.single_source. The single-source calibration carries over '
                 'to pair maps on MEA30 (90% region: 92-93% coverage for auditory-cortex '
                 'pairs, 81-83% for random pairs; bayes-sl reports/TB_mirror_pair.md).',
        implementation=f'{_EP}:MirrorPairPosterior',
        tests='tests/test_ensemble_posterior.py',
        reference='bayes-sl reports/TB_mirror_pair.md',
        caveat='Pairs very close to the midline look like one central source; at '
               'high SNR a single deep source can pass for a near-midline pair '
               '(bayes-sl tb16 control).'),
    'locate.source_set': ModelSpec(
        'locate.source_set', 1,
        'any fixed set of dipoles, with model evidence',
        'A configuration q places d/3 free-orientation dipoles with stacked whitened '
        'leadfield G_q. Per column: y_j | q, k ~ N(0, I + tau^2 G_qk G_qk^T), each '
        'moment ~ N(0, tau^2 I3) independently, averaged over head models. '
        'log_evidence averages the likelihood over the set\'s configurations, so '
        'different sets can be compared.',
        priors=('prior.position_uniform', 'prior.moment_empirical_bayes',
                'prior.head_ensemble_pa_phase2'),
        validity='As locate.single_source; calibration checked for one source and '
                 'for mirror pairs only.',
        implementation=f'{_EP}:SourceSetPosterior',
        tests='tests/test_ensemble_posterior.py',
        reference=f'module docstring of {_EP}'),
}

LOCATION_PRIORS: Dict[str, ModelSpec] = {
    'prior.position_uniform': ModelSpec(
        'prior.position_uniform', 1,
        'uniform over candidate positions or configurations',
        'p(x) constant over the grid points inside the brain (or over a source '
        'set\'s configurations).',
        priors=('uniform; the grid spacing sets its resolution (0.5 mm by default)',),
        implementation=f'{_EP}:SourceSetPosterior',
        caveat='Because gain falls with depth, deep sources get broad posteriors '
               'under it; that is a statement about the data, not the prior.'),
    'prior.moment_empirical_bayes': ModelSpec(
        'prior.moment_empirical_bayes', 1,
        'dipole moments Gaussian, scale estimated per pattern',
        'Each dipole moment ~ N(0, tau^2 I3). tau^2 is an empirical-Bayes estimate '
        'per data pattern: max(||y||^2 / c - s m, 1e-3 m) / median_x '
        '||G_x,nominal||_F^2 (c columns, m channels, noise variance s = 1 after '
        'whitening). With several dipoles it is a per-source variance.',
        priors=('tau^2 from the data, not a fixed prior; floor 1e-3 m',),
        implementation=f'{_EP}:SourceSetPosterior.moment_variance',
        reference='probability-atlas pa.posterior (the reference implementation)'),
    'prior.moment_given_sd': ModelSpec(
        'prior.moment_given_sd', 1,
        'dipole moment Gaussian with a given SD, or profiled out',
        'With moment_std given: the moment ~ N(0, moment_std^2 I3) and is '
        'marginalized. Without it: no prior; the moment is profiled out (its '
        'maximum-likelihood value is used).',
        priors=('moment_std set by the caller; None profiles the moment out',),
        implementation='source_localization.validation.posterior:DipolePosterior.posterior',
        tests='tests/test_posterior.py'),
    'prior.head_ensemble_pa_phase2': ModelSpec(
        'prior.head_ensemble_pa_phase2', 1,
        'head-model uncertainty: the 33-model ensemble',
        'The nominal head model plus 32 draws from HeadModelPrior: electrode-'
        'registration shift ~ N(0, 0.3 mm) per axis, skull-conductivity factor '
        'log-uniform on [0.5, 2.0]. Draws from numpy default_rng([20261005, 1, 0]); '
        'leadfields on the 0.5 mm volumetric grid, unreferenced as MNE returns them.',
        priors=('shift SD 0.3 mm (x, y, z)', 'skull factor log-uniform 0.5-2.0',
                'K = 1 nominal + 32 draws', 'seed [20261005, 1, 0]'),
        validity='MEA30 on the ellipsoid BEM. Every calibration in this catalogue '
                 'was measured with this ensemble.',
        implementation='source_localization.locate.head_ensemble:ENSEMBLE_PA_PHASE2',
        tests='tests/test_locate.py',
        reference='Ported from probability-atlas Phase 2 (pa/ensemble.py, scripts/phase2_simulate.py '
                  'VARIANTS["primary"], K_ENS 32, rng_for(1, 0)); draws identical, leadfields equal to '
                  'a relative 2e-16 on the Sentinel grid (field-method plan, F1)'),
    'prior.model_odds_equal': ModelSpec(
        'prior.model_odds_equal', 1,
        'equal prior odds between source sets',
        'P(set | y) from the sets\' log evidences (locate.source_set.log_evidence) '
        'with equal prior probability for each set compared (for example one '
        'source vs a mirror pair).',
        priors=('equal prior probability per set',),
        validity='P(pair) roughly calibrated, slightly overconfident, on MEA30 '
                 '(patterns with P(pair) 0.95-1.0 were pairs 94.5% of the time; '
                 'bayes-sl reports/TB_mirror_pair.md).',
        implementation=f'{_EP}:model_probabilities'),
}

MAP_CALIBRATIONS: Dict[str, ModelSpec] = {
    'calibration.none': ModelSpec(
        'calibration.none', 1,
        'the raw averaged posterior',
        'Temperature 1, no blur.',
        priors=('temperature 1.0', 'blur 0 mm'),
        implementation=f'{_EP}:UNCALIBRATED',
        caveat='Overconfident under head-model error: the 90% region held the truth '
               'about 55-65% of the time on MEA30 (bayes-sl reports/TB_map_focality.md).'),
    'calibration.volumetric': ModelSpec(
        'calibration.volumetric', 1,
        'tempered and blurred, for volumetric maps',
        'Normalized log posterior divided by T = 1.5, then an isotropic Gaussian '
        'blur of 0.25 mm over the candidate positions (mass-conserving).',
        priors=('temperature 1.5', 'blur 0.25 mm'),
        validity='MEA30, prior.head_ensemble_pa_phase2, 0.5 mm volumetric grid, '
                 'single-source truths, whitened real noise. 90% region held the truth '
                 '88-89% of the time at half, primary and double SNR (80%: 79-81%; '
                 '95%: 92-94%). Chosen on one simulated set, checked on another.',
        implementation=f'{_EP}:VOLUMETRIC',
        tests='tests/test_ensemble_posterior.py',
        reference='bayes-sl reports/TB_map_calibration.md'),
    'calibration.surface': ModelSpec(
        'calibration.surface', 1,
        'tempered, for the cortical surface readout',
        'Normalized log posterior divided by T = 2.5; no blur (blurring made it worse).',
        priors=('temperature 2.5', 'blur 0 mm'),
        validity='MEA30, prior.head_ensemble_pa_phase2, cortical/cerebellar/olfactory '
                 'surface positions. Approximately calibrated: 90% region 88-91% at half '
                 'and primary SNR, 83-85% at double SNR.',
        implementation=f'{_EP}:SURFACE',
        tests='tests/test_ensemble_posterior.py',
        reference='bayes-sl reports/TB_map_calibration.md'),
}

_WH = 'source_localization.locate.whitening:NoiseWhitener'
_WH_MODEL = ('Data and leadfields are projected onto the n-1 average-reference directions (the reference electrode '
             'drops out). Each recording\'s noise scale is the RMS of its referenced noise samples; the spatial '
             'covariance shape C is pooled over recordings from their scale-normalised samples, estimated by {est}. '
             'Whitener W = C^(-1/2) B\'; a recording\'s pattern y becomes W y / scale. Noise samples are, for '
             'trial averages, (odd-trial average - even-trial average) / 2, so their variance is the noise in the '
             'full average; complex steady-state noise enters as [Re, Im] columns.')
_WH_VALIDITY = ('Assumes every recording shares one noise shape and differs only in size. Which estimator to use '
                'is chosen per measure by held-out Gaussian log-likelihood among those whose held-out whitened '
                'power is within 1 +- 0.15 (choose_whitener; probability-atlas 2026-10-05 rule).')

WHITENERS: Dict[str, ModelSpec] = {
    'whiten.sample': ModelSpec(
        'whiten.sample', 1, 'pooled sample covariance shape, per-recording scale',
        _WH_MODEL.format(est='the plain sample covariance'),
        priors=('estimator "sample"',), validity=_WH_VALIDITY, implementation=_WH, tests='tests/test_locate.py',
        caveat='Amplifies noise it was not fitted on (held-out whitened power 2.2x for ASSR, 15-24x for chirp '
               'peaks in probability-atlas Phase 1); prefer a shrinkage estimator.'),
    'whiten.ledoit_wolf': ModelSpec(
        'whiten.ledoit_wolf', 1, 'Ledoit-Wolf shrinkage covariance shape, per-recording scale',
        _WH_MODEL.format(est="scikit-learn's Ledoit-Wolf shrinkage (zero-mean samples)"),
        priors=('estimator "ledoit_wolf"',), validity=_WH_VALIDITY, implementation=_WH,
        tests='tests/test_locate.py'),
    'whiten.oas': ModelSpec(
        'whiten.oas', 1, 'OAS shrinkage covariance shape, per-recording scale',
        _WH_MODEL.format(est="scikit-learn's oracle approximating shrinkage (OAS; zero-mean samples)"),
        priors=('estimator "oas"',),
        validity=_WH_VALIDITY + ' Used by the field method for the frozen Sentinel fields '
                                '(surface-evoked localize_fields_bayes.py).',
        implementation=_WH, tests='tests/test_locate.py'),
    'whiten.diagonal': ModelSpec(
        'whiten.diagonal', 1, 'independent electrode noise, then referenced',
        _WH_MODEL.format(est="C = B' diag(d) B, d the per-electrode mean square of the samples "
                             "(independent noise per electrode)"),
        priors=('estimator "diagonal"',), validity=_WH_VALIDITY, implementation=_WH,
        tests='tests/test_locate.py'),
}

_FD = 'source_localization.field.define'
_FD_VALIDITY = ('Electrode space only: no head model. A field is the data\'s pattern, not a generator; where it '
                'comes from is answered by locating it (locate.*). Its sign is fixed by the template\'s sign rule.')

FIELD_DEFINITIONS: Dict[str, ModelSpec] = {
    'field.response_svd': ModelSpec(
        'field.response_svd', 1, 'leading spatial singular vectors of a group-mean evoked response',
        'Per unit (for example a mouse-day): the mean of its recordings\' trial averages, scaled to unit Frobenius '
        'norm over the time span, so every unit weighs the same. The fields are the first k left singular vectors '
        'of the units\' mean over that span (orthogonal by construction).',
        priors=('time span (Sentinel chirp: 5-300 ms)', 'k fields (Sentinel chirp: 2)'),
        validity=_FD_VALIDITY + ' Reproduces Sentinel chirp_D and chirp_U2 (surface-evoked '
                                'sentinel_fields_v1.json) to 1e-16 from their defining data.',
        implementation=f'{_FD}:response_svd', tests='tests/test_field.py',
        reference='surface-evoked troubleshooting log 67; docs/methods/FIELD_TEMPLATES.md',
        caveat='Successive components are orthogonal; physiological fields may be mixtures of them.'),
    'field.phase_aligned': ModelSpec(
        'field.phase_aligned', 1, 'real pattern of a steady-state response, phase-aligned across recordings',
        'Complex coefficients at the driving rate per recording. Iterate: normalise the pattern D; rotate each '
        'recording onto D\'s phase and scale it to unit norm; average within units; D = real part of the units\' '
        'mean. The result depends on the starting pattern only through which phase it aligns to.',
        priors=('starting pattern', 'iterations (Sentinel ASSR: 3, from log 66\'s group pattern)'),
        validity=_FD_VALIDITY + ' Reproduces Sentinel assr40_D and assr80_D exactly from their defining data.',
        implementation=f'{_FD}:phase_aligned_pattern', tests='tests/test_field.py',
        reference='surface-evoked troubleshooting log 69; docs/methods/FIELD_TEMPLATES.md'),
    'field.phase_free': ModelSpec(
        'field.phase_free', 1, 'leading eigenvector of the phase-free steady-state cross-product',
        'The leading eigenvector of sum_r w_r Re(z_r z_r^H) / |z_r|^2 over recordings\' complex patterns z_r: no '
        'starting pattern and no iteration.',
        priors=('recording weights w (default equal)',),
        validity=_FD_VALIDITY,
        implementation=f'{_FD}:phase_free_pattern', tests='tests/test_field.py',
        reference='surface-evoked method_crossover.FieldModel.decompose (ASSR)'),
    'field.csd_leading': ModelSpec(
        'field.csd_leading', 1, 'leading eigenvector of the mean trace-normalised cross-spectral density',
        'For an ongoing band: each subject\'s band-integrated cross-spectral matrix (real part) divided by its trace, '
        'averaged with weights; the field is the leading eigenvector.',
        priors=('band (bayes-sl L2/L3: 30-55 Hz)', 'subject weights (default equal)'),
        validity=_FD_VALIDITY,
        implementation=f'{_FD}:csd_leading_pattern', tests='tests/test_field.py',
        reference='surface-evoked method_crossover.FieldModel.decompose (rest); bayes-sl reports/2026-11-04_l3_plan.md'),
}

#: The catalogue's sections, in the order the generated page lists them.
CATALOGUE: Dict[str, Dict[str, ModelSpec]] = {
    'Location posteriors': LOCATION_POSTERIORS,
    'Priors of the location posteriors': LOCATION_PRIORS,
    'Map calibrations': MAP_CALIBRATIONS,
    'Noise whitening': WHITENERS,
    'Field definitions': FIELD_DEFINITIONS,
}


def catalogue_entries() -> Dict[str, ModelSpec]:
    """Every catalogued model by id, across sections."""
    return {k: v for section in CATALOGUE.values() for k, v in section.items()}


def catalogue_markdown() -> str:
    """The catalogue as the Markdown page docs/methods/CATALOGUE.md."""
    out = ['# Model catalogue', '',
           'Every probability model this package uses to say where a signal comes from: the '
           'location posteriors, the priors they rest on, the calibrations applied to their '
           'maps, the noise whitening they assume, and the definitions of the electrode fields they locate. Each has a stable **id** and a **version**; results name the ids and versions '
           'that produced them.', '',
           '*Generated from `source_localization/methods.py` (`python -m '
           'source_localization.methods > docs/methods/CATALOGUE.md`); do not edit by hand. '
           '`tests/test_catalogue.py` checks it is current.*', '']
    for title, section in CATALOGUE.items():
        out += [f'## {title}', '', '| id | version | what it is |', '|---|---|---|']
        out += [f'| `{m.id}` | {m.version} | {m.summary} |' for m in section.values()]
        out.append('')
        for m in section.values():
            out += [f'### `{m.id}` (version {m.version})', '', f'{m.summary[0].upper()}{m.summary[1:]}.', '',
                    f'- **Model:** {m.model}']
            if m.priors:
                label = 'Rests on' if all(p in catalogue_entries() for p in m.priors) else 'Values'
                items = ', '.join(f'`{p}`' if p in catalogue_entries() else p for p in m.priors)
                out.append(f'- **{label}:** {items}')
            for field, name in (('validity', 'Validity'), ('caveat', 'Caveat'),
                                ('implementation', 'Implementation'), ('tests', 'Tests'),
                                ('reference', 'Reference')):
                text = getattr(m, field)
                if text:
                    out.append(f'- **{name}:** ' + (f'`{text}`' if field in ('implementation', 'tests')
                                                    else text))
            out.append('')
    return '\n'.join(out).rstrip('\n') + '\n'


if __name__ == '__main__':
    print(catalogue_markdown(), end='')
