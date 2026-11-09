"""Where a pattern comes from: location posteriors and what they rest on.

- :mod:`.ensemble_posterior`: the head-model-ensemble location posteriors and map calibrations
  (``locate.single_source``, ``locate.mirror_pair``, ``locate.source_set``; ``calibration.*``).
- :mod:`.head_ensemble`: the head-model prior and named ensembles (``prior.head_ensemble_*``).
- :mod:`.geometry`: the head geometry and leadfields under a perturbed head model.
- :mod:`.whitening`: noise whitening from real noise (``whiten.*``).
- :mod:`.regions`: region labels, aggregation and credible sets.
- :mod:`.space`: all of the above for one study's head, with its provenance.

Every model here is declared in the catalogue (``source_localization.methods``, ``docs/methods/CATALOGUE.md``) with an
id and a version. The field-method plan (bayes-sl ``reports/2026-11-09_field_method_library_plan.md``) builds the
field model on this package. No group statistics live here.
"""
from .ensemble_posterior import (SURFACE, UNCALIBRATED, VOLUMETRIC, EnsemblePosterior, MapCalibration,  # noqa: F401
                                 MirrorPairPosterior, SourceSetPosterior, model_probabilities)
from .head_ensemble import (ENSEMBLE_PA_PHASE2, ENSEMBLES, EnsembleSpec, HeadModel, HeadModelPrior,  # noqa: F401
                            draw_ensemble, ensemble_leadfields)
from .regions import RegionMap, credible_set, labelling  # noqa: F401
from .whitening import NoiseWhitener, average_reference_basis  # noqa: F401
