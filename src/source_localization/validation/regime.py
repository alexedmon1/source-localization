"""The validation *regime*: what a validation run's simulation assumes.

This module is the **single definition** of the two regimes. Everything else (the runner, the CLI flag, the
config key, ``metrics.json``, the docs) refers to these names and descriptions instead of restating them.

``"realistic"`` — the default from v0.6.0
    - **Truths** are simulated at the *requested* (off-grid) position, through a head model drawn from
      :class:`~source_localization.validation.head_models.HeadModelPrior`: a registration shift of the electrode
      array and a skull-conductivity factor. The inverse keeps the nominal model, so model error is present.
    - **Noise** is a recorded background when ``validation.background`` names one. Otherwise it is generated
      (``noise_type``), and the output records that it was generated.
    - **Noise-only control:** the readout is also run on noise alone, and its false-confidence statistics are
      reported (see :func:`noise_only_summary`).

``"legacy"`` — the v0.5.x behaviour, kept **only to reproduce numbers produced before v0.6.0**
    - **Truths** are simulated with the inversion's *own* forward model, at the source-grid position nearest the
      requested one: the "inverse crime". The one exception is the ``forward_model_mismatch`` option, which swaps
      in fixed ground-truth conductivities.
    - **Noise** is always generated (white unless ``noise_type`` says otherwise), never recorded.
    - **No noise-only control.**

    Legacy numbers are **best-case** and are **not comparable** with realistic numbers. "Legacy" here means
    *this exact set of assumptions*, nothing else: not "deprecated code", not "old atlas", not the legacy
    *atlas names* (``full``, ``coarse_22roi``) the validation CLI also accepts. Every run in the legacy regime
    logs :data:`LEGACY_WARNING` and records :func:`describe` in its output.

Select with ``validation.regime: realistic | legacy`` in the config, or ``--regime`` on the CLI.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["REALISTIC", "LEGACY", "REGIMES", "DEFAULT_REGIME", "LEGACY_WARNING", "Regime", "resolve_regime",
           "regime_name", "describe", "noise_only_summary"]

REALISTIC = "realistic"
LEGACY = "legacy"
REGIMES = (REALISTIC, LEGACY)
DEFAULT_REGIME = REALISTIC

LEGACY_WARNING = (
    "validation regime 'legacy' (v0.5.x assumptions): truths use the inversion's own forward model at grid-snapped "
    "positions (the 'inverse crime'), noise is generated, and no noise-only control is run. These numbers are "
    "best-case and NOT comparable with the default 'realistic' regime. Use 'legacy' only to reproduce results "
    "produced before v0.6.0. See source_localization.validation.regime."
)

_DESCRIPTIONS = {
    REALISTIC: ("truths simulated at the requested positions through head models drawn from a prior (registration "
                "shift + skull conductivity); recorded background noise if configured, else generated; noise-only "
                "control reported"),
    LEGACY: ("v0.5.x assumptions, for reproducing pre-0.6.0 numbers only: truths through the inversion's own forward "
             "model at grid-snapped positions (inverse crime); generated noise; no noise-only control. Best-case; "
             "not comparable with 'realistic'"),
}


@dataclass(frozen=True)
class Regime:
    """The resolved settings of one validation run. Build it with :func:`resolve_regime`, never by hand."""

    name: str
    perturbed_truths: bool          # truths through a head model drawn from the prior, at requested positions
    recorded_background: bool       # a background was configured (realistic only)
    noise_only_control: bool
    head_model_prior: Optional[Dict[str, Any]] = None   # HeadModelPrior fields, realistic only
    n_head_models: int = 0
    noise_only_trials: int = 0
    seed: int = 0

    @property
    def is_legacy(self) -> bool:
        return self.name == LEGACY


def regime_name(val_config: Optional[Mapping[str, Any]] = None, override: Optional[str] = None) -> str:
    """The regime's name only (``override`` first, then ``validation.regime``, then the default), validated, with
    no logging. Used where a run must know its regime before setup, e.g. to choose its output directory."""
    name = str(override or (val_config or {}).get("regime") or DEFAULT_REGIME).lower()
    if name not in REGIMES:
        raise ValueError(f"validation regime must be one of {REGIMES}; got {name!r}")
    return name


def resolve_regime(val_config: Optional[Mapping[str, Any]] = None, override: Optional[str] = None) -> Regime:
    """The run's regime from the ``validation:`` block, with ``override`` (the CLI's ``--regime``) taking precedence.

    Realistic-regime options in ``validation:`` (all optional)::

        regime: realistic                    # or legacy
        truth_head_model:
          shift_sd_mm: 0.3                   # per-axis registration SD, or [x, y, z]
          skull_factor_range: [0.5, 2.0]     # log-uniform multiple of the skull conductivity
          n_models: 16                       # head models drawn per run; each trial uses one at random
        background:                          # recorded noise (see noise.RecordedBackground)
          files: [/path/a.set, /path/b.set]  # or a glob string
        noise_only_control:
          n_trials: 200
        regime_seed: 20261009

    Unknown regime names raise. In the legacy regime the realistic options are ignored, and a warning says so.
    """
    val_config = dict(val_config or {})
    name = regime_name(val_config, override)
    seed = int(val_config.get("regime_seed", 20261009))
    if name == LEGACY:
        logger.warning(LEGACY_WARNING)
        ignored = [k for k in ("truth_head_model", "background", "noise_only_control") if k in val_config]
        if ignored:
            logger.warning("validation regime 'legacy' ignores %s", ignored)
        return Regime(name=LEGACY, perturbed_truths=False, recorded_background=False, noise_only_control=False,
                      seed=seed)
    thm = dict(val_config.get("truth_head_model") or {})
    prior = {"shift_sd_mm": thm.get("shift_sd_mm", 0.3),
             "skull_factor_range": tuple(thm.get("skull_factor_range", (0.5, 2.0)))}
    noc = dict(val_config.get("noise_only_control") or {})
    return Regime(name=REALISTIC, perturbed_truths=True, recorded_background=bool(val_config.get("background")),
                  noise_only_control=True, head_model_prior=prior, n_head_models=int(thm.get("n_models", 16)),
                  noise_only_trials=int(noc.get("n_trials", 200)), seed=seed)


def describe(regime: Regime) -> Dict[str, Any]:
    """The block recorded in every validation output, so a number always says which regime produced it."""
    out = asdict(regime)
    out["description"] = _DESCRIPTIONS[regime.name]
    out["comparable_with"] = regime.name
    out["defined_in"] = "source_localization.validation.regime"
    return out


def noise_only_summary(attributed_rois, reachable_rois, flag_ratio: float = 3.0) -> Dict[str, Any]:
    """False-confidence statistics of a readout run on noise alone.

    Parameters
    ----------
    attributed_rois : sequence of int
        The ROI the readout named on each noise-only trial.
    reachable_rois : sequence of int
        The ROIs the readout could name (those with at least one source); the uniform reference.
    flag_ratio : float
        Flag when one ROI attracts more than this multiple of its uniform share.

    Returns
    -------
    dict: ``n_trials``, ``top_roi``, ``top_share``, ``uniform_share``, ``ratio_to_uniform``, ``flagged``,
    ``normalized_entropy`` (1 = spread like uniform, 0 = always the same ROI) and ``counts``.
    """
    rois = np.asarray(list(attributed_rois), int)
    reach = sorted(set(int(r) for r in reachable_rois if int(r) > 0))
    if len(rois) == 0 or not reach:
        return {"n_trials": int(len(rois)), "flagged": False}
    vals, cnt = np.unique(rois, return_counts=True)
    p = cnt / cnt.sum()
    k = int(np.argmax(cnt))
    uniform = 1.0 / len(reach)
    ent = float(-(p * np.log(p)).sum() / np.log(len(reach))) if len(reach) > 1 else 1.0
    ratio = float(p[k] / uniform)
    return {"n_trials": int(len(rois)), "top_roi": int(vals[k]), "top_share": float(p[k]),
            "uniform_share": float(uniform), "ratio_to_uniform": ratio, "flagged": bool(ratio > flag_ratio),
            "normalized_entropy": ent, "counts": {int(v): int(c) for v, c in zip(vals, cnt)}}
