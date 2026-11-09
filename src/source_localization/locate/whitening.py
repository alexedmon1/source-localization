"""Noise whitening from real, correlated noise.

Ported unchanged in behaviour from probability-atlas ``pa/whitening.py`` (field-method plan, phase F1). The
estimators are catalogued (``methods.WHITENERS``) as ``whiten.<estimator>``.

The single-source posterior assumes white noise of known size; the recordings' noise is neither.
This estimates it from real noise samples and whitens data and leadfields by it, as the reference
(``diag_probabilistic_location.py``) does:

* **Noise samples.** One (n_channels, n_columns) matrix per recording that estimates the noise *in
  that recording's response pattern*. For trial-averaged responses the reference uses
  (odd-trial average - even-trial average) / 2, whose variance equals that of the noise in the
  full average when the halves have equal trial counts. Complex steady-state noise enters as
  [Re, Im] columns.
* **Average reference.** Data and leadfields are projected onto the n-1 orthonormal directions
  orthogonal to the all-ones vector (the basis of ``average_reference_basis``), so the reference
  electrode drops out. Whitened quantities therefore have n-1 rows.
* **Shape pooled, scale per recording.** Each recording's noise gets its own scale, the RMS of its
  referenced noise samples; the spatial covariance *shape* is the average over recordings of the
  scale-normalised sample covariances (weighted by column counts). Assumes every recording shares
  one noise shape and differs only in size.
* **Whitener.** W = C^{-1/2} B^T with C^{-1/2} the symmetric inverse square root; recording r's
  pattern y is whitened as W y / scale_r, giving noise of unit variance per whitened channel.

Note for calibration: with few samples (65 recordings x 2 columns for ASSR, 64 x 1 for a chirp
peak, in 29 dimensions) the covariance shape is itself noisy, and the reference's simulations
draw their noise from the same samples the whitener was fitted on (in-sample).

**Phase 2 (decision of 2026-10-05, reports/2026-10-05_phase2_rule.md).** The plain sample covariance
(Phase 1, ``estimator="sample"``) amplifies noise it was not fitted on (held-out whitened power 2.2x
for ASSR, 15-24x for the chirp peaks). The shape estimator is therefore chosen per measure, by
held-out Gaussian log-likelihood (:func:`heldout_scores`), among:

  estimator  "sample" | "ledoit_wolf" | "oas" | "diagonal"
  samples    the measure's own noise vectors (one per recording and column), or more noise samples
             per recording (``shape_samples``: ASSR neighbouring DFT bins, chirp time points in the
             peak's window), each recording's samples normalised by their own RMS.

"diagonal" is per-electrode independent noise, then average-referenced: C = B' diag(d) B, with d
the per-electrode mean square of the (scale-normalised, average-referenced) samples. Ledoit-Wolf
and OAS are scikit-learn's, on zero-mean samples (``assume_centered``). The per-recording *scale*
is always the RMS of the recording's own measure noise vector, as in Phase 1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = ["average_reference_basis", "NoiseWhitener", "estimate_covariance", "gaussian_loglik",
           "heldout_scores", "choose_whitener", "ESTIMATORS"]

ESTIMATORS = ("sample", "ledoit_wolf", "oas", "diagonal")


def average_reference_basis(n: int) -> np.ndarray:
    """(n, n-1) orthonormal basis of the space orthogonal to the all-ones vector (as the reference)."""
    return np.linalg.svd(np.eye(n) - 1.0 / n)[0][:, : n - 1]


@dataclass
class NoiseWhitener:
    """Whitening by pooled real-noise shape and per-recording scale (module docstring).

    Attributes
    ----------
    W : (m, n) array
        Shape whitener including the average-reference projection (m = n-1), or the plain
        C^{-1/2} (m = n) when fitted with ``average_reference=False``.
    scales : (n_recordings,) array
        Per-recording noise SD in the referenced space.
    covariance : (m, m) array
        The pooled, scale-normalised noise covariance shape.
    """

    W: np.ndarray
    scales: np.ndarray
    covariance: np.ndarray

    @classmethod
    def fit(cls, noise: Sequence[np.ndarray], average_reference: bool = True, estimator: str = "sample",
            shape_samples: Sequence[np.ndarray] | None = None) -> "NoiseWhitener":
        """Fit from one noise matrix (n_channels, n_columns) per recording.

        ``estimator`` and ``shape_samples`` choose the covariance-shape estimate (module docstring);
        the defaults reproduce Phase 1 exactly. Scales always come from ``noise``.
        """
        noise = [np.asarray(z, float).reshape(len(z), -1) for z in noise]
        n = noise[0].shape[0]
        B = average_reference_basis(n) if average_reference else np.eye(n)
        scales = np.array([cls.scale_of(z, B) for z in noise])
        if shape_samples is None:
            Nz = [B.T @ z for z in noise]
            if estimator == "sample":          # Phase 1's exact arithmetic, so its results reproduce bit for bit
                C = sum((z / s) @ (z / s).T for z, s in zip(Nz, scales)) / sum(z.shape[1] for z in Nz)
            else:
                C = estimate_covariance(np.concatenate([z / s for z, s in zip(Nz, scales)], axis=1), estimator, B)
        else:
            X = np.concatenate([(B.T @ z) / cls.scale_of(z, B)
                                for z in (np.asarray(x, float).reshape(n, -1) for x in shape_samples)], axis=1)
            C = estimate_covariance(X, estimator, B)
        w, V = np.linalg.eigh(C)
        W = V @ np.diag(w ** -0.5) @ V.T @ B.T
        return cls(W=W, scales=scales, covariance=C)

    @staticmethod
    def scale_of(z: np.ndarray, B: np.ndarray) -> float:
        """RMS of a noise matrix in the referenced space (the per-recording scale)."""
        v = B.T @ np.asarray(z, float).reshape(B.shape[0], -1)
        return float(np.sqrt(np.sum(v ** 2) / v.size))

    @property
    def n_whitened(self) -> int:
        return self.W.shape[0]

    def whiten(self, y: np.ndarray, recording: int | None = None) -> np.ndarray:
        """Whiten a pattern (n,) or (n, cols); divide by recording's scale if ``recording`` is given."""
        out = self.W @ np.asarray(y, float)
        return out if recording is None else out / self.scales[recording]

    def whiten_leadfield(self, G: np.ndarray) -> np.ndarray:
        """Whiten leadfields (..., n_channels, 3) -> (..., m, 3). Shape only: scale cancels into the moment."""
        return np.einsum("ac,...ck->...ak", self.W, G)


def estimate_covariance(X: np.ndarray, estimator: str = "sample", B: np.ndarray | None = None) -> np.ndarray:
    """Covariance (m, m) of zero-mean samples ``X`` (m, N) in the referenced basis.

    "diagonal" needs ``B`` (n, m), the basis X is expressed in: C = B' diag(d) B with d the
    per-electrode mean square of B X (per-electrode independent noise, then referenced).
    """
    X = np.asarray(X, float)
    if estimator == "sample":
        return X @ X.T / X.shape[1]
    if estimator == "ledoit_wolf":
        from sklearn.covariance import ledoit_wolf
        return ledoit_wolf(X.T, assume_centered=True)[0]
    if estimator == "oas":
        from sklearn.covariance import oas
        return oas(X.T, assume_centered=True)[0]
    if estimator == "diagonal":
        if B is None:
            raise ValueError("diagonal needs the basis B")
        d = np.mean((B @ X) ** 2, axis=1)
        return B.T @ np.diag(d) @ B
    raise ValueError(f"unknown estimator {estimator!r}")


def gaussian_loglik(C: np.ndarray, V: np.ndarray) -> float:
    """Mean log N(v; 0, C) over the columns v of V (m, N)."""
    w, U = np.linalg.eigh(C)
    q = np.sum((U.T @ V) ** 2 / w[:, None], axis=0)
    return float(np.mean(-0.5 * (q + np.sum(np.log(w)) + len(w) * np.log(2 * np.pi))))


def heldout_scores(fit_noise: Sequence[np.ndarray], test_noise: Sequence[np.ndarray], estimator: str,
                   fit_shape_samples: Sequence[np.ndarray] | None = None) -> dict:
    """Fit a whitener on ``fit_noise`` and score it on ``test_noise`` (recordings not in the fit).

    Each test recording's measure noise vectors are scale-normalised by their own RMS and
    expressed in the average-reference basis. Returns ``loglik`` (mean Gaussian log-likelihood per
    held-out vector under the fitted shape) and ``power`` (mean held-out whitened power per
    channel; 1 if the shape is right).
    """
    wh = NoiseWhitener.fit(fit_noise, estimator=estimator, shape_samples=fit_shape_samples)
    n = np.asarray(test_noise[0]).shape[0]
    B = average_reference_basis(n)
    V = [(B.T @ np.asarray(z, float).reshape(n, -1)) / NoiseWhitener.scale_of(z, B) for z in test_noise]
    Vall = np.concatenate(V, axis=1)
    power = float(np.mean([np.sum((wh.W @ B @ v) ** 2) / v.size for v in V]))
    return {"loglik": gaussian_loglik(wh.covariance, Vall), "power": power}


def choose_whitener(candidates: dict, power_tolerance: float = 0.15) -> str | None:
    """The selection criterion of the 2026-10-05 decision.

    ``candidates``: name -> {"median_loglik", "median_power"}. Among those whose median held-out
    whitened power is within 1 +- ``power_tolerance``, the highest median held-out log-likelihood;
    ``None`` if none qualifies (the rule then says stop).
    """
    ok = {k: v for k, v in candidates.items() if abs(v["median_power"] - 1.0) <= power_tolerance}
    return max(ok, key=lambda k: ok[k]["median_loglik"]) if ok else None
