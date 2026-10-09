"""Reference directed-connectivity measures for network validation: PSI, Granger causality, time-reversed GC.

Ported unchanged from the probability-atlas method workspace (``pa/directed.py``, 2026-10). Every measure is a net,
antisymmetric matrix N with N[i, j] > 0 meaning i leads j:

- **PSI** (Nolte et al. 2008, PMID 18643502): Im sum_f C_ij(f)* C_ij(f + df) over the band, from per-epoch spectra.
- **GC**: net spectral Granger causality (Geweke) from a bivariate VAR per pair, band-averaged.
- **TRGC** (Haufe et al. 2013, PMID 23006806): net GC minus net GC on the time-reversed data. The VAR is fitted by
  Yule-Walker from lagged covariances; a reversed series' covariances are the transposes, so no refit is needed.

On volume-conducted data naive GC reports spurious direction under instantaneous mixing; TRGC and PSI do not (on
average). Other directed measures (dPLI, DTF, TE) are injected by the caller, as for undirected metrics.
"""
from __future__ import annotations

import numpy as np

__all__ = ["lagged_cov", "net_gc", "net_psi", "directed_net", "to_net", "node_net"]


# ---------------------------------------------------------------- Granger causality
def lagged_cov(X: np.ndarray, order: int) -> np.ndarray:
    """(order + 1, n, n) R[k] = E[x(t) x(t - k)^T] of (n_epochs, n, n_samples) data, within epochs."""
    X = X - X.mean(axis=-1, keepdims=True)
    T = X.shape[-1]
    R = np.empty((order + 1, X.shape[1], X.shape[1]))
    for k in range(order + 1):
        R[k] = np.einsum("eit,ejt->ij", X[:, :, k:], X[:, :, :T - k]) / (X.shape[0] * (T - k))
    return R


def _yule_walker(R: np.ndarray):
    """Batched bivariate Yule-Walker. R (m, order + 1, 2, 2) -> A (m, order, 2, 2), Sigma (m, 2, 2)."""
    m, p1 = R.shape[:2]
    p = p1 - 1
    Rf = lambda k: R[:, k] if k >= 0 else np.swapaxes(R[:, -k], -1, -2)   # R(-k) = R(k)^T
    G = np.empty((m, 2 * p, 2 * p))
    for a in range(p):
        for b in range(p):
            G[:, 2 * a:2 * a + 2, 2 * b:2 * b + 2] = Rf(b - a)                 # Gamma[l, k] = R(k - l)
    rhs = np.concatenate([R[:, k] for k in range(1, p + 1)], axis=-1)        # (m, 2, 2p) = [R(1) .. R(p)]
    Acat = np.linalg.solve(np.swapaxes(G, -1, -2), np.swapaxes(rhs, -1, -2))  # A Gamma = rhs
    Acat = np.swapaxes(Acat, -1, -2)                                         # (m, 2, 2p)
    A = np.stack([Acat[:, :, 2 * l:2 * l + 2] for l in range(p)], axis=1)
    Sig = R[:, 0] - sum(A[:, l] @ np.swapaxes(R[:, l + 1], -1, -2) for l in range(p))
    return A, Sig


def _gc_net_band(A: np.ndarray, Sig: np.ndarray, freqs: np.ndarray, sfreq: float) -> np.ndarray:
    """Net Geweke GC (0 -> 1 minus 1 -> 0), averaged over ``freqs``. A (m, p, 2, 2), Sig (m, 2, 2) -> (m,)."""
    p = A.shape[1]
    z = np.exp(-2j * np.pi * np.outer(freqs, np.arange(1, p + 1)) / sfreq)  # (F, p)
    Af = np.eye(2)[None, None] - np.einsum("fl,mlij->mfij", z, A)           # (m, F, 2, 2)
    H = np.linalg.inv(Af)
    S = H @ Sig[:, None] @ np.conj(np.swapaxes(H, -1, -2))
    s11, s22, s12 = Sig[:, 0, 0, None], Sig[:, 1, 1, None], Sig[:, 0, 1, None]
    S11, S22 = S[..., 0, 0].real, S[..., 1, 1].real
    gc_1to0 = np.log(S11 / (S11 - (s22 - s12 ** 2 / s11) * np.abs(H[..., 0, 1]) ** 2))
    gc_0to1 = np.log(S22 / (S22 - (s11 - s12 ** 2 / s22) * np.abs(H[..., 1, 0]) ** 2))
    return (gc_0to1 - gc_1to0).mean(axis=1)


def net_gc(X: np.ndarray, sfreq: float, band, order: int, df: float = 0.25):
    """(naive net GC, TRGC), each (n, n) antisymmetric, from pairwise bivariate VARs on (n_epochs, n, n_samples)."""
    n = X.shape[1]
    R = lagged_cov(X, order)
    iu, ju = np.triu_indices(n, 1)
    Rp = np.empty((len(iu), order + 1, 2, 2))
    Rp[:, :, 0, 0], Rp[:, :, 1, 1] = R[:, iu, iu].T, R[:, ju, ju].T
    Rp[:, :, 0, 1], Rp[:, :, 1, 0] = R[:, iu, ju].T, R[:, ju, iu].T
    freqs = np.arange(band[0], band[1] + df / 2, df)
    fwd = _gc_net_band(*_yule_walker(Rp), freqs, sfreq)
    rev = _gc_net_band(*_yule_walker(np.swapaxes(Rp, -1, -2)), freqs, sfreq)
    return to_net(fwd, iu, ju, n), to_net(fwd - rev, iu, ju, n)


# ---------------------------------------------------------------- PSI
def net_psi(X: np.ndarray, sfreq: float, band) -> np.ndarray:
    """(n, n) PSI from per-epoch Hann-windowed spectra of (n_epochs, n, n_samples); PSI[i, j] > 0: i leads j."""
    T = X.shape[-1]
    F = np.fft.rfft((X - X.mean(axis=-1, keepdims=True)) * np.hanning(T), axis=-1)
    f = np.fft.rfftfreq(T, 1.0 / sfreq)
    sel = np.flatnonzero((f >= band[0]) & (f <= band[1]))
    Fb = F[:, :, sel]
    S = np.einsum("eif,ejf->ijf", Fb, np.conj(Fb)) / F.shape[0]
    d = np.sqrt(np.einsum("iif->if", S).real)
    C = S / (d[:, None, :] * d[None, :, :])
    return np.imag((np.conj(C[:, :, :-1]) * C[:, :, 1:]).sum(axis=-1))


# ---------------------------------------------------------------- helpers
def to_net(v: np.ndarray, iu, ju, n: int) -> np.ndarray:
    N = np.zeros((n, n))
    N[iu, ju], N[ju, iu] = v, -v
    return N


def directed_net(X: np.ndarray, sfreq: float, band, order: int) -> dict:
    """PSI, naive GC and TRGC net matrices of (n_epochs, n, n_samples)."""
    gc, trgc = net_gc(X, sfreq, band, order)
    return {"psi": net_psi(X, sfreq, band), "gc": gc, "trgc": trgc}


def node_net(N: np.ndarray, units: list, node_of: dict, nodes: list) -> np.ndarray:
    """Node-level net matrix: mean of a unit-level net matrix over the unit pairs between two nodes."""
    lab = np.array([node_of[u] for u in units], dtype=object)
    M = np.zeros((len(nodes), len(nodes)))
    for a, na in enumerate(nodes):
        ia = np.flatnonzero(lab == na)
        for b in range(a + 1, len(nodes)):
            ib = np.flatnonzero(lab == nodes[b])
            if len(ia) and len(ib):
                M[a, b] = N[np.ix_(ia, ib)].mean()
                M[b, a] = -M[a, b]
    return M
