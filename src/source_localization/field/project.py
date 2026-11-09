"""Projecting recordings onto a frozen field template: the field mode's ``project`` step.

Each recording's epochs are average-referenced over the template's channels, picked in the template's channel order
(by name, so the file's order does not matter; a missing channel raises), and reduced to one **virtual channel** per
field. Two catalogued projections (``methods.FIELD_DEFINITIONS``):

- ``field.project_inner`` (default): each field separately, s_k(t) = u_k . x(t). The field method's readout since
  surface-evoked troubleshooting log 70 (``rerun070_fields.py``); reproduces its series exactly.
- ``field.project_joint``: all selected fields at once by least squares, s(t) = (M M^T)^-1 M x(t). Equals the inner
  projection when the fields are orthogonal; when they overlap it splits the shared part between them.

The **residual** is what the selected fields do not capture: x - P x, P the orthogonal projector onto their span
(the same for both projections). It is stored as its trial average (the Conductor, 2026-11-09), so what the fields
miss can still be located or tested.

**Output** (one directory per recording, the layout source-analytics already reads for ROI series):

- ``data/step6_roi_timeseries_signed.pkl``: {field name: 1-D series, epochs concatenated (epoch-major)}
- ``data/roi_timeseries_signed.set``: a MAT file with ``srate`` (what source-analytics reads the rate from)
- ``data/field_residual_average.npz``: the residual's trial average (channels x times), channel names, times
- ``data/field_run.json``: the run record (``readout_mode: field``; the template's name, version, sha256 and
  definition ids; the projection's catalogue id and version; the maps; reference; source file and its sha256;
  epochs, samples, rate, tmin; package version). source-analytics reads ``readout_mode`` to refuse pooling a field
  run with a source run.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from .template import FieldTemplate, load

__all__ = ["PROJECTIONS", "pick_channels", "project", "write_run", "project_file"]

#: Catalogue id of each projection method.
PROJECTIONS = {"inner": "field.project_inner", "joint": "field.project_joint"}


def pick_channels(data: np.ndarray, ch_names: Sequence[str], channels: Sequence[str]) -> np.ndarray:
    """``data`` (..., n_file_channels, times) reduced to ``channels`` in that order, by name. Raises if one is missing."""
    names = list(ch_names)
    missing = [c for c in channels if c not in names]
    if missing:
        raise ValueError(f"channels missing from the recording: {missing}")
    return np.asarray(data)[..., [names.index(c) for c in channels], :]


def project(X: np.ndarray, template: FieldTemplate, maps: Optional[Sequence[str]] = None, method: str = "inner",
            residual_of: Optional[Sequence[str]] = None) -> dict:
    """Project epochs ``X`` (epochs, channels, times), already in the template's channel order.

    The data are average-referenced over these channels first. Returns ``series`` {map: (epochs, times)},
    ``residual_average`` (channels, times) and the maps used. ``maps`` defaults to every map in the template,
    ``residual_of`` to ``maps``.
    """
    if method not in PROJECTIONS:
        raise ValueError(f"unknown projection {method!r}; choose from {list(PROJECTIONS)}")
    X = np.asarray(X, float)
    if X.ndim != 3 or X.shape[1] != len(template.channels):
        raise ValueError(f"X must be (epochs, {len(template.channels)}, times) in the template's channel order")
    maps = list(template.maps) if maps is None else list(maps)
    residual_of = maps if residual_of is None else list(residual_of)
    unknown = [m for m in set(maps) | set(residual_of) if m not in template.maps]
    if unknown:
        raise ValueError(f"maps not in template {template.name} v{template.version}: {unknown}")
    X = X - X.mean(axis=1, keepdims=True)
    M = np.stack([template.maps[k] for k in maps])                       # (k, channels)
    if method == "inner":
        S = np.einsum("kc,ect->ket", M, X)
    else:
        S = np.einsum("kc,ect->ket", np.linalg.pinv(M).T, X)
    R = np.stack([template.maps[k] for k in residual_of])
    P = np.linalg.pinv(R) @ R                                            # projector onto span(R) in channel space
    residual = X.mean(0) - P @ X.mean(0)
    return {"series": {k: S[i] for i, k in enumerate(maps)}, "residual_average": residual, "maps": maps,
            "residual_of": residual_of, "method": method}


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_run(out_dir, result: dict, template: FieldTemplate, sfreq: float, tmin: float, times: np.ndarray,
              source: Optional[str] = None, extra: Optional[dict] = None) -> Path:
    """Write one recording's field run under ``out_dir/data`` (module docstring). Returns the data directory."""
    from scipy.io import savemat

    from .. import __version__
    from ..methods import catalogue_entries
    data = Path(out_dir) / "data"
    data.mkdir(parents=True, exist_ok=True)
    series = {k: np.asarray(v).reshape(-1) for k, v in result["series"].items()}
    with open(data / "step6_roi_timeseries_signed.pkl", "wb") as f:
        pickle.dump(series, f)
    savemat(str(data / "roi_timeseries_signed.set"), {"srate": float(sfreq)})
    np.savez_compressed(data / "field_residual_average.npz", residual=result["residual_average"],
                        channels=np.array(template.channels), times=np.asarray(times),
                        residual_of=np.array(result["residual_of"]))
    pid = PROJECTIONS[result["method"]]
    n_ep, n_t = next(iter(result["series"].values())).shape
    record = {
        "readout_mode": "field", "step": "project",
        "template": template.provenance(),
        "projection": {"id": pid, "version": catalogue_entries()[pid].version},
        "maps": result["maps"], "residual_of": result["residual_of"], "residual": "trial average",
        "reference": "average over the template's channels, re-applied", "channels": template.channels,
        "n_epochs": int(n_ep), "n_times": int(n_t), "sfreq": float(sfreq), "tmin": float(tmin),
        "source": None if source is None else {"path": str(source), "sha256": _sha256(source)},
        "package": {"name": "source-localization", "version": __version__},
        **(extra or {})}
    (data / "field_run.json").write_text(json.dumps(record, indent=1))
    return data


def project_file(eeg_file, template, out_dir, maps: Optional[Sequence[str]] = None, method: str = "inner",
                 residual_of: Optional[Sequence[str]] = None, extra: Optional[dict] = None) -> Path:
    """Read an epoched EEGLAB file, project it onto ``template`` (a FieldTemplate or a path) and write its run."""
    import mne
    template = load(template) if not isinstance(template, FieldTemplate) else template
    ep = mne.read_epochs_eeglab(str(eeg_file), verbose="ERROR")
    X = pick_channels(ep.get_data(), ep.ch_names, template.channels)
    res = project(X, template, maps, method, residual_of)
    return write_run(out_dir, res, template, ep.info["sfreq"], ep.tmin, ep.times, source=eeg_file, extra=extra)
