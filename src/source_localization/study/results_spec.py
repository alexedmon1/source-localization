"""The study's localization, described by the shared results specification.

A reviewer of an EEG study needs to see how its recordings were localized:
which recipe, whether every recording got the same one, how each went, and --
for Monte Carlo source sampling -- which parcels the montage could not resolve.
Each recording records that in its own folder (``config_resolved.yaml``,
``monte_carlo_report.json``, ``pipeline_report.html``); this gathers it into one
study-level analysis folder, beside the QC report, in the format of
neurobox-dev's results specification (schema 0.4): an ``analysis.json`` with
``role: diagnostic`` and ``domain: localization``, two tables with column
dictionaries, the QC figures, and ``provenance.json``.

``recordings.csv`` names each recording explicitly -- subject, session, group,
its output folder and the paths of its report and figures -- so a reader joins
on fields, never by parsing folder names such as ``sub-801_ses-01``.

Optional: neurobox-dev is the ``results`` extra. Without it nothing is written;
a folder that does not conform is logged and never fails the QC.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:                                               # the optional `results` extra
    import neurobox
    from neurobox.spec import default_run_id as _default_run_id
except ImportError:                                # pragma: no cover - depends on install
    neurobox = None

INSTALL_HINT = "install the optional extra: pip install 'source-localization[results]'"

#: A parcel is near-collinear with another when their sensor topographies
#: correlate at or above this (``realizations.monte_carlo_roi``'s
#: ``collinear_threshold``); such a parcel's own value is not interpretable.
COLLINEARITY_THRESHOLD = 0.99
#: Below this share of draws a parcel's row averages too few placements to be
#: compared as an amplitude (the warning in ``realizations.py``).
COVERAGE_THRESHOLD = 0.5

RECORDINGS = "recordings.csv"
PARCELS = "parcel_coverage.csv"

#: The recipe fields compared across recordings: (name, path in config_resolved's ``config``).
RECIPE = (
    ("preset", ("provenance", "preset")), ("atlas", ("provenance", "atlas")),
    ("bem_type", ("pipeline", "bem_type")), ("source_type", ("pipeline", "source_type")),
    ("surface_method", ("source_space", "surface", "method")),
    ("spacing_mm", ("source_space", "surface", "spacing_mm")),
    ("source_sampling", ("source_space", "source_sampling")),
    ("monte_carlo", ("source_space", "monte_carlo")),
    ("inverse_method", ("inverse", "method")), ("snr", ("inverse", "snr")),
    ("orientation", ("inverse", "orientation")), ("output_variants", ("outputs", "output_variants")),
)

#: The QC report's plots (qc.generate_qc_plots), by file-name stem.
FIGURE_CAPTIONS = {
    "source_amplitude": "Source and ROI amplitude per recording, by group",
    "forward_condition": "Forward-model condition number per recording, by group",
    "timepoints": "Time points per recording",
    "sources_per_roi": "Sources per ROI per recording",
    "processing_time": "Processing time per recording",
    "zscore_heatmap": "QC metrics as z-scores per recording (outliers stand out)",
}

#: Descriptions of qc_metrics' columns, carried into recordings.csv.
METRIC_DOCS = {
    "n_channels": "EEG channels after registration",
    "sfreq": ("sampling frequency", "Hz"),
    "n_sources": "sources in the source space (the Monte Carlo pool, when sampled)",
    "forward_condition_number": "condition number of the leadfield",
    "forward_leadfield_norm_mean": "mean column norm of the leadfield",
    "forward_leadfield_norm_std": "SD of the leadfield's column norms",
    "stc_n_sources": "sources in the source estimate (empty under Monte Carlo sampling)",
    "stc_n_times": "time points in the source estimate",
    "stc_amp_min": "smallest source amplitude",
    "stc_amp_max": "largest source amplitude",
    "stc_amp_mean": "mean source amplitude",
    "stc_amp_std": "SD of source amplitude",
    "stc_amp_median": "median source amplitude",
    "n_rois": "ROIs in the ROI time series",
    "roi_amp_mean": "mean absolute ROI amplitude",
    "roi_amp_std": "SD of absolute ROI amplitude",
    "sources_per_roi_mean": "mean sources per ROI",
    "sources_per_roi_std": "SD of sources per ROI",
    "sources_per_roi_min": "fewest sources in an ROI",
    "sources_per_roi_max": "most sources in an ROI",
    "processing_time_sec": ("time the recording's localization took", "s"),
    "outlier_flag": "flagged as an outlier on at least one QC metric (see the QC report)",
}


def available() -> bool:
    return neurobox is not None


def _get(d: Any, path: tuple) -> Any:
    for key in path:
        if not isinstance(d, dict):
            return None
        d = d.get(key)
    return d


def _rel(path: Optional[Path], start: Path) -> str:
    return os.path.relpath(path, start).replace(os.sep, "/") if path else ""


def _read_yaml(path: Path) -> Optional[dict]:
    try:
        import yaml

        with open(path) as fh:
            return yaml.safe_load(fh)
    except Exception as exc:                       # noqa: BLE001 - a missing record is reported, not fatal
        logger.debug("Could not read %s: %s", path, exc)
        return None


def _read_json(path: Path) -> Optional[dict]:
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception as exc:                       # noqa: BLE001
        logger.debug("Could not read %s: %s", path, exc)
        return None


def recipe(resolved: Optional[dict]) -> Dict[str, Any]:
    """The recipe fields of one recording's ``config_resolved.yaml``."""
    cfg = (resolved or {}).get("config") or {}
    out = {name: _get(cfg, path) for name, path in RECIPE}
    out["version"] = (resolved or {}).get("source_localization_version")
    if out["source_sampling"] is None and cfg:
        out["source_sampling"] = "fixed"
    return out


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return "" if value != value else repr(value)          # NaN -> empty
    return str(value)


def _write_csv(path: Path, columns: List[str], rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(columns)
        for r in rows:
            writer.writerow([_cell(r.get(c)) for c in columns])


def collect(config: Any, qc_dir: Path, metrics_df: Any = None) -> Dict[str, Any]:
    """Read every recording's records: rows of both tables, recipes, problems. Pure reading."""
    qc_dir = Path(qc_dir)
    metrics = {}
    if metrics_df is not None and len(metrics_df):
        metrics = {str(r["subject_id"]): r for r in metrics_df.to_dict("records")}
    log = _read_json(Path(config.derivatives_dir) / "processing_log.json") or {}
    logged = {str(e.get("subject_id")): e for e in log.get("subjects") or [] if isinstance(e, dict)}

    recordings, recipes, mc_reports = [], {}, {}
    for s in config.subjects:
        folder = Path(config.get_subject_output_dir(s))
        pipe = folder / "pipeline"
        data = pipe / "data"
        report = pipe / "pipeline_report.html"
        registration = pipe / "electrode_registration" / "electrode_registration_validation.png"
        figures = sorted((pipe / "figures").glob("*.png")) if (pipe / "figures").is_dir() else []
        localized = (folder / "roi_timeseries").is_dir() or data.is_dir()
        entry = logged.get(str(s.subject_id))
        if entry is not None and entry.get("success") is False:
            status = "failed"
        else:
            status = "localized" if localized else "missing"
        row: Dict[str, Any] = {
            "subject_id": s.subject_id, "session": s.session, "recording": s.recording, "group": s.group,
            "folder": _rel(folder, qc_dir), "status": status,
            "error": (entry or {}).get("error_message"),
            "report": _rel(report, qc_dir) if report.exists() else "",
            "registration_figure": _rel(registration, qc_dir) if registration.exists() else "",
            "figures": "; ".join(_rel(f, qc_dir) for f in figures),
        }
        resolved = _read_yaml(data / "config_resolved.yaml") if data.is_dir() else None
        if resolved:
            recipes[s.output_id] = recipe(resolved)
            row["localized_with"] = recipes[s.output_id].get("version")
        mc = _read_json(data / "monte_carlo_report.json") if data.is_dir() else None
        if mc:
            mc_reports[s.output_id] = mc
            pp = mc.get("per_parcel") or {}
            cov = [float(v["coverage"]) for v in pp.values() if v.get("coverage") is not None]
            gain = [float(v["gain"]) for v in pp.values() if v.get("gain") is not None]
            row.update(mc_draws=mc.get("n_draws"), mc_sources_per_draw=mc.get("n_sources"),
                       mc_min_coverage=min(cov) if cov else None, mc_min_gain=min(gain) if gain else None,
                       mc_n_collinear=sum(1 for v in pp.values() if v.get("collinear_with")),
                       mc_n_undersampled=sum(1 for c in cov if c < COVERAGE_THRESHOLD))
        m = metrics.get(str(s.subject_id))
        if m:
            row.update({k: m.get(k) for k in METRIC_DOCS})
        recordings.append(row)

    parcels = _parcel_rows(mc_reports)
    return {"recordings": recordings, "parcels": parcels, "recipes": recipes, "mc_reports": mc_reports}


def _parcel_rows(mc_reports: Dict[str, dict]) -> List[Dict[str, Any]]:
    by_parcel: Dict[str, List[dict]] = {}
    for rep in mc_reports.values():
        for parcel, entry in (rep.get("per_parcel") or {}).items():
            by_parcel.setdefault(parcel, []).append(entry)
    order = next((list(r.get("parcels") or []) for r in mc_reports.values() if r.get("parcels")), [])
    order += sorted(p for p in by_parcel if p not in order)
    rows = []
    for parcel in order:
        entries = by_parcel.get(parcel) or []
        if not entries:
            continue

        def nums(key):
            return [float(e[key]) for e in entries if e.get(key) is not None]

        cov, gain, coll = nums("coverage"), nums("gain"), nums("max_collinearity")
        partners: Counter = Counter()
        for e in entries:
            p = e.get("collinear_with")
            for q in ([p] if isinstance(p, str) else (p or [])):
                partners[q] += 1
        partner, n_partner = partners.most_common(1)[0] if partners else (None, 0)
        n_coll = sum(1 for e in entries if e.get("collinear_with"))
        n_under = sum(1 for c in cov if c < COVERAGE_THRESHOLD)
        rows.append({
            "parcel": parcel, "n_recordings": len(entries),
            "coverage_mean": sum(cov) / len(cov) if cov else None, "coverage_min": min(cov) if cov else None,
            "gain_mean": sum(gain) / len(gain) if gain else None, "gain_min": min(gain) if gain else None,
            "gain_max": max(gain) if gain else None,
            "max_collinearity_mean": sum(coll) / len(coll) if coll else None,
            "max_collinearity_max": max(coll) if coll else None,
            "n_collinear": n_coll, "collinear_with": partner, "collinear_with_n": n_partner,
            "n_undersampled": n_under,
            "reliable": n_coll == 0 and n_under == 0,
        })
    return rows


def _recording_columns(rows: List[dict]) -> Dict[str, dict]:
    docs: Dict[str, dict] = {
        "subject_id": {"Description": "the subject, as the study config names it"},
        "session": {"Description": "the session, as the study config names it (empty when none)"},
        "recording": {"Description": "the recording number within the session (empty when none)"},
        "group": {"Description": "the subject's group"},
        "folder": {"Description": "the recording's output folder, relative to this folder "
                                  "(sub-<id>[_ses-<session>][_rec-<n>])"},
        "status": {"Description": "how the recording's localization went",
                   "Levels": {"localized": "outputs present", "failed": "the processing log records "
                              "a failure", "missing": "no outputs for this recording"}},
        "error": {"Description": "the failure the processing log records, if any"},
        "report": {"Description": "the recording's pipeline report (HTML), relative to this folder"},
        "registration_figure": {"Description": "the recording's electrode-registration check, "
                                               "relative to this folder"},
        "figures": {"Description": "the recording's per-step figures, relative to this folder, "
                                   "separated by '; '"},
        "localized_with": {"Description": "the source-localization version that localized it"},
        "mc_draws": {"Description": "Monte Carlo draws (source placements)"},
        "mc_sources_per_draw": {"Description": "sources placed per Monte Carlo draw"},
        "mc_min_coverage": {"Description": "the lowest parcel coverage (share of draws that place a "
                                           "source in the parcel)"},
        "mc_min_gain": {"Description": "the lowest parcel SNR gain over a single draw"},
        "mc_n_collinear": {"Description": f"parcels near-collinear (sensor-topography r >= "
                                          f"{COLLINEARITY_THRESHOLD}) with another"},
        "mc_n_undersampled": {"Description": f"parcels with coverage below {COVERAGE_THRESHOLD}"},
    }
    for k, v in METRIC_DOCS.items():
        docs[k] = {"Description": v[0], "Units": v[1]} if isinstance(v, tuple) else {"Description": v}
    cols = [c for c in docs if any(c in r for r in rows)]
    return {c: docs[c] for c in cols}


def _parcel_columns() -> Dict[str, dict]:
    return {
        "parcel": {"Description": "the atlas parcel (ROI)", "Standard": "element"},
        "n_recordings": {"Description": "recordings whose Monte Carlo report covers the parcel"},
        "coverage_mean": {"Description": "mean coverage: the share of draws that place a source in "
                                         "the parcel"},
        "coverage_min": {"Description": "lowest coverage over the recordings"},
        "gain_mean": {"Description": "mean SNR gain of the parcel's averaged operator over a single draw"},
        "gain_min": {"Description": "lowest gain over the recordings"},
        "gain_max": {"Description": "highest gain over the recordings"},
        "max_collinearity_mean": {"Description": "mean of the parcel's highest sensor-topography "
                                                 "correlation with another parcel"},
        "max_collinearity_max": {"Description": "the highest such correlation in any recording"},
        "n_collinear": {"Description": f"recordings in which the parcel is near-collinear "
                                       f"(r >= {COLLINEARITY_THRESHOLD}) with another"},
        "collinear_with": {"Description": "the parcel it is most often near-collinear with"},
        "collinear_with_n": {"Description": "recordings in which it is collinear with that parcel"},
        "n_undersampled": {"Description": f"recordings in which its coverage is below {COVERAGE_THRESHOLD}"},
        "reliable": {"Description": "neither near-collinear nor undersampled in any recording: its own "
                                    "value can be read; otherwise read it with its partner, or not at all"},
    }


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9_.-]+", "_", str(text).lower()).strip("_") or "study"


def write(config: Any, qc_dir: Path, metrics_df: Any = None, outlier_flags: Optional[dict] = None,
          warnings: Optional[List[str]] = None, plot_paths: Optional[List[Path]] = None,
          started: Optional[str] = None) -> Any:
    """Write the localization summary into ``qc_dir``; the checker's report, or None. Never raises."""
    if neurobox is None:
        logger.info("Localization summary not written: %s", INSTALL_HINT)
        return None
    try:
        return _write(config, Path(qc_dir), metrics_df, outlier_flags or {}, warnings or [],
                      plot_paths or [], started)
    except Exception as exc:  # noqa: BLE001 - a description must never fail the QC
        logger.warning("Localization summary not written: %s", exc)
        return None


def _write(config, qc_dir, metrics_df, outlier_flags, qc_warnings, plot_paths, started):
    from .. import __version__

    started = started or datetime.now(timezone.utc).isoformat(timespec="seconds")
    got = collect(config, qc_dir, metrics_df)
    recordings, parcels, recipes = got["recordings"], got["parcels"], got["recipes"]
    rec_cols = _recording_columns(recordings)
    _write_csv(qc_dir / RECORDINGS, list(rec_cols), recordings)
    neurobox.write_columns(qc_dir / RECORDINGS, rec_cols)
    tables = [{"path": RECORDINGS, "role": "other", "headline": True, "n_rows": len(recordings),
               "rows": "one recording of the study",
               "description": "every recording in the study config: how its localization went, where its "
                              "report and figures are, and its Monte Carlo health"}]
    if parcels:
        _write_csv(qc_dir / PARCELS, list(_parcel_columns()), parcels)
        neurobox.write_columns(qc_dir / PARCELS, _parcel_columns())
        tables.append({"path": PARCELS, "role": "descriptives", "n_rows": len(parcels),
                       "rows": "one atlas parcel, summarized over the recordings",
                       "description": "Monte Carlo coverage, gain and collinearity per parcel: which "
                                      "parcels the montage resolves"})

    caveats: List[str] = []
    distinct = {json.dumps(r, sort_keys=True, default=str) for r in recipes.values()}
    if len(distinct) > 1:
        fields = sorted({k for r in recipes.values() for k in r
                         if len({json.dumps(x.get(k), sort_keys=True, default=str)
                                 for x in recipes.values()}) > 1})
        caveats.append(f"The recordings were not all localized with the same recipe: they differ in "
                       f"{', '.join(fields)} (see each recording's config_resolved.yaml).")
    status = Counter(r["status"] for r in recordings)
    if status.get("failed") or status.get("missing"):
        caveats.append(f"Of {len(recordings)} recordings, {status.get('failed', 0)} failed and "
                       f"{status.get('missing', 0)} have no outputs.")
    unreliable = [p["parcel"] for p in parcels if not p["reliable"]]
    if unreliable:
        caveats.append(f"{len(unreliable)} of {len(parcels)} parcels are near-collinear with another or "
                       f"undersampled in at least one recording, so their own values should not be read "
                       f"alone: {', '.join(unreliable)} (parcel_coverage.csv).")
    for metric, ids in outlier_flags.items():
        caveats.append(f"QC outliers on {metric}: {', '.join(map(str, ids))}.")
    caveats += [f"QC: {w}" for w in qc_warnings]

    figures = []
    for p in plot_paths:
        p = Path(p)
        try:
            rel = p.resolve().relative_to(qc_dir.resolve()).as_posix()
        except ValueError:
            continue
        key = next((k for k in FIGURE_CAPTIONS if k in p.stem), None)
        figures.append({"path": rel, "kind": "diagnostic", "drawn_by": "analysis",
                        "caption": FIGURE_CAPTIONS.get(key, p.stem.replace("_", " ")),
                        "shows": RECORDINGS})

    first = next(iter(recipes.values()), {})
    localized = [r for r in recordings if r["status"] == "localized"]
    groups = Counter(r["group"] for r in localized if r.get("group"))
    root_name = Path(config.root_dir).name
    run_id = _default_run_id()
    analysis = {
        "id": f"eeg/other/localization/{_slug(root_name)}",
        "title": f"Source localization: {getattr(config, 'name', '') or root_name}",
        "description": "How the study's recordings were localized: the recipe, whether every recording "
                       "got the same one, how each recording went, and which atlas parcels the montage "
                       "resolves. A diagnostic summary: it tests nothing.",
        "analysis_type": "other", "modality": "eeg", "domain": "localization", "role": "diagnostic",
        "design": {"n": max(len(localized), 1), "groups": dict(sorted(groups.items()))},
        "inference": {"method": "none: a descriptive summary of the localization",
                      "correction": {"p_kind": "none", "family": "none", "statement": "no tests"}},
        "effect": {"measure": None, "reason": "a diagnostic summary: it describes how the recordings were "
                                              "localized and tests nothing"},
        "tables": tables, "caveats": caveats,
        "run": {"id": run_id, "label": "study QC"},
    }
    if figures:
        analysis["figures"] = figures
    if first.get("atlas"):
        analysis["atlas"] = {"name": str(first["atlas"])}

    settings = {"recipe": first, "recipe_consistent": len(distinct) <= 1,
                "pipeline": {"preset": getattr(config, "pipeline_preset", None),
                             "atlas": getattr(config, "pipeline_atlas", None),
                             "overrides": getattr(config, "pipeline_overrides", None)},
                "collinearity_threshold": COLLINEARITY_THRESHOLD, "coverage_threshold": COVERAGE_THRESHOLD}
    prov = neurobox.provenance(
        [{"Name": "source-localization", "Version": str(__version__)}], status="completed",
        start=started, end=datetime.now(timezone.utc).isoformat(timespec="seconds"), run_id=run_id,
        inputs=[{"path": str(Path(config.derivatives_dir)), "role": "the study's per-recording outputs"}],
        subjects={"n": len(localized), "groups": dict(sorted(groups.items()))},
        settings=json.loads(json.dumps(settings, default=str)))
    report = neurobox.write_analysis(qc_dir, analysis, prov, strict=False)
    if report.ok:
        logger.info("Localization summary: %s conforms (%d warning(s))", qc_dir, len(report.warnings))
    else:
        logger.warning("Localization summary: %s does not conform: %s", qc_dir, "; ".join(report.errors))
    for w in report.warnings:
        logger.info("  results spec warning: %s", w)
    return report
