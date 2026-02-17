"""
Study-Level Quality Control Module

Extracts per-subject pipeline metrics, detects outliers, and generates
a summary QC report (HTML + CSV) for each study folder.

Usage:
    from source_localization.study.qc import run_qc
    result = run_qc(config)
"""

import gc
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SubjectMetrics:
    """Per-subject pipeline metrics extracted from pickle outputs."""

    subject_id: str
    group: Optional[str] = None

    # Step 1: EEG info
    n_channels: Optional[int] = None
    sfreq: Optional[float] = None

    # Step 3: Source space
    n_sources: Optional[int] = None

    # Step 4: Forward solution
    forward_condition_number: Optional[float] = None
    forward_leadfield_norm_mean: Optional[float] = None
    forward_leadfield_norm_std: Optional[float] = None

    # Step 5: STC (magnitude)
    stc_n_sources: Optional[int] = None
    stc_n_times: Optional[int] = None
    stc_amp_min: Optional[float] = None
    stc_amp_max: Optional[float] = None
    stc_amp_mean: Optional[float] = None
    stc_amp_std: Optional[float] = None
    stc_amp_median: Optional[float] = None

    # Step 6: ROI timeseries (magnitude)
    n_rois: Optional[int] = None
    roi_amp_mean: Optional[float] = None
    roi_amp_std: Optional[float] = None
    sources_per_roi_mean: Optional[float] = None
    sources_per_roi_std: Optional[float] = None
    sources_per_roi_min: Optional[int] = None
    sources_per_roi_max: Optional[int] = None

    # Processing
    processing_time_sec: Optional[float] = None


@dataclass
class QCResult:
    """Study-level QC result container."""

    metrics_df: pd.DataFrame
    outlier_flags: Dict[str, List[str]]
    warnings: List[str]
    report_path: Optional[Path] = None


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

def extract_subject_metrics(
    subject_id: str,
    output_dir: Path,
    group: Optional[str] = None,
    processing_time_sec: Optional[float] = None,
) -> SubjectMetrics:
    """
    Extract pipeline metrics from a single subject's pickle outputs.

    Loads each pickle one at a time and frees memory after extraction.

    Parameters
    ----------
    subject_id : str
        Subject identifier.
    output_dir : Path
        Subject output directory (e.g., derivatives/source_localization/sub-X).
    group : str, optional
        Treatment group label.
    processing_time_sec : float, optional
        Processing time from the processing log.

    Returns
    -------
    SubjectMetrics
    """
    import pickle

    data_dir = Path(output_dir) / "pipeline" / "data"
    metrics = SubjectMetrics(
        subject_id=subject_id,
        group=group,
        processing_time_sec=processing_time_sec,
    )

    # --- Step 1: Info ---
    try:
        with open(data_dir / "step1_info.pkl", "rb") as f:
            info = pickle.load(f)
        metrics.n_channels = len(info.ch_names)
        metrics.sfreq = info["sfreq"]
        del info
    except Exception as e:
        logger.debug("step1_info.pkl failed for %s: %s", subject_id, e)

    # --- Step 3: Source space ---
    try:
        with open(data_dir / "step3_source_space.pkl", "rb") as f:
            src = pickle.load(f)
        metrics.n_sources = src[0]["nuse"]
        del src
    except Exception as e:
        logger.debug("step3_source_space.pkl failed for %s: %s", subject_id, e)

    # --- Step 4: Forward solution ---
    try:
        with open(data_dir / "step4_forward.pkl", "rb") as f:
            fwd = pickle.load(f)
        leadfield = fwd["sol"]["data"]
        metrics.forward_condition_number = float(np.linalg.cond(leadfield))
        col_norms = np.linalg.norm(leadfield, axis=0)
        metrics.forward_leadfield_norm_mean = float(np.mean(col_norms))
        metrics.forward_leadfield_norm_std = float(np.std(col_norms))
        del fwd, leadfield, col_norms
    except Exception as e:
        logger.debug("step4_forward.pkl failed for %s: %s", subject_id, e)

    # --- Step 5: STC magnitude ---
    try:
        with open(data_dir / "step5_stc_magnitude.pkl", "rb") as f:
            stc = pickle.load(f)
        data = stc.data
        metrics.stc_n_sources = data.shape[0]
        metrics.stc_n_times = data.shape[1]
        metrics.stc_amp_min = float(np.min(data))
        metrics.stc_amp_max = float(np.max(data))
        metrics.stc_amp_mean = float(np.mean(data))
        metrics.stc_amp_std = float(np.std(data))
        metrics.stc_amp_median = float(np.median(data))
        del stc, data
    except Exception as e:
        logger.debug("step5_stc_magnitude.pkl failed for %s: %s", subject_id, e)

    # --- Step 6: ROI timeseries magnitude ---
    try:
        with open(data_dir / "step6_roi_timeseries_magnitude.pkl", "rb") as f:
            roi_dict = pickle.load(f)

        metrics.n_rois = len(roi_dict)

        # Each value is a 1-D numpy array (timeseries for that ROI)
        all_vals = np.concatenate(list(roi_dict.values()))
        metrics.roi_amp_mean = float(np.mean(all_vals))
        metrics.roi_amp_std = float(np.std(all_vals))
        del all_vals

        # Sources per ROI — loaded from source_space mapping if available
        try:
            with open(data_dir / "step6_roi_timeseries.pkl", "rb") as f2:
                roi_meta = pickle.load(f2)
            if isinstance(roi_meta, dict) and any(
                isinstance(v, dict) and "source_indices" in v
                for v in roi_meta.values()
            ):
                counts = [
                    len(v["source_indices"])
                    for v in roi_meta.values()
                    if isinstance(v, dict) and "source_indices" in v
                ]
            else:
                # Fallback: equal split approximation
                counts = None
            del roi_meta
        except Exception:
            counts = None

        if counts:
            arr = np.array(counts)
            metrics.sources_per_roi_mean = float(np.mean(arr))
            metrics.sources_per_roi_std = float(np.std(arr))
            metrics.sources_per_roi_min = int(np.min(arr))
            metrics.sources_per_roi_max = int(np.max(arr))

        del roi_dict
    except Exception as e:
        logger.debug("step6_roi_timeseries_magnitude.pkl failed for %s: %s", subject_id, e)

    gc.collect()
    return metrics


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------

def detect_outliers(
    df: pd.DataFrame,
    threshold: float = 2.0,
) -> Dict[str, List[str]]:
    """
    Flag subjects whose metrics deviate > *threshold* standard deviations
    from the group mean (z-score method).

    Parameters
    ----------
    df : DataFrame
        Metrics table (one row per subject, must have ``subject_id`` column).
    threshold : float
        Z-score cutoff for flagging.

    Returns
    -------
    dict
        ``{metric_name: [list of flagged subject_ids]}``.
    """
    metrics_to_check = [
        "forward_condition_number",
        "stc_amp_mean",
        "stc_amp_max",
        "stc_n_times",
    ]

    flags: Dict[str, List[str]] = {}

    for col in metrics_to_check:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.dropna().empty or series.std() == 0:
            continue
        z = (series - series.mean()) / series.std()
        outlier_mask = z.abs() > threshold
        flagged = df.loc[outlier_mask, "subject_id"].tolist()
        if flagged:
            flags[col] = flagged

    return flags


# ---------------------------------------------------------------------------
# Consistency checks
# ---------------------------------------------------------------------------

def check_consistency(df: pd.DataFrame) -> List[str]:
    """
    Warn when metrics that should be constant across subjects actually vary.

    Parameters
    ----------
    df : DataFrame
        Metrics table.

    Returns
    -------
    list of str
        Warning messages.
    """
    warnings = []
    constant_cols = ["n_sources", "n_channels", "sfreq", "n_rois"]

    for col in constant_cols:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        unique = series.unique()
        if len(unique) > 1:
            warnings.append(
                f"{col} varies across subjects: {sorted(unique.tolist())} "
                f"(expected constant for same pipeline)"
            )

    return warnings


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def generate_qc_plots(
    df: pd.DataFrame,
    outlier_flags: Dict[str, List[str]],
    output_dir: Path,
) -> List[Path]:
    """
    Generate 6 diagnostic plots and save to *output_dir*/figures/.

    Returns list of saved figure paths.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = Path(output_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []

    # Colour map for groups
    groups = df["group"].unique() if "group" in df.columns else []
    palette = {}
    default_colours = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12"]
    for i, g in enumerate(sorted(groups)):
        palette[g] = default_colours[i % len(default_colours)]

    def _group_colors(df_):
        return [palette.get(g, "#888888") for g in df_["group"]]

    # Helper: highlight outlier bars
    def _edge_colors(df_, metric):
        flagged = set(outlier_flags.get(metric, []))
        return ["red" if sid in flagged else "none" for sid in df_["subject_id"]]

    def _edge_widths(df_, metric):
        flagged = set(outlier_flags.get(metric, []))
        return [2.0 if sid in flagged else 0.0 for sid in df_["subject_id"]]

    x_labels = df["subject_id"].tolist()
    x = np.arange(len(x_labels))

    # --- Plot 1: Source amplitude by subject ---
    if "stc_amp_mean" in df.columns:
        fig, ax = plt.subplots(figsize=(max(10, len(x_labels) * 0.4), 5))
        bars = ax.bar(
            x, df["stc_amp_mean"], color=_group_colors(df),
            edgecolor=_edge_colors(df, "stc_amp_mean"),
            linewidth=_edge_widths(df, "stc_amp_mean"),
        )
        if "stc_amp_std" in df.columns:
            ax.errorbar(x, df["stc_amp_mean"], yerr=df["stc_amp_std"],
                        fmt="none", ecolor="gray", alpha=0.5, capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Source Amplitude (mean)")
        ax.set_title("Source Amplitude by Subject")
        # Legend
        for g in sorted(palette):
            ax.bar([], [], color=palette[g], label=g)
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = fig_dir / "01_source_amplitude.png"
        fig.savefig(p, dpi=150)
        paths.append(p)
        plt.close(fig)

    # --- Plot 2: Forward condition number ---
    if "forward_condition_number" in df.columns:
        fig, ax = plt.subplots(figsize=(max(10, len(x_labels) * 0.4), 5))
        vals = df["forward_condition_number"].fillna(0)
        ax.bar(
            x, vals, color=_group_colors(df),
            edgecolor=_edge_colors(df, "forward_condition_number"),
            linewidth=_edge_widths(df, "forward_condition_number"),
        )
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Condition Number (log scale)")
        ax.set_title("Forward Model Condition Number")
        fig.tight_layout()
        p = fig_dir / "02_forward_condition.png"
        fig.savefig(p, dpi=150)
        paths.append(p)
        plt.close(fig)

    # --- Plot 3: Total timepoints per subject ---
    if "stc_n_times" in df.columns:
        fig, ax = plt.subplots(figsize=(max(10, len(x_labels) * 0.4), 5))
        ax.bar(
            x, df["stc_n_times"].fillna(0), color=_group_colors(df),
            edgecolor=_edge_colors(df, "stc_n_times"),
            linewidth=_edge_widths(df, "stc_n_times"),
        )
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Timepoints")
        ax.set_title("Total Timepoints per Subject")
        fig.tight_layout()
        p = fig_dir / "03_timepoints.png"
        fig.savefig(p, dpi=150)
        paths.append(p)
        plt.close(fig)

    # --- Plot 4: Sources per ROI ---
    if "sources_per_roi_mean" in df.columns:
        fig, ax = plt.subplots(figsize=(max(10, len(x_labels) * 0.4), 5))
        vals = df["sources_per_roi_mean"].fillna(0)
        errs = df["sources_per_roi_std"].fillna(0) if "sources_per_roi_std" in df.columns else None
        ax.bar(x, vals, color=_group_colors(df))
        if errs is not None:
            ax.errorbar(x, vals, yerr=errs, fmt="none", ecolor="gray",
                        alpha=0.5, capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Sources per ROI (mean)")
        ax.set_title("ROI Coverage: Sources per ROI")
        fig.tight_layout()
        p = fig_dir / "04_sources_per_roi.png"
        fig.savefig(p, dpi=150)
        paths.append(p)
        plt.close(fig)

    # --- Plot 5: Processing time ---
    if "processing_time_sec" in df.columns and df["processing_time_sec"].notna().any():
        fig, ax = plt.subplots(figsize=(max(10, len(x_labels) * 0.4), 5))
        ax.bar(x, df["processing_time_sec"].fillna(0), color=_group_colors(df))
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Time (seconds)")
        ax.set_title("Processing Time per Subject")
        fig.tight_layout()
        p = fig_dir / "05_processing_time.png"
        fig.savefig(p, dpi=150)
        paths.append(p)
        plt.close(fig)

    # --- Plot 6: Z-scored metrics heatmap ---
    z_cols = [
        "forward_condition_number", "stc_amp_mean", "stc_amp_max",
        "stc_amp_std", "stc_n_times", "roi_amp_mean",
    ]
    available = [c for c in z_cols if c in df.columns]
    if available:
        z_df = df[available].apply(pd.to_numeric, errors="coerce")
        z_df = (z_df - z_df.mean()) / z_df.std()
        z_df = z_df.fillna(0)

        fig, ax = plt.subplots(figsize=(max(8, len(available) * 1.2),
                                         max(6, len(x_labels) * 0.3)))
        vmax = max(3.0, z_df.abs().max().max())
        im = ax.imshow(z_df.values, aspect="auto", cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax)
        ax.set_xticks(np.arange(len(available)))
        ax.set_xticklabels(available, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(x_labels)))
        ax.set_yticklabels(x_labels, fontsize=7)
        ax.set_title("Z-Scored Metrics Heatmap")
        fig.colorbar(im, ax=ax, label="Z-score")
        fig.tight_layout()
        p = fig_dir / "06_zscore_heatmap.png"
        fig.savefig(p, dpi=150)
        paths.append(p)
        plt.close(fig)

    return paths


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

_QC_CSS = """\
body {
    font-family: Arial, sans-serif;
    margin: 40px;
    background-color: #f5f5f5;
}
.container {
    max-width: 1200px;
    margin: 0 auto;
    background-color: white;
    padding: 30px;
    box-shadow: 0 0 10px rgba(0,0,0,0.1);
}
h1 {
    color: #2c3e50;
    border-bottom: 3px solid #3498db;
    padding-bottom: 10px;
}
h2 {
    color: #34495e;
    margin-top: 30px;
    border-left: 4px solid #3498db;
    padding-left: 10px;
}
.metadata {
    background-color: #ecf0f1;
    padding: 15px;
    border-radius: 5px;
    margin: 15px 0;
}
.metric-label {
    font-weight: bold;
    color: #555;
}
.warning {
    background-color: #fff3cd;
    border-left: 4px solid #f39c12;
    padding: 10px 15px;
    margin: 8px 0;
}
.flag {
    background-color: #f8d7da;
    border-left: 4px solid #e74c3c;
    padding: 10px 15px;
    margin: 8px 0;
}
.ok {
    background-color: #d4edda;
    border-left: 4px solid #27ae60;
    padding: 10px 15px;
    margin: 8px 0;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 15px 0;
    font-size: 13px;
}
th, td {
    border: 1px solid #ddd;
    padding: 6px 8px;
    text-align: left;
}
th {
    background-color: #3498db;
    color: white;
    cursor: pointer;
}
th:hover {
    background-color: #2980b9;
}
tr:nth-child(even) {
    background-color: #f2f2f2;
}
tr.outlier {
    background-color: #fce4e4;
}
img {
    max-width: 100%;
    height: auto;
    margin: 15px 0;
    border: 1px solid #ddd;
}
"""

_SORT_JS = """\
function sortTable(table, col) {
    var rows = Array.from(table.querySelectorAll('tbody tr'));
    var asc = table.dataset.sortCol == col ? !JSON.parse(table.dataset.sortAsc || 'true') : true;
    table.dataset.sortCol = col;
    table.dataset.sortAsc = asc;
    rows.sort(function(a, b) {
        var va = a.cells[col].textContent.trim();
        var vb = b.cells[col].textContent.trim();
        var na = parseFloat(va), nb = parseFloat(vb);
        if (!isNaN(na) && !isNaN(nb)) return asc ? na - nb : nb - na;
        return asc ? va.localeCompare(vb) : vb.localeCompare(va);
    });
    var tbody = table.querySelector('tbody');
    rows.forEach(function(r) { tbody.appendChild(r); });
}
"""


def generate_qc_report(
    config: Any,
    df: pd.DataFrame,
    outlier_flags: Dict[str, List[str]],
    warnings: List[str],
    plot_paths: List[Path],
    output_path: Path,
) -> Path:
    """
    Generate an HTML QC report.

    Parameters
    ----------
    config : StudyConfig
        Study configuration.
    df : DataFrame
        Per-subject metrics.
    outlier_flags : dict
        Outlier flags from ``detect_outliers()``.
    warnings : list
        Warning messages from ``check_consistency()``.
    plot_paths : list of Path
        Paths to saved figures.
    output_path : Path
        Where to save the HTML file.

    Returns
    -------
    Path
        The output path.
    """
    import base64

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # All flagged subjects
    all_flagged = set()
    for sids in outlier_flags.values():
        all_flagged.update(sids)

    # Study info
    study_name = getattr(config, "name", "Unknown Study")
    n_subjects = len(df)
    groups = df["group"].value_counts().to_dict() if "group" in df.columns else {}
    preset = getattr(config, "pipeline_preset", "N/A")
    if preset == "N/A":
        pipeline = getattr(config, "pipeline", None)
        if pipeline and isinstance(pipeline, dict):
            preset = pipeline.get("preset", "N/A")

    # --- Build HTML ---
    parts = []
    parts.append(f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>QC Report: {study_name}</title>
<style>{_QC_CSS}</style>
<script>{_SORT_JS}</script>
</head>
<body>
<div class="container">
<h1>QC Report: {study_name}</h1>
<div class="metadata">
  <p><span class="metric-label">Date:</span> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
  <p><span class="metric-label">Preset:</span> {preset}</p>
  <p><span class="metric-label">Subjects:</span> {n_subjects}</p>
</div>
""")

    # --- Processing summary ---
    parts.append("<h2>Processing Summary</h2>")
    if "processing_time_sec" in df.columns and df["processing_time_sec"].notna().any():
        total = df["processing_time_sec"].sum()
        mean = df["processing_time_sec"].mean()
        parts.append(f'<p>Total processing time: {total:.1f}s '
                     f'(mean {mean:.1f}s/subject)</p>')

    # --- Warnings / flags ---
    parts.append("<h2>Warnings &amp; Outlier Flags</h2>")
    if not warnings and not outlier_flags:
        parts.append('<div class="ok">No warnings or outlier flags.</div>')
    for w in warnings:
        parts.append(f'<div class="warning">{w}</div>')
    for metric, sids in outlier_flags.items():
        parts.append(
            f'<div class="flag"><strong>{metric}:</strong> '
            f'{", ".join(sids)} flagged as outliers</div>'
        )

    # --- Group balance ---
    if groups:
        parts.append("<h2>Group Balance</h2><table><tr><th>Group</th><th>N</th></tr>")
        for g, n in sorted(groups.items()):
            parts.append(f"<tr><td>{g}</td><td>{n}</td></tr>")
        parts.append("</table>")

    # --- Metrics table ---
    parts.append("<h2>Per-Subject Metrics</h2>")
    display_cols = [c for c in df.columns if c != "subject_id"]
    parts.append('<table id="metricsTable"><thead><tr>')
    parts.append(f'<th onclick="sortTable(this.closest(\'table\'),0)">subject_id</th>')
    for i, col in enumerate(display_cols, 1):
        parts.append(f'<th onclick="sortTable(this.closest(\'table\'),{i})">{col}</th>')
    parts.append("</tr></thead><tbody>")
    for _, row in df.iterrows():
        cls = ' class="outlier"' if row["subject_id"] in all_flagged else ""
        parts.append(f"<tr{cls}>")
        parts.append(f"<td>{row['subject_id']}</td>")
        for col in display_cols:
            val = row[col]
            if pd.isna(val):
                parts.append("<td>N/A</td>")
            elif isinstance(val, float):
                parts.append(f"<td>{val:.4g}</td>")
            else:
                parts.append(f"<td>{val}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")

    # --- Plots ---
    if plot_paths:
        parts.append("<h2>Diagnostic Plots</h2>")
        for pp in plot_paths:
            pp = Path(pp)
            try:
                with open(pp, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                title = pp.stem.replace("_", " ").title()
                parts.append(f"<h3>{title}</h3>")
                parts.append(f'<img src="data:image/png;base64,{b64}" alt="{title}">')
            except Exception:
                parts.append(f'<p><em>Could not embed {pp.name}</em></p>')

    parts.append("</div></body></html>")

    output_path.write_text("\n".join(parts), encoding="utf-8")
    return output_path


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_qc(
    config: Any,
    study_result: Any = None,
    outlier_threshold: float = 2.0,
    output_dir: Optional[Path] = None,
    verbose: bool = False,
) -> QCResult:
    """
    Run study-level QC: extract metrics, detect outliers, generate report.

    Parameters
    ----------
    config : StudyConfig
        Study configuration (loaded from YAML).
    study_result : StudyResult, optional
        If provided, processing times are taken from here.
    outlier_threshold : float
        Z-score threshold for outlier flagging (default 2.0).
    output_dir : Path, optional
        Override the default QC output location.
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    QCResult
    """
    from .config import StudyConfig

    derivatives_dir = config.derivatives_dir
    qc_dir = Path(output_dir) if output_dir else derivatives_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    # Build processing time lookup from log or study_result
    proc_times: Dict[str, float] = {}
    if study_result is not None:
        for sr in getattr(study_result, "subject_results", []):
            if hasattr(sr, "processing_time_sec") and sr.processing_time_sec is not None:
                proc_times[sr.subject_id] = sr.processing_time_sec
    else:
        # Try loading from processing_log.json
        log_path = derivatives_dir / "processing_log.json"
        if log_path.exists():
            try:
                with open(log_path) as f:
                    log_data = json.load(f)
                for entry in log_data.get("subjects", []):
                    sid = entry.get("subject_id")
                    pt = entry.get("processing_time_sec")
                    if sid and pt is not None:
                        proc_times[sid] = pt
            except Exception as e:
                logger.warning("Could not read processing_log.json: %s", e)

    # Extract metrics for each subject
    all_metrics: List[SubjectMetrics] = []
    for i, subject in enumerate(config.subjects):
        sid = subject.subject_id
        group = subject.group
        subj_dir = config.get_subject_output_dir(subject)
        data_dir = subj_dir / "pipeline" / "data"

        if not data_dir.exists():
            if verbose:
                print(f"  [{i+1}/{len(config.subjects)}] {sid}: SKIP (no pipeline data)")
            continue

        if verbose:
            print(f"  [{i+1}/{len(config.subjects)}] {sid}: extracting metrics...")

        m = extract_subject_metrics(
            subject_id=sid,
            output_dir=subj_dir,
            group=group,
            processing_time_sec=proc_times.get(sid),
        )
        all_metrics.append(m)

    if not all_metrics:
        logger.warning("No subjects found with pipeline data — QC skipped.")
        empty_df = pd.DataFrame()
        return QCResult(
            metrics_df=empty_df,
            outlier_flags={},
            warnings=["No subjects found with pipeline data."],
        )

    # Build DataFrame
    df = pd.DataFrame([asdict(m) for m in all_metrics])

    # Detect outliers
    outlier_flags = detect_outliers(df, threshold=outlier_threshold)

    # Consistency checks
    warnings = check_consistency(df)

    # Add outlier flag column
    all_flagged = set()
    for sids in outlier_flags.values():
        all_flagged.update(sids)
    df["outlier_flag"] = df["subject_id"].isin(all_flagged)

    # Save CSV
    csv_path = qc_dir / "qc_metrics.csv"
    df.to_csv(csv_path, index=False)
    if verbose:
        print(f"  Metrics saved to {csv_path}")

    # Generate plots
    plot_paths = generate_qc_plots(df, outlier_flags, qc_dir)
    if verbose:
        print(f"  Generated {len(plot_paths)} diagnostic plots")

    # Generate HTML report
    report_path = qc_dir / "qc_report.html"
    generate_qc_report(config, df, outlier_flags, warnings, plot_paths, report_path)
    if verbose:
        print(f"  Report saved to {report_path}")

    # Summary to stdout
    n_flagged = len(all_flagged)
    if verbose or n_flagged > 0:
        print(f"\nQC Summary: {len(df)} subjects, {n_flagged} outlier(s), "
              f"{len(warnings)} warning(s)")
        for w in warnings:
            print(f"  WARNING: {w}")
        for metric, sids in outlier_flags.items():
            print(f"  OUTLIER [{metric}]: {', '.join(sids)}")

    return QCResult(
        metrics_df=df,
        outlier_flags=outlier_flags,
        warnings=warnings,
        report_path=report_path,
    )
