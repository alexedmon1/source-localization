"""
Batch Processing Module for Multi-Subject Studies

Provides functions to process multiple subjects with source localization.
Analysis (connectivity, spectral) should be done with MNE-Python after pipeline runs.

Example workflow:
    # Run pipeline on all subjects
    source-localization study run study_config.yaml --jobs 4

    # Then in Python/MNE:
    import mne
    from mne_connectivity import spectral_connectivity_epochs

    epochs = mne.io.read_epochs_eeglab('derivatives/.../roi_timeseries_signed.set')
    conn = spectral_connectivity_epochs(epochs, method='coh', fmin=30, fmax=80)
"""

import json
import logging
import shutil
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .config import StudyConfig, SubjectInfo

logger = logging.getLogger(__name__)


@dataclass
class SubjectResult:
    """Result from processing a single subject."""

    subject_id: str
    success: bool
    output_dir: Optional[Path] = None
    error_message: Optional[str] = None
    processing_time_sec: Optional[float] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class StudyResult:
    """Result from processing an entire study."""

    study_name: str
    n_subjects: int
    n_successful: int
    n_failed: int
    subject_results: List[SubjectResult]
    processing_time_sec: float
    output_dir: Path

    @property
    def success_rate(self) -> float:
        """Percentage of successfully processed subjects."""
        return (self.n_successful / self.n_subjects * 100) if self.n_subjects > 0 else 0.0

    def summary(self) -> str:
        """Generate a summary of processing results."""
        lines = [
            f"Study: {self.study_name}",
            f"Subjects processed: {self.n_subjects}",
            f"Successful: {self.n_successful} ({self.success_rate:.1f}%)",
            f"Failed: {self.n_failed}",
            f"Total processing time: {self.processing_time_sec:.1f}s",
            f"Output directory: {self.output_dir}",
        ]

        if self.n_failed > 0:
            lines.append("\nFailed subjects:")
            for result in self.subject_results:
                if not result.success:
                    lines.append(f"  - {result.subject_id}: {result.error_message}")

        return "\n".join(lines)


def process_subject(
    config: StudyConfig,
    subject: Union[str, SubjectInfo],
    skip_existing: bool = True,
) -> SubjectResult:
    """
    Process a single subject with source localization.

    The pipeline produces MNE-compatible .set files with ROI time series.
    Further analysis (connectivity, spectral) should be done with MNE-Python.

    Args:
        config: Study configuration
        subject: Subject ID or SubjectInfo object
        skip_existing: Skip if output already exists

    Returns:
        SubjectResult with processing outcome
    """
    import time
    start_time = time.time()

    # Get subject info
    if isinstance(subject, str):
        subject_info = config.get_subject_by_id(subject)
        if subject_info is None:
            return SubjectResult(
                subject_id=subject,
                success=False,
                error_message=f"Subject '{subject}' not found in study configuration",
            )
    else:
        subject_info = subject

    subject_id = subject_info.subject_id
    output_dir = config.get_subject_output_dir(subject_info)

    # Check if already processed
    if skip_existing and (output_dir / "roi_timeseries" / "roi_timeseries_signed.set").exists():
        logger.info(f"Subject {subject_id}: already processed, skipping")
        return SubjectResult(
            subject_id=subject_id,
            success=True,
            output_dir=output_dir,
            metadata={'skipped': True},
        )

    try:
        # Import here to avoid circular imports
        from ..pipeline import Pipeline

        logger.info(f"Processing subject {subject_id}: {subject_info.eeg_file}")

        # Create output directories
        output_dir.mkdir(parents=True, exist_ok=True)
        roi_dir = output_dir / "roi_timeseries"
        roi_dir.mkdir(exist_ok=True)

        # Run source localization pipeline
        pipeline = Pipeline.from_preset(config.pipeline_preset, **config.pipeline_overrides)
        pipeline.run(
            eeg_file=str(subject_info.eeg_file),
            output_dir=str(output_dir / "pipeline"),
            include_spectral=False,
            include_visualization=False,
        )

        # Copy ROI timeseries to standard location
        pipeline_data_dir = output_dir / "pipeline" / "data"
        if pipeline_data_dir.exists():
            for f in pipeline_data_dir.glob("roi_timeseries_*.set"):
                target = roi_dir / f.name
                if not target.exists():
                    shutil.copy2(f, target)

        # Clean up to prevent memory leaks between subjects
        del pipeline
        import gc
        import matplotlib.pyplot as plt
        plt.close('all')  # Close any lingering figures
        gc.collect()

        processing_time = time.time() - start_time
        logger.info(f"Subject {subject_id}: completed in {processing_time:.1f}s")

        return SubjectResult(
            subject_id=subject_id,
            success=True,
            output_dir=output_dir,
            processing_time_sec=processing_time,
            metadata={
                'group': subject_info.group,
                'eeg_file': str(subject_info.eeg_file),
            },
        )

    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Subject {subject_id}: failed - {error_msg}")
        logger.debug(traceback.format_exc())

        # Clean up even on failure
        import gc
        import matplotlib.pyplot as plt
        plt.close('all')
        gc.collect()

        return SubjectResult(
            subject_id=subject_id,
            success=False,
            output_dir=output_dir,
            error_message=error_msg,
            processing_time_sec=processing_time,
        )


def process_study(
    config: StudyConfig,
    n_jobs: int = 1,
    skip_existing: bool = True,
    subjects: Optional[List[str]] = None,
    progress_callback: Optional[callable] = None,
) -> StudyResult:
    """
    Process all subjects in a study.

    Args:
        config: Study configuration
        n_jobs: Number of parallel jobs (1 for sequential)
        skip_existing: Skip subjects with existing outputs
        subjects: Optional list of subject IDs to process (default: all)
        progress_callback: Optional callback function(completed, total, subject_id, success)

    Returns:
        StudyResult with all processing outcomes
    """
    import time
    start_time = time.time()

    # Validate configuration
    errors = config.validate()
    if errors:
        raise ValueError(f"Invalid study configuration:\n" + "\n".join(errors))

    # Determine subjects to process
    if subjects:
        subjects_to_process = [
            s for s in config.subjects if s.subject_id in subjects
        ]
    else:
        subjects_to_process = config.subjects

    n_subjects = len(subjects_to_process)
    logger.info(f"Processing {n_subjects} subjects from study '{config.name}'")

    # Create output directories
    config.derivatives_dir.mkdir(parents=True, exist_ok=True)
    config.group_dir.mkdir(exist_ok=True)

    # Process subjects
    results = []

    if n_jobs == 1:
        # Sequential processing
        for i, subject in enumerate(subjects_to_process):
            result = process_subject(config, subject, skip_existing)
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, n_subjects, subject.subject_id, result.success)
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            future_to_subject = {
                executor.submit(process_subject, config, subject, skip_existing): subject
                for subject in subjects_to_process
            }

            completed = 0
            for future in as_completed(future_to_subject):
                subject = future_to_subject[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = SubjectResult(
                        subject_id=subject.subject_id,
                        success=False,
                        error_message=f"Process error: {e}",
                    )
                results.append(result)
                completed += 1

                if progress_callback:
                    progress_callback(completed, n_subjects, subject.subject_id, result.success)

    # Save processing log
    processing_time = time.time() - start_time
    n_successful = sum(1 for r in results if r.success)
    n_failed = n_subjects - n_successful

    processing_log = {
        'study_name': config.name,
        'timestamp': datetime.now().isoformat(),
        'n_subjects': n_subjects,
        'n_successful': n_successful,
        'n_failed': n_failed,
        'processing_time_sec': processing_time,
        'pipeline_preset': config.pipeline_preset,
        'subjects': [
            {
                'subject_id': r.subject_id,
                'success': r.success,
                'output_dir': str(r.output_dir) if r.output_dir else None,
                'error_message': r.error_message,
                'processing_time_sec': r.processing_time_sec,
            }
            for r in results
        ],
    }

    log_path = config.derivatives_dir / "processing_log.json"
    with open(log_path, 'w') as f:
        json.dump(processing_log, f, indent=2)

    logger.info(f"Processing complete: {n_successful}/{n_subjects} successful")

    return StudyResult(
        study_name=config.name,
        n_subjects=n_subjects,
        n_successful=n_successful,
        n_failed=n_failed,
        subject_results=results,
        processing_time_sec=processing_time,
        output_dir=config.derivatives_dir,
    )


def create_study_from_folder(
    folder: Union[str, Path],
    name: Optional[str] = None,
    participants_file: Optional[str] = None,
    preset: str = "roi_based_ellipsoid",
    save_config: bool = True,
) -> StudyConfig:
    """
    Create a study configuration by scanning a folder for EEG files.

    Args:
        folder: Root folder containing EEG data
        name: Study name (defaults to folder name)
        participants_file: Path to CSV with subject metadata
        preset: Pipeline preset to use
        save_config: Whether to save study_config.yaml

    Returns:
        StudyConfig ready for processing
    """
    folder = Path(folder)

    config = StudyConfig.from_folder(
        folder=folder,
        name=name,
        participants_file=participants_file,
        group_from_folder=True,
    )
    config.pipeline_preset = preset

    if save_config:
        config.save_yaml()
        logger.info(f"Saved study configuration to {folder / 'study_config.yaml'}")

    return config


def collect_group_results(config: StudyConfig) -> 'pd.DataFrame':
    """
    Collect processing results for group-level summary.

    This creates a simple CSV with subject metadata and processing status.
    For actual analysis (band power, connectivity), use MNE-Python directly
    on the .set files.

    Args:
        config: Study configuration

    Returns:
        DataFrame with subject info and processing status
    """
    import pandas as pd

    rows = []
    for subject in config.subjects:
        output_dir = config.get_subject_output_dir(subject)
        roi_file = output_dir / "roi_timeseries" / "roi_timeseries_signed.set"

        rows.append({
            'subject_id': subject.subject_id,
            'group': subject.group,
            'processed': roi_file.exists(),
            'eeg_file': str(subject.eeg_file),
            'roi_timeseries_file': str(roi_file) if roi_file.exists() else None,
        })

    df = pd.DataFrame(rows)

    # Save to group directory
    group_file = config.group_dir / "subjects.csv"
    df.to_csv(group_file, index=False)
    logger.info(f"Saved subject list to {group_file}")

    return df
