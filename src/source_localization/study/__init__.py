"""
Study Configuration and Batch Processing Module

This module provides tools for organizing and processing multi-subject EEG studies
with source localization.

Folder Hierarchy (BIDS-inspired):
    study_folder/
    ├── study_config.yaml        # Study-level configuration
    ├── sourcedata/              # Raw EEG files (or symlinks)
    │   ├── sub-001.set
    │   └── sub-002.set
    ├── participants.csv         # Subject metadata
    └── derivatives/             # Processed outputs
        └── source_localization/
            ├── sub-001/
            │   ├── source_timeseries/
            │   ├── roi_timeseries/
            │   ├── analysis/
            │   └── figures/
            └── group/
                ├── group_statistics.csv
                └── figures/

Usage:
    from source_localization.study import StudyConfig, process_study

    # Load or create study configuration
    config = StudyConfig.from_yaml('study_folder/study_config.yaml')

    # Process all subjects
    results = process_study(config, n_jobs=4)

    # Or process a single subject
    result = process_subject(config, 'sub-001')

Author: Claude Code
Date: 2026-01-26
"""

from .config import StudyConfig, SubjectInfo
from .batch import (
    process_study,
    process_subject,
    create_study_from_folder,
    collect_group_results,
)
from .analysis import (
    analyze_subject,
    analyze_study,
    collect_connectivity_matrices,
    compute_group_connectivity,
    DEFAULT_BANDS,
)

__all__ = [
    # Configuration
    'StudyConfig',
    'SubjectInfo',

    # Pipeline processing
    'process_study',
    'process_subject',
    'create_study_from_folder',
    'collect_group_results',

    # Analysis (MNE wrappers)
    'analyze_subject',
    'analyze_study',
    'collect_connectivity_matrices',
    'compute_group_connectivity',
    'DEFAULT_BANDS',
]
