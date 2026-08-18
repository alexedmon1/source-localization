"""
Validation metrics for source localization accuracy assessment.

This module computes quantitative metrics for validating source localization
results against known ground truth positions. Metrics include:

- Localization error (Euclidean distance)
- ROI classification accuracy (exact and hierarchical)
- Point Spread Function (PSF) analysis
- Crosstalk Function (CTF) analysis
- Amplitude recovery
- Depth bias
- Two-dipole resolution

Functions
---------
compute_localization_error
    Euclidean distance between true and estimated positions
compute_roi_classification_accuracy
    Binary ROI classification (correct/incorrect)
compute_hierarchical_roi_accuracy
    Fuzzy ROI matching with spatial proximity
compute_point_spread_function
    PSF metrics (FWHM, spread volume, centroid shift)
compute_crosstalk_function
    CTF metrics (power leakage to incorrect ROIs)
compute_amplitude_recovery
    Dipole amplitude recovery assessment
compute_depth_bias
    Systematic errors vs source depth
compute_two_dipole_resolution
    Multi-source resolution capability
summarize_validation_results
    Aggregate statistics across multiple tests

Examples
--------
>>> from source_localization.validation.metrics import (
...     compute_localization_error,
...     compute_roi_classification_accuracy
... )
>>> error = compute_localization_error([0, 0, 5], [0.2, -0.1, 5.3])
>>> print(f"Localization error: {error:.2f} mm")
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr

__all__ = [
    'EXTERIOR_ROI_ID',
    'compute_localization_error',
    'compute_roi_classification_accuracy',
    'compute_hierarchical_roi_accuracy',
    'compute_point_spread_function',
    'compute_crosstalk_function',
    'compute_amplitude_recovery',
    'compute_depth_bias',
    'compute_two_dipole_resolution',
    'summarize_validation_results',
    'compute_source_depths',
    'compute_depth_stratified_error',
    'DEFAULT_DEPTH_BINS_MM'
]

# Default depth bins (distance to nearest electrode, in mm)
# Note: Very few brain sources are <1mm from an electrode, so bins start at 1mm
DEFAULT_DEPTH_BINS_MM = [
    (1, 2, '1-2mm'),
    (2, 3, '2-3mm'),
    (3, 4, '3-4mm'),
    (4, 5, '4-5mm'),
    (5, float('inf'), '5+mm')
]


def compute_localization_error(
    true_position_mm: np.ndarray,
    estimated_position_mm: np.ndarray
) -> float:
    """
    Compute Euclidean distance between true and estimated source positions.

    Parameters
    ----------
    true_position_mm : array-like, shape (3,)
        True dipole position [x, y, z] in mm
    estimated_position_mm : array-like, shape (3,)
        Estimated peak source position [x, y, z] in mm

    Returns
    -------
    error_mm : float
        Localization error in millimeters
    """
    true_pos = np.asarray(true_position_mm)
    est_pos = np.asarray(estimated_position_mm)

    error_mm = np.linalg.norm(true_pos - est_pos)

    return error_mm


def compute_orientation_error(
    true_orientation: np.ndarray,
    estimated_orientation: np.ndarray,
    degrees: bool = True
) -> float:
    """
    Angle between a true and a recovered dipole moment direction.

    Dipole polarity is not identifiable: a source pointing one way with a
    positive time course and one pointing the opposite way with a negative
    time course produce the same sensor data. So the angle is taken between
    *axes*, not vectors, via the absolute value of the dot product. This
    matches the convention already used for signed source fidelity, where
    |r| is reported rather than r for the same reason.

    The consequence is that the error is bounded at 90 degrees, and chance
    performance is not 90 but the mean angle between random axes in 3-D,
    which is 57.3 degrees. Compare against that, not against 90.

    Parameters
    ----------
    true_orientation : array-like, shape (3,)
        Ground-truth dipole moment direction. Normalised internally.
    estimated_orientation : array-like, shape (3,)
        Recovered dipole moment direction. Normalised internally.
    degrees : bool, default=True
        Return degrees rather than radians.

    Returns
    -------
    error : float
        Angle in [0, 90] degrees (or [0, pi/2] radians). NaN if either
        vector has effectively zero length, which happens for source spaces
        that carry no normals.

    Examples
    --------
    >>> compute_orientation_error([1, 0, 0], [1, 0, 0])
    0.0
    >>> compute_orientation_error([1, 0, 0], [-1, 0, 0])  # polarity ignored
    0.0
    >>> round(compute_orientation_error([1, 0, 0], [0, 1, 0]), 6)
    90.0
    """
    true_vec = np.asarray(true_orientation, dtype=float).ravel()
    est_vec = np.asarray(estimated_orientation, dtype=float).ravel()

    true_norm = np.linalg.norm(true_vec)
    est_norm = np.linalg.norm(est_vec)
    if true_norm < 1e-12 or est_norm < 1e-12:
        return float('nan')

    cos_angle = np.dot(true_vec / true_norm, est_vec / est_norm)
    # abs() folds the antipodal direction onto the same axis
    cos_angle = np.clip(abs(cos_angle), 0.0, 1.0)

    angle = np.arccos(cos_angle)
    return float(np.degrees(angle)) if degrees else float(angle)


# Mean angle between two random axes in 3-D, in degrees. This is the chance
# level for compute_orientation_error, and the number any orientation result
# has to beat to mean anything.
RANDOM_AXIS_ANGLE_DEG = 57.2958


def extract_dipole_orientation(
    source_activity: np.ndarray,
    n_comp: int = 3
) -> np.ndarray:
    """
    Recover a dipole moment direction from a 3-component source time course.

    Uses the dominant left singular vector, i.e. the direction explaining the
    most variance over time, which is what the pipeline already uses to
    collapse free-orientation estimates to a signed scalar.

    Parameters
    ----------
    source_activity : ndarray, shape (n_comp, n_times) or (n_comp,)
        Source activity at one location.
    n_comp : int, default=3
        Components per source. With 1 the orientation is fixed by
        construction and cannot be recovered from the data, so NaNs are
        returned.

    Returns
    -------
    orientation : ndarray, shape (3,)
        Unit vector, or NaNs when it is not recoverable.
    """
    if n_comp != 3:
        return np.full(3, np.nan)

    activity = np.asarray(source_activity, dtype=float)
    if activity.ndim == 1:
        activity = activity[:, np.newaxis]
    if activity.shape[0] != 3:
        raise ValueError(
            f"expected 3 components, got {activity.shape[0]}"
        )

    if not np.any(np.abs(activity) > 0):
        return np.full(3, np.nan)

    U, _, _ = np.linalg.svd(activity, full_matrices=False)
    return U[:, 0]


# Exterior ROI ID - sources here are outside the brain and should be excluded
EXTERIOR_ROI_ID = 0


def compute_roi_classification_accuracy(
    true_roi: int,
    estimated_roi: int,
    exclude_exterior: bool = True
) -> Optional[bool]:
    """
    Check if the estimated peak source is in the correct ROI.

    Parameters
    ----------
    true_roi : int
        True ROI ID where dipole was placed
    estimated_roi : int
        ROI ID of the estimated peak source
    exclude_exterior : bool, default=True
        If True, returns None when either ROI is Exterior (ID 0),
        indicating this trial should be excluded from accuracy calculations.

    Returns
    -------
    correct : bool or None
        True if ROI classification is correct, False if incorrect,
        None if either ROI is Exterior and exclude_exterior=True
    """
    # Exterior (ROI 0) should not be a valid ROI for validation
    if exclude_exterior:
        if true_roi == EXTERIOR_ROI_ID or estimated_roi == EXTERIOR_ROI_ID:
            return None  # Exclude from accuracy calculation

    return true_roi == estimated_roi


def compute_hierarchical_roi_accuracy(
    true_roi: int,
    estimated_roi: int,
    roi_distance_matrix: np.ndarray,
    roi_ids: np.ndarray,
    adjacency_threshold_mm: float = 2.0,
    nearby_threshold_mm: float = 3.0,
    close_threshold_mm: float = 5.0
) -> Dict[str, bool]:
    """
    Compute hierarchical ROI classification accuracy with fuzzy matching.

    Uses spatial proximity between ROI centroids to allow nearby ROIs to count
    as partially correct. This reflects the practical reality that small
    localization errors across ROI boundaries shouldn't be penalized as heavily.

    Parameters
    ----------
    true_roi : int
        True ROI ID where dipole was placed
    estimated_roi : int
        ROI ID of the estimated peak source
    roi_distance_matrix : ndarray, shape (n_rois, n_rois)
        Distance matrix between ROI centroids in mm
    roi_ids : ndarray, shape (n_rois,)
        ROI IDs corresponding to distance matrix indices
    adjacency_threshold_mm : float, default=2.0
        Maximum distance (mm) to consider ROIs adjacent
    nearby_threshold_mm : float, default=3.0
        Maximum distance (mm) to consider ROIs nearby
    close_threshold_mm : float, default=5.0
        Maximum distance (mm) to consider ROIs close

    Returns
    -------
    accuracy_levels : dict
        Dictionary with hierarchical accuracy flags:
        - 'exact': True if estimated_roi == true_roi
        - 'adjacent': True if distance <= adjacency_threshold_mm
        - 'nearby': True if distance <= nearby_threshold_mm
        - 'close': True if distance <= close_threshold_mm
        - 'same_hemisphere': True if both ROIs in same hemisphere
        - 'distance_mm': Distance between ROI centroids

    Notes
    -----
    Hierarchy levels (most to least stringent):
    1. Exact match: Same ROI (0 mm)
    2. Adjacent: Touching/bordering ROIs (≤2 mm between centroids)
    3. Nearby: Close ROIs (≤3 mm between centroids)
    4. Close: Moderately close ROIs (≤5 mm between centroids)
    5. Same hemisphere: Left vs right brain

    Example
    -------
    >>> # Thalamus_L (ROI 5) neighbors:
    >>> # - Thalamus_R: 1.0 mm (adjacent)
    >>> # - Corpus_Callosum_Splenium_L: 1.9 mm (adjacent)
    >>> # - Hypothalamus_L: 2.2 mm (nearby but not adjacent)
    >>> accuracy = compute_hierarchical_roi_accuracy(
    ...     true_roi=5,  # Thalamus_L
    ...     estimated_roi=29,  # Hypothalamus_L
    ...     roi_distance_matrix=dist_matrix,
    ...     roi_ids=roi_ids
    ... )
    >>> print(accuracy)
    {'exact': False, 'adjacent': False, 'nearby': True,
     'close': True, 'same_hemisphere': True, 'distance_mm': 2.2}
    """
    # Exact match
    exact = (true_roi == estimated_roi)

    # If exact match, all levels are True
    if exact:
        return {
            'exact': True,
            'adjacent': True,
            'nearby': True,
            'close': True,
            'same_hemisphere': True,
            'distance_mm': 0.0
        }

    # Find indices in distance matrix
    try:
        true_idx = np.where(roi_ids == true_roi)[0][0]
        est_idx = np.where(roi_ids == estimated_roi)[0][0]
    except IndexError:
        # ROI not found in distance matrix
        return {
            'exact': False,
            'adjacent': False,
            'nearby': False,
            'close': False,
            'same_hemisphere': False,
            'distance_mm': np.nan
        }

    # Get distance between ROIs
    distance_mm = roi_distance_matrix[true_idx, est_idx]

    # Check hierarchy levels
    adjacent = (distance_mm <= adjacency_threshold_mm)
    nearby = (distance_mm <= nearby_threshold_mm)
    close = (distance_mm <= close_threshold_mm)

    # Check hemisphere (heuristic: odd ROI IDs = left, even = right)
    # This assumes ROI IDs follow Antwerp atlas convention
    # ROI 1 = Exterior (no hemisphere)
    # ROIs 2-23: Left hemisphere (even IDs)
    # ROIs 24-46: Right hemisphere (even IDs)
    if true_roi == 1 or estimated_roi == 1:
        same_hemisphere = False  # Exterior has no hemisphere
    elif true_roi <= 23 and estimated_roi <= 23:
        same_hemisphere = True  # Both left
    elif true_roi >= 24 and estimated_roi >= 24:
        same_hemisphere = True  # Both right
    else:
        same_hemisphere = False  # Different hemispheres

    accuracy_levels = {
        'exact': exact,
        'adjacent': adjacent,
        'nearby': nearby,
        'close': close,
        'same_hemisphere': same_hemisphere,
        'distance_mm': float(distance_mm)
    }

    return accuracy_levels


def compute_point_spread_function(
    source_power: np.ndarray,
    source_positions_mm: np.ndarray,
    peak_idx: int,
    threshold_fraction: float = 0.5
) -> Dict[str, float]:
    """
    Compute point-spread function (PSF) metrics.

    PSF quantifies how focal the source reconstruction is around the true source.

    Parameters
    ----------
    source_power : ndarray, shape (n_sources,)
        Reconstructed source power (time-averaged)
    source_positions_mm : ndarray, shape (n_sources, 3)
        Source positions in mm
    peak_idx : int
        Index of peak source
    threshold_fraction : float
        Fraction of peak power to define PSF extent (default: 0.5 = FWHM)

    Returns
    -------
    psf_metrics : dict
        - 'fwhm_mm': Full width at half maximum
        - 'n_sources_active': Number of sources above threshold
        - 'spread_volume_mm3': Approximate volume of active sources
        - 'centroid_shift_mm': Distance from peak to power-weighted centroid
    """
    peak_power = source_power[peak_idx]
    threshold = threshold_fraction * peak_power

    # Find sources above threshold
    active_mask = source_power >= threshold
    active_indices = np.where(active_mask)[0]
    n_sources_active = len(active_indices)

    if n_sources_active == 0:
        return {
            'fwhm_mm': 0.0,
            'n_sources_active': 0,
            'spread_volume_mm3': 0.0,
            'centroid_shift_mm': 0.0
        }

    # Compute distances from peak to all active sources
    peak_pos = source_positions_mm[peak_idx]
    active_positions = source_positions_mm[active_mask]
    distances = np.linalg.norm(active_positions - peak_pos, axis=1)

    # FWHM: average distance of sources at half-max
    fwhm_mm = np.mean(distances)

    # Spread volume: approximate as sphere with radius = max distance
    max_distance = np.max(distances)
    spread_volume_mm3 = (4/3) * np.pi * (max_distance ** 3)

    # Centroid shift: power-weighted centroid vs peak position
    active_power = source_power[active_mask]
    power_weighted_centroid = np.average(
        active_positions,
        axis=0,
        weights=active_power
    )
    centroid_shift_mm = np.linalg.norm(power_weighted_centroid - peak_pos)

    psf_metrics = {
        'fwhm_mm': fwhm_mm,
        'n_sources_active': n_sources_active,
        'spread_volume_mm3': spread_volume_mm3,
        'centroid_shift_mm': centroid_shift_mm,
        'max_spread_mm': max_distance
    }

    return psf_metrics


def compute_crosstalk_function(
    source_power: np.ndarray,
    roi_mapping: np.ndarray,
    true_roi: int,
    top_n: int = 5
) -> Dict[str, any]:
    """
    Compute crosstalk function (CTF) metrics.

    CTF quantifies how much activity "leaks" into incorrect ROIs.

    Parameters
    ----------
    source_power : ndarray, shape (n_sources,)
        Reconstructed source power
    roi_mapping : ndarray, shape (n_sources,)
        ROI assignment for each source
    true_roi : int
        True ROI where dipole was placed
    top_n : int
        Number of top ROIs to report

    Returns
    -------
    ctf_metrics : dict
        - 'true_roi_power_fraction': Fraction of total power in correct ROI
        - 'top_rois': List of (roi_id, power_fraction) tuples
        - 'n_rois_active': Number of ROIs with >1% of total power
        - 'ctf_ratio': Ratio of true ROI power to next strongest ROI
    """
    # Compute total power per ROI
    unique_rois = np.unique(roi_mapping)
    roi_powers = {}

    for roi_id in unique_rois:
        roi_mask = roi_mapping == roi_id
        roi_powers[roi_id] = np.sum(source_power[roi_mask])

    total_power = np.sum(source_power)

    # Power fraction in true ROI
    true_roi_power = roi_powers.get(true_roi, 0.0)
    true_roi_power_fraction = true_roi_power / total_power if total_power > 0 else 0.0

    # Sort ROIs by power
    sorted_rois = sorted(
        roi_powers.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Top N ROIs with power fractions
    top_rois = [
        (roi_id, power / total_power if total_power > 0 else 0.0)
        for roi_id, power in sorted_rois[:top_n]
    ]

    # Number of significantly active ROIs (>1% of total power)
    n_rois_active = sum(
        1 for roi_id, power in roi_powers.items()
        if (power / total_power) > 0.01
    )

    # CTF ratio: true ROI power / next strongest ROI power
    if len(sorted_rois) >= 2:
        # Find next strongest ROI (that's not the true ROI)
        other_rois = [roi for roi, _ in sorted_rois if roi != true_roi]
        if other_rois:
            next_roi = other_rois[0]
            next_power = roi_powers[next_roi]
            ctf_ratio = true_roi_power / next_power if next_power > 0 else np.inf
        else:
            ctf_ratio = np.inf
    else:
        ctf_ratio = np.inf

    ctf_metrics = {
        'true_roi_power_fraction': true_roi_power_fraction,
        'top_rois': top_rois,
        'n_rois_active': n_rois_active,
        'ctf_ratio': ctf_ratio
    }

    return ctf_metrics


def compute_amplitude_recovery(
    true_amplitude_nAm: float,
    estimated_peak_power: float,
    estimated_total_power: float
) -> Dict[str, float]:
    """
    Assess amplitude recovery accuracy.

    Parameters
    ----------
    true_amplitude_nAm : float
        True dipole amplitude in nAm
    estimated_peak_power : float
        Peak source power from inverse solution
    estimated_total_power : float
        Total power across all sources

    Returns
    -------
    amp_metrics : dict
        - 'amplitude_ratio': Estimated / True amplitude
        - 'amplitude_error_percent': Percentage error
    """
    # Note: This is approximate since inverse solution units differ from nAm
    # We can't directly compare absolute values, but can assess relative recovery

    # For now, return ratios (qualitative assessment)
    amp_metrics = {
        'estimated_peak_power': estimated_peak_power,
        'estimated_total_power': estimated_total_power,
        'peak_to_total_ratio': estimated_peak_power / estimated_total_power if estimated_total_power > 0 else 0.0
    }

    return amp_metrics


def compute_depth_bias(
    localization_errors_mm: np.ndarray,
    source_depths_mm: np.ndarray
) -> Dict[str, float]:
    """
    Quantify depth bias in localization errors.

    Depth bias occurs when deeper sources are systematically mislocalized
    more than superficial sources.

    Parameters
    ----------
    localization_errors_mm : ndarray, shape (n_tests,)
        Localization errors for each test dipole
    source_depths_mm : ndarray, shape (n_tests,)
        True depth of each test dipole (e.g., Z coordinate)

    Returns
    -------
    depth_metrics : dict
        - 'correlation': Pearson correlation between depth and error
        - 'slope': Linear regression slope (mm error per mm depth)
        - 'superficial_error_mm': Mean error for superficial sources
        - 'deep_error_mm': Mean error for deep sources
    """
    # Pearson correlation
    if len(localization_errors_mm) > 1:
        corr, p_value = pearsonr(source_depths_mm, localization_errors_mm)
    else:
        corr, p_value = np.nan, np.nan

    # Linear regression slope
    if len(localization_errors_mm) > 1:
        slope, intercept = np.polyfit(source_depths_mm, localization_errors_mm, 1)
    else:
        slope, intercept = np.nan, np.nan

    # Split by depth (median split)
    median_depth = np.median(source_depths_mm)
    superficial_mask = source_depths_mm >= median_depth
    deep_mask = source_depths_mm < median_depth

    superficial_error_mm = np.mean(localization_errors_mm[superficial_mask]) if np.any(superficial_mask) else np.nan
    deep_error_mm = np.mean(localization_errors_mm[deep_mask]) if np.any(deep_mask) else np.nan

    depth_metrics = {
        'correlation': corr,
        'p_value': p_value,
        'slope': slope,
        'intercept': intercept,
        'superficial_error_mm': superficial_error_mm,
        'deep_error_mm': deep_error_mm,
        'median_depth_mm': median_depth
    }

    return depth_metrics


def compute_two_dipole_resolution(
    source_power: np.ndarray,
    source_positions_mm: np.ndarray,
    true_pos1_mm: np.ndarray,
    true_pos2_mm: np.ndarray,
    min_separation_mm: float = 2.0
) -> Dict[str, any]:
    """
    Assess ability to resolve two simultaneous dipoles.

    Parameters
    ----------
    source_power : ndarray, shape (n_sources,)
        Reconstructed source power
    source_positions_mm : ndarray, shape (n_sources, 3)
        Source positions
    true_pos1_mm, true_pos2_mm : array-like, shape (3,)
        True positions of the two dipoles
    min_separation_mm : float
        Minimum separation to consider peaks resolved

    Returns
    -------
    resolution_metrics : dict
        - 'resolved': Whether two distinct peaks were found
        - 'n_peaks': Number of detected peaks
        - 'peak1_error_mm': Error for first peak
        - 'peak2_error_mm': Error for second peak
        - 'estimated_separation_mm': Distance between estimated peaks
        - 'true_separation_mm': True dipole separation
    """
    true_pos1 = np.asarray(true_pos1_mm)
    true_pos2 = np.asarray(true_pos2_mm)
    true_separation_mm = np.linalg.norm(true_pos1 - true_pos2)

    # Find local maxima (simple approach: top 2 peaks)
    sorted_indices = np.argsort(source_power)[::-1]

    # Get top 2 peaks
    peak1_idx = sorted_indices[0]
    peak1_pos = source_positions_mm[peak1_idx]

    # Find second peak that's far enough from first peak
    peak2_idx = None
    for idx in sorted_indices[1:]:
        candidate_pos = source_positions_mm[idx]
        if np.linalg.norm(candidate_pos - peak1_pos) >= min_separation_mm:
            peak2_idx = idx
            break

    if peak2_idx is None:
        # No second peak found - unresolved
        return {
            'resolved': False,
            'n_peaks': 1,
            'peak1_error_mm': min(
                np.linalg.norm(peak1_pos - true_pos1),
                np.linalg.norm(peak1_pos - true_pos2)
            ),
            'peak2_error_mm': np.nan,
            'estimated_separation_mm': 0.0,
            'true_separation_mm': true_separation_mm
        }

    peak2_pos = source_positions_mm[peak2_idx]
    estimated_separation_mm = np.linalg.norm(peak1_pos - peak2_pos)

    # Match peaks to true dipoles (assign based on minimum distance)
    dist_11 = np.linalg.norm(peak1_pos - true_pos1)
    dist_12 = np.linalg.norm(peak1_pos - true_pos2)
    dist_21 = np.linalg.norm(peak2_pos - true_pos1)
    dist_22 = np.linalg.norm(peak2_pos - true_pos2)

    # Hungarian assignment (simple case: two points)
    if (dist_11 + dist_22) < (dist_12 + dist_21):
        # peak1 -> dipole1, peak2 -> dipole2
        peak1_error_mm = dist_11
        peak2_error_mm = dist_22
    else:
        # peak1 -> dipole2, peak2 -> dipole1
        peak1_error_mm = dist_12
        peak2_error_mm = dist_21

    resolution_metrics = {
        'resolved': True,
        'n_peaks': 2,
        'peak1_error_mm': peak1_error_mm,
        'peak2_error_mm': peak2_error_mm,
        'estimated_separation_mm': estimated_separation_mm,
        'true_separation_mm': true_separation_mm,
        'separation_error_mm': abs(estimated_separation_mm - true_separation_mm)
    }

    return resolution_metrics


def summarize_validation_results(
    results: List[Dict]
) -> Dict[str, any]:
    """
    Compute summary statistics across multiple validation tests.

    Parameters
    ----------
    results : list of dict
        List of per-test validation results

    Returns
    -------
    summary : dict
        Summary statistics (mean, median, std, percentiles)
    """
    # Extract metrics
    localization_errors = [r['localization_error_mm'] for r in results if 'localization_error_mm' in r]
    roi_correct = [r['roi_correct'] for r in results if 'roi_correct' in r]
    psf_fwhm = [r['psf_metrics']['fwhm_mm'] for r in results if 'psf_metrics' in r]
    ctf_ratios = [r['ctf_metrics']['ctf_ratio'] for r in results if 'ctf_metrics' in r and np.isfinite(r['ctf_metrics']['ctf_ratio'])]

    summary = {
        'n_tests': len(results),
        'localization_error': {
            'mean_mm': float(np.mean(localization_errors)) if localization_errors else np.nan,
            'median_mm': float(np.median(localization_errors)) if localization_errors else np.nan,
            'std_mm': float(np.std(localization_errors)) if localization_errors else np.nan,
            'min_mm': float(np.min(localization_errors)) if localization_errors else np.nan,
            'max_mm': float(np.max(localization_errors)) if localization_errors else np.nan,
            'percentile_25': float(np.percentile(localization_errors, 25)) if localization_errors else np.nan,
            'percentile_75': float(np.percentile(localization_errors, 75)) if localization_errors else np.nan,
            'percentile_95': float(np.percentile(localization_errors, 95)) if localization_errors else np.nan,
        },
        'roi_classification': {
            'accuracy': float(np.mean(roi_correct)) if roi_correct else np.nan,
            'n_correct': int(np.sum(roi_correct)) if roi_correct else 0,
            'n_incorrect': int(len(roi_correct) - np.sum(roi_correct)) if roi_correct else 0
        },
        'psf': {
            'mean_fwhm_mm': float(np.mean(psf_fwhm)) if psf_fwhm else np.nan,
            'median_fwhm_mm': float(np.median(psf_fwhm)) if psf_fwhm else np.nan,
        },
        'ctf': {
            'mean_ratio': float(np.mean(ctf_ratios)) if ctf_ratios else np.nan,
            'median_ratio': float(np.median(ctf_ratios)) if ctf_ratios else np.nan,
        }
    }

    return summary


def compute_source_depths(
    source_coords_mm: np.ndarray,
    electrode_coords_mm: np.ndarray,
    method: str = 'min'
) -> np.ndarray:
    """
    Compute depth of each source from the electrode array.

    Depth is defined as the minimum (or mean/weighted) distance from each source
    to the nearest electrode. This provides a standardized measure of how "deep"
    each source is relative to the recording surface.

    Parameters
    ----------
    source_coords_mm : ndarray, shape (n_sources, 3)
        Source positions [x, y, z] in mm
    electrode_coords_mm : ndarray, shape (n_electrodes, 3)
        Electrode positions [x, y, z] in mm
    method : str, default='min'
        Method for computing depth:
        - 'min': Minimum distance to any electrode (default, recommended)
        - 'mean': Mean distance to all electrodes
        - 'weighted': Distance to nearest electrode weighted by 1/r^2

    Returns
    -------
    depths_mm : ndarray, shape (n_sources,)
        Depth of each source from the electrode array in mm

    Examples
    --------
    >>> source_coords = np.array([[0, 0, 5], [0, 0, 3], [0, 0, 1]])
    >>> electrode_coords = np.array([[0, 0, 6.5], [1, 0, 6.5], [-1, 0, 6.5]])
    >>> depths = compute_source_depths(source_coords, electrode_coords)
    >>> print(f"Depths: {depths}")  # [1.5, 3.5, 5.5] mm approximately
    """
    source_coords_mm = np.asarray(source_coords_mm)
    electrode_coords_mm = np.asarray(electrode_coords_mm)

    # Compute distance matrix: (n_sources, n_electrodes)
    distances = cdist(source_coords_mm, electrode_coords_mm)

    if method == 'min':
        # Minimum distance to any electrode
        depths_mm = np.min(distances, axis=1)
    elif method == 'mean':
        # Mean distance to all electrodes
        depths_mm = np.mean(distances, axis=1)
    elif method == 'weighted':
        # Weighted average with 1/r^2 weighting (closer electrodes matter more)
        # Avoid division by zero
        weights = 1.0 / (distances ** 2 + 1e-10)
        weights_normalized = weights / weights.sum(axis=1, keepdims=True)
        depths_mm = np.sum(distances * weights_normalized, axis=1)
    else:
        raise ValueError(f"Unknown depth method: {method}. Use 'min', 'mean', or 'weighted'.")

    return depths_mm


def compute_depth_stratified_error(
    results: List[Dict],
    depth_bins: Optional[List[Tuple]] = None,
    depth_key: str = 'depth',
    error_key: str = 'localization_error'
) -> Dict[str, Dict[str, float]]:
    """
    Compute localization error stratified by source depth from electrodes.

    This function bins validation results by depth and computes error statistics
    for each bin. This reveals how localization accuracy varies with depth,
    which is critical for understanding the limitations of source localization.

    Parameters
    ----------
    results : list of dict
        List of validation results. Each dict should contain:
        - depth_key: Depth of the source (mm from electrodes)
        - error_key: Localization error (mm)
    depth_bins : list of tuple, optional
        List of (min_mm, max_mm, label) tuples defining depth bins.
        Default: [(1,2,'1-2mm'), (2,3,'2-3mm'), ..., (5,inf,'5+mm')]
    depth_key : str, default='depth'
        Key in results dict for depth values
    error_key : str, default='localization_error'
        Key in results dict for error values

    Returns
    -------
    error_by_depth : dict
        Dictionary mapping bin labels to error statistics:
        {
            '1-2mm': {'mean': 0.5, 'median': 0.3, 'std': 0.4, 'n': 50},
            '2-3mm': {'mean': 1.1, 'median': 0.8, 'std': 0.7, 'n': 80},
            ...
        }

    Examples
    --------
    >>> results = [
    ...     {'depth': 1.5, 'localization_error': 0.5},
    ...     {'depth': 2.5, 'localization_error': 1.2},
    ...     {'depth': 4.0, 'localization_error': 2.8},
    ... ]
    >>> error_by_depth = compute_depth_stratified_error(results)
    >>> print(error_by_depth['1-2mm'])
    {'mean': 0.5, 'median': 0.5, 'std': 0.0, 'n': 1}

    Notes
    -----
    Depth is measured as distance to nearest electrode. Typical bins for mouse EEG:
    - 1-2mm: Shallow sources (electrodes almost directly above, good localization)
    - 2-3mm: Mid-shallow sources (good localization expected)
    - 3-4mm: Mid-deep sources (moderate localization)
    - 4-5mm: Deep sources (degraded localization)
    - 5+mm: Deepest sources (worst localization expected)

    Expected behavior: Error should increase monotonically with depth due to
    the physics of EEG (deeper sources produce weaker, more diffuse signals).
    """
    if depth_bins is None:
        depth_bins = DEFAULT_DEPTH_BINS_MM

    error_by_depth = {}

    for bin_min, bin_max, bin_label in depth_bins:
        # Filter results to this depth bin
        bin_errors = [
            r[error_key] for r in results
            if depth_key in r and error_key in r
            and bin_min <= r[depth_key] < bin_max
        ]

        if bin_errors:
            error_by_depth[bin_label] = {
                'mean': float(np.mean(bin_errors)),
                'median': float(np.median(bin_errors)),
                'std': float(np.std(bin_errors)),
                'min': float(np.min(bin_errors)),
                'max': float(np.max(bin_errors)),
                'n': len(bin_errors)
            }
        else:
            error_by_depth[bin_label] = {
                'mean': np.nan,
                'median': np.nan,
                'std': np.nan,
                'min': np.nan,
                'max': np.nan,
                'n': 0
            }

    return error_by_depth
