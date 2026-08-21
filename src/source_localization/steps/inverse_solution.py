"""Step 5: Inverse Solution.

Apply inverse method (MNE, dSPM, sLORETA, LCMV, DICS) to estimate source activity.

Uses custom inverse implementations optimized for small animal brains,
bypassing MNE-Python's human-brain-specific assumptions.

Beamformer methods (LCMV, DICS) offer:
- Better focal source reconstruction
- Improved deep source localization
- Adaptive noise suppression

Output Types:
- Magnitude: Always positive (norm across 3 orientations). Use for power analysis.
- Signed: Preserves sign via SVD (dominant orientation). Use for connectivity analysis.

Memory Optimization:
- Epoch-wise processing to minimize peak memory usage
- Processes one epoch at a time instead of concatenating all epochs
- Peak memory reduced from ~8GB to ~200MB for typical datasets
"""

import mne
from mne.beamformer import make_lcmv, apply_lcmv, make_dics, apply_dics_csd
from mne.time_frequency import csd_morlet
import numpy as np


def combine_orientations(source_activity_3d):
    """
    Combine 3-orientation source activity into scalar time series.

    Returns both magnitude (always positive) and signed (SVD-based) versions.

    Parameters
    ----------
    source_activity_3d : ndarray
        Source activity with shape (n_sources, 3, n_times)

    Returns
    -------
    magnitude : ndarray
        Magnitude (norm) of source activity, shape (n_sources, n_times).
        Always positive. Use for power/spectral analysis.
    signed : ndarray
        Signed source activity via SVD projection, shape (n_sources, n_times).
        Preserves sign for connectivity analysis. Note: has inherent sign ambiguity.
    """
    n_sources, _, n_times = source_activity_3d.shape

    # Magnitude: L2 norm across orientations (always positive)
    magnitude = np.linalg.norm(source_activity_3d, axis=1)

    # Signed: SVD to find dominant orientation and project onto it
    signed = np.zeros((n_sources, n_times), dtype=np.float32)
    for i in range(n_sources):
        # source_activity_3d[i] has shape (3, n_times)
        # SVD finds the direction of maximum variance
        U, S, Vt = np.linalg.svd(source_activity_3d[i], full_matrices=False)
        # U[:, 0] is the dominant orientation (3,)
        # Project the 3D activity onto this direction to get signed scalar
        dominant_orientation = U[:, 0]
        signed[i, :] = dominant_orientation @ source_activity_3d[i]

    return magnitude, signed


def apply_orientation_constraint(fwd, orientation='free', loose=0.2, verbose=True):
    """Constrain source orientation to the cortical normal, or not.

    Until this existed, per-vertex normals were computed by the source spaces
    and never consumed: the forward stayed free-orientation and orientation was
    collapsed downstream by SVD or L2 norm, so anatomy never entered the
    inverse. That is the defect this addresses.

    Parameters
    ----------
    fwd : mne.Forward
        Forward solution.
    orientation : {'free', 'loose', 'fixed'}
        ``free`` leaves the leadfield untouched, three components per source,
        and is bit-identical to the behaviour that produced every published
        number. ``fixed`` keeps only the normal component. ``loose`` keeps all
        three but penalises the tangential ones.
    loose : float
        Tangential penalty for ``loose``, in (0, 1]. MNE's convention: the
        source covariance of the two tangential components is scaled by this,
        which is a weight of ``sqrt(loose)`` on the leadfield columns.

    Returns
    -------
    fwd_out : mne.Forward
        Orientation-converted forward.
    n_comp : int
        Components per source: 1 for ``fixed``, 3 otherwise.
    weights : ndarray or None
        Per-column leadfield weights for ``loose``, else None. These must be
        applied to the leadfield *columns* and then to the operator *rows*;
        see :func:`compute_inverse_operator`.
    """
    import mne

    orientation = (orientation or 'free').lower()
    if orientation not in ('free', 'loose', 'fixed'):
        raise ValueError(
            f"Unknown orientation: {orientation}. Valid: free, loose, fixed"
        )

    if orientation == 'free':
        if verbose:
            print("    Orientation: free (3 components per source, leadfield unchanged)")
        return fwd, 3, None

    # fixed and loose both need meaningful normals. Six of the ten presets use
    # source spaces that write nn = zeros, so fail loudly rather than silently
    # constraining every source to the zero vector.
    nn = np.vstack([s['nn'] for s in fwd['src']])
    norms = np.linalg.norm(nn, axis=1)
    if not np.all(norms > 1e-6):
        n_bad = int((norms <= 1e-6).sum())
        raise ValueError(
            f"orientation='{orientation}' needs source normals, but "
            f"{n_bad}/{len(norms)} sources have none. Source spaces that "
            "write nn = zeros (cartesian, roi_based) support only "
            "orientation='free'."
        )

    if orientation == 'fixed':
        fwd_out = mne.convert_forward_solution(
            fwd, surf_ori=True, force_fixed=True, use_cps=False, verbose=False
        )
        if verbose:
            print("    Orientation: fixed (1 component per source, along the "
                  "cortical normal)")
        return fwd_out, 1, None

    if not 0.0 < loose <= 1.0:
        raise ValueError(f"loose must be in (0, 1], got {loose}")

    # surf_ori rotates each triplet so component 0 is the normal and 1, 2 are
    # tangential; only then does penalising "the tangential components" mean
    # anything.
    fwd_out = mne.convert_forward_solution(
        fwd, surf_ori=True, force_fixed=False, use_cps=False, verbose=False
    )
    n_dipoles = fwd_out['sol']['data'].shape[1]
    weights = np.ones(n_dipoles, dtype=float)
    weights[1::3] = np.sqrt(loose)
    weights[2::3] = np.sqrt(loose)

    if verbose:
        print(f"    Orientation: loose {loose} (tangential components weighted "
              f"{np.sqrt(loose):.4f})")
    return fwd_out, 3, weights


def compute_inverse_operator(fwd, method='sLORETA', snr=3.0, lambda2=None, depth=0.8,
                             max_iter=20, tol=1e-6, verbose=True,
                             orientation='free', loose=0.2):
    """
    Compute inverse operator W for a given forward solution.

    This can be computed once and reused for all epochs.

    Parameters
    ----------
    fwd : mne.Forward
        Forward solution
    method : str
        Inverse method ('MNE', 'dSPM', 'sLORETA', 'eLORETA')
    snr : float
        Signal-to-noise ratio assumption
    lambda2 : float, optional
        Regularization parameter
    depth : float
        Depth weighting for MNE (ignored for dSPM/sLORETA/eLORETA)
    max_iter : int
        Maximum iterations for eLORETA (default: 20)
    tol : float
        Convergence tolerance for eLORETA (default: 1e-6)
    verbose : bool
        Print progress

    Returns
    -------
    W : ndarray
        Inverse operator, shape (n_dipoles, n_channels)
    normalizer : ndarray or None
        Normalization factors for dSPM/sLORETA/eLORETA, shape (n_dipoles,)
    n_comp : int
        Components per source: 1 under fixed orientation, else 3. Callers need
        this to reshape source activity.
    """
    if lambda2 is None:
        lambda2 = 1.0 / snr ** 2

    fwd, n_comp, ori_weights = apply_orientation_constraint(
        fwd, orientation=orientation, loose=loose, verbose=verbose
    )

    G_ori = fwd['sol']['data']
    n_channels, n_dipoles = G_ori.shape
    n_sources = n_dipoles // n_comp

    # Loose orientation is a source-covariance scaling. With R = diag(r),
    # W = R G' (G R G' + lam C)^-1 factors as: weight the leadfield columns by
    # sqrt(r), solve as usual, then weight the operator rows by sqrt(r). This
    # is the same shape as the depth weighting already applied below.
    if ori_weights is None:
        G = G_ori
    else:
        G = G_ori * ori_weights[np.newaxis, :]

    if verbose:
        print(f"    Computing inverse operator ({method})")
        print(f"    SNR: {snr}, Lambda2: {lambda2:.6f}")
        # D20: fixed orientation drops the leadfield from 3 columns per source
        # to 1, which changes trace(GG') and so the absolute size of the
        # regularisation term. Logged so the side effect is visible rather than
        # silently absorbed.
        trace_ggt = float(np.trace(G @ G.T))
        print(f"    Components/source: {n_comp}  (n_sources={n_sources:,}, "
              f"n_dipoles={n_dipoles:,})")
        print(f"    trace(GG')/n_ch: {trace_ggt / n_channels:.6e}  "
              f"effective reg: {lambda2 * trace_ggt / n_channels:.6e}")

    method_upper = method.upper()

    if method_upper == 'MNE':
        # MNE with depth weighting
        column_norms = np.linalg.norm(G, axis=0)
        if depth > 0:
            depth_weights = np.power(column_norms + 1e-10, -depth / 2)
            depth_weights /= np.mean(depth_weights)
            G_weighted = G * depth_weights[np.newaxis, :]
        else:
            G_weighted = G
            depth_weights = np.ones(n_dipoles)

        GGT = G_weighted @ G_weighted.T
        GGT_reg = GGT + lambda2 * np.trace(GGT) / n_channels * np.eye(n_channels)
        GGT_inv = np.linalg.inv(GGT_reg)
        W = G_weighted.T @ GGT_inv
        W = W * depth_weights[:, np.newaxis]
        normalizer = None

    elif method_upper == 'DSPM':
        GGT = G @ G.T
        GGT_reg = GGT + lambda2 * np.trace(GGT) / n_channels * np.eye(n_channels)
        GGT_inv = np.linalg.inv(GGT_reg)
        W = G.T @ GGT_inv
        # dSPM normalizer
        normalizer = np.sqrt(np.sum(W ** 2, axis=1))

    elif method_upper == 'SLORETA':
        GGT = G @ G.T
        GGT_reg = GGT + lambda2 * np.trace(GGT) / n_channels * np.eye(n_channels)
        GGT_inv = np.linalg.inv(GGT_reg)
        W = G.T @ GGT_inv
        # sLORETA normalizer
        normalizer = np.sqrt(np.sum(W * G.T, axis=1))

    elif method_upper == 'ELORETA':
        # eLORETA: iterative weight optimization for exact localization
        if verbose:
            print(f"    eLORETA max iterations: {max_iter}, tolerance: {tol}")

        # Initialize source weights (one per source location, whatever its
        # component count)
        D = np.ones(n_sources)

        # Iterative eLORETA weight estimation
        for iteration in range(max_iter):
            D_old = D.copy()

            # Apply current weights to leadfield
            G_weighted = G.copy()
            for i in range(n_sources):
                G_weighted[:, i*n_comp:(i+1)*n_comp] *= D[i]

            # Compute inverse operator with weighted leadfield
            GGT = G_weighted @ G_weighted.T
            GGT_reg = GGT + lambda2 * np.trace(GGT) / n_channels * np.eye(n_channels)
            GGT_inv = np.linalg.inv(GGT_reg)
            W_temp = G_weighted.T @ GGT_inv

            # Update weights based on resolution matrix
            for i in range(n_sources):
                W_i = W_temp[i*n_comp:(i+1)*n_comp, :]
                G_i = G[:, i*n_comp:(i+1)*n_comp]
                R_i = W_i @ G_i
                trace_R = np.trace(R_i)
                D[i] = 1.0 / (np.sqrt(trace_R / n_comp) + 1e-10)

            # Check convergence
            change = np.max(np.abs(D - D_old) / (np.abs(D_old) + 1e-10))
            if verbose and (iteration < 2 or change < tol):
                print(f"      Iteration {iteration + 1}: max weight change = {change:.2e}")

            if change < tol:
                if verbose:
                    print(f"      Converged at iteration {iteration + 1}")
                break

        # Compute final inverse operator with converged weights
        G_weighted = G.copy()
        for i in range(n_sources):
            G_weighted[:, i*n_comp:(i+1)*n_comp] *= D[i]

        GGT = G_weighted @ G_weighted.T
        GGT_reg = GGT + lambda2 * np.trace(GGT) / n_channels * np.eye(n_channels)
        GGT_inv = np.linalg.inv(GGT_reg)
        W = G_weighted.T @ GGT_inv

        # Apply source weights to inverse operator
        for i in range(n_sources):
            W[i*n_comp:(i+1)*n_comp, :] *= D[i]

        # Compute eLORETA normalizer (resolution-based, per dipole)
        normalizer = np.zeros(n_dipoles)
        for i in range(n_sources):
            W_i = W[i*n_comp:(i+1)*n_comp, :]
            G_i = G[:, i*n_comp:(i+1)*n_comp]
            R_i = W_i @ G_i
            norm_val = np.sqrt(np.trace(R_i) / n_comp) + 1e-10
            normalizer[i*n_comp:(i+1)*n_comp] = norm_val

    else:
        raise ValueError(f"Unknown method: {method}. Supported: MNE, dSPM, sLORETA, eLORETA")

    # Undo the leadfield column weighting on the operator rows (see above).
    # No-op under free orientation, which keeps that path bit-identical.
    if ori_weights is not None:
        W = W * ori_weights[:, np.newaxis]

        # dSPM and sLORETA normalise each component by its own resolution,
        # which divides the penalised tangential components by a
        # correspondingly smaller number and re-inflates exactly what the
        # loose constraint suppressed. Measured: the constraint takes the
        # tangential/normal amplitude ratio from 5.21 to 0.88, and
        # per-component normalisation puts it back to 1.73, against free's
        # 1.90 — so loose was doing almost nothing.
        #
        # Normalise per source instead, applying one value to all three
        # components, which is what MNE does for non-fixed orientations and
        # what the eLORETA branch above already does. Only reached when
        # ori_weights is set, so the free path stays bit-identical.
        if normalizer is not None:
            per_source = normalizer.reshape(n_sources, n_comp)
            combined = np.sqrt((per_source ** 2).sum(axis=1) / n_comp)
            normalizer = np.repeat(combined, n_comp)

    return W, normalizer, n_comp


def apply_inverse_to_epoch(W, normalizer, epoch_data, n_sources, n_comp=3):
    """
    Apply precomputed inverse operator to a single epoch.

    Memory efficient: processes one epoch at a time.

    Parameters
    ----------
    W : ndarray
        Inverse operator, shape (n_dipoles, n_channels)
    normalizer : ndarray or None
        Normalization factors
    epoch_data : ndarray
        EEG data for one epoch, shape (n_channels, n_times)
    n_sources : int
        Number of sources (n_dipoles / n_comp)
    n_comp : int
        Components per source: 1 under fixed orientation, else 3.

    Returns
    -------
    magnitude : ndarray
        Magnitude source activity, shape (n_sources, n_times)
    signed : ndarray
        Signed source activity, shape (n_sources, n_times)
    """
    # Apply inverse operator
    source_activity = W @ epoch_data

    # Apply normalization if provided
    if normalizer is not None:
        source_activity = source_activity / (normalizer[:, np.newaxis] + 1e-10)

    # Combine orientations
    source_activity_nc = source_activity.reshape(n_sources, n_comp, -1)

    # Magnitude: L2 norm (fast, vectorized)
    magnitude = np.linalg.norm(source_activity_nc, axis=1).astype(np.float32)

    if n_comp == 1:
        # Under fixed orientation the single component IS the projection onto
        # the cortical normal, so its sign is anatomically meaningful rather
        # than an arbitrary SVD convention. This is what makes signed measures
        # such as ERP polarity interpretable.
        signed = source_activity_nc[:, 0, :].astype(np.float32)
        return magnitude, signed

    # Signed: use first component weighted by sign of max variance direction
    # This is faster than full SVD and gives similar results for single epochs
    signed = np.zeros((n_sources, epoch_data.shape[1]), dtype=np.float32)
    for i in range(n_sources):
        # Find dominant orientation from variance
        variances = np.var(source_activity_nc[i], axis=1)
        dominant_idx = np.argmax(variances)
        signed[i, :] = source_activity_nc[i, dominant_idx, :]

    return magnitude, signed


def apply_inverse_custom_dSPM(fwd, eeg_data, snr=3.0, lambda2=None, verbose=True):
    """
    Custom dSPM implementation for mouse brains.

    Avoids MNE-Python's human-brain assumptions about depth weighting
    and noise covariance.

    Parameters
    ----------
    fwd : mne.Forward
        Forward solution
    eeg_data : ndarray
        EEG data with shape (n_channels, n_times)
    snr : float
        Signal-to-noise ratio assumption
    lambda2 : float, optional
        Regularization parameter. If None, computed from SNR.

    Returns
    -------
    magnitude : ndarray
        Magnitude (norm) of source activity, shape (n_sources, n_times)
    signed : ndarray
        Signed source activity via SVD, shape (n_sources, n_times)
    """
    if lambda2 is None:
        lambda2 = 1.0 / snr ** 2

    # Get leadfield
    G = fwd['sol']['data']
    n_channels, n_dipoles = G.shape

    if verbose:
        print(f"    Using custom dSPM implementation (mouse brain optimized)")
        print(f"    SNR: {snr}")
        print(f"    Lambda2: {lambda2:.6f}")
        print(f"    Leadfield: {G.shape}")

    # Compute inverse operator
    GGT = G @ G.T
    GGT_reg = GGT + lambda2 * np.trace(GGT) / n_channels * np.eye(n_channels)
    GGT_inv = np.linalg.inv(GGT_reg)
    W = G.T @ GGT_inv

    # Apply to data
    source_activity = W @ eeg_data

    # dSPM normalization: divide by noise sensitivity (Dale et al., 2000)
    # For white noise (identity covariance): noise_norm = sqrt(diag(W @ W.T))
    noise_norm = np.sqrt(np.sum(W ** 2, axis=1))  # sqrt(diag(W @ W.T))
    source_activity_norm = source_activity / (noise_norm[:, np.newaxis] + 1e-10)

    # Combine orientations - returns both magnitude and signed
    n_sources = n_dipoles // 3
    source_activity_reshaped = source_activity_norm.reshape(n_sources, 3, -1)
    magnitude, signed = combine_orientations(source_activity_reshaped)

    return magnitude, signed


def apply_inverse_custom_MNE(fwd, eeg_data, snr=3.0, lambda2=None, depth=0.8, verbose=True):
    """
    Custom MNE implementation for mouse brains with depth weighting.

    Parameters
    ----------
    fwd : mne.Forward
        Forward solution
    eeg_data : ndarray
        EEG data with shape (n_channels, n_times)
    snr : float
        Signal-to-noise ratio (default: 3.0)
    lambda2 : float, optional
        Regularization parameter (default: 1/snr^2)
    depth : float
        Depth weighting exponent (default: 0.8). Higher values give more
        compensation for deep sources. Set to 0 for no depth weighting.
    verbose : bool
        Print progress messages

    Returns
    -------
    magnitude : ndarray
        Magnitude (norm) of source activity, shape (n_sources, n_times)
    signed : ndarray
        Signed source activity via SVD, shape (n_sources, n_times)
    """
    if lambda2 is None:
        lambda2 = 1.0 / snr ** 2

    G = fwd['sol']['data']
    n_channels, n_dipoles = G.shape

    if verbose:
        print(f"    Using custom MNE implementation (mouse brain optimized)")
        print(f"    SNR: {snr}")
        print(f"    Lambda2: {lambda2:.6f}")
        print(f"    Depth weighting: {depth}")

    # Compute depth weights from leadfield column norms
    # Deeper sources have smaller leadfield norms, so we weight them up
    column_norms = np.linalg.norm(G, axis=0)  # Shape: (n_dipoles,)

    # Depth weighting: w_i = ||g_i||^(-depth/2)
    # The /2 is because we apply weights to both sides of the source covariance
    # This compensates for the 1/r^2 decay of the leadfield
    if depth > 0:
        # Avoid division by zero for very small norms
        depth_weights = np.power(column_norms + 1e-10, -depth / 2)
        # Normalize to preserve overall scale
        depth_weights /= np.mean(depth_weights)
        # Create weighted leadfield: G_weighted = G @ diag(depth_weights)
        G_weighted = G * depth_weights[np.newaxis, :]
    else:
        G_weighted = G
        depth_weights = np.ones(n_dipoles)

    # Compute inverse with weighted leadfield
    # This is equivalent to: W = R @ G.T @ (G @ R @ G.T + λI)^-1
    # where R = diag(depth_weights^2) is the source covariance prior
    GGT = G_weighted @ G_weighted.T
    GGT_reg = GGT + lambda2 * np.trace(GGT) / n_channels * np.eye(n_channels)
    GGT_inv = np.linalg.inv(GGT_reg)
    W = G_weighted.T @ GGT_inv  # Shape: (n_dipoles, n_channels)

    # Apply depth weights to transform back to source space
    # W_final = diag(depth_weights) @ W
    W = W * depth_weights[:, np.newaxis]

    source_activity = W @ eeg_data

    # Combine orientations - returns both magnitude and signed
    n_sources = n_dipoles // 3
    source_activity_reshaped = source_activity.reshape(n_sources, 3, -1)
    magnitude, signed = combine_orientations(source_activity_reshaped)

    return magnitude, signed


def apply_inverse_custom_sLORETA(fwd, eeg_data, snr=3.0, lambda2=None, verbose=True):
    """
    Custom sLORETA implementation for mouse brains.

    Parameters
    ----------
    fwd : mne.Forward
        Forward solution
    eeg_data : ndarray
        EEG data with shape (n_channels, n_times)
    snr : float
        Signal-to-noise ratio assumption
    lambda2 : float, optional
        Regularization parameter. If None, computed from SNR.

    Returns
    -------
    magnitude : ndarray
        Magnitude (norm) of source activity, shape (n_sources, n_times)
    signed : ndarray
        Signed source activity via SVD, shape (n_sources, n_times)
    """
    if lambda2 is None:
        lambda2 = 1.0 / snr ** 2

    G = fwd['sol']['data']
    n_channels, n_dipoles = G.shape

    if verbose:
        print(f"    Using custom sLORETA implementation (mouse brain optimized)")
        print(f"    SNR: {snr}")
        print(f"    Lambda2: {lambda2:.6f}")

    GGT = G @ G.T
    GGT_reg = GGT + lambda2 * np.trace(GGT) / n_channels * np.eye(n_channels)
    GGT_inv = np.linalg.inv(GGT_reg)
    W = G.T @ GGT_inv

    source_activity = W @ eeg_data

    # sLORETA normalization
    resolution_diagonal = np.sum(W * G.T, axis=1)
    source_activity_norm = source_activity / (np.sqrt(resolution_diagonal[:, np.newaxis]) + 1e-10)

    # Combine orientations - returns both magnitude and signed
    n_sources = n_dipoles // 3
    source_activity_reshaped = source_activity_norm.reshape(n_sources, 3, -1)
    magnitude, signed = combine_orientations(source_activity_reshaped)

    return magnitude, signed


def apply_inverse_custom_eLORETA(fwd, eeg_data, snr=3.0, lambda2=None, max_iter=20, tol=1e-6, verbose=True):
    """
    Custom eLORETA (exact LORETA) implementation for mouse brains.

    eLORETA (Pascual-Marqui, 2007) achieves exact, zero-error localization
    for single point sources through iterative weight matrix optimization.

    Parameters
    ----------
    fwd : mne.Forward
        Forward solution
    eeg_data : ndarray
        EEG data with shape (n_channels, n_times)
    snr : float
        Signal-to-noise ratio assumption
    lambda2 : float, optional
        Regularization parameter. If None, computed from SNR.
    max_iter : int
        Maximum iterations for weight optimization (default: 20)
    tol : float
        Convergence tolerance (default: 1e-6)
    verbose : bool
        Print progress messages

    Returns
    -------
    magnitude : ndarray
        Magnitude (norm) of source activity, shape (n_sources, n_times)
    signed : ndarray
        Signed source activity via SVD, shape (n_sources, n_times)

    References
    ----------
    Pascual-Marqui, R.D. (2007). Discrete, 3D distributed, linear imaging
    methods of electric neuronal activity. Part 1: exact, zero error
    localization. arXiv:0710.3341
    """
    if lambda2 is None:
        lambda2 = 1.0 / snr ** 2

    G = fwd['sol']['data']
    n_channels, n_dipoles = G.shape
    n_sources = n_dipoles // 3

    if verbose:
        print(f"    Using custom eLORETA implementation (mouse brain optimized)")
        print(f"    SNR: {snr}")
        print(f"    Lambda2: {lambda2:.6f}")
        print(f"    Max iterations: {max_iter}")

    # eLORETA computes weights per source location (not per dipole component)
    # Initialize weight matrix D (diagonal, one weight per source location)
    # D has shape (n_sources,) - one weight per 3-component source

    # Reshape G to group by source: (n_channels, n_sources, 3)
    G_3d = G.reshape(n_channels, n_sources, 3)

    # Initialize source weights (one per location)
    D = np.ones(n_sources)

    # Iterative eLORETA weight estimation
    for iteration in range(max_iter):
        D_old = D.copy()

        # Apply current weights to leadfield
        # G_weighted[:, i*3:(i+1)*3] = G[:, i*3:(i+1)*3] * D[i]
        G_weighted = G.copy()
        for i in range(n_sources):
            G_weighted[:, i*3:(i+1)*3] *= D[i]

        # Compute inverse operator with weighted leadfield
        GGT = G_weighted @ G_weighted.T
        GGT_reg = GGT + lambda2 * np.trace(GGT) / n_channels * np.eye(n_channels)
        GGT_inv = np.linalg.inv(GGT_reg)

        # Compute resolution matrix diagonal for each source location
        # For eLORETA, we compute the average resolution across 3 orientations
        W = G_weighted.T @ GGT_inv

        # Update weights: D_i = 1 / sqrt(trace(R_i) / 3)
        # where R_i is the 3x3 resolution submatrix for source i
        for i in range(n_sources):
            # Get 3x3 resolution submatrix: W[i*3:(i+1)*3, :] @ G[:, i*3:(i+1)*3]
            W_i = W[i*3:(i+1)*3, :]  # (3, n_channels)
            G_i = G[:, i*3:(i+1)*3]   # (n_channels, 3)
            R_i = W_i @ G_i           # (3, 3) resolution submatrix

            # eLORETA weight: average of diagonal elements
            trace_R = np.trace(R_i)
            D[i] = 1.0 / (np.sqrt(trace_R / 3.0) + 1e-10)

        # Check convergence
        change = np.max(np.abs(D - D_old) / (np.abs(D_old) + 1e-10))
        if verbose and (iteration < 3 or iteration == max_iter - 1):
            print(f"      Iteration {iteration + 1}: max weight change = {change:.2e}")

        if change < tol:
            if verbose:
                print(f"      Converged at iteration {iteration + 1}")
            break

    # Apply final weights to compute source activity
    G_weighted = G.copy()
    for i in range(n_sources):
        G_weighted[:, i*3:(i+1)*3] *= D[i]

    GGT = G_weighted @ G_weighted.T
    GGT_reg = GGT + lambda2 * np.trace(GGT) / n_channels * np.eye(n_channels)
    GGT_inv = np.linalg.inv(GGT_reg)
    W = G_weighted.T @ GGT_inv

    # Apply weights to inverse operator
    for i in range(n_sources):
        W[i*3:(i+1)*3, :] *= D[i]

    # Apply to data
    source_activity = W @ eeg_data

    # eLORETA normalization (final pass)
    # Normalize each source by its resolution matrix diagonal
    for i in range(n_sources):
        W_i = W[i*3:(i+1)*3, :]
        G_i = G[:, i*3:(i+1)*3]
        R_i = W_i @ G_i
        norm_factor = np.sqrt(np.trace(R_i) / 3.0) + 1e-10
        source_activity[i*3:(i+1)*3, :] /= norm_factor

    # Combine orientations - returns both magnitude and signed
    source_activity_reshaped = source_activity.reshape(n_sources, 3, -1)
    magnitude, signed = combine_orientations(source_activity_reshaped)

    return magnitude, signed


def apply_inverse_LCMV(fwd, epochs, info, reg=0.05, weight_norm='unit-noise-gain'):
    """
    Apply LCMV (Linearly Constrained Minimum Variance) beamformer.

    LCMV is a spatial filter that estimates activity at each source location
    independently while suppressing contributions from other sources.

    Advantages over minimum norm methods:
    - Better focal source reconstruction
    - Improved deep source localization
    - Adaptive noise suppression based on data covariance

    Parameters
    ----------
    fwd : mne.Forward
        Forward solution
    epochs : mne.Epochs
        Epoched EEG data (used for covariance estimation)
    info : mne.Info
        Measurement info
    reg : float
        Regularization parameter (default: 0.05)
    weight_norm : str
        Weight normalization method. Options:
        - 'unit-noise-gain': Normalizes by noise sensitivity (recommended)
        - 'unit-noise-gain-invariant': Orientation-invariant normalization
        - None: No normalization (has depth bias)

    Returns
    -------
    magnitude : ndarray
        Magnitude (abs) of source activity, shape (n_sources, n_times)
    signed : ndarray
        Signed source activity (optimal orientation), shape (n_sources, n_times)
    stc : mne.SourceEstimate
        MNE source estimate object
    """
    print(f"    Using LCMV beamformer")
    print(f"    Regularization: {reg}")
    print(f"    Weight normalization: {weight_norm}")

    # Compute data covariance from epochs
    # Use entire epoch for covariance (signal + noise)
    print(f"    Computing data covariance from {len(epochs)} epochs...")
    data_cov = mne.compute_covariance(epochs, method='empirical')

    # Create LCMV spatial filter
    print(f"    Creating LCMV spatial filter...")
    filters = make_lcmv(
        info=info,
        forward=fwd,
        data_cov=data_cov,
        reg=reg,
        pick_ori='max-power',  # Scalar output (optimal orientation)
        weight_norm=weight_norm,
        depth=None  # Let weight_norm handle depth bias
    )

    # Apply beamformer to averaged data
    evoked = epochs.average()
    stc = apply_lcmv(evoked, filters)

    # Extract both magnitude and signed source activity
    # Beamformer with pick_ori='max-power' already gives scalar per source
    magnitude = np.abs(stc.data)
    signed = stc.data.copy()  # Preserve sign from optimal orientation

    print(f"    ✓ LCMV complete: {magnitude.shape}")

    return magnitude, signed, stc


def apply_inverse_DICS(fwd, epochs, info, freq_band=(30, 55), reg=0.05, weight_norm='unit-noise-gain'):
    """
    Apply DICS (Dynamic Imaging of Coherent Sources) beamformer.

    DICS is a frequency-domain beamformer optimized for oscillatory activity.
    Particularly useful for analyzing specific frequency bands (e.g., gamma).

    Advantages:
    - Designed for oscillatory signals
    - Better frequency-specific source localization
    - Reduces broadband noise interference

    Parameters
    ----------
    fwd : mne.Forward
        Forward solution
    epochs : mne.Epochs
        Epoched EEG data
    info : mne.Info
        Measurement info
    freq_band : tuple
        Frequency band of interest (fmin, fmax) in Hz (default: low_gamma 30-55 Hz)
    reg : float
        Regularization parameter (default: 0.05)
    weight_norm : str
        Weight normalization method (default: 'unit-noise-gain')

    Returns
    -------
    magnitude : ndarray
        Magnitude (abs) of source activity, shape (n_sources, n_times)
    signed : ndarray
        Signed source activity, shape (n_sources, n_times)
        Note: DICS is inherently power-based, so signed=magnitude for this method.
    stc : mne.SourceEstimate
        MNE source estimate object
    """
    fmin, fmax = freq_band
    print(f"    Using DICS beamformer")
    print(f"    Frequency band: {fmin}-{fmax} Hz")
    print(f"    Regularization: {reg}")
    print(f"    Weight normalization: {weight_norm}")

    # Compute cross-spectral density (CSD) matrix
    # Use Morlet wavelets for time-frequency decomposition
    freqs = np.linspace(fmin, fmax, num=5)  # Sample frequencies in band
    n_cycles = freqs / 2.0  # Adaptive cycles (higher freq = more cycles)

    print(f"    Computing CSD at frequencies: {freqs} Hz...")
    csd = csd_morlet(
        epochs,
        frequencies=freqs,
        n_cycles=n_cycles,
        decim=1
    )

    # Create DICS spatial filter
    print(f"    Creating DICS spatial filter...")
    filters = make_dics(
        info=info,
        forward=fwd,
        csd=csd.mean(),  # Average CSD across frequencies
        reg=reg,
        pick_ori='max-power',
        weight_norm=weight_norm,
        depth=None
    )

    # Apply DICS to get source power
    # Use CSD averaged across frequencies
    stc, freqs_out = apply_dics_csd(csd.mean(), filters)

    # Extract source power
    # Note: DICS outputs power (real, positive), not amplitude
    magnitude = np.abs(stc.data)

    # Replicate across time dimension for compatibility with downstream code
    # DICS gives a single power estimate per source
    n_times = len(epochs.times)
    if magnitude.ndim == 1:
        magnitude = np.tile(magnitude[:, np.newaxis], (1, n_times))

    # For DICS, signed is same as magnitude (power is inherently positive)
    # This is expected behavior for frequency-domain beamformers
    signed = magnitude.copy()

    print(f"    ✓ DICS complete: {magnitude.shape}")
    print(f"    Note: DICS outputs power (always positive), signed=magnitude")

    return magnitude, signed, stc


def run(config, previous_outputs):
    """
    Compute inverse solution.

    Computes both magnitude (always positive, for power analysis) and signed
    (SVD-based, for connectivity analysis) source time courses.

    Uses epoch-wise processing for memory efficiency: processes one epoch at a
    time instead of concatenating all epochs, reducing peak memory from ~8GB
    to ~200MB for typical datasets.

    Parameters
    ----------
    config : Config
        Pipeline configuration
    previous_outputs : dict
        Outputs from previous steps containing:
        - 'fwd': mne.Forward - Forward solution
        - 'epochs': mne.Epochs - EEG epochs
        - 'info': mne.Info - EEG measurement info

    Returns
    -------
    outputs : dict
        Dictionary containing:
        - 'stc': mne.VolSourceEstimate - Magnitude source time courses (backward compat)
        - 'stc_magnitude': mne.VolSourceEstimate - Magnitude (always positive)
        - 'stc_signed': mne.VolSourceEstimate - Signed (SVD-based, for connectivity)
        - 'method': str - Inverse method used
    """
    method = config['inverse']['method']
    snr = config['inverse']['snr']

    # Compute lambda2 from SNR if not specified
    if config['inverse']['lambda2'] is not None:
        lambda2 = config['inverse']['lambda2']
    else:
        lambda2 = 1.0 / (snr ** 2)

    print(f"  Computing inverse solution:")
    print(f"    Method: {method}")
    print(f"    SNR: {snr}")
    print(f"    λ²: {lambda2:.6f}")

    # Extract required inputs
    fwd = previous_outputs['fwd']
    epochs = previous_outputs['epochs']
    sfreq = epochs.info['sfreq']

    # Get depth weighting from config (default 0.0 for mouse brain)
    depth_weighting = config['inverse'].get('depth_weighting', 0.0)

    # Get beamformer-specific parameters if applicable
    beamformer_reg = config['inverse'].get('beamformer_reg', 0.05)
    weight_norm = config['inverse'].get('weight_norm', 'unit-noise-gain')
    freq_band = config['inverse'].get('freq_band', None)

    # Orientation constraint. Paired with the anatomical surface default in
    # source_space/surface.py — see the note there for why the two have to move
    # together.
    #
    # The default is resolved against the source space rather than fixed as a
    # constant, because `fixed` is only meaningful where normals exist. Six of
    # the ten presets use source spaces that write nn = zeros (cartesian,
    # roi_based, shell), and `apply_orientation_constraint` refuses those
    # loudly and correctly. Refusing a value the *user* asked for is right;
    # refusing a value they never mentioned is not, so an unset default resolves
    # to `fixed` where the anatomy supports it and `free` where it does not.
    # An explicit `orientation:` in config is passed through untouched and still
    # raises if the source space cannot honour it.
    orientation = config['inverse'].get('orientation')
    if orientation is None:
        # Keyed off the source space, not off whether `nn` happens to be
        # non-zero. The icosphere also carries normals — radial ones — and they
        # differ from true cortical normals by 49.36 degrees on average, so
        # constraining to them asserts close to noise. Measured, that is the one
        # cell where fixed loses: icosphere fixed 2.370 mm against free's
        # 2.347 mm, where anatomical fixed is 1.026 mm against free's 2.473 mm.
        _surf = (config.get('source_space') or {}).get('surface') or {}
        _anatomical = (
            config.get('pipeline', {}).get('source_type') == 'surface'
            and _surf.get('method', 'anatomical') == 'anatomical'
        )
        orientation = 'fixed' if _anatomical else 'free'
        print(f"    Orientation: {orientation} (default for "
              f"{'the anatomical surface' if _anatomical else 'this source space'})")
    loose = float(config['inverse'].get('loose', 0.2))

    # Determine number of sources. Fixed orientation collapses each source to a
    # single component, so the divisor is not always 3.
    n_dipoles = fwd['sol']['data'].shape[1]
    n_sources = n_dipoles // 3

    # Check if we should use beamformer methods (which have different processing)
    if method.upper() in ['LCMV', 'DICS']:
        # Beamformers use their own epoch handling
        if method.upper() == 'LCMV':
            source_magnitude, source_signed, _ = apply_inverse_LCMV(
                fwd, epochs, previous_outputs['info'],
                reg=beamformer_reg,
                weight_norm=weight_norm
            )
        else:  # DICS
            if freq_band is None:
                primary_band = config['spectral'].get('primary_band', 'low_gamma')
                freq_bands = config['spectral'].get('frequency_bands', {})
                freq_band = freq_bands.get(primary_band, (30, 55))
            source_magnitude, source_signed, _ = apply_inverse_DICS(
                fwd, epochs, previous_outputs['info'],
                freq_band=freq_band,
                reg=beamformer_reg,
                weight_norm=weight_norm
            )
    else:
        # Use memory-efficient epoch-wise processing for MNE/dSPM/sLORETA
        print(f"    Using memory-efficient epoch-wise processing")

        # Compute inverse operator once (this is small and reusable)
        W, normalizer, n_comp = compute_inverse_operator(
            fwd, method=method, snr=snr, lambda2=lambda2,
            depth=depth_weighting, verbose=True,
            orientation=orientation, loose=loose
        )
        # Derive the source count from the OPERATOR, not from the original
        # leadfield. n_dipoles here is the free-orientation column count; under
        # fixed orientation the forward is converted inside
        # compute_inverse_operator, so n_dipoles // n_comp overcounts threefold
        # (10,440 instead of 3,480) and the epoch reshape then fails.
        n_sources = W.shape[0] // n_comp

        # Get epochs data
        epochs_data = epochs.get_data()  # (n_epochs, n_channels, n_times)
        n_epochs, n_channels, n_times_per_epoch = epochs_data.shape
        n_total_times = n_epochs * n_times_per_epoch

        print(f"    Processing {n_epochs} epochs × {n_times_per_epoch} samples = {n_total_times:,} total")
        print(f"    Sources: {n_sources:,} (peak memory: ~{n_sources * n_times_per_epoch * 4 * 2 / 1e6:.0f} MB per epoch)")

        # Pre-allocate output arrays (float32 to save memory)
        source_magnitude = np.zeros((n_sources, n_total_times), dtype=np.float32)
        source_signed = np.zeros((n_sources, n_total_times), dtype=np.float32)

        # Process epochs one at a time
        for epoch_idx in range(n_epochs):
            if (epoch_idx + 1) % 50 == 0 or epoch_idx == 0:
                print(f"    Processing epoch {epoch_idx + 1}/{n_epochs}...")

            epoch_data = epochs_data[epoch_idx]  # (n_channels, n_times)

            # Apply inverse to this epoch
            mag_epoch, signed_epoch = apply_inverse_to_epoch(
                W, normalizer, epoch_data, n_sources, n_comp=n_comp
            )

            # Store in output arrays
            start_idx = epoch_idx * n_times_per_epoch
            end_idx = start_idx + n_times_per_epoch
            source_magnitude[:, start_idx:end_idx] = mag_epoch
            source_signed[:, start_idx:end_idx] = signed_epoch

            # Free epoch memory
            del mag_epoch, signed_epoch

        # Free epochs array
        del epochs_data

        print(f"    ✓ Epoch-wise processing complete")

    # Set tmin for SourceEstimate (concatenated epochs start at t=0)
    tmin = 0.0

    # Create MNE SourceEstimate objects for both magnitude and signed
    vertices = [np.arange(len(source_magnitude))]

    stc_magnitude = mne.VolSourceEstimate(
        data=source_magnitude,
        vertices=vertices,
        tmin=tmin,
        tstep=1.0 / sfreq,
        subject='mouse'
    )

    stc_signed = mne.VolSourceEstimate(
        data=source_signed,
        vertices=vertices,
        tmin=tmin,
        tstep=1.0 / sfreq,
        subject='mouse'
    )

    # Get statistics
    n_sources = len(stc_magnitude.data)
    n_times = stc_magnitude.data.shape[1]

    print(f"    ✓ Source estimate: {n_sources:,} sources × {n_times} timepoints")
    print(f"    Magnitude - min: {source_magnitude.min():.2e}, max: {source_magnitude.max():.2e}")
    print(f"    Signed    - min: {source_signed.min():.2e}, max: {source_signed.max():.2e}")
    print(f"    Signed has negative values: {np.any(source_signed < 0)}")

    # Save intermediate data and create visualizations
    if config['outputs'].get('save_intermediate', True):
        from ..utils.io_utils import save_pickle, get_data_dir, get_figures_dir, get_output_variants
        from ..utils.step_visualizations import visualize_step5_inverse

        data_dir = get_data_dir(config)
        figures_dir = get_figures_dir(config)

        variants = get_output_variants(config)

        if 'magnitude' in variants:
            save_pickle(stc_magnitude, data_dir / 'step5_stc_magnitude.pkl')
            print(f"    Saved: {data_dir / 'step5_stc_magnitude.pkl'}")
        if 'signed' in variants:
            save_pickle(stc_signed, data_dir / 'step5_stc_signed.pkl')
            print(f"    Saved: {data_dir / 'step5_stc_signed.pkl'}")

        # Export source-level .set files for parametric mapping
        # Each source becomes a "channel" with its 3D coordinates
        source_coords_mm = previous_outputs['source_coords_mm']
        # Handle source count mismatch
        if len(stc_magnitude.data) != len(source_coords_mm):
            source_coords_mm_export = source_coords_mm[:len(stc_magnitude.data)]
        else:
            source_coords_mm_export = source_coords_mm

        from ..utils.export_set import export_source_to_set
        sfreq = stc_magnitude.sfreq if hasattr(stc_magnitude, 'sfreq') else previous_outputs.get('sfreq', 500.0)

        if 'magnitude' in variants:
            export_source_to_set(
                stc_magnitude.data, source_coords_mm_export, sfreq,
                data_dir / 'source_timeseries_magnitude.set',
                subject_id='source_magnitude'
            )
        if 'signed' in variants:
            export_source_to_set(
                stc_signed.data, source_coords_mm_export, sfreq,
                data_dir / 'source_timeseries_signed.set',
                subject_id='source_signed'
            )

        # Inverse solution QC visualizations
        import matplotlib.pyplot as plt

        if 'magnitude' in variants:
            fig = visualize_step5_inverse(stc_magnitude, source_coords_mm_export, method,
                                          figures_dir / 'step5_inverse_magnitude.png',
                                          title_suffix=' (Magnitude)')
            print(f"    Saved: {figures_dir / 'step5_inverse_magnitude.png'}")
            plt.close(fig)

        if 'signed' in variants:
            fig = visualize_step5_inverse(stc_signed, source_coords_mm_export, method,
                                          figures_dir / 'step5_inverse_signed.png',
                                          title_suffix=' (Signed)')
            print(f"    Saved: {figures_dir / 'step5_inverse_signed.png'}")
            plt.close(fig)

    return {
        'stc': stc_magnitude,  # Backward compatibility
        'stc_magnitude': stc_magnitude,
        'stc_signed': stc_signed,
        'method': method
    }
