"""Step 4: Forward Solution Computation.

Compute forward solution (leadfield matrix) mapping sources to sensors.
"""

import mne
import numpy as np


def run(config, previous_outputs):
    """
    Compute forward solution.

    Parameters
    ----------
    config : Config
        Pipeline configuration
    previous_outputs : dict
        Outputs from previous steps containing:
        - 'info': mne.Info - EEG measurement info
        - 'src': mne.SourceSpaces - Source space
        - 'bem': BEM model (sphere or ellipsoid)
        - 'bem_params': dict - BEM parameters

    Returns
    -------
    outputs : dict
        Dictionary containing:
        - 'fwd': mne.Forward - Forward solution
    """
    bem_type = config['pipeline']['bem_type']
    source_type = config['pipeline']['source_type']

    print(f"  Computing forward solution:")
    print(f"    BEM type: {bem_type}")
    print(f"    Source type: {source_type}")

    # Extract required inputs
    info = previous_outputs['info']
    src = previous_outputs['src']
    bem = previous_outputs['bem']

    # Handle sphere vs. ellipsoid BEM differently
    # Sphere BEM can be analytical (is_sphere=True) or numerical (has surfaces)
    if bem_type == 'sphere':
        # Check if this is a numerical sphere BEM (has surfaces, is_sphere=False)
        is_numerical_sphere = isinstance(bem, dict) and 'surfs' in bem and not bem.get('is_sphere', True)

        if is_numerical_sphere:
            # Numerical sphere BEM - use same approach as ellipsoid
            print(f"    Using NUMERICAL sphere BEM solution (boundary integral)")

            # Create BEM solution from sphere surfaces
            bem_solution = mne.make_bem_solution(bem['surfs'], verbose=False)

            fwd = mne.make_forward_solution(
                info,
                trans=None,  # No coordinate transform needed
                src=src,
                bem=bem_solution,
                eeg=config['forward']['eeg'],
                meg=config['forward']['meg'],
                mindist=config['forward']['mindist'],
                n_jobs=1,
                verbose=False
            )
        else:
            # Analytical sphere BEM (Berg-Scherg approximation)
            print(f"    Using analytical sphere BEM solution")

            fwd = mne.make_forward_solution(
                info,
                trans=None,  # No coordinate transform needed (already aligned)
                src=src,
                bem=bem,
                eeg=config['forward']['eeg'],
                meg=config['forward']['meg'],
                mindist=config['forward']['mindist'],
                n_jobs=1,
                verbose=False
            )

    elif bem_type == 'ellipsoid':
        # Ellipsoid BEM requires numerical solution
        print(f"    Computing numerical ellipsoid BEM solution")

        # Create BEM solution from ellipsoid surfaces
        bem_solution = mne.make_bem_solution(bem['surfs'], verbose=False)

        fwd = mne.make_forward_solution(
            info,
            trans=None,  # No coordinate transform needed
            src=src,
            bem=bem_solution,
            eeg=config['forward']['eeg'],
            meg=config['forward']['meg'],
            mindist=config['forward']['mindist'],
            n_jobs=1,
            verbose=False
        )

    else:
        raise ValueError(f"Unknown BEM type: {bem_type}")

    # Get forward solution statistics
    n_sources = fwd['nsource']
    n_channels = fwd['nchan']

    print(f"    ✓ Forward solution: {n_channels} channels × {n_sources} sources")

    # MNE silently discards sources that fall outside the inner-skull surface,
    # so a source space and the forward built from it can disagree with nothing
    # in the log to say so. That is how 868dea9's olfactory-bulb problem went
    # unnoticed — 30% of bulb sources outside the standard ellipsoid — and how
    # the sphere BEM was found to drop 25% of the anatomical surface (X40).
    #
    # Anything downstream that indexes source-space arrays by forward column
    # (parcel assignments, per-vertex normals) is silently misaligned when this
    # happens, so it is worth saying loudly.
    n_requested = int(sum(s['nuse'] for s in src))
    kept = kept_source_indices(src, fwd['src'])
    if len(kept) != n_sources:  # pragma: no cover - MNE invariant
        raise RuntimeError(
            f"forward has {n_sources} sources but its source space lists "
            f"{len(kept)} in-use vertices"
        )

    # Re-emit every per-source array restricted to the sources MNE kept, in
    # forward column order. Downstream steps (inverse export, ROI extraction)
    # index these by STC row, and until this existed they truncated from the
    # front instead, which assigns every row after the first dropped source to
    # the wrong coordinate and parcel.
    source_coords_mm = np.asarray(previous_outputs['source_coords_mm'])
    if len(source_coords_mm) != n_requested:
        raise ValueError(
            f"source_coords_mm has {len(source_coords_mm)} rows but the source "
            f"space has {n_requested} in-use vertices"
        )
    source_coords_kept = source_coords_mm[kept]

    roi_assignments = None
    if len(src) > 0 and src[0].get('roi_assignments') is not None:
        roi_assignments = np.asarray(src[0]['roi_assignments'])
        if len(roi_assignments) != n_requested:
            raise ValueError(
                f"roi_assignments has {len(roi_assignments)} entries but the "
                f"source space has {n_requested} in-use vertices"
            )
        roi_assignments = roi_assignments[kept]

    if n_sources < n_requested:
        dropped = n_requested - n_sources
        print(f"    ⚠️  BEM CONTAINMENT: {dropped:,} of {n_requested:,} sources "
              f"({dropped / n_requested:.1%}) fall outside the inner skull and "
              f"were dropped from the forward.")
        print(f"        Either the conductor is too small for this source space, "
              f"or the source space extends beyond the brain. Source "
              f"coordinates and parcel assignments have been restricted to the "
              f"{n_sources:,} sources the forward kept.")

    # Save intermediate data and create visualizations
    if config['outputs'].get('save_intermediate', True):
        from ..utils.io_utils import save_pickle, get_data_dir, get_figures_dir
        from ..utils.step_visualizations import visualize_step4_forward

        data_dir = get_data_dir(config)
        figures_dir = get_figures_dir(config)

        # Save forward solution
        save_pickle(fwd, data_dir / 'step4_forward.pkl')
        print(f"    Saved: {data_dir / 'step4_forward.pkl'}")

        # Create forward solution visualization
        fig = visualize_step4_forward(fwd, info, figures_dir / 'step4_forward.png')
        print(f"    Saved: {figures_dir / 'step4_forward.png'}")
        import matplotlib.pyplot as plt
        plt.close(fig)

    return {
        'fwd': fwd,
        # Per-source arrays aligned with the forward's columns. These override
        # the source-space step's versions in `previous_outputs`.
        'source_coords_mm': source_coords_kept,
        'n_sources': int(n_sources),
        'roi_assignments': roi_assignments,
        'kept_source_indices': kept,
    }


def kept_source_indices(src_before, src_after):
    """Row indices (into the pre-forward source arrays) of the sources a forward kept.

    ``mne.make_forward_solution`` discards sources outside the inner skull and
    records the survivors only in ``fwd['src'][i]['vertno']``. Per-source
    arrays built by the source-space step are ordered entry by entry, each in
    ``vertno`` order, so the position of each surviving vertex in the original
    entry's ``vertno`` (plus the entry offset) is its row in those arrays.
    """
    idx = []
    offset = 0
    for before, after in zip(src_before, src_after):
        before_v = np.asarray(before['vertno'])
        after_v = np.asarray(after['vertno'])
        pos = {int(v): i for i, v in enumerate(before_v)}
        try:
            rows = np.array([pos[int(v)] for v in after_v], dtype=int)
        except KeyError as e:
            raise ValueError(
                f"forward source space lists vertex {e} that the input source "
                f"space does not have"
            ) from None
        idx.append(offset + rows)
        offset += len(before_v)
    if len(src_before) != len(src_after):
        raise ValueError(
            f"source space has {len(src_before)} entries, forward's has "
            f"{len(src_after)}"
        )
    return np.concatenate(idx) if idx else np.zeros(0, dtype=int)
