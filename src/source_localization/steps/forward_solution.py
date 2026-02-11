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
        'fwd': fwd
    }
