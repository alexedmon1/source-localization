"""Step 1.5: EEG Data Loading.

Load EEG data from file and prepare epochs for inverse solution.
"""

import mne
import numpy as np
from pathlib import Path


def align_channels_to_info(inst, info, label="EEG"):
    """Reorder (and if necessary trim) an Epochs/Raw object to ``info``'s channel order.

    The forward solution is built from the electrode-registration ``info``, so
    its rows follow that channel order. The inverse step multiplies the
    operator by ``epochs.get_data()`` directly, which follows the *file's*
    channel order. Nothing used to reconcile the two, so a ``.set`` whose
    channels were stored in a different order (the eeg-preprocess output stores
    them E23, E22, E30, ...) silently paired every leadfield row with the wrong
    channel's data. Positions were copied by name, so the misalignment left no
    trace in the info either.

    Raises if any registered electrode is absent from the recording. Channels
    present in the recording but not registered are dropped, loudly.
    """
    wanted = list(info['ch_names'])
    have = list(inst.ch_names)

    missing = [c for c in wanted if c not in have]
    if missing:
        raise ValueError(
            f"{label} recording lacks {len(missing)} registered electrode(s): "
            f"{missing}. Recording channels: {have}. The forward solution is "
            f"built for the registered electrodes, so the data cannot be aligned."
        )

    extra = [c for c in have if c not in wanted]
    if extra:
        print(f"    ⚠️  Dropping {len(extra)} channel(s) not in the electrode "
              f"registration: {extra}")

    if have != wanted:
        if not extra:
            print(f"    Reordering channels to match electrode registration "
                  f"(file order started {have[:4]}..., registration order "
                  f"starts {wanted[:4]}...)")
        inst.reorder_channels(wanted)

    if list(inst.ch_names) != wanted:
        raise RuntimeError("channel alignment failed")  # pragma: no cover
    return inst


def run(config, previous_outputs):
    """
    Load EEG data and create epochs.

    Parameters
    ----------
    config : Config
        Pipeline configuration
    previous_outputs : dict
        Outputs from previous steps containing:
        - 'info': mne.Info - EEG measurement info from electrode registration

    Returns
    -------
    outputs : dict
        Dictionary containing:
        - 'epochs': mne.Epochs - EEG epochs
        - 'raw': mne.io.Raw - Raw EEG data (optional, if loaded from raw)
    """
    print("  Loading EEG Data")

    # Get EEG file path from config
    eeg_file = Path(config['inputs']['eeg_file'])

    if not eeg_file.exists():
        raise FileNotFoundError(f"EEG file not found: {eeg_file}")

    print(f"    EEG file: {eeg_file.name}")

    # Get info from electrode registration
    info = previous_outputs['info']

    # Try loading as epochs first (EEGLAB .set files can be epochs or raw)
    try:
        print(f"    Attempting to load as epochs...")
        epochs = mne.io.read_epochs_eeglab(str(eeg_file), verbose=False)

        # Put the channels in the registration's order (the forward's row
        # order) before anything downstream reads the data array.
        align_channels_to_info(epochs, info)

        # Update channel info from electrode registration (has correct positions)
        for ch_idx, ch_name in enumerate(epochs.ch_names):
            info_idx = info['ch_names'].index(ch_name)
            epochs.info['chs'][ch_idx]['loc'] = info['chs'][info_idx]['loc']

        # Set average EEG reference (required for inverse modeling)
        print(f"    Setting average EEG reference...")
        epochs.set_eeg_reference(projection=True, verbose=False)

        print(f"    ✓ Loaded epochs: {len(epochs)} epochs, {epochs.info['sfreq']} Hz")
        print(f"      Channels: {len(epochs.ch_names)}")
        print(f"      Time range: [{epochs.times[0]:.3f}, {epochs.times[-1]:.3f}] s")

        return {
            'epochs': epochs,
            'raw': None
        }

    except Exception as e:
        # Fall back to loading as raw
        print(f"    Could not load as epochs, trying as raw...")
        try:
            raw = mne.io.read_raw_eeglab(str(eeg_file), preload=True, verbose=False)

            align_channels_to_info(raw, info)

            # Update channel info from electrode registration
            for ch_idx, ch_name in enumerate(raw.ch_names):
                info_idx = info['ch_names'].index(ch_name)
                raw.info['chs'][ch_idx]['loc'] = info['chs'][info_idx]['loc']

            # Set average EEG reference (required for inverse modeling)
            print(f"    Setting average EEG reference...")
            raw.set_eeg_reference(projection=True, verbose=False)

            print(f"    ✓ Loaded raw: {len(raw.ch_names)} channels, {raw.info['sfreq']} Hz, {raw.times[-1]:.2f}s")

            # Create epochs from raw data
            # Use a simple approach: create fixed-length epochs
            print(f"    Creating epochs from raw data...")

            # Create events every 1 second
            n_times = len(raw.times)
            sfreq = raw.info['sfreq']
            epoch_length = 1.0  # 1 second epochs
            n_samples_per_epoch = int(epoch_length * sfreq)

            # Create events at regular intervals
            n_epochs = n_times // n_samples_per_epoch
            events = np.zeros((n_epochs, 3), dtype=int)
            events[:, 0] = np.arange(n_epochs) * n_samples_per_epoch
            events[:, 2] = 1  # Event ID

            # Create epochs
            epochs = mne.Epochs(
                raw,
                events=events,
                event_id={'stim': 1},
                tmin=0,
                tmax=epoch_length - 1/sfreq,  # Just under 1 second
                baseline=None,
                preload=True,
                verbose=False
            )

            print(f"    ✓ Created {len(epochs)} epochs of {epoch_length}s each")

            return {
                'epochs': epochs,
                'raw': raw
            }

        except Exception as e2:
            raise RuntimeError(f"Could not load EEG file as epochs or raw: {e2}")
