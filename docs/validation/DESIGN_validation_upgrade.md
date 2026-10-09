# Design: honest-by-default validation, and planted-network validation

**Status:** proposed 2026-10-09, for the maintainer's approval. Nothing here is implemented yet. The user guide
for what exists today is [the validation README](README.md).

## 1. Why

The validation package answers "how well does this pipeline localize?" with a forward-inverse simulation. Three
of its defaults make the answer optimistic, and one question it cannot answer at all:

| Default today | Consequence |
|---|---|
| The same forward model simulates the data and inverts it (the "inverse crime"), unless `validation.forward_model_mismatch` sets other ground-truth conductivities | Model error, which every real recording has, is absent. Accuracy is a ceiling |
| Noise is white Gaussian in the CLI runner; synthetic correlated and 1/f noise (`noise.py`) only in `RobustnessTest` and the sweep scripts | Real EEG backgrounds contain structured, non-stationary activity, which no synthetic generator reproduces |
| No noise-only condition | A method that reports a confident location (or parcel, or edge) from pure noise passes every test |
| `connectivity.py` compares electrode and ROI connectivity on real data, with no ground truth | It cannot say whether any network is resolvable, and it scores mne-connectivity rather than the metrics an analysis actually uses |

The probability-atlas method workspace (2026-10-02 to 2026-10-09, local, not part of this repo) built and tested
each missing piece on MEA30 data. It found:
- With fresh head models per truth and real backgrounds, the calibrated location readout kept its nominal
  coverage (89-94% at 90%) on Sentinel ASSR data.
- A pooled 2 Hz location readout on SL data reported 0.993 confidence for a "location" in **pure noise**. Only a
  pre-registered noise-only control caught it.
- On MEA30, planted coupling is detected (AUC 0.92-0.99 at +5 dB) but **no node pair is resolvable** in
  electrodes, parcels or merged regions. Direction is not readable on the cortical axes. dPLI's sign followed the
  sources' polarity, and DTF was biased toward the stronger source. These findings are in source-analytics'
  `docs/methods/CONNECTIVITY_METHODS.md` (commit e11dba5, unreleased after v0.8.2).

The probability atlas was closed as an atlas on 2026-10-09: it gave no usable location answer at the SNRs these
studies have. Its **validation components** are what this design brings into the package.

## 2. Decisions (taken with the maintainer, 2026-10-09)

1. **The honest settings become the defaults** (A1-A3 below). A `legacy` switch restores today's behaviour, so
   old numbers can be reproduced. Old and new numbers are not comparable, and reports must say which they used.
2. **Planted-network validation takes its connectivity metrics as an injected callable.** Neither this package
   nor source-analytics depends on the other. source-analytics' `compute_connectivity_matrix` already has the
   required signature.
3. **Directed scoring is included as an option** in the first version of B.
4. **The ensemble location posterior (A4) is deferred.** Its use is localization, which does not pay off on MEA30.
   A1-A3 already make the validation numbers honest.

## 3. Components

### A1. Truths under a perturbed head model

- **New module `validation/head_models.py`**, ported from probability-atlas `pa/ensemble.py` and `pa/geometry.py`
  (tests in its `tests/test_readout.py`):
  - `HeadModel(shift_mm=(dx, dy, dz), skull_factor)`;
  - `HeadModelPrior(shift_sd_mm=0.3, skull_range=(0.5, 2.0))`: per-axis normal registration shift, log-uniform
    skull-conductivity factor on 0.0042 S/m;
  - `Geometry.leadfield(model, positions)`: rebuilds the BEM with the scaled skull conductivity and shifts the
    electrodes, cached by a key built from the geometry, model and positions.
- **Use:** every simulated truth draws its own head model from the prior (seeded), and simulates through it. The
  inverse keeps the nominal model.
- **Config:** `validation.truth_head_model: {prior: default | legacy, shift_sd_mm, skull_range}`.
  - `legacy` = the inversion's own forward (today's behaviour).
  - The existing `forward_model_mismatch` option (fixed ground-truth conductivities) becomes the special case
    `shift_sd_mm: 0` with a fixed skull factor.
- **Known caveat, carried over:** MNE projects shifted electrodes back onto the BEM scalp, so a z shift acts about
  a third as strongly as the same shift in x or y. Documented, not corrected.

### A2. Real-noise backgrounds

- **`noise.py` gains `RecordedBackground`:** epochs from the user's own recordings (EEGLAB `.set` or MNE
  epochs).
  - Channels are matched to the montage by name, then average-referenced.
  - Each simulated recording draws N epochs without replacement, from a seeded, alternating draw over
    recordings. Groups can be balanced, as probability-atlas balanced KO and WT.
- **SNR definitions** (both available):
  - the existing broadband ratio;
  - the band-power ratio at each source's most sensitive electrode. This is the planted-network one, needed for
    oscillatory signals.
- **The no-plant reference:** every statistic that compares a planted condition with a null uses **the same
  background, unplanted**. Real backgrounds contain real structure, which this removes.
- **Fallback:** synthetic noise stays when no recordings are given, and the report says which was used.

### A3. The noise-only control, in every suite

Each suite runs a noise-only condition through the same readout and reports a false-confidence rate:

| Suite | Noise-only statistic | Flag |
|---|---|---|
| `runner` / `robustness` (peak localization, ROI accuracy) | Distribution of attributed ROIs under noise; the largest single ROI's share | One ROI attracts > 3x its uniform share |
| `posterior`, `two_source`, `roi_certainty` | The largest posterior probability under noise; the fraction ≥ 0.9; the two-source Bayes factor | Any fraction ≥ 0.9 above the nominal false-alarm rate |
| Planted networks (B) | Built in: control 1, no plant, sets every z | — |

Thresholds are reported, not used to fail a run.

### B. Planted-network validation (replaces `connectivity.py`)

**New subpackage `validation/networks/`**, ported from probability-atlas `pa/networks.py`, `pa/directed.py`
(tests in `tests/test_networks.py` and `tests/test_directed.py`) and the Phase 9-11 scripts. A1 provides the truths and A2 the backgrounds.

| Part | What it does |
|---|---|
| `nodes` | Readouts: **electrodes** (each assigned to the node whose sources it sees most strongly; the table is written out); **atlas parcels** through the existing Monte Carlo `build_roi_operator`; options to **collapse volume parcels into one Deep node** or to apply a **merge map** |
| `planting` | Two sources per truth, at positions drawn uniformly within each node of a declared pair. Kinds: lagged (fractional lag, under half a cycle of the band), zero-lag, envelope; controls: single, uncoupled, no plant. Directed options: both orientations, weak driver, zero-lag with unequal strength |
| `metrics` | The injected callable: `f(node_ts: dict[str, ndarray], sfreq, bands) -> (results[band][metric] (n x n), names)`, i.e. source-analytics' `compute_connectivity_matrix` contract. A list of metric names says which are directed (`i -> j` convention) |
| `scoring` | Edge z against the no-plant draws (per readout, metric, band, edge); detection AUC; top-1 attribution; false edges under the controls; "resolvable" = top-1 ≥ 0.70 and false edges under a single source ≤ 0.10 at that SNR and above. Directed: correct / wrong / none at the exact pair and on coarse axes (anterior-posterior, left-right, cortex-Deep), with false-direction rates under every control |
| `cli` | `source-localization validate networks --pipeline-dir DIR --background 'GLOB' --atlas NAME --pairs pairs.yaml --metrics module:function [--directed] [--workers N]` |
| outputs | `resolvable_networks.csv`, `directed_readability.csv` (optional), `detection.csv`, `summary.md`, and the raw node-level matrices (`.npy`), so other scorings need no re-run |

**Performance requirement:** pin each worker to one BLAS thread. Without it, probability-atlas ran 34x slower.
With it, 36,000 simulations take about 1-2 h on 9 workers.

**Acceptance test:** on the FORGE/MEA30 geometry and backgrounds, the port reproduces probability-atlas Phase 9's
resolvable-network table, and Phase 10's for the directed option. Same seeds, so identical up to floating point;
the port is a refactor, not a re-design.

**Default readout:** decided by probability-atlas Phase 11 (running at the time of writing). It compares the full
26 parcels, the parcels with the volume parcels collapsed, and merged regions.

### A4 (deferred). Ensemble location posterior

`DipolePosterior` assumes an exact forward model and white noise of known size. The probability-atlas
`EnsemblePosterior` averages the likelihood over head models and whitens with real noise, chosen by held-out
likelihood. It is not scheduled: revisit if a study needs calibrated location under model error.

## 4. Order of work (one pull request each)

| PR | Content | Changes numbers? |
|---|---|---|
| 1 | A1 + A2 + A3, the `legacy` switch, tests, README updates. Housekeeping: fix stale examples (`source-localization validate --test ...` in the main CLI help; `run_validation(test_name=...)` in `runner.py`'s docstring; neither exists) | **Yes:** validation results under the new defaults differ from earlier ones |
| 2 | B, with the acceptance test against Phase 9/10; remove `connectivity.py` (or keep it as `legacy` for one release) | New capability |
| 3 | Docs: move the validation write-ups from the external staging area (`validation-tests/docs/`) into `docs/validation/` | No |

## 5. Compatibility

- **Version:** 0.5.1 → 0.6.0 for PR 1 (the defaults change).
- **Reproducibility:** `validation.legacy: true`, or `--legacy` on the CLI, reproduces 0.5.x numbers.
- Every validation output records the truth head-model prior, the background source (recorded or synthetic) and
  the noise-only results. A result without them is a 0.5.x result.

## 6. Open questions

1. **Where the default backgrounds come from.** No recordings ship with the package, so a user must point at their
   own. A small anonymised background set could be bundled for the tests.
2. **The registration prior.** 0.3 mm per axis is an assumption. Per-mouse placement records would replace it.
3. **Electrode-shift projection** (the A1 caveat): accept it, or shift in the tangent plane instead.
