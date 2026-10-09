# Validating source localization (and a new atlas)

This is the user guide to `source_localization.validation`: what each tool answers, how to run it, how to
validate a **new atlas** end to end, and what to report. It describes the code as it is in v0.6.0. Changes not
yet made are in [the validation design](DESIGN_validation_upgrade.md) and are marked **planned** below. The methods
supplements behind sections 5-6 (single source, two sources, ROI certainty, and their reasoning record) are in
[`methods/`](methods/). The older module reference (v0.4.0, API-level detail) is
[`src/source_localization/validation/README.md`](../../src/source_localization/validation/README.md).

## Contents

1. [What validation can and cannot tell you](#1-what-validation-can-and-cannot-tell-you)
2. [Which tool answers which question](#2-which-tool-answers-which-question)
3. [Validating a new atlas, step by step](#3-validating-a-new-atlas-step-by-step)
4. [Forward-inverse accuracy: the `validate` CLI](#4-forward-inverse-accuracy-the-validate-cli)
5. [Parcel certainty and displacement](#5-parcel-certainty-and-displacement)
6. [Calibrated location and two sources](#6-calibrated-location-and-two-sources)
7. [Connectivity: planted-network validation](#7-connectivity-planted-network-validation)
8. [What to report](#8-what-to-report)
9. [Results you should not quote](#9-results-you-should-not-quote)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. What validation can and cannot tell you

Every tool here plants sources with a known position, simulates the scalp EEG through a forward model, and scores
what comes back against the truth. The answers are only as honest as the simulation, so every run states its
**regime**. Both regimes are defined in one place, `source_localization.validation.regime`:

| | **`realistic`** (default from v0.6.0) | **`legacy`** (v0.5.x behaviour) |
|---|---|---|
| Truth signal | At the **requested** position, through a head model drawn from a prior: registration shift N(0, 0.3 mm) per axis and skull conductivity 0.5-2x. The inverse keeps the nominal model, so model error is present | The inversion's **own** forward model, at the nearest source-grid position (the "inverse crime") |
| Noise | **Recorded** background from your files if `validation.background` is set; otherwise generated, and the output says so | Always generated (white by default) |
| Noise-only control | Run, and reported as `noise_only_control` | None |
| Use it for | All new results | **Only** reproducing numbers produced before v0.6.0 |

**"Legacy" means exactly the right-hand column and nothing else.** It is not "deprecated code" or "old atlas", and
it is unrelated to the legacy *atlas names* (`full`, `coarse_22roi`) the validation CLI accepts. Legacy numbers are
best-case and **not comparable** with realistic numbers. A legacy run logs a warning, and every output records
which regime produced it (`validation_regime` in `metrics.json`).

**Every simulating component follows the regime:**
- the `validate` runner;
- `validate --batch` (`BatchValidationRunner`, option `regime=`);
- `RobustnessTest` (`regime=`, and `--regime` on the sweep scripts). `RobustnessTest`'s realistic regime needs the
  BEM: `from_pipeline_dir` loads it. A recorded background replaces **white** noise only; tests that sweep
  coloured noise types keep them, since varying them is the point.

Select the regime with `validation.regime` in the config, or `--regime realistic|legacy` on the CLI (section 4).
In the realistic regime, other model errors still exist: head size, array rotation, per-electrode misplacement, BEM
shape. The prior covers registration and skull conductivity only. A forward-mismatch check with fixed ground-truth
conductivities is `validation.forward_model_mismatch: true` plus `ground_truth_conductivities: [brain, skull,
scalp]`; in the realistic regime the head-model prior perturbs around those ground-truth conductivities.

**Validation is a property of the configuration** (montage, BEM, source space, inverse method, sampling mode, atlas
and SNR), not of the package. Re-run it for the configuration you use, and report depth-stratified results.

## 2. Which tool answers which question

| Question | Tool | Entry point | Status |
|---|---|---|---|
| Is the atlas consistent with the template? | `atlas_bundle` | `load_bundle(...)` | current |
| How far off is the reconstructed peak, by depth and SNR? | `runner`, `robustness`, `metrics` | `source-localization validate` | current (inverse crime by default) |
| How often does the pipeline name the right parcel, and how much should a named parcel be believed? | `roi_certainty` | `scripts/run_roi_certainty.py` | current |
| How far, in mm, is an attribution off? | `displacement` | Python API | current (the MS1 statistic) |
| Given the data, where is the source, with calibrated credible regions? | `posterior` (`DipolePosterior`) | `scripts/run_posterior_orbital.py` | current; assumes an exact forward model and white noise |
| Are there two sources, and how far apart? | `two_source` | `scripts/run_two_source_posterior.py` | current |
| How blurred is the operator (noise-free)? | `resolution` | `scripts/run_resolution_map.py` | partly superseded (section 9) |
| Can two blobs be seen at a threshold? | `separability` | `scripts/run_separability.py` | superseded framing (section 9) |
| Is a connectivity network resolvable? Is its direction readable? | `networks` | `source-localization validate --networks spec.yaml` | current (section 7). The old `connectivity` module has no ground truth and is not a validation |

## 3. Validating a new atlas, step by step

You need:
- the package installed (`uv venv && uv pip install -e .`);
- your montage's electrode CSV;
- one EEG recording from that montage (`.set`). It is used only to build the geometry; its data are not scored.

### Step 1. Register the atlas

1. Put the label volume and its `roi_mapping.json` (and optionally `roi_categories.yaml`) under
   `src/source_localization/data/atlas/<your_atlas>/`.
2. Add an entry to `src/source_localization/data/atlas/registry.yaml` (copy an `allen32`-style entry; `inputs:`
   are paths, `meta:` is descriptive). That is the only step needed to make `--atlas <name>` selectable in both
   CLIs.
3. Check the bundle **before anything else**: template and labels must share one voxel grid and one affine
   convention.

   ```python
   from source_localization.validation import load_bundle
   bundle = load_bundle(template="data/atlas/Atlas_3DRois.nii",
                        labels="data/atlas/<your_atlas>/labels.nii.gz",
                        roi_mapping="data/atlas/<your_atlas>/roi_mapping.json",
                        name="<your_atlas>")
   print(bundle.summary())       # raises BundleConsistencyError on a mismatch
   ```

   Details: [Adding an atlas](../guides/adding_an_atlas.md).
4. Add its parcel count and coverage to `meta:`. `tests/test_atlas_registry.py` checks them against the files, so
   run `pytest tests/test_atlas_registry.py`.

### Step 2. Build one pipeline run for the geometry

The scripts in sections 5-6 read the BEM, montage and source space from a pipeline run (`--pipeline-dir`). They
need an **ellipsoid BEM**, so use an ellipsoid preset. `shell_ellipsoid` is what `sloreta_attributor` expects:

```bash
source-localization run --preset shell_ellipsoid --atlas <your_atlas> \
    --eeg /path/to/one_recording.set --output /path/to/val/pipeline_shell_ellipsoid
```

The run directory must contain `bem_cache/ellipsoid_3layer.pkl`, `data/step1_info.pkl`,
`data/step3_source_space.pkl`, `data/step3_source_coords_mm.npy` and `data/step4_forward.pkl`. Keep it: every
later step reuses it.

### Step 3. Forward-inverse accuracy with your atlas

See section 4. In short:

```bash
source-localization validate --test-dir /path/to/val --config configs/ --all \
    --atlas <your_atlas> --test-mode combined --snr 0 10 20 --trials 25
```

### Step 4. Parcel certainty and displacement

See section 5. In short:

```bash
python scripts/run_roi_certainty.py --pipeline-dir /path/to/val/pipeline_shell_ellipsoid \
    --output-dir /path/to/val/roi_certainty --atlas <your_atlas> --n-per-parcel 30
```

Then compute peak displacement from the same confusion matrix (section 5.2).

### Step 5. Decide what the atlas can support

- Parcels with low recall **or** low precision cannot carry a per-parcel claim at that SNR and depth.
- Report them merged with the parcels they are confused with, or not at all.
- The confusion matrix from step 4 shows the merges. A displacement near the parcel's radius means "this parcel
  or its neighbour".
- Always state the SNR: an atlas that resolves at +20 dB may collapse at 0 dB.

## 4. Forward-inverse accuracy: the `validate` CLI

### Layout

```
my_validation/
├── configs/                 # one pipeline config per configuration under test
│   └── shell_ellipsoid_sLORETA.yaml
└── results/                 # written by the runner, one directory per config (atlas-suffixed if not antwerp)
```

### Config

A validation config is a normal pipeline config plus a `validation:` block (full example in the main README,
[Validation config format](../../README.md#validation-config-format)):

```yaml
validation:
  snr_levels: [0, 10, 20]   # dB
  n_trials: 25              # per test position
  test_mode: combined       # roi_centroids | uniform_grid | combined (recommended)
  grid_spacing_mm: 1.0
  dipole: {amplitude_nAm: 50.0, duration_s: 1.0, sfreq: 500.0}

  # --- the regime (section 1); everything below is optional ---
  regime: realistic                       # or legacy (v0.5.x assumptions; reproducing old numbers only)
  truth_head_model:                       # realistic only
    shift_sd_mm: 0.3                      # registration SD per axis, or [x, y, z]
    skull_factor_range: [0.5, 2.0]        # log-uniform multiple of the skull conductivity
    n_models: 16                          # head models drawn per run; each trial uses one at random
  background:                             # realistic only: recorded noise instead of generated
    files: "/data/rest/*.set"             # EEGLAB .set (epoched or continuous), or a list of paths
  noise_only_control: {n_trials: 200}     # realistic only
  regime_seed: 20261009
```

**Backgrounds** must contain every montage channel by name, at the simulation's sampling rate (resample first
otherwise). Each trial takes a random window of `duration_s` from a random epoch, average-referenced. Use
recordings from the same montage and preparation as the data you will analyse. Real backgrounds contain real
activity, which is the point.

### Commands

```bash
source-localization validate --test-dir ./my_validation --config configs/ --list     # what will run
source-localization validate --test-dir ./my_validation --config configs/ --all --atlas <your_atlas>
source-localization validate --test-dir ./my_validation --config configs/ --all --quick   # 5 ROIs, 1 trial, SNR 10
source-localization validate --summarize ./my_validation/results/                      # depth-stratified summary
source-localization validate --compare ./my_validation/results/a/ ./my_validation/results/b/
```

| Option | Meaning |
|---|---|
| `--atlas` | Any registry name. **The validation CLI defaults to `full` (= Antwerp)**, so always pass it |
| `--test-mode` | `roi_centroids` (ROI accuracy at parcel centroids), `uniform_grid` (localization error and depth), `combined` |
| `--regime` | `realistic` (default) or `legacy`; overrides `validation.regime` (section 1) |
| `--snr`, `--trials`, `--rois` | Override the config |
| `--batch --presets ... --methods ...` | Sweep presets × inverse methods (`batch_runner`); follows `--regime` |

**Outputs:** `results/<config>/metrics.json`, `validation_report.html` and `figures/`. `metrics.json` holds:
- per SNR: localization error, ROI accuracy, depth-stratified error;
- `validation_regime`: which regime produced the numbers, with its full description;
- realistic regime only:
  - `truth_head_models`: the prior and the drawn models, and how many positions fell outside the inner skull
    (those are simulated at the snapped grid source);
  - `noise_only_control`: on noise alone, the share of answers going to the most-named ROI (`top_share`), its
    ratio to a uniform spread (`ratio_to_uniform`; flagged above 3), and `normalized_entropy` (1 = spread like
    uniform). A flagged control means that ROI attracts the readout from noise. Treat that ROI's accuracy with
    suspicion.

**Forward mismatch:** set `forward_model_mismatch: true` and `ground_truth_conductivities: [brain, skull, scalp]`
in the `validation:` block. The data are then simulated with those conductivities and inverted with the config's
own. In the realistic regime the head-model prior perturbs around them.

**Robustness sweeps** (follow the regime; `RobustnessTest.from_pipeline_dir(..., regime=...)`) (`robustness.RobustnessTest.from_pipeline_dir(...)`) cover SNR (`run_snr_test`), noise level
(`run_noise_test`), noise type (`run_noise_type_test`), amplitude (`run_amplitude_test`), two dipoles
(`run_two_dipole_test`) and resolvability (`run_resolvability_test`). Section 9 lists which of their
conclusions are superseded.

## 5. Parcel certainty and displacement

### 5.1 `run_roi_certainty.py`

It scores two arms on the same simulated truths and noise:
- the **ceiling:** the calibrated posterior's best parcel;
- the **deployed:** the sLORETA peak's parcel on the pipeline's shell.

```bash
python scripts/run_roi_certainty.py --pipeline-dir DIR --output-dir OUT --atlas <your_atlas> \
    [--spacing-mm 0.5] [--n-per-parcel 30] [--headline-snr 10] [--seed 0] [--replot]
```

| Output | What it shows |
|---|---|
| `roi_accuracy_vs_snr.png` | Overall parcel accuracy by SNR, both arms |
| `roi_confusion.png` | True × attributed parcel, at the headline SNR |
| `roi_reliability.png` | Calibration: when the posterior says p for a parcel, is it right p of the time? |
| `roi_tolerance.png` | Accuracy if a neighbouring parcel counts as correct |
| `roi_belief_by_parcel.png` | How much belief each true parcel receives |
| `cache/roi_certainty_<atlas>_*.npz` | The simulated data; `--replot` redraws from it |

Truths are balanced across parcels (`n_per_parcel` each), so small parcels are not drowned out by large ones.

### 5.2 Peak displacement (Python)

`displacement` turns a confusion matrix into millimetres: the expected distance between a source's parcel and the
parcel it is credited to, with a chance reference (`null_displacement_mm`) and `information_gain` (1 = perfect,
0 = chance). The example below runs as written (checked 2026-10-09 against a shell-ellipsoid run):

```python
import numpy as np
import pandas as pd
from source_localization.validation.posterior import DipolePosterior
from source_localization.validation import roi_certainty as rc
from source_localization.validation.displacement import parcel_geometry, displacement_table

PIPE, ATLAS = "/path/to/val/pipeline_shell_ellipsoid", "<your_atlas>"
dp = DipolePosterior.from_pipeline_dir(PIPE, spacing_mm=0.5, cache_dir="out/cache")
pmap = rc.ParcelMap(dp.positions_mm, atlas=ATLAS, electrode_pos_mm=dp.electrode_pos_mm)
rng = np.random.default_rng(0)
truth = rc.sample_truth_positions(pmap, n_per_parcel=30, rng=rng)

attribute = rc.sloreta_attributor(PIPE, ATLAS)          # or rc.posterior_attributor(dp, pmap)
pooled, _ = rc.confusion_matrices(attribute, dp, pmap, truth, snr_db=10.0, estimator="sloreta", rng=rng)
confusion = pd.DataFrame(pooled.counts, index=pooled.parcel_names, columns=pooled.parcel_names)

name_of = dict(zip(pmap.parcel_ids, pmap.parcel_names))
geometry = parcel_geometry(dp.positions_mm, [name_of.get(l) for l in pmap.labels])
print(displacement_table(confusion, geometry))           # per true parcel, sorted by displacement
```

- **Scoring a different operator** (Monte Carlo, hybrid): use `displacement.operator_attributor(operator,
  operator_parcels, pmap.parcel_names)` as the attributor.
- **Comparing operators:** score them against the **same** centroids, or the comparison is not matched.

## 6. Calibrated location and two sources

These do not depend on the atlas (they work on positions), but they answer the question an atlas claim rests on.

> **Naming:** `DipolePosterior` is a Bayesian posterior over **source location** ("where is this source?"). It is
> unrelated to Bayesian **group statistics** (effect sizes across animals, ROPE decisions), which live in the
> analysis packages, not here. The validation regime (section 1) is likewise about the simulation, not about any
> statistical prior.

```bash
# single source: does a p% credible region contain the truth p% of the time? Plus "orbital" figures
python scripts/run_posterior_orbital.py --pipeline-dir DIR --output-dir OUT [--spacing-mm 0.5] [--n-trials N]

# two sources: Bayes factor (two vs one) against a matched one-source null, separation posterior
python scripts/run_two_source_posterior.py --pipeline-dir DIR --output-dir OUT [--depth-series] [--replot]
```

- `DipolePosterior` assumes the forward model is exact and the noise white with known size. Its coverage is
  verified under those assumptions, not under model error (design, A4).
- **Noise-only control:**
  - `run_posterior_orbital.py` writes `posterior_noise_only_control.json`: credible radii on noise alone at each
    SNR's noise level. They should approach the whole-grid radius; small radii mean the posterior reports a
    location where there is none.
  - `run_roi_certainty.py` writes `roi_noise_only_control.json`: how often a parcel reaches p >= 0.5 or 0.9 on
    noise, and which parcel the noise-only argmax favours. For the MEA30 shell run at 10 dB: never confident
    (0%), but the argmax drifts to Cerebellum at 9x uniform, so treat low-probability Cerebellum attributions with
    suspicion.
  - `--noise-only-trials 0` skips either.
- **Two sources:** `run_two_source_posterior.py` already scores every Bayes factor against a **matched one-source
  null** (a 10% false-alarm threshold), a stricter control than noise alone.

## 7. Connectivity: planted-network validation

**`validation.connectivity` is not a validation.** It compares electrode and ROI connectivity on real data with
no ground truth, so it cannot tell you whether a network is resolvable. Use **planted-network validation**
(`validation.networks`) instead.

### What it does

For declared node pairs it plants coupled sources at known positions. Each truth is simulated through its own
head model drawn from the prior (section 1), into **real resting EEG of your montage**. The connectivity is then
read out at the electrodes and at source nodes, with **your own metric code**. Every edge is z-scored against the
**same background with nothing planted**, which removes the real coupling the backgrounds contain.

- **Undirected (default).** The resolvable-network table: for every pair, readout, metric, band and coupling kind
  (lagged, envelope, zero-lag), the lowest SNR at which:
  - the pair is the strongest edge (top-1) in >= 70% of truths;
  - a single uncoupled source creates <= 10% false edges;
  - both hold at that SNR and every higher one.

  Also reported: detection AUC (does the planted edge rise at all?) and false-edge rates under the controls.
- **Directed (optional).** Whether the readout tells which side leads, both at the exact pair and along coarse
  axes you define (e.g. anterior-posterior), with controls that have no direction (zero-lag coupling of equal and
  unequal strength, single sources, uncoupled sources).
  - Built-in reference measures: PSI, Granger and time-reversed Granger.
  - Your own directed measures (dPLI, DTF, TE...) are injected like the undirected ones.

### Running it

```bash
source-localization validate --networks network_spec.yaml [--networks-output DIR] [--workers 9]
```

`network_spec.yaml` (all keys: `validation.networks.load_spec`):

```yaml
pipeline_dir: /path/to/run              # one completed pipeline run of your montage (hybrid recommended)
backgrounds:                            # real resting EEG, same montage; draws alternate across groups
  - {group: KO, files: "/data/rest/ko/*.set"}
  - {group: WT, files: "/data/rest/wt/*.set"}
n_epochs: 30                            # 2 s epochs per simulated recording (60 s)
n_truths: 50
bands: {theta: [6, 8], beta: [15, 25], low_gamma: [35, 45]}
snrs_db: [-10, -5, 0, 5]
truth_labels: {collapse_volume: true}   # pair node names: parcels, volume parcels as one "Deep"
readouts:                               # these three are the default; R-region is opt-in
  - {name: R-el, type: electrodes}
  - {name: R-parcel, type: nodes, collapse_volume: true}
  - {name: R-parcel-full, type: nodes, collapse_volume: false}   # volume parcels kept, scored at the truth level
  - {name: R-region, type: nodes, collapse_volume: true, merge_map: regions.json}    # {parcel: region}
pairs:
  - {class: dorsal_far, a: Frontal_Anterior, b: Retrosplenial_L, axis: AP}   # axis only for the directed stage
  - {class: homologous, a: Motor_L, b: Motor_R, axis: LR}
metrics:                                # your analysis's own code: 'module:function' or '/path/file.py:function'
  callable: /path/to/source-analytics/src/source_analytics/spectral/connectivity.py:compute_connectivity_matrix
  names: [imag_coherence, wpli, aec, coherence]
  abs: []
workers: 9
directed:                               # optional
  axes:
    AP: {anterior: [Frontal_Anterior, Motor_L, Motor_R, Olfactory_Bulb], posterior: [Retrosplenial_L, ...]}
    LR: {left: [Motor_L, Somatosensory_L, ...], right: [Motor_R, Somatosensory_R, ...]}
  measures:
    dpli: {callable: ".../connectivity.py:compute_connectivity_matrix", key: dpli, net: antisym,
           kwargs: {metrics: [dpli]}}
```

The metric callable must follow source-analytics' `compute_connectivity_matrix` contract:
`f(node_ts: dict[name, 1-D array], sfreq, {band: (lo, hi)})` returning `(results[band][metric] -> (n, n) array,
names)`.

**Runtime:** about 1-2 h for 12 pairs x 3 bands x 4 SNRs x 5 kinds x 50 truths on 9 workers. Workers are pinned to
one BLAS thread each.

### Outputs

| File | What it holds |
|---|---|
| `resolvable_networks.csv` | The deliverable: `resolvable_from_db` (lowest SNR, or `never`) and per-SNR top-1 and false-edge rates |
| `detection.csv` | AUC of the planted edge vs a single source and vs two uncoupled sources |
| `sims.csv.gz` | One row per simulation, readout and metric |
| `electrode_assignment_<readout>.csv` | Which node each electrode was scored as |
| `directed_readability.csv`, `directed_outcomes.csv.gz` | Directed stage, if run |
| `summary.json` | Counts and the spec |

### Reading it

- **Detected but not resolvable** (high AUC, top-1 well under 0.70) means:
  - the readout sees that coupling changed, but names a neighbouring pair ("ghost interactions");
  - report edge-level results as "coupling changed near these nodes", not "between them";
  - or merge the nodes it confuses and re-run.
- **Choosing nodes (MEA30, probability-atlas Phase 11):**
  - all readouts **detect** planted coupling about equally (median AUC 0.94-0.98);
  - what differs is **attribution**:
    - electrodes attribute cortical pairs best;
    - volume parcels collapsed into one Deep node are slightly better than the full parcels for cortical pairs;
    - the **full parcels** (the 11 deep parcels kept as separate filters, scored as Deep) are better for
      **cortex-Deep** pairs (e.g. Motor_L-Deep top-1 0.37 vs 0.09), though they cannot name *which* deep
      structure (top-1 <= 0.08).
  - **Merged regions hide the coupling inside them** by construction, and on MEA30 lost the frontal pairs. Use
    them only for region-level questions.
  - None of these makes a network resolvable.
- **Direction:**
  - a measure whose direction flips with the planted orientation, and stays quiet on the controls, reads
    direction;
  - anything else does not.
  - dPLI's sign also depends on the relative polarity of the two sources, which is unknown in real data.

### Known results for MEA30

These are from the probability-atlas runs this harness reproduces; the acceptance tests are in
`tests/test_network_validation.py`.
- Coupling is detected (AUC 0.92-0.99 at +5 dB), but **no node pair is resolvable** in electrodes, parcels or
  merged regions.
- **Direction is not readable** on the cortical axes. dPLI's sign followed the sources' polarity, and DTF was
  biased toward the stronger source.
- Details: source-analytics `docs/methods/CONNECTIVITY_METHODS.md`, "Validation on MEA30".

## 8. What to report

For any validation number, state:
- the **regime** (`realistic` or `legacy`; from `validation_regime` in the output). Never compare across regimes;
- the **configuration:** preset, BEM, source space, inverse method and SNR parameter, source sampling mode, atlas
  (registry name), package version;
- **depth-stratified** results, never a pooled headline;
- **several SNRs** (e.g. 0, 10, 20 dB): resolvability is strongly SNR-dependent;
- the **noise model** (white, correlated or 1/f, recorded) and whether the forward model was **mismatched**;
- for confidence statements, the **noise-only** false-confidence rate;
- for parcel claims, recall **and** precision (or displacement) for the parcels claimed.

## 9. Results you should not quote

From earlier validation rounds, now replaced by the calibrated-posterior work:
- **Spatial dispersion** (`resolution`): it scores 78-96% of what a completely flat map would, so it has almost no
  dynamic range. Peak localization error from the same module stands.
- **Threshold separability percentages** ("two blobs at 25% of peak"): a display threshold, not an auditable
  probability. The depth-matching finding (unequal reconstructed amplitudes across depth) stands.
- **Any resolvability claim from a single SNR or pooled across depth,** e.g. "cannot resolve two sources at any
  separation". Resolvability depends strongly on SNR and depth.
- **A single headline ROI accuracy** for a preset: accuracy depends on depth, atlas and source space.

## 10. Troubleshooting

| Symptom | Likely cause |
|---|---|
| ROI accuracy collapses (e.g. ~5%) or all sources map to a few ROIs | Affine-convention mismatch between labels and template. Run `load_bundle`, and read any label volume through `utils/atlas.py` (`get_true_affine`), never `nii.affine` |
| `--atlas` rejected | Name not in `registry.yaml`. The validation CLI also accepts the legacy names `full` (antwerp) and `coarse_22roi` (coarse22) |
| Results land in an unexpected directory | The validation CLI suffixes results with the atlas name unless it is Antwerp |
| Small parcels missing from the confusion matrix | No grid position strictly inside them at the chosen `--spacing-mm`; use 0.5 mm or check coverage in `registry.yaml` `meta:` |
| A script cannot find `bem_cache/ellipsoid_3layer.pkl` | The pipeline run used a sphere BEM; rebuild with an ellipsoid preset (step 2) |
| Validation numbers differ from an older report | Check the configuration list in section 8. Atlas, sampling mode and noise model all change the numbers |
