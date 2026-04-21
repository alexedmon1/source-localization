---
author: Alex Edmondson
affiliation: CCHMC
email: alex.edmondson@cchmc.org
study: {{STUDY_NAME}}
preset: {{PRESET}}                    # roi_based_ellipsoid (recommended) | shell_ellipsoid | ellipsoid_surface | ...
atlas: {{ATLAS}}                      # antwerp (default, 47 ROIs) | allen (64 ROIs)
phase: source-localization
project_dir: ~/research/{{STUDY_NAME}}/source-localization
study_dir: /mnt/arborea/{{STUDY_NAME}}
pipeline_dir: /home/edm9fd/sandbox/source-localization
pipeline_venv: /home/edm9fd/sandbox/source-localization/.venv/bin/python
---

<!-- AI Instructions:
Source-localization IRL project for mouse EEG.
- $PROJECT (this repo) is small, git-tracked; holds plan, QC notes, activity log
- $STUDY on /mnt/arborea holds raw EEG + derivatives (source localization output)
- ROI timeseries + per-subject pipeline outputs go to $DERIV; never into $PROJECT
- source-localization is invoked via the `source-localization` CLI (installed in $PIPELINE/.venv) or `uv run source-localization ...` from $PIPELINE
- Long batch jobs (study run, study analyze) MUST use nohup, logs to $STUDY/logs
- Before launching CPU-heavy batch jobs: `ps aux | grep -E 'source-localization|python.*pipeline.py'`
-->

# {{STUDY_NAME}} — Source Localization Plan

## 📁 Paths — Single source of truth

### `$PROJECT` — this IRL project (small, git-tracked on home drive)

- **`$PROJECT`** — `~/research/{{STUDY_NAME}}/source-localization` — this repo root
- **`$PLAN`** — `$PROJECT/plans` — main-plan.md, activity log, CSV log
- **`$QC_NOTES`** — `$PROJECT/qc-notes` — markdown notes on per-subject QC and exclusion decisions
- **`$VALIDATION`** — `$PROJECT/validation` — optional, dipole-simulation validation configs + reports

### `$STUDY` — study data (on arborea; can be local if volume fits)

- **`$STUDY`** — `/mnt/arborea/{{STUDY_NAME}}` — study root
- **`$SOURCEDATA`** — `$STUDY/sourcedata` — raw EEGLAB `.set/.fdt` files (or symlinks)
- **`$PARTICIPANTS`** — `$STUDY/participants.csv` — subject metadata (subject_id, group, ...)
- **`$DERIV`** — `$STUDY/derivatives/source_localization/{{PRESET}}` — per-subject pipeline outputs
  - `$DERIV/{subject}/pipeline/` — source localization intermediates
  - `$DERIV/{subject}/roi_timeseries/` — ROI timeseries .set files
  - `$DERIV/{subject}/analysis/` — band power / connectivity CSVs
  - `$DERIV/group/` — group-aggregated CSVs
- **`$STUDY_CFG`** — `$STUDY/study_config.yaml` — source-localization study config (authoritative)
- **`$EXCL`** — `$STUDY/exclusions/source_localization.csv` — canonical exclusions (subject,reason,date_added)
- **`$LOGS`** — `$STUDY/logs`

### Pipeline (read-only)

- **`$PIPELINE`** — `/home/edm9fd/sandbox/source-localization` — source-localization repo
- **`$SL`** — `source-localization` CLI (from `$PIPELINE/.venv/bin/`); invoke as `uv run source-localization ...` from `$PIPELINE` when in doubt

Rule: every section below refers to these by shorthand. If you need a new absolute path, add it here first.

---

## 🔧 First Time Setup — Run once when establishing the study

1. **Verify CLI installed**:
   ```bash
   cd $PIPELINE
   uv run source-localization --help
   uv run source-localization study --help
   ```
2. **Initialize study** (creates `$STUDY_CFG` + folder layout):
   ```bash
   cd $PIPELINE
   uv run source-localization study init $SOURCEDATA \
       --name "{{STUDY_NAME}}" \
       --preset {{PRESET}} \
       --atlas {{ATLAS}} \
       --output-dir $STUDY
   ```
3. **Populate `$PARTICIPANTS`** with `subject_id,group,...` (one row per `.set` file)
4. **Edit `$STUDY_CFG`** for this study (frequency bands, connectivity methods, epoch length)
5. **Initialize exclusion CSV**:
   ```bash
   echo "subject,reason,date_added" > $EXCL
   ```
6. **Snapshot `$STUDY_CFG` + `$PARTICIPANTS`** into `$PROJECT` (copy, don't symlink)
7. **Commit baseline** in `$PROJECT`: plan, config snapshot, empty QC notes

### Common skill library
<!-- Uncomment to use -->
<!-- Install Quarto: https://github.com/posit-dev/skills/tree/main/quarto/authoring -->

---

## ✅ Before Each Loop

- **Clean git tree** in `$PROJECT`: `git status`
- **Running-jobs check**: `ps aux | grep -E 'source-localization|python.*pipeline.py'`
- **Disk check**: `df -h /mnt/arborea`
- **Pipeline version**: `cd $PIPELINE && git log -1` — record for reproducibility
- Any step that writes to `$DERIV` must be idempotent (re-run = no-op or byte-identical output)
- Only `## One-Time Instructions` is plan-editable without explicit permission

---

## 🔁 Instruction Loop — Define the work for each iteration

<!-- 👤 AUTHOR AREA: Edit each loop. -->

### Loop task (current)

- **Phase:** <!-- single-subject validation | batch source localization | spectral/connectivity analysis | QC pass | preset comparison -->
- **Subjects:** <!-- all / subset / single -->
- **Preset:** <!-- roi_based_ellipsoid (default) | shell_ellipsoid | ellipsoid_surface | ... -->
- **Atlas:** <!-- antwerp | allen -->
- **Expected output:** <!-- $DERIV/{subject}/pipeline/, $DERIV/{subject}/roi_timeseries/, $DERIV/{subject}/analysis/ -->

### Command templates

**Single-subject (validation / spot-check):**
```bash
cd $PIPELINE
uv run source-localization run \
    --preset {{PRESET}} --atlas {{ATLAS}} \
    --eeg $SOURCEDATA/{{SUBJECT}}.set \
    --output $DERIV/{{SUBJECT}}
```

**Batch source localization across subjects:**
```bash
cd $PIPELINE
nohup uv run source-localization study run $STUDY_CFG \
    --jobs {{N_JOBS}} --verbose \
    > $LOGS/study_run_$(date +%Y%m%d_%H%M).log 2>&1 &
```

**Spectral + connectivity analysis (uses MNE wrappers):**
```bash
cd $PIPELINE
nohup uv run source-localization study analyze $STUDY_CFG \
    --bands delta theta alpha beta gamma \
    --connectivity coherence plv wpli \
    --epoch-length 2.0 \
    --jobs {{N_JOBS}} --verbose \
    > $LOGS/study_analyze_$(date +%Y%m%d_%H%M).log 2>&1 &
```

**Status + collect group results:**
```bash
cd $PIPELINE
uv run source-localization study status $STUDY_CFG
uv run source-localization study collect $STUDY_CFG    # builds $DERIV/group/*.csv
```

**Validation (dipole simulations; optional, for preset comparison):**
```bash
cd $PIPELINE
nohup uv run source-localization validate \
    --test-dir $VALIDATION \
    --config $VALIDATION/configs/ --all \
    --test-mode combined --snr 10 --trials 25 \
    > $LOGS/validate_$(date +%Y%m%d_%H%M).log 2>&1 &
```

**QC pass:**
- Review `pipeline_report.html` per subject in `$DERIV/{subject}/pipeline/`
- For each failure, add row to `$EXCL` with concrete `reason`
- Record rationale in `$QC_NOTES/{subject}_qc.md`

### One-Time Instructions — Tasks that should only execute once

<!-- 👤 AUTHOR AREA: Add tasks. Move to Completed once done. -->

- [ ] `study init` for `$STUDY`
- [ ] Populate `$PARTICIPANTS` with groups/covariates
- [ ] Edit `$STUDY_CFG` for this study's bands + connectivity methods
- [ ] Initialize empty exclusion CSV
- [ ] Snapshot `$STUDY_CFG` + `$PARTICIPANTS` into `$PROJECT`
- [ ] Pilot single-subject run to validate preset + atlas choice
- [ ] First batch source-localization pass (`study run`)
- [ ] First batch spectral/connectivity pass (`study analyze`)
- [ ] First QC pass, populate exclusions

#### Completed (don't re-run)
<!-- Move checked items here with date -->

### Formatting Guidelines

- **QC notes** → `$QC_NOTES/{subject}_qc.md` with: date, pipeline report items reviewed, failures + reason, exclusion decision
- **Exclusion rows** — every row in `$EXCL` needs a concrete `reason` string; expand rationale in `$QC_NOTES`
- **Paths** — always shorthand from `## Paths`; never `../../`

---

## 📝 After Each Loop

- **Update activity log** (`$PLAN/main-plan-activity.md`, append 1–2 lines):
  - Phase, subjects processed, preset, outputs produced
  - Timestamp (UTC), `$PROJECT` git hash, `$PIPELINE` git hash
  - Exclusions added this loop (count + pointer to `$QC_NOTES/`)

- **Update plan log** (`$PLAN/main-plan-log.csv`):
  `timestamp,phase,subject_range,preset,atlas,n_processed,n_excluded,output_path,status,project_hash,pipeline_hash`

- **Commit `$PROJECT`** — plan edits, QC notes, log updates, config/participants snapshots only
  - Never commit anything from `$STUDY`
  - Commit message: `sl: {phase} {preset}/{atlas} — {outcome}`

- **Feedback to AUTHOR**:
  1. Phase progress, subjects remaining
  2. QC findings needing attention (bad channels, poor localization, excessive motion)
  3. Pipeline issues worth filing upstream in source-localization

---

## 📚 Skill Library — Community skills (optional)
<!-- Uncomment to use -->

---

## 📌 Study-specific conventions

### Preset / atlas selection
- `roi_based_ellipsoid` + `antwerp` is the recommended default (76.9% ROI accuracy, 1.67mm error)
- Use `shell_ellipsoid` for whole-brain parametric / depth-stratified mapping
- Use `ellipsoid_surface` if best spatial localization matters more than ROI statistics
- `allen` atlas (64 ROIs, anatomically constrained) is an alternative to `antwerp` (47 ROIs)

### Output structure (per preset)
`$DERIV/{subject}/` produces:
- `pipeline/step5_stc.pkl` — MNE SourceEstimate (required for vertex-level analyses downstream)
- `pipeline/step3_source_coords_mm.npy` — source coordinates (required for vertex-level)
- `roi_timeseries/step6_roi_timeseries_{magnitude,signed}.pkl` — ROI timeseries (required for source-analytics)
- `roi_timeseries/*.set/.fdt` — EEGLAB-compatible export
- `analysis/band_power.csv`, `analysis/connectivity_*.csv` — if `study analyze` has been run

### Exclusion system
- `$EXCL` is the only source of truth for downstream analysis (source-analytics reads it)
- Every exclusion carries a `reason` string
- Detailed rationale lives in `$QC_NOTES`

### Handoff to source-analytics
When localization + analysis are complete (all subjects, `$DERIV/group/*.csv` populated, exclusions populated), tag a `sl-v1` commit in `$PROJECT` and note it in the activity log. A downstream `source-analytics` project will point at `$DERIV` as its `discovery.root_dir`.
