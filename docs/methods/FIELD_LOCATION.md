# Locating a field

Where does an electrode field come from? The field method answers with calibrated location posteriors over an
ensemble of head models: one source anywhere on the grid, or a mirror-image pair of sources, compared by their
evidence. This page covers the location step (`source_localization.locate.field_model`); the field itself is defined
and frozen as in [Field templates](FIELD_TEMPLATES.md). Every model named here is in the
[model catalogue](CATALOGUE.md) with an id and a version, and every result records them.

The answer is a distribution: where a source set would best reproduce the field, given the head-model uncertainty.
It does not show that the field is a generator, and it cannot separate structures closer together than its maps
resolve (for Sentinel's chirp field, the tectum from the thalamus is not separable).

## The location space

A `LocationSpace` holds what the posteriors need about one study's head: the geometry from a pipeline run (ellipsoid
BEM, registered electrodes), a catalogued head-model ensemble (`prior.head_ensemble_pa_phase2`: the nominal model plus
32 seeded draws of registration shift and skull conductivity), the ensemble's leadfields on the 0.5 mm grid, region
labels from the atlas, and the average-reference basis.

```python
from source_localization.locate.space import LocationSpace
sp = LocationSpace("localization/sentinel_40hz_cartesian_mc48/<recording>", cache_dir="scratch/location_cache")
sp.provenance()     # ensemble id, version, prior values and seed; geometry signature; labelling
```

## Calibrations hold only where they were measured

The raw ensemble maps are overconfident; a **map calibration** (temper, then blur) makes their credible regions
honest. A calibration was measured on one array, one ensemble, one grid and one kind of positions, and holds only
there. Each catalogued calibration carries that domain:

| calibration | array | ensemble | grid | positions |
|---|---|---|---|---|
| `calibration.volumetric` (T 1.5, blur 0.25 mm) | `mea30_sentinel` | `prior.head_ensemble_pa_phase2` | 0.5 mm | volumetric |
| `calibration.surface` (T 2.0, no blur) | `mea30_sentinel` | `prior.head_ensemble_pa_phase2` | 0.5 mm | surface |

The array is identified by `data/calibration/mea30_sentinel.json`: its channel names in order, and every electrode
within 0.1 mm of its registered position. **The field locator refuses a space outside the domain** (decided
2026-11-09): another array, ensemble, spacing or position set raises `CalibrationDomainError` and has to be calibrated
first. An uncalibrated map (`calibration.none`) is given only when asked for explicitly
(`allow_uncalibrated=True`), for diagnostics.

## The locator

```python
from source_localization.locate.field_model import FieldLocator, summarize

loc = FieldLocator(sp)                       # calibration.volumetric, whiten.oas, pair rule 0.5; checks the domain
loc.fit(noise_blocks)                        # one (channels, k) noise block per unit: whitener + both posteriors
yw = loc.whiten(column, noise=column_noise)  # W y / the column's own noise scale
loc.locate(yw)                               # ([right, left], P(pair)) or ([peak], P(pair))  -- locate.field_choice
summarize(loc, yw, labels, names)            # SNR; single and pair peaks, 90% region volumes, region masses, log BF
loc.provenance()                             # catalogue ids and versions, calibration, space
```

- **Whitening** (`whiten.oas` by default): average reference, noise shape pooled over units by OAS shrinkage, each
  unit's own scale.
- **Posteriors:** `locate.single_source` and `locate.mirror_pair`, on the whitened ensemble.
- **Choice** (`locate.field_choice`): the mirror pair at the pair map's peak if P(pair) ≥ 0.5, otherwise the
  single-source peak.

`report_response_svd_field` gives the full report for a `field.response_svd` field from its defining data: the grand
mean, each unit, and a bootstrap over subjects that re-derives the field per replicate.

**Check (2026-11-09).** For Sentinel's frozen chirp fields (`chirp_D`, `chirp_U2`), the library reproduces
surface-evoked's `localize_fields_bayes.json` with no difference at its stored precision: the grand mean, all 40
mouse-days, and the 200-replicate bootstrap including its random draws (bayes-sl
`scripts/f4_check_localize_fields.py`).

## The field model over a sample of subjects

`FieldModel` (`locate.field_model_map`) answers "where, and how stable is that over resamples of the subjects?". Records
are one per recording, of one of three kinds:

| kind | record | the sample's field |
|---|---|---|
| `evoked` | `xbar` (channels × times), `x_flip` (k × channels × times) | first time course of the mean unit-normalised response |
| `steady` | `z` (channels, complex), `z_flip` (k × channels) | `field.phase_free` |
| `ongoing` | `csd` (channels × channels), `csd_q` (k × channels × channels) | `field.csd_leading` |

```python
from source_localization.locate.field_model import FieldModel, Units, inner_draws

fm = FieldModel(sp, "ongoing")                       # checks the calibration domain
units = Units(record_subject)                        # record -> subject; weights per subject
loc = fm.setup(recs, units)                          # whitener on one noise block per subject, through the sample's field
mass, share_pair = fm.map(recs, units, inner_draws(np.ones(units.U), 100, rng), loc)
```

Each draw resamples the subjects, re-derives the field, forms the weighted grand-mean column and noise, and locates it
(`locate.field_choice`); the map is the located mass per grid point, and `share_pair` the share of draws called a pair.

## The model filter of a located pair

`model_filter_pair(space, right_mm, target)` (`field.model_filter_pair`) turns a located mirror pair into a spatial
filter: under each head model one mirrored moment is fitted to the target field, the patterns are averaged and
normalised. It can be projected like a template field.

**Checks (2026-11-09; bayes-sl `scripts/f4b_check_field_model.py`).**
- The model filter rebuilds surface-evoked's `templates/h038_model_filter.json` exactly (members at
  [±0.56, −2.73, 0.80] mm, cos 0.990 with `chirp_D`, per-model fits 0.955-0.986).
- On three benchmark cohorts (Thalamus 1.0 SD, cortex 0.5 SD, noise only) at every group size (8-30 per group), the
  field model's maps equal bayes-sl L3's stored maps with no difference (located mass, share called pair, the pooled
  field), with L2's seeds.

## Calibrating a new space

A space outside every catalogued calibration's domain (another array, ensemble, grid spacing or position set) is
refused until it has its own calibration. `locate.calibrate.calibrate` measures one (`calibration.procedure`):

```python
from source_localization.locate.calibrate import calibrate

cal, report = calibrate(sp, whitener, noise_calib, noise_valid,      # whitened held-out noise columns, two halves
                        fresh_models_calib, fresh_models_valid,      # from the ensemble's prior, not in the ensemble
                        snrs=[s / 2, s, 2 * s], n_per_snr=400, out_dir="calibrations/mystudy",
                        depth_of=lambda i: "deep" if is_deep[i] else "cortex", name="calibration.mystudy")
loc = FieldLocator(sp, calibration=cal)      # accepted: its domain is this space
```

1. Simulate single-source truths in the space through fresh head models, at several SNRs, with real held-out noise;
   two independent sets (calibration, validation).
2. Score each raw map on a grid of temperature × blur by the posterior mass above the truth.
3. Choose, on the calibration set, the setting with the smallest worst |coverage − level| over the levels 0.5, 0.8, 0.9,
   0.95 and over SNR × depth class.
4. Report its coverage on the validation set; write the array reference and `<name>.json` (setting, domain, coverage,
   losses).

The result is a `MapCalibration` whose domain is this space; the locator accepts it there and nowhere else.
Calibration covers single-source truths; mirror-pair coverage should be checked separately (as bayes-sl
`TB_mirror_pair.md` did for MEA30).

**Check (2026-11-09).** Applied to bayes-sl tb11's stored scores (7,200 maps per setting), the selection chooses the
version 1 settings again: volumetric T 1.5, blur 0.25 mm (90% coverage 88-89% on validation), surface T 2.5, no blur
(84-91%) (bayes-sl `scripts/f4c_check_calibration.py`).

**Version 2 (2026-11-14).** Version 1 was measured in the inner-first BEM. `scripts/recalibrate_maps.py` repeats tb11's
design on this package's geometry, with the corrected solve. Run in the old order, it reproduces tb11's stored scores.
Under the corrected solve it chooses volumetric T 1.5, blur 0.25 mm again (90% coverage 87-88% on validation) and
surface T 2.0, no blur (85-91%) (validation-tests `map_calibration_bemfix/REPORT.md`).

## Each located region's field, and each subject's value along it

After the field model has located a sample's field (the located mass per position and the pooled field u), each
region with located mass gets its own electrode field (`field.region_field`): u projected on the span of each located
position's leadfield (mean over head models, average-referenced), weighted by located mass within the region and
normalised. Each subject's band power along it is `field.region_power_db`, 10 log10(g′ C g) in dB, with C the
subject's average-referenced cross-spectral matrix. The table (one row per region and subject, with the region's
located mass as `p_source`) is what source-analytics' Bayesian engine reads with no ROI, one outcome per region.

```python
from source_localization.locate.field_model import region_fields, region_value_table, write_region_values
fields = region_fields(sp, u, located_idx, located_mass, region_of_position)
write_region_values("values.csv", region_value_table(fields, csd_by_subject), {"space": sp.provenance()})
```

```bash
source-localization field region-values --run-dir RUN --map map.npz --csd csd.npz --out values.csv
#   map.npz: idx, mass, u     csd.npz: subject, csd (subjects x channels x channels)
```

`values.csv.json` records the catalogue ids and versions and the location space.

**Check (2026-11-09; bayes-sl `scripts/f5_check_region_values.py`).** From L3's stored maps, the library reproduces
bayes-sl L3's per-subject tables: the FORGE real recordings in full (`y_forge_real.csv`, 700 rows; also through the
command line) and three benchmark cohorts at every group size (`y_v1.csv`, 3,664 rows), to 3e-14 dB.
