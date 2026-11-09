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
| `calibration.surface` (T 2.5, no blur) | `mea30_sentinel` | `prior.head_ensemble_pa_phase2` | 0.5 mm | surface |

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
