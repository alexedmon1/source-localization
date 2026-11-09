# Field templates

A **field** is a fixed pattern over the electrodes that carries a response: unit norm, average-referenced, one value
per channel. The field method defines a field once from a group's data, freezes it in a **field template**, and then
uses the template unchanged: every recording is projected onto it (one virtual channel per field), and the field is
located with the location posteriors (`source_localization.locate`). This page covers the template file and how
fields are defined (`source_localization.field`). The projection and the location steps have their own pages as
they are built (plan: bayes-sl `reports/2026-11-09_field_method_library_plan.md`).

A field is the data's dominant pattern. It is not a generator: where it comes from is answered by locating it, as a
distribution.

## Defining a field

Each definition is catalogued with an id and a version ([model catalogue](CATALOGUE.md), "Field definitions"):

| id | for | what it computes |
|---|---|---|
| `field.response_svd` | evoked responses (chirp) | leading spatial singular vectors of the group-mean response over a time span, each unit first scaled to unit norm over the span |
| `field.phase_aligned` | steady-state responses (ASSR) | real pattern of the complex coefficients after rotating every recording onto the pattern's phase, iterated |
| `field.phase_free` | steady-state responses | leading eigenvector of the phase-free cross-product, no starting pattern |
| `field.csd_leading` | ongoing bands (rest) | leading eigenvector of the mean trace-normalised cross-spectral density |

```python
import numpy as np
from source_localization.field import unit_index, scale_units, response_svd, apply_sign_rule

keys, units = unit_index(list(zip(mouse, day)))                  # one unit per mouse-day
resp = np.array([trial_avg[j].mean(0) for j in units])           # (units, channels, times), average-referenced
span = (times >= 0.005) & (times <= 0.300)
Y, _ = scale_units(resp, span)
maps, s = response_svd(Y.mean(0), span, k=2)
chirp_D = apply_sign_rule(maps[:, 0], channels, {"electrodes": ["E3", "E28"], "sign": 1})
```

A field's sign is arbitrary until a **sign rule** fixes it: the sum over named electrodes is positive (or negative).
The rule is stored in the template; it is never a constant in code.

**Check (2026-11-09).** Re-defining surface-evoked's four frozen Sentinel fields (`sentinel_fields_v1.json`) from
their defining data with these functions reproduces all four exactly (cosine 1, largest element difference 1e-16;
bayes-sl `scripts/f2_check_redefine.py`).

## The template file

Schema `source_localization.field_template/1`, JSON. Required:

| key | content |
|---|---|
| `schema` | `source_localization.field_template/1` |
| `name`, `version` | the template's name and an integer version; a new definition is a new version |
| `frozen` | date written |
| `channels`, `electrodes_mm` | the array's channel order and positions; every map follows this order |
| `reference` | how the data must be referenced before projection (average reference) |
| `maps` | field name → one value per channel (unit norm, sums to 0) |
| `sign_rules` | field name → `{"electrodes": [...], "sign": 1}` |
| `definitions` | field name → `{"id": <catalogue id>, "version": <int>, ...parameters}` |

Recommended: `status`, `sign_conventions` (text), `measurement_conventions`, `defining_data` (study, groups, units,
counts), `provenance` (commits, script and input hashes), `caveats`, `cos_between_maps`.

```python
from source_localization.field import FieldTemplate, freeze, load

t = freeze(FieldTemplate(name="mystudy_fields", version=1, channels=channels, electrodes_mm=el, reference="average",
                         maps={"chirp_D": chirp_D},
                         sign_rules={"chirp_D": {"electrodes": ["E3", "E28"], "sign": 1}},
                         definitions={"chirp_D": {"id": "field.response_svd", "version": 1,
                                                  "span_s": [0.005, 0.3], "component": 0}},
                         extra={"defining_data": {...}, "caveats": [...]}),
           "templates/mystudy_fields_v1.json")
t.sha256          # identifies the template in every output that used it
t.provenance()    # name, version, sha256, maps, definition ids and versions
```

`freeze` validates (lengths, unit norm, average reference, sign rules, a definition and a sign rule for every map)
and **refuses to overwrite** an existing file. `load` reads and validates a template and computes its sha256.

**Templates written before the schema** (surface-evoked `sentinel_fields_v1.json`, 2026-10-14) load unchanged. Their
sign rules are read from the `sign_conventions` text ("… (E3 + E28) positive"); `definitions` is empty; their
provenance reports `schema: null`. Nothing is written back to them.

## Projecting recordings (the field mode)

Each recording is reduced to one **virtual channel** per field. The epochs are average-referenced over the
template's channels and picked by name in the template's order (the file's channel order does not matter; a missing
channel raises). Two catalogued projections:

| id | what it computes |
|---|---|
| `field.project_inner` (default) | each field separately: s_k(t) = u_k · x(t). The field method's readout since surface-evoked log 70 |
| `field.project_joint` | all selected fields at once by least squares: s(t) = (M Mᵀ)⁻¹ M x(t). Equals the inner projection for orthogonal fields |

Overlapping fields share signal: with the inner projection each series also carries what its field shares with the
others (Sentinel's `chirp_D` and `assr40_D` are not orthogonal).

The **residual** is what the selected fields do not capture, x − P x with P the orthogonal projector onto their span.
It is stored as its **trial average** (decided 2026-11-09), so what the fields miss can still be located or tested.

**Command line:**

```bash
source-localization field show templates/sentinel_fields_v1.json
source-localization field project --template templates/sentinel_fields_v1.json \
    --roster recordings.csv --out localization/sentinel_fields_v1      # roster columns: id, path
source-localization field project --template T.json --eeg a.set b.set --out OUT --maps chirp_D chirp_U2
```

**In a pipeline config** (`pipeline.mode: field`; no head model sections needed):

```yaml
pipeline: {name: sentinel_fields, mode: field}
field:
  template: templates/sentinel_fields_v1.json
  template_sha256: b141891da92765a4dbe351e886302188308b8a6560d67b1fa6b2db3a6c0d98a9   # optional guard
  maps: [chirp_D, chirp_U2]          # optional; default every map
  method: inner                      # or joint
```

**Output** (per recording, the layout source-analytics reads for ROI series):

| file | content |
|---|---|
| `data/step6_roi_timeseries_signed.pkl` | {field: 1-D series, epochs concatenated} |
| `data/roi_timeseries_signed.set` | MAT file with `srate` |
| `data/field_residual_average.npz` | residual trial average (channels × times), channel names, times, which fields it excludes |
| `data/field_run.json` | `readout_mode: field`; the template's name, version, sha256 and definition ids; the projection's id and version; maps; reference; source file and its sha256; epochs, samples, rate, tmin; package version |

**Check (2026-11-09).** For all 192 recordings of surface-evoked's held-out roster (Sentinel D1pre, groups 1-6;
chirp, 40 Hz and 80 Hz ASSR), the library's series equal the study's (`rerun070_fields.py`) to a relative 6e-16
(bayes-sl `scripts/f3_check_projection.py`).

## What a template does not carry over

- **Another array or channel order** needs its own template: the maps follow the channel order of the defining data.
- **The calibrations of the location posteriors** hold only on the array, ensemble and grid they were measured with
  (see the catalogue's validity entries).
- **A field defined on two groups pooled** under-weights a field present in only one of them.
