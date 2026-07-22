# source-localization

EEG source reconstruction for rodent recordings — raw cleaned EEG in, per-subject
ROI timeseries and whole-brain source estimates out. First stage of a
three-package pipeline:

```
source-localization  ──►  source-analytics  ──►  source-lightbox
(reconstruct sources)     (stats + figures)       (render the gallery)
```

The full usage guide — installation, presets, the CLI, validation — lives in the
[README](https://github.com/alexedmon1/source-localization#readme). This site
holds the **methods and operational documentation**: how to extend the pipeline,
and the known limitations you should read before trusting a number.

## Guides

| Page | Covers |
|---|---|
| [Adding an atlas](guides/adding_an_atlas.md) | Registering a new atlas for the verification benchmark (a data task, not a code change) |
| [Electrode coordinate setup](guides/ELECTRODE_COORDINATE_SETUP.md) | Aligning an electrode array to a mouse brain atlas |

## Known issues

Read these before interpreting output from the affected pipeline.

| Page | Affects |
|---|---|
| [Shell ROI coverage at 215 vertices](known-issues/SHELL_ROI_COVERAGE.md) | `shell_ellipsoid` — thin lateral ROIs can be dropped by the sampling grid |
| [Regularization scaling](known-issues/REGULARIZATION_SCALING_ISSUE.md) | Inverse solution scaling for mouse-scale head models |

## Two rules that keep this site useful

1. **The `nav:` in `mkdocs.yml` is the curation.** A page not listed there is not
   part of the site. Adding a document means deciding where it belongs — which is
   what stops reference material from being buried under working notes.
2. **Known issues are durable; TODOs are not.** A document describing a *real
   limitation of the output* belongs in Known issues and stays until fixed. A
   document describing *work we intend to do* belongs in an issue tracker.

## Local preview

```bash
uv run --no-project --with "mkdocs-material>=9.5,<10" mkdocs serve   # http://127.0.0.1:8000
uv run --no-project --with "mkdocs-material>=9.5,<10" mkdocs build   # render to site/
```

Pushes to `main` publish automatically via GitHub Actions
(`.github/workflows/docs.yml`).
