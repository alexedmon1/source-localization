"""The field method's readout side: field templates, defining fields from group data, projecting onto them.

- :mod:`.template`: the frozen field-template file (schema ``source_localization.field_template/1``).
- :mod:`.define`: field definitions from group data (catalogued ``field.*``).
- :mod:`.project`: the field mode's project step: virtual channels, residual, run record (``field.project_*``).
- :mod:`.cli`: ``source-localization field show|project``.

Locating a field (where a source set would best reproduce it) is :mod:`source_localization.locate`'s. No group
statistics live here. Plan: bayes-sl ``reports/2026-11-09_field_method_library_plan.md``.
"""
from .define import (apply_sign_rule, csd_leading_pattern, phase_aligned_pattern, phase_free_pattern,  # noqa: F401
                     response_svd, scale_units, unit_index)
from .template import SCHEMA, FieldTemplate, TemplateError, freeze, load  # noqa: F401
from .project import PROJECTIONS, pick_channels, project, project_file, write_run  # noqa: F401,E402
