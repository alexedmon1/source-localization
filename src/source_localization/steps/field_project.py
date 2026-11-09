"""Field mode, step ``project``: one recording onto a frozen field template (:mod:`source_localization.field.project`).

Config (``pipeline.mode: field``)::

    field:
      template: path/to/<name>_v<version>.json   # a frozen field template
      template_sha256: <hex>                      # optional: refuse a template whose content differs
      maps: [chirp_D, chirp_U2]                   # optional: default every map in the template
      method: inner                               # inner (field.project_inner, default) or joint
      residual_of: [chirp_D, chirp_U2]            # optional: default the projected maps
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from ..field.project import project_file
from ..field.template import load


def run(config, previous_outputs: Dict[str, Any]) -> Dict[str, Any]:
    f = config['field']
    template = load(f['template'])
    want = f.get('template_sha256')
    if want and want != template.sha256:
        raise ValueError(f"field template {f['template']} has sha256 {template.sha256}, config expects {want}")
    eeg = config['inputs']['eeg_file']
    out = Path(config['outputs']['dir'])
    print(f"    Template: {template.name} v{template.version} ({template.sha256[:12]}), maps "
          f"{f.get('maps') or list(template.maps)}, projection {f.get('method', 'inner')}")
    data = project_file(eeg, template, out, maps=f.get('maps'), method=f.get('method', 'inner'),
                        residual_of=f.get('residual_of'))
    print(f"    ✓ Virtual channels and residual written to {data}")
    return {'field_run': data / 'field_run.json', 'field_template': template}
