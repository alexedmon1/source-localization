"""Field templates: frozen electrode fields, with everything needed to use them correctly.

A field template is a JSON file holding one or more fields (unit-norm, average-referenced electrode patterns) for one
electrode array, together with how they were defined and how they are to be measured. It is frozen: projections
and locations always use the file unchanged, and a new definition is a new version, never an overwrite. Its sha256
identifies it in every output that used it.

**Schema ``source_localization.field_template/1``.** Required keys:

- ``name``, ``version`` (int), ``frozen`` (date), ``channels`` (array channel order), ``electrodes_mm``
  (positions, same order), ``reference`` (text; fields are average-referenced), ``maps`` (name -> list of floats,
  one per channel);
- ``sign_rules``: map -> ``{"electrodes": [...], "sign": +1}``, the rule that fixed the field's sign;
- ``definitions``: map -> ``{"id": <catalogue id, e.g. field.response_svd>, "version": int, ...parameters}``.

Recommended (free text or structured): ``status``, ``sign_conventions``, ``measurement_conventions``,
``defining_data``, ``provenance``, ``caveats``, ``cos_between_maps``.

**Templates written before the schema** (surface-evoked's ``sentinel_fields_v1.json``, 2026-10-14) have no ``schema``,
``sign_rules`` or ``definitions`` keys. They load unchanged: the sign rules are read from their
``sign_conventions`` text ("... (E3 + E28) positive"), and ``definitions`` stays empty. Nothing is written back.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import re
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

__all__ = ["SCHEMA", "FieldTemplate", "load", "freeze", "TemplateError"]

SCHEMA = "source_localization.field_template/1"
_REQUIRED = ("name", "version", "channels", "electrodes_mm", "reference", "maps")
_SIGN_TEXT = re.compile(r"\(?\s*(E\d+(?:\s*\+\s*E\d+)*)\s*\)?\s*positive")


class TemplateError(ValueError):
    """A template that fails validation."""


@dataclass
class FieldTemplate:
    name: str
    version: int
    channels: List[str]
    electrodes_mm: np.ndarray
    reference: str
    maps: Dict[str, np.ndarray]
    sign_rules: Dict[str, dict] = dc_field(default_factory=dict)
    definitions: Dict[str, dict] = dc_field(default_factory=dict)
    extra: Dict[str, object] = dc_field(default_factory=dict)
    sha256: Optional[str] = None
    path: Optional[Path] = None
    legacy: bool = False

    @property
    def matrix(self) -> np.ndarray:
        """The fields as rows, (n_maps, n_channels), in ``maps`` order."""
        return np.stack([self.maps[k] for k in self.maps])

    def validate(self, tol: float = 1e-6) -> None:
        """Raise :class:`TemplateError` if a map has the wrong length, is not unit norm, is not average-referenced,
        or breaks its sign rule; or if a sign rule names an unknown channel."""
        n = len(self.channels)
        if len(set(self.channels)) != n:
            raise TemplateError("duplicate channel names")
        if np.shape(self.electrodes_mm) != (n, 3):
            raise TemplateError(f"electrodes_mm must be ({n}, 3)")
        for k, v in self.maps.items():
            if v.shape != (n,):
                raise TemplateError(f"map {k}: {v.shape[0]} values for {n} channels")
            if abs(np.linalg.norm(v) - 1) > tol:
                raise TemplateError(f"map {k}: norm {np.linalg.norm(v):.6g}, not 1")
            if abs(v.sum()) > 1e-6 * np.sqrt(n):
                raise TemplateError(f"map {k}: sums to {v.sum():.3g}, not average-referenced")
        for k, rule in self.sign_rules.items():
            if k not in self.maps:
                raise TemplateError(f"sign rule for unknown map {k}")
            missing = set(rule["electrodes"]) - set(self.channels)
            if missing:
                raise TemplateError(f"sign rule of {k} names unknown channels {sorted(missing)}")
            s = sum(self.maps[k][self.channels.index(c)] for c in rule["electrodes"])
            if np.sign(s) != np.sign(rule.get("sign", 1)):
                raise TemplateError(f"map {k} breaks its sign rule {rule}")
        for k, d in self.definitions.items():
            if k not in self.maps or "id" not in d:
                raise TemplateError(f"definition of {k} must name a map and a catalogue id")

    def provenance(self) -> dict:
        """What an output records about the template it used."""
        return {"name": self.name, "version": self.version, "sha256": self.sha256, "maps": list(self.maps),
                "definitions": {k: {"id": d.get("id"), "version": d.get("version")} for k, d in
                                self.definitions.items()},
                "schema": None if self.legacy else SCHEMA}

    def to_json(self) -> dict:
        doc = {"schema": SCHEMA, "name": self.name, "version": int(self.version), **{k: v for k, v in self.extra.items()
                                                                                       if k not in ("schema",)}}
        doc.update({"channels": list(self.channels), "electrodes_mm": np.asarray(self.electrodes_mm).tolist(),
                    "reference": self.reference, "maps": {k: np.asarray(v, float).tolist() for k, v in
                                                          self.maps.items()},
                    "sign_rules": self.sign_rules, "definitions": self.definitions})
        doc.setdefault("frozen", datetime.date.today().isoformat())
        return doc


def _legacy_sign_rules(doc: dict) -> Dict[str, dict]:
    out = {}
    for k, text in (doc.get("sign_conventions") or {}).items():
        m = _SIGN_TEXT.search(text or "")
        if m:
            out[k] = {"electrodes": [e.strip() for e in m.group(1).split("+")], "sign": 1}
    return out


def load(path, validate: bool = True) -> FieldTemplate:
    """Read a template (schema or pre-schema), compute its sha256, and validate it."""
    path = Path(path)
    raw = path.read_bytes()
    doc = json.loads(raw)
    missing = [k for k in _REQUIRED if k not in doc]
    if missing:
        raise TemplateError(f"{path.name}: missing {missing}")
    legacy = "schema" not in doc
    if not legacy and doc["schema"] != SCHEMA:
        raise TemplateError(f"{path.name}: schema {doc['schema']!r}, this package reads {SCHEMA!r}")
    known = set(_REQUIRED) | {"schema", "sign_rules", "definitions"}
    t = FieldTemplate(name=doc["name"], version=int(doc["version"]), channels=list(doc["channels"]),
                      electrodes_mm=np.asarray(doc["electrodes_mm"], float), reference=doc["reference"],
                      maps={k: np.asarray(v, float) for k, v in doc["maps"].items()},
                      sign_rules=_legacy_sign_rules(doc) if legacy else dict(doc.get("sign_rules") or {}),
                      definitions={} if legacy else dict(doc.get("definitions") or {}),
                      extra={k: v for k, v in doc.items() if k not in known},
                      sha256=hashlib.sha256(raw).hexdigest(), path=path, legacy=legacy)
    if validate:
        t.validate()
    return t


def freeze(template: FieldTemplate, path) -> FieldTemplate:
    """Validate and write ``template`` to ``path`` under the current schema. Refuses to overwrite: a new definition
    is a new version with a new file name. Returns the template as read back (with its sha256)."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; a frozen template is never overwritten. Write a new version instead.")
    if not template.definitions or set(template.definitions) != set(template.maps):
        raise TemplateError("freeze needs a definition (catalogue id and parameters) for every map")
    if set(template.sign_rules) != set(template.maps):
        raise TemplateError("freeze needs a sign rule for every map")
    template.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(template.to_json(), indent=1))
    return load(path)
