"""``source-localization field``: field templates and field projection from the command line.

    source-localization field show TEMPLATE
    source-localization field project --template T --eeg a.set b.set ... --out ROOT [--maps A B] [--method inner|joint]
    source-localization field project --template T --roster recordings.csv --out ROOT      # columns: id, path

Each recording is written to ROOT/<id>/data (``id`` from the roster, or the file name without extension), in the
layout of :mod:`source_localization.field.project`. Recordings whose field_run.json exists are skipped unless
``--overwrite``.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path


def create_parser(subparsers):
    parser = subparsers.add_parser('field', help='Field templates and field projection (the field method)',
                                   description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='field_command', title='field commands')
    show = sub.add_parser('show', help='Validate a field template and print what it holds')
    show.add_argument('template')
    proj = sub.add_parser('project', help='Project epoched recordings onto a field template')
    proj.add_argument('--template', required=True)
    proj.add_argument('--template-sha256', help='Refuse the template if its content hash differs')
    src = proj.add_mutually_exclusive_group(required=True)
    src.add_argument('--eeg', nargs='+', help='Epoched EEGLAB .set files')
    src.add_argument('--roster', help='CSV with columns id, path')
    proj.add_argument('--out', required=True, help='Output root; one directory per recording')
    proj.add_argument('--maps', nargs='+', help='Fields to project (default: all in the template)')
    proj.add_argument('--residual-of', nargs='+', help='Fields the residual excludes (default: the projected ones)')
    proj.add_argument('--method', choices=['inner', 'joint'], default='inner',
                      help='inner: each field separately (field.project_inner); joint: least squares together')
    proj.add_argument('--overwrite', action='store_true')
    return parser


def run(args) -> int:
    from .project import project_file
    from .template import load
    if args.field_command == 'show':
        t = load(args.template)
        print(json.dumps({**t.provenance(), 'channels': len(t.channels), 'sign_rules': t.sign_rules,
                          'legacy (pre-schema)': t.legacy}, indent=1))
        return 0
    if args.field_command != 'project':
        print('usage: source-localization field {show,project} ...')
        return 1
    t = load(args.template)
    if args.template_sha256 and args.template_sha256 != t.sha256:
        raise SystemExit(f'{args.template}: sha256 {t.sha256} differs from --template-sha256 {args.template_sha256}')
    if args.roster:
        items = [(r['id'], r['path']) for r in csv.DictReader(open(args.roster))]
    else:
        items = [(Path(p).stem, p) for p in args.eeg]
    ids = [i for i, _ in items]
    if len(set(ids)) != len(ids):
        raise SystemExit('recording ids repeat; give a roster with unique ids')
    done = skipped = 0
    for rid, path in items:
        out = Path(args.out) / rid
        if (out / 'data' / 'field_run.json').exists() and not args.overwrite:
            skipped += 1
            continue
        project_file(path, t, out, maps=args.maps, method=args.method, residual_of=args.residual_of)
        done += 1
    print(f'{done} recordings projected onto {t.name} v{t.version} ({t.sha256[:12]}); {skipped} already present')
    return 0
