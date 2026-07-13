"""
Module entry point for running validation as a standalone module.

Usage:
    python -m source_localization.validation --test original
    python -m source_localization.validation --test dipole_size --quick
    python -m source_localization.validation --list
"""

import sys
from .cli import main

if __name__ == '__main__':
    sys.exit(main())
