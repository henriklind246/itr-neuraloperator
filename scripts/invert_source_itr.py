"""Compatibility shim for the renamed inverse harness.

Use ``scripts/invert.py`` for new commands. This path remains so existing
source_itr inversion commands and Slurm scripts continue to work.
"""

from scripts.invert import *  # noqa: F401,F403
from scripts.invert import main


if __name__ == "__main__":
    raise SystemExit(main())
