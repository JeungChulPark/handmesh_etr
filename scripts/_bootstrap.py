"""Restore the import paths these scripts had when they lived at the repo root.

They import repo-root modules (``from utils import ...``, ``from models.mobrecon_ds
import ...``) and each other (``from infer_rgbd_captures import load_capture``).
Running ``python scripts/infer/foo.py`` only puts ``scripts/infer`` on sys.path,
so put the repo root and every ``scripts/<group>`` directory back.

Import it first in every entry-point script::

    import _bootstrap  # noqa: F401
"""
import os
import sys

# realpath, not abspath: each scripts/<group>/_bootstrap.py is a symlink to this
# file, and abspath would leave _HERE pointing at the group directory.
_HERE = os.path.dirname(os.path.realpath(__file__))
_ROOT = os.path.dirname(_HERE)

_paths = [_ROOT]
_paths += sorted(
    os.path.join(_HERE, d)
    for d in os.listdir(_HERE)
    if os.path.isdir(os.path.join(_HERE, d)) and not d.startswith(("_", "."))
)
for _p in _paths:
    if _p not in sys.path:
        sys.path.insert(0, _p)
