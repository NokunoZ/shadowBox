"""Thin wrapper over trainDataSet/classType.py -- the ONLY place that scans folders.

Classes are declared by folders, so the scan has to happen somewhere; happening in
exactly one place is what stops training and inference disagreeing about what index 3
means.

Inference does not call this. It reads the class list out of the checkpoint, because
adding a folder renumbers every class after it. See modelholder/noted.md.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from .paths import ROOT

_CLASSTYPE = ROOT / "trainDataSet" / "classType.py"
_MODULE_NAME = "classType"


def _load_classtype():
    """Import classType.py by path, as the module name it would have anyway.

    trainDataSet/ is not a package and `code/` cannot become one -- it would shadow
    the standard library's `code` module -- so a normal import is not available.

    The name matters: dataSpliter.py does a plain `from classType import ...`, and
    registering this under the same name means both routes reach ONE module object.
    Loaded twice under two names there would be two ClassScanError classes, and an
    `except` for one would sail straight past the other.
    """
    if _MODULE_NAME in sys.modules:
        return sys.modules[_MODULE_NAME]
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _CLASSTYPE)
    if spec is None or spec.loader is None:            # pragma: no cover - unreachable
        raise ImportError(f"cannot import {_CLASSTYPE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


_classtype = _load_classtype()

ClassScanError = _classtype.ClassScanError
clip_paths = _classtype.clip_paths
scan = _classtype.scan
class_map = _classtype.class_map
class_counts = _classtype.class_counts
require_classes = _classtype.require_classes


def classes_in(split_dir: Path) -> list[str]:
    """Class names present in a generated split folder (trainData/ or testData/).

    Sorted the same way as the dataset scan, since the indices have to line up.
    """
    return sorted(d.name for d in Path(split_dir).iterdir() if d.is_dir())


def clips_by_class(split_dir: Path) -> dict[str, list[Path]]:
    """class name -> its clips, for one split folder."""
    split_dir = Path(split_dir)
    return {name: clip_paths(split_dir / name) for name in classes_in(split_dir)}
