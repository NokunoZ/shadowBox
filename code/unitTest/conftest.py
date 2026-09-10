"""Shared fixtures.

Tests must not need a GPU and must not download weights, so nothing here imports
ultralytics or touches modelholder/ for real. Anything that writes runs against
tmp_path -- a test that promotes or overwrites for real is a test that can destroy
the working model.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_by_path(name: str, path: Path):
    """Import a module that is not importable normally (trainDataSet/ is not a package)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def spliter():
    return load_by_path("test_dataspliter_module", ROOT / "trainDataSet" / "dataSpliter.py")


@pytest.fixture
def make_dataset(tmp_path):
    """Build a fake dataSet/ tree: {"jab": 4, "cross": 3} -> folders with that many clips.

    The clips are empty files. Nothing under test decodes video -- class discovery and
    splitting only ever look at names and suffixes -- so real footage would buy nothing
    and would have to fight .gitignore to exist.
    """
    def build(counts: dict[str, int], root: Path | None = None, suffix: str = ".mp4") -> Path:
        root = root or (tmp_path / "dataSet")
        root.mkdir(parents=True, exist_ok=True)
        for name, n in counts.items():
            folder = root / name
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                (folder / f"{name}_{i:03d}{suffix}").write_bytes(b"")
        return root
    return build


@pytest.fixture
def skeleton():
    """A synthetic COCO-17 pose, (F, 17, 3), with a torso of a known size.

    Hips at y=100, shoulders at y=60 -> torso length 40, hip midpoint at (50, 100).
    """
    def build(frames: int = 8, offset: tuple[float, float] = (0.0, 0.0), scale: float = 1.0):
        person = np.zeros((17, 3), dtype=np.float32)
        person[:, 2] = 1.0
        person[5] = (40.0, 60.0, 1.0)     # left shoulder
        person[6] = (60.0, 60.0, 1.0)     # right shoulder
        person[11] = (40.0, 100.0, 1.0)   # left hip
        person[12] = (60.0, 100.0, 1.0)   # right hip
        person[9] = (30.0, 80.0, 1.0)     # left wrist
        person[10] = (70.0, 80.0, 1.0)    # right wrist

        clip = np.repeat(person[None, ...], frames, axis=0)
        # a moving wrist, so a windowed clip is not a constant and resampling is visible.
        # Added BEFORE the shift and rescale, or the "same pose, moved and resized" clip
        # would carry a punch of a different size and the two would rightly not match.
        clip[:, 10, 0] += np.linspace(0.0, 20.0, frames, dtype=np.float32)
        clip[:, :, :2] = clip[:, :, :2] * scale + np.asarray(offset, dtype=np.float32)
        return clip
    return build
