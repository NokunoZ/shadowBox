"""The one place that knows modelVersion/ and runningVersion/.

Contract (modelholder/noted.md):

* training writes a NEW numbered file into modelVersion/ and never overwrites one
* running code loads runningVersion/classifier.pt and looks nowhere else
* promotion between the two is a deliberate copy, never automatic

A checkpoint carries everything needed to use it -- weights, class list, preprocessing
settings, architecture, and its own version tag -- because the promoted copy is
renamed to classifier.pt and would otherwise be anonymous.
"""

from __future__ import annotations

import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from . import features as F
from . import model as M
from .paths import MODEL_VERSION_DIR, POSE_WEIGHTS, RUNNING_CLASSIFIER, RUNNING_VERSION_DIR

CHECKPOINT_FORMAT = 1
ARCHIVE_PATTERN = re.compile(r"^classifier_v(\d+)\.pt$")


class CheckpointError(RuntimeError):
    """Raised rather than guessing when a checkpoint does not fit its caller."""


def archive_name(version: int) -> str:
    return f"classifier_v{version}.pt"


def archived_versions(archive_dir: Path = MODEL_VERSION_DIR) -> dict[int, Path]:
    archive_dir = Path(archive_dir)
    if not archive_dir.is_dir():
        return {}
    found = {}
    for path in archive_dir.iterdir():
        match = ARCHIVE_PATTERN.match(path.name)
        if match:
            found[int(match.group(1))] = path
    return found


def next_version(archive_dir: Path = MODEL_VERSION_DIR) -> int:
    """max(existing) + 1, discovered by scanning -- never a number kept in a file.

    Scanning means nothing has to be bumped by hand, and an interior gap is never
    refilled: delete v2 of v1..v3 and the next run is still v4.

    The one case where a number does come round again is deleting the HIGHEST version
    -- max+1 cannot know it ever existed. That is the argument for not deleting from
    the archive at all: two different models both called classifier_v3.pt make every
    note and every confident.csv row that names v3 ambiguous.
    """
    versions = archived_versions(archive_dir)
    return max(versions) + 1 if versions else 1


def preprocessing_settings(window: int) -> dict[str, Any]:
    return {
        "window": window,
        "normalisation": F.NORMALISATION,
        "keypoint_format": F.KEYPOINT_FORMAT,
        "keypoint_count": F.NUM_KEYPOINTS,
        "num_features": F.NUM_FEATURES,
        "pose_weights": POSE_WEIGHTS.name,
    }


def save(
    net: M.TemporalCNN,
    classes: list[str],
    window: int,
    archive_dir: Path = MODEL_VERSION_DIR,
    metrics: dict[str, Any] | None = None,
    class_counts: dict[str, int] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write the next version into the archive. Refuses to overwrite anything."""
    archive_dir = Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)

    version = next_version(archive_dir)
    target = archive_dir / archive_name(version)
    if target.exists():                      # a race, or a stale file appearing mid-run
        raise CheckpointError(
            f"{target} already exists -- refusing to overwrite a trained model. "
            "Every file in modelVersion/ is immutable once written."
        )

    payload = {
        "format": CHECKPOINT_FORMAT,
        "version": version,
        "archive_name": target.name,
        "classes": list(classes),
        "arch": dict(net.config),
        "preprocessing": preprocessing_settings(window),
        "state_dict": net.state_dict(),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metrics": metrics or {},
        "class_counts": class_counts or {},
    }
    if extra:
        payload.update(extra)

    torch.save(payload, target)
    return target


def _validate(payload: dict[str, Any], source: Path) -> dict[str, Any]:
    missing = [k for k in ("classes", "arch", "preprocessing", "state_dict") if k not in payload]
    if missing:
        raise CheckpointError(
            f"{source} is missing {', '.join(missing)}. A bare state_dict is not enough "
            "to run the model -- see modelholder/noted.md."
        )
    if not payload["classes"]:
        raise CheckpointError(f"{source} stores an empty class list")
    return payload


def load_file(path: Path, device: torch.device | str = "cpu") -> tuple[M.TemporalCNN, dict[str, Any]]:
    """Load a checkpoint by explicit path and rebuild its model."""
    path = Path(path)
    if not path.exists():
        raise CheckpointError(f"no checkpoint at {path}")
    payload = _validate(torch.load(path, map_location=device, weights_only=True), path)

    # width comes from the stored list, never from a fresh scan of dataSet/
    net = M.build(len(payload["classes"]), **payload["arch"])
    net.load_state_dict(payload["state_dict"])
    net.to(device).eval()
    return net, payload


def running_classifier(running_dir: Path = RUNNING_VERSION_DIR) -> Path:
    """The one classifier in runningVersion/, or a clear error saying what to do."""
    running_dir = Path(running_dir)
    candidates = sorted(p for p in running_dir.glob("*.pt") if p.name != POSE_WEIGHTS.name)
    if not candidates:
        raise CheckpointError(
            f"no classifier in {running_dir}.\n"
            "Promote one:  cp modelholder/modelVersion/classifier_v1.pt "
            f"{RUNNING_CLASSIFIER}"
        )
    if len(candidates) > 1:
        raise CheckpointError(
            f"{len(candidates)} classifiers in {running_dir} "
            f"({', '.join(p.name for p in candidates)}).\n"
            "Exactly one runs at a time -- remove the others rather than leaving the "
            "choice to whichever script sorts first."
        )
    return candidates[0]


def load_running(device: torch.device | str = "cpu", running_dir: Path = RUNNING_VERSION_DIR):
    """What the demo and the score both load. Never looks in modelVersion/."""
    return load_file(running_classifier(running_dir), device=device)


def describe(payload: dict[str, Any]) -> str:
    """One line naming the loaded version, since the running filename is constant."""
    pre = payload.get("preprocessing", {})
    return (
        f"{payload.get('archive_name', '?')} (v{payload.get('version', '?')}, "
        f"{len(payload['classes'])} classes, T={pre.get('window', '?')}, "
        f"trained {payload.get('trained_at', 'at an unrecorded time')})"
    )


def require_classes(payload: dict[str, Any], expected: list[str], where: str) -> None:
    """Raise, never warn, when a caller's class list differs from the stored one.

    Adding a folder to dataSet/ renumbers every class after it, so a six-class model
    scored against a seven-class test set produces a real-looking, meaningless number.
    """
    stored = list(payload["classes"])
    if stored == list(expected):
        return
    added = [c for c in expected if c not in stored]
    removed = [c for c in stored if c not in expected]
    detail = []
    if added:
        detail.append(f"not in the model: {', '.join(added)}")
    if removed:
        detail.append(f"missing from {where}: {', '.join(removed)}")
    if not detail:
        detail.append(f"same names, different order: model {stored} vs {where} {list(expected)}")
    raise CheckpointError(
        f"class list mismatch between the checkpoint and {where}.\n  "
        + "\n  ".join(detail)
        + "\nThe model's output indices mean what its own class list says. Retrain, or "
        "score against the split this model was trained for."
    )


def require_preprocessing(payload: dict[str, Any], window: int | None = None) -> dict[str, Any]:
    """Check the caller's preprocessing matches the model's, and hand back the settings."""
    pre = payload["preprocessing"]
    if pre.get("keypoint_format") != F.KEYPOINT_FORMAT or pre.get("normalisation") != F.NORMALISATION:
        raise CheckpointError(
            f"this checkpoint expects {pre.get('keypoint_format')} / {pre.get('normalisation')}, "
            f"but common/features.py now produces {F.KEYPOINT_FORMAT} / {F.NORMALISATION}. "
            "Preprocessing changed since it was trained -- retrain rather than feeding it "
            "inputs it has never seen."
        )
    if window is not None and window != pre["window"]:
        raise CheckpointError(
            f"this checkpoint was trained with T={pre['window']}, caller asked for T={window}"
        )
    return pre


def promote(version: int, archive_dir: Path = MODEL_VERSION_DIR,
            running_path: Path = RUNNING_CLASSIFIER) -> Path:
    """Copy an archived version into runningVersion/. Copy, never move.

    Deliberately not called by training: finishing a run must not change what the demo
    and the score are using.
    """
    versions = archived_versions(archive_dir)
    if version not in versions:
        known = ", ".join(f"v{v}" for v in sorted(versions)) or "nothing archived yet"
        raise CheckpointError(f"no classifier_v{version}.pt in {archive_dir} ({known})")
    running_path = Path(running_path)
    running_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(versions[version], running_path)
    return running_path
