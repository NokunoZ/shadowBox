"""Stage 1: video -> keypoints. Loaded once, used by training and by the live demo.

The pose model is frozen pretrained weights and we never train it. This module exists
so both sides of the project run the SAME weights from the SAME path -- the one in
modelholder/runningVersion/, which is what "running" means.

Pose estimation is far slower than the classifier, so clip keypoints are cached to
.npy and computed once, not once per epoch.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .features import NUM_KEYPOINTS
from .paths import CACHE_DIR, POSE_WEIGHTS


class PoseError(RuntimeError):
    pass


class PoseEstimator:
    """Thin wrapper over ultralytics YOLO-pose.

    Ultralytics is imported lazily so that unit tests, which must not need weights or
    a GPU, can import everything else in common/ without it.
    """

    def __init__(self, weights: Path = POSE_WEIGHTS, device: str | None = None,
                 conf: float = 0.25, verbose: bool = False) -> None:
        weights = Path(weights)
        if not weights.exists():
            raise PoseError(
                f"no pose weights at {weights}.\n"
                "A fresh clone has none -- *.pt is gitignored. Download the YOLO26-pose "
                "weights into modelholder/runningVersion/ (see its noted.md)."
            )
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - environment problem, not logic
            raise PoseError(
                "ultralytics is not installed. "
                "pip install -r requirements.txt --extra-index-url "
                "https://download.pytorch.org/whl/cu128"
            ) from exc

        self.weights = weights
        self.conf = conf
        self.verbose = verbose
        self.device = device
        self.model = YOLO(str(weights))

    # ---- single frames (the live demo) ----------------------------------------

    def frame_keypoints(self, frame: np.ndarray) -> np.ndarray | None:
        """One BGR frame -> (17, 3) for the most prominent person, or None if nobody."""
        results = self.model.predict(
            frame, conf=self.conf, device=self.device, verbose=self.verbose
        )
        return _largest_person(results[0])

    # ---- whole clips (training and scoring) -----------------------------------

    def clip_keypoints(self, clip: Path) -> np.ndarray:
        """A video file -> (F, 17, 3). Frames with nobody in them are dropped.

        Dropping is deliberate: a frame of zeros is a pose claiming every joint sits
        at the origin, which normalisation would then happily scale into nonsense.
        """
        clip = Path(clip)
        frames: list[np.ndarray] = []
        for result in self.model.predict(
            str(clip), stream=True, conf=self.conf, device=self.device, verbose=self.verbose
        ):
            person = _largest_person(result)
            if person is not None:
                frames.append(person)

        if not frames:
            raise PoseError(
                f"no person detected in any frame of {clip.name}. "
                "Check the whole body is in shot -- the pose model cannot find a hip it "
                "cannot see."
            )
        return np.stack(frames).astype(np.float32)

    def cached_clip_keypoints(self, clip: Path, cache_dir: Path = CACHE_DIR) -> np.ndarray:
        """clip_keypoints(), memoised on disk. Re-runs if the clip or weights change."""
        clip = Path(clip)
        cache_path = Path(cache_dir) / f"{_cache_key(clip, self.weights)}.npy"
        if cache_path.exists():
            return np.load(cache_path)

        keypoints = self.clip_keypoints(clip)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, keypoints)
        return keypoints


def _largest_person(result) -> np.ndarray | None:
    """Pick one skeleton per frame: the largest box, i.e. the boxer, not a bystander."""
    keypoints = getattr(result, "keypoints", None)
    if keypoints is None or keypoints.data is None or len(keypoints.data) == 0:
        return None

    data = keypoints.data.cpu().numpy()          # (people, 17, 3)
    if data.shape[0] > 1 and getattr(result, "boxes", None) is not None:
        boxes = result.boxes.xywh.cpu().numpy()
        index = int(np.argmax(boxes[:, 2] * boxes[:, 3]))
    else:
        index = 0

    person = data[index]
    if person.shape != (NUM_KEYPOINTS, 3):
        raise PoseError(
            f"expected {NUM_KEYPOINTS} keypoints with (x, y, confidence), got {person.shape}. "
            "The pose weights do not use the COCO-17 layout common/features.py assumes."
        )
    return person.astype(np.float32)


def _cache_key(clip: Path, weights: Path) -> str:
    """Identity of a cache entry: which clip, and which pose weights read it.

    Deliberately keyed on the clip's NAME, size and mtime rather than its path. The
    same clip is reachable as dataSet/jab/x.mp4, trainData/jab/x.mp4 and (after the
    next re-split) testData/jab/x.mp4 -- keying on the path would recompute pose for
    every one of those, and throw the whole cache away each time dataSpliter runs.

    Includes the weights file so swapping the pose model invalidates the cache instead
    of silently training on keypoints from the old one.
    """
    stat = clip.stat()
    raw = f"{clip.name}|{stat.st_size}|{int(stat.st_mtime)}|{weights.name}|{weights.stat().st_size}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"{clip.stem}_{digest}"
