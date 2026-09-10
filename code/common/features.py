"""Keypoints -> the fixed-size tensor the classifier eats.

Two jobs, both of which must be byte-identical at training time and at run time:

* normalise, so the model learns the move and not where the boxer stood
* window, so every clip becomes exactly T frames

Both are recorded in the checkpoint (see checkpoint.py) -- a model is only meaningful
against the preprocessing it was trained with.
"""

from __future__ import annotations

import numpy as np

# COCO-17, the layout YOLO-pose returns.
KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
NUM_KEYPOINTS = len(KEYPOINT_NAMES)

LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_HIP, RIGHT_HIP = 11, 12

KEYPOINT_FORMAT = "coco17_xy_conf"
NORMALISATION = "hip_centre_torso_scale"

DEFAULT_WINDOW = 32

# x and y per joint, plus that joint's detection confidence as its own channel:
# a joint the pose model could not see is different from a joint at the origin, and
# without the confidence channel the model cannot tell those two apart.
NUM_FEATURES = NUM_KEYPOINTS * 3

_EPS = 1e-6


def normalise(keypoints: np.ndarray) -> np.ndarray:
    """Centre on the hip midpoint, scale by torso length. Shape (F, 17, 3) in and out.

    Without this the model learns "person stands near the left of the frame" instead
    of "this is a jab", and a boxer filmed one step closer to the camera looks like a
    different move.
    """
    keypoints = np.asarray(keypoints, dtype=np.float32)
    if keypoints.ndim != 3 or keypoints.shape[1:] != (NUM_KEYPOINTS, 3):
        raise ValueError(
            f"expected keypoints shaped (F, {NUM_KEYPOINTS}, 3), got {keypoints.shape}"
        )

    out = keypoints.copy()
    xy = out[:, :, :2]

    hip_centre = (xy[:, LEFT_HIP] + xy[:, RIGHT_HIP]) / 2.0
    shoulder_centre = (xy[:, LEFT_SHOULDER] + xy[:, RIGHT_SHOULDER]) / 2.0
    torso = np.linalg.norm(shoulder_centre - hip_centre, axis=-1)

    # a frame with no visible torso would otherwise divide by ~0 and produce inf;
    # fall back to the median torso of the clip, and to 1.0 if the whole clip is bad
    usable = torso > _EPS
    fallback = float(np.median(torso[usable])) if usable.any() else 1.0
    torso = np.where(usable, torso, fallback)

    out[:, :, :2] = (xy - hip_centre[:, None, :]) / torso[:, None, None]
    return out


def window(keypoints: np.ndarray, length: int = DEFAULT_WINDOW) -> np.ndarray:
    """Resample a clip to exactly `length` frames, however long it started.

    Uniformly spaced sampling rather than crop-or-pad: a punch fills its clip, so
    stretching keeps the whole move and dropping frames keeps its shape. Padding
    instead would teach the model that "ends with frozen frames" means "short clip".
    """
    keypoints = np.asarray(keypoints, dtype=np.float32)
    if length < 1:
        raise ValueError(f"window length must be >= 1, got {length}")
    if keypoints.shape[0] == 0:
        raise ValueError("cannot window an empty clip -- no frames were detected")
    if keypoints.shape[0] == length:
        return keypoints
    idx = np.linspace(0, keypoints.shape[0] - 1, num=length)
    return keypoints[np.rint(idx).astype(int)]


def to_features(keypoints: np.ndarray) -> np.ndarray:
    """(F, 17, 3) -> (51, F): channels first, which is what Conv1d wants."""
    keypoints = np.asarray(keypoints, dtype=np.float32)
    frames = keypoints.shape[0]
    flat = keypoints.reshape(frames, NUM_FEATURES)
    return np.ascontiguousarray(flat.T)


def prepare(keypoints: np.ndarray, length: int = DEFAULT_WINDOW) -> np.ndarray:
    """The whole path, in the one order everything must use: normalise, window, lay out.

    Call this and nothing else. Training and inference both go through here, so they
    cannot drift apart.
    """
    return to_features(window(normalise(keypoints), length))
