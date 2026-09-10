"""The live demo: capture video, show which class is firing and its confidence.

in   webcam (or --source a video file, for repeatable debugging)
     modelholder/runningVersion/  -- both stages, at fixed paths
out  an on-screen overlay. Qualitative only, produces no numbers.

This is NOT where accuracy comes from: a webcam has no ground truth, so nothing
downstream of it can compute "correct". That is getTeststatistic.py's job, against the
labelled testData/ split.

Labels come from the checkpoint, never from a fresh scan of dataSet/ -- that folder can
gain a class at any time, and mapping output index 3 through today's folder list while
the loaded model was trained against last week's is how a demo lies confidently.
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2                                                       # noqa: E402
import torch                                                     # noqa: E402

from common import checkpoint, features, model as M              # noqa: E402
from common.pose import PoseEstimator, PoseError                 # noqa: E402

DEFAULT_SMOOTHING = 5
DEFAULT_FLOOR = 0.5
DEFAULT_CLASSIFY_EVERY = 2

FONT = cv2.FONT_HERSHEY_SIMPLEX

SKELETON = (
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6),
)


def class_colour(index: int, total: int) -> tuple[int, int, int]:
    """Evenly spaced hues, generated from the class count.

    Keyed off the checkpoint's list rather than a hand-written colour table, so a new
    class shows up in the demo with no edit to this file.
    """
    hue = int(180 * index / max(total, 1))
    bgr = cv2.cvtColor(np.uint8([[[hue, 200, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_skeleton(frame: np.ndarray, person: np.ndarray, colour: tuple[int, int, int]) -> None:
    for a, b in SKELETON:
        if person[a, 2] > 0.3 and person[b, 2] > 0.3:
            cv2.line(frame, tuple(person[a, :2].astype(int)), tuple(person[b, :2].astype(int)), colour, 2)
    for x, y, conf in person:
        if conf > 0.3:
            cv2.circle(frame, (int(x), int(y)), 3, colour, -1)


def draw_overlay(frame: np.ndarray, label: str, confidence: float, colour: tuple[int, int, int],
                 classes: list[str], probabilities: np.ndarray | None, version: str) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 44), (0, 0, 0), -1)
    cv2.putText(frame, f"{label}  {confidence:.2f}" if label else "--", (12, 32), FONT, 1.0, colour, 2)
    cv2.putText(frame, version, (frame.shape[1] - 360, 28), FONT, 0.45, (170, 170, 170), 1)

    if probabilities is None:
        return
    # one bar per class, laid out by looping the checkpoint's list
    top = 60
    for index, name in enumerate(classes):
        y = top + index * 22
        width = int(160 * float(probabilities[index]))
        cv2.rectangle(frame, (12, y - 11), (12 + width, y + 4), class_colour(index, len(classes)), -1)
        cv2.putText(frame, f"{name} {probabilities[index]:.2f}", (180, y), FONT, 0.5, (230, 230, 230), 1)


def main() -> int:
    parser = argparse.ArgumentParser(description="shadowBox live demo")
    parser.add_argument("--source", default="0", help="webcam index, or a path to a video file")
    parser.add_argument("--device", default=None, help="cuda / cpu (default: cuda if present)")
    parser.add_argument("--smoothing", type=int, default=DEFAULT_SMOOTHING,
                        help="frames of prediction averaging; 1 disables it")
    parser.add_argument("--floor", type=float, default=DEFAULT_FLOOR,
                        help="stay silent below this confidence rather than guessing")
    parser.add_argument("--classify-every", type=int, default=DEFAULT_CLASSIFY_EVERY,
                        help="run the classifier every N frames; pose still runs every frame")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else M.pick_device()

    try:
        net, payload = checkpoint.load_running(device=device)
    except checkpoint.CheckpointError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    classes: list[str] = payload["classes"]
    settings = checkpoint.require_preprocessing(payload)
    window = int(settings["window"])
    version = checkpoint.describe(payload)
    print(f"loaded {version}")
    print(f"classes: {', '.join(classes)}")

    try:
        estimator = PoseEstimator(device=args.device)
    except PoseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    source: int | str = int(args.source) if args.source.isdigit() else args.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        print(f"error: cannot open video source {args.source!r}", file=sys.stderr)
        return 1

    buffer: deque[np.ndarray] = deque(maxlen=window)
    recent: deque[np.ndarray] = deque(maxlen=max(args.smoothing, 1))
    probabilities: np.ndarray | None = None
    frame_number = 0

    print("running -- q or Esc to quit")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_number += 1

            person = estimator.frame_keypoints(frame)
            if person is not None:
                buffer.append(person)
                draw_skeleton(frame, person, (90, 90, 90))

            # rolling buffer of the last T frames, through the SAME preprocessing as
            # training -- common/features.prepare is the only path either side uses
            if len(buffer) == window and frame_number % max(args.classify_every, 1) == 0:
                batch = features.prepare(np.stack(buffer), window)[None, ...]
                with torch.no_grad():
                    logits = net(torch.from_numpy(batch).to(device))
                    recent.append(torch.softmax(logits, dim=1)[0].cpu().numpy())
                probabilities = np.mean(recent, axis=0)

            label, confidence, colour = "", 0.0, (200, 200, 200)
            if probabilities is not None:
                index = int(np.argmax(probabilities))
                confidence = float(probabilities[index])
                if confidence >= args.floor:
                    label = classes[index]
                    colour = class_colour(index, len(classes))
                else:
                    label = "..."          # silent rather than guessing
            elif len(buffer) < window:
                label = f"filling buffer {len(buffer)}/{window}"

            draw_overlay(frame, label, confidence, colour, classes, probabilities, version)
            cv2.imshow("shadowBox", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
