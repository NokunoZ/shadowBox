"""The score: percentage of correct guesses, into confident.csv.

in   trainDataSet/testData/      (labelled clips -- the ground truth TestRun.py lacks)
     modelholder/runningVersion/ (both stages, fixed paths)
out  confident.csv, one row per test clip

A separate offline pass, NOT a consumer of TestRun.py's output. TestRun.py reads a
webcam, which has no labels, so nothing downstream of it can compute "correct".

This is also how a candidate version gets judged: promote it into runningVersion/,
run this, compare with the previous version's numbers, keep or roll back.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch                                                       # noqa: E402

from common import checkpoint, features, labels, model as M        # noqa: E402
from common.paths import CACHE_DIR, CONFIDENT_CSV, TEST_DIR        # noqa: E402
from common.pose import PoseEstimator, PoseError                   # noqa: E402


def confusion(y_true: list[int], y_pred: list[int], n: int) -> np.ndarray:
    """n x n, sized from the class list so a new class widens the report by itself."""
    matrix = np.zeros((n, n), dtype=int)
    for true, predicted in zip(y_true, y_pred):
        matrix[true, predicted] += 1
    return matrix


def print_confusion(matrix: np.ndarray, classes: list[str]) -> None:
    width = max(max(len(c) for c in classes), 9)
    print("\nconfusion (rows = truth, columns = prediction)")
    print(" " * (width + 2) + "  ".join(f"{c[:6]:>6}" for c in classes))
    for index, name in enumerate(classes):
        row = "  ".join(f"{value:>6}" for value in matrix[index])
        print(f"{name:<{width}}  {row}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Score the running model against testData/.")
    parser.add_argument("--test-dir", type=Path, default=TEST_DIR)
    parser.add_argument("--out", type=Path, default=CONFIDENT_CSV)
    parser.add_argument("--device", default=None, help="cuda / cpu (default: cuda if present)")
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
    print(f"scoring {version}")

    if not args.test_dir.is_dir():
        print(f"error: {args.test_dir} does not exist -- run trainDataSet/dataSpliter.py first",
              file=sys.stderr)
        return 1

    # scoring a six-class model against a seven-class test set produces a real-looking
    # number that is meaningless, so the folders must match the checkpoint exactly
    try:
        checkpoint.require_classes(payload, labels.classes_in(args.test_dir), "testData/")
    except checkpoint.CheckpointError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        estimator = PoseEstimator(device=args.device)
    except PoseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    rows: list[dict[str, object]] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    skipped = 0

    for index, name in enumerate(classes):
        for clip in labels.clip_paths(args.test_dir / name):
            try:
                keypoints = estimator.cached_clip_keypoints(clip, CACHE_DIR)
            except PoseError as exc:
                print(f"  skipping {name}/{clip.name}: {exc}", file=sys.stderr)
                skipped += 1
                continue

            batch = features.prepare(keypoints, window)[None, ...]
            with torch.no_grad():
                probabilities = torch.softmax(net(torch.from_numpy(batch).to(device)), dim=1)[0]
            predicted = int(probabilities.argmax().item())

            y_true.append(index)
            y_pred.append(predicted)
            rows.append({
                "clip": clip.name,
                # names, never bare indices: an index in a CSV is unreadable the moment
                # a class is added
                "true_label": name,
                "predicted_label": classes[predicted],
                "confidence": round(float(probabilities[predicted].item()), 4),
                "correct": int(predicted == index),
                "model_version": payload.get("archive_name", "unknown"),
            })

    if not rows:
        print(f"error: no scorable clips under {args.test_dir}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        # the model version travels in every row: runningVersion/classifier.pt is a
        # constant path whose contents change on every promotion, so a CSV without it
        # cannot be attributed to the model that produced it
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    matrix = confusion(y_true, y_pred, len(classes))
    accuracy = float(np.trace(matrix) / matrix.sum())

    print(f"\noverall accuracy: {accuracy:.3f}  ({int(np.trace(matrix))}/{matrix.sum()} clips)")
    print("\nper class")
    width = max(len(c) for c in classes)
    for index, name in enumerate(classes):
        total = matrix[index].sum()
        shown = f"{matrix[index, index] / total:.3f}" if total else "no test clips"
        print(f"  {name:<{width}}  {shown}  ({matrix[index, index]}/{total})")

    print_confusion(matrix, classes)
    if skipped:
        print(f"\n{skipped} clip(s) skipped -- no person detected")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
