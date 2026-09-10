"""Train the stage-2 classifier.

in   trainDataSet/trainData/   (a validation slice is carved out of it, see --val-ratio)
out  modelholder/modelVersion/classifier_v<N>.pt

Never trains YOLO -- stage 1 is frozen pretrained weights.
Never writes to modelholder/runningVersion/ -- finishing a run must not change what
the demo and the score are using. Promoting is a separate, deliberate copy:

    cp modelholder/modelVersion/classifier_v3.pt modelholder/runningVersion/classifier.pt

Nothing here assumes how many classes there are. The final layer is sized from the
discovered list, and so is every weight vector and metric.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch                                                          # noqa: E402
import torch.nn as nn                                                 # noqa: E402
from torch.utils.data import DataLoader, TensorDataset                # noqa: E402

from common import checkpoint, features, labels, model as M           # noqa: E402
from common.paths import CACHE_DIR, DATASET_DIR, MODEL_VERSION_DIR, TRAIN_DIR  # noqa: E402
from common.pose import PoseEstimator, PoseError                      # noqa: E402

DEFAULT_EPOCHS = 60
DEFAULT_BATCH = 16
DEFAULT_LR = 1e-3
DEFAULT_VAL_RATIO = 0.2
DEFAULT_SEED = 1337
MIN_VAL_CLIPS_PER_CLASS = 1


# ---------------------------------------------------------------- data ------

def load_split(split_dir: Path, classes: list[str], window: int, estimator: PoseEstimator
               ) -> tuple[np.ndarray, np.ndarray, list[Path], dict[str, int]]:
    """Every clip in a split -> (N, 51, T) features, labels, paths, raw frame counts.

    Pose runs ONCE per clip here and the result is cached to .npy. Re-running it every
    epoch would dominate training time -- the classifier is the cheap half.
    """
    xs: list[np.ndarray] = []
    ys: list[int] = []
    paths: list[Path] = []
    raw_lengths: dict[str, int] = {}

    for index, name in enumerate(classes):
        clips = labels.clip_paths(split_dir / name)
        for clip in clips:
            try:
                keypoints = estimator.cached_clip_keypoints(clip, CACHE_DIR)
            except PoseError as exc:
                print(f"  skipping {name}/{clip.name}: {exc}", file=sys.stderr)
                continue
            raw_lengths[f"{name}/{clip.name}"] = int(keypoints.shape[0])
            xs.append(features.prepare(keypoints, window))
            ys.append(index)
            paths.append(clip)
        print(f"  {name:<12} {len(clips):>4} clip(s)")

    if not xs:
        raise SystemExit(
            f"no usable clips under {split_dir}. Run trainDataSet/dataSpliter.py first, "
            "and check the clips actually contain a visible person."
        )
    return np.stack(xs), np.array(ys, dtype=np.int64), paths, raw_lengths


def carve_validation(y: np.ndarray, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split indices into train and val, stratified per class.

    trainDataSet/dataSpliter.py produces train and test only, and tuning epochs or
    learning rate against the test set silently inflates the reported score. So the
    val slice comes out of TRAIN and the test split is never touched here.
    """
    rng = random.Random(seed)
    by_class: dict[int, list[int]] = defaultdict(list)
    for i, label in enumerate(y):
        by_class[int(label)].append(i)

    train_idx: list[int] = []
    val_idx: list[int] = []
    for label in sorted(by_class):
        indices = by_class[label][:]
        rng.shuffle(indices)
        n_val = max(MIN_VAL_CLIPS_PER_CLASS, round(len(indices) * val_ratio))
        n_val = min(n_val, max(len(indices) - 1, 0))   # never leave a class with no training clip
        val_idx += indices[:n_val]
        train_idx += indices[n_val:]

    return np.array(sorted(train_idx), dtype=np.int64), np.array(sorted(val_idx), dtype=np.int64)


def class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    """Inverse-frequency weights, computed from the data -- never a hand-written table.

    A table goes stale the first time a folder is added; this cannot.
    """
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


# ------------------------------------------------------------ training ------

def run_epoch(net, loader, criterion, device, optimiser=None) -> tuple[float, float, np.ndarray, np.ndarray]:
    training = optimiser is not None
    net.train(training)

    total_loss = 0.0
    total = 0
    correct = 0
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []

    with torch.set_grad_enabled(training):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = net(batch_x)
            loss = criterion(logits, batch_y)

            if training:
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()

            predicted = logits.argmax(dim=1)
            total_loss += float(loss.item()) * batch_y.size(0)
            correct += int((predicted == batch_y).sum().item())
            total += batch_y.size(0)
            all_true.append(batch_y.cpu().numpy())
            all_pred.append(predicted.cpu().numpy())

    return (
        total_loss / max(total, 1),
        correct / max(total, 1),
        np.concatenate(all_true) if all_true else np.array([]),
        np.concatenate(all_pred) if all_pred else np.array([]),
    )


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict[str, float]:
    """Sized from the class list, so a new class widens the report with no edit here."""
    out: dict[str, float] = {}
    for index, name in enumerate(classes):
        mask = y_true == index
        out[name] = float((y_pred[mask] == index).mean()) if mask.any() else float("nan")
    return out


def report_lengths(raw_lengths: dict[str, int], window: int) -> None:
    """Print the real clip lengths, since T should be chosen from footage, not guessed."""
    if not raw_lengths:
        return
    lengths = np.array(sorted(raw_lengths.values()))
    print(
        f"\nclip length (frames with a person detected): "
        f"min {lengths.min()}, median {int(np.median(lengths))}, max {lengths.max()}"
    )
    print(f"window T={window} -- "
          f"{int((lengths < window).sum())} clip(s) stretched, "
          f"{int((lengths > window).sum())} thinned, "
          f"{int((lengths == window).sum())} exact")


# ---------------------------------------------------------------- main ------

def main() -> int:
    parser = argparse.ArgumentParser(description="Train the shadowBox stage-2 classifier.")
    parser.add_argument("--window", type=int, default=features.DEFAULT_WINDOW,
                        help="T, frames per clip after resampling")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO,
                        help="validation slice carved out of trainData (the test split "
                             "is never touched here)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default=None, help="cuda / cpu (default: cuda if present)")
    parser.add_argument("--resume", type=Path, default=None,
                        help="an explicit modelVersion/classifier_vN.pt to continue from; "
                             "never 'whatever is running'")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # --- the class list, logged before anything else ------------------------
    try:
        classes, _ = labels.require_classes(DATASET_DIR)
    except labels.ClassScanError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not TRAIN_DIR.is_dir():
        print(f"error: {TRAIN_DIR} does not exist -- run trainDataSet/dataSpliter.py first",
              file=sys.stderr)
        return 1

    split_classes = labels.classes_in(TRAIN_DIR)
    if split_classes != classes:
        print(
            "error: trainData/ does not match dataSet/.\n"
            f"  dataSet:   {', '.join(classes)}\n"
            f"  trainData: {', '.join(split_classes)}\n"
            "Re-run trainDataSet/dataSpliter.py -- the split is regenerated, not patched.",
            file=sys.stderr,
        )
        return 1

    print(f"{len(classes)} class(es): {', '.join(classes)}")

    device = torch.device(args.device) if args.device else M.pick_device()
    print(f"device: {device}")

    estimator = PoseEstimator(device=args.device)
    print("\nreading trainData/ (pose runs once per clip, then it is cached):")
    x, y, _, raw_lengths = load_split(TRAIN_DIR, classes, args.window, estimator)
    report_lengths(raw_lengths, args.window)

    train_idx, val_idx = carve_validation(y, args.val_ratio, args.seed)
    print(f"\n{len(train_idx)} training clip(s), {len(val_idx)} validation clip(s)")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x[train_idx]), torch.from_numpy(y[train_idx])),
        batch_size=args.batch_size, shuffle=True, drop_last=len(train_idx) > args.batch_size,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x[val_idx]), torch.from_numpy(y[val_idx])),
        batch_size=args.batch_size,
    )

    # --- the model ----------------------------------------------------------
    net = M.TemporalCNN(len(classes))          # width from len(classes), never a literal

    if args.resume:
        previous, payload = checkpoint.load_file(args.resume, device="cpu")
        # a differing class list is a hard stop: the last layer's shape no longer
        # matches and the surviving weights are wired to the old ordering
        checkpoint.require_classes(payload, classes, "dataSet/")
        checkpoint.require_preprocessing(payload, args.window)
        net.load_state_dict(previous.state_dict())
        print(f"resumed from {checkpoint.describe(payload)}")

    net.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights(y[train_idx], len(classes)).to(device))
    optimiser = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    print(f"\ntraining for {args.epochs} epoch(s)")
    best_val_acc = 0.0
    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, _, _ = run_epoch(net, train_loader, criterion, device, optimiser)
        val_loss, val_acc, val_true, val_pred = run_epoch(net, val_loader, criterion, device)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": val_acc})

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            # train and val tracked separately: train loss alone cannot show overfitting
            print(f"  epoch {epoch:>3}  train loss {train_loss:.4f} acc {train_acc:.3f}"
                  f"   val loss {val_loss:.4f} acc {val_acc:.3f}")

    net.load_state_dict(best_state)
    net.to(device)
    _, _, val_true, val_pred = run_epoch(net, val_loader, criterion, device)
    breakdown = per_class_accuracy(val_true, val_pred, classes)

    print(f"\nbest validation accuracy: {best_val_acc:.3f}")
    for name, accuracy in breakdown.items():
        shown = "no val clips" if np.isnan(accuracy) else f"{accuracy:.3f}"
        print(f"  {name:<12} {shown}")

    # --- save: a NEW version in the archive, nothing promoted ---------------
    saved = checkpoint.save(
        net.cpu(),
        classes=classes,
        window=args.window,
        archive_dir=MODEL_VERSION_DIR,
        metrics={
            "best_val_accuracy": best_val_acc,
            "per_class_val_accuracy": breakdown,
            "epochs": args.epochs,
            "history": history[-1] if history else {},
        },
        class_counts={name: int((y == i).sum()) for i, name in enumerate(classes)},
        extra={"seed": args.seed, "resumed_from": str(args.resume) if args.resume else None},
    )

    print(f"\nsaved {saved}")
    print("nothing was promoted -- the demo and the score still use the previous model.")
    print(f"  to promote:  cp {saved} modelholder/runningVersion/classifier.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
