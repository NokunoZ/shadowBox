"""Create testData/ and trainData/ from classType.py, then split dataSet into them.

Three rules decide whether the accuracy number means anything (trainDataSet/noted.md):

1. whole clips are split, never frames -- frames of one clip on both sides means the
   model has already seen the test set and the score is fiction
2. clips filmed of the same person in the same session stay on the same side, so the
   model cannot score by recognising the boxer instead of the punch
3. the shuffle is seeded, so a re-run reproduces the same split

The split is REGENERATED, not patched: both output folders are cleared first, or a
renamed or deleted class leaves a stale folder behind that training picks up as real.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from classType import DATASET_DIR, ClassScanError, clip_paths, require_classes  # noqa: E402

ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "trainData"
TEST_DIR = ROOT / "testData"

DEFAULT_TEST_RATIO = 0.2
DEFAULT_SEED = 1337

# Filming convention: everything before a double underscore identifies who was filmed
# and when -- `kasi_s01__jab_004.mp4` groups with `kasi_s01__jab_005.mp4`. A clip with
# no `__` is its own group, which is the right default while one person is filming.
GROUP_SEPARATOR = "__"

MIN_TEST_CLIPS = 2


def group_key(clip: Path) -> str:
    """Which person/session a clip belongs to. Groups are never split across sides."""
    stem = clip.stem
    return stem.split(GROUP_SEPARATOR)[0] if GROUP_SEPARATOR in stem else stem


def split_one_class(clips: list[Path], test_ratio: float, rng: random.Random
                    ) -> tuple[list[Path], list[Path], bool]:
    """Split whole groups of whole clips, aiming at test_ratio by clip count.

    Returns (train, test, fell_back) -- see below for what the fallback means.
    """
    groups: dict[str, list[Path]] = defaultdict(list)
    for clip in clips:
        groups[group_key(clip)].append(clip)

    ordered = sorted(groups)          # deterministic before the seeded shuffle
    rng.shuffle(ordered)
    target = round(len(clips) * test_ratio)

    if len(ordered) < 2:
        # One person/session for the whole class: keeping groups intact would put
        # every clip on one side and leave the other empty. Split whole CLIPS instead
        # and say so -- rule 1 (never split frames) still holds, but rule 2 cannot,
        # and a score measured this way partly reflects recognising the boxer.
        shuffled = sorted(clips)
        rng.shuffle(shuffled)
        return sorted(shuffled[target:]), sorted(shuffled[:target]), True

    test: list[Path] = []
    train: list[Path] = []
    unassigned = len(clips)
    for key in ordered:
        group = groups[key]
        unassigned -= len(group)
        # groups are indivisible; fill the test side until it hits the target, but
        # never take the last group that would otherwise leave train empty
        if len(test) < target and (unassigned > 0 or train):
            test.extend(group)
        else:
            train.extend(group)
    return sorted(train), sorted(test), False


def place(src: Path, dst: Path, mode: str) -> None:
    """hardlink by default: no duplicated video on disk, no admin rights needed."""
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "symlink":
        os.symlink(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        # different volume, or a filesystem without hardlinks -- copying still works
        shutil.copy2(src, dst)


def clear(directory: Path) -> None:
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)


def split(
    dataset_dir: Path = DATASET_DIR,
    train_dir: Path = TRAIN_DIR,
    test_dir: Path = TEST_DIR,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = DEFAULT_SEED,
    mode: str = "hardlink",
) -> dict[str, tuple[int, int]]:
    """Regenerate both splits. Returns class -> (train count, test count)."""
    classes, _ = require_classes(dataset_dir)
    rng = random.Random(seed)

    clear(train_dir)
    clear(test_dir)

    result: dict[str, tuple[int, int]] = {}
    thin: list[str] = []
    single_group: list[str] = []
    for name in classes:
        clips = clip_paths(Path(dataset_dir) / name)
        train, test, fell_back = split_one_class(clips, test_ratio, rng)
        if fell_back:
            single_group.append(name)

        for side, files in ((train_dir, train), (test_dir, test)):
            (side / name).mkdir(parents=True, exist_ok=True)
            for clip in files:
                place(clip, side / name / clip.name, mode)

        result[name] = (len(train), len(test))
        if len(test) < MIN_TEST_CLIPS:
            thin.append(f"{name}: {len(test)} test clip(s) out of {len(clips)}")

    if single_group:
        print(
            "\nNOTE -- these classes were filmed in a single person/session, so clips "
            "of that\nsession sit on both sides of the split:\n  "
            + ", ".join(single_group)
            + f"\n  Name clips <person>_<session>{GROUP_SEPARATOR}<whatever>.mp4 once more "
            "than one person\n  or session is filmed, or the score partly measures "
            "recognising the boxer.",
            file=sys.stderr,
        )

    if thin:
        print("\nWARNING -- these classes cannot be scored meaningfully:", file=sys.stderr)
        for line in thin:
            print(f"  {line}", file=sys.stderr)
        print(
            f"  A class with fewer than {MIN_TEST_CLIPS} test clips scores 0% or 100% "
            "and means nothing. Film more.",
            file=sys.stderr,
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="change only if you want a different split; the default "
                             "is what makes a re-run reproducible")
    parser.add_argument("--mode", choices=("hardlink", "copy", "symlink"), default="hardlink",
                        help="how clips land in the split folders (default: hardlink, "
                             "which costs no extra disk)")
    args = parser.parse_args()

    try:
        counts = split(test_ratio=args.test_ratio, seed=args.seed, mode=args.mode)
    except ClassScanError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    width = max(len(n) for n in counts)
    print(f"split (seed={args.seed}, test_ratio={args.test_ratio}, mode={args.mode})")
    for name, (n_train, n_test) in counts.items():
        print(f"  {name:<{width}}  train {n_train:>4}   test {n_test:>4}")
    total_train = sum(a for a, _ in counts.values())
    total_test = sum(b for _, b in counts.values())
    print(f"  {'total':<{width}}  train {total_train:>4}   test {total_test:>4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
