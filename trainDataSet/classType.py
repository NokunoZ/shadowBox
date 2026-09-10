"""Return all classes of the dataset.

Classes are DISCOVERED, never declared: one folder under dataSet/ is one class.
Nothing here holds a list of names or a count -- see trainDataSet/dataSet/noted.md.

This scan feeds training and splitting only. Inference reads its class list out of
the checkpoint instead, because adding a folder renumbers every class after it and a
scan run today need not agree with a model trained last week.
"""

from __future__ import annotations

import re
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent / "dataSet"

VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm"})

# lowercase, digits, underscores between words. Enforced so `body-shot`, `Body Shot`
# and `bodyShot` cannot become three classes that all mean the same move.
NAME_PATTERN = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")

NAME_RULE = "lowercase letters, digits and single underscores, e.g. 'jab', 'body_shot'"


class ClassScanError(RuntimeError):
    """Raised when the dataset folder cannot be turned into a usable class list."""


def is_ignored(name: str) -> bool:
    """Scaffolding folders: `_raw/`, `_rejected/`, `.DS_Store` and friends."""
    return name.startswith(".") or name.startswith("_")


def clip_paths(class_dir: Path) -> list[Path]:
    """Video files directly inside one class folder, sorted for a stable order."""
    return sorted(
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
    )


def scan(dataset_dir: Path | str = DATASET_DIR) -> list[str]:
    """The class names, sorted.

    A folder counts as a class when it is not ignored, its name matches the naming
    convention, and it holds at least one video file. An empty folder is scaffolding:
    counting it would add an output unit the model can never learn but which still
    eats probability mass.
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise ClassScanError(f"dataset folder does not exist: {dataset_dir}")

    names: list[str] = []
    rejected: list[str] = []
    for entry in sorted(dataset_dir.iterdir()):
        if not entry.is_dir() or is_ignored(entry.name):
            continue
        if not NAME_PATTERN.match(entry.name):
            rejected.append(entry.name)
            continue
        if clip_paths(entry):
            names.append(entry.name)

    if rejected:
        raise ClassScanError(
            "folder names break the naming convention ("
            + NAME_RULE
            + "): "
            + ", ".join(sorted(rejected))
            + "\nRename them. A near-duplicate name silently becomes a second class."
        )

    # sorted() above already orders them; being explicit because everything downstream
    # -- checkpoint indices, confusion matrices, saved models -- depends on this order
    # never drifting.
    return sorted(names)


def class_counts(dataset_dir: Path | str = DATASET_DIR) -> dict[str, int]:
    """Class name -> number of clips. Used for logging and for class weights."""
    dataset_dir = Path(dataset_dir)
    return {name: len(clip_paths(dataset_dir / name)) for name in scan(dataset_dir)}


def class_map(dataset_dir: Path | str = DATASET_DIR) -> tuple[list[str], dict[str, int]]:
    """Both directions at once: the ordered names, and name -> index.

    Returning both is what stops callers rebuilding the mapping with their own
    `.index()` call and getting a different answer.
    """
    names = scan(dataset_dir)
    return names, {name: i for i, name in enumerate(names)}


def require_classes(dataset_dir: Path | str = DATASET_DIR) -> tuple[list[str], dict[str, int]]:
    """class_map(), but fails loudly when the dataset has no footage yet."""
    names, to_index = class_map(dataset_dir)
    if not names:
        raise ClassScanError(
            f"no classes found in {Path(dataset_dir)}\n"
            "Every class folder is empty, so there is nothing to train on. "
            "Add video clips -- see trainDataSet/dataSet/noted.md."
        )
    return names, to_index


if __name__ == "__main__":
    counts = class_counts()
    if not counts:
        print(f"no classes found in {DATASET_DIR} (every folder is empty)")
    else:
        width = max(len(n) for n in counts)
        for i, (name, n) in enumerate(counts.items()):
            print(f"{i:>3}  {name:<{width}}  {n} clip(s)")
        print(f"\n{len(counts)} class(es)")
