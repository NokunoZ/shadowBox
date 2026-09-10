"""classType.py -- the highest-value tests in the project.

If class ORDER ever changes between training and inference, every prediction is
mislabelled and nothing crashes. These tests pin the order down.
"""

from __future__ import annotations

import pytest

from common import labels


def test_order_is_sorted_and_stable(make_dataset):
    root = make_dataset({"uppercut": 2, "jab": 2, "block": 2, "cross": 2})
    assert labels.scan(root) == ["block", "cross", "jab", "uppercut"]
    assert labels.scan(root) == labels.scan(root)      # repeat scans agree


def test_folder_with_clips_is_a_class(make_dataset):
    root = make_dataset({"jab": 1})
    assert labels.scan(root) == ["jab"]


def test_empty_folder_is_not_a_class(make_dataset):
    root = make_dataset({"jab": 2, "hook": 0})
    # scaffolding, not a class: counting it would add an output unit the model can
    # never learn but which still eats probability mass
    assert labels.scan(root) == ["jab"]


def test_underscore_and_dot_folders_are_ignored(make_dataset):
    root = make_dataset({"jab": 2})
    make_dataset({"_raw": 3, "_rejected": 2}, root=root)
    (root / ".hidden").mkdir()
    (root / ".hidden" / "x.mp4").write_bytes(b"")
    assert labels.scan(root) == ["jab"]


def test_non_video_files_do_not_make_a_class(make_dataset):
    root = make_dataset({"jab": 2})
    (root / "hook").mkdir()
    (root / "hook" / "noted.md").write_text("a note is not footage")
    assert labels.scan(root) == ["jab"]


def test_adding_a_folder_mid_alphabet_renumbers_the_classes_after_it(make_dataset):
    """The trap, asserted explicitly so it is recorded rather than discovered later
    by a model that mislabels everything."""
    root = make_dataset({"cross": 1, "hook": 1, "jab": 1, "uppercut": 1})
    _, before = labels.class_map(root)
    assert before == {"cross": 0, "hook": 1, "jab": 2, "uppercut": 3}

    make_dataset({"idle": 1}, root=root)
    _, after = labels.class_map(root)

    assert after == {"cross": 0, "hook": 1, "idle": 2, "jab": 3, "uppercut": 4}
    assert before["jab"] != after["jab"]           # everything after idle shifted by one
    assert before["cross"] == after["cross"]       # everything before it did not


@pytest.mark.parametrize("bad", ["body-shot", "Body Shot", "bodyShot", "Jab", "body shot"])
def test_names_breaking_the_convention_are_rejected(make_dataset, bad):
    """A near-duplicate name must not quietly become a second class."""
    # the good class is `cross`, not `jab`: on Windows a `Jab/` folder would simply BE
    # the existing `jab/` folder, and the test would pass for the wrong reason
    root = make_dataset({"cross": 1})
    make_dataset({bad: 1}, root=root)
    with pytest.raises(labels.ClassScanError) as excinfo:
        labels.scan(root)
    assert bad in str(excinfo.value)


def test_class_map_returns_both_directions(make_dataset):
    root = make_dataset({"jab": 1, "cross": 1})
    names, to_index = labels.class_map(root)
    assert names == ["cross", "jab"]
    assert [to_index[n] for n in names] == [0, 1]


def test_class_counts_are_discovered_not_declared(make_dataset):
    root = make_dataset({"jab": 5, "cross": 2})
    assert labels.class_counts(root) == {"cross": 2, "jab": 5}


def test_empty_dataset_fails_loudly(make_dataset):
    root = make_dataset({"jab": 0, "cross": 0})
    assert labels.scan(root) == []
    with pytest.raises(labels.ClassScanError):
        labels.require_classes(root)
