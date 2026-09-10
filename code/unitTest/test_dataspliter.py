"""dataSpliter.py -- the split decides whether the accuracy number means anything."""

from __future__ import annotations

import pytest

from common import labels


def run_split(spliter, tmp_path, dataset, **kwargs):
    train = tmp_path / "trainData"
    test = tmp_path / "testData"
    counts = spliter.split(dataset_dir=dataset, train_dir=train, test_dir=test,
                           mode="copy", **kwargs)
    return train, test, counts


def test_split_ratio_is_respected(spliter, tmp_path, make_dataset):
    dataset = make_dataset({"jab": 10, "cross": 10})
    _, _, counts = run_split(spliter, tmp_path, dataset, test_ratio=0.2)
    for name, (n_train, n_test) in counts.items():
        assert n_train + n_test == 10, name
        assert n_test == 2, name


def test_no_clip_appears_in_both_sides(spliter, tmp_path, make_dataset):
    dataset = make_dataset({"jab": 10, "cross": 8})
    train, test, _ = run_split(spliter, tmp_path, dataset)
    for name in labels.classes_in(train):
        in_train = {p.name for p in (train / name).iterdir()}
        in_test = {p.name for p in (test / name).iterdir()}
        assert not (in_train & in_test)


def test_whole_clips_and_nothing_lost(spliter, tmp_path, make_dataset):
    dataset = make_dataset({"jab": 7})
    train, test, _ = run_split(spliter, tmp_path, dataset)
    originals = {p.name for p in (dataset / "jab").iterdir()}
    split = {p.name for p in (train / "jab").iterdir()} | {p.name for p in (test / "jab").iterdir()}
    assert split == originals


def test_same_seed_reproduces_the_split(spliter, tmp_path, make_dataset):
    dataset = make_dataset({"jab": 10})
    train, _, _ = run_split(spliter, tmp_path, dataset, seed=7)
    first = sorted(p.name for p in (train / "jab").iterdir())
    train, _, _ = run_split(spliter, tmp_path, dataset, seed=7)
    assert sorted(p.name for p in (train / "jab").iterdir()) == first


def test_clips_of_one_person_and_session_stay_on_one_side(spliter, tmp_path, make_dataset):
    """Otherwise the model can score well by recognising the boxer, not the punch."""
    dataset = tmp_path / "dataSet"
    (dataset / "jab").mkdir(parents=True)
    for person in ("kasi_s01", "sam_s01"):
        for i in range(5):
            (dataset / "jab" / f"{person}__jab_{i}.mp4").write_bytes(b"")

    train, test, _ = run_split(spliter, tmp_path, dataset, test_ratio=0.5)
    for side in (train, test):
        groups = {spliter.group_key(p) for p in (side / "jab").iterdir()}
        assert len(groups) <= 1, "a person/session was split across both sides"


def test_a_new_class_needs_no_code_change(spliter, tmp_path, make_dataset):
    dataset = make_dataset({"jab": 4, "cross": 4})
    make_dataset({"idle": 4}, root=dataset)
    train, test, counts = run_split(spliter, tmp_path, dataset)
    assert "idle" in counts
    assert (train / "idle").is_dir() and (test / "idle").is_dir()


def test_rerun_regenerates_and_leaves_no_stale_folder(spliter, tmp_path, make_dataset):
    """A removed class must not leave a folder behind for training to pick up."""
    dataset = make_dataset({"jab": 4, "cross": 4, "hook": 4})
    train, test, _ = run_split(spliter, tmp_path, dataset)
    assert (train / "hook").is_dir()

    for clip in (dataset / "hook").iterdir():
        clip.unlink()
    (dataset / "hook").rmdir()

    train, test, counts = run_split(spliter, tmp_path, dataset)
    assert "hook" not in counts
    assert not (train / "hook").exists()
    assert not (test / "hook").exists()


def test_thin_class_is_reported_loudly(spliter, tmp_path, make_dataset, capsys):
    """A class with fewer than two test clips scores 0% or 100% and means nothing."""
    dataset = make_dataset({"jab": 20, "cross": 3})
    run_split(spliter, tmp_path, dataset, test_ratio=0.2)
    assert "cross" in capsys.readouterr().err


def test_empty_dataset_fails_rather_than_making_empty_folders(spliter, tmp_path, make_dataset):
    dataset = make_dataset({"jab": 0})
    with pytest.raises(labels.ClassScanError):
        run_split(spliter, tmp_path, dataset)


def test_one_person_and_session_still_yields_both_sides(spliter, tmp_path):
    """The group rule cannot apply when a class has a single session. It must fall
    back to splitting whole clips -- never leave train or test empty."""
    dataset = tmp_path / "dataSet"
    (dataset / "jab").mkdir(parents=True)
    for i in range(14):
        (dataset / "jab" / f"kasi_s01__jab_{i}.mp4").write_bytes(b"")

    train, test, counts = run_split(spliter, tmp_path, dataset, test_ratio=0.2)

    assert counts["jab"] == (11, 3)
    assert list((train / "jab").iterdir()) and list((test / "jab").iterdir())


def test_the_single_session_fallback_is_announced(spliter, tmp_path, make_dataset, capsys):
    """Silently splitting one session across both sides would inflate the score."""
    dataset = make_dataset({"jab": 10})          # no `__`, so every clip is its own group
    (dataset / "cross").mkdir()
    for i in range(10):
        (dataset / "cross" / f"kasi_s01__cross_{i}.mp4").write_bytes(b"")

    run_split(spliter, tmp_path, dataset)

    err = capsys.readouterr().err
    assert "cross" in err
    assert "jab" not in err.split("split")[0]    # per-clip groups need no fallback
