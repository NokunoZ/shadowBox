"""The modelVersion/ + runningVersion/ contract.

Everything here runs against tmp_path. A test that promotes or overwrites for real is
a test that can destroy the working model.
"""

from __future__ import annotations

import pytest
import torch

from common import checkpoint
from common import features
from common import model as M

CLASSES = ["block", "cross", "dodge", "hook", "jab", "uppercut"]


@pytest.fixture
def archive(tmp_path):
    path = tmp_path / "modelVersion"
    path.mkdir()
    return path


@pytest.fixture
def running(tmp_path):
    path = tmp_path / "runningVersion"
    path.mkdir()
    return path


def small(num_classes: int) -> M.TemporalCNN:
    return M.TemporalCNN(num_classes, channels=(16, 32), kernel_size=3)


# ---- round trip -------------------------------------------------------------

def test_round_trip_gives_identical_outputs(archive):
    net = small(len(CLASSES)).eval()
    batch = torch.randn(2, features.NUM_FEATURES, 32)
    with torch.no_grad():
        before = net(batch)

    saved = checkpoint.save(net, CLASSES, window=32, archive_dir=archive)
    loaded, payload = checkpoint.load_file(saved)
    with torch.no_grad():
        after = loaded(batch)

    assert torch.allclose(before, after, atol=1e-6)
    assert payload["classes"] == CLASSES          # same list, same order


def test_checkpoint_carries_its_identity_and_settings(archive):
    saved = checkpoint.save(small(len(CLASSES)), CLASSES, window=48, archive_dir=archive)
    _, payload = checkpoint.load_file(saved)

    # the promoted copy is renamed to classifier.pt, so these fields are the only
    # thing identifying which archive file is running
    assert payload["version"] == 1
    assert payload["archive_name"] == "classifier_v1.pt"
    assert payload["preprocessing"]["window"] == 48
    assert payload["preprocessing"]["normalisation"] == features.NORMALISATION
    assert payload["preprocessing"]["keypoint_format"] == features.KEYPOINT_FORMAT
    assert payload["arch"]["arch"] == M.ARCH_NAME


def test_a_bare_state_dict_is_refused(tmp_path):
    path = tmp_path / "classifier_v1.pt"
    torch.save(small(3).state_dict(), path)
    with pytest.raises(checkpoint.CheckpointError):
        checkpoint.load_file(path)


# ---- versioning never overwrites --------------------------------------------

def test_next_version_is_max_plus_one(archive):
    assert checkpoint.next_version(archive) == 1
    checkpoint.save(small(3), ["a", "b", "c"], 32, archive_dir=archive)
    checkpoint.save(small(3), ["a", "b", "c"], 32, archive_dir=archive)
    assert checkpoint.next_version(archive) == 3
    assert sorted(p.name for p in archive.iterdir()) == ["classifier_v1.pt", "classifier_v2.pt"]


def test_saving_never_replaces_an_existing_version(archive):
    """The test that protects every previously trained model."""
    first = checkpoint.save(small(3), ["a", "b", "c"], 32, archive_dir=archive)
    original = first.read_bytes()

    checkpoint.save(small(3), ["a", "b", "c"], 32, archive_dir=archive)

    assert first.read_bytes() == original
    assert (archive / "classifier_v2.pt").exists()


def test_a_collision_raises_rather_than_overwriting(archive, monkeypatch):
    checkpoint.save(small(3), ["a", "b", "c"], 32, archive_dir=archive)
    monkeypatch.setattr(checkpoint, "next_version", lambda *_a, **_k: 1)
    with pytest.raises(checkpoint.CheckpointError):
        checkpoint.save(small(3), ["a", "b", "c"], 32, archive_dir=archive)


def test_an_interior_gap_is_not_refilled(archive):
    """Deleting v2 of v1..v3 must not make the next run v2 as well -- two different
    models sharing a version number make every note that cites it ambiguous."""
    checkpoint.save(small(3), ["a", "b", "c"], 32, archive_dir=archive)
    second = checkpoint.save(small(3), ["a", "b", "c"], 32, archive_dir=archive)
    checkpoint.save(small(3), ["a", "b", "c"], 32, archive_dir=archive)

    second.unlink()

    assert checkpoint.next_version(archive) == 4


def test_deleting_the_highest_version_does_reuse_its_number(archive):
    """The documented limit of max+1, pinned here so it is a known property rather
    than a surprise: do not delete the newest file from the archive."""
    checkpoint.save(small(3), ["a", "b", "c"], 32, archive_dir=archive)
    second = checkpoint.save(small(3), ["a", "b", "c"], 32, archive_dir=archive)
    second.unlink()
    assert checkpoint.next_version(archive) == 2


def test_saving_does_not_touch_the_running_folder(archive, running):
    (running / "classifier.pt").write_bytes(b"the model that is serving")
    before = (running / "classifier.pt").read_bytes()

    checkpoint.save(small(3), ["a", "b", "c"], 32, archive_dir=archive)

    assert (running / "classifier.pt").read_bytes() == before
    assert list(running.iterdir()) == [running / "classifier.pt"]


# ---- class list mismatches --------------------------------------------------

def test_matching_class_list_passes(archive):
    saved = checkpoint.save(small(len(CLASSES)), CLASSES, 32, archive_dir=archive)
    _, payload = checkpoint.load_file(saved)
    checkpoint.require_classes(payload, CLASSES, "testData/")


def test_an_added_class_raises_and_names_it(archive):
    """Build for N, save, load against N+1: labels shifted by one must not run."""
    saved = checkpoint.save(small(len(CLASSES)), CLASSES, 32, archive_dir=archive)
    _, payload = checkpoint.load_file(saved)

    with pytest.raises(checkpoint.CheckpointError) as excinfo:
        checkpoint.require_classes(payload, sorted(CLASSES + ["idle"]), "dataSet/")
    assert "idle" in str(excinfo.value)


def test_a_removed_class_raises(archive):
    saved = checkpoint.save(small(len(CLASSES)), CLASSES, 32, archive_dir=archive)
    _, payload = checkpoint.load_file(saved)
    with pytest.raises(checkpoint.CheckpointError):
        checkpoint.require_classes(payload, [c for c in CLASSES if c != "hook"], "testData/")


def test_reordered_class_list_raises(archive):
    saved = checkpoint.save(small(len(CLASSES)), CLASSES, 32, archive_dir=archive)
    _, payload = checkpoint.load_file(saved)
    with pytest.raises(checkpoint.CheckpointError):
        checkpoint.require_classes(payload, list(reversed(CLASSES)), "dataSet/")


def test_model_width_comes_from_the_stored_list(archive):
    saved = checkpoint.save(small(3), ["a", "b", "c"], 32, archive_dir=archive)
    net, _ = checkpoint.load_file(saved)
    assert net(torch.zeros(1, features.NUM_FEATURES, 32)).shape == (1, 3)


def test_a_different_window_raises(archive):
    saved = checkpoint.save(small(3), ["a", "b", "c"], 32, archive_dir=archive)
    _, payload = checkpoint.load_file(saved)
    checkpoint.require_preprocessing(payload, 32)
    with pytest.raises(checkpoint.CheckpointError):
        checkpoint.require_preprocessing(payload, 64)


# ---- the running folder -----------------------------------------------------

def test_running_folder_with_no_classifier_says_how_to_promote(running):
    with pytest.raises(checkpoint.CheckpointError) as excinfo:
        checkpoint.running_classifier(running)
    assert "promote" in str(excinfo.value).lower()


def test_the_pose_model_is_not_mistaken_for_a_classifier(archive, running):
    """Stage 1 shares the folder and the .pt suffix, but it is not the classifier."""
    (running / "yolo26s-pose.pt").write_bytes(b"stage 1")
    with pytest.raises(checkpoint.CheckpointError):
        checkpoint.running_classifier(running)

    checkpoint.save(small(3), ["a", "b", "c"], 32, archive_dir=archive)
    checkpoint.promote(1, archive_dir=archive, running_path=running / "classifier.pt")
    assert checkpoint.running_classifier(running).name == "classifier.pt"


def test_two_classifiers_raise_rather_than_guessing(running):
    (running / "classifier.pt").write_bytes(b"a")
    (running / "classifier_v9.pt").write_bytes(b"b")
    with pytest.raises(checkpoint.CheckpointError):
        checkpoint.running_classifier(running)


def test_promote_copies_and_leaves_the_archive_intact(archive, running):
    saved = checkpoint.save(small(len(CLASSES)), CLASSES, 32, archive_dir=archive)
    target = running / "classifier.pt"

    checkpoint.promote(1, archive_dir=archive, running_path=target)

    assert saved.exists(), "promote must copy, never move"
    assert target.read_bytes() == saved.read_bytes()

    _, payload = checkpoint.load_file(target)
    assert payload["archive_name"] == "classifier_v1.pt"    # traceable despite the rename


def test_rollback_is_the_same_copy_with_an_older_version(archive, running):
    checkpoint.save(small(3), ["a", "b", "c"], 32, archive_dir=archive)
    checkpoint.save(small(3), ["a", "b", "c"], 32, archive_dir=archive)
    target = running / "classifier.pt"

    checkpoint.promote(2, archive_dir=archive, running_path=target)
    assert checkpoint.load_file(target)[1]["version"] == 2

    checkpoint.promote(1, archive_dir=archive, running_path=target)
    assert checkpoint.load_file(target)[1]["version"] == 1


def test_promoting_a_version_that_does_not_exist_raises(archive, running):
    with pytest.raises(checkpoint.CheckpointError):
        checkpoint.promote(7, archive_dir=archive, running_path=running / "classifier.pt")
