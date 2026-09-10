"""The keypoint cache -- pose runs once per clip, not once per epoch or per re-split.

Only the cache KEY is tested here: running the pose model would need weights and a
GPU, which tests must not.
"""

from __future__ import annotations

import shutil

import pytest

from common.pose import _cache_key


@pytest.fixture
def clip_and_weights(tmp_path):
    clip = tmp_path / "dataSet" / "jab" / "kasi_s01__jab_001.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"pretend this is footage")
    weights = tmp_path / "yolo26s-pose.pt"
    weights.write_bytes(b"pretend these are weights")
    return clip, weights


def test_same_clip_through_a_split_folder_hits_the_same_entry(clip_and_weights, tmp_path):
    """dataSet/, trainData/ and testData/ all reach one clip. Keying on the path would
    recompute pose for each, and bin the cache every time dataSpliter runs."""
    clip, weights = clip_and_weights
    split = tmp_path / "trainData" / "jab"
    split.mkdir(parents=True)
    shutil.copy2(clip, split / clip.name)

    assert _cache_key(clip, weights) == _cache_key(split / clip.name, weights)


def test_a_different_clip_gets_a_different_entry(clip_and_weights, tmp_path):
    clip, weights = clip_and_weights
    other = clip.with_name("kasi_s01__jab_002.mp4")
    other.write_bytes(b"different footage entirely")
    assert _cache_key(clip, weights) != _cache_key(other, weights)


def test_edited_clip_invalidates_its_entry(clip_and_weights):
    clip, weights = clip_and_weights
    before = _cache_key(clip, weights)
    clip.write_bytes(b"a re-trimmed version of the same take")
    assert _cache_key(clip, weights) != before


def test_swapping_the_pose_model_invalidates_the_cache(clip_and_weights):
    """Otherwise training silently continues on keypoints from the old stage 1."""
    clip, weights = clip_and_weights
    before = _cache_key(clip, weights)
    weights.write_bytes(b"a different, larger pose model")
    assert _cache_key(clip, weights) != before
