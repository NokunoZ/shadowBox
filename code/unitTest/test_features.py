"""Normalisation and windowing -- the preprocessing both sides of the project share."""

from __future__ import annotations

import numpy as np
import pytest

from common import features


def test_normalise_centres_on_the_hips(skeleton):
    out = features.normalise(skeleton(frames=4))
    hip_mid = (out[:, features.LEFT_HIP, :2] + out[:, features.RIGHT_HIP, :2]) / 2
    assert np.allclose(hip_mid, 0.0, atol=1e-5)


def test_normalise_scales_by_torso_length(skeleton):
    out = features.normalise(skeleton(frames=4))
    shoulder_mid = (out[:, features.LEFT_SHOULDER, :2] + out[:, features.RIGHT_SHOULDER, :2]) / 2
    # torso runs from the hip centre (now the origin) to the shoulder centre, so its
    # length is 1 by construction after scaling
    assert np.allclose(np.linalg.norm(shoulder_mid, axis=-1), 1.0, atol=1e-5)


def test_same_pose_shifted_and_resized_normalises_identically(skeleton):
    """The whole point: the model must learn the move, not where the boxer stood."""
    plain = features.normalise(skeleton(frames=6))
    moved = features.normalise(skeleton(frames=6, offset=(300.0, -120.0), scale=2.5))
    assert np.allclose(plain, moved, atol=1e-4)


def test_confidence_is_carried_through_untouched(skeleton):
    clip = skeleton(frames=3)
    clip[:, 4, 2] = 0.1                      # an unreliable joint
    out = features.normalise(clip)
    assert np.allclose(out[:, :, 2], clip[:, :, 2])


def test_normalise_survives_a_frame_with_no_visible_torso(skeleton):
    clip = skeleton(frames=5)
    clip[2, [5, 6, 11, 12], :2] = 0.0        # torso collapses to a point on one frame
    out = features.normalise(clip)
    assert np.isfinite(out).all()


def test_normalise_rejects_the_wrong_shape():
    with pytest.raises(ValueError):
        features.normalise(np.zeros((4, 12, 3), dtype=np.float32))


@pytest.mark.parametrize("frames", [1, 5, 32, 200])
def test_windowing_always_produces_T_frames(skeleton, frames):
    """A clip shorter than T and a clip longer than T both come out at length T."""
    out = features.window(skeleton(frames=frames), length=32)
    assert out.shape == (32, features.NUM_KEYPOINTS, 3)


def test_windowing_keeps_the_first_and_last_frame(skeleton):
    clip = skeleton(frames=50)
    out = features.window(clip, length=10)
    assert np.allclose(out[0], clip[0])
    assert np.allclose(out[-1], clip[-1])


def test_windowing_an_empty_clip_raises():
    with pytest.raises(ValueError):
        features.window(np.zeros((0, 17, 3), dtype=np.float32), length=8)


def test_prepare_produces_channels_first(skeleton):
    out = features.prepare(skeleton(frames=17), length=32)
    assert out.shape == (features.NUM_FEATURES, 32)
    assert out.dtype == np.float32


def test_prepare_is_the_same_path_for_any_caller(skeleton):
    """Training and inference must not be able to drift apart."""
    clip = skeleton(frames=40)
    by_hand = features.to_features(features.window(features.normalise(clip), 24))
    assert np.allclose(features.prepare(clip, 24), by_hand)
