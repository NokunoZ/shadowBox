"""The architecture must not care how many classes there are, or how long T is."""

from __future__ import annotations

import pytest
import torch

from common import features
from common import model as M


@pytest.mark.parametrize("num_classes", [2, 6, 7, 20])
def test_output_width_follows_the_class_count(num_classes):
    """Catches a hardcoded 6 in the architecture."""
    net = M.TemporalCNN(num_classes)
    out = net(torch.zeros(3, features.NUM_FEATURES, 32))
    assert out.shape == (3, num_classes)


def test_one_class_is_refused():
    with pytest.raises(ValueError):
        M.TemporalCNN(1)


@pytest.mark.parametrize("window", [16, 32, 64])
def test_window_length_does_not_reshape_the_head(window):
    """Global pooling over time, so changing T is not an architecture change."""
    net = M.TemporalCNN(6)
    assert net(torch.zeros(2, features.NUM_FEATURES, window)).shape == (2, 6)


def test_build_reconstructs_from_a_stored_config():
    net = M.TemporalCNN(6, channels=(32, 64), kernel_size=3, dropout=0.1)
    rebuilt = M.build(6, **net.config)          # config carries extra bookkeeping keys
    assert rebuilt.config == net.config
    rebuilt.load_state_dict(net.state_dict())   # identical shapes, so this cannot fail


def test_config_records_what_the_checkpoint_needs():
    net = M.TemporalCNN(6)
    for key in ("arch", "arch_version", "in_channels", "channels", "kernel_size", "dropout"):
        assert key in net.config


def test_runs_on_cpu_without_a_gpu():
    """Tests must not need a GPU."""
    net = M.TemporalCNN(4).to(M.pick_device(prefer_gpu=False))
    net.eval()
    with torch.no_grad():
        out = net(torch.randn(1, features.NUM_FEATURES, 32))
    assert torch.isfinite(out).all()
