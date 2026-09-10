"""The stage-2 classifier: a 1D temporal CNN over the keypoint sequence.

Why 1D and not an image CNN: stage 2 never sees pixels. Its input is a (51, T) block
-- 51 keypoint channels over T frames -- so the convolution slides along TIME, and
what it learns are motion patterns: how fast a wrist travels, when a hip rotates
relative to a shoulder. That is the shape of a boxing move.

Imported by both training and inference so there is exactly one copy of the
architecture. num_classes is an argument; no class count is baked in anywhere, so a
new folder in dataSet/ never reaches this file.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .features import NUM_FEATURES

ARCH_NAME = "TemporalCNN"
ARCH_VERSION = 1

DEFAULT_CHANNELS = (128, 256, 256)
DEFAULT_KERNEL = 5
DEFAULT_DROPOUT = 0.3


class TemporalCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int = NUM_FEATURES,
        channels: tuple[int, ...] = DEFAULT_CHANNELS,
        kernel_size: int = DEFAULT_KERNEL,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError(
                f"a classifier needs at least 2 classes, got {num_classes}. "
                "Check that dataSet/ has more than one non-empty class folder."
            )

        blocks: list[nn.Module] = []
        previous = in_channels
        for width in channels:
            blocks += [
                nn.Conv1d(previous, width, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(width),
                nn.ReLU(inplace=True),
                nn.Conv1d(width, width, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(width),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
            ]
            previous = width
        self.features = nn.Sequential(*blocks)

        # global pooling over what is left of the time axis, so the architecture does
        # not care what T is -- changing the window length does not reshape the head
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(previous, num_classes),
        )

        self.config = {
            "arch": ARCH_NAME,
            "arch_version": ARCH_VERSION,
            "in_channels": in_channels,
            "channels": list(channels),
            "kernel_size": kernel_size,
            "dropout": dropout,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, 51, T) -> (batch, num_classes) logits."""
        return self.head(self.pool(self.features(x)))


def build(num_classes: int, **config) -> TemporalCNN:
    """Construct from a config dict, dropping the bookkeeping keys a checkpoint adds."""
    known = {"in_channels", "channels", "kernel_size", "dropout"}
    kwargs = {k: v for k, v in config.items() if k in known}
    if "channels" in kwargs:
        kwargs["channels"] = tuple(kwargs["channels"])
    return TemporalCNN(num_classes, **kwargs)


def pick_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
