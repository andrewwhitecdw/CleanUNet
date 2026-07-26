"""Regression test for the high-band STFT loss axis bug.

With ``band="high"``, ``STFTLoss.forward()`` must compute the losses on the
upper half of the frequency bins only. Before the fix it sliced axis 1
(time frames) instead of the frequency axis, so the high-band loss was
computed over low frequencies and half the utterance.

Run with CPU torch:
    uv run --python 3.11 --with torch --with pytest \
        --index-url https://download.pytorch.org/whl/cpu \
        pytest tests/test_stft_high_band.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stft_loss import STFTLoss  # noqa: E402


class ShapeRecorder(torch.nn.Module):
    """Replacement sub-loss that records the shapes it receives."""

    def __init__(self):
        super().__init__()
        self.shapes = []

    def forward(self, x_mag, y_mag):
        self.shapes.append((tuple(x_mag.shape), tuple(y_mag.shape)))
        return torch.tensor(0.0)


def test_high_band_masks_frequency_axis_not_frames():
    fft_size = 64
    loss = STFTLoss(fft_size=fft_size, shift_size=16, win_length=32,
                    band="high")
    loss.spectral_convergence_loss = ShapeRecorder()
    loss.log_stft_magnitude_loss = ShapeRecorder()

    x = torch.randn(2, 1000)
    y = torch.randn(2, 1000)
    loss(x, y)

    n_frames = 1 + 1000 // 16  # 63 frames (center-padded STFT)
    n_bins = fft_size // 2 + 1              # 33 frequency bins
    high_bins = n_bins - n_bins // 2        # upper half: 17 bins

    for recorder in (loss.spectral_convergence_loss,
                     loss.log_stft_magnitude_loss):
        assert len(recorder.shapes) == 1
        x_shape, y_shape = recorder.shapes[0]
        assert x_shape == y_shape
        # Frames must be intact; only the frequency axis is halved.
        assert x_shape == (2, n_frames, high_bins), (
            f"expected (batch, {n_frames} frames, {high_bins} high-freq "
            f"bins), got {x_shape} — high-band mask hit the wrong axis"
        )
