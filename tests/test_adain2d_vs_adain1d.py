"""Verify ``AdaIN2d`` is numerically equivalent to ``AdaIN1d`` when fed the
unsqueezed rank-4 form of the same input.

This is the PyTorch-level parity gate for the rank-3 → rank-4 rewrite at
``kokoro/istftnet.py::AdaIN2d`` (see README/Plans/ane-decoder-har-rank4-rewrite-v1.md
Phase 1). ``AdaIN1d`` is intentionally untouched in this plan — it stays
rank-3 for the Decoder path's ``AdainResBlk1d``. ``AdaIN2d`` is its rank-4
sibling for the Generator path.

Tolerance is tight (atol=1e-5 / rtol=1e-5) because the math is the same and
both paths run in fp32 on the CPU here — any drift would indicate a bug in
the rank-4 reduction or broadcast.
"""

from __future__ import annotations

import pytest
import torch

from kokoro.istftnet import AdaIN1d, AdaIN2d


@pytest.mark.parametrize(
    "batch,channels,style_dim,seq_len",
    [
        (1, 128, 128, 64),
        (1, 256, 128, 128),
        (1, 128, 64, 32),
        # B>1 case — guards against any per-batch broadcast / view mistake in
        # AdaIN2d's `fc(s).view(B, 2C, 1, 1)` step, since per-channel mean/var
        # is computed independently per batch element.
        (2, 128, 128, 64),
    ],
)
def test_adain2d_matches_adain1d_on_unsqueezed_input(batch, channels, style_dim, seq_len):
    torch.manual_seed(0)

    a1 = AdaIN1d(style_dim, channels).eval()
    a2 = AdaIN2d(style_dim, channels).eval()

    # Copy fc weights so a1 and a2 compute identical (gamma, beta) from the
    # same style vector. Random init would otherwise diverge.
    with torch.no_grad():
        a2.fc.weight.copy_(a1.fc.weight)
        a2.fc.bias.copy_(a1.fc.bias)

    x3 = torch.randn(batch, channels, seq_len)
    s = torch.randn(batch, style_dim)
    x4 = x3.unsqueeze(-2)  # (B, C, 1, T)

    with torch.no_grad():
        y3 = a1(x3, s)              # (B, C, T)
        y4 = a2(x4, s)              # (B, C, 1, T)
        y4_back = y4.squeeze(-2)    # (B, C, T)

    assert y3.shape == y4_back.shape
    max_abs_diff = (y3 - y4_back).abs().max().item()
    assert torch.allclose(y3, y4_back, atol=1e-5, rtol=1e-5), (
        f"AdaIN2d output diverges from AdaIN1d "
        f"(batch={batch}, channels={channels}, T={seq_len}): "
        f"max abs diff {max_abs_diff:.3e}"
    )


def test_adain2d_rejects_missing_h_axis():
    a2 = AdaIN2d(128, 64)
    x_bad = torch.randn(1, 64, 2, 32)  # H=2 instead of H=1
    s = torch.randn(1, 128)
    with pytest.raises(AssertionError, match="H=1"):
        a2(x_bad, s)


def test_adain2d_rejects_channel_mismatch():
    a2 = AdaIN2d(128, 64)
    x = torch.randn(1, 32, 1, 16)  # 32 channels, expects 64
    s = torch.randn(1, 128)
    with pytest.raises(AssertionError, match="channel mismatch"):
        a2(x, s)
