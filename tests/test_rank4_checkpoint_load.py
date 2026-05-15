"""Verify the rank-4 ``Generator`` loads the pretrained ``hexgrad/Kokoro-82M``
checkpoint without missing or unexpected keys, courtesy of the
``register_load_state_dict_pre_hook`` that reshapes rank-3 conv weights and
rank-3 alpha parameters to rank-4 idempotently.

This is the checkpoint-compat gate for the rank-3 → rank-4 rewrite. See
``README/Plans/ane-decoder-har-rank4-rewrite-v1.md`` Phase 1 and the shared
``kokoro.istftnet._rank3_to_rank4_conv_state_dict`` helper.

The kokoro checkpoint is at
``~/.cache/huggingface/hub/models--hexgrad--Kokoro-82M/snapshots/*/kokoro-v1_0.pth``
— test is skipped if the snapshot isn't present (CI without HF cache).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from kokoro.istftnet import Generator


def _kokoro_checkpoint_path() -> Path | None:
    base = Path.home() / ".cache" / "huggingface" / "hub" / "models--hexgrad--Kokoro-82M" / "snapshots"
    if not base.is_dir():
        return None
    for snap in sorted(base.iterdir()):
        ckpt = snap / "kokoro-v1_0.pth"
        if ckpt.is_file():
            return ckpt
    return None


def _build_rank4_generator() -> Generator:
    """Match the istftnet hyperparameters baked into the kokoro checkpoint."""
    return Generator(
        style_dim=128,
        resblock_kernel_sizes=[3, 7, 11],
        upsample_rates=[10, 6],
        upsample_initial_channel=512,
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_kernel_sizes=[20, 12],
        gen_istft_n_fft=20,
        gen_istft_hop_size=5,
        disable_complex=True,
    )


@pytest.mark.skipif(
    _kokoro_checkpoint_path() is None,
    reason="hexgrad/Kokoro-82M checkpoint not cached locally",
)
def test_rank4_generator_loads_pretrained_checkpoint_without_missing_or_unexpected():
    ckpt_path = _kokoro_checkpoint_path()
    sd = torch.load(str(ckpt_path), weights_only=True, map_location="cpu")
    # Checkpoint structure: top-level "decoder" -> module-prefixed flat dict.
    dec = sd["decoder"]
    generator_sd = {
        k.removeprefix("module.generator."): v
        for k, v in dec.items()
        if k.startswith("module.generator.")
    }
    assert generator_sd, "no generator.* keys found in checkpoint"

    gen = _build_rank4_generator()
    result = gen.load_state_dict(generator_sd, strict=False)

    # ``stft.*`` buffers (``window`` / ``weight_forward_real`` etc.) are
    # initialised at module construction from ``torch.hann_window`` and the
    # DFT basis; they are NOT in the kokoro checkpoint and never have been.
    # Filter them out so the test only flags rank-4-rewrite regressions
    # (missing conv weights or alpha parameters).
    learnable_missing = [k for k in result.missing_keys if not k.startswith("stft.")]

    # The pre-hook on Generator + each AdaINResBlock1 should reshape every
    # rank-3 conv weight / alpha to rank-4 in place. After that, the loader
    # finds an exact key match for the learnable parameters.
    assert not learnable_missing, (
        f"rank-4 Generator expected learnable keys not present in kokoro checkpoint: "
        f"{learnable_missing}"
    )
    assert not result.unexpected_keys, (
        f"kokoro checkpoint has keys the rank-4 Generator doesn't expect: "
        f"{result.unexpected_keys}"
    )


@pytest.mark.skipif(
    _kokoro_checkpoint_path() is None,
    reason="hexgrad/Kokoro-82M checkpoint not cached locally",
)
def test_rank4_generator_forward_after_load_is_finite():
    ckpt_path = _kokoro_checkpoint_path()
    sd = torch.load(str(ckpt_path), weights_only=True, map_location="cpu")
    dec = sd["decoder"]
    generator_sd = {
        k.removeprefix("module.generator."): v
        for k, v in dec.items()
        if k.startswith("module.generator.")
    }

    gen = _build_rank4_generator()
    gen.load_state_dict(generator_sd, strict=False)
    gen.eval()

    # Small synthetic batch — small magnitudes match the export script's
    # numeric gate inputs to keep AdaIN's mean/var stable.
    torch.manual_seed(0)
    B = 1
    T_asr = 50
    style_dim = 128
    x = torch.clamp(torch.randn(B, 512, T_asr, dtype=torch.float32) * 0.02, -0.05, 0.05)
    s = torch.randn(B, style_dim, dtype=torch.float32) * 0.01
    # f0_upsamp scale = math.prod([10, 6]) * 5 = 300; pick a length that
    # multiplies cleanly so transposed conv shapes line up.
    f0 = torch.zeros(B, T_asr * 2, dtype=torch.float32)

    with torch.no_grad():
        wave = gen(x, s, f0)

    assert torch.isfinite(wave).all(), "rank-4 Generator output is non-finite"
    # iSTFT returns rank-3 (B, 1, samples); cope with either layout.
    assert wave.dim() in (2, 3), f"unexpected output rank {wave.dim()}"
