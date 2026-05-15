# Audio Synthesis and iSTFT-based Neural Vocoder Components
#
# This module implements the core audio generation pipeline for Kokoro TTS,
# featuring an iSTFT-based neural vocoder that synthesizes high-quality
# waveforms from intermediate acoustic features.
#
# Key Components:
# - Generator: Main neural vocoder with HiFi-GAN-style architecture
# - Decoder: High-level wrapper combining feature processing and generation
# - SourceModuleHnNSF: Harmonic/noise source modeling for F0-conditioned synthesis
# - AdaINResBlock1: Style-adaptive residual blocks for voice conditioning
# - TorchSTFT/CustomSTFT: STFT implementations with CoreML compatibility options
#
# Architecture Philosophy:
# - iSTFT-based synthesis for high-fidelity audio generation
# - Style-conditioned layers throughout for voice adaptation
# - Harmonic plus noise source modeling for natural speech characteristics
# - Multi-scale residual processing for rich spectral detail
#
# CoreML Export Considerations:
# - CustomSTFT used when disable_complex=True for ONNX/CoreML compatibility
# - TorchSTFT used for native PyTorch inference (higher quality)
# - Complex number operations avoided in CustomSTFT variant
#
# Cross-file dependencies:
# - Imports from: custom_stft.py (CustomSTFT for export compatibility)
# - Used by: model.py (KModel.decoder), modules.py (ProsodyPredictor components)
# - Based on: StyleTTS2 iSTFTNet with Kokoro-specific optimizations
#
# Performance Notes:
# - Optimized for 24kHz synthesis with 600-sample hop length
# - Multi-resolution processing for efficient high-quality generation
# - Style conditioning enables zero-shot voice cloning capabilities

# Adapted from StyleTTS2: https://github.com/yl4579/StyleTTS2/blob/main/Modules/istftnet.py

try:
    from kokoro.custom_stft import CustomSTFT  # normal import when used as a package
except Exception:
    # Fallback for local script loading without package context
    import importlib.util, pathlib
    _ROOT = pathlib.Path(__file__).resolve().parent
    _p = (_ROOT / 'custom_stft.py').resolve()
    _spec = importlib.util.spec_from_file_location('kokoro_custom_stft', _p)
    _mod = importlib.util.module_from_spec(_spec)
    assert _spec and _spec.loader
    _spec.loader.exec_module(_mod)
    CustomSTFT = _mod.CustomSTFT
from torch.nn.utils.parametrizations import weight_norm
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m, mean=0.0, std=0.01):
    # Initialize convolutional layer weights with normal distribution.
    #
    # This utility function provides consistent weight initialization
    # across all convolutional layers in the vocoder architecture.
    # Proper initialization is critical for stable training and convergence.
    #
    # Parameters:
    # - m: PyTorch module to initialize
    # - mean: Normal distribution mean (default: 0.0)
    # - std: Normal distribution standard deviation (default: 0.01)
    #
    # Applied to:
    # - All Conv1d layers in Generator architecture
    # - Ensures consistent initialization across the network
    #
    # From: StyleTTS2 utilities
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

def get_padding(kernel_size, dilation=1):
    # Calculate padding for 'same' convolution with dilation.
    #
    # This utility ensures that convolutional operations maintain
    # the input sequence length, which is essential for the iSTFT
    # generation pipeline where temporal alignment must be preserved.
    #
    # Parameters:
    # - kernel_size: Convolution kernel size
    # - dilation: Dilation factor for dilated convolutions
    #
    # Returns:
    # - int: Padding value for same-size output
    #
    # Used throughout:
    # - AdaINResBlock1: Dilated convolutions in residual blocks
    # - Generator: Upsampling and processing layers
    #
    return int((kernel_size*dilation - dilation)/2)


class AdaIN1d(nn.Module):
    # Adaptive Instance Normalization for 1D sequences with style conditioning.
    #
    # **Rank-3 contract.** This module consumes ``(B, C, T)`` tensors and is the
    # AdaIN used by ``AdainResBlk1d`` (Decoder path). It is **intentionally
    # distinct** from ``AdaIN2d`` (rank-4, Generator path) below; do **not**
    # merge or refactor the two into a shared class. The two paths export to
    # different Core ML packages with different ANE-alignment requirements —
    # the rank-3 Decoder path keeps AdaIN1d untouched as a frozen contract; the
    # rank-4 Generator path uses AdaIN2d. Merging them couples HAR-post and
    # decoder_pre rewrites in a way the rank-4 plan explicitly avoids (see
    # ``README/Plans/ane-decoder-har-rank4-rewrite-v1.md`` Open Questions §
    # "Should AdaIN1d be rewritten in place...?").
    #
    # This module implements style-conditioned normalization that adapts
    # the normalization statistics based on voice characteristics. It's a
    # key component enabling voice cloning and style transfer capabilities.
    #
    # Mathematical Operation:
    # 1. Instance normalization: x_norm = InstanceNorm(x)
    # 2. Style-dependent parameters: gamma, beta = Linear(style)
    # 3. Adaptive transformation: output = (1 + gamma) * x_norm + beta
    #
    # Manual channel-wise normalization (no nn.InstanceNorm1d) avoids exporter
    # shape/broadcast bugs and keeps the MIL graph clean for ANE tracing.
    #
    # Note: gamma/beta are explicitly ``.expand``ed across T (see forward
    # below). The rank-4 sibling ``AdaIN2d`` drops the explicit expand and
    # relies on implicit broadcasting — that change is **only safe at rank 4**
    # (it triggers Espresso "Shape computation issue" events at rank 3 on
    # macOS 26 ANE compilers; see the iteration-1 Phase 3 evidence in the
    # rank-4 plan). Keep the explicit expand here.
    #
    # Parameters:
    # - style_dim: Dimension of input style vector (typically 128)
    # - num_features: Number of channels to normalize
    #
    # Used by:
    # - AdainResBlk1d (kokoro/istftnet.py, Decoder path) — frozen rank-3
    #   contract.
    # - Generator (legacy, pre-rank-4): no longer used; ``AdaINResBlock1`` now
    #   uses ``AdaIN2d`` instead.
    #
    def __init__(self, style_dim, num_features):
        super().__init__()
        # Use manual channel-wise normalization to avoid exporter shape/broadcast bugs.
        # Keep num_features for gamma/beta projection sizing.
        self.num_features = num_features
        self.eps = 1e-5
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        # Apply adaptive instance normalization with style conditioning.
        # Manual per-channel normalization to keep shapes explicit for exporters.
        # x: (B, C, T), s: (B, style_dim)
        B, C, T = x.shape
        # Compute channel-wise mean/var over time
        mean = x.mean(dim=2, keepdim=True)
        var = x.var(dim=2, unbiased=False, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Project style to gamma/beta: (B, 2C) -> (B, C, 1)
        # AdaIN1d is always constructed with num_features == channels for the
        # convs it normalizes (AdaINResBlock1, AdainResBlk1d), so C == num_features
        # at runtime.  The old slice/pad branch (torch.cat with zeros) was dead code
        # and would have poisoned the ANE trace with a banned concat op (Orion #1).
        assert C == self.num_features, f"AdaIN1d channel mismatch: got {C}, expected {self.num_features}"
        h = self.fc(s).view(B, 2 * self.num_features, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        # Expand across time to avoid implicit broadcasting pitfalls
        gamma_exp = gamma.expand(B, C, T)
        beta_exp = beta.expand(B, C, T)
        return (1.0 + gamma_exp) * x_norm + beta_exp


def rank3_to_rank4_conv_state_dict(state_dict, prefix, conv_module_paths, alpha_param_paths):
    """Idempotently reshape rank-3 conv weights and rank-3 alpha params to rank-4
    inside a state_dict, supporting bare and weight_norm-wrapped (legacy and new
    parametrizations API) forms.

    Used by ``AdaINResBlock1`` and ``Generator`` ``register_load_state_dict_pre_hook``
    so the pretrained rank-3 ``hexgrad/Kokoro-82M`` checkpoint loads into the
    rank-4 modules introduced for ANE alignment (see
    ``README/Plans/ane-decoder-har-rank4-rewrite-v1.md``).

    Designed as a public cross-module utility: the follow-on ``decoder_pre``
    rank-4 plan (``ane-decoder-pre-rank4-rewrite-v1.md``) will call it from
    ``AdainResBlk1d``'s load hook with its own key lists. The name is
    intentionally unprefixed so cross-module callers don't have to reach across
    a single-underscore boundary.

    Args:
        state_dict: dict being loaded (mutated in place).
        prefix: module prefix supplied by PyTorch's load hook (e.g.
            ``"generator.resblocks.0."``).
        conv_module_paths: iterable of relative module paths for conv layers
            (e.g. ``"convs1.0"``, ``"ups.0"``). For each, ALL weight-tensor
            sub-keys are reshaped if rank-3:
                - ``<path>.weight``               (bare Conv1d/ConvTranspose1d)
                - ``<path>.weight_g``             (legacy weight_norm magnitude)
                - ``<path>.weight_v``             (legacy weight_norm direction)
                - ``<path>.parametrizations.weight.original0`` (new API magnitude)
                - ``<path>.parametrizations.weight.original1`` (new API direction)
            A 3D tensor becomes 4D via ``.unsqueeze(-2)`` (inserts H=1 just before
            the kernel/time axis). 4D tensors pass through unchanged.
        alpha_param_paths: iterable of relative parameter paths for the Snake1D
            alpha parameters (e.g. ``"alpha1.0"``). 3D ``(1, C, 1)`` becomes 4D
            ``(1, C, 1, 1)`` via ``.unsqueeze(-1)``; 4D tensors pass through.

    Raises:
        AssertionError: if any matched tensor has a rank other than 3 or 4
            (the only two ranks this helper is intended to handle). Catching
            rank-2 or rank-5 surprises here gives a clear diagnostic; the
            alternative is a confusing shape mismatch deep inside
            ``load_state_dict``.
    """
    weight_suffixes = (
        "weight",
        "weight_g",
        "weight_v",
        "parametrizations.weight.original0",
        "parametrizations.weight.original1",
    )
    for module_path in conv_module_paths:
        for weight_suffix in weight_suffixes:
            key = f"{prefix}{module_path}.{weight_suffix}"
            t = state_dict.get(key)
            if t is None:
                continue
            assert t.dim() in (3, 4), (
                f"rank3_to_rank4_conv_state_dict: expected rank 3 or 4 for {key!r}, "
                f"got rank {t.dim()} shape {tuple(t.shape)}"
            )
            if t.dim() == 3:
                state_dict[key] = t.unsqueeze(-2)
    for param_path in alpha_param_paths:
        key = f"{prefix}{param_path}"
        t = state_dict.get(key)
        if t is None:
            continue
        assert t.dim() in (3, 4), (
            f"rank3_to_rank4_conv_state_dict: expected rank 3 or 4 for alpha param "
            f"{key!r}, got rank {t.dim()} shape {tuple(t.shape)}"
        )
        if t.dim() == 3:
            state_dict[key] = t.unsqueeze(-1)


class AdaIN2d(nn.Module):
    # Rank-4 AdaIN used by the Generator's ANE-aligned rewrite.
    #
    # **Rank-4 contract.** Consumes ``(B, C, 1, T)`` tensors so the surrounding
    # Conv2d / ConvTranspose2d stack stays rank-4 throughout — Apple's ANE
    # prefers last-axis-largest rank-4 layouts (see CLAUDE.md Part 4.1).
    #
    # **Intentionally distinct from ``AdaIN1d`` above.** Same math (manual
    # instance norm over time + style-conditioned gamma/beta) but operating
    # on a different rank. Do **not** merge or refactor the two into a
    # shared class. ``AdaIN1d`` is frozen as the rank-3 Decoder contract;
    # this class is the rank-4 Generator replacement. The 95% code
    # similarity is acknowledged and intentional — see the corresponding
    # warning on ``AdaIN1d`` and Risks and Mitigations in the rank-4 plan
    # (``README/Plans/ane-decoder-har-rank4-rewrite-v1.md``).
    #
    # The decoder_pre follow-on plan will reuse this same ``AdaIN2d`` (rank-4)
    # rather than introduce a parallel rank-4 AdaIN; ``AdaIN1d`` becomes dead
    # code at that point and can be removed.
    #
    # **Note vs ``AdaIN1d``:** AdaIN1d explicitly ``.expand``s gamma/beta over
    # T to avoid Espresso "Shape computation issue" events on the rank-3
    # graph. At rank 4 the explicit expand is not needed — PyTorch broadcast
    # of ``(B, C, 1, 1)`` against ``(B, C, 1, T)`` does not trigger the same
    # ANE shape-inference pitfall (verified by Phase 3 of the rank-4 plan:
    # 12 ``Shape computation issue`` events on rank 3 → 0 on rank 4). Keep
    # the implicit broadcast here; do not copy the explicit expand back.
    #
    # Parameters:
    # - style_dim:    Dimension of input style vector (typically 128).
    # - num_features: Number of channels to normalize (C).
    #
    # Forward I/O:
    # - x: (B, C, 1, T)
    # - s: (B, style_dim)
    # - returns: (B, C, 1, T)

    def __init__(self, style_dim, num_features):
        super().__init__()
        self.num_features = num_features
        self.eps = 1e-5
        # Linear projection to (gamma, beta). Identical fc.weight shape to
        # AdaIN1d (2C, style_dim), so checkpoints with AdaIN1d-shaped fc load
        # without translation.
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        # x: (B, C, 1, T), s: (B, style_dim)
        B, C, H, T = x.shape
        assert H == 1, f"AdaIN2d expects H=1 axis, got H={H}"
        assert C == self.num_features, f"AdaIN2d channel mismatch: got {C}, expected {self.num_features}"

        # Reduce over the time axis only; keep_dim broadcasts back over T.
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Project to (gamma, beta) — reshape to (B, 2C, 1, 1) so broadcast
        # against rank-4 x is implicit (no torch.expand call needed; explicit
        # expand+reshape sequences were what triggered the rank-3 graph's
        # Shape computation issue events on ANE).
        h = self.fc(s).view(B, 2 * self.num_features, 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1.0 + gamma) * x_norm + beta


class AdaINResBlock1(nn.Module):
    # Rank-4 residual block used by the Generator. Operates on (B, C, 1, T)
    # tensors so the surrounding Generator stack stays ANE-aligned end to end
    # (see README/Plans/ane-decoder-har-rank4-rewrite-v1.md).
    #
    # Conv1d → Conv2d((1, k), dilation=(1, d), padding=(0, p)). Snake1D alpha
    # parameters are stored as (1, C, 1, 1). AdaIN normalization uses the
    # rank-4 ``AdaIN2d`` sibling — ``AdaIN1d`` (rank-3) is kept untouched for
    # the Decoder path's ``AdainResBlk1d``.
    #
    # A ``register_load_state_dict_pre_hook`` reshapes rank-3
    # ``hexgrad/Kokoro-82M`` checkpoint tensors to rank-4 idempotently so
    # pretrained weights load without retraining. See
    # ``rank3_to_rank4_conv_state_dict`` for the shared reshape utility.

    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), style_dim=64):
        super(AdaINResBlock1, self).__init__()
        # convs1: kernel_size, varying dilation per resblock entry.
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv2d(
                channels, channels, (1, kernel_size), (1, 1),
                dilation=(1, d),
                padding=(0, get_padding(kernel_size, d)),
            ))
            for d in dilation
        ])
        self.convs1.apply(init_weights)
        # convs2: dilation=1 always, one Conv2d per convs1 entry.
        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv2d(
                channels, channels, (1, kernel_size), (1, 1),
                dilation=(1, 1),
                padding=(0, get_padding(kernel_size, 1)),
            ))
            for _ in dilation
        ])
        self.convs2.apply(init_weights)
        self.adain1 = nn.ModuleList(
            AdaIN2d(style_dim, channels) for _ in dilation
        )
        self.adain2 = nn.ModuleList(
            AdaIN2d(style_dim, channels) for _ in dilation
        )
        self.alpha1 = nn.ParameterList(
            [nn.Parameter(torch.ones(1, channels, 1, 1)) for _ in dilation]
        )
        self.alpha2 = nn.ParameterList(
            [nn.Parameter(torch.ones(1, channels, 1, 1)) for _ in dilation]
        )

        self._register_load_state_dict_pre_hook(self._reshape_rank3_to_rank4_hook)

    @staticmethod
    def _reshape_rank3_to_rank4_hook(state_dict, prefix, *_args, **_kwargs):
        # Pretrained kokoro-v1_0.pth was saved with Conv1d weights ((C,C,k))
        # and Snake1D alpha ((1,C,1)). Reshape to Conv2d ((C,C,1,k)) and
        # (1,C,1,1) idempotently so checkpoints load without retraining. The
        # 3-element list lengths match the ``dilation`` triple in __init__
        # (kokoro uses dilation=(1,3,5)); these hard-coded indices are
        # AdaINResBlock1's contract with kokoro-v1_0.pth.
        conv_module_paths = [f"convs1.{i}" for i in range(3)] + [f"convs2.{i}" for i in range(3)]
        alpha_param_paths = [f"alpha1.{i}" for i in range(3)] + [f"alpha2.{i}" for i in range(3)]
        rank3_to_rank4_conv_state_dict(state_dict, prefix, conv_module_paths, alpha_param_paths)

    def forward(self, x, s):
        # x: (B, C, 1, T), s: (B, style_dim) — returns (B, C, 1, T)
        for c1, c2, n1, n2, a1, a2 in zip(self.convs1, self.convs2, self.adain1, self.adain2, self.alpha1, self.alpha2):
            xt = n1(x, s)
            xt = xt + (1 / a1) * (torch.sin(a1 * xt) ** 2)  # Snake1D
            xt = c1(xt)
            xt = n2(xt, s)
            xt = xt + (1 / a2) * (torch.sin(a2 * xt) ** 2)  # Snake1D
            xt = c2(xt)
            x = xt + x
        return x


class TorchSTFT(nn.Module):
    def __init__(self, filter_length=800, hop_length=200, win_length=800, window='hann'):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        assert window == 'hann', window
        self.window = torch.hann_window(win_length, periodic=True, dtype=torch.float32)

    def transform(self, input_data):
        forward_transform = torch.stft(
            input_data,
            self.filter_length, self.hop_length, self.win_length, window=self.window.to(input_data.device),
            return_complex=True)
        return torch.abs(forward_transform), torch.angle(forward_transform)

    def inverse(self, magnitude, phase):
        inverse_transform = torch.istft(
            magnitude * torch.exp(phase * 1j),
            self.filter_length, self.hop_length, self.win_length, window=self.window.to(magnitude.device))
        return inverse_transform.unsqueeze(-2)  # unsqueeze to stay consistent with conv_transpose1d implementation

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction


class SineGen(nn.Module):
    """ Definition of sine generator
    SineGen(samp_rate, harmonic_num = 0,
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0,
            flag_for_pulse=False)
    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-wavefrom (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_thoreshold: F0 threshold for U/V classification (default 0)
    flag_for_pulse: this SinGen is used inside PulseGen (default False)
    Note: when flag_for_pulse is True, the first time step of a voiced
        segment is always sin(torch.pi) or cos(0)
    """
    def __init__(self, samp_rate, upsample_scale, harmonic_num=0,
                 sine_amp=0.1, noise_std=0.003,
                 voiced_threshold=0,
                 flag_for_pulse=False):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse
        self.upsample_scale = upsample_scale

    def _f02uv(self, f0):
        # generate uv signal
        uv = (f0 > self.voiced_threshold).to(f0.dtype)
        return uv

    def _f02sine(self, f0_values):
        """ f0_values: (batchsize, length, dim)
            where dim indicates fundamental tone and overtones
        """
        # convert to F0 in rad. The interger part n can be ignored
        # because 2 * torch.pi * n doesn't affect phase
        rad_values = (f0_values / self.sampling_rate) % 1
        # initial phase noise (no noise for fundamental component)
        rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], device=f0_values.device)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
        # instantanouse phase sine[t] = sin(2*pi \sum_i=1 ^{t} rad)
        if not self.flag_for_pulse:
            # Avoid zero-length downsampling when scale_factor < 1 by using explicit sizes
            # target length after downsample must be at least 1
            B, L, D = rad_values.shape
            down_len = max(1, int((L + self.upsample_scale - 1) // self.upsample_scale))
            # Downsample by specifying size instead of fractional scale to prevent floor-to-zero
            rad_values_ds = F.interpolate(
                rad_values.transpose(1, 2), size=down_len, mode="linear"
            ).transpose(1, 2)
            phase = torch.cumsum(rad_values_ds, dim=1) * 2 * torch.pi
            up_len = down_len * self.upsample_scale
            phase_up = F.interpolate(
                (phase.transpose(1, 2) * self.upsample_scale), size=up_len, mode="linear"
            ).transpose(1, 2)
            sines = torch.sin(phase_up)
        else:
            # If necessary, make sure that the first time step of every
            # voiced segments is sin(pi) or cos(0)
            # This is used for pulse-train generation
            # identify the last time step in unvoiced segments
            uv = self._f02uv(f0_values)
            uv_1 = torch.roll(uv, shifts=-1, dims=1)
            uv_1[:, -1, :] = 1
            u_loc = (uv < 1) * (uv_1 > 0)
            # get the instantanouse phase
            tmp_cumsum = torch.cumsum(rad_values, dim=1)
            # different batch needs to be processed differently
            for idx in range(f0_values.shape[0]):
                temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
                temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
                # stores the accumulation of i.phase within
                # each voiced segments
                tmp_cumsum[idx, :, :] = 0
                tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum
            # rad_values - tmp_cumsum: remove the accumulation of i.phase
            # within the previous voiced segment.
            i_phase = torch.cumsum(rad_values - tmp_cumsum, dim=1)
            # get the sines
            sines = torch.cos(i_phase * 2 * torch.pi)
        return sines

    def forward(self, f0):
        """ sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
                  f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """
        f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
        # fundamental component
        # Build harmonics without advanced broadcasting ops to aid CoreML conversion
        harmonics = []
        for i in range(self.harmonic_num + 1):
            coef = torch.tensor(float(i + 1), dtype=f0.dtype, device=f0.device)
            harmonics.append(f0 * coef)
        fn = torch.cat(harmonics, dim=2)
        # generate sine waveforms
        sine_waves = self._f02sine(fn) * self.sine_amp
        # generate uv signal
        # uv = torch.ones(f0.shape)
        # uv = uv * (f0 > self.voiced_threshold)
        uv = self._f02uv(f0)
        # noise: for unvoiced should be similar to sine_amp
        #        std = self.sine_amp/3 -> max value ~ self.sine_amp
        #        for voiced regions is self.noise_std
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)
        # first: set the unvoiced part to 0 by uv
        # then: additive noise
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class SourceModuleHnNSF(nn.Module):
    """ SourceModule for hn-nsf
    SourceModule(sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0)
    sampling_rate: sampling_rate in Hz
    harmonic_num: number of harmonic above F0 (default: 0)
    sine_amp: amplitude of sine source signal (default: 0.1)
    add_noise_std: std of additive Gaussian noise (default: 0.003)
        note that amplitude of noise in unvoiced is decided
        by sine_amp
    voiced_threshold: threhold to set U/V given F0 (default: 0)
    Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
    F0_sampled (batchsize, length, 1)
    Sine_source (batchsize, length, 1)
    noise_source (batchsize, length 1)
    uv (batchsize, length, 1)
    """
    def __init__(self, sampling_rate, upsample_scale, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0):
        super(SourceModuleHnNSF, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        # to produce sine waveforms
        self.l_sin_gen = SineGen(sampling_rate, upsample_scale, harmonic_num,
                                 sine_amp, add_noise_std, voiced_threshod)
        # to merge source harmonics into a single excitation
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, x):
        """
        Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
        F0_sampled (batchsize, length, 1)
        Sine_source (batchsize, length, 1)
        noise_source (batchsize, length 1)
        """
        # source for harmonic branch
        with torch.no_grad():
            sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        # source for noise branch, in the same shape as uv
        noise = torch.randn_like(uv) * self.sine_amp / 3
        return sine_merge, noise, uv


class Generator(nn.Module):
    # Rank-4 vocoder Generator. The internal convolution / AdaIN stack operates
    # on ``(B, C, 1, T)`` tensors so the entire body is ANE-aligned (see
    # README/Plans/ane-decoder-har-rank4-rewrite-v1.md and CLAUDE.md Part 4.1
    # on the last-axis-largest rule). The public ``forward(x, s, f0)`` signature
    # still consumes and returns **rank-3** tensors at the module boundary —
    # the rank promotion happens internally so callers like
    # ``Decoder.forward`` (kokoro/istftnet.py::Decoder) and any other rank-3
    # consumers are unchanged. ``GeneratorFromHar`` (export_synth/wrappers.py)
    # is the inference entry point that the CoreML export traces; it mirrors
    # this same rank-3 → rank-4 → rank-3 boundary pattern.
    #
    # ``noise_convs``, ``ups``, ``conv_post``, ``reflection_pad`` are rank-4
    # versions of their Conv1d / ConvTranspose1d / ReflectionPad1d counterparts.
    # ``resblocks`` and ``noise_res`` hold rank-4 ``AdaINResBlock1`` instances
    # which in turn use rank-4 ``AdaIN2d``.
    #
    # A ``register_load_state_dict_pre_hook`` reshapes the pretrained kokoro
    # checkpoint's rank-3 weights to rank-4 idempotently (see
    # ``rank3_to_rank4_conv_state_dict``).

    def __init__(self, style_dim, resblock_kernel_sizes, upsample_rates, upsample_initial_channel, resblock_dilation_sizes, upsample_kernel_sizes, gen_istft_n_fft, gen_istft_hop_size, disable_complex=False):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.m_source = SourceModuleHnNSF(
                    sampling_rate=24000,
                    upsample_scale=math.prod(upsample_rates) * gen_istft_hop_size,
                    harmonic_num=8, voiced_threshod=10)
        self.f0_upsamp = nn.Upsample(scale_factor=math.prod(upsample_rates) * gen_istft_hop_size)
        self.noise_convs = nn.ModuleList()
        self.noise_res = nn.ModuleList()
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                nn.ConvTranspose2d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                   (1, k), (1, u), padding=(0, (k-u)//2))))
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes,resblock_dilation_sizes)):
                self.resblocks.append(AdaINResBlock1(ch, k, d, style_dim))
            c_cur = upsample_initial_channel // (2 ** (i + 1))
            if i + 1 < len(upsample_rates):
                stride_f0 = math.prod(upsample_rates[i + 1:])
                self.noise_convs.append(nn.Conv2d(
                    gen_istft_n_fft + 2, c_cur,
                    kernel_size=(1, stride_f0 * 2),
                    stride=(1, stride_f0),
                    padding=(0, (stride_f0+1) // 2)))
                self.noise_res.append(AdaINResBlock1(c_cur, 7, [1,3,5], style_dim))
            else:
                self.noise_convs.append(nn.Conv2d(gen_istft_n_fft + 2, c_cur, kernel_size=(1, 1)))
                self.noise_res.append(AdaINResBlock1(c_cur, 11, [1,3,5], style_dim))
        self.post_n_fft = gen_istft_n_fft
        self.conv_post = weight_norm(nn.Conv2d(ch, self.post_n_fft + 2, (1, 7), (1, 1), padding=(0, 3)))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        # ReflectionPad2d args are (left, right, top, bottom); H axis gets 0/0.
        self.reflection_pad = nn.ReflectionPad2d((1, 0, 0, 0))
        self.stft = (
            CustomSTFT(filter_length=gen_istft_n_fft, hop_length=gen_istft_hop_size, win_length=gen_istft_n_fft)
            if disable_complex
            else TorchSTFT(filter_length=gen_istft_n_fft, hop_length=gen_istft_hop_size, win_length=gen_istft_n_fft)
        )

        self._register_load_state_dict_pre_hook(self._reshape_rank3_to_rank4_hook)

    def _reshape_rank3_to_rank4_hook(self, state_dict, prefix, *_args, **_kwargs):
        # Pretrained kokoro-v1_0.pth has Conv1d weights at noise_convs.* /
        # ConvTranspose1d weights at ups.* / Conv1d weights at conv_post.
        # Reshape to Conv2d / ConvTranspose2d shapes idempotently so pretrained
        # weights load without retraining. resblocks and noise_res have their
        # own pre-hooks on each AdaINResBlock1 instance — PyTorch recurses
        # into children automatically.
        conv_module_paths: list[str] = []
        conv_module_paths.extend(f"ups.{i}" for i in range(len(self.ups)))
        conv_module_paths.extend(f"noise_convs.{i}" for i in range(len(self.noise_convs)))
        conv_module_paths.append("conv_post")
        rank3_to_rank4_conv_state_dict(state_dict, prefix, conv_module_paths, alpha_param_paths=[])

    def vocoder_body(self, x, s, har):
        """Run the rank-4 conv/AdaIN/iSTFT body of the Generator.

        Both ``Generator.forward`` (training/Decoder path) and
        ``GeneratorFromHar.forward`` (Core ML inference entry) call this
        method so the rank-3 → rank-4 → rank-3 boundary lives in exactly
        one place. Inputs and outputs are rank-3 / rank-2 so callers stay
        on the original public contract.

        Args:
            x:   ``(B, C, T_in)`` — pre-vocoder features. Generator gets these
                 from ``Decoder``; GeneratorFromHar gets them from the
                 ``x_pre`` input it traces against.
            s:   ``(B, style_dim)`` — voice style vector.
            har: ``(B, C_har, T_har)`` — concat of ``[har_spec, har_phase]``
                 from the F0-driven harmonic-noise source. Computed inside
                 ``Generator.forward`` (PyTorch CPU); supplied directly as an
                 input by ``GeneratorFromHar``.

        Returns:
            Rank-3 waveform from ``self.stft.inverse`` (typically
            ``(B, 1, T_out)``).
        """
        # Promote to rank-4 (B, C, 1, T) so the Conv2d / ConvTranspose2d /
        # AdaIN2d stack stays ANE-aligned end to end.
        x = x.unsqueeze(-2)      # (B, C, 1, T_in)
        har = har.unsqueeze(-2)  # (B, C_har, 1, T_har)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, negative_slope=0.1)
            x_source = self.noise_convs[i](har)
            x_source = self.noise_res[i](x_source, s)
            x = self.ups[i](x)
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)
            # Harmonic branch vs upsampled feature length can differ by a few
            # samples (STFT / conv output rounding). Align before add —
            # required for stable torch.jit.trace and Core ML.
            tx = x.size(-1)
            ts = x_source.size(-1)
            if ts < tx:
                x_source = F.pad(x_source, (0, tx - ts))
            elif ts > tx:
                x_source = x_source[:, :, :, :tx]
            x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x, s)
                else:
                    # Use ``xs = xs + ...`` (not ``+=``) — torch.jit.trace
                    # prefers non-in-place additions in loop bodies.
                    xs = xs + self.resblocks[i * self.num_kernels + j](x, s)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)        # (B, post_n_fft + 2, 1, T_post)
        x = x.squeeze(-2)            # drop H=1; iSTFT expects rank-3
        spec = torch.exp(x[:, : self.post_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.post_n_fft // 2 + 1 :, :])
        return self.stft.inverse(spec, phase)

    def forward(self, x, s, f0):
        # Inputs:
        #   x:   (B, C, T_in)   — rank-3 from upstream (e.g. ``Decoder``).
        #   s:   (B, style_dim) — rank-2 style vector.
        #   f0:  (B, T_f0)      — rank-2 F0 contour.
        # Output:
        #   waveform (B, 1, T_out) — rank-3 boundary preserved for legacy
        #   callers. The rank-4 promotion happens inside ``vocoder_body``.
        #
        # The Core ML inference path (``GeneratorFromHar``) bypasses this
        # method's F0 / m_source / STFT preamble (which is not Core
        # ML-traceable) and calls ``self.vocoder_body(x_pre, s, har)``
        # directly with a pre-computed ``har``.
        with torch.no_grad():
            f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # bs,n,t
            har_source, noi_source, uv = self.m_source(f0)
            har_source = har_source.transpose(1, 2).squeeze(1)
            har_spec, har_phase = self.stft.transform(har_source)
            har = torch.cat([har_spec, har_phase], dim=1)  # (B, C_har, T_har)
        return self.vocoder_body(x, s, har)


class UpSample1d(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        else:
            return F.interpolate(x, scale_factor=2, mode='nearest')


class AdainResBlk1d(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, actv=nn.LeakyReLU(0.2), upsample='none', dropout_p=0.0):
        super().__init__()
        self.actv = actv
        self.upsample_type = upsample
        self.upsample = UpSample1d(upsample)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)
        self.dropout = nn.Dropout(dropout_p)
        if upsample == 'none':
            self.pool = nn.Identity()
        else:
            self.pool = weight_norm(nn.ConvTranspose1d(dim_in, dim_in, kernel_size=3, stride=2, groups=dim_in, padding=1, output_padding=1))

    def _build_weights(self, dim_in, dim_out, style_dim):
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_out, dim_out, 3, 1, 1))
        self.norm1 = AdaIN1d(style_dim, dim_in)
        self.norm2 = AdaIN1d(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))

    def _shortcut(self, x):
        x = self.upsample(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.pool(x)
        x = self.conv1(self.dropout(x))
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(self.dropout(x))
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        out = (out + self._shortcut(x)) * torch.rsqrt(torch.tensor(2.0))
        return out


class Decoder(nn.Module):
    def __init__(self, dim_in, style_dim, dim_out, 
                 resblock_kernel_sizes,
                 upsample_rates,
                 upsample_initial_channel,
                 resblock_dilation_sizes,
                 upsample_kernel_sizes,
                 gen_istft_n_fft, gen_istft_hop_size,
                 disable_complex=False):
        super().__init__()
        self.encode = AdainResBlk1d(dim_in + 2, 1024, style_dim)
        self.decode = nn.ModuleList()
        self.decode.append(AdainResBlk1d(1024 + 2 + 64, 1024, style_dim))
        self.decode.append(AdainResBlk1d(1024 + 2 + 64, 1024, style_dim))
        self.decode.append(AdainResBlk1d(1024 + 2 + 64, 1024, style_dim))
        self.decode.append(AdainResBlk1d(1024 + 2 + 64, 512, style_dim, upsample=True))
        self.F0_conv = weight_norm(nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1))
        self.N_conv = weight_norm(nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1))
        self.asr_res = nn.Sequential(weight_norm(nn.Conv1d(512, 64, kernel_size=1)))
        self.generator = Generator(style_dim, resblock_kernel_sizes, upsample_rates, 
                                   upsample_initial_channel, resblock_dilation_sizes, 
                                   upsample_kernel_sizes, gen_istft_n_fft, gen_istft_hop_size, disable_complex=disable_complex)

    def forward(self, asr, F0_curve, N, s):
        F0 = self.F0_conv(F0_curve.unsqueeze(1))
        N = self.N_conv(N.unsqueeze(1))
        x = torch.cat([asr, F0, N], axis=1)
        x = self.encode(x, s)
        asr_res = self.asr_res(asr)
        res = True
        for block in self.decode:
            if res:
                x = torch.cat([x, asr_res, F0, N], axis=1)
            x = block(x, s)
            if block.upsample_type != "none":
                res = False
        x = self.generator(x, s, F0_curve)
        return x
