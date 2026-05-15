# ANE Rank-4 Rewrite — `decoder_har_post` Plan

**Date:** 2026-05-15
**Status:** Planned

## Executive Summary

Rewrite the kokoro Generator path from rank-3 `(B, C, T)` to rank-4
`(B, C, 1, T)` so that the `kokoro_decoder_har_post_{3s,10s}.mlpackage`
artifacts actually engage Apple's Neural Engine. The current 10s package
engages **0 of ~1238 ops** on the Neural Engine on M3 Max / macOS 26.4 and
silently falls back to GPU; the Espresso compiler emits 322
`Unsupported op` events and 12 `Shape computation issue` events, all at
rank-3 shape sites
([investigation note](../Notes/ane-decoder-har-post-investigation.md)).
Target outcome: re-exported packages with **Neural Engine column count
> 0** in Xcode's Performance Report and `.all` no longer bit-equal to
`.cpuAndGPU`.

## Problem Statement

- **Symptom:** `MLComputeUnits.all` schedules **zero** ops to the Neural
  Engine for `kokoro_decoder_har_post_10s.mlpackage`. `.all` output is
  byte-identical to `.cpuAndGPU` (sha256 `d0bbdce3360a0506`). `.all` pays
  a **55× cold-load penalty** (26.5 s vs 0.48 s on `.cpuAndGPU`) for a
  doomed ANE compile that retries 2× per segment before falling back to
  the MPSGraph class.
- **Root Cause:** The Generator (`kokoro/istftnet.py::Generator`) and the
  vocoder wrapper (`export_synth/wrappers.py::GeneratorFromHar`) operate
  on rank-3 `(B, C, T)` tensors throughout. Apple's ANE requires rank-4
  `(B, C, 1, T)` with `T` as the largest (last) axis to satisfy a 64-byte
  alignment penalty (see [CLAUDE.md Part 4.1](../../CLAUDE.md)). The
  segmenter rejects every rank-3 `conv → 1×C×T`, every AdaIN
  `reduce_mean → 1×C×1`, and every rank-3 `mul → 1×C×T`. The leading
  fp32→fp16 cast is also rejected, but that op is supposed to fall back
  to CPU/GPU at the boundary — the graph bails because the **body** is
  unsupported, not because of the entry cast.
- **Impact:** All HAR-post predict time runs on GPU (MPSGraph) on
  consumer Macs; the M2 Air ablation in
  [Core ML compute-unit ablation](../Notes/coreml-compute-unit-ablation.md)
  shows `.all` is up to **2.5–3× slower** than `.cpuAndGPU` because of
  the failed-ANE-compile overhead (15s/30s buckets: 4803 ms vs 1592 ms).
  Latency target left on the table; battery and thermal cost paid for
  zero benefit.

## Goals and Non-Goals

### Goals

- [ ] **Neural Engine engagement.** Xcode Performance Report
  `Neural Engine` column count is **> 0** on the re-exported
  `kokoro_decoder_har_post_10s.mlpackage`. 0 is failure.
- [ ] **No more silent fallback.** Python probe
  ([`/tmp/ane-investigation/probe.py`](../Notes/ane-decoder-har-post-investigation.md#6-reproducer))
  shows `.all` output sha256 **differs** from `.cpuAndGPU` (a different
  hash on warm predict means the compute path differs).
- [ ] **Cold-load recovery.** `.all` cold-load time drops from ~26 s to
  the same order as `.cpuAndGPU` (≤ 2 s target; ≤ 5 s acceptable).
- [ ] **PyTorch parity.** Rank-4 `GeneratorFromHar` output matches the
  rank-3 baseline on real `(x_pre, ref_s, har)` inputs to fp32 rounding
  tolerance (existing
  [`scripts/compare_decoder_har_post_waveforms.py`](../../scripts/compare_decoder_har_post_waveforms.py)
  gates: Pearson **r > 0.99**, **SNR ≥ 40 dB**, **max abs Δ ≤ 1e-2**).
- [ ] **Checkpoint compatibility.** Pretrained `hexgrad/Kokoro-82M`
  weights load without retraining via
  `register_load_state_dict_pre_hook` that reshapes Conv1d weights
  `(C_out, C_in, k)` → `(C_out, C_in, 1, k)` and `alpha` parameters
  `(1, C, 1)` → `(1, C, 1, 1)`.
- [ ] **Both shipping buckets re-exported.** `coreml/kokoro_decoder_har_post_3s.mlpackage`
  and `coreml/kokoro_decoder_har_post_10s.mlpackage` rebuilt with the new
  rank-4 graph.
- [ ] **Results-log row appended** to the section at the bottom of this
  plan, matching the
  [`ane-optimization-v1.md` results-log style](./ane-optimization-v1.md#results-log-commit-this).

### Non-Goals

- **fp16 input dtypes.** Step 4(b) from the
  [investigation brief](../Notes/ane-decoder-har-post-investigation.md#4-hypotheses-tested).
  Out of scope here. If Phase 3 verification shows residual non-ANE ops
  concentrated at the input boundary, open a follow-on plan to switch
  `x_pre`, `ref_s`, `har` to `np.float16` and update runtime callers.
  Bundling fp16-inputs into this plan adds runtime-caller churn (Python
  harness, Swift pipeline, bench scripts) that we don't need until we
  know we need it.
- **`minimum_deployment_target` bump.** Step 4(a). Already ruled out per
  the investigation brief (macOS13 → macOS15 took effect; ANE column
  still 0/1238). Keep `target = ct.target.macOS13` in
  [`export_synth/convert.py`](../../export_synth/convert.py) for this
  plan.
- **`decoder_pre_*.mlpackage` rewrite implementation.**
  [`Decoder.encode`](../../kokoro/istftnet.py#L526) /
  [`Decoder.decode`](../../kokoro/istftnet.py#L526) /
  [`AdainResBlk1d`](../../kokoro/istftnet.py#L482) have the same rank-3
  disqualification but are exported via a separate path
  ([`export_decoder_pre.py::DecoderPreWrapper`](../../export_decoder_pre.py#L63)).
  Sequence as: this plan (HAR-post) → validate ANE engagement on real
  audio → follow-on plan
  (`ane-decoder-pre-rank4-rewrite-v1.md`) for the decoder_pre rewrite.
  Doubling the rewrite surface in one PR doubles audit risk. **However,
  this plan's design *must* make that retrofit cheap** — see
  [Retrofit-Readiness for `decoder_pre` Rewrite](#retrofit-readiness-for-decoder_pre-rewrite)
  below. Concretely: `AdaIN2d` is the shared rank-4 AdaIN used by both
  (the Generator's `AdaINResBlock1` now, the Decoder's `AdainResBlk1d`
  later); the weight-reshape utility for `Conv1d → Conv2d` checkpoint
  loading is a shared function reused by both rewrites; the
  unsqueeze/squeeze pattern at the wrapper boundary is identical between
  `GeneratorFromHar` (now) and the future rank-4 `DecoderPreWrapper`.
- **iSTFT tail.** `kokoro/istftnet.py::CustomSTFT.inverse` and
  `TorchSTFT.inverse` stay rank-3. The Generator's `conv_post` output
  squeezes the H=1 axis before the iSTFT call. Revisit only if Phase 3
  shows the iSTFT is the next-gating chunk on ANE engagement.
- **Linear-vs-Conv1d for `AdaIN1d.fc`.** The
  [previous plan](./ane-optimization-v1.md) landed this and it was
  reverted in commit `5278e88` for "no Core ML predict win." That was
  measured without confirming ANE engagement; the rank cause was always
  the dominant lever. Do not re-litigate Linear vs Conv1d in this plan;
  keep `nn.Linear` and project to rank-4 via reshape at the AdaIN
  consumption site.
- **Training / fine-tuning compatibility for the Generator.** The
  rewritten Generator is inference-only here. If anyone later retrains
  Kokoro, the rank-4 modules can either be ported back or the load-hook
  pattern reversed. Out of scope to make the rank-4 modules
  training-compatible in this PR.
- **Audio quality re-tuning.** Pearson > 0.99 against the pre-rewrite
  baseline is the bar; perceptual A/B tests against the baseline are
  optional confirmation and not blocking.

## Scope and Constraints

- **Scope (code):**
  [`kokoro/istftnet.py::Generator`](../../kokoro/istftnet.py#L391),
  [`kokoro/istftnet.py::AdaINResBlock1`](../../kokoro/istftnet.py#L153),
  [`kokoro/istftnet.py::AdaIN1d`](../../kokoro/istftnet.py#L98) (read-only;
  new sibling class `AdaIN2d` lives next to it),
  [`export_synth/wrappers.py::GeneratorFromHar`](../../export_synth/wrappers.py#L67).
- **Scope (artifacts):** `coreml/kokoro_decoder_har_post_3s.mlpackage`,
  `coreml/kokoro_decoder_har_post_10s.mlpackage`.
- **Constraints:**
  - **Decoder path unchanged.**
    [`kokoro/istftnet.py::AdainResBlk1d`](../../kokoro/istftnet.py#L482),
    [`Decoder.encode`/`Decoder.decode`](../../kokoro/istftnet.py#L526) keep
    using rank-3 `AdaIN1d`. Decoder uses
    [`IdentityAdaIN`](../../export_synth/wrappers.py#L466) swap-in for
    HAR-post export anyway — its `AdaIN1d` doesn't even reach the
    decoder-har JIT graph — so the rewrite must not change `AdaIN1d`'s
    rank-3 contract.
  - **iSTFT boundary is rank-3.** Squeeze H=1 axis at conv_post output
    before the iSTFT call. iSTFT internals untouched.
  - **No mask threading work.** Current
    `GeneratorFromHar.forward(x_pre, ref_s, har)` has no mask parameter
    on `main`. The investigation brief mentioned `_align_mask_to` on a
    `mask-aware-bucketing-v1` branch that isn't reflected on main.
    Confirmed by grep: no `_align_mask_to` exists in this tree.
- **Guardrails:** PyTorch parity ≥ Pearson > 0.99 / SNR ≥ 40 dB /
  max abs Δ ≤ 1e-2 on the existing
  [`scripts/compare_decoder_har_post_waveforms.py`](../../scripts/compare_decoder_har_post_waveforms.py)
  harness, against pre-rewrite `/tmp/...` baseline packages.

## Ground Truth Contracts (Do Not Violate)

- **Pretrained checkpoint shape mapping.** `nn.Conv1d.weight`
  `(C_out, C_in, k)` reshapes to `nn.Conv2d.weight` `(C_out, C_in, 1, k)`
  by `tensor.unsqueeze(-2)`. `nn.ConvTranspose1d.weight`
  `(C_in, C_out_per_group, k)` reshapes to `nn.ConvTranspose2d.weight`
  `(C_in, C_out_per_group, 1, k)` by `tensor.unsqueeze(-2)`. Conv bias
  shape `(C_out,)` is unchanged between Conv1d and Conv2d. `alpha`
  Parameter shape `(1, C, 1)` reshapes to `(1, C, 1, 1)` by
  `tensor.unsqueeze(-1)`. The pre-hook applies these reshapes
  idempotently — if the tensor already has the target rank, the hook is
  a no-op.
- **iSTFT input contract.** `TorchSTFT.inverse(spec, phase)` and
  `CustomSTFT.inverse(spec, phase)` accept rank-3
  `(B, n_fft//2 + 1, T)`. The rank-4 Generator must squeeze the H=1
  axis at conv_post before slicing into `spec` / `phase` and calling
  `inverse`.
- **Style vector stays rank-2.** `ref_s` / `s` is `(B, style_dim)`.
  Project to rank-4 via reshape (`view`) when broadcasting against
  rank-4 `x`, do not unsqueeze at the entry.
- **`AdaIN1d` rank-3 invariant.** The original
  [`AdaIN1d`](../../kokoro/istftnet.py#L98) class continues to consume
  and return rank-3 `(B, C, T)` tensors. It is the Decoder path's
  contract. The new sibling class `AdaIN2d` is what the rank-4 Generator
  uses; the two coexist with no shared state.
- **External API surface (`GeneratorFromHar.forward(x_pre, ref_s, har)`).**
  Inputs remain rank-3 `(B, C, T)` and rank-2 `(B, style_dim)` at the
  module boundary — the rank-4 promotion happens inside the wrapper at
  `unsqueeze(-2)` time, and the output is squeezed back to rank-3 before
  the iSTFT call so existing Python and Swift callers don't change.

## Already Shipped (Do Not Re-Solve)

- **MIL op histogram script.**
  [`scripts/count_mil_ops.py`](../../scripts/count_mil_ops.py) — works
  as-is; reused in Phase 0 / Phase 3.
- **Waveform parity gates.**
  [`scripts/compare_decoder_har_post_waveforms.py`](../../scripts/compare_decoder_har_post_waveforms.py)
  — Pearson / SNR / max abs Δ thresholds already wired.
- **Predict-latency fallback bench.**
  [`scripts/bench_decoder_har_post_predict.py`](../../scripts/bench_decoder_har_post_predict.py)
  — warmup + median of N predicts. Reuse for Phase 3 wall-clock.
- **Investigation scratch tooling.**
  `/tmp/ane-investigation/probe.py`,
  `/tmp/ane-investigation/capture_logs.sh`,
  `/tmp/ane-investigation/dump_mil_ops.py`. Move to `scripts/` only if
  Phase 3 acceptance reuses them on every re-export.
- **`IdentityAdaIN` swap-in.**
  [`export_synth/wrappers.py::IdentityAdaIN`](../../export_synth/wrappers.py#L466)
  replaces `AdainResBlk1d.norm1/norm2` on the shared `kmodel` before
  trace for every export mode. Confirms that **Decoder `AdaIN1d`
  instances never enter the decoder-har JIT graph** — only the
  Generator's `AdaIN1d` (inside `AdaINResBlock1`) does. Reinforces
  the Constraint that `AdaIN1d` rank-3 contract stays untouched.

## Fresh Baseline (Current State)

- **MIL op histogram (rank-3 10s baseline, 2026-05-15, coremltools 8.3.0):**
  2207 total ops; `const` 1166, `add` 218, `mul` 148, `tile` 96,
  `reduce_mean` 88, `conv` 51, `sin` 50, `linear` 48, `reshape` 48,
  `split` 48, `pow` 48, `sub` 45, `square` 44, `sqrt` 44, `real_div` 44,
  `slice_by_index` 6, `cast` 5, `conv_transpose` 4, `leaky_relu` 3,
  `pad` 1, `exp` 1, `cos` 1. See
  [investigation note §1](../Notes/ane-decoder-har-post-investigation.md#1-model-identification).
- **ANE engagement (Xcode Compute Unit Mapping):**
  `All: 1,238   CPU: 0   GPU: 1,238   Neural Engine: 0`.
- **Compute-unit probe (random inputs, M3 Max / macOS 26.4):**

  | `MLComputeUnits` | Cold load (s) | Warm predict (s) | sha256(out) |
  | --- | --- | --- | --- |
  | `.all` | 26.485 | 0.116 | `d0bbdce3360a0506` |
  | `.cpuAndGPU` | 0.476 | 0.119 | `d0bbdce3360a0506` (matches `.all`) |
  | `.cpuAndNE` | 420.194 | 0.768 | `0222e613c45c1104` (differs) |

- **Espresso `log stream` during `.all` load:** 322 `Unsupported op`
  events (indices 7..1041, all unique), 12 `Shape computation issue`
  events at layers 37 / 44 / 51, 4 `class was unable to load... going to
  use another class` events (silent fallback), 8 "Model enabled
  BNNSGraph as a preferred cpu backend" events.

## Solution Overview

```text
PyTorch rank-3 input        Rank-4 promotion at entry
+-------------+             +----------------------+
| x_pre 1×C×T |--unsqueeze->| x 1×C×1×T            |
| har 1×Ch×Th |             | har 1×Ch×1×Th        |
+-------------+             +----------------------+
                                       |
                            +----------v-----------+
                            | Generator body       |
                            |  - Conv2d (1, k)     |
                            |  - ConvTranspose2d   |
                            |  - AdaIN2d (B,C,1,T) |
                            |  - Snake1D, leaky_relu|
                            +----------+-----------+
                                       |
                            +----------v-----------+
                            | conv_post Conv2d     |
                            |  -> 1×(n_fft+2)×1×T  |
                            +----------+-----------+
                                       |
                            squeeze H=1 axis (back to rank-3)
                                       |
                            +----------v-----------+
                            | iSTFT inverse (rank-3)|
                            +----------------------+
```

| Phase | Purpose |
| --- | --- |
| 0 | MIL audit lock + Conv2d(1,k) lowering probe (current stack) |
| 1 | Rank-4 module rewrite (`AdaIN2d`, `AdaINResBlock1`, `Generator`, `GeneratorFromHar`) + load hook + PyTorch parity tests |
| 2 | Re-export `kokoro_decoder_har_post_{3s,10s}.mlpackage`; waveform parity vs `/tmp` baselines |
| 3 | ANE placement verification (Xcode + probe triplet + `log stream`); results-log row |

## Retrofit-Readiness for `decoder_pre` Rewrite

The decoder_pre stack has the same rank-3 disqualification as HAR-post
and will need its own rank-4 rewrite next (out of scope here — see
[Non-Goals](#goals-and-non-goals)). This plan is designed so that the
follow-on `ane-decoder-pre-rank4-rewrite-v1.md` plan reuses the same
building blocks instead of duplicating them.

### Decoder-pre components that will need rank-4 conversion (later)

| File:line | Module | Current rank-3 form | Target rank-4 form |
| --- | --- | --- | --- |
| [`kokoro/istftnet.py::AdainResBlk1d`](../../kokoro/istftnet.py#L482) | `_build_weights` | `nn.Conv1d` × 3 (`conv1`, `conv2`, optional `conv1x1`) | `nn.Conv2d((1, k))` × 3 |
| [`kokoro/istftnet.py::AdainResBlk1d`](../../kokoro/istftnet.py#L482) | `norm1`, `norm2` | `AdaIN1d` | `AdaIN2d` (this plan's class) |
| [`kokoro/istftnet.py::AdainResBlk1d`](../../kokoro/istftnet.py#L494) | `pool` (when `upsample != 'none'`) | `nn.ConvTranspose1d` | `nn.ConvTranspose2d((1, k), stride=(1, 2))` |
| [`kokoro/istftnet.py::AdainResBlk1d`](../../kokoro/istftnet.py#L487) | `upsample` (`UpSample1d`) | `F.interpolate(scale_factor=2, mode='nearest')` on rank-3 | `F.interpolate(scale_factor=(1, 2), mode='nearest')` on rank-4 |
| [`kokoro/istftnet.py::Decoder.F0_conv`](../../kokoro/istftnet.py#L542) | `F0_conv`, `N_conv` | `nn.Conv1d(1, 1, k=3, s=2, p=1)` | `nn.Conv2d((1, 3), stride=(1, 2), padding=(0, 1))` |
| [`kokoro/istftnet.py::Decoder.asr_res`](../../kokoro/istftnet.py#L544) | `asr_res` | `nn.Conv1d(512, 64, k=1)` | `nn.Conv2d((1, 1))` |
| [`export_decoder_pre.py::DecoderPreWrapper.forward`](../../export_decoder_pre.py#L77) | wrapper boundary | rank-3 inputs (asr, f0, n_input, ref_s) | unsqueeze inputs at entry, squeeze x_pre at exit so the downstream `GeneratorFromHar` consumer is unchanged |

### Design choices in this plan that keep the retrofit cheap

- **`AdaIN2d` is generic, not Generator-specific.**
  `AdaIN2d(style_dim, num_features).forward(x: (B, C, 1, T), s: (B, style_dim))` →
  `(B, C, 1, T)`. The implementation has no knowledge of the surrounding
  module. When `AdainResBlk1d`'s rank-4 rewrite lands, it switches
  `self.norm1 = AdaIN1d(...)` → `self.norm1 = AdaIN2d(...)` with no
  other module changes inside `AdaIN2d`. Reflected in the
  [Open Questions](#open-questions): we picked the sibling-class
  approach specifically so two callers (now Generator, later Decoder)
  can share without coupling.
- **Weight-reshape state-dict hook is a shared utility, not an inlined
  closure.** Add a module-level helper next to `AdaIN1d` in
  [`kokoro/istftnet.py`](../../kokoro/istftnet.py) (no new files):

  ```python
  # In kokoro/istftnet.py, near AdaIN1d.
  def _rank3_to_rank4_conv_state_dict(state_dict, prefix, conv_weight_keys, alpha_keys):
      """Idempotently reshape Conv1d / ConvTranspose1d weights to Conv2d /
      ConvTranspose2d shape and alpha parameters from (1, C, 1) to (1, C, 1, 1)
      inside a state_dict.

      Handles both bare weights (e.g. `*.weight`) and weight_norm-wrapped
      forms (`*.weight_orig` / `*.weight_v` / `*.weight_g`).

      Reused by:
        - AdaINResBlock1 / Generator (this plan, HAR-post).
        - AdainResBlk1d / Decoder (follow-on plan, decoder_pre).
      """
      # implementation: see Phase 1 task list.
  ```

  Then both `AdaINResBlock1` and (later) `AdainResBlk1d` register
  load hooks that call this helper with their own key lists. No
  cross-class coupling beyond the function call.
- **Unsqueeze/squeeze pattern is the same at every rank-4 wrapper.**
  `GeneratorFromHar.forward` (this plan) and the future
  `DecoderPreWrapper.forward` rewrite (follow-on) both: unsqueeze
  rank-3 input tensors at `-2`, run the rank-4 body, squeeze `-2` from
  the rank-4 output. Identical idiom. Capture this in the
  `GeneratorFromHar` docstring so the future plan can cargo-cult the
  pattern verbatim.
- **Rank-3 modules left untouched.** `AdaIN1d`, `AdainResBlk1d`,
  `Decoder.encode`/`Decoder.decode`, `DecoderPreWrapper`,
  `Decoder.F0_conv`/`Decoder.N_conv`/`Decoder.asr_res` all stay rank-3.
  No partial conversion, no shape-detecting modules. The decoder_pre
  retrofit is a self-contained subsequent change.

### Retrofit checklist (preview, for the follow-on plan)

When `ane-decoder-pre-rank4-rewrite-v1.md` is written, the implementer
should:

1. Reuse `AdaIN2d` and `_rank3_to_rank4_conv_state_dict` from this plan
   — do not introduce a parallel `AdaIN2d_v2` or a separate hook.
2. Convert `AdainResBlk1d` / `Decoder.encode` / `Decoder.decode` /
   `Decoder.F0_conv` / `Decoder.N_conv` / `Decoder.asr_res` per the
   table above.
3. Update `DecoderPreWrapper.forward` to unsqueeze inputs at entry and
   squeeze the output before returning. Note: `x_pre` is consumed
   downstream by `GeneratorFromHar` — keep the output rank-3 so this
   plan's `GeneratorFromHar` (which already unsqueezes its own
   `x_pre` input) stays unchanged.
4. Re-export `coreml/kokoro_decoder_pre_{3s,5s,7s,10s,15s,30s}.mlpackage`
   (or whatever bucket set `export_decoder_pre.py` ships).
5. Run the same Xcode + probe + `log stream` ANE-engagement gate as
   this plan's Phase 3, plus the existing
   `tests/test_decoder_pre*` parity gates (if they exist;
   otherwise model them on this plan's parity tests).

## Implementation Phases

> Do one phase at a time. Verify before proceeding.

### Phase 0: MIL audit lock + Conv2d(1,k) lowering probe

**Goal:** Confirm the rank-3 baseline op-type taxonomy is stable on the
current `coremltools 8.3.0` + macOS 26 stack and verify that
`nn.Conv2d(in, out, (1, k))` lowers to a single MIL `conv` op for k > 1.
The prior plan ([Phase 0 of ane-optimization-v1.md](./ane-optimization-v1.md#phase-0-mil-and-op-audit))
verified Conv2d(1, 1); this plan needs Conv2d(1, k) where k is the
kernel size of the real convs (3, 7, and the various
upsample/noise-conv kernels).

**Tasks:**

- [ ] Re-run `uv run python scripts/count_mil_ops.py coreml/kokoro_decoder_har_post_10s.mlpackage`
  and confirm the 2207-op rank-3 baseline histogram matches the
  [Fresh Baseline](#fresh-baseline-current-state) above. Record any
  delta and resolve before moving on.
- [ ] Extend
  [`scripts/count_mil_ops.py`](../../scripts/count_mil_ops.py)'s
  `--probe-conv-lowering` mode to also probe `Conv2d(in, out, (1, k))`
  with k in `{3, 7}` (the real Generator kernels), and
  `ConvTranspose2d(in, out, (1, k))` with stride `(1, u)` and padding
  `(0, (k-u)//2)` for k=u=10..20 (the upsample stride space). Each must
  lower to a single MIL `conv` or `conv_transpose` op with no inserted
  `reshape` or `transpose`. If any pattern lowers to extra reshape ops,
  fail Phase 0 and reconsider the rewrite strategy.
- [ ] **Lock taxonomy:** Phase 3 will compare op histograms using the
  exact `op.type` strings produced by this script. Pin the version of
  the script in commit history so the comparison is reproducible.

**Verification:** Conv2d(1, k) and ConvTranspose2d((1, k), stride=(1, u))
both lower to a single MIL `conv` / `conv_transpose` op for the real
kernel/stride sizes used in `kokoro/istftnet.py::Generator`. Histogram
deltas vs the baseline are zero (or explained).

---

### Phase 1: Rank-4 module rewrite + PyTorch parity

**Goal:** Replace the Generator path's rank-3 modules with rank-4
counterparts, keep pretrained checkpoint loading working, and prove
numerical equivalence in PyTorch before touching Core ML.

**Tasks (`kokoro/istftnet.py`):**

- [ ] Add new class `AdaIN2d` next to
  [`AdaIN1d` at line 98](../../kokoro/istftnet.py#L98). API:
  `forward(x, s)` where `x` is `(B, C, 1, T)`, `s` is `(B, style_dim)`,
  output is `(B, C, 1, T)`. Implementation:
  - `self.fc = nn.Linear(style_dim, num_features * 2)` (unchanged from
    `AdaIN1d` — Linear is fine; the projection just reshapes to rank-4
    after).
  - `mean = x.mean(dim=-1, keepdim=True)` and
    `var = x.var(dim=-1, unbiased=False, keepdim=True)` reduce over T
    only, producing `(B, C, 1, 1)`.
  - `x_norm = (x - mean) / torch.sqrt(var + self.eps)`.
  - `h = self.fc(s).view(B, 2 * num_features, 1, 1)`;
    `gamma, beta = torch.chunk(h, 2, dim=1)`.
  - Output: `(1.0 + gamma) * x_norm + beta` (broadcast handles the T
    expansion implicitly — no explicit `expand` needed; that avoids
    the kind of rank-3 `expand`+`reshape` pattern that triggered
    "Shape computation issue" in the rank-3 graph).
  - **Do not** modify the original `AdaIN1d`. It stays untouched for
    the Decoder path's `AdainResBlk1d`.
- [ ] Convert `AdaINResBlock1` (
  [line 153](../../kokoro/istftnet.py#L153)) to rank-4:
  - `self.convs1` / `self.convs2`: `nn.Conv1d(C, C, k, 1, dilation=d, padding=p)`
    → `nn.Conv2d(C, C, (1, k), 1, dilation=(1, d), padding=(0, p))`.
  - `self.adain1` / `self.adain2`: now hold `AdaIN2d` instances.
  - `self.alpha1` / `self.alpha2`: shape `(1, channels, 1)` → `(1, channels, 1, 1)`.
  - Forward: same control flow, no shape changes needed — convs and
    AdaIN are now rank-4 native; `torch.sin(a * xt) ** 2` works
    elementwise on rank-4.
- [ ] Convert `Generator` (
  [line 391](../../kokoro/istftnet.py#L391)) to rank-4:
  - `self.noise_convs`: `nn.Conv1d(...)` →
    `nn.Conv2d(..., kernel_size=(1, k), stride=(1, stride_f0), padding=(0, (stride_f0+1)//2))`
    for the strided branch; `nn.Conv2d(..., (1, 1))` for the k=1 branch.
  - `self.ups`: `nn.ConvTranspose1d(...)` →
    `nn.ConvTranspose2d(in, out, (1, k), stride=(1, u), padding=(0, (k-u)//2))`.
  - `self.conv_post`: `nn.Conv1d(ch, n_fft+2, 7, 1, padding=3)` →
    `nn.Conv2d(ch, n_fft+2, (1, 7), 1, padding=(0, 3))`.
  - `self.reflection_pad`: `nn.ReflectionPad1d((1, 0))` →
    `nn.ReflectionPad2d((1, 0, 0, 0))` (PyTorch's 2d pad is
    `(left, right, top, bottom)`).
  - Forward: leave alone — `Generator.forward` is the training path that
    operates on a rank-3 `(B, C, T)` upstream; **`GeneratorFromHar` is
    the inference entry point that does the rank-4 promotion**. If
    `Generator.forward` is also traced from anywhere
    (`export_synth/convert.py` `mode == "full"` for instance), wrap with
    its own rank-3 → rank-4 → rank-3 squeeze/unsqueeze adapter.
- [ ] Add a module-level helper `_rank3_to_rank4_conv_state_dict` next to
  [`AdaIN1d`](../../kokoro/istftnet.py#L98). **Shared with the future
  `decoder_pre` rewrite** — see
  [Retrofit-Readiness for `decoder_pre` Rewrite](#retrofit-readiness-for-decoder_pre-rewrite).
  Signature:

  ```python
  def _rank3_to_rank4_conv_state_dict(state_dict, prefix, conv_weight_keys, alpha_keys):
      """Idempotently reshape Conv1d / ConvTranspose1d weights to Conv2d /
      ConvTranspose2d shape and alpha parameters from (1, C, 1) to (1, C, 1, 1)
      inside a state_dict.

      conv_weight_keys: iterable of relative key suffixes for conv weights
        (e.g. ``("convs1.0.weight", "convs2.1.weight", ...)``). Both bare
        and weight_norm-wrapped forms (``weight``, ``weight_orig``,
        ``weight_v``, ``weight_g``) are reshaped.
      alpha_keys: iterable of relative key suffixes for `alpha`
        Parameters (Snake1D's ``alpha1``/``alpha2``).
      """
      ...
  ```

  Caller modules supply their own key lists, so the helper has no
  knowledge of `AdaINResBlock1` vs `AdainResBlk1d` and can be reused
  verbatim in the follow-on decoder_pre plan.
- [ ] Register `register_load_state_dict_pre_hook` on `AdaINResBlock1`
  and `Generator` that calls the shared helper with the key lists for
  this plan's modules:
  - `AdaINResBlock1` keys: `convs1.0.weight`, `convs1.1.weight`,
    `convs1.2.weight`, `convs2.0.weight`, `convs2.1.weight`,
    `convs2.2.weight`, plus the `weight_norm`-wrapped equivalents.
    Alpha keys: `alpha1.0`, `alpha1.1`, `alpha1.2`, `alpha2.0`,
    `alpha2.1`, `alpha2.2`.
  - `Generator` keys: `ups.{i}.weight*` (ConvTranspose1d),
    `noise_convs.{i}.weight*`, `conv_post.weight*`.
  - Verify hook arity against the actual env's torch
    (`torch==2.6.0` per `uv run python -c "import torch; print(torch.__version__)"`;
    `requirements-export.txt` pins 2.5.0). `register_load_state_dict_pre_hook`
    signature on torch 2.5/2.6 is `(module, state_dict, prefix,
    local_metadata, strict, missing_keys, unexpected_keys, error_msgs)`.

**Tasks (`export_synth/wrappers.py`):**

- [ ] Convert
  [`GeneratorFromHar.forward`](../../export_synth/wrappers.py#L87) to
  rank-4 internally. Inputs stay rank-3 / rank-2 at the boundary:
  - Entry: `x = x_pre.unsqueeze(-2)`,
    `har = har.unsqueeze(-2)`. (`ref_s` stays rank-2.)
  - Body loop: same control flow as today; `gen.noise_convs[i](har)`,
    `gen.noise_res[i](x_source, s)`, `gen.ups[i](x)`,
    `gen.reflection_pad(x)`, and `gen.resblocks[...]` all operate on
    rank-4 in the new module layout.
  - Size alignment: change
    [`tx = x.size(2)`](../../export_synth/wrappers.py#L98) →
    `tx = x.size(3)`; same for `ts`. The
    `F.pad(x_source, (0, tx - ts))` and `x_source[:, :, :, :tx]`
    statements pad / slice the last axis — F.pad in PyTorch always
    pads from the last dim, so no F.pad change needed; the slice
    becomes 4-axis (`x_source[:, :, :, :tx]`).
  - Exit: `x = gen.conv_post(x)` produces rank-4 `(B, n_fft+2, 1, T)`;
    `x = x.squeeze(-2)` produces rank-3 `(B, n_fft+2, T)`; rest of the
    code (spec/phase slicing + `gen.stft.inverse`) is unchanged.
- [ ] Document the rank promotion at the top of `GeneratorFromHar` so a
  future reader sees the rank-4 internal invariant explicitly (LLM-first
  doc style per [CLAUDE.md](../../CLAUDE.md)).

**Tasks (tests):**

- [ ] Add `tests/test_adain2d_vs_adain1d.py`: rank-3 `AdaIN1d` reference
  vs rank-4 `AdaIN2d` on synthetic `(B, C, T)` input
  (unsqueezed/squeezed for the rank-4 path);
  `torch.allclose(rank3_out, rank4_out.squeeze(-2), atol=1e-5, rtol=1e-5)`.
- [ ] Add `tests/test_generator_from_har_rank4.py`: rank-3 `GeneratorFromHar`
  reference vs rank-4 rewrite on real `(x_pre, ref_s, har)` inputs
  produced by
  [`kokoro.synthesis_backends.build_decoder_har_post_inputs_np`](../../kokoro/synthesis_backends.py)
  (or its existing usage in
  [`scripts/bench_decoder_har_post_predict.py`](../../scripts/bench_decoder_har_post_predict.py));
  Pearson > 0.99999 and max abs Δ < 1e-5 on PyTorch fp32 — much tighter
  tolerance than the Core ML cross-runtime gate because we're comparing
  PyTorch-to-PyTorch.
- [ ] Add `tests/test_rank4_checkpoint_load.py`: load real
  `hexgrad/Kokoro-82M` checkpoint (the same one
  `export_synth/main.py` loads) into the rank-4 Generator; assert no
  missing keys, no unexpected keys, and a forward pass on synthetic
  rank-4 input is finite.
- [ ] Re-run the existing
  [`tests/test_adain1d_decoder_smoke.py`](../../tests/test_adain1d_decoder_smoke.py)
  and
  [`tests/test_export_wrappers_shapes.py`](../../tests/test_export_wrappers_shapes.py)
  to confirm the Decoder path is unaffected (since `AdaIN1d` is
  unchanged and only the Generator path moved to `AdaIN2d`).
- [ ] `uv run pytest tests/` — full suite green.

**Verification:** PyTorch parity tests pass at fp32 tolerance; full
pytest green; pretrained checkpoint loads with no missing/unexpected
keys.

---

### Phase 2: Re-export `kokoro_decoder_har_post_{3s,10s}.mlpackage`

**Goal:** Produce the rank-4 Core ML packages and prove waveform parity
against the pre-rewrite baseline.

**Tasks:**

- [ ] Copy current `coreml/kokoro_decoder_har_post_{3s,10s}.mlpackage`
  to `/tmp/kokoro_decoder_har_post_{3s,10s}.baseline.mlpackage` before
  re-exporting (the in-repo path will be overwritten).
- [ ] Re-export:
  `uv run --no-sync python -m export_synth.main --mode decoder-har --buckets 3s,10s -o coreml`.
- [ ] **Export gates (existing):** the export script's "decoder-har
  numeric gate: traced vs Core ML waveform shape ..., all finite" line
  must pass for both buckets (already wired in
  [`export_synth/convert.py`](../../export_synth/convert.py)).
- [ ] **Waveform parity:**
  `uv run python scripts/compare_decoder_har_post_waveforms.py`
  comparing each re-exported bucket against its `/tmp` baseline using
  real `(x_pre, har, ref_s)` from the production pipeline path
  (`HybridTTSPipeline`, `_select_bucket_seconds` matched). Gates:
  Pearson **> 0.99**, **SNR ≥ 40 dB**, **max abs Δ ≤ 1e-2**.
- [ ] `uv run pytest tests/test_mlpackage_exports.py -q` — pass.

**Verification:** Both packages save, smoke export gate passes, Pearson
/ SNR / max abs Δ gates pass on `3s` and `10s` against the `/tmp`
baselines.

---

### Phase 3: ANE placement verification

**Goal:** Prove the rank-4 graph engages the Neural Engine. **This is
the new hard gate that did not exist in
[`ane-optimization-v1.md`](./ane-optimization-v1.md). 0 ops on Neural
Engine is failure.**

**Tasks (graph + placement):**

- [ ] **Xcode Performance Report:** open the re-exported
  `coreml/kokoro_decoder_har_post_10s.mlpackage` in Xcode, add the
  local Mac as the performance target, run a performance report, click
  **Compute Unit Mapping**, and screenshot the per-op table. Record:
  `All: N, CPU: a, GPU: b, Neural Engine: c`. **Hard gate: c > 0.**
  Save the screenshot to `outputs/ane_rank4/xcode_compute_unit_map_10s.png`
  (gitignored).
- [ ] **Python probe triplet:** run
  `/tmp/ane-investigation/probe.py` (or its hardened sibling, if moved
  to `scripts/`) with units `all`, `gpu`, `ne` in fresh subprocesses.
  **Hard gate: `.all` output sha256 differs from `.cpuAndGPU`** — that
  signals the compute path is no longer identical to the silent-fallback
  GPU path. **Cold-load gate: `.all` cold load ≤ 5 s** (down from 26.485
  s baseline).
- [ ] **Espresso log stream:** run
  `/tmp/ane-investigation/capture_logs.sh ... all ...` while loading
  with `.all`. Count `Unsupported op N` events. **Hard gate: count
  drops from 322 to a small boundary residue (target ≤ 50; ideally ≤ 20
  — the entry cast, output cast, and any iSTFT-adjacent squeeze).**
  Confirm zero `Shape computation issue at layer N` events at the
  former 37 / 44 / 51 sites; new sites elsewhere are evaluated case
  by case.
- [ ] **MIL op histogram (rank-4):** run
  `scripts/count_mil_ops.py` against the new 10s package; record the
  new histogram. `linear` count is expected to remain 48 (we kept
  `nn.Linear` for AdaIN's style projection); `conv` count is expected
  to rise (Conv1d → Conv2d still lowers to `conv`, plus additional
  noise_convs and conv_post). Some `tile` or `expand` ops may
  disappear if the rank-4 broadcasts eliminate explicit expansions.

**Tasks (wall-clock — fallback path):**

- [ ] `uv run python scripts/bench_decoder_har_post_predict.py --baseline /tmp/kokoro_decoder_har_post_10s.baseline.mlpackage`
  — median predict ms before/after. Expect predict-only ms to drop
  meaningfully on ANE engagement; if predict-only is unchanged but ANE
  count > 0, that's still success per the Hard Requirements (the
  cold-load and Espresso-event drops carry the result).
- [ ] **Optional:** `sudo powermetrics -i 1000 --samplers ane` for a few
  seconds during steady-state predict. Non-zero ANE Power is
  corroboration but not gating.
- [ ] **Optional:** `MLComputePlan.load(contentsOf: modelURL, ...)` from
  a small Swift snippet to read per-op compute device assignment
  programmatically. Document only if it shifts the verdict from one
  gate.

**Tasks (results log):**

- [ ] Append a row to the [Results log](#results-log-commit-this)
  section at the bottom of this plan: git SHA, coremltools version,
  hardware, Xcode Neural Engine count, probe sha256s, cold-load times,
  Pearson/SNR/maxΔ.

**Verification:** Xcode Neural Engine column count > 0 on the 10s
package; `.all` probe sha differs from `.cpuAndGPU` probe sha; `.all`
cold load ≤ 5 s; Espresso `Unsupported op` event count drops below 50;
waveform parity gates still green from Phase 2.

#### Escalation paths (pre-named so we don't scramble)

- **If Xcode Neural Engine count > 0 but partial (say 30-80% of ops):**
  inspect the residual `Unsupported op` indices via
  `/tmp/ane-investigation/capture_logs.sh` + `dump_mil_ops.py`. Identify
  which op-types are still rejected. Likely candidates: residual `tile`
  (96 in the rank-3 baseline; might be reduced by rank-4 broadcast but
  not eliminated), `slice_by_index` (6 occurrences, related to the
  `ref_s[:, :128]` style slice and the spec/phase split after
  conv_post), or boundary `cast`. Open a follow-on plan named
  `ane-decoder-har-residual-ops-v1.md` to tackle the specific op-types.
  **Do not** speculatively bundle fp16 inputs into this PR.
- **If Xcode Neural Engine count is still 0:** Step 4(b) (fp16 input
  dtypes) becomes the next experiment, in its own follow-on plan. Open
  `ane-decoder-har-fp16-inputs-v1.md`. The smoking-gun signal: re-capture
  the Espresso log stream and confirm the entry-cast op (former index 8)
  is no longer in the rejection list — if other ops fill its place at
  the input boundary, fp16 alone won't help either and the rank-4
  hypothesis was incomplete.
- **If PyTorch parity passes but Core ML waveform parity fails (Phase 2
  gates):** revert the load-hook portion first (the most likely
  fp16-rounding suspect); re-run with `--precision fp32` to isolate fp16
  drift from rank-4 graph drift. If the rank-4 fp32 export passes parity
  but rank-4 fp16 doesn't, document the per-layer drift and consider
  mixed-precision via `op_selector` for the AdaIN reductions.

## Success Criteria

### Hard Requirements (must pass)

- [ ] `AdaIN1d` (rank-3) **unchanged** in
  [`kokoro/istftnet.py`](../../kokoro/istftnet.py#L98) (line 98).
  Decoder path contract is preserved.
- [ ] New `AdaIN2d` class added next to `AdaIN1d`, consumed only by the
  rank-4 `AdaINResBlock1`. Round-trip vs `AdaIN1d` passes
  `torch.allclose(atol=1e-5, rtol=1e-5)` on synthetic input.
- [ ] `AdaINResBlock1` / `Generator` rebuilt with Conv2d / ConvTranspose2d
  / ReflectionPad2d; `alpha1` / `alpha2` reshape to `(1, C, 1, 1)`.
- [ ] `GeneratorFromHar.forward(x_pre, ref_s, har)` signature unchanged;
  rank promotion happens inside; output is rank-3.
- [ ] Pretrained checkpoint loads via `register_load_state_dict_pre_hook`
  — no missing keys, no unexpected keys, no manual reshaping at the
  call site.
- [ ] `uv run pytest tests/` green, including the new rank-4 tests
  AND the existing Decoder smoke
  ([`tests/test_adain1d_decoder_smoke.py`](../../tests/test_adain1d_decoder_smoke.py)).
- [ ] `coreml/kokoro_decoder_har_post_3s.mlpackage` and
  `coreml/kokoro_decoder_har_post_10s.mlpackage` re-exported; export
  numeric gate green for both buckets.
- [ ] [`scripts/compare_decoder_har_post_waveforms.py`](../../scripts/compare_decoder_har_post_waveforms.py)
  gates pass on both buckets against the `/tmp` baseline:
  **Pearson > 0.99**, **SNR ≥ 40 dB**, **max abs Δ ≤ 1e-2**.
- [ ] Xcode Performance Report **Neural Engine column count > 0** on
  `kokoro_decoder_har_post_10s.mlpackage`. Screenshot saved to
  `outputs/ane_rank4/`.
- [ ] Python probe: `.all` sha256(out) **differs from** `.cpuAndGPU`
  sha256(out). Cold-load `.all` ≤ 5 s.
- [ ] Espresso `Unsupported op` event count on `.all` load **≤ 50** (down
  from 322).
- [ ] Results-log row appended to this plan.

### Definition of Done

- [ ] All Phase 0–3 tasks checked.
- [ ] Hard Requirements all checked.
- [ ] `README/Notes/ane-decoder-har-post-investigation.md` updated with
  a "Resolved" footer that points back to this plan and the merged PR.
- [ ] Either the
  [`ane-optimization-v1.md` results-log](./ane-optimization-v1.md#results-log-commit-this)
  is annotated with a "superseded by rank-4" pointer, or that plan is
  marked **Complete** if its own success criteria are still satisfied
  (Phase 1–3 of `ane-optimization-v1.md` were `[x]` before the
  `5278e88` revert; cross-check whether to revive or retire).

## Open Questions

### Resolved

- **Q:** Should `AdaIN1d` be rewritten in place to rank-4 or should a new
  sibling `AdaIN2d` be introduced?
- **A:** New sibling `AdaIN2d`. `AdaIN1d` is shared with the Decoder
  path (`AdainResBlk1d`), which is out of scope for **implementation**
  here but is explicitly **in scope for design-reuse** — see
  [Retrofit-Readiness for `decoder_pre` Rewrite](#retrofit-readiness-for-decoder_pre-rewrite).
  Keeping `AdaIN1d`'s rank-3 contract untouched avoids cross-coupling
  the HAR-post and decoder_pre rewrites at the module boundary. The
  follow-on decoder_pre plan **will** point `AdainResBlk1d.norm1` /
  `.norm2` at this plan's `AdaIN2d`; option (b) ("leave `AdaIN1d` and
  rewrite again") is rejected in advance — duplicate rank-4 AdaIN
  classes would be the wrong kind of complexity.
- **Q:** Does this plan need to update mask threading
  (`_align_mask_to`)?
- **A:** No. Current
  [`GeneratorFromHar.forward(x_pre, ref_s, har)`](../../export_synth/wrappers.py#L87)
  has no mask parameter on `main`. The investigation brief's mention of
  `_align_mask_to` referred to a branch that has not been merged. Grep
  confirms no such method exists in this tree.
- **Q:** Should we bundle Step 4(b) (fp16 input dtypes) in this PR?
- **A:** No. Bundling adds runtime-caller churn (Python harness, Swift
  pipeline, bench scripts) before we know the rank-4 rewrite is
  insufficient on its own. If Phase 3 verification shows residual
  boundary-cast rejections, fp16 inputs become a follow-on plan
  (`ane-decoder-har-fp16-inputs-v1.md`). See
  [user framing in the investigation note](../Notes/ane-decoder-har-post-investigation.md#33-why-the-entry-cast-is-a-red-herring-not-the-root-cause).
- **Q:** Should `Generator.forward` be updated too, or only
  `GeneratorFromHar.forward`?
- **A:** Only `GeneratorFromHar.forward` does the rank-4 promotion. The
  underlying `Generator` modules become rank-4 (their `__init__`
  changes), but `Generator.forward` is the training path and not the
  inference entry point. If anyone later traces `Generator.forward`
  directly (e.g., `mode == "full"` in
  [`export_synth/convert.py`](../../export_synth/convert.py)), the same
  unsqueeze/squeeze wrapping should be applied at that call site — flag
  this when the time comes.
- **Q:** Is `cp=65568` in the Espresso `Loaded network` log line a
  smoking gun for fp32 input boundary?
- **A:** Insufficient evidence in this plan's scope. `cp=65568` =
  `0x10020`, which decodes as FLOAT32 in coremltools' precision enum,
  but Espresso's `cp` field semantics aren't fully documented in the
  public API. Treat as suggestive, not load-bearing. The rank-4 fix
  doesn't depend on resolving this.

### Unresolved

- **Q:** Will `nn.ConvTranspose2d` with stride `(1, u)` and padding
  `(0, (k-u)//2)` produce a single MIL `conv_transpose` op on the
  current coremltools 8.3.0 + macOS 26 stack, or does it lower to a
  `conv_transpose` + `reshape` combo that fragments ANE eligibility?
- **Options:** Phase 0's `--probe-conv-lowering` extension answers this.
  If it fragments, fall back to (a) `nn.ConvTranspose1d` with explicit
  unsqueeze/squeeze around it (worse — re-introduces reshape ops in
  the body), or (b) `nn.Upsample(scale_factor=u, mode='linear')` +
  `nn.Conv2d` to emulate, with a load hook that re-flows the
  ConvTranspose1d weight into the Upsample+Conv2d pair. **Current lean:**
  expect ConvTranspose2d to lower cleanly; if not, prefer (b) over (a).
- **Q:** Will `tile` (96 occurrences in rank-3 baseline) drop to zero
  after rank-4, or does some tiling persist (e.g., for repeating the
  style vector along T inside AdaIN2d)?
- **Options:** Inspect the rank-4 MIL histogram in Phase 3. If `tile`
  persists and is unsupported on ANE, evaluate whether explicit
  broadcasting (rely on PyTorch's implicit rank-4 broadcast) can
  eliminate it. **Current lean:** `tile` drops substantially but not
  to zero; revisit only if Phase 3 ANE count is partial.
- **Q:** Should this plan move the investigation scratch tools under
  `/tmp/ane-investigation/` (probe.py, capture_logs.sh, dump_mil_ops.py)
  into `scripts/` for permanent reuse?
- **Options:** (a) yes, harden and add as `scripts/probe_ane_engagement.py`
  etc.; (b) no, treat them as one-shot investigation scratch. **Current
  lean:** (a) — they will be re-used in any future ANE investigation,
  and the cost of hardening (~30 LOC of arg handling) is small. Do this
  in Phase 3 as a pre-commit task, not as a separate phase.

## References

### Internal

- [ANE decoder-har-post investigation](../Notes/ane-decoder-har-post-investigation.md)
  — root-cause analysis that motivates this plan (today's findings).
- [ANE Graph Optimization Plan](./ane-optimization-v1.md) — prior
  Phase 0/1/2/3 work (Linear → Conv1d, since-reverted). Template
  source for this plan.
- [Core ML Compute Unit Scheduling Guide](../Guides/apple-silicon/CoreML-Compute-Unit-Scheduling-guide.md)
  — `.all` / `.cpuAndGPU` / `.cpuAndNeuralEngine` semantics, silent
  fallback verification, `powermetrics` recipe.
- [Core ML compute-unit ablation](../Notes/coreml-compute-unit-ablation.md)
  — F/G/G-prime/G-double-prime cross-machine ablation; the open
  hypothesis #3 (silent fallback) is what this plan addresses.
- [CLAUDE.md, Part 4.1 — ANE memory layout](../../CLAUDE.md) —
  canonical `(B, C, 1, T)` rule with `T` largest; 64-byte alignment
  penalty.
- [`scripts/count_mil_ops.py`](../../scripts/count_mil_ops.py),
  [`scripts/compare_decoder_har_post_waveforms.py`](../../scripts/compare_decoder_har_post_waveforms.py),
  [`scripts/bench_decoder_har_post_predict.py`](../../scripts/bench_decoder_har_post_predict.py),
  [`tests/test_adain1d_linear_vs_conv1d.py`](../../tests/test_adain1d_linear_vs_conv1d.py),
  [`tests/test_adain1d_decoder_smoke.py`](../../tests/test_adain1d_decoder_smoke.py)
  — existing infrastructure that this plan reuses.

### External

- Apple Research — [Deploying Transformers on the ANE](https://machinelearning.apple.com/research/neural-engine-transformers)
  — canonical "rank-4 `(B, C, 1, S)` with `S` largest" guidance.
- [Orion: Characterizing Apple's Neural Engine](https://arxiv.org/abs/2603.06728)
  — ANE op support catalog and alignment constraints (referenced from
  [ane-optimization-v1.md](./ane-optimization-v1.md)).

## Files Likely to Change

| File | Change Type | Notes |
| --- | --- | --- |
| [`kokoro/istftnet.py`](../../kokoro/istftnet.py) | Modify | Add **`AdaIN2d`** (rank-4 sibling of `AdaIN1d`) and shared **`_rank3_to_rank4_conv_state_dict`** helper (designed for reuse by the future `decoder_pre` rewrite); convert `AdaINResBlock1` to Conv2d/ConvTranspose2d/`AdaIN2d`; convert `Generator.__init__` modules to rank-4; register load hooks on `Generator` and `AdaINResBlock1` that call the shared helper with their own key lists. **Do not touch** `AdaIN1d`, `AdainResBlk1d`, `Decoder.encode`/`Decoder.decode`, `Decoder.F0_conv`/`Decoder.N_conv`/`Decoder.asr_res`. |
| [`export_synth/wrappers.py`](../../export_synth/wrappers.py) | Modify | `GeneratorFromHar.forward`: unsqueeze at entry, change `.size(2)` → `.size(3)` and the slice index, squeeze at exit before `gen.stft.inverse`. |
| [`scripts/count_mil_ops.py`](../../scripts/count_mil_ops.py) | Modify | Extend `--probe-conv-lowering` to also probe Conv2d(1, k) and ConvTranspose2d((1, k), stride=(1, u)) for real k/u values. |
| `tests/test_adain2d_vs_adain1d.py` | Create | New `AdaIN2d` ↔ `AdaIN1d` parity test at fp32 tolerance. |
| `tests/test_generator_from_har_rank4.py` | Create | Rank-3 vs rank-4 `GeneratorFromHar` parity on real inputs. |
| `tests/test_rank4_checkpoint_load.py` | Create | `hexgrad/Kokoro-82M` checkpoint loads into rank-4 Generator with no missing/unexpected keys. |
| `coreml/kokoro_decoder_har_post_3s.mlpackage` | Rebuild | Phase 2 re-export. |
| `coreml/kokoro_decoder_har_post_10s.mlpackage` | Rebuild | Phase 2 re-export. |
| `outputs/ane_rank4/results.json` | Create | Gitignored Phase 3 metrics (probe sha256s, cold-load times, MIL histogram). |
| `outputs/ane_rank4/xcode_compute_unit_map_10s.png` | Create | Gitignored Phase 3 Xcode screenshot. |
| `README/Notes/ane-decoder-har-post-investigation.md` | Modify | Add "Resolved by [this plan](../Plans/ane-decoder-har-rank4-rewrite-v1.md)" footer after Phase 3 succeeds. |
| `scripts/probe_ane_engagement.py` | Optionally create | If Open Question (c) resolves to "yes, harden tools." |

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| `nn.ConvTranspose2d((1, k), stride=(1, u))` doesn't lower to a single MIL `conv_transpose` on coremltools 8.3.0 / macOS 26 | Phase 0 `--probe-conv-lowering` extension catches this before Phase 1. Fallback path documented in Open Question (a). |
| Load hook regresses on `weight_norm`-wrapped weights | Test against the real `hexgrad/Kokoro-82M` checkpoint in `tests/test_rank4_checkpoint_load.py` (not synthetic state_dict). The kokoro checkpoint uses `weight_norm`; if the hook misses `weight_orig` / `weight_v`, the test will catch it. |
| Rank-4 graph passes PyTorch parity but Core ML fp16 export diverges (fp16 rounding in AdaIN reductions) | Phase 3 escalation: re-export with `--precision fp32` to isolate; document per-layer drift; consider mixed precision via `op_selector` for AdaIN. |
| Pseudo-success: Xcode count moves from 0 to a few hundred but the real-world latency doesn't improve | Hard Requirements include cold-load drop (≤ 5 s) AND Espresso `Unsupported op` event drop (≤ 50). Neither moves without genuine engagement. Predict-only ms is reported but not gating. |
| Sharing `Generator` modules between training (rank-3) and inference (rank-4) breaks anyone retraining Kokoro | Documented as a Non-Goal. The load hook is idempotent (3D→4D once; 4D→4D no-op), so retraining downstream would require either reversing the hook or rebuilding from rank-3 — fine for the inference-only-shipping posture. |
| Decoder path silently regresses because someone later edits `AdaIN1d` thinking it's "the rank-4 one" | LLM-first comment on both `AdaIN1d` and `AdaIN2d`: state that they are **deliberately distinct** classes for rank-3 Decoder vs rank-4 Generator, and that merging them couples HAR-post and decoder_pre exports. |
| The `tile` count doesn't drop and ANE engagement stays partial | Escalation path #1 (specific-op follow-on plan). Phase 3 inspects residual unsupported indices. |
| `.all` cold-load is still slow because of GPU class-load races even after ANE engages | Cold-load gate is ≤ 5 s, not ≤ 0.5 s. Some GPU class initialization is normal. Drop from 26.485 s by an order of magnitude is the real signal. |

## Results log (commit this)

| Run | git SHA | coremltools | torch | Hardware | OS | Xcode NE count | Cold-load `.all` (s) | `.all` sha vs `.gpu` | Pearson 3s / 10s | SNR / Δ | MIL note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline (rank-3) | `842473e` | 8.3.0 | 2.6.0 | M3 Max 36 GB | macOS 26.4 | 0 / 1238 | 26.485 | match (silent GPU fallback) | n/a | n/a | `linear` 48, `conv` 51, `tile` 96 |
| Rank-4 | _fill in PR SHA_ | _fill in_ | _fill in_ | _fill in_ | _fill in_ | _fill in_ | _fill in_ | _fill in_ | _fill in_ | _fill in_ | _fill in_ |

## Rollback

- **How to revert (Phase 1):** `git revert <conv1d-to-conv2d commit>`
  reverses the module changes. The load hook is idempotent for either
  direction; pre-rewrite checkpoints still load on rank-3 (no-op) and
  rank-4 (reshape).
- **How to revert (Phase 2):** `git checkout <pre-rewrite SHA> --
  coreml/kokoro_decoder_har_post_{3s,10s}.mlpackage` to restore the
  baseline packages. Or re-export from the reverted Phase 1 source.
- **Time to rollback:** < 5 min for source; ~2 min for re-export.
- **Data recovery needed:** no. No persistent state outside of the
  Core ML packages and git history.
