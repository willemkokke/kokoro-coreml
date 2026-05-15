# ANE Rank-4 Rewrite — `decoder_har_post` Plan

**Date:** 2026-05-15
**Status:** Executed (all 4 phases ran end-to-end); **ANE engagement gate
failed** — follow-on plan
`README/Plans/ane-decoder-har-fp16-inputs-v1.md` covers the next
iteration (Step 4(b)). Real wins kept: shape-inference fix, 25×
cold-load speedup, –9% MIL op count, audio-quality parity preserved. See
[Conclusion and next iteration](#conclusion-and-next-iteration).

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
  **FAILED (2026-05-15):** 0 / 948 on M3 Max / macOS 26.4 /
  `ct.target.macOS13`. Rank-4 alone did not engage ANE; the follow-on
  plan `ane-decoder-har-fp16-inputs-v1.md` tackles the next lever.
- [ ] **No more silent fallback.** Python probe
  ([`/tmp/ane-investigation/probe.py`](../Notes/ane-decoder-har-post-investigation.md#6-reproducer))
  shows `.all` output sha256 **differs** from `.cpuAndGPU` (a different
  hash on warm predict means the compute path differs).
  **FAILED:** `.all` and `.cpuAndGPU` both produce sha
  `28041dfea5c8b6e1`. Same disqualification → silent GPU fallback as
  rank-3, just much cheaper.
- [x] **Cold-load recovery.** `.all` cold-load time drops from ~26 s to
  the same order as `.cpuAndGPU` (≤ 2 s target; ≤ 5 s acceptable).
  **PASSED:** 26.485 s → 1.075 s (**25× faster**, well under the 5 s
  acceptable bar and inside the 2 s stretch target). The ANE compile
  attempt now fast-fails in a single segmentation pass with zero
  retries.
- [x] **PyTorch parity.** Rank-4 `GeneratorFromHar` output matches the
  rank-3 baseline on real `(x_pre, ref_s, har)` inputs to fp32 rounding
  tolerance (existing
  [`scripts/compare_decoder_har_post_waveforms.py`](../../scripts/compare_decoder_har_post_waveforms.py)
  gates: Pearson **r > 0.99**, **SNR ≥ 40 dB**, **max abs Δ ≤ 1e-2**).
  **PASSED on 10s** (Pearson 0.999994, SNR 49.90 dB, max abs Δ
  3.17e-3). 3s waveform parity skipped because no pre-rewrite 3s
  baseline was preserved on disk; per-bucket rank-4 architecture is
  identical so 10s passing covers the architectural claim.
- [x] **Checkpoint compatibility.** Pretrained `hexgrad/Kokoro-82M`
  weights load without retraining via
  `register_load_state_dict_pre_hook` that reshapes Conv1d weights
  `(C_out, C_in, k)` → `(C_out, C_in, 1, k)` and `alpha` parameters
  `(1, C, 1)` → `(1, C, 1, 1)`.
  **PASSED:** `tests/test_rank4_checkpoint_load.py` verifies no
  missing learnable keys and no unexpected keys; subsequent forward is
  finite. The shared `_rank3_to_rank4_conv_state_dict` helper handles
  bare `weight`, legacy `weight_g`/`weight_v`, and new
  `parametrizations.weight.original0/1` key forms.
- [x] **Both shipping buckets re-exported.** `coreml/kokoro_decoder_har_post_3s.mlpackage`
  and `coreml/kokoro_decoder_har_post_10s.mlpackage` rebuilt with the new
  rank-4 graph.
  **PASSED.** Export numeric gate (traced vs Core ML, all finite)
  green for both buckets.
- [x] **Results-log row appended** to the section at the bottom of this
  plan, matching the
  [`ane-optimization-v1.md` results-log style](./ane-optimization-v1.md#results-log-commit-this).
  **Done.**

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

- [x] Re-run `uv run python scripts/count_mil_ops.py coreml/kokoro_decoder_har_post_10s.mlpackage`
  and confirm the 2207-op rank-3 baseline histogram matches the
  [Fresh Baseline](#fresh-baseline-current-state) above. **Done
  2026-05-15:** baseline confirmed exactly — 2207 total ops; all per-type
  counts match the Fresh Baseline (`const` 1166, `add` 218, `mul` 148,
  `tile` 96, `reduce_mean` 88, `conv` 51, `sin` 50, `linear` 48,
  `reshape` 48, `split` 48, `pow` 48, `sub` 45, `square` 44, `sqrt` 44,
  `real_div` 44, `slice_by_index` 6, `cast` 5, `conv_transpose` 4,
  `leaky_relu` 3, `pad` 1, `exp` 1, `cos` 1). JSON saved at
  `outputs/ane_rank4/phase0_baseline_histogram_10s.json` (gitignored).
- [x] Extend
  [`scripts/count_mil_ops.py`](../../scripts/count_mil_ops.py)'s
  `--probe-conv-lowering` mode to also probe `Conv2d(in, out, (1, k))`
  for the real Generator kernels and `ConvTranspose2d(in, out, (1, k))`
  with stride `(1, u)`. **Done 2026-05-15:** the probe now covers 15
  cases:
  - Conv k=1 baseline.
  - AdaINResBlock1 convs1: k ∈ {3, 7, 11} × dilation ∈ {1, 3, 5} (9 cases).
  - noise_convs[0]: k=12 stride=6 padding=3.
  - conv_post: k=7 stride=1 padding=3.
  - ups[0]: ConvTranspose k=20 stride=10 padding=5.
  - ups[1]: ConvTranspose k=12 stride=6 padding=3.
  - ReflectionPad1d((1, 0)) vs ReflectionPad2d((1, 0, 0, 0)).

  All 15 cases lower to the **same MIL op-type set** between rank-3 and
  rank-4 variants (`{cast, const, conv}` for conv cases,
  `{cast, const, conv_transpose}` for transpose cases,
  `{cast, const, pad}` for the reflection pad case). `all_equivalent:
  true` — exit code 0. Full JSON saved at
  `outputs/ane_rank4/phase0_probe_results.json` (gitignored).
  **Open Question (a) resolved in the affirmative**: ConvTranspose2d
  with `(1, k)` and stride `(1, u)` does lower to a single MIL
  `conv_transpose` op on coremltools 8.3.0 + macOS 26.4 + torch 2.6.0.
  No fragmentation, no extra reshape / transpose ops, no fallback to
  the `Upsample + Conv2d` alternative needed.
- [x] **Lock taxonomy:** the extended probe is committed in this phase.
  Phase 3 compares op histograms using the same `op.type` strings
  produced by this script.

**Verification:** Conv2d(1, k) and ConvTranspose2d((1, k), stride=(1, u))
both lower to a single MIL `conv` / `conv_transpose` op for the real
kernel/stride sizes used in `kokoro/istftnet.py::Generator`. Histogram
delta vs the baseline is zero (re-run on the same package returns the
same counts).

---

### Phase 1: Rank-4 module rewrite + PyTorch parity

**Goal:** Replace the Generator path's rank-3 modules with rank-4
counterparts, keep pretrained checkpoint loading working, and prove
numerical equivalence in PyTorch before touching Core ML.

**Tasks (`kokoro/istftnet.py`):**

- [x] Add new class `AdaIN2d` next to
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
- [x] Convert `AdaINResBlock1` (
  [line 153](../../kokoro/istftnet.py#L153)) to rank-4:
  - `self.convs1` / `self.convs2`: `nn.Conv1d(C, C, k, 1, dilation=d, padding=p)`
    → `nn.Conv2d(C, C, (1, k), 1, dilation=(1, d), padding=(0, p))`.
  - `self.adain1` / `self.adain2`: now hold `AdaIN2d` instances.
  - `self.alpha1` / `self.alpha2`: shape `(1, channels, 1)` → `(1, channels, 1, 1)`.
  - Forward: same control flow, no shape changes needed — convs and
    AdaIN are now rank-4 native; `torch.sin(a * xt) ** 2` works
    elementwise on rank-4.
- [x] Convert `Generator` (
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
  - **Deviation from original plan, made in Phase 1c (2026-05-15):**
    `Generator.forward` is NOT left alone — it now does the rank-3 →
    rank-4 → rank-3 promotion internally. Reason: `Decoder.forward` at
    [`kokoro/istftnet.py::Decoder` line 562](../../kokoro/istftnet.py#L562)
    calls `self.generator(x, s, F0_curve)` with rank-3 `x`, so leaving
    `Generator.forward` rank-3 while the inner modules are rank-4 would
    have broken the Decoder path (which is in-scope to keep working).
    The boundary pattern in `Generator.forward` is identical to
    `GeneratorFromHar.forward`: `unsqueeze(-2)` on `x` and `har` at the
    body entry, `squeeze(-2)` on `x` after `conv_post`, then the existing
    rank-3 spec/phase slicing + `stft.inverse` call. The integration
    test
    [`tests/test_export_wrappers_shapes.py::test_synthesizer_model_forward_runs_and_returns_1d_audio`](../../tests/test_export_wrappers_shapes.py)
    exercises this Decoder → Generator path and passes after the
    rewrite.
- [x] Add a module-level helper `_rank3_to_rank4_conv_state_dict` next to
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
- [x] Register `register_load_state_dict_pre_hook` on `AdaINResBlock1`
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

- [x] Convert
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
- [x] Document the rank promotion at the top of `GeneratorFromHar` so a
  future reader sees the rank-4 internal invariant explicitly (LLM-first
  doc style per [CLAUDE.md](../../CLAUDE.md)).

**Tasks (tests):**

- [x] Add `tests/test_adain2d_vs_adain1d.py`: rank-3 `AdaIN1d` reference
  vs rank-4 `AdaIN2d` on synthetic `(B, C, T)` input
  (unsqueezed/squeezed for the rank-4 path);
  `torch.allclose(rank3_out, rank4_out.squeeze(-2), atol=1e-5, rtol=1e-5)`.
  **Done:** 5 tests cover the (channels, style_dim, seq_len) matrix
  `[(128, 128, 64), (256, 128, 128), (128, 64, 32)]` plus AdaIN2d's
  H=1-axis and channel-mismatch assertions.
- [ ] **Deferred to Phase 2 (waveform parity gate)** — `tests/test_generator_from_har_rank4.py`:
  rank-3 `GeneratorFromHar` reference vs rank-4 rewrite on real inputs.
  **Deviation:** the rank-3 `GeneratorFromHar` reference no longer
  exists in-process (it was overwritten by the rank-4 rewrite, by
  design — the boundary preserves the rank-3 public API but the body
  is rank-4 only). An in-process diff would require restoring rank-3
  code as a separate class, which contradicts the
  "do not duplicate AdaIN2d" decision in
  [Open Questions §Resolved](#resolved). The cross-runtime parity gate
  in [Phase 2](#phase-2-re-export-kokoro_decoder_har_post_3s10smlpackage)
  via
  [`scripts/compare_decoder_har_post_waveforms.py`](../../scripts/compare_decoder_har_post_waveforms.py)
  against the pre-rewrite `/tmp/...baseline.mlpackage` files is the
  actual numerical parity check (Pearson > 0.99, SNR ≥ 40 dB, max abs
  Δ ≤ 1e-2). The Phase 1 smoke gate at
  `tests/test_rank4_checkpoint_load.py::test_rank4_generator_forward_after_load_is_finite`
  covers the "forward runs and is finite" end of the test pyramid.
- [x] Add `tests/test_rank4_checkpoint_load.py`: load real
  `hexgrad/Kokoro-82M` checkpoint (the same one
  `export_synth/main.py` loads) into the rank-4 Generator; assert no
  missing keys (excluding `stft.*` internal buffers which were never
  in the checkpoint), no unexpected keys, and a forward pass on
  synthetic rank-4 input is finite. **Done:** 2 tests; both pass when
  the HF snapshot is cached locally and skip otherwise.
- [x] Re-run the existing
  [`tests/test_adain1d_decoder_smoke.py`](../../tests/test_adain1d_decoder_smoke.py)
  and
  [`tests/test_export_wrappers_shapes.py`](../../tests/test_export_wrappers_shapes.py)
  to confirm the Decoder path is unaffected (since `AdaIN1d` is
  unchanged and only the Generator path moved to `AdaIN2d`).
  **Done:** both files green;
  `test_synthesizer_model_forward_runs_and_returns_1d_audio` in
  particular exercises the full Decoder → Generator path through the
  new rank-3 → rank-4 → rank-3 boundary.
- [x] `uv run python -m pytest tests/ -q` — full suite green.
  **2026-05-15:** 41 passed, 9 skipped, 0 failed. Skips are
  environment-conditional (decoder_pre / decoder_only / kokoro_duration
  mlpackages not on disk; Phase 2 re-export work).

**Verification:** PyTorch parity tests pass at fp32 tolerance; full
pytest green; pretrained checkpoint loads with no missing/unexpected
keys.

---

### Phase 2: Re-export `kokoro_decoder_har_post_{3s,10s}.mlpackage`

**Goal:** Produce the rank-4 Core ML packages and prove waveform parity
against the pre-rewrite baseline.

**Tasks:**

- [x] Copy current `coreml/kokoro_decoder_har_post_10s.mlpackage` to
  `/tmp/kokoro_decoder_har_post_10s.baseline.mlpackage` before
  re-exporting. **Deviation:** the in-tree `coreml/` folder is
  gitignored and on this machine only had the rank-3 10s package on
  disk (the 3s package was never present pre-rewrite), so only the 10s
  baseline could be saved. The 3s waveform parity gate is therefore
  skipped; the rank-4 rewrite is bucket-agnostic at the architecture
  level and the strong 10s parity result is sufficient evidence the
  rewrite is numerically correct.
- [x] Re-export:
  `uv run --no-sync python -m export_synth.main --mode decoder-har --buckets 3s,10s -o coreml`.
  **Done 2026-05-15:** both packages saved to
  `coreml/kokoro_decoder_har_post_{3s,10s}.mlpackage`.
- [x] **Export gates (existing):** the export script's "decoder-har
  numeric gate: traced vs Core ML waveform shape ..., all finite" line
  must pass for both buckets. **Done:** `3s (72000,)` and
  `10s (240000,)`, both all finite.
- [x] **Waveform parity** (10s only, see deviation above):
  `uv run --no-sync python scripts/compare_decoder_har_post_waveforms.py
  --baseline /tmp/kokoro_decoder_har_post_10s.baseline.mlpackage
  --candidate coreml/kokoro_decoder_har_post_10s.mlpackage
  --bucket-sec 10 --text "..."`. **Done 2026-05-15:**
  - **Pearson: 0.999994** (gate: > 0.99)
  - **SNR: 49.90 dB** (gate: ≥ 40 dB)
  - **max abs Δ: 3.17e-3** (gate: ≤ 1e-2)

  All three gates pass with substantial margin. The rank-4 rewrite is
  numerically equivalent to the rank-3 baseline within fp16 export
  tolerance.
- [x] `uv run --no-sync python -m pytest tests/test_mlpackage_exports.py -q`.
  **Done:** 2 passed, 8 skipped (skips need `decoder_pre_*` /
  `decoder_only_3s` / `kokoro_synthesizer_3s` / `kokoro_duration`
  packages — all out of scope for this plan).

**Verification:** Both rank-4 packages save; export numeric gate passes
for both; 10s waveform parity vs `/tmp` baseline passes all three gates
with substantial margin (Pearson 0.999994 > 0.99, SNR 49.9 ≥ 40, max
abs Δ 3.17e-3 ≤ 1e-2).

**MIL diff vs baseline (10s):** rank-3 had **2207 ops**; rank-4 has
**2021 ops** (–186 net, –9%).

- `tile`: **96 → 0** (eliminated — rank-4 PyTorch broadcasting
  replaces the explicit tile ops that the rank-3 graph used to expand
  AdaIN gamma/beta over T).
- `const`: 1166 → 1073 (–93 boilerplate consts paired with the tile
  chains).
- `linear`: 48 (unchanged — `AdaIN2d.fc` is still `nn.Linear` as
  planned).
- New boundary ops: `expand_dims: 2` (the two unsqueeze(-2) calls at
  GeneratorFromHar's body entry) and `squeeze: 1` (before
  `stft.inverse`). Three boundary ops for ~93 internal `tile`/`const`
  ops eliminated — a net win even ignoring the ANE engagement
  motivation.

Histograms saved at `outputs/ane_rank4/phase2_rank4_histogram_{3s,10s}.json`.

---

### Phase 3: ANE placement verification

**Goal:** Prove the rank-4 graph engages the Neural Engine. **This is
the new hard gate that did not exist in
[`ane-optimization-v1.md`](./ane-optimization-v1.md). 0 ops on Neural
Engine is failure.**

**Tasks (graph + placement):**

- [x] **Xcode Performance Report:** opened the re-exported
  `coreml/kokoro_decoder_har_post_10s.mlpackage` in Xcode. **Result
  (2026-05-15, M3 Max / macOS 26.4):** `All: 948  CPU: 0  GPU: 948
  Neural Engine: 0`. **HARD GATE FAILED — Neural Engine count is still
  0.** Every visible op (`ios16.cast`, `ios16.conv`,
  `ios16.leaky_relu`, `ios16.reduce_mean`, `expand_dims`,
  `slice_by_index`, `ios16.sub`, `ios16.square`) has an empty diamond
  in the Neural Engine column — graph-level disqualification, same
  pattern as rank-3. Median Prediction 103.22 ms / Load 37.47 ms /
  Compilation 112.76 ms. Op count dropped 1238 → 948 (–23%) but
  ANE engagement is unchanged.
- [x] **Python probe triplet:** ran with units `all`, `gpu`, `ne` in
  fresh subprocesses. **Results:**

  | Units | Cold load (s) | Warm predict (s) | sha256(out) |
  | --- | ---: | ---: | --- |
  | `.all` | **1.075** | 0.110 | `28041dfea5c8b6e1` |
  | `.cpuAndGPU` | 0.515 | 0.106 | `28041dfea5c8b6e1` (matches `.all`) |
  | `.cpuAndNE` | **4.495** | 0.288 | `a7968e9334aab13f` (differs) |

  **Sha-differs gate FAILED** — `.all` still bit-equals `.cpuAndGPU`,
  confirming silent GPU fallback persists.
  **Cold-load gate PASSED with margin** — `.all` cold load is **1.075 s**
  (gate: ≤ 5 s; baseline was 26.485 s, **25× improvement**).
  `.cpuAndNE` cold load is 4.495 s (baseline was 420 s, **93× faster**)
  — the ANE compile attempt fails fast in one segmentation pass instead
  of exhausting retries. The dramatic cold-load improvement is real
  even though ANE engagement itself is unchanged.
- [x] **Espresso log stream:** captured during `.all` load.
  **Results:**
  - **`Shape computation issue at layer N` events: 12 → 0** ✓
    (rank-3 fired this 4× each at layers 37 / 44 / 51 — those sites are
    completely gone, confirming the rank-4 rewrite did fix the shape
    inference disqualifier).
  - **Retries gone:** rank-3 had 6 `Failed to create E5 execution stream
    operation ... Retry 1/2 / Retry 2/2` events and 4 silent
    class-fallback events. Rank-4 has zero retries and a single
    segmentation attempt. The compiler now fast-fails on the rank-4
    graph instead of exhausting retries — this explains the 25× cold-load
    speedup.
  - **`Unsupported op N` events: 322 → 298** (–24, ~7.5% drop).
    **HARD GATE FAILED** — gate was ≤ 50; we're at 298. Indices span
    [7..948] (was [7..1041]). Per-op rejection has a different cause
    than rank-3 shape inference: rank-4 conv ops on perfect
    last-axis-largest shapes like `1×256×1×8000` are STILL rejected,
    as are AdaIN math ops (`reduce_mean → 1×256×1×1`, `sub`, `square`,
    `sqrt`, `real_div`, `mul`), `sin` (Snake1D), and a
    `reshape → 1×512×1×1` (`AdaIN2d.fc.view(B, 2C, 1, 1)`). Per-op
    Xcode columns confirm: even `ios16.conv` and `ios16.leaky_relu` on
    rank-4 are not ANE-eligible on this OS+target combo. Captured
    log saved at `outputs/ane_rank4/phase3_log_all.ndjson` (gitignored).
- [x] **MIL op histogram (rank-4):** captured in Phase 2; see
  [Phase 2 MIL diff](#phase-2-re-export-kokoro_decoder_har_post_3s10smlpackage)
  above. Total 2207 → 2021 ops; `tile` 96 → 0; `linear` 48 (unchanged);
  new boundary ops `expand_dims: 2`, `squeeze: 1`.

**Tasks (wall-clock — fallback path):**

- [x] Wall-clock predict (rank-3 vs rank-4) — already covered by the
  probe triplet above: warm predict went 0.116 s (rank-3 .all) →
  0.110 s (rank-4 .all), essentially unchanged. Both run on GPU
  (MPSGraph) since ANE never engaged. **Skipping
  `scripts/bench_decoder_har_post_predict.py --baseline`** — the
  probe data already shows GPU-on-GPU parity in predict time and
  there is no ANE-vs-GPU comparison to make.
- [ ] **Skipped:** `sudo powermetrics -i 1000 --samplers ane` — would
  require interactive sudo and would only confirm what Xcode already
  shows (zero ANE power → zero ANE engagement).
- [ ] **Skipped:** `MLComputePlan` Swift snippet — Xcode's Compute Unit
  Mapping is the same data source. Not informative when Xcode already
  shows 0 / 948 on Neural Engine.

**Tasks (results log):**

- [x] Append a row to the [Results log](#results-log-commit-this)
  section at the bottom of this plan.

**Verification (2026-05-15, M3 Max / macOS 26.4):**

| Gate | Target | Result | Verdict |
| --- | --- | --- | --- |
| Xcode Neural Engine count > 0 | required | 0 / 948 | **FAIL** |
| `.all` sha differs from `.cpuAndGPU` | required | identical (`28041dfea5c8b6e1`) | **FAIL** |
| `.all` cold load ≤ 5 s | required | 1.075 s | **PASS** (25× faster vs rank-3) |
| Espresso `Unsupported op` count ≤ 50 | required | 298 | **FAIL** (was 322) |
| Waveform parity (10s) | from Phase 2 | Pearson 0.999994 / SNR 49.90 dB / max abs Δ 3.17e-3 | **PASS** |
| `Shape computation issue` events ≤ 0 | (informal) | 0 | **PASS** (was 12) |

**Two of six gates fail. The rank-4 rewrite is necessary but not
sufficient on macOS 26.4 / iOS16-target.** See
[Conclusion and next iteration](#conclusion-and-next-iteration) below.

#### Escalation paths (pre-named so we don't scramble)

- **If Xcode Neural Engine count > 0 but partial (say 30-80% of ops):**
  inspect the residual `Unsupported op` indices via
  `/tmp/ane-investigation/capture_logs.sh` + `dump_mil_ops.py`. Identify
  which op-types are still rejected. Likely candidates on the rank-4
  graph: boundary `cast` (the fp32→fp16 input casts and the fp16→fp32
  output cast — the entry-cast hypothesis from
  [the investigation note §3.3](../Notes/ane-decoder-har-post-investigation.md#33-why-the-entry-cast-is-a-red-herring-not-the-root-cause)),
  `slice_by_index` (6 occurrences, related to the `ref_s[:, :128]`
  style slice and the spec/phase split after conv_post), and the new
  rank-4 boundary ops `expand_dims` (2 at GeneratorFromHar's body
  entry) / `squeeze` (1 before `stft.inverse`). Open a follow-on plan
  named `ane-decoder-har-residual-ops-v1.md` to tackle the specific
  op-types. **Do not** speculatively bundle fp16 inputs into this PR.
  (Note: residual `tile` is no longer a candidate — Phase 2 confirmed
  `tile` 96 → 0 on rank-4; see the Resolved entry above.)
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

### Conclusion and next iteration

**Verdict (2026-05-15):** the rank-4 rewrite is **necessary but not
sufficient** on M3 Max / macOS 26.4 / `ct.target.macOS13`. ANE engagement
remains 0 / 948. **However**, the rewrite is keep-able on its own
merits:

What rank-4 fixed (real wins, independent of ANE):

- `Shape computation issue at layer N` events: **12 → 0**. The rank-3
  graph fired these 4× each at layers 37 / 44 / 51 (AdaIN's
  reduce_mean / split_sizes / scalar consts). The rank-4 graph emits
  zero. The shape-inference disqualifier called out in the
  investigation note is genuinely resolved.
- `.all` cold-load time: **26.485 s → 1.075 s (25× faster).** The ANE
  compiler now fast-fails in one segmentation pass instead of
  exhausting 6 retry attempts and 4 silent class-fallbacks. End users
  feel this on every cold model load even though ANE never engages.
- `.cpuAndNE` cold-load time: **420.194 s → 4.495 s (93× faster).** The
  same fast-fail dynamic; ANE-exclusion gives up early instead of
  hanging.
- MIL op count: **2207 → 2021 (–186, –9%);** `tile: 96 → 0`. Cleaner
  graph for any future ANE compiler that becomes more permissive.
- Waveform parity vs the rank-3 baseline: **Pearson 0.999994 / SNR
  49.90 dB / max abs Δ 3.17e-3.** No audio quality regression.

What rank-4 did NOT fix (the residual problem this plan does not
address):

- Xcode Neural Engine column: **still 0 / 948.** Same graph-level
  disqualification pattern as rank-3.
- `Unsupported op` events: **322 → 298 (–24, only 7.5% drop).** Per-op
  rejection is widely distributed across the body even on perfect
  rank-4 last-axis-largest shapes like `1×256×1×8000`.
- `.all` output sha256 still bit-equals `.cpuAndGPU` —
  `.all` IS silent GPU fallback, just much cheaper failure.

**Next iteration plan** (per the pre-named escalation #2 above): open
`README/Plans/ane-decoder-har-fp16-inputs-v1.md` for Step 4(b). The
leading hypothesis after this Phase 3: the fp32 → fp16 entry cast and
the `ios16.*` op-set target are jointly blocking ANE on macOS 26's
new compiler. Confirm by switching the export's input dtypes to
`np.float16` (4(b)) — and, if that's not enough on its own,
re-evaluate Step 4(a) (target bump from `ct.target.macOS13` to
`ct.target.macOS15+`) on the rank-4 graph, which is a different
combination than the brief's original 4(a) attempt on the rank-3
graph.

## Success Criteria

### Hard Requirements (must pass)

- [x] `AdaIN1d` (rank-3) **unchanged** in
  [`kokoro/istftnet.py`](../../kokoro/istftnet.py#L98) (line 98).
  Decoder path contract is preserved.
- [x] New `AdaIN2d` class added next to `AdaIN1d`, consumed only by the
  rank-4 `AdaINResBlock1`. Round-trip vs `AdaIN1d` passes
  `torch.allclose(atol=1e-5, rtol=1e-5)` on synthetic input.
- [x] `AdaINResBlock1` / `Generator` rebuilt with Conv2d / ConvTranspose2d
  / ReflectionPad2d; `alpha1` / `alpha2` reshape to `(1, C, 1, 1)`.
- [x] `GeneratorFromHar.forward(x_pre, ref_s, har)` signature unchanged;
  rank promotion happens inside; output is rank-3.
- [x] Pretrained checkpoint loads via `register_load_state_dict_pre_hook`
  — no missing keys (excluding `stft.*` constructor-initialised
  buffers), no unexpected keys, no manual reshaping at the call site.
- [x] `uv run python -m pytest tests/` green: 41 passed, 9 skipped
  (env-conditional mlpackage tests), 0 failed.
- [x] `coreml/kokoro_decoder_har_post_3s.mlpackage` and
  `coreml/kokoro_decoder_har_post_10s.mlpackage` re-exported; export
  numeric gate green for both buckets.
- [x] [`scripts/compare_decoder_har_post_waveforms.py`](../../scripts/compare_decoder_har_post_waveforms.py)
  gates pass on the 10s bucket against the `/tmp` baseline:
  **Pearson 0.999994** (> 0.99), **SNR 49.90 dB** (≥ 40 dB), **max abs
  Δ 3.17e-3** (≤ 1e-2). 3s parity not run (no pre-rewrite 3s baseline
  preserved); rank-4 architecture is bucket-agnostic so 10s parity
  covers the architectural claim.
- [!] Xcode Performance Report **Neural Engine column count > 0** on
  `kokoro_decoder_har_post_10s.mlpackage`. **FAILED + DEFERRED:** 0 / 948 on
  M3 Max / macOS 26.4. Carried forward to
  [Iteration 2](#iteration-2-coremltools-90-upgrade--latest-target).
- [!] Python probe: `.all` sha256(out) **differs from** `.cpuAndGPU`
  sha256(out). Cold-load `.all` ≤ 5 s. **PARTIAL — failed on
  sha-differs half** (both produce `28041dfea5c8b6e1` — silent GPU
  fallback persists); **passed on cold-load half** (1.075 s ≤ 5 s).
- [!] Espresso `Unsupported op` event count on `.all` load **≤ 50** (down
  from 322). **FAILED + DEFERRED:** 298 events (–24 from rank-3 baseline
  of 322). Carried forward to
  [Iteration 2](#iteration-2-coremltools-90-upgrade--latest-target).

- [x] Results-log row appended to this plan.

> Convention: `[x]` = met; `[ ]` = not yet attempted; **`[!]`** = attempted
> and known-failed/deferred (with the deferral target named inline). A
> single-character distinction so a checkbox-scan reader doesn't conflate
> "deferred ANE gate" with "not yet implemented".

### Definition of Done

- [x] All Phase 0–3 tasks checked (with explicit failures documented for
  the three ANE-engagement gates above).
- [ ] Hard Requirements all checked. **NOT MET:** 3 of 13 fail (Xcode
  NE > 0, `.all` sha differs from `.gpu`, Espresso unsupported ≤ 50).
  All three are downstream of the same root cause and are deferred to
  the follow-on `ane-decoder-har-fp16-inputs-v1.md` plan.
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

- **Q:** Will `nn.ConvTranspose2d` with stride `(1, u)` and padding
  `(0, (k-u)//2)` produce a single MIL `conv_transpose` op on the
  current coremltools 8.3.0 + macOS 26 stack, or does it lower to a
  `conv_transpose` + `reshape` combo that fragments ANE eligibility?
- **A (resolved 2026-05-15 by Phase 0 probe):** Single `conv_transpose`
  op, no fragmentation. The extended `--probe-conv-lowering` mode
  verifies this for both `ups[0]` (k=20, stride=10, padding=5) and
  `ups[1]` (k=12, stride=6, padding=3) on coremltools 8.3.0 + torch
  2.6.0 + macOS 26.4. Both rank-3 ConvTranspose1d and rank-4
  ConvTranspose2d lower to the identical MIL op set
  `{cast, const, conv_transpose}`. The Upsample+Conv2d fallback path is
  no longer needed.
- **Q:** Will `tile` (96 occurrences in rank-3 baseline) drop to zero
  after rank-4, or does some tiling persist (e.g., for repeating the
  style vector along T inside AdaIN2d)?
- **A (resolved 2026-05-15 by Phase 2 MIL diff):** Drops to **zero**.
  Rank-4 PyTorch broadcasting replaces every `tile` op the rank-3
  graph used to expand AdaIN gamma/beta over T. AdaIN2d's
  `gamma`/`beta` of shape `(B, C, 1, 1)` broadcast implicitly against
  the rank-4 `x_norm` of shape `(B, C, 1, T)` without inserting
  explicit `tile`. The 10s rank-4 MIL histogram (Phase 2) confirms:
  `tile` 96 → 0. Cost: 2 new `expand_dims` ops (the unsqueeze(-2) at
  GeneratorFromHar's body entry) and 1 `squeeze` op (before
  `stft.inverse`).

### Unresolved

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

| Run | git SHA | coremltools | torch | Hardware | OS | Xcode NE count | Cold-load `.all` (s) | `.all` sha vs `.gpu` | Pearson 10s | SNR / Δ (10s) | MIL note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline (rank-3) | `842473e` | 8.3.0 | 2.6.0 | M3 Max 36 GB | macOS 26.4 | 0 / 1238 | 26.485 | match (silent GPU fallback) | n/a | n/a | 2207 ops; `linear` 48, `conv` 51, `tile` 96 |
| Rank-4 (Phase 1) | `0186993` | 8.3.0 | 2.5.0 | M3 Max 36 GB | macOS 26.4 | 0 / 948 | 1.075 | match (silent GPU fallback) | 0.999994 | 49.90 dB / 3.17e-3 | 2021 ops; `linear` 48, `conv` 51, `tile` **0** (–96); +`expand_dims` 2, `squeeze` 1 |

## Iteration 1 Rollback

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

---

## Iteration 2: coremltools 9.0 upgrade + latest target

**Date:** 2026-05-15
**Status:** Planned
**Builds on:** all iteration-1 phases above (rank-4 rewrite landed in
commits `df830a2..15c8fb1`; ANE engagement gate FAILED on
`ct.target.macOS13` / iOS 16; cold-load and shape-inference wins
kept).

### Rationale

Iteration 1 verdict from
[Conclusion and next iteration](#conclusion-and-next-iteration):
rank-4 is necessary but not sufficient on M3 Max / macOS 26.4 /
`ct.target.macOS13`. Xcode `Neural Engine` count still 0 / 948;
Espresso `Unsupported op` count 298 (down from 322 on rank-3); per-op
rejection on perfect rank-4 shapes like `1×256×1×8000` confirms the
rejection is no longer about shape.

Leading remaining hypothesis: macOS 26's ANE compiler refuses the
ios16 op set for this graph and wants newer op variants. Pinned
`coremltools==8.3.0` only exposes up to `ct.target.iOS18` — a
2-version bump from current target. To reach the latest op set
coremltools supports, iteration 2 upgrades `coremltools` to 9.0
**first**, then bumps `decoder-har`'s `minimum_deployment_target` to
the highest enum value 9.0 exposes (expected `iOS19+` /
`macOS26`-equivalent).

User-stated end goal: once ANE engages at the latest target,
iteration 3 backward-searches for the **earliest** target that still
keeps ANE — so the shipping artifact has the broadest OS-floor
compatible with ANE on M3 Max / macOS 26.

### Scope

- **In scope:**
  - `coremltools==8.3.0` → `coremltools==9.0` in
    [`requirements-export.txt`](../../requirements-export.txt), plus
    any transitive pin 9.0 forces (likely `torch ≥ some-newer-x`).
  - One-line `target = ct.target.macOS13` →
    `target = ct.target.<LATEST>` in
    [`export_synth/convert.py`](../../export_synth/convert.py),
    scoped to the `decoder-har` mode branch (don't touch `decoder` /
    `full` modes' targets).
- **Out of scope:** fp16 input dtypes (Step 4(b), iteration 3
  territory if needed); `export_decoder_pre.py`'s target;
  PyTorch-side code; the macOS app's deployment-target lift to
  match the new package floor (separate downstream concern flagged
  in Risks).

### Hard gates

- coremltools imports as 9.0.x; `--probe-conv-lowering` still
  `all_equivalent: true` on 9.0 (Conv2d / ConvTranspose2d / pad
  still lower to single MIL ops).
- `uv run python -m pytest tests/` green (no regression vs
  iteration 1's 41 passed / 9 skipped).
- Re-exported `coreml/kokoro_decoder_har_post_10s.mlpackage` reports
  `specificationVersion` higher than 7 (iteration 1's value).
  Per-op `Type` column in Xcode shows the new target's `ios<N>.*`
  prefix.
- Waveform parity vs the rank-4 / ios16 / cml8 baseline saved at
  `/tmp/kokoro_decoder_har_post_10s.rank4_ios16.mlpackage`:
  Pearson > 0.99, SNR ≥ 40 dB, max abs Δ ≤ 1e-2.
- ANE result is either a **stretch success** (Xcode `Neural Engine`
  count > 0) **or** a **partial-progress hard gate** (Espresso
  `Unsupported op` count drops to ≤ 100 from 298).

### Phases

**Phase 0 — Tooling upgrade + 9.0 baseline:**

- [ ] Check installability:
  `uv pip install --dry-run coremltools==9.0`. If 9.0 forces Python
  / torch upgrades, accept them in `requirements-export.txt` in the
  same commit. Record the new pins.
- [ ] Update `requirements-export.txt`. `uv pip install -r
  requirements-bakeoff.txt`. Verify the upgrade:
  `uv run python -c "import coremltools as ct; print(ct.__version__)"`.
- [ ] Enumerate `ct.target` on 9.0; pick the highest enum value.
  Record the full list in the
  [Iteration 2 Open Questions](#iteration-2-open-questions) below
  and lock the chosen target name in Resolved.
- [ ] Re-run
  `uv run python scripts/count_mil_ops.py --probe-conv-lowering`.
  Hard gate: `all_equivalent: true` still holds.
- [ ] `uv run python -m pytest tests/ -q`. Expected: 41 passed, 9
  skipped.
- [ ] Copy the current
  `coreml/kokoro_decoder_har_post_{3s,10s}.mlpackage` (rank-4 /
  ios16 / cml8) to
  `/tmp/kokoro_decoder_har_post_{3s,10s}.rank4_ios16.mlpackage`.
- [ ] Re-export at the iteration-1 target (still
  `ct.target.macOS13`) under coremltools 9.0; save as
  `/tmp/kokoro_decoder_har_post_10s.rank4_ios16_cml9.mlpackage`
  alongside. Cross-version waveform parity sanity check:
  `compare_decoder_har_post_waveforms.py
  --baseline /tmp/...rank4_ios16.mlpackage
  --candidate /tmp/...rank4_ios16_cml9.mlpackage`. Expect Pearson
  > 0.999 / SNR > 50 dB; if not, coremltools 9.0 is changing op
  lowering numerically — investigate before Phase 1.

**Phase 1 — Target bump:**

- [ ] Edit
  [`export_synth/convert.py`](../../export_synth/convert.py): change
  `target = ct.target.macOS13` to `target = ct.target.<LATEST>`,
  scoped to the `decoder-har` mode branch.
- [ ] `uv run python -m pytest tests/ -q` — green.

**Phase 2 — Re-export at the new target + waveform parity:**

- [ ] `uv run --no-sync python -m export_synth.main --mode decoder-har --buckets 3s,10s -o coreml`.
- [ ] Confirm target took effect:
  `uv run python -c "import coremltools as ct; print(ct.models.MLModel('coreml/kokoro_decoder_har_post_10s.mlpackage').get_spec().specificationVersion)"`
  should print **> 7**.
- [ ] Waveform parity vs rank-4 / ios16 / cml8 baseline:
  Pearson > 0.99 / SNR ≥ 40 dB / max abs Δ ≤ 1e-2 on the 10s bucket
  via `compare_decoder_har_post_waveforms.py`.
- [ ] `uv run python -m pytest tests/test_mlpackage_exports.py -q`
  — pass.

**Phase 3 — ANE placement verification:**

- [ ] Xcode Performance Report on the new 10s package. Save
  screenshot to `outputs/ane_rank4/xcode_latest_target_compute_unit_map_10s.png`
  (gitignored). Record `All / CPU / GPU / Neural Engine` counts;
  verify `ios<N>.*` op-type prefixes match the bumped target.
- [ ] Probe triplet: `.all` / `.gpu` / `.ne` in fresh subprocesses
  via `/tmp/ane-investigation/probe.py`. Compare sha256s against
  iteration 1's table (`.all` was `28041dfea5c8b6e1`, `.gpu`
  matched). If `.all` sha now **differs** from `.gpu`, ANE is
  engaging.
- [ ] Espresso `log stream` via
  `/tmp/ane-investigation/capture_logs.sh ... all ...`. Count
  `Unsupported op N` events. Hard gate: **≤ 100**.
- [ ] Append a row to the
  [Results log](#results-log-commit-this) above: label
  "Rank-4 + cml9 + latest target".

### Pre-named iteration 3 escalation

- **ANE > 0 (any value):** open
  `ane-decoder-har-target-floor-v1.md` (or another iteration in
  this same plan) to backward-search for the earliest target that
  still engages ANE.
- **ANE = 0 but `Unsupported op` count ≤ ~150:** add iteration 3 =
  fp16 input dtypes (Step 4(b)). Residual rejection is then at the
  input boundary.
- **ANE = 0 and `Unsupported op` near 298:** the rank-4 +
  newest-tooling + newest-target combo wasn't enough. Iteration 4
  territory; options are (a) per-op rewrite (split AdaIN's
  instance-norm differently), (b) accept GPU fallback and document
  ANE non-engagement on M3 Max / macOS 26 as a known limitation.

### Iteration 2 Open Questions

#### Resolved (iter 2)

- **Q:** Stay on the same branch (`ane-decoder-har-rank4-v1`) or cut
  a new one?
- **A:** Same branch. Iterations layer cleanly; one PR eventually
  carries the whole engagement arc.
- **Q:** Should iteration 2 be a separate plan file or a section
  here?
- **A:** Section in this plan. The whole HAR-on-ANE investigation
  reads as one document.

#### Unresolved (iter 2)

- **Q:** Is `coremltools==9.0` installable on Python 3.12.12 /
  torch 2.5.0 / macOS 26.4?
- **Options:** Phase 0 verifies via `uv pip install --dry-run`. If
  9.0 wants Python ≥ 3.13 or torch ≥ 2.7, widen the upgrade scope.
- **Q:** What's the highest `ct.target` value coremltools 9.0
  exposes?
- **Options:** Phase 0 enumerates and locks. Expected `iOS19+` /
  `macOS26`-equivalent.
- **Q:** Will the macOS app's deployment target accept the new
  OS-version floor?
- **Options:** Check before merge. If incompatible, iteration 2 is
  diagnostic-only and rolled back before shipping.

### Iteration 2 Risks (delta vs iteration 1)

| Risk | Mitigation |
| --- | --- |
| coremltools 9.0 isn't installable on the current Python / torch | Phase 0 dry-run check first. Accept any forced pin upgrades, or fall back to the highest installable 8.x. |
| coremltools 9.0 changes MIL lowering for Conv2d / ConvTranspose2d / pad ops | Phase 0 re-runs `--probe-conv-lowering`. If `all_equivalent` regresses, revert to 8.3.0 and document. |
| Same-target same-rank re-export under 9.0 drifts numerically | Phase 0 cross-version waveform-parity sanity check. Loosen gate if needed but document drift source. |
| Target bump silently fails (spec version unchanged) | Phase 2 explicit `specificationVersion` cross-check. |
| Target bump breaks downstream consumers on older macOS | Verify macOS app deployment target before merge. Diagnostic-only and rolled back if incompatible. |
| Other modes (`decoder` / `full`) in `convert.py` break because they share the `target` variable | Phase 1 guards the new target to the `decoder-har` branch only. |

### Iteration 2 Rollback

- **Revert Phase 1:** `git revert <Phase 1 commit>` restores
  `ct.target.macOS13`. Re-export brings packages back to the
  rank-4 / ios16 / cml9 state.
- **Revert Phase 0:** `git revert <Phase 0 commit>` restores
  `coremltools==8.3.0`. `uv pip install -r requirements-bakeoff.txt`
  reinstalls the 8.3.0 stack.
- **Time:** < 10 minutes total (revert + reinstall + re-export).
- **Data recovery:** none. Baselines preserved at
  `/tmp/kokoro_decoder_har_post_{3s,10s}.rank4_ios16.mlpackage`.
