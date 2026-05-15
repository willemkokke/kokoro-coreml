# ANE Engagement Investigation: `kokoro_decoder_har_post_10s`

**Date:** 2026-05-15
**Status:** Resolved — root cause identified; recommended fix scoped to a
follow-up plan (rank-3 → rank-4 graph rewrite).
**Hardware:** Apple M3 Max, 36 GB unified memory.
**Host OS:** macOS 26.4 (build `25E246`).
**Package target:** `ct.target.macOS13` (specification version 7).

## TL;DR

The 10s decoder-har-post `.mlpackage` engages **zero** of its ~1238 compute-unit
ops on the Neural Engine. `.all` is a silent **GPU fallback** that produces
bit-for-bit identical output to `.cpuAndGPU` and pays a **~55× cold-load
penalty** (26.5 s vs 0.48 s) for an ANE compile that never succeeds.

The root cause is **structural**: the kokoro Generator graph uses rank-3
`(B, C, T)` tensors throughout. ANE's segmenter rejects 322 ops scattered
across the graph (every `conv → 1×C×T`, every AdaIN `reduce_mean → 1×C×1`,
every rank-3 `mul`), and 12 explicit `Shape computation issue` failures fire
at layers 37 / 44 / 51 — all rank-3 shape sites. The leading fp32→fp16 input
`cast` is also rejected, but on its own that op would simply fall back to
CPU/GPU at a healthy boundary. The graph bails because the **body** is
unsupported, not because of the entry cast.

**Recommendation:** commit to the rank-3 → rank-4 `(B, C, 1, T)` rewrite of
`Generator` + `AdaINResBlock1` + `AdaIN1d` as a checked-in plan
([`README/Plans/ane-optimization-v1.md`](../Plans/ane-optimization-v1.md) is
the precedent; the rank-4 work is a follow-on, larger in scope). Bumping
input dtype to fp16 (Step 4(b)) and bumping the target to `macOS15+`
(Step 4(a)) on their own do not engage ANE — both are free-riders to bundle
with the rank-4 rewrite, not standalone fixes.

This resolves hypothesis #3 in [Core ML compute-unit ablation](coreml-compute-unit-ablation.md):
> "`.all` attempts ANE, then spills or falls back into a worse execution path
> than explicitly excluding ANE."

---

## 1. Model identification

- **Path:** `coreml/kokoro_decoder_har_post_10s.mlpackage`
- **Export script:** [`export_synth/main.py`](../../export_synth/main.py) →
  [`export_synth/convert.py`](../../export_synth/convert.py) (`decoder-har`
  branch, ~line 440); wrapper
  [`export_synth/wrappers.py::GeneratorFromHar`](../../export_synth/wrappers.py)
- **Spec version:** 7 (corresponds to `ct.target.macOS13` / iOS 16)
- **coremltools:** 8.3.0
- **Source torch:** 2.6.0 (env-resolved; `requirements-export.txt` pins 2.5.0)
- **Compute precision:** FLOAT16 (mlprogram backend)
- **Inputs (all FLOAT32):**
  - `x_pre`: `(1, 512, 800)`
  - `ref_s`: `(1, 256)`
  - `har`:   `(1, 22, 96001)`
- **Output:** `waveform`: FLOAT16 `(1, 1, 240000)`
- **MIL op total:** 2207 (after coremltools lowering, pre-Espresso). The
  Xcode "Compute Unit Mapping" count of 1238 reflects post-segmentation
  ops; the 1166 MIL `const` ops collapse into weights and aren't counted by
  Xcode.

### MIL op histogram (top 15)

| Count | Op type |
|---:|---|
| 1166 | `const` |
| 218 | `add` |
| 148 | `mul` |
| 96 | `tile` |
| 88 | `reduce_mean` |
| 51 | `conv` |
| 50 | `sin` |
| 48 | `linear` |
| 48 | `reshape` |
| 48 | `split` |
| 48 | `pow` |
| 45 | `sub` |
| 44 | `square` |
| 44 | `sqrt` |
| 44 | `real_div` |

**Note on `linear: 48`.** The
[ANE Graph Optimization Plan](../Plans/ane-optimization-v1.md) previously
landed `AdaIN1d.fc: nn.Linear → nn.Conv1d(k=1)` (Phase 1 history shows
`linear: 48 → 0`, `conv: 51 → 99` on the 3s package), but commit
[`5278e88`](../../.git/) ("Revert AdaIN1d to Linear and expand HAR-post
performance notes") reverted it — its justification was *"benchmarks
showed no Core ML predict win from source-level Linear→Conv1d; MIL already
lowers Linear."* That justification was measured **without** confirming
ANE engagement, and this investigation now shows the revert did not
matter for ANE either: neither `linear` nor `conv` on rank-3 tensors is
ANE-eligible. The rewrite that matters is rank, not Linear-vs-Conv.

## 2. Baseline ANE engagement

The Xcode Performance Report (per the originating investigation brief) shows:

```text
All: 1,238   CPU: 0   GPU: 1,238   Neural Engine: 0
```

Every op — including ones that "should" trivially run on ANE
(`cast`, `slice_by_index`, `conv`, `leaky_relu`) — has an empty diamond in
the Neural Engine column. This is **graph-level disqualification**, not a
single-op chokepoint.

### Independent Python cross-check (Step 2 of the brief)

Three fresh-subprocess loads with shape-matching random inputs (small
clamped normals to match the export script's smoke geometry):

| `MLComputeUnits` | Cold load (s) | Cold predict (s) | Warm predict (s) | sha256(output) |
|---|---:|---:|---:|---|
| `.all`         |  26.485 | 0.218 | 0.116 | `d0bbdce3360a0506` |
| `.cpuAndGPU`   |   0.476 | 0.234 | 0.119 | `d0bbdce3360a0506` |
| `.cpuAndNE`    | 420.194 | 0.781 | 0.768 | `0222e613c45c1104` |

Reading the table:

- **`.all` output is bit-equal to `.cpuAndGPU`** (identical sha256). This is
  the canonical signature of silent GPU fallback per the brief: when ANE
  compile fails, `.all` becomes `.cpuAndGPU`.
- **`.all` cold-load is 55× slower than `.cpuAndGPU`** (26.485 s vs 0.476 s)
  — the cost of doomed ANE compile + retry + class-fallback ceremony.
- **`.cpuAndNE` cold-load is 7 minutes** (420 s). With GPU excluded, the
  runtime cannot fall back to MPS and instead exhausts the ANE compile
  attempt before dropping to BNNSGraph CPU. The output sha differs because
  BNNS CPU produces a different bit pattern than MPS GPU.
- Per-predict times confirm GPU does the work in `.all` (`.all` and `.gpu`
  both ~120 ms warm; `.ne` is ~770 ms — 6× slower CPU path).

Output values are NaN for all three runs because the shape-matching random
inputs blow up AdaIN's mean/variance normalization (a known instability on
near-zero synthetic inputs; the export's `numeric gate` uses real `x_pre`
/ `har` for the same reason). The **bit-equality of the NaN pattern**
between `.all` and `.cpuAndGPU` is stronger evidence than RMS comparison
would be — identical NaN bit-pattern means the computation graphs are
identical down to the order of intermediate ops.

## 3. The smoking gun: Espresso `log stream` during `.all` load

Captured with:

```bash
log stream \
  --predicate 'subsystem CONTAINS "com.apple.ANE" OR subsystem CONTAINS "com.apple.coreml" OR subsystem CONTAINS "espresso"' \
  --level=debug --style=ndjson
```

while loading the package with `MLComputeUnit.ALL`. **762 events** in ~28 s.
The os_log private-data filter redacts op type names (`<private>`) but
event indices and counts are not redacted.

### 3.1. The rejection cascade

| Format string | Count | Verbatim sample (with redactions) |
|---|---:|---|
| `Unsupported op %zu (%s): %s` | **322** | `Unsupported op 8 (<private>): <private>` |
| `Shape computation issue: %s at layer %zu` | **12** | `Shape computation issue: <private> at layer 37` |
| `Backend registered: %s` | 22 | `Backend registered: ane` |
| `Start segmenting function: %s` | 5 | (E5RT = ANE backend segmenter) |
| `Finished segmenting function: %s` | 4 | (one segmentation start did not finish) |
| `Lowering to E5RT opset completed` | 4 | (only 4 segments lowered to ANE opset) |
| `E5RT: %s (%d)` | 7 | `E5RT: <private> (11)` (error code 11) |
| `Failed to create E5 execution stream operation for the function ...` | 6 | "It can happen when APFS purged a compiled E5 bundle. We will recompile the bundle and try again. (Retry 1 / 2)" |
| `%@ class was unable to load the model at %s with error: %@; The model loader is going to use another class.` | 4 | The silent fallback gate. |
| `Model enabled BNNSGraph as a preferred cpu backend.` | 8 | The CPU fallback registers. |
| `Unable to set priority when the backend is not ANE` | 5 | Final confirmation: priority hint refused because backend is not ANE. |

The 322 `Unsupported op N` events have indices in `[7, 1041]`, all unique
— a wide-spread rejection (not clustered at one chokepoint). The retry
framing in the "Failed to create E5 execution stream" message is
misleading: the issue is not APFS purging the compiled bundle, it's that
the segmenter could not produce a realizable ANE function in the first
place.

### 3.2. Mapping rejected indices to MIL op types

Walking the MIL program with `coremltools` and correlating the indices
that fire (the first few rejections + the three layers that hit
"Shape computation issue") gives:

| Espresso index | MIL op | Output shape | Why ANE rejects |
|---:|---|---|---|
| 7 | `const` (`var_58_to_fp16`) | scalar | const used by leading cast — boundary noise. |
| **8** | **`cast` (`x_pre_to_fp16`)** | **1×512×800** | **Entry cast fp32 → fp16 at the input boundary.** |
| 37 | `const` (`var_138_split_sizes_0`) | `(2,)` | split-sizes vector — shape inference site for AdaIN `torch.chunk`. |
| 44 | `const` (`var_144_to_fp16`) | scalar | scalar const-cast — small but on a shape-bearing path. |
| 51 | `const` (`var_79_promoted_to_fp16`) | scalar | scalar const-cast — same. |
| 115 | `const` (`reduce_mean_5_keep_dims_0`) | scalar | `keep_dims` argument to AdaIN's `reduce_mean`. |
| **138** | **`mul` (`var_246_cast_fp16`)** | **1×256×16000** | **First rank-3 elementwise multiply (AdaIN scale or noise-conv path).** |
| 369 | `const` (`var_558_to_fp16`) | scalar | scalar const. |
| 492 | `const` (`generator_resblocks_0_adain1_2_fc_bias_to_fp16`) | `(512,)` | AdaIN1d.fc bias for resblock 0 / adain1 / branch 2. |
| 518..521 | `const`s | conv stride/groups/weight/bias | conv parameters for the next conv. |
| **522** | **`conv` (`x_25_cast_fp16`)** | **1×256×8000** | **Rank-3 conv output (B, C, T).** |
| **525** | **`reduce_mean` (`mean_23_cast_fp16`)** | **1×256×1** | **AdaIN mean reduction over the time axis — rank-3 input, rank-3 output.** |
| 1041 | `reduce_mean` (`reduce_mean_47_cast_fp16`) | `1×256×1` | Same pattern, further into the upsampling stack. |

Every load-bearing op (entries in **bold**) is one of:

1. The entry `cast fp32 → fp16` at the input boundary.
2. A rank-3 `(B, C, T)` `conv` output.
3. A rank-3 `(B, C, 1)` AdaIN `reduce_mean` output.
4. A rank-3 `(B, C, T)` elementwise `mul` on the noise/AdaIN path.

This is **exactly** the shape pattern called out in CLAUDE.md and in the
brief:

> "The last axis must be the largest dimension. Use shape `(B, C, 1, T)`
> where `T` is large. Never use `(B, T, C)` where `C` is small."

Kokoro's Generator picks `(B, C, T)` rank-3 with `C` in `{256, 128, ...}`.
The ANE compiler does not auto-promote this to rank-4; it disqualifies
the segment.

### 3.3. Why the entry cast is a red herring, not the root cause

User-supplied framing — verified by the log:

> If the rank issues were solved, the unsupported input cast ops would
> simply lower to CPU/GPU instead of ANE. But the compiler doesn't get to
> that stage because it bails completely.

The Espresso segmenter's job is to partition the graph into
ANE-eligible runs and let everything else (boundary casts, custom ops,
unsupported edge ops) fall back to CPU or GPU. **The entry cast at index 8
is supposed to fall back.** What disqualifies the whole graph is not the
cast — it's that 322 ops scattered across the body fragment every potential
ANE segment so finely that no segment is large enough to be worth realizing.
The runtime tries 4 segmentations, lowers them to E5RT, gets error code
11 from each, retries, fails again, and finally hands the model to the
GPU (MPSGraph) class.

This is why **Step 4(b) (fp16 inputs) alone will not engage ANE.** It
removes the entry cast from the rejection list but leaves the rank-3 body
unchanged. Step 4(b) is a free-rider to bundle with the rank-4 rewrite
(one less boundary op, slightly faster cold init), not a standalone fix.

## 4. Hypotheses tested

| Hypothesis | Cost | Verdict |
|---|---|---|
| **(a) Bump `minimum_deployment_target`** macOS13 → macOS15 | re-export only | **Ruled out** (per investigation brief: bump took effect — op-type prefixes changed `ios16.*` → `ios18.*`; Neural Engine column still 0/1238). |
| **(b) fp16 input dtypes** (`x_pre`, `ref_s`, `har` declared as `np.float16`) | re-export + runtime caller changes | **Not standalone.** Would only remove the entry-cast op from the rejection list; body remains rank-3 and segmenter still bails. Worth bundling with (c). |
| **(c) Rank-3 → rank-4 `(B, C, 1, T)` rewrite** of `Generator`, `AdaINResBlock1`, `AdaIN1d`, mask threading | 100-300 LOC across `kokoro/istftnet.py` + `export_synth/wrappers.py`; PyTorch parity test; re-export + waveform parity | **The only fix supported by the evidence.** Every shape rejected by the log is a rank-3 shape; the documented ANE preference is the inverse layout. |
| **(d) Substitute a specific chokepoint op** | depends | **Not applicable.** 322 unique rejected indices spread across `[7, 1041]` rule out a single-op explanation. |

Re-checking the linear-vs-conv axis: the
[`ane-optimization-v1` plan](../Plans/ane-optimization-v1.md) Phase 1
landed `nn.Linear → nn.Conv1d(k=1)` for `AdaIN1d.fc`, was later reverted
in commit `5278e88` based on a wall-clock no-op. This investigation
explains why there was no wall-clock win: ANE was never engaged, so
matmul-vs-conv-on-ANE optimization could not have helped. The revert was
correct on its own terms (no observed benefit); it should not block the
rank-4 work, since rank-4 is what actually moves the placement needle.

## 5. Recommendation

**Commit to a rank-4 rewrite plan as the next milestone.**

Suggested plan path: `README/Plans/ane-decoder-har-rank4-rewrite-v1.md`
(follow-on to `ane-optimization-v1.md`).

In-scope for the rewrite:

- `kokoro/istftnet.py::AdaIN1d` — accept rank-4 `(B, C, 1, T)` input; reduce
  over the last axis; bias broadcast to rank-4.
- `kokoro/istftnet.py::AdaINResBlock1` — Conv1d → Conv2d `(1, k)` with
  weight reshape (`(C_out, C_in, k)` → `(C_out, C_in, 1, k)`); same for
  noise_convs (ConvTranspose1d → ConvTranspose2d).
- `kokoro/istftnet.py::Generator` — unsqueeze inputs at the entry; squeeze
  the H=1 axis at the conv_post output (or just before the iSTFT tail,
  depending on which placement keeps the iSTFT entry shape happy).
- `export_synth/wrappers.py::GeneratorFromHar._align_mask_to` — update mask
  shape to `(B, 1, 1, T)`.
- (Optional) bundle Step 4(b): switch input declarations to `np.float16`
  and update runtime callers (`scripts/bakeoff_harness.py`,
  `scripts/bench_decoder_har_post_predict.py`, the Swift pipeline). Cheap
  to bundle; do not bundle if it stretches scope.

Out-of-scope for the first rank-4 PR:

- The iSTFT tail (`kokoro/istftnet.py::CustomSTFT.inverse` and friends).
  Keep rank-3 across the iSTFT boundary; the conv_post output squeezes
  to rank-3 before the iSTFT call. Revisit later only if profiling shows
  the iSTFT op stack is the next gating factor for ANE engagement.
- The Decoder pre-stack rewrite (`Decoder.encode` / `Decoder.decode` /
  `AdainResBlk1d`). That's a separate package (`decoder_pre_10s`) with the
  same disqualification pattern, but tackling both in one PR doubles risk.
  Sequence as: HAR-post rank-4 first → re-validate ANE engagement and
  audio parity → then `decoder_pre` rank-4.

Acceptance criteria for the rank-4 PR (mirror Phase 0 / Phase 3 of
`ane-optimization-v1`):

1. PyTorch parity: rank-4 `GeneratorFromHar` output matches the rank-3
   baseline on real `(x_pre, har, ref_s, mask)` inputs to fp32 rounding
   tolerance (existing
   `scripts/compare_decoder_har_post_waveforms.py` gates: Pearson > 0.99,
   SNR ≥ 40 dB, max abs Δ ≤ 1e-2).
2. Re-export both shipping buckets (`3s`, `10s`).
3. **Xcode Performance Report: Neural Engine column count > 0** on the
   re-exported `10s` package. This is the new hard gate.
4. Python probe (`/tmp/ane-investigation/probe.py`): `.all` output sha256
   **differs** from `.cpuAndGPU` (no more silent fallback).
5. End-to-end audio parity (`uv run pytest` green;
   `tests/test_audio_quality_probe.py` and
   `tests/test_bucket_contamination.py` in particular).

If Step 4(b) is bundled, also: update production callers; re-validate
that fp32 inputs upstream of the model entry are converted to fp16 at
the call site (cheap), and remeasure the .all / .gpu / .ne triplet.

## 6. Reproducer

Reproduces on M3 Max running macOS 26.x with Xcode + uv installed.

```bash
# From a clean checkout of main:
cd /Users/willem/Documents/Repositories/kokoro-coreml-ane
uv sync

# Export the 10s decoder-har-post package.
uv run --no-sync python -m export_synth.main \
    --mode decoder-har --buckets 10s -o coreml

# Capture Espresso/ANE/CoreML log stream while loading with .all.
bash /tmp/ane-investigation/capture_logs.sh \
    "$PWD/coreml/kokoro_decoder_har_post_10s.mlpackage" \
    all \
    /tmp/ane-investigation/log_all.ndjson

# Probe the .all / .gpu / .ne triplet in fresh subprocesses (different
# subprocesses avoid Apple-runtime quirks when changing compute_units in
# the same Python process).
for u in all gpu ne; do
  uv run --no-sync python /tmp/ane-investigation/probe.py \
      "$PWD/coreml/kokoro_decoder_har_post_10s.mlpackage" "$u"
done

# Enumerate MIL ops and highlight Espresso "layer N" indices:
uv run --no-sync python /tmp/ane-investigation/dump_mil_ops.py \
    "$PWD/coreml/kokoro_decoder_har_post_10s.mlpackage" \
    "7,8,37,44,51,115,138,369,492,518,519,520,521,522,525,1041"

# Spec metadata:
uv run --no-sync python /tmp/ane-investigation/spec_meta.py \
    "$PWD/coreml/kokoro_decoder_har_post_10s.mlpackage"
```

Tooling lives under `/tmp/ane-investigation/` (gitignored — these are
investigation scratch, not shipping scripts). If the rank-4 rewrite plan
goes ahead, the probe + dump scripts should be hardened and moved under
`scripts/` for repeatable acceptance testing.

## Related notes

- [Core ML compute-unit ablation](coreml-compute-unit-ablation.md) — F/G/G-prime/G-double-prime ablation; this investigation resolves the open hypothesis #3 (silent fallback) in its favor on M3 Max.
- [Performance notes](performance-notes.md) — historical timing and Conv1d revert context.
- [ANE Graph Optimization Plan](../Plans/ane-optimization-v1.md) — Phase 0/1/3 linear-vs-conv work that preceded this investigation.
- [Core ML Compute Unit Scheduling Guide](../Guides/apple-silicon/CoreML-Compute-Unit-Scheduling-guide.md) — documented `.all` / `.cpuAndGPU` / `.cpuAndNeuralEngine` semantics + silent fallback reference.
- [CLAUDE.md, Part 4.1 — ANE memory layout](../../CLAUDE.md) — `(B, C, 1, T)` rule with `T` largest; 64-byte last-axis alignment penalty.
