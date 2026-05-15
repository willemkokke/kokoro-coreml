# Core ML Compute-Unit Ablation Notes

Institutional memory for isolating Core ML `.all`, `.cpuAndGPU`,
`.cpuAndNeuralEngine`, and `.cpuOnly` behavior in the Swift Kokoro pipeline.

**Quick filter:** `grep -n "— Active" README/Notes/coreml-compute-unit-ablation.md`

---

## Issue: Swift Core ML `.all` Is Slower Than `.cpuAndGPU` On M2 Air-Class Hardware — Active

**First spotted:** 2026-04-17
**Status:** Active

### Summary

The first F/G ablation shows that allowing ANE via Core ML `.all` does not help
on the local Apple M2 24 GB machine. Config G (`.cpuAndGPU`) beats Config F
(`.all`) at every benchmark length, especially 15s and 30s, where the slowdown
is concentrated in `GeneratorFromHar`.

Latency alone does not prove `.all` actually used ANE. It proves only that
allowing ANE did not help. The next ablations must distinguish a bad ANE path
from a bad `.all` mixed execution plan and from GPU doing the useful work.

### Current Evidence

Completed run:

```bash
BAKEOFF_SKIP_SMOKE=1 uv run --no-sync python scripts/bakeoff_harness.py run \
  --configs f,g \
  --iterations 5 \
  --order-seed 0 \
  --machine-id ane_ablation_fg_local
```

Result file:

- `outputs/bakeoff/results_ane_ablation_fg_local.json`

Machine:

- Apple M2, 24 GB unified memory
- macOS 15.7.5
- Git commit `9738030122ac00fe5fbe25930c094c467e70552a`
- Dirty tree: true

Median wall time:

| Input | F `.all` | G `.cpuAndGPU` | G/F |
| --- | ---: | ---: | ---: |
| 3s | 227.5 ms | 175.7 ms | 0.77 |
| 7s | 478.8 ms | 384.2 ms | 0.80 |
| 15s | 2051.9 ms | 802.3 ms | 0.39 |
| 30s | 4803.5 ms | 1592.2 ms | 0.33 |

Median `GeneratorFromHar` time:

| Input | F `.all` | G `.cpuAndGPU` |
| --- | ---: | ---: |
| 3s | 195.3 ms | 131.7 ms |
| 7s | 421.6 ms | 302.3 ms |
| 15s | 1911.9 ms | 638.8 ms |
| 30s | 4512.6 ms | 1273.7 ms |

### Interpretation Boundary

This result does **not** prove ANE execution was bad, because `.all` is a
scheduler request, not proof of active ANE placement. It establishes a narrower
but important claim:

> On this machine and artifact set, Core ML `.all` is slower than excluding ANE
> with `.cpuAndGPU` for the Swift decomposed pipeline.

Three mechanisms remain possible:

1. **ANE path hurts:** the ANE placement itself is slow for these bucket shapes.
2. **Mixed `.all` plan hurts:** `.all` creates a bad CPU/GPU/ANE partition or
   synchronization pattern, while ANE-only plus CPU might be fine.
3. **ANE fallback hurts:** `.all` attempts ANE, then spills or falls back into a
   worse execution path than explicitly excluding ANE.

> **2026-05-15 update — hypothesis #3 confirmed on M3 Max.**
> See [ANE decoder-har-post investigation](ane-decoder-har-post-investigation.md).
> Xcode shows 0 of ~1238 ops on Neural Engine for
> `kokoro_decoder_har_post_10s.mlpackage`; `.all` output is bit-equal to
> `.cpuAndGPU` (silent GPU fallback); `.all` cold-load is 55× slower than
> `.cpuAndGPU` (26.5 s vs 0.48 s) for an ANE compile that never succeeds.
> Espresso emits 322 `Unsupported op` events and 12 `Shape computation
> issue` events. Root cause is structural: the kokoro Generator graph uses
> rank-3 `(B, C, T)` tensors and ANE wants rank-4 `(B, C, 1, T)`. The fix
> is the rank-4 rewrite, not a compute-unit re-routing.

### Required Ablations

Add two Swift Core ML compute-unit controls to the harness:

| Human label | Suggested harness id | Core ML compute units | Purpose |
| --- | --- | --- | --- |
| G-prime | `gne` | `.cpuAndNeuralEngine` | Force CPU + ANE only; exclude GPU. |
| G-double-prime | `gcpu` | `.cpuOnly` | Exclude both GPU and ANE. |

The full compute-unit matrix:

| Config | Core ML compute units | Meaning |
| --- | --- | --- |
| F | `.all` | Let Core ML choose CPU, GPU, and ANE. |
| G | `.cpuAndGPU` | Exclude ANE. |
| G-prime | `.cpuAndNeuralEngine` | Exclude GPU. |
| G-double-prime | `.cpuOnly` | CPU baseline for Swift Core ML packages. |

Decision logic:

| Result pattern | Interpretation |
| --- | --- |
| G-prime is catastrophic like F at 15s/30s | ANE path is likely the problem, or `.all` and CPU+ANE share the same bad fallback. |
| G-prime is fast while F is slow | The problem is specifically the `.all` mixed plan. |
| G-double-prime is about 2x slower than G | GPU is doing useful work in G. |
| G-double-prime is near G | The win is mostly "not ANE"; GPU is not carrying much useful work. |

### Cross-Machine Gates

Before rewriting the paper framing, run F/G at minimum on Ultra and Mini with
the same artifacts and harness shape.

| Scenario | Paper implication |
| --- | --- |
| G beats F on Ultra, Air, and Mini | Thesis is inverted: Swift + decomposed Core ML on CPU+GPU beats PyTorch-on-MPS, and ANE does not help this workload. |
| G beats F on Air, but F beats G on Ultra | Most interesting result: ANE behavior is hardware-dependent for generative audio, and consumer-tier chips can be penalized by `.all`. |
| F beats G on Ultra and Mini, but G beats F only on Air | Air may be an outlier due to thermal state, memory pressure, or a chip-specific Core ML plan. Paper can survive with a caveat. |

### Related Guides

- [Core ML compute-unit scheduling](../Guides/apple-silicon/CoreML-Compute-Unit-Scheduling-guide.md) - explains `.all`, `.cpuAndGPU`, `.cpuAndNeuralEngine`, silent fallback, and powermetrics verification.
- [Bakeoff results v2](bakeoff-results-v2.md) - current cross-machine A/D/E/F benchmark context before this F/G ablation.
- [Performance notes](performance-notes.md) - historical timing and Core ML performance observations.

### Verification Gap

`powermetrics` was not captured during the first F/G run because non-interactive
sudo was unavailable:

```log
sudo: a password is required
```

For publication claims about ANE participation, latency comparisons must be
paired with telemetry:

```bash
sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000
```

If the sampler names differ on the target macOS version, use the ANE-only form
from the scheduling guide:

```bash
sudo powermetrics -i 1000 --samplers ane
```

### Next Steps

- [ ] Add Swift Core ML `G-prime` (`.cpuAndNeuralEngine`) to the harness.
- [ ] Add Swift Core ML `G-double-prime` (`.cpuOnly`) to the harness.
- [ ] Run F/G/G-prime/G-double-prime on the local M2 24 GB machine.
- [ ] Run F/G on M2 Ultra.
- [ ] Run F/G on M1 Mini.
- [ ] Capture powermetrics during steady-state F and G loops.
- [ ] Update [bakeoff results v2](bakeoff-results-v2.md) only after cross-machine data is collected.

### Investigation Log

**2026-04-17**

- **Hypothesis:** Config F's speedup over Config A may not isolate ANE, because
  F changes host language, orchestration, and the number of Core ML models.
- **Tried:** Added Config G as Swift + Core ML with `.cpuAndGPU`, preserving the
  same Swift pipeline and model packages as Config F.
- **Outcome:** Config G was faster than F on the local Apple M2 24 GB machine.
  The result invalidates an ANE-latency-win claim for this machine, but does not
  yet identify whether `.all` used ANE, mixed a bad plan, or fell back poorly.

