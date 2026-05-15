#!/usr/bin/env python3
"""Histogram MIL operation types for an MLProgram .mlpackage (ANE optimization Phase 0).

Walks ``spec.mlProgram`` function blocks for reproducible Phase 3 comparisons.

Example::

    uv run python scripts/count_mil_ops.py coreml/kokoro_decoder_har_post_3s.mlpackage
    uv run python scripts/count_mil_ops.py --probe-conv-lowering
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn


def _iter_operation_types(spec) -> list[str]:
    mp = spec.mlProgram
    types: list[str] = []
    for fn in mp.functions.values():
        for block in fn.block_specializations.values():
            for op in block.operations:
                types.append(op.type)
    return types


def histogram_for_mlpackage(path: Path) -> tuple[Counter, int]:
    model = ct.models.MLModel(str(path))
    spec = model.get_spec()
    types = _iter_operation_types(spec)
    return Counter(types), len(types)


def _export_minimal_and_top_ops(module: nn.Module, x: torch.Tensor) -> list[str]:
    m = module.eval()
    with torch.no_grad():
        traced = torch.jit.trace(m, (x,), strict=False)
    ml = ct.convert(
        traced,
        inputs=[ct.TensorType(shape=x.shape, dtype=np.float32)],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
    )
    spec = ml.get_spec()
    types = _iter_operation_types(spec)
    return sorted(set(types))


class _TinyConv1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Conv1d(4, 8, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c(x)


class _TinyConv2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Conv2d(4, 8, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c(x)


class _ParamConv1d(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int, stride: int = 1,
                 padding: int = 0, dilation: int = 1, transpose: bool = False):
        super().__init__()
        if transpose:
            self.c = nn.ConvTranspose1d(in_c, out_c, k, stride=stride, padding=padding)
        else:
            self.c = nn.Conv1d(in_c, out_c, k, stride=stride, padding=padding, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c(x)


class _ParamConv2d(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int, stride: int = 1,
                 padding: int = 0, dilation: int = 1, transpose: bool = False):
        super().__init__()
        if transpose:
            self.c = nn.ConvTranspose2d(in_c, out_c, (1, k), stride=(1, stride), padding=(0, padding))
        else:
            self.c = nn.Conv2d(in_c, out_c, (1, k), stride=(1, stride),
                               padding=(0, padding), dilation=(1, dilation))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c(x)


class _ReflectionPad1d(nn.Module):
    def __init__(self, padding):
        super().__init__()
        self.p = nn.ReflectionPad1d(padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.p(x)


class _ReflectionPad2d(nn.Module):
    def __init__(self, padding):
        super().__init__()
        self.p = nn.ReflectionPad2d(padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.p(x)


def _probe_pair(label: str, rank3_module: nn.Module, rank4_module: nn.Module,
                rank3_shape, rank4_shape) -> dict:
    x3 = torch.zeros(*rank3_shape, dtype=torch.float32)
    x4 = torch.zeros(*rank4_shape, dtype=torch.float32)
    t3 = _export_minimal_and_top_ops(rank3_module, x3)
    t4 = _export_minimal_and_top_ops(rank4_module, x4)
    return {
        "case": label,
        "rank3_shape": list(rank3_shape),
        "rank4_shape": list(rank4_shape),
        "rank3_op_types": t3,
        "rank4_op_types": t4,
        "type_sets_equal": sorted(t3) == sorted(t4),
        "rank4_extra_ops": sorted(set(t4) - set(t3)),
        "rank4_missing_ops": sorted(set(t3) - set(t4)),
    }


def probe_conv_lowering() -> dict:
    """Probe whether rank-4 Conv2d / ConvTranspose2d / ReflectionPad2d variants
    used in the planned Generator rewrite lower to the same MIL op set as their
    rank-3 counterparts. Phase 0 gate for the rank-4 rewrite plan
    (README/Plans/ane-decoder-har-rank4-rewrite-v1.md).

    Real kernel/stride values come from the kokoro istftnet config:
      - resblock_kernel_sizes = [3, 7, 11]   (AdaINResBlock1.convs1/convs2)
      - resblock_dilation_sizes = [1, 3, 5]  (AdaINResBlock1.convs1)
      - upsample_kernel_sizes = [20, 12]     (Generator.ups)
      - upsample_rates = [10, 6]             (Generator.ups stride)
      - noise_convs:  (k=12, stride=6, padding=3)  and  (k=1)
      - conv_post:    (k=7, stride=1, padding=3)
      - reflection_pad: ReflectionPad1d((1, 0))
    """
    cases: list[dict] = []

    # Baseline: minimal Conv1d(k=1) vs Conv2d(1, 1) — sanity-check that the
    # historical Phase 0 result (sets equal, both lower to MIL ``conv``) still
    # holds on the current stack.
    cases.append(_probe_pair(
        "Conv k=1 (baseline)",
        _TinyConv1d(), _TinyConv2d(),
        (1, 4, 16), (1, 4, 1, 16),
    ))

    # AdaINResBlock1.convs1/convs2 — kernels {3, 7, 11} with dilation {1, 3, 5}
    # (only the dilation=1 cases for k=7 and k=11; AdaINResBlock1 uses dilation=1
    # for convs2 across all kernels).
    for k in (3, 7, 11):
        for d in (1, 3, 5):
            if d > 1 and k == 1:
                continue
            padding = (k * d - d) // 2  # matches get_padding in istftnet.py
            cases.append(_probe_pair(
                f"Conv k={k} dilation={d} (AdaINResBlock1 convs1)",
                _ParamConv1d(8, 8, k, padding=padding, dilation=d),
                _ParamConv2d(8, 8, k, padding=padding, dilation=d),
                (1, 8, 32), (1, 8, 1, 32),
            ))

    # noise_convs[0]: k=12, stride=6, padding=3 (the strided downsampling case).
    cases.append(_probe_pair(
        "Conv k=12 stride=6 padding=3 (noise_convs[0])",
        _ParamConv1d(22, 256, 12, stride=6, padding=3),
        _ParamConv2d(22, 256, 12, stride=6, padding=3),
        (1, 22, 96), (1, 22, 1, 96),
    ))

    # conv_post: k=7, stride=1, padding=3.
    cases.append(_probe_pair(
        "Conv k=7 stride=1 padding=3 (conv_post)",
        _ParamConv1d(128, 22, 7, padding=3),
        _ParamConv2d(128, 22, 7, padding=3),
        (1, 128, 32), (1, 128, 1, 32),
    ))

    # Generator.ups[0]: ConvTranspose1d k=20, stride=10, padding=5.
    cases.append(_probe_pair(
        "ConvTranspose k=20 stride=10 padding=5 (ups[0])",
        _ParamConv1d(512, 256, 20, stride=10, padding=5, transpose=True),
        _ParamConv2d(512, 256, 20, stride=10, padding=5, transpose=True),
        (1, 512, 16), (1, 512, 1, 16),
    ))

    # Generator.ups[1]: ConvTranspose1d k=12, stride=6, padding=3.
    cases.append(_probe_pair(
        "ConvTranspose k=12 stride=6 padding=3 (ups[1])",
        _ParamConv1d(256, 128, 12, stride=6, padding=3, transpose=True),
        _ParamConv2d(256, 128, 12, stride=6, padding=3, transpose=True),
        (1, 256, 32), (1, 256, 1, 32),
    ))

    # ReflectionPad: 1d (left=1, right=0) vs 2d (left=1, right=0, top=0, bottom=0).
    cases.append(_probe_pair(
        "ReflectionPad (1, 0) vs (1, 0, 0, 0)",
        _ReflectionPad1d((1, 0)),
        _ReflectionPad2d((1, 0, 0, 0)),
        (1, 4, 16), (1, 4, 1, 16),
    ))

    all_equivalent = all(c["type_sets_equal"] for c in cases)
    return {
        "stack": {
            "coremltools": ct.__version__,
            "torch": torch.__version__,
            "minimum_deployment_target": "macOS13",
        },
        "cases": cases,
        "all_equivalent": all_equivalent,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mlpackage", nargs="?", type=Path, help="Path to .mlpackage directory")
    p.add_argument("--json", action="store_true", help="Histogram as JSON")
    p.add_argument(
        "--probe-conv-lowering",
        action="store_true",
        help="Minimal Conv1d vs Conv2d MIL op comparison (Phase 0 gate)",
    )
    args = p.parse_args(argv)

    print(f"coremltools {ct.__version__}", file=sys.stderr)

    if args.probe_conv_lowering:
        out = probe_conv_lowering()
        print(json.dumps(out, indent=2))
        return 0 if out["all_equivalent"] else 2

    if args.mlpackage is None:
        p.error("mlpackage path required unless --probe-conv-lowering")

    path = args.mlpackage
    if not path.is_dir():
        print(f"Not a directory: {path}", file=sys.stderr)
        return 1

    counts, total = histogram_for_mlpackage(path)
    if args.json:
        print(json.dumps({"total_ops": total, "counts": dict(sorted(counts.items()))}, indent=2))
    else:
        print(f"total_ops\t{total}")
        for op, n in counts.most_common():
            print(f"{op}\t{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
