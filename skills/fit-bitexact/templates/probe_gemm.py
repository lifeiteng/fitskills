#!/usr/bin/env python3
"""GEMM probe: generate inputs, compute reference results with several candidate reduction trees,
and compare their agreement rate against the target side's output.

Full workflow:
  1. python probe_gemm.py gen  OUT_DIR
     Generates probe-MxNxK.a.f16 / .b.f16 (A is MxK, B is KxN, row-major FP16)
  2. Run the GEMM on the target side with the same inputs and write the output to
     OUT_DIR/probe-MxNxK.c.f16 (MxN, FP16)
  3. python probe_gemm.py fit  OUT_DIR
     Prints each candidate's agreement rate per shape. Only a candidate that reaches 100% may go
     into the reference implementation.

The candidate reduction trees are only a starting point covering common forms. Determine the real rule
by reading the target-side source (dispatch, dot pairing, number of split-K segments).
Computation is row by row: memory is O(N*K), and no (M,N,K) 3-D tensor is materialized.
This is a reference-principle implementation for fitting small probes and is slow; for real layers,
rewrite the reference implementation as a tiled torch/Triton version.
"""
import os
import sys
import numpy as np

SHAPES = [  # (M, N, K)
    (128, 128, 1024), (130, 72, 1024),     # full tile / edge tile
    (256, 256, 4096), (256, 256, 8192),    # split-K may use 2 / 4 segments (thresholds per target-side source)
    (3, 512, 5120), (8, 512, 5120),        # small M; some libraries use cooperative GEMV for 2-8 rows
    (17, 384, 4096), (48, 1024, 2048),     # both sides of the small-M threshold
]


def gen(out, seed=0):
    rng = np.random.default_rng(seed)
    os.makedirs(out, exist_ok=True)
    for M, N, K in SHAPES:
        a = rng.uniform(-1, 1, (M, K)).astype(np.float16)
        b = rng.uniform(-1, 1, (K, N)).astype(np.float16)
        a[0, :8] = [0, -0.0, 6e-8, -6e-8, 65504, 1, -1, 0]  # special values: ±0, subnormals, max
        a.tofile(f"{out}/probe-{M}x{N}x{K}.a.f16")
        b.tofile(f"{out}/probe-{M}x{N}x{K}.b.f16")
    print(f"wrote {len(SHAPES)} probes to {out}")


def chain(p):
    """FP32 linear chain: add along the last dim starting from +0, rounding to FP32 at every step, no fused FMA."""
    acc = np.zeros(p.shape[:-1], np.float32)
    for k in range(p.shape[-1]):
        acc = acc + p[..., k]
    return acc


def dot4_pairs(p):
    """Groups of 16 K; within a group pair as (p0+p1)+(p8+p9), (p2+p3)+(p10+p11), ...; results feed the accumulation chain in order."""
    out = []
    for g in range(0, p.shape[-1], 16):
        q = p[..., g:g + 16]
        for b in range(4):
            i = 2 * b
            out.append((q[..., i] + q[..., i + 1]) + (q[..., i + 8] + q[..., i + 9]))
    return np.stack(out, -1)


def row_products(a_row, b):
    return a_row.astype(np.float32)[None, :] * b.T.astype(np.float32)  # (N, K); FP16×FP16 is exact in FP32


def candidate_linear_fp32(a, b):
    """Candidate 1: a single FP32 linear chain over all of K, rounded to half once at the end."""
    return np.stack([chain(row_products(r, b)) for r in a]).astype(np.float16)


def candidate_dot4_splitk_fp32(a, b, splits=1, seg_align=128):
    """Candidate 2: dot4 pairing + split-K (each segment in FP32, added serially by segment index), rounded once at the end."""
    K = a.shape[1]
    seg = (K // splits) // seg_align * seg_align
    rows = []
    for r in a:
        p = row_products(r, b)
        acc = np.zeros(p.shape[0], np.float32)
        for s in range(splits):
            lo, hi = s * seg, (K if s == splits - 1 else (s + 1) * seg)
            acc = acc + chain(dot4_pairs(p[:, lo:hi]))
        rows.append(acc)
    return np.stack(rows).astype(np.float16)


def candidate_half_per_block(a, b, block=64):
    """Candidate 3 (common defect): round the accumulator to half every `block` K."""
    rows = []
    for r in a:
        p = row_products(r, b)
        acc = np.zeros(p.shape[0], np.float16)
        for lo in range(0, a.shape[1], block):
            acc = (acc.astype(np.float32) + chain(p[:, lo:lo + block])).astype(np.float16)
        rows.append(acc)
    return np.stack(rows)


CANDIDATES = {
    "linear_fp32": candidate_linear_fp32,
    "dot4_fp32_split1": lambda a, b: candidate_dot4_splitk_fp32(a, b, 1),
    "dot4_fp32_split2": lambda a, b: candidate_dot4_splitk_fp32(a, b, 2),
    "dot4_fp32_split4": lambda a, b: candidate_dot4_splitk_fp32(a, b, 4),
    "half_per_64": candidate_half_per_block,
}


def fit(out):
    for M, N, K in SHAPES:
        base = f"{out}/probe-{M}x{N}x{K}"
        if not os.path.exists(base + ".c.f16"):
            print(f"{M}x{N}x{K}: missing target-side output {base}.c.f16, skipping")
            continue
        a = np.fromfile(base + ".a.f16", np.float16).reshape(M, K)
        b = np.fromfile(base + ".b.f16", np.float16).reshape(K, N)
        c = np.fromfile(base + ".c.f16", np.float16).reshape(M, N)
        scores = {name: float((f(a, b).view(np.uint16) == c.view(np.uint16)).mean())
                  for name, f in CANDIDATES.items()}
        best = max(scores, key=scores.get)
        line = "  ".join(f"{k}={v:.4%}" for k, v in scores.items())
        print(f"{M}x{N}x{K}: {line}  -> best={best}{'  (BITEXACT)' if scores[best] == 1.0 else ''}")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("gen", "fit"):
        raise SystemExit("usage: probe_gemm.py gen|fit OUT_DIR")
    (gen if sys.argv[1] == "gen" else fit)(sys.argv[2])
