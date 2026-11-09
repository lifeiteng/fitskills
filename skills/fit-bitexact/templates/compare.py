#!/usr/bin/env python3
"""Compare two tensor files bitwise and report ULP diagnostics.

Usage:
  python compare.py target.bin reference.bin --dtype f16 --shape 8192,4096
  (without --shape the data is treated as 1-D: only element-level results are reported, no row counts)

Criterion: only identical bit patterns pass (prints BITEXACT). ULP is used only to judge
"how far off, and whether it looks like a reduction-order issue".

Terminology: ordered-integer distance = map the float bit pattern to an integer so that two adjacent
     representable floats differ by exactly 1. Two numbers n ulps apart have distance n. Both ±0 map
     to 0, so +0 and -0 are at distance 0, but their bit patterns differ and they still count as "different".
"""
import argparse
import numpy as np

INT = {"f16": np.uint16, "bf16": np.uint16, "f32": np.uint32}
BITS = {"f16": 16, "bf16": 16, "f32": 32}


def ordered(u, bits):
    sign = np.int64(1) << (bits - 1)
    u = u.astype(np.int64)
    return np.where(u & sign, -(u & (sign - 1)), u)


def to_float(u, dtype):
    if dtype == "f16":
        return u.view(np.float16).astype(np.float64)
    if dtype == "f32":
        return u.view(np.float32).astype(np.float64)
    return (u.astype(np.uint32) << 16).view(np.float32).astype(np.float64)  # bf16


def main():
    p = argparse.ArgumentParser()
    p.add_argument("target")
    p.add_argument("reference")
    p.add_argument("--dtype", required=True, choices=INT)
    p.add_argument("--shape", help="e.g. 8192,4096; dim 0 is treated as rows (tokens)")
    p.add_argument("--show", type=int, default=5, help="print bit patterns of the first N differing elements")
    args = p.parse_args()
    a = np.fromfile(args.target, INT[args.dtype])
    b = np.fromfile(args.reference, INT[args.dtype])
    if a.size != b.size:
        raise SystemExit(f"element counts differ: {a.size} != {b.size} (first check that shape/dtype/dump refer to the same tensor)")
    shape = tuple(int(x) for x in args.shape.split(",")) if args.shape else (a.size,)
    a, b = a.reshape(shape), b.reshape(shape)

    diff = a != b
    n = int(diff.sum())
    print(f"elements={a.size} differ={n}")
    if n == 0:
        print("BITEXACT")
        return

    idx = np.argwhere(diff)
    print(f"first_differ_index={tuple(int(i) for i in idx[0])}")
    if a.ndim > 1:
        print(f"rows_differ={len(np.unique(idx[:, 0]))} / {a.shape[0]}")
    w = BITS[args.dtype] // 4
    for i in idx[: args.show]:
        i = tuple(i)
        print(f"  at {tuple(int(x) for x in i)}: target=0x{int(a[i]):0{w}x} ({float(to_float(a[i], args.dtype))})"
              f"  reference=0x{int(b[i]):0{w}x} ({float(to_float(b[i], args.dtype))})")

    fa, fb = to_float(a, args.dtype), to_float(b, args.dtype)
    nan_a, nan_b = np.isnan(fa), np.isnan(fb)
    finite = ~nan_a & ~nan_b
    same_sign = (np.signbit(fa) == np.signbit(fb)) | (fa == 0) | (fb == 0)
    print(f"nan_mismatch={int((nan_a != nan_b).sum())} opposite_sign={int((diff & finite & ~same_sign).sum())}")
    sel = diff & finite & same_sign
    ulp = np.abs(ordered(a, BITS[args.dtype]) - ordered(b, BITS[args.dtype]))[sel]
    if ulp.size:
        hist = np.bincount(np.minimum(ulp, 8), minlength=9)
        print(f"ulp max={int(ulp.max())} mean={ulp.mean():.3f} "
              f"count[0(±0),1,2,3,4,5,6,7,>=8]={hist.tolist()}")
    err = np.abs(fa - fb)[finite]
    rms = float(np.sqrt(np.mean(fa[finite] ** 2))) or 1.0
    print(f"abs max={err.max():.3e} mean={err.mean():.3e} rel_rms={np.sqrt(np.mean(err ** 2)) / rms:.3e}")
    print("hint: differences concentrated at 1-2 ulp -> most likely a different reduction order or rounding point; "
          "go to S2 and fit with a probe. Large differences, opposite signs, or whole rows differing -> "
          "first check input identity, dtype, dispatch path, and discrete selection (top-k).")


if __name__ == "__main__":
    main()
