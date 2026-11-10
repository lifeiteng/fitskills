# Reference side: reproducing the target side bit-exactly on another GPU / framework

The thresholds and rules in the code examples come from measurements on Apple GPU (Metal/MFA) and are illustrative only; for other GPUs or engines, refit them using the same method.

Principle: the reference side does not aim to be "more accurate"; it **writes out the reduction tree of every target-side kernel**. Every rule must be fitted to 100% with a probe before it goes into the reference implementation. If the reference implementation is also used for training: the forward pass uses the numerics fitted here (bit-exact with the target side); the backward pass uses the gradient of the ideal operator, passing gradients straight through (STE) non-differentiable steps such as rounding, quantization, and top-k.

## 1. Dispatch table (first thing to do)
Read the dispatch conditions from the target-side source, then confirm them with dispatch logs. Sketch:

```python
def gemm_kernel(rows, N, K):   # Example: dispatch order of a GEMM library on Apple GPU (the GEMV check comes before SmallM)
    if rows == 1:                               return "gemv"
    if 2 <= rows <= 8 and N <= 512 and K >= 512: return "coop_gemv"   # easily mistaken for small_m
    if rows <= 48 and K < 16384:               return "small_m"
    return "tiled_gemm"
```
- Decide based on the **actual row count of each call**. Treating several consecutive rows within prefill as one call yields the wrong row count.
- `raise` on unverified shape combinations; do not fall back to an approximate implementation.

## 2. GEMM reduction tree
Parameters to read from the source and then confirm with probes:
- **Pairing in the vectorized dot**: e.g., groups of 16 K, dot4 = (p0+p1)+(p8+p9), added to the accumulation chain for b=0..3 in order.
- **Whether FMA is fused**. In PyTorch, simulate step by step, either by adding one at a time in `torch.float32` or by disabling FMA contraction in Triton.
- **K-direction block length, number of split-K segments, and the segment-length formula**, e.g., 2 segments for K≥4096, 4 segments for K≥8192, segment length = K/S rounded down to a multiple of 128.
- **Edge-tile handling**, and the **final rounding point**: before the fix, rounding once every 64 K; after the fix, rounding only once at the end.

Reference implementation sketch (FP32 chain, fixed order, a single rounding at the end). **For explaining the principle only**: it materializes an (M,N,K) 3-D tensor and will blow up memory on real layers. Use `templates/probe_gemm.py` (row by row) for small probes; for real layers, rewrite it as a tiled torch/Triton implementation.
```python
def gemm_ref(a16, b16, splits):                                  # a16: (M,K), b16: (K,N)
    p = a16.float()[:, None, :] * b16.float().t()[None, :, :]   # p: (M,N,K); FP16×FP16 products are exact in FP32
    acc = torch.zeros(p.shape[:2])
    for s in range(splits):                                      # add segments serially by segment index
        part = torch.zeros(p.shape[:2])
        for k in seg_range(s):                                   # within a segment, add one by one in kernel order
            part = part + p[..., k]
        acc = acc + part
    return acc.half()                                            # round only once
```


## 3. MMA / matrix hardware units
Every vendor's matrix units have a fixed internal pairing order, usually only recoverable by fitting with probes. For example, one piece of hardware uses segments of 128 K, takes quads {b, b+1, b+8, b+9} for `b & 9 == 0`, computes each group as RN(RN(p_b+p_{b+1}) + RN(p_{b+8}+p_{b+9})), and chains the groups in ascending b into a single FP32 chain. Fitting method:
1. Use sparse inputs (only 1–2 nonzero K) to determine which products are paired first;
2. Use inputs with widely different magnitudes to determine at which step rounding happens.

## 4. Reductions
- A pairwise tree within a SIMD group/warp (xor shuffle) and a linear chain give different results.
- A multi-lane denominator may be reassociated by the compiler into two different values, with different output dimensions using different copies. This is only visible from the assembly or from probes.

## 5. Quantized GEMV
Write each quantization format separately: whether scale is applied before or after summation, at which step the offset term is added, which steps are contracted into FMA, the lane stride, and how the dots within a group are accumulated in sequence.

## 6. Hardware fast functions (the most underestimated)
- Hardware fast functions (e.g., `fast::exp2`, hardware reciprocal, and rsqrt on Apple GPU) are usually **not IEEE correctly rounded**. Nor can you expect an approximate instruction on another GPU (e.g., NVIDIA PTX `ex2.approx`) to match them exactly. Measured: for `127/m`, the Apple GPU hardware reciprocal differs from the correctly rounded result on 8108 of 31743 FP16 inputs.
- Reproduce them with **lookup tables**:
  - exp2 = 2^floor(x) · g(floor(frac · 2^23)), with g a 2^23-entry table;
  - reciprocal: enumerate the entire FP16 input domain to build the table;
  - rsqrt(m · 4^k) = rsqrt_hw(m) · 2^-k, also modeling the flush-to-zero rule and sign for subnormals.
- Do not assume: in Metal, both `fast::divide(1,x)` and `1.0f/x` are the hardware reciprocal; only `precise::divide` is IEEE division. Also watch for compiler CSE: different spellings in the same kernel may be merged, so test probes in separate kernels, or you will draw the wrong conclusion.
- If the system math library (cos/sin) is not correctly rounded, model it too. Measured: macOS libm cosf/sinf differ from glibc in FP32 on 17,216 input values, 240 of which appear in the RoPE table. Reproduce with an FP64 baseline plus an exception table; export the exception table from the target platform, because building the table on another platform itself introduces differences.

## 7. Discrete selection
- Top-k routing: the tie-break rule (e.g., lower index wins) and the order in which weights are summed.
- Top-k in the sparse-attention indexer: the cross-head summation order and the SIMD group count must match the target side, and the group count may vary with sequence length (measured on Apple GPU: 8 groups at T=1; at T≥1024, 4 groups when C≤1024, otherwise 2). Measured: in one long sequence, 2 decode positions selected different keys at layer 2, and by the entry of layer 10 this had spread to 1,508 positions. In short sequences the number of keys is below k, so all of them are selected and the problem stays hidden.

## 8. Write-back points
Model every place an intermediate result is written back to low precision, e.g., RMSNorm first computes `deviation(half) *= rstd`, then multiplies by weight, with two intermediate write-backs to FP16; the write-back precision of embedding dequantization must also be modeled.

## 9. Fitting workflow
```
Read the source and list candidate reduction trees → build probes (sparse / wide magnitude range / random / special values)
→ compute the agreement rate for each candidate → put the one that reaches 100% into the reference implementation + unit test → write PROBE-<name>-FIT.md
```
When the agreement rate gets stuck, check first: do inputs come from the same run? Are the dtypes right? Was the dispatch misjudged?
