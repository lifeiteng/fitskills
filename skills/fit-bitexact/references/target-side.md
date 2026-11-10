# Target side: common defects and fixes

The target side fixes only "implementation defects or precision loss". Code examples are labeled by platform (Metal means Apple GPU, CUDA means NVIDIA GPU); for engines such as PyTorch / vLLM / SGLang, find the corresponding kernel or operator implementation and fix it on the same principles. Inherent hardware behavior is left for the reference side to model. Every fix must come with a test, and after the fix, dump again and hand it to the reference side for re-verification.

## 1. float → half / bf16 truncates instead of rounding to nearest
Symptom: the least significant bit of low-precision outputs is systematically biased toward zero.
Fix: prefer the platform's built-in round-to-nearest conversion over hand-written bit manipulation:
- CUDA: `__float2half_rn`, `__float2bfloat16_rn`
- PyTorch: `x.to(torch.float16)`, `x.to(torch.bfloat16)` (both RNE)
- numpy: `x.astype(np.float16)` (RNE)
- Metal: `half(x)`, `static_cast<half>(x)`

If you must hand-write it, handle all of these: round to nearest with ties to even, subnormals, overflow to Inf (from 65520 for FP16), and NaN preserved as a quiet NaN (it must not become Inf because low mantissa bits were dropped).
Test: enumerate all 2^32 floats and compare bit by bit against the platform conversion above (or numpy `astype`); for NaN, compare only "is it NaN".

## 2. fast-math reassociation
Symptom: the result of dequantization `acc * a * b` changes with the compiler; `(1 + x) - 1` is simplified to `x`, and the log1p compensation term disappears.
```metal
// Metal (Apple GPU)
// Before: fast-math may reorder the multiplication into acc * (a * b)
float value = (float)sum * a_scale[m] * b_scale[n];
// After: locally disable fast-math and fix the order with parentheses
{
#pragma METAL fp math_mode(safe)
  value = ((float)sum * a_scale[m]) * b_scale[n];
}
```
```metal
// Metal (Apple GPU)
inline float stable_log1p(float x) {
  // fast-math simplifies (1 + x) - 1 back to x, and the compensation term is gone
#pragma METAL fp math_mode(safe)
  const float xp1 = 1.0f + x;
  if (xp1 == 1.0f) return x;
  return x * (precise::log(xp1) / (xp1 - 1.0f));
}
```
CUDA (NVIDIA GPU) equivalent: use `__fmul_rn` / `__fadd_rn` to prevent contraction into FMA, or add `-fmad=false` for a single file; for Triton and PyTorch custom kernels, also check whether compiler flags contract FMAs or reorder operations.

## 3. Division rewritten as multiplication by an approximate reciprocal
Symptom: the activation-quantization scale `127 / max_abs` differs from the CPU by 1 ulp, so the quantized result of the whole row differs.
```metal
// Metal (Apple GPU): under fast-math, both 127.0f / max_abs and fast::divide compile to multiplication by the hardware reciprocal
float scale = precise::divide(127.0f, max_abs);
```

## 4. Approximate transcendental functions
Switch sigmoid, SwiGLU, exp, etc. to `precise::exp` / `precise::divide`. If performance requires keeping the approximate version, keep it, and let the reference side reproduce it with lookup tables (see reference-side.md).

## 5. Low-precision accumulator (FP16 GEMM)
Symptom: FP16 GEMM accumulates in half, differing by hundreds to hundreds of thousands of elements across shapes (measured over 20 shapes: as few as 835, as many as 694,981). Three common forms: full tiles round once every 64 K; edge tiles round once after accumulating the whole K; split-K partial sums are added in half.
- Use an FP32 accumulator and **round to half only once at the end**.
- Store split-K partial sums in FP32 scratch and **add them serially by segment index** `((s0+s1)+s2)+…`; do not use atomic adds (their order is nondeterministic).
- On the small-M path, each sub-packet also uses FP32 partial sums, added in packet order starting from +0 (or bias), rounded once at the end.
- Edge tiles: if only segment 0 computes the full K and the other segments write +0, document it. An exact −0 plus +0 becomes +0, while a tiny negative number underflows to −0 when converted to half.

## 6. FP32 GEMM with relaxed precision enabled
Symptom: a small overall bias in results, because FP32 operands are internally truncated to a 10-bit mantissa with rounding toward zero (measured on Apple GPU: gate outputs were scaled by about ×0.99965 per token). Fix: disable relaxed precision and measure the performance cost.

## 7. Dispatch-dependent implementation forks
Symptom: the same op uses a custom kernel at ≤16 tokens and a graph framework (e.g., Apple's MPSGraph) at ≥17, or cuBLAS / cuBLASLt picks different algorithms for different shapes, and the two paths produce different results.
Fix: make every path share one core function with a single order of operations.

```metal
// Metal (Apple GPU) sketch: shared by all token counts; selection, tie-break, and weight summation order are defined only here
inline void route_token(device const float* logits, uint token, uint lane,
                        thread uint* experts, thread float* weights);
// T == 1      -> kernel_t1    : route_token
// T <= 16     -> kernel_tn    : route_token
// T  > 16     -> select/group/gather three stages : the first stage also calls route_token
```

## 8. Nondeterministic tie-break
Symptom: when several experts tie, which one is chosen varies with batch size or implementation. For example, MPSGraph top-k picks "larger 5-bit-reversed group index first, then smaller segment index first", while the CPU picks the lowest index. Such problems often surface only in deep layers after all earlier layers are bit-exact.
Fix: make lowest index win the tie-break; sum weights serially in slot order; sort output pairs by (expert, token, slot).
Test: construct tied inputs and verify that T = 1 / 16 / 17 / 100 select the same experts, and that the weights are bit-identical to the single-token case.

## 9. Testing requirements after a fix
- Every fix comes with a unit test that reads the probe's full set of expected values.
- Output order is part of the result: add a test that compares position by position, and run a mutation experiment to prove the test fails when the order is shuffled.
- Confirm the model actually reaches the path. Do not fix paths it never reaches, but state this in the PR.
