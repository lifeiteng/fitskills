# Common root causes and modeling checklist

## 1. Target-side implementation defects (fix)
| Category | Symptom | Fix |
|---|---|---|
| Truncating conversion | float→half/bf16 truncates toward zero; the least significant bit is systematically biased | Round to nearest even (RNE), keep NaN as NaN; test by enumerating all 2^32 floats |
| fast-math reassociation | `acc*a*b` gets reordered; `(1+x)-1` gets folded to `x`, losing the log1p compensation | Local safe/precise math mode + explicit parentheses |
| fast-math rewritten division | `c/x` compiled to `c*rcp_approx(x)` | Exact division |
| Approximate transcendentals (conditional) | exp/sigmoid off by 1–2 ulp | Only a defect if it violates the declared semantics (e.g., the source asks for `precise::exp` but fast-math substitutes the approximation) or loses precision it should not; then use exact versions. If the approximate version is a declared design choice (e.g., for speed), it is not a defect: the reference side models it with lookup tables (section 2) |
| Low-precision accumulator | FP16 GEMM accumulates in half, rounding per chunk | FP32 accumulation, split-K partial sums added in FP32 in a fixed segment order, a single rounding at the end |
| relaxed precision | FP32 GEMM truncates mantissas internally | Disable relaxed precision |
| Dispatch-dependent implementation forks (conditional) | Different token counts go through different libraries/implementations with different results | Only a defect if the op declares one rule for all sizes (e.g., batch invariance, a single tie-break) or a path loses precision; then all paths share one core function that defines a single order. If the separate paths are a declared design choice (e.g., GEMV for decode), the reference side models the dispatch table (section 2) |
| Nondeterministic tie-break | Which tied element is selected varies with batch size or implementation | Fixed rule (e.g., lower index wins), weights summed serially in fixed slot order |

## 2. Target-side behaviors the reference side must model (follow)
- **Dispatch table**: per-call row count/shape → concrete kernel. Read thresholds from the source and confirm with probes.
- **GEMM reduction tree**: grouping in the vectorized dot (which products are paired first), intra-/inter-group order, whether FMA is fused, K-direction block length, number of split-K segments and segment-length formula, edge-tile handling (whether a single segment computes the full K), final rounding point.
- **Internal pairing order of matrix hardware units** (tensor cores / simdgroup matrix, etc.): usually requires dedicated probes to fit.
- **Reductions**: pairwise tree vs. linear chain within a SIMD group/warp, cross-lane xor order, compiler reassociation of multi-lane denominators.
- **Quantized GEMV**: unpacking and accumulation order for each quantization format (scale before or after summation, what is contracted into FMA, where the offset term goes).
- **Hardware fast functions**: exp2, reciprocal, and rsqrt are usually not IEEE correctly rounded. The reference side reproduces them bit-exactly on other hardware with lookup tables (using the other framework's approximate functions directly usually matches only a small fraction). Also model the flush-to-zero rule and sign for subnormals.
- **System math library**: if cos/sin etc. are not correctly rounded, reproduce with a high-precision baseline + exception table; the table must be built on the same CPU path as the target.
- **Write-back points**: every place an intermediate result is written back to low precision (e.g., two FP16 write-backs of Norm intermediates).
- **Discrete selection**: top-k group order, and the rule by which the group count varies with sequence length.

## 3. Common pitfalls
- **±0**: if only the first segment computes a split-K edge tile and the other segments contribute +0, an exact −0 plus +0 yields +0, while a tiny negative number underflows to −0 when converted to low precision. Bitwise comparison must be able to explain these.
- **Misjudged shape dispatch**: a shape actually goes through cooperative GEMV rather than the small-M kernel; modeling the wrong kernel leaves a small number of "unexplainable" 1-ulp differences.
- **Unused paths**: first confirm the model really takes the path; do not fix paths it never takes.
- **NaN payload**: when both operands are NaN, FMA on different hardware produces different payloads; handle these separately in comparison.
- **JIT compilation overhead**: first-time compilation in Triton etc. pollutes timing; use a fixed cache directory.
- **Cross-run fitting**: inputs and outputs from different runs make the fit rate stall at a very low level.
