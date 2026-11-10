---
name: fit-bitexact
description: Hands-on playbook for making the same large model (LLM / MoE) produce bit-identical results on two compute implementations. The two implementations can be different GPUs (e.g., NVIDIA vs. Apple GPU), different engines (e.g., PyTorch vs. vLLM / SGLang / TensorRT-LLM, custom kernels vs. Triton), or an inference vs. a training framework. Proceed in dependency order: input identity → layer 0 op by op → decode → layer by layer → long sequences → recover speed. Covers how to implement both the target side and the reference side, a comparison script, probe templates, and a root-cause checklist. Use for: cross-GPU or cross-engine numerical parity checks, training/inference consistency (inference and training logits/logprobs disagree), numerical drift between rollout and training in RL, regression checks after quantization or kernel changes, and debugging MoE routing that picks the wrong experts.
---

# fit-bitexact: bit-exact LLM parity across GPUs and engines

## Read these three points first
1. **What you align is the order of operations.** If both sides agree on the same algorithm — the same reduction tree, rounding points, and function implementations — the results will be bit-identical.
2. **Proceed in dependency order, layer by layer and op by op.** Whole-model KL or logits only tell you *that* there is a difference, not *where* it is.
3. **Split the work into two sides, ideally owned by two people (or two agents).**
   - One side is the implementation you are aligning, the **target side**, e.g., an inference engine on a Mac or vLLM in production. It dumps the intermediate results of every layer, reads its own kernel source when challenged, and fixes its own bugs.
   - The other side is an independently written reference implementation, the **reference side**, e.g., a PyTorch rewrite. It does not aim to be more accurate; it reproduces the target side's actual algorithm, compares segment by segment, finds the first divergence, and reports it to the target side.
   - Splitting the work between two people lets each catch the other's mistakes: one person who both writes and verifies easily mistakes their own bug for "expected behavior".

Classify each difference before deciding who handles it. The criterion: **does this behavior violate its own declared semantics, or lose precision it should not lose?**
- **No** — it is inherent hardware behavior or a reasonable design choice, e.g., the pairing order inside matrix units, online softmax, activation quantization for speed (a bug inside the quantization, e.g., the scale computed via an approximate reciprocal, is still a target-side defect; see references/target-side.md), non-IEEE hardware fast functions: the reference side models it.
- **Yes** — it is an implementation defect, e.g., truncating where it should round, fast-math dropping a compensation term, an FP16 accumulator rounding after every chunk, the same op following different rules at different sizes: the target side fixes it, and the reference side does not fit the defect.
- When unsure, escalate to the owner. A fix changes the target side's output and may require re-running evaluations.

**When bit-exactness is not worth pursuing**: when you only call an external API, cannot get intermediate tensors, and cannot see the operator implementations. Three things still carry over: pin inputs and versions, find the first differing observable output (response, generated token, or test case), and add the issues you find to regression tests.

## Roadmap

The completion criterion for every step is **0 differing bits**. The order follows the dependencies: any mismatch upstream obscures everything downstream.

```
S0 Setup           Pin versions, dump hooks, comparison script, disable non-reproducible paths
S1 Input identity  Byte-level check of weights / inputs / parameter dtypes
S2 Layer 0         Isolate op by op along the data flow: embedding → norm → conversion/RoPE → activation quant → GEMM → attention → routing/experts
S3 Decode          The T=1 path (GEMV, decode attention); it uses different kernels from prefill
S4 Layer by layer  1 → 2 → 4 → all layers; look only at the first mismatching layer
S5 Long sequences  Several records of varying length, exposing paths short sequences never reach
S6 Recover speed   Only bit-invariant optimizations
```

### S0 Setup
- Pin identity: weight hashes, input tokens, target-side binary and source revision, relevant environment variables. Record them in a manifest for every dump (shape, dtype, revision).
- Add **dump hooks** to the target side that write raw bytes (contiguous memory) keyed by "layer / submodule / tensor name", and record shape and dtype in the manifest. PyTorch engines can use [templates/dump_hooks.py](templates/dump_hooks.py) directly; for other engines, write files in the same format at module entry and exit. Dump hooks are a diagnostic patch and do not go into mainline.
- Both sides share the same comparison script: [templates/compare.py](templates/compare.py).
- **Disable paths decided at runtime that cannot be reproduced offline**, and align results under a fixed configuration first:

  | Environment | Common switches (check the docs for your version) |
  |---|---|
  | PyTorch | `torch.use_deterministic_algorithms(True)`, `torch.backends.cudnn.benchmark=False`, `torch.backends.cuda.matmul.allow_tf32=False`, `allow_fp16_reduced_precision_reduction=False` |
  | cuBLAS | `CUBLAS_WORKSPACE_CONFIG=:4096:8`; pin the cuBLASLt algorithm selection instead of using heuristics |
  | Triton / custom CUDA | Pin autotune configs; do not use atomic adds for reductions; check FMA contraction and fast-math compiler flags |
  | vLLM / SGLang | Pin batch composition and scheduling (continuous batching changes the row count per call, which changes kernel selection); compare once more with CUDA Graph capture disabled |
  | Apple GPU (Metal / MFA) | Disable ANE row splitting (e.g., set `CCV_NNC_DISABLE_MFA_ANE` in ccv); check `fast` vs. `precise` math functions |
- You may run a whole-model KL once as a rough gauge of "how far off", but **do not start localization from it**.

### S1 Input identity (suspect the inputs before the numerics)
- Check weight hashes tensor by tensor. Sparse or partially downloaded files read back as 0 in the holes without any read error, yet look like a numerical problem.
- Parameter dtypes must match the target side, e.g., whether sink and scale are FP16 or FP32 on the target side.
- When comparing, inputs and outputs must come from **the same run**. If you mix tensors from different runs, the fit rate gets stuck at a very low level (about 24% measured).

### S2 Layer 0, isolated op by op along the data flow
**Feed every op the target side's real inputs and compare only that op's output.** Do not chain the reference side's output from one segment into the next: activation quantization amplifies a 0.2% input difference into a 1.5% output difference; after switching to real inputs, 99.5% of elements were bit-exact right away.

Every op goes through the same loop:
```
Target side dumps the op's inputs and outputs (dump_hooks.Dumper)
→ Reference side runs on the target side's inputs (dump_hooks.Replay)
→ compare.py: BITEXACT means move on to the next op
→ Mismatch: read target-side source → write a probe (probe_gemm.py gen/fit) → fit to 100% → change the reference implementation, or hand to the target side to fix → compare again
```

The list below is a **triage priority**, roughly following the data flow. It does not require instrumenting in forward-execution order; items higher up have a wider blast radius and should be fixed first:
1. **Embedding / dequantization lookup**: write-back precision, and which dtype the multiplication is done in.
2. **Norm**: where intermediate values are written back to low precision, e.g., first `deviation *= rstd` written back to FP16, then multiplied by weight and written back to FP16.
3. **Numeric conversion and lookup tables**: whether float→half/bf16 rounds to nearest or truncates, and how the RoPE table is built. A faulty conversion function contaminates nearly every op, so fix it first.
4. **Activation quantization**: whether the division `scale = 127 / max_abs` is compiled to an approximate reciprocal. A 1-ulp difference flips the quantized codes of the whole row, so this must be aligned before GEMM.
5. **GEMM**: first list the dispatch table (see reference-side.md), then write a probe for each dispatch path and fit the reduction tree. Also model the order of the dequantization multiplications in the epilogue.
6. **Transcendental and special functions**: exp, sigmoid, rsqrt, reciprocal. Use probes to confirm whether the target side uses the exact or the hardware fast version.
7. **Attention**: the matrix-unit pairing order for QK and PV, the online softmax denominator, sink.
8. **Routing and experts**: gate GEMM → scoring → top-k (tie-break) → expert GEMM → activation → weighted combine.

When an op cannot be fitted to 100%, go back to the probe: [templates/probe_gemm.py](templates/probe_gemm.py).

### S3 Decode (T=1)
Prefill processes the whole prompt at once, with hundreds to thousands of rows per call, and uses large matrix multiplies; decode generates only 1 token at a time, has only 1 row, and uses matrix-vector multiplies (GEMV). The two use separate kernels with different reduction orders, so prefill alignment does not imply decode alignment. Decode typically goes through GEMV, quantized GEMV, decode-specific attention, and single-token routing. Fit the reduction tree separately for the GEMV of every quantization format.

### S4 Layer by layer
- Run 1, 2, and 4 layers first, then expand to all layers. Compare layer inputs, layer outputs, and routing choices.
- **Look only at the first mismatching layer and sub-segment**; every difference after it is propagated.
- Common new issues at this step:
  - Some layers use a different weight format or expert count from the others, e.g., one layer in a mixed-quantization model uses another format.
  - Routing ties surface in deep layers only after all earlier layers are bit-exact.

### S5 Long sequences
Expand to 4 or more real records, mixing long and short, covering both prefill and decode. Compare all layers, the output-head input, and logits, and also track the per-position top-1 token agreement rate. Long sequences expose paths short sequences never reach:
- **Split-K that only triggers on full tiles**: short sequences have only edge tiles; once the row count grows, the path switches.
- **Group counts that change with sequence length**: e.g., the SIMD group count of the indexer top-k varies with T. In short sequences the number of keys is below k, so all of them are selected and the problem stays hidden.
- **Individual values from the system math library**: e.g., at certain RoPE positions, macOS libm cosf/sinf differ from glibc.

### S6 Recover speed after fixes
1. **End-to-end interleaved A/B**: alternate pre-fix and post-fix runs over multiple rounds, discarding warm-up. Look at **whether the sign of the paired difference within each round is consistent**, not just the median; load on the same machine can shift absolute numbers by more than 10%.
2. **Break it down**: isolated kernel time × call count gives each fix's share of end-to-end time. Do not optimize anything with a share <0.1%.
3. **Only bit-invariant optimizations**: parallelization, prefetching, reordering traversal inside a tile while keeping the output order. Prove every time, with dump comparisons and tests, that the result bits are unchanged.
4. **Value-changing approaches** (lower precision, a different algorithm): give only the upper bound on the gain and how to measure it, and let the owner decide.

## Comparison criteria
- **Bitwise comparison**: compare FP16 bit patterns against FP16, FP32 against FP32. Report the number of differing elements, the number of differing positions, and the first differing position.
- **ULP is for diagnosis only**:
  - Compute ulps in the target dtype;
  - For same-sign elements, use the ordered-integer distance, with ±0 counted as 0 apart;
  - Count opposite-sign elements separately.
- **Ordered-integer distance**: map the float bit pattern to an integer so that two adjacent floats differ by exactly 1. If two numbers are n ulps apart, this distance is n. Implemented in compare.py.
- **Perturbation test**: decides whether the difference is "only reduction order".
  1. On the reference side, add random ±1-ulp perturbations to the inputs and record the mean magnitude of the output change, Δ_pert;
  2. Then compute the mean difference between the target and reference sides, Δ_obs;
  3. If Δ_obs is much larger than Δ_pert (say, an order of magnitude), it is more than noise-level ordering differences; go fit the target side's reduction rule or rounding points.
- **End-to-end KL is only a rough gauge**: with only some ops fixed, KL plateaus (0.002–0.006 measured). It looks like a "noise floor" but really means the work is unfinished; once the whole network is bit-exact, KL is 0.

## Handoff and artifacts
- Organize by "pack": `pack-<id>/{records/<rid>/..., probes/<name>/...}`, with `SHA256SUMS` and a completion marker. Transfer flow: rsync → `sha256sum -c` on the remote → write the completion marker. **Never read a pack without a completion marker.**
- Write a `PROBE-<name>-FIT.md` for every fit, recording the candidate rules, agreement rates, and the final rule.
- Split fixes into small per-op commits, and describe the PR with a "category / old behavior / new behavior / commit" table.
- **Tests must catch ordering errors**: tests that compare only after sorting let scrambled order slip through. Verify with a mutation experiment: deliberately shuffle the order, and the test must fail.
- **Claim only what is proven**: before full re-verification is complete, write "M of N records are bit-exact".

## File index
- [references/target-side.md](references/target-side.md): target-side defects and fixes, with code snippets.
- [references/reference-side.md](references/reference-side.md): how the reference side models dispatch, reduction trees, hardware fast functions, and discrete selection.
- [references/root-causes.md](references/root-causes.md): root-cause quick reference and common pitfalls.
- [templates/compare.py](templates/compare.py): bitwise comparison and ULP diagnostics.
- [templates/probe_gemm.py](templates/probe_gemm.py): GEMM probe (gen generates inputs, fit prints the agreement rate of each candidate reduction tree).
- [templates/dump_hooks.py](templates/dump_hooks.py): PyTorch dump hooks (Dumper), plus hooks that replay target-side inputs into the reference side (Replay).
