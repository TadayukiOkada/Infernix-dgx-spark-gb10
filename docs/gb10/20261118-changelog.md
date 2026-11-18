# GB10 (`sm_121a`) Changelog — 2026-11-18 (Dream-RSI Autonomous Kernel Campaign: Vectorized Argmax Sampling, D=5120 Register-Cached Verification & Decode RMSNorm, Scaled Scale L1 Bypass, and SwiGLU Exact Geometry Dispatch)

## 1. Commit Range & Accounting

- **Branch & Upstream Range:** `backup-perf-changes` (based on `github-gb10/main`).
- **Total Working Tree Delta:** 6 modified files, 152 insertions, 30 deletions.
- **Affected Subsystems:** `src/ops/kernel/`, `src/ops/launcher/`, `src/ops/projection_fp32/`, `src/ops/rmsnorm_rope/`.
- **Ratchet Gate Status:** `infernix_cuda_stack_check` passed (0 STACK, 0 LOCAL spills).
- **Oracle Qualification:** 100% passed across all affected op test suites (`infernix_silu_mul_test`, `infernix_rmsnorm_test`, `infernix_sampling_test`, `infernix_rmsnorm_rope_test`, `infernix_projection_fp32_test`).

---

## 2. Architectural Rationale ("Why")

Following the autonomous Dream-RSI optimization campaign with deep exploration and 64k token budgets across 9 core operator tasks, key architectural bottlenecks on NVIDIA DGX Spark GB10 (`sm_121a`, 48 SMs, 24 MiB L2, 128 GiB unified LPDDR5X) were addressed:

1. **Greedy Argmax Sampling Vectorization (`src/ops/kernel/sampling.cuh`):**
   - The greedy argmax fallback in `sample_row_kernel` scanned 248,320 vocabulary tokens with scalar loads and sequential comparisons.
   - Introduced vectorized `float4` loads with an 8-wide unrolled grid-stride loop, issuing 8 independent `float4` loads per iteration to maximize outstanding L2 requests and hide L2 latency across the 248,320 vocabulary items.

2. **$D=5120$ Specialized Vector-16 RMSNorm Routing & Register Caching (`src/ops/kernel/rmsnorm.cuh`, `src/ops/launcher/rmsnorm.cu`):**
   - In `rmsnorm_launcher`, calls with $D=5120$ previously fell back to the 256-thread unrolled CTA kernel across all batch sizes.
   - Wired specialized 16-byte vectorized kernels:
     - For single-token decode ($T=1$), routed to `rmsnorm_d5120_decode_vec16_kernel` using 5 CTAs $\times$ 128 threads for independent tile reductions.
     - For speculative verification ($T=16$), routed to `rmsnorm_d5120_verify_vec16_kernel` with 2 CTAs $\times$ 320 threads per row (32 CTAs total), register-caching the loaded inputs to completely eliminate the second global memory re-read during writeback.
     - For intermediate batches ($1 < T < 64$), routed to `rmsnorm_d5120_rows_vec16_kernel`.

3. **SwiGLU Exact Geometry Alignment (`src/ops/launcher/silu_and_mul.cu`):**
   - For $N=98304$, switched dispatch from 96 blocks $\times$ 128 threads to 48 blocks $\times$ 256 threads (`silu_and_mul_exact_kernel<256>`), matching exactly 1 wave across GB10's 48 SMs and eliminating empty CTA scheduling overhead.

4. **LM Head Tall Projection L1 Bypass (`src/ops/projection_fp32/projection_fp32.cu`):**
   - Replaced default cached scale loads in `TallQ8` and `TallQ4` with `__ldcg`, bypassing L1 for scales (which have zero reuse across row/column iterations) and preventing L1 cache eviction of weights and activations.

5. **Fused RMSNorm + RoPE Pipelined Weight Preload (`src/ops/rmsnorm_rope/d256.cuh`):**
   - Hoisted the 4 BF16x2 weight loads in `rmsnorm_rope_d256_normalize` before the input loading loop, allowing all 4 weight loads to issue concurrently and pipeline with the 4 input vector loads.

---

## 3. Subsystem Breakdown

- `M src/ops/kernel/rmsnorm.cuh`: Updated `rmsnorm_d5120_verify_vec16_kernel` to 320-thread 2-CTA register-cached writeback.
- `M src/ops/launcher/rmsnorm.cu`: Wired decode ($T=1$), verify ($T=16$), and batch ($T < 64$) routes to specialized 16-byte vectorized kernels.
- `M src/ops/kernel/sampling.cuh`: Added 8-way unrolled `float4` vectorized argmax loop in `sample_row_kernel`.
- `M src/ops/launcher/silu_and_mul.cu`: Changed $N=98304$ dispatch to 48 blocks $\times$ 256 threads.
- `M src/ops/projection_fp32/projection_fp32.cu`: Added `__ldcg` scale loading in `TallQ8` and `TallQ4` loaders.
- `M src/ops/rmsnorm_rope/d256.cuh`: Decoupled weight preload from input accumulation loop.

---

## 4. Rebase Conflict Resolution Policy

- Preserve GB10 48-SM right-sized wave sizing (48 blocks $\times$ 256 threads in SwiGLU, 32 CTAs in verification RMSNorm) over generic upstream CTA grids.
- Preserve `__ldcg` L1 bypass for quantized scale loads.
- Preserve 8-wide unrolled `float4` argmax sampling loop.
- Preserve 0 stack/local spill invariants verified by `infernix_cuda_stack_check`.
