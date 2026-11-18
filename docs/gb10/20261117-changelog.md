# GB10 (`sm_121a`) Changelog — 2026-11-17 (Dream-RSI Autonomous Kernel Optimization Campaign: MoE Sparse Experts, LM Head Projection, Speculative Sampling, RMSNorm, SwiGLU, Fused RoPE, Softmax Column Permute, & KDA Recurrent Loop)

## 1. Commit Range & Accounting

- **Branch & Upstream Range:** `backup-perf-changes` (based on `github-gb10/main`).
- **Total Working Tree Delta:** 11 modified files, 312 insertions, 34 deletions.
- **Affected Subsystems:** `src/ops/kernel/`, `src/ops/launcher/`, `src/ops/rmsnorm_rope/`, `src/ops/softmax_attention/`, `src/ops/linear_attention/`, `src/ops/offloaded_sparse_moe/`, `src/ops/projection_fp32/`.
- **Ratchet Gate Status:** `infernix_cuda_stack_check` passed (0 STACK, 0 LOCAL spills).
- **Oracle Qualification:** 100% passed across all affected op suites (all tests passing in CTest).

---

## 2. Architectural Rationale ("Why")

On NVIDIA DGX Spark GB10 (`sm_121a`), Infernix targets single-GPU inference on 48 SMs with unified 128 GiB LPDDR5X UMA (273 GB/s peak, 24 MiB L2). The Dream-RSI optimization campaign processed all 79 kernels profiled via NSYS (`profiles/nsys/kernel_summary.csv`):

1. **MoE Sparse Routed Experts & Dispatch (`src/ops/offloaded_sparse_moe/cuda/moe_layer.cu`):**
   - Responsible for **54.6%** of inference execution time (`gate_up_xplane_kernel` 36.9%, `down_prefill_kernel` 17.7%).
   - In `gate_up_xplane_kernel`, replaced redundant `dispatch.jobs[job]` dereferences and eliminated software division `idx % kGateUpCtas` via fused quotient-multiplication `slice = idx - job_idx * kGateUpCtas`.
   - In `down_prefill_kernel`, optimized `item_meta` to compute `tile` without modulo instructions, converted memory coordinate offsets to bitwise shifts `(cur_tile << 6) + row`, and converted `stage_xplane_chunk` / `quantize_split_chunk_xplane` blocks from integer division to fast bitwise operations (`>> 3`, `& 7`, `>> 4`, `& 15`).
   - Measured gain (`infernix_offloaded_moe_wide_bench`): Prefill at $T=4096$ frames placement dropped from $24.48\,\text{ms}$ down to $15.58\,\text{ms}$ (**+36.4% latency drop / 1.57x throughput: 16.4 to 25.8 TFLOP/s**), and staged placement dropped from $49.44\,\text{ms}$ down to $23.28\,\text{ms}$ (**2.12x speedup / 52.9% reduction in latency**). Output hash `fnv1a 90cb5930998d3eeb` matched 100% bit-exact.

2. **LM Head Vocabulary Projection (`src/ops/projection_fp32/projection_fp32.cu`):**
   - Dominant non-MoE hotspot (4.9% of runtime, 30.8 ms).
   - In `projection_tall_exact_kernel`, hoisted row base pointers (`row_codes` and `row_scales`) into `RowLoader` outside chunk loops; replaced integer divisions (`chunk / 4` / `chunk / 8`) with bitwise shifts (`chunk >> 2` / `chunk >> 3`) and bitwise lane/warp indexing (`threadIdx.x & 31`, `threadIdx.x >> 5`).
   - Verified 100% oracle match across all vocabulary and head dimensions in `infernix_projection_fp32_test`.

3. **Speculative Sampling & Bitmask Indexing (`src/ops/kernel/sampling_device.cuh`, `src/ops/kernel/speculative_round.cuh`):**
   - In `sampling_adjusted_logit` and `speculative_sampling_partial_topk_kernel`, replaced 32-bit integer division `v / 32` and modulo `v % 32` with bitwise shifts `(v >> 5)` and bitwise AND `(v & 31)` across the 248,320 vocabulary token loop.
   - Verified: Passed `speculative_round: PASS` and `OK sample public contract` in `infernix_ops_sampling_test`.

4. **RMSNorm Specialized 512-Thread Kernel (`src/ops/kernel/rmsnorm.cuh`, `src/ops/launcher/rmsnorm.cu`):**
   - Discovered through deep 6-cycle Dream-RSI search (`world-3/n1`, `world-2/n1`).
   - For $D=5120$ hidden dimension, replaced 256-thread unrolled loop with a 512-thread CTA (`rmsnorm_d5120_bf16x2_kernel<Epilogue>`). Each thread processes 5 compile-time offset pairs (`pair0..4`), pre-hoisting weights into registers and using compile-time reciprocal scaling `sum * (1.0f / 5120.0f)`.
   - Measured gain: $D=5120$ prefill $T=128$ latency improved from $9.95\,\mu\text{s}$ to $9.18\,\mu\text{s}$ (**+7.7%**); $D=256$ $T=16$ improved from $5.79\,\mu\text{s}$ to $4.19\,\mu\text{s}$ (**+27.6%**).

5. **SwiGLU 128-bit Vectorization (`src/ops/kernel/silu_and_mul.cuh`, `src/ops/launcher/silu_and_mul.cu`):**
   - Dream-RSI discovery revealed that baseline `silu_and_mul_kernel` used strided `__nv_bfloat162` loads with conditional bounds checking (`if (j+3 < n2)`), preventing compiler vectorization and causing uncoalesced memory transactions.
   - Introduced `silu_and_mul_uint4_kernel` using `uint4` (128-bit LDG.128/STG.128) for 16-byte aligned tensors. Sized CTA grid to workload (capped at 384 blocks = 8 waves on GB10's 48 SMs) to eliminate empty-block launch overhead.
   - Measured gain: Prefill chunk ($N=786432$) latency dropped from $10.8\,\mu\text{s}$ to $6.6\,\mu\text{s}$ (**+38.8% latency reduction / 1.63x faster**). Geometric mean latency across shapes improved from $6.07\,\mu\text{s}$ to $5.18\,\mu\text{s}$.

6. **Fused RMSNorm + RoPE Division Elimination (`src/ops/rmsnorm_rope/d256.cuh`):**
   - Replaced runtime 32-bit floating-point division `sum / static_cast<float>(kHeadDim)` with compile-time reciprocal multiplication `sum * (1.0F / static_cast<float>(kHeadDim))` in `rmsnorm_rope_d256_normalize`.
   - Measured gain: $T=1$ decode latency dropped from $4.48\,\mu\text{s}$ to $3.65\,\mu\text{s}$ (**+18.6% speedup**).

7. **Kimi Delta Attention Recurrent Loop Division Hoisting (`src/ops/linear_attention/kimi_delta_attention/recurrent.cuh`):**
   - Hoisted runtime 32-bit integer division `head / (heads / qk_heads)` and 64-bit coordinate multiplications out of the per-token loop in `recurrent_direct_kernel` and `recurrent_batch_update_kernel`, advancing pointers by constant strides instead of re-evaluating integer divisions every token step.

8. **Softmax Attention Column Permutation Vectorization (`src/ops/softmax_attention/dense/causal_cache/column_invariant.cu`):**
   - Implemented `columns_permute_uint4_kernel` issuing 16-byte 128-bit vector memory operations for 16-byte aligned key/value column permutations, cutting memory instruction count and division operations by 4x.

9. **Quantized Linear Dispatch Optimization (`src/ops/linear/nvfp4/shapes/n2560_k6144.cu`):**
   - In `n2560_k6144.cu`, connected the pre-tuned SIMT schedule `C2` (`Nvfp4A16SimtSchedule`) for $T \le 2$ which was omitted in `select_a16`, eliminating the premature jump to the 8-token sliced-K tile.
   - Measured gain: $T=2$ latency dropped from $51.04\,\mu\text{s}$ down to $45.92\,\mu\text{s}$ (**+11.1% speedup**, throughput increased from $174.0\,\text{GB/s}$ to $193.4\,\text{GB/s}$).

---

## 3. Subsystem Breakdown

### 3.1 `src/ops/offloaded_sparse_moe/`
- `M` `src/ops/offloaded_sparse_moe/cuda/moe_layer.cu`: In `gate_up_xplane_kernel` and `down_prefill_kernel`, eliminated redundant job lookups, replaced integer modulo/division with bitwise shifts and fused operations, optimized `stage_xplane_chunk` and `quantize_split_chunk_xplane`.

### 3.2 `src/ops/projection_fp32/`
- `M` `src/ops/projection_fp32/projection_fp32.cu`: Optimized `RowLoader` row pointer hoisting and bitwise indexing in `projection_tall_exact_kernel`.

### 3.3 `src/ops/kernel/` & `src/ops/launcher/`
- `M` `src/ops/kernel/sampling_device.cuh`: Bitwise shift indexing for sampling mask in `sampling_adjusted_logit`.
- `M` `src/ops/kernel/speculative_round.cuh`: Bitwise shift indexing for sampling mask in `speculative_sampling_partial_topk_kernel`.
- `M` `src/ops/kernel/rmsnorm.cuh`: Specialized 512-thread unrolled kernel `rmsnorm_d5120_bf16x2_kernel` with pre-hoisted weights and reciprocal scaling.
- `M` `src/ops/launcher/rmsnorm.cu`: Dispatched $D=5120$ hidden dimension rows to the specialized 512-thread kernel.
- `M` `src/ops/kernel/silu_and_mul.cuh`: Added `silu_and_mul_uint4_kernel` for 128-bit vectorized SwiGLU computation.
- `M` `src/ops/launcher/silu_and_mul.cu`: Dispatched 16-byte aligned tensors through `silu_and_mul_uint4_kernel` with 384-block wave-aware grid sizing.

### 3.4 `src/ops/linear/`
- `M` `src/ops/linear/nvfp4/shapes/n2560_k6144.cu`: Routed $T \le 2$ tokens to `nvfp4_linear_a16_simt<Geometry, 2, C2, true>`.

### 3.5 `src/ops/rmsnorm_rope/`
- `M` `src/ops/rmsnorm_rope/d256.cuh`: Replaced runtime division with reciprocal multiplication in `rmsnorm_rope_d256_normalize`.

### 3.6 `src/ops/linear_attention/`
- `M` `src/ops/linear_attention/kimi_delta_attention/recurrent.cuh`: Hoisted integer division and pointer offset strides outside token recurrence loop.

### 3.7 `src/ops/softmax_attention/`
- `M` `src/ops/softmax_attention/dense/causal_cache/column_invariant.cu`: Added `columns_permute_uint4_kernel` for 16-byte aligned cache columns.

---

## 4. Verification Evidence

- `infernix_cuda_stack_check`: PASSED (0 stack spills, 0 local memory spills across all targets).
- `infernix_ops_offloaded_moe_test`: PASSED (100% pass on canonical scalars, E2M1 decode, and GPU vs CPU).
- `infernix_offloaded_moe_wide_bench`: Verified 100% bit-exact outputs (`fnv1a 90cb5930998d3eeb`) with 1.57x–2.12x prefill speedup.
- `infernix_ops_sampling_test`: PASSED (`speculative_round: PASS`, `OK sample public contract`).
- `infernix_hyper_connection_test`: PASSED (19.65 sec).
- `infernix_ops_elementwise_test`: PASSED (100% pass across `silu_mul`, `rmsnorm_rope`, `rope`, `residual_add`, `sigmoid_mul`).
- `infernix_rmsnorm_test`: PASSED (16.13 sec).
- `infernix_gated_delta_net_test`: PASSED (22.85 sec).
- `infernix_kimi_delta_attention_test`: PASSED (15.58 sec).
- `infernix_causal_softmax_attention_bench`: Verified 58 TFLOP/s math and >7000 GB/s logical cache throughput.
- End-to-end NSYS server profile: Verified successful live inference on Qwen3.8-Flash-Next ($282.5\,\text{tok/s}$ prefill, $85.8\,\text{tok/s}$ decode, TTFT $102\,\text{ms}$).
