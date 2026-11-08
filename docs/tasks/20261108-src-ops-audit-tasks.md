# GB10 (`sm_121a`) `src/ops` Package Audit & Optimization Task Roadmap (2026-11-08)

- **Date:** 2026-11-08
- **Target Architecture:** NVIDIA DGX Spark GB10 (`sm_121a`, 48 SMs, 99 KiB L1/SMEM per SM, 65,536 32-bit registers per SM, 24 MiB L2 cache, 128 GiB unified LPDDR5X UMA at ~273 GB/s, 20-core Grace ARMv9.2 CPU, 1:1 FP32/FP16 Tensor Core accumulation parity)
- **Baseline Commit:** `8834c95ff` (`dev-rebased-20261028`)
- **Scope Audited:** Entire `src/ops/` package (954 files across 32 subdirectories)
- **Focus Areas:**
  1. **Specific RTX 5090 (`sm_120a`, 170 SMs, 96 MiB L2, discrete PCIe, half-rate FP32 accumulation) Constants & Heuristics Left in Sources**
  2. **Unnecessary Reads/Writes, Device Workspace Over-Allocations, Host Heap Allocations, and Memory Transfers**
  3. **Inefficient or Redundant Operations, Speculative Dual-Pipeline Kernel Launches, Uncached CUDA Runtime/Driver Queries, and Register/Stack Spills (`255` `STACK > 0` Kernels)**

---

## Mandatory GB10 (`sm_121a`) Workflow & Architectural Invariants

1. **Dead CUDA Depfiles in `build-native` (CRITICAL)**:
   - Editing `.cuh` or `.h` headers does **not** automatically recompile `.cu.o` objects in `build-native`.
   - Whenever you edit any header under `src/ops/`, **always** `touch` the consuming `.cu` files before running Ninja (e.g., `find src/ops/<subdir> -name '*.cu' -exec touch {} +`) and verify `.cu.o` modification timestamps with `stat`.
2. **Single-Invocation Ninja Build & `nvlink` Amortization**:
   - Never run concurrent `ninja` builds against `build-native` (causes `cmake_device_link.o` fatbin hash mismatches `__fatbinwrap_*`).
   - Always build `infernix-serve`, benchmarks, and test binaries in a **single** Ninja invocation to share the ~55.8s device link:
     ```bash
     ninja -C build-native -j "$(nproc)" infernix-serve <test_and_bench_targets...>
     ```
3. **One-Way CUDA Stack-Spill Ratchet Gate (`infernix_cuda_stack_check`)**:
   - Every task must pass `ctest --test-dir build-native -R infernix_cuda_stack_check --output-on-failure`.
   - Never run `check_cuda_stack.py update` to mask newly introduced or increased stack spills. Only run `update` when ratcheting `tools/cuda_stack/cuda_stack_allowlist.json` down.
4. **1:1 FP32 Tensor Core Accumulation & Oracle Verification**:
   - Preserve 1:1 FP32 Tensor Core accumulation on production routes unless a qualified fast path is explicitly measured faster and passes model/oracle gates.
5. **Mandatory GB10 Changelog Entry (`docs/gb10/YYYYMMDD-changelog.md`)**:
   - Every completed task must record its accounting, architectural rationale, subsystem breakdown, verification numbers, regressions/adverse results, and accepted gaps in `docs/gb10/`.

---

## 1. Executive Summary & Audit Inventory Tables

### 1.1 Specific RTX 5090 (`sm_120a`) Constants & Heuristics Left in Active Code (COMPLETED 2026-11-08)

- [x] **96 MiB L2 Cache — [`sparse_moe_small_t_plan.cpp`](file:///opt/etc/Infernix/src/ops/sparse_moe/small_t/sparse_moe_small_t_plan.cpp#L41-L72):** Replaced fixed `8 MiB` `kNextWeightPrefetchLimit` with `device_prefetch_limit()` querying `cudaDevAttrL2CacheSize` and clamping `l2 / 12` to `[2 MiB, 8 MiB]` (`2 MiB` on GB10's 24 MiB L2).
- [x] **96 MiB L2 Cache — [`bf16_vector.cuh`](file:///opt/etc/Infernix/src/ops/common/bf16_vector.cuh#L22-L25) (`add_bias.cu`, `gelu.cu`):** Lowered `kBf16x8CacheSizedMaxElements` from `32M` (`64 MiB`/tensor) to `8M` (`16 MiB`/tensor) for GB10's 24 MiB L2.
- [x] **96 MiB L2 Cache — [`ggml_linear.cu`](file:///opt/etc/Infernix/src/ops/ggml/ggml_linear.cu#L36-L42), [`ggml_linear_a8.h`](file:///opt/etc/Infernix/src/ops/ggml/ggml_linear_a8.h#L16-L20):** Lowered `kPlaneBytes` and `kA8WidePlaneBytes` from `64 MiB` to `16 MiB` (`2/3` of GB10's 24 MiB L2), saving `48 MiB` of workspace per plane while keeping each pass's decoded plane L2-resident.
- [x] **96 MiB L2 Cache — [`nvfp4_a4_quantize.cuh`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a4_quantize.cuh#L165-L173):** Lowered CTA order reversal threshold in `nvfp4_a4_row_scaled_residual_quantize_kernel` from `total_blocks > 1024` (96 MiB L2) to `total_blocks > 256` (24 MiB L2).
- [x] **Discrete PCIe / 1:1 FP32 Acc — [`miss_service.h`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cpu/miss_service.h#L43), [`wide_expert.h`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert.h#L248-L256):** Changed `CpuMissService::Options::pcie_divisor` default from `3` to `0` (UMA bypass) and lowered `kGgmlStageDecodeMinColumns` from `512` to `128` (matching `kBf16WideMinColumns = 128`).
- [x] **170 SMs — [`sparse_moe_decode_kernels.cu`](file:///opt/etc/Infernix/src/ops/sparse_moe/decode/sparse_moe_decode_kernels.cu#L29-L36):** Lowered `kAdaptiveD3Blocks` and `kAdaptiveD4Blocks` from `5 * 512 = 2560` to `16 * 48 = 768` (`16` waves of 48 SMs).
- [x] **170 SMs — [`context_query.cuh`](file:///opt/etc/Infernix/src/ops/softmax_attention/common/context_query.cuh#L19-L21), [`sliding_window/kernel.cuh`](file:///opt/etc/Infernix/src/ops/softmax_attention/sliding_window/kernel.cuh#L72-L77):** Lowered `kContextQueryMaxSplit` from `85` (`170 / 2` SMs) to `64` and used `32` (`kSlidingWindowMaxSplits`) in `sliding_window_attention_reduce_kernel`.
- [x] **170 SMs — [`lossless_attention.cu`](file:///opt/etc/Infernix/src/ops/lossless/lossless_attention.cu#L69):** Lowered `kTargetCtas` from `384` (170 SMs) to `144` (`3 * 48` SMs).
- [x] **170 SMs — [`rmsnorm_rope/launch.cu`](file:///opt/etc/Infernix/src/ops/rmsnorm_rope/launch.cu#L34-L48):** Added `tokens <= 4` branch in `launch_text` launching `rmsnorm_rope_d256_text_kernel<QHeads, KHeads, 1>` (`18 * T` CTAs for `<16, 2>` instead of `6 * T` CTAs).
- [x] **170 SMs — [`launcher/rope.cu`](file:///opt/etc/Infernix/src/ops/launcher/rope.cu#L29-L30), [`lines 98-105`](file:///opt/etc/Infernix/src/ops/launcher/rope.cu#L98-L105):** Lowered `launch_fixed_pair` `DflashText1D` `<8, 32, 8>` split cutoff from `tokens <= 400` (170 SMs) to `tokens <= 128` (48 SMs).
- [x] **170 SMs — [`launcher/embed_gather.cu`](file:///opt/etc/Infernix/src/ops/launcher/embed_gather.cu#L77), [`lines 145-166`](file:///opt/etc/Infernix/src/ops/launcher/embed_gather.cu#L145-L166):** Raised `embed_gather_q8_2048_launch` `Grouped` threshold from `T <= 6` to `T <= 24` (`8 * T` CTAs across 48 SMs) and lowered `<10, 128>` -> `<5, 128>` crossover in `embed_gather_q8_launch` / `embed_gather_fp8_launch` from `T <= 128` / `176` to `T <= 48`.
- [x] **170 SMs — [`launcher/argmax.cu`](file:///opt/etc/Infernix/src/ops/launcher/argmax.cu#L18-L35):** Updated `kFullAggregateBlock` from `128` to `256` and selected `kShortlistAggregateBlock` (`256`, `32` CTAs/col) for `131072` rows across all `t_count`.
- [x] **170 SMs — [`launcher/vision_pos_embed.cu`](file:///opt/etc/Infernix/src/ops/launcher/vision_pos_embed.cu#L15-L31):** Lowered `kSmallPTargetBlocks` from `4096` (`24 * 170` SMs) to `1152` (`24 * 48` SMs) and `x.ne[1] < 1024` (`6 * 170` SMs) to `x.ne[1] < 288` (`6 * 48` SMs).
- [x] **170 SMs — [`kv_cache/append/vq2_launch.cu`](file:///opt/etc/Infernix/src/ops/kv_cache/append/vq2_launch.cu#L49-L60):** Removed hardcoded RTX 5090 `if (rows >= 512 && rows <= 8192)` override in `vq_append_split` so the `2 * multiprocessors` (`96` CTAs on 48-SM GB10) loop governs all row counts.
- [x] **RTX 5090 Tile Tables & Snapshot Plan — [`nvfp4_gdn_snapshot_plan.cpp`](file:///opt/etc/Infernix/src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.cpp#L51-L56), [`fp8_attn_input_a8.cu`](file:///opt/etc/Infernix/src/ops/attn_input_proj/fp8/fp8_attn_input_a8.cu#L119-L125), [`fp8_gdn_input_a8.cu`](file:///opt/etc/Infernix/src/ops/gdn_input_proj/fp8/fp8_gdn_input_a8.cu#L103-L172):** Aligned `nvfp4_gdn_conv_resolve_plan` `AllowA4` `SmallTFusedA16` threshold (`tokens <= 16`) with `nvfp4_gdn_input_plan.cpp`, returned `0` from `fp8_gdn_input_partial_capacity_bytes`, and removed the 170-SM `65..104` / `65..96` `T32R128` detours in `fp8_{attn,gdn}_input_a8_residual_launch`.
- [x] **Native `sm_121a` `QMMA` — [`common/mma.cuh`](file:///opt/etc/Infernix/src/ops/common/mma.cuh#L98-L143):** Emitted native full-rate `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` (`QMMA`) on `__CUDA_ARCH__ == 1210` (`sm_121a`) in `mma_fp8_e4m3`, avoiding dummy `0x7f` block-scale operands required on `sm_120a`.
- [x] **Remaining `/shapes/` Files — `src/ops/linear/{bf16,q4,q6,mxfp8,fp8}/shapes/*`:** Updated all 17 remaining shape files so all 111 shape files in `ckt shapes list` report `origin = GB10` (`q4 [131072, 3072]` `1.23x` faster at `T=64` and `1.43x` faster at `T=128`; `bf16 [248320, 2560]` uses `Grouped, 8` raster at `T >= 97`; `q6 [1152, 1536]` and `q6 [34816, 5120]` aligned to 48-SM wave boundaries; `mxfp8` `launch_f16` aligned to GB10 TMA tiles).
- [x] **Bench Denominators — [`bench/ops/roofline.h`](file:///opt/etc/Infernix/bench/ops/roofline.h):** Aliased roofline constants directly to `gpu_reference.h` (GB10 273 GB/s spec, 246 GB/s sustained read, 97/99/220/440 TFLOP/s), eliminating obsolete RTX 5090 figures (1792 GB/s, 1674.5 GB/s, 838/419/209.5 TF) that caused `mxfp8_flash_next_sweep` and other bench tools to understate efficiency by ~6.5x.
- [x] **GB10 UMA Transfer Fallback — [`expert_offload.h`](file:///opt/etc/Infernix/src/runtime/expert_offload/expert_offload.h#L181-L184), [`expert_offload.cpp`](file:///opt/etc/Infernix/src/runtime/expert_offload/expert_offload.cpp#L400-L425):** Added `kReferenceUmaLinkBytesPerSecond = 19.4e9` fallback when integrated memory (`vram_->query().integrated`) is detected and no unallocated staging buffers are available to probe with; updated `reference_link_bytes_per_second()`, `measure_link()`, `split_columns()`, and `promotion_interval()`.
- [x] **GB10 UMA Transfer Fallback — [`program_impl.h`](file:///opt/etc/Infernix/src/models/qwen4_exp/program/program_impl.h#L6172-L6205):** Added `kReferenceUmaLinkBytesPerSecond = 19.4e9` and `reference_link_bytes_per_second()` helper in `ProgramImpl`, updating `measure_link()`, `promotion_interval()`, and `split_columns()` to scale against `19.4e9` when running on integrated UMA.
- [x] **Flash-Next Diagnostics — [`vram_probe_kernels.cu`](file:///opt/etc/Infernix/tools/flash_next_probe/vram_probe_kernels.cu#L53-L58), [`gpu_probe.cu`](file:///opt/etc/Infernix/tools/flash_next_probe/gpu_probe.cu#L298-L308):** Replaced hardcoded `170` SM blocks in `probe_stream_read` and `pdl_chain` with dynamic queries to `cudaDevAttrMultiProcessorCount`.

| Category | Subsystem / File | Symbol / Constant | RTX 5090 Assumption | Impact on GB10 (`48 SMs`, `24 MiB L2`, UMA, 1:1 FP32 Acc) | Task |
|---|---|---|---|---|---|
| **96 MiB L2 Cache** | [`sparse_moe_small_t_plan.cpp`](file:///opt/etc/Infernix/src/ops/sparse_moe/small_t/sparse_moe_small_t_plan.cpp#L41-L46) | `kNextWeightPrefetchLimit = 8 MiB` | `1/12` of 96 MiB L2 | Prefetches up to `8 MiB` (`33%` of GB10's 24 MiB L2) in `sparse_moe_decode_d4_small_t_kernel`, evicting live MoE activations/weights (unlike `sparse_moe_decode_plan.cpp:25` which clamps to `l2_bytes / 12 = 2 MiB`) | **T1** |
| **96 MiB L2 Cache** | [`bf16_vector.cuh`](file:///opt/etc/Infernix/src/ops/common/bf16_vector.cuh#L22-L25), [`gelu.cu`](file:///opt/etc/Infernix/src/ops/launcher/gelu.cu#L18-L19) | `kBf16x8CacheSizedMaxElements = 32M` (`64 MiB` / tensor, `128 MiB` R+W) | 96 MiB L2 | Uses `.wb`/`.ca` cache hints up to `128 MiB` working set (`5.3x` GB10's 24 MiB L2) instead of streaming `.cs` above `~8–12 MiB` | **T11** |
| **96 MiB L2 Cache** | [`ggml_linear.cu`](file:///opt/etc/Infernix/src/ops/ggml/ggml_linear.cu#L28), [`ggml_linear_a8.h`](file:///opt/etc/Infernix/src/ops/ggml/ggml_linear_a8.h#L20) | `kWidePlaneBytes = 32 MiB`, `kA8WidePlaneBytes = 16 MiB` | 96 MiB L2 | 32 MiB plane exceeds GB10's entire 24 MiB L2 cache | **T10** |
| **96 MiB L2 Cache** | [`nvfp4_a4_quantize.cuh`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a4_quantize.cuh#L171-L173) | `total_blocks > 1024` CTA reversal threshold | 96 MiB L2 (`1024` tokens < 25% of 96 MiB) | At `K = 17408` (`10.88 KiB`/token), 1,024 tokens is `~22 MiB` (`~92%` of 24 MiB L2); forward CTA order at `T = 512..1024` evicts early tokens before TMA GEMM starts | **T5** |
| **Discrete PCIe** | [`miss_service.h`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cpu/miss_service.h#L43), [`moe_layer.cu`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/moe_layer.cu#L3203-L3210) | `pcie_divisor = 3`, `kStageCtas = 16`, `kStageChunk = 16384` | Discrete GPU over PCIe | Stages 1/3 of cache misses via GPU `stage_kernel` copy while running 2/3 on CPU; on GB10 UMA, "PCIe staging" is a redundant LPDDR5X-to-LPDDR5X copy on the same memory controller | **T3** |
| **Discrete PCIe / 96 MiB L2** | [`wide_expert.h`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert.h#L248-L255) | `kGgmlStageDecodeMinColumns = 512` | Calibrated on RTX 5090 (`327 us` vs `365 us`) | Untuned threshold for staging GGML expert weights before wide MMA on GB10 | **T3** |
| **170 SMs** | [`sparse_moe_decode_kernels.cu`](file:///opt/etc/Infernix/src/ops/sparse_moe/decode/sparse_moe_decode_kernels.cu#L32-L33) | `kAdaptiveD3Blocks = 2560`, `kAdaptiveD4Blocks = 2560` | `T=5` on 170 SMs (`~3–5` waves) | Launches `2,560` persistent atomic work-stealing CTAs (`18–26` waves on 48-SM GB10) at [`lines 283, 452, 611, 670`](file:///opt/etc/Infernix/src/ops/sparse_moe/decode/sparse_moe_decode_kernels.cu#L283) | **T2** |
| **170 SMs** | [`context_query.cuh`](file:///opt/etc/Infernix/src/ops/softmax_attention/common/context_query.cuh#L20), [`sliding_window/kernel.cuh`](file:///opt/etc/Infernix/src/ops/softmax_attention/sliding_window/kernel.cuh#L75) | `kContextQueryMaxSplit = 85` (`85 * 2 = 170` SMs) | 170 SMs | Over-allocates shared memory `weights[WarpsPerBlock][85]` in `context_query_reduce_body` even though `context` caps splits at `64` and `sliding_window` caps splits at `32` | **T6** |
| **170 SMs** | [`lossless_attention.cu`](file:///opt/etc/Infernix/src/ops/lossless/lossless_attention.cu#L69) | `kTargetCtas = 384` | `~2.25` waves on 170 SMs | Launches `8` full waves on 48-SM GB10 (`choose_splits` at [`lines 1845-1861`](file:///opt/etc/Infernix/src/ops/lossless/lossless_attention.cu#L1845-L1861)), inflating split-KV partial workspace and `combine_kernel` overhead | **T9** |
| **170 SMs** | [`ggml_linear.cu`](file:///opt/etc/Infernix/src/ops/ggml/ggml_linear.cu#L527) | `kMmaTileTarget = 512` | `~3` tiles/SM on 170 SMs | Forces `split_k` up to 8 (`>10` tiles/SM on 48-SM GB10), increasing `atomicAdd` reduction traffic | **T10** |
| **170 SMs / `sm_120a`** | [`rmsnorm_rope/launch.cu`](file:///opt/etc/Infernix/src/ops/rmsnorm_rope/launch.cu#L26-L36) | `kTextHeadsPerBlock = 3` | Measured on `sm_120a` | At `T=1` with 18 heads (`16Q + 2KV`), launches only **6 CTAs** across 48 SMs (`12.5%` SM utilization) | **T11** |
| **170 SMs** | [`launcher/rope.cu`](file:///opt/etc/Infernix/src/ops/launcher/rope.cu#L57-L68), [`lines 98-105`](file:///opt/etc/Infernix/src/ops/launcher/rope.cu#L98-L105) | `tokens <= 6` (1 CTA/token); `tokens <= 400` (`2000` CTAs) | 170 SMs | At `T=1`, `launch_fixed_text` launches **1 CTA** on the entire GPU (`2.1%` SM utilization); `launch_fixed_pair` launches 7 waves on 48 SMs | **T11** |
| **170 SMs** | [`launcher/embed_gather.cu`](file:///opt/etc/Infernix/src/ops/launcher/embed_gather.cu#L77), [`lines 145-166`](file:///opt/etc/Infernix/src/ops/launcher/embed_gather.cu#L145-L166) | `T <= 6` (`Grouped` vs `Row`), `T <= 176`, `T <= 128` | 170 SMs (`176` CTAs = 1 wave) | At `T = 7..47`, `Row` route launches only `T` CTAs (`7..47` CTAs across 48 SMs, leaving up to 85% of SMs idle) | **T11** |
| **170 SMs** | [`launcher/argmax.cu`](file:///opt/etc/Infernix/src/ops/launcher/argmax.cu#L18-L34) | `kArgmaxBlock = 512` (`8,192` rows/tile) | Large vocab split | For `248,077` vocab rows at `T=1`, launches only **31 CTAs**, leaving 17 of 48 SMs (`35%`) idle | **T11** |
| **170 SMs** | [`launcher/vision_pos_embed.cu`](file:///opt/etc/Infernix/src/ops/launcher/vision_pos_embed.cu#L15), [`lines 29-31`](file:///opt/etc/Infernix/src/ops/launcher/vision_pos_embed.cu#L29-L31) | `kSmallPTargetBlocks = 4096` | `~24` waves on 170 SMs | Forces `warps_per_block = 1` for `patches < 820`, causing 18x redundant global reloads of bilinear patch indices/weights per CTA | **T11** |
| **170 SMs** | [`kv_cache/append/vq2_launch.cu`](file:///opt/etc/Infernix/src/ops/kv_cache/append/vq2_launch.cu#L49-L69) | `vq_append_split` (`rows >= 512 && rows <= 8192`) | Measured on RTX 5090 | Hardcodes RTX 5090 row split factors (`8, 4, 2`) before falling back to `2 * multiprocessors` | **T8** |
| **RTX 5090 Tile Tables** | [`fp8_attn_input_a8.cu`](file:///opt/etc/Infernix/src/ops/attn_input_proj/fp8/fp8_attn_input_a8.cu#L119-L125), [`fp8_gdn_input_a8.cu`](file:///opt/etc/Infernix/src/ops/gdn_input_proj/fp8/fp8_gdn_input_a8.cu#L166-L172), [`fp8_attn_input_n17408_k3072.cu`](file:///opt/etc/Infernix/src/ops/attn_input_proj/fp8/fp8_attn_input_n17408_k3072.cu#L67-L86), [`fp8_gdn_input_n20480_k3072.cu`](file:///opt/etc/Infernix/src/ops/gdn_input_proj/fp8/fp8_gdn_input_n20480_k3072.cu#L74-L102) | `tokens <= 32`, `65..104`, `65..96` band tables | 170-SM RTX 5090 wave boundaries | Residual A8 and 122B A16 input projection launchers were not updated when the 1-term A8 routes were retuned for 48-SM GB10 | **T1** |
| **Half-Rate FP32 Acc** | [`common/mma.cuh`](file:///opt/etc/Infernix/src/ops/common/mma.cuh#L98-L136) | `mma_fp8_e4m3` (`kind::mxf8f6f4.block_scale` + `kUnitScale = 0x7Fu`) | RTX 5090 half-rate `.kind::f8f6f4` | Allocates and passes dummy `0x7Fu` scale registers into every `m16n8k32` FP8 MMA instruction even though GB10 `sm_121a` has 1:1 FP32 accumulation parity | **T5** |
| **Half-Rate FP32 Acc** | [`f16acc_plan.h`](file:///opt/etc/Infernix/src/ops/linear/f16acc/f16acc_plan.h#L3-L6), [`f16acc_mma.cuh`](file:///opt/etc/Infernix/src/ops/linear/f16acc/f16acc_mma.cuh#L511-L527) | `LinearRoutes::F16Accumulate`, `kF16AccSplitRaster`, `rows * tokens < (1 << 28)` | RTX 5090 `505` vs `254` TFLOP/s FP16/FP32 acc | On GB10 (1:1 FP16/FP32 parity), `launch_f16_accumulate_encode` adds an extra global memory read+write pass of `[K, T]` activations with zero Tensor Core throughput gain | **T5** |
| **Untuned `/shapes/`** | `src/ops/linear/{bf16,q4,q6,mxfp8,fp8}/shapes/*` | 16 shape files (`origin: unknown` or RTX 5090 thresholds) | RTX 5090 schedules & `kPairMinTokens` | Suboptimal tile sizes, wave quantization cliffs on 48 SMs, and unused type aliases (`Gemv`, `C2`, `C4`) | **T5** |

---

### 1.2 Unnecessary Reads/Writes, Allocations, and Memory Transfers

| Category | Subsystem / File | Mechanism / Defect | Wasted Memory / Traffic / Overhead | Task |
|---|---|---|---|---|
| **Unfused Global Round-Trip** | [`nvfp4_gdn_snapshot_plan.cpp`](file:///opt/etc/Infernix/src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.cpp#L51-L53) vs [`nvfp4_gdn_input_plan.cpp`](file:///opt/etc/Infernix/src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.cpp#L36) | `AllowA4` switches from `SmallTFusedA16` to `Materialized` at `tokens = 4`, while `nvfp4_gdn_input_plan.cpp` stays on `A16` up to `tokens = 16` | At `T = 4..16` under `AllowA4`, allocates `[10240, T]` BF16 buffer, writes it to global memory in `nvfp4_gdn_input_a16_launch`, and re-reads it in `nvfp4_gdn_snapshot_post_launch` (`36` layers/pass) | **T1** |
| **Dead Workspace + 40x–80x Redundant Global Reads** | [`wide_expert.cu`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert.cu#L368-L370), [`moe_layer.cu`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/moe_layer.cu#L1527-L1744), [`line 1967`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/moe_layer.cu#L1967) | `o.narrow_acts` is allocated in workspace, but `moe_layer.cu:1967` comments out `/*narrow_acts*/` and passes `nullptr` to narrow expert kernels | Every narrow expert job (`40–80` CTAs) redundantly reads BF16 `x` from global memory and re-quantizes it `40–80` times per job (`48` layers/pass) | **T3** |
| **Unused Split-K Workspace** | [`fp8_gdn_input_a8.cu`](file:///opt/etc/Infernix/src/ops/gdn_input_proj/fp8/fp8_gdn_input_a8.cu#L39-L43), [`lines 103-105`](file:///opt/etc/Infernix/src/ops/gdn_input_proj/fp8/fp8_gdn_input_a8.cu#L103-L105) | `fp8_gdn_input_partial_capacity_bytes` returns `Bulk::kPartialBytes` for `max_tokens > 256` even though `kRoutes` never uses `MidBulk` or `Bulk` | Unnecessarily inflates engine workspace reservation for `max_tokens > 256` | **T1** |
| **Unused Staged `k`/`v` Workspace** | [`causal_softmax_attention.cpp`](file:///opt/etc/Infernix/src/ops/softmax_attention/dense/causal_cache/causal_softmax_attention.cpp#L279-L290), [`lines 422-534`](file:///opt/etc/Infernix/src/ops/softmax_attention/dense/causal_cache/causal_softmax_attention.cpp#L422-L534) | `stage_columns` allocates `staged.k` and `staged.v` even when `verify_rows_attention(cache.storage)` is `true` | Wastes `2 * 256 * kv_heads * width * batch * 2` bytes of workspace across BF16, INT8, FP8, NVFP4, and K8V4 caches | **T6** |
| **Unused `Direct` Partial Workspace** | [`context_softmax_attention.cpp`](file:///opt/etc/Infernix/src/ops/softmax_attention/dense/context/context_softmax_attention.cpp#L170-L174) | Calls `allocate_workspace` unconditionally even when `plan.route == ContextAttentionRoute::Direct` | Allocates unused `partial.acc`, `partial.m`, `partial.l` buffers on every `Direct` context attention call | **T6** |
| **Unconditional Route Workspace** | [`mxfp8_dispatch.cpp`](file:///opt/etc/Infernix/src/ops/linear/mxfp8/mxfp8_dispatch.cpp#L41-L56), [`fp8_dispatch.cpp`](file:///opt/etc/Infernix/src/ops/linear/fp8/fp8_dispatch.cpp#L56-L75), [`nvfp4_dispatch.cpp`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_dispatch.cpp#L32-L46) | `*_linear_workspace_capacity_bytes` reserves workspace for `F16Accumulate` / `Pair` routes without checking enabled `LinearRoutes` | Reserves `2 * K * max_tokens` bytes even when those routes are disabled | **T5** |
| **800x Redundant Global Reads + Unfused Combine** | [`resident_moe.cu`](file:///opt/etc/Infernix/src/ops/resident_moe/resident_moe.cu#L179-L219), [`lines 300-315`](file:///opt/etc/Infernix/src/ops/resident_moe/resident_moe.cu#L300-L315) | `gate_up_kernel` reads and converts `x[token]` to FP32 in every CTA (`800x`/token); `down_kernel` reads `h[entry]` `80x`/entry and writes uncombined `[H, k*T]` | `800x` redundant `x` reads, `80x` redundant `h` reads, plus an extra global `[H, k*T]` write+read pass before `moe_combine` | **T4** |
| **8x Redundant Gather + 3-Pass Combine** | [`sparse_moe_prefill_kernels.cu`](file:///opt/etc/Infernix/src/ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu#L302-L320), [`lines 1785-1859`](file:///opt/etc/Infernix/src/ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu#L1785-L1859) | Q8 prefill gathers BF16 `x` `k=8` times into `expert_input`, quantizes `k*T` rows, and runs 3 global passes on expert output (`down` -> `scale` -> `combine`) | `8x` redundant BF16 global copy + quantization, `3x` global read/write passes of `[H, k*T]` | **T2** |
| **1.05 MiB Unconditional Global Zeroing** | [`qsa_select.cu`](file:///opt/etc/Infernix/src/ops/qsa/qsa_select.cu#L923-L960) | `clear_words = l.group * (1 + kScoreBins)` (`262,400` words = `1.05 MiB`) is zeroed even when `n < l.group` or `tile_blocks <= top_blocks` (dense regime) | Writes up to `1.05 MiB` of global memory per `qsa_select` call even when `histograms` and `arrivals` are never read (`12` layers/pass) | **T7** |
| **Uncoalesced Scalar Global/Shared Accesses** | [`qsa_prepare.cu`](file:///opt/etc/Infernix/src/ops/qsa/qsa_prepare.cu#L139-L203), [`qsa_prompt.cu`](file:///opt/etc/Infernix/src/ops/qsa/qsa_prompt.cu#L329-L345), [`context_kv_materialize/kernel.cuh`](file:///opt/etc/Infernix/src/ops/context_kv_materialize/kernel.cuh#L294-L361), [`column_invariant.cu`](file:///opt/etc/Infernix/src/ops/softmax_attention/dense/causal_cache/column_invariant.cu#L31-L58), [`row_major_mma_epilogue.cuh`](file:///opt/etc/Infernix/src/ops/linear_swiglu/row_major_mma_epilogue.cuh#L75-L96) | Stride-32 scalar 1-byte/2-byte loads/stores in `qsa_prepare`, `qsa_prompt`, and `context_kv_mma_kernel`; 4-byte copies in `columns_permute_kernel`; strided scalar BF16 writes in `SwiGluRowMajorMmaEpilogue` | Fragmented sector transactions and 4x–8x excess load/store instruction issue | **T6, T7, T8, T12** |
| **Host Heap Allocations & Sync on Hot Paths** | [`lossless_expert.cuh`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/lossless_expert.cuh#L193-L221), [`wrapper/sparse_moe.cpp`](file:///opt/etc/Infernix/src/ops/wrapper/sparse_moe.cpp#L38-L160), [`weight_input.cpp`](file:///opt/etc/Infernix/src/ops/weight_input.cpp#L17-L324), [`ggml_linear.cu`](file:///opt/etc/Infernix/src/ops/ggml/ggml_linear.cu#L470-L493), [`cpu/*_expert.cpp`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cpu/a4r_expert.cpp#L221-L224), [`replay.cpp`](file:///opt/etc/Infernix/src/ops/linear_attention/gated_delta_net/replay.cpp#L151-L156) | Per-call `std::vector` / `std::string` heap allocations; unpinned heap `cudaMemcpyAsync` + `cudaStreamSynchronize` in `LosslessMoE::forward`; duplicate `concatenate_rows` in `weight_input.cpp` | Host CPU allocator contention, unpinned DMA fallback, and full stream stalls | **T3, T9, T10, T13** |

---

### 1.3 Inefficient / Redundant Operations, Speculative Launches, Uncached Queries, & Spills

| Category | Subsystem / File | Mechanism / Defect | Impact | Task |
|---|---|---|---|---|
| **Double Decode + Double 128-Bit Hadamard** | [`lossless_attention.cu`](file:///opt/etc/Infernix/src/ops/lossless/lossless_attention.cu#L1500-L1592), [`lines 2035-2038`](file:///opt/etc/Infernix/src/ops/lossless/lossless_attention.cu#L2035-L2038) | Launches `codec_pack_kernel` twice (`mode=0` and `mode=1`), decoding `vplane` and computing the 8-stage 128-bit Hadamard transform `hadamard_i128` twice on the same data | Doubles global reads and 128-bit integer arithmetic in lossless value preparation | **T9** |
| **Speculative Dual-Pipeline Launches** | [`sparse_moe_prefill_kernels.cu`](file:///opt/etc/Infernix/src/ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu#L1219-L1386) | Adaptive route (`T=47..52`) launches **both** small-T (`2x 2,560` CTAs) **and** full grouped-prefill pipelines; narrow Q4 (`T=9..46`) launches **both** `Packed` (`1,536` CTAs) and `Spread` (`1,536` CTAs) gate/up kernels | Launches thousands of no-op CTAs that read `plan->use_grouped` in global memory and exit | **T2** |
| **8x Redundant CPU Column Scans & Affinity Syscalls** | [`expert_team.cpp`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cpu/expert_team.cpp#L174-L186), [`lines 244-256`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cpu/expert_team.cpp#L244-L256), [`lines 595-620`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cpu/expert_team.cpp#L595-L620) | Every output slice `s = 0..7` rescans the entire `hidden` column for `a16_column_exponent` / `a4r_row_scale_of`; `CpuExpertTeam::run` calls `pthread_getaffinity_np` + `pthread_setaffinity_np` twice per call | `8x` redundant CPU memory scans + 2 kernel syscalls per MoE layer | **T3** |
| **Per-Warp `powf` Recomputation & Dead Split Code** | [`qsa_prepare.cu`](file:///opt/etc/Infernix/src/ops/qsa/qsa_prepare.cu#L95-L129), [`lines 597-604`](file:///opt/etc/Infernix/src/ops/qsa/qsa_prepare.cu#L597-L604); [`qsa_prompt.cu`](file:///opt/etc/Infernix/src/ops/qsa/qsa_prompt.cu#L143-L145), [`lines 1092-1111`](file:///opt/etc/Infernix/src/ops/qsa/qsa_prompt.cu#L1092-L1111) | `args.inverse[32]` is left unpopulated when `!a.yarn`, forcing `31` warps/column to recompute `powf(a.theta, ...)` on GPU; `qsa_prompt.cu` retains dead `SplitGather` double scale slots | Redundant transcendental SFU pressure in `qsa_prepare` and wasted SMEM in `qsa_prompt` | **T7** |
| **Scalar SIMT Fallback on BF16 FP32-Output Linear** | [`bf16_dispatch.cpp`](file:///opt/etc/Infernix/src/ops/linear/bf16/bf16_dispatch.cpp#L63-L68), [`bf16_general.cu`](file:///opt/etc/Infernix/src/ops/linear/bf16/bf16_general.cu#L42-L57) | Routes `out.dtype == DType::FP32` to `bf16_general_gemm_kernel` (`T > 8`), a naive scalar SIMT loop with zero Tensor Core usage | Orders-of-magnitude slower than `projection_fp32_bf16_mma` / `projection_fp32_bf16_wide_mma` | **T5** |
| **Uncached CUDA Runtime / Driver Calls on Hot Launch Paths** | [`gated_delta_net/chunked/recurrence.cu`](file:///opt/etc/Infernix/src/ops/linear_attention/gated_delta_net/chunked/recurrence.cu#L521-L522), [`kimi_delta_attention/chunked/recurrence.cu`](file:///opt/etc/Infernix/src/ops/linear_attention/kimi_delta_attention/chunked/recurrence.cu#L324-L325), [`materialize.cu`](file:///opt/etc/Infernix/src/ops/context_kv_materialize/materialize.cu#L393-L396), [`causal_softmax_attention.cpp`](file:///opt/etc/Infernix/src/ops/softmax_attention/dense/causal_cache/causal_softmax_attention.cpp#L371-L372), [`sparse_moe_decode_plan.cpp`](file:///opt/etc/Infernix/src/ops/sparse_moe/decode/sparse_moe_decode_plan.cpp#L22), [`sliced_k_pipelined.cuh`](file:///opt/etc/Infernix/src/ops/linear/common/sliced_k_pipelined.cuh#L513), [`ggml_linear_a8.cuh`](file:///opt/etc/Infernix/src/ops/ggml/ggml_linear_a8.cuh#L456-L490) | Uncached `cudaFuncSetAttribute`, `cudaGetDevice`, and `cudaDeviceGetAttribute` on every launch | Adds driver lock / dispatch overhead on every layer launch | **T13** |
| **255 Kernels with `STACK > 0` Spills** | `ggml_linear*` (up to `2,448 B`), `offloaded_sparse_moe` (up to `1,568 B`), `gated_delta_net/recurrent.cu` (`656 B`), `lossless_attention.cu` (`368 B`), `kv_cache/append/vq2_kernels.cuh` (`32–144 B`) | Local-array staging and register pressure exceeding 255/launch-bound registers | Local-memory (`STL`/`LDL`) traffic thrashing L1/L2 cache | **T14** |

---

### 1.4 Prioritized Task Matrix (`T1`–`T14`)

| Task | Priority | Subsystem | Summary |
|---|---|---|---|
| **T1** | **P0** | `gdn_input_proj`, `attn_input_proj`, `sparse_moe/small_t` | Fix NVFP4 GDN conv `T=4..16` unfused materialization bug, remove unused FP8 GDN partial workspace, retune FP8 A8 residual / 122B A16 band tables, and clamp small-T MoE prefetch to GB10 L2 |
| **T2** | **P0** | `sparse_moe` (`decode` & `prefill`) | Right-size 2,560-CTA persistent decode grids for 48 SMs, eliminate speculative dual-pipeline prefill launches, and fuse Q8 prefill gather/scale passes |
| **T3** | **P0** | `offloaded_sparse_moe` (`cuda` & `cpu`) | Wire pre-allocated `narrow_acts` to eliminate 40x–80x redundant per-CTA activation quantization, bypass PCIe staging on GB10 UMA, fix 8x redundant CPU column scans, and remove per-call `pthread_setaffinity_np` & heap allocations |
| **T4** | **P1** | `resident_moe` | Eliminate 800x redundant global `x` reads and 80x redundant `h` reads, and fuse `moe_combine` into `down_kernel` |
| **T5** | **P0** | `linear/*` (`nvfp4`, `fp8`, `mxfp8`, `bf16`, `q4`, `q6`, `f16acc`, `common/mma.cuh`) | Fix `n12800_k2560` threshold & `nvfp4_a4_tma_linear_narrow` `K=5120`, lower L2 reversal threshold from `1024` to GB10 L2 size, gate optional route workspaces, replace scalar BF16 FP32-out fallback with Tensor Core MMA, and retune remaining 16 RTX 5090 `/shapes/` files |
| **T6** | **P1** | `softmax_attention` (`context`, `sliding_window`, `causal_cache`) | Eliminate unused `Direct` partial workspace and unused `staged.k`/`staged.v` allocations, right-size `kContextQueryMaxSplit = 85`, remove redundant SMEM zeroing & barriers, and vectorize `columns_permute_kernel` to 16-byte `uint4` |
| **T7** | **P1** | `qsa` (`qsa_prepare`, `qsa_select`, `qsa_prompt`, `qsa`) | Precompute `args.inverse` for non-YaRN to eliminate 31x/column GPU `powf`, vectorize stride-32 scalar loads/stores, guard `1.05 MiB` `qsa_select` global zeroing, and remove dead `SplitGather` SMEM slots in `qsa_prompt` |
| **T8** | **P1** | `context_kv_materialize`, `kv_cache` | Vectorize scalar stride-32 stores in `context_kv_mma_kernel`, hoist host `std::pow` table in `materialize.cu`, and retune RTX 5090 `vq_append_split` row bands in `vq2_launch.cu` |
| **T9** | **P1** | `lossless`, `offloaded_sparse_moe/cuda/lossless_expert.cuh` | Fuse double `codec_pack_kernel` launch (`hadamard_i128`) in `lossless_attention.cu`, right-size `kTargetCtas = 384` for 48 SMs, and replace unpinned heap `cudaMemcpyAsync` + `cudaStreamSynchronize` in `lossless_expert.cuh` |
| **T10** | **P2** | `ggml` (`ggml_linear*`) | Right-size `kMmaTileTarget = 512` and `kWidePlaneBytes = 32 MiB` for GB10, deduplicate `ggml_encode.cuh` helpers, and eliminate per-call `std::vector<std::vector<Slab>>` heap allocations |
| **T11** | **P1** | `rmsnorm_rope`, `launcher/*`, `common/bf16_vector.cuh` | Retune 170-SM / 96-MiB-L2 constants in `rmsnorm_rope` (`kTextHeadsPerBlock`), `rope.cu`, `embed_gather.cu`, `argmax.cu`, `vision_pos_embed.cu`, `gelu.cu`, and `bf16_vector.cuh` for 48-SM GB10 |
| **T12** | **P1** | `linear_swiglu`, `linear_add`, `linear_pair`, `linear_topk`, `projection_fp32` | Vectorize `SwiGluRowMajorMmaEpilogue::finish_tile` via SMEM staging, prune dead `q8_linear_swiglu` kernel launches, and add vectorized `int8` unpacking in `projection_fp32` |
| **T13** | **P1** | `wrapper/sparse_moe.cpp`, `weight_input.cpp`, `linear_attention`, hot launch paths | Cache all per-launch `cudaFuncSetAttribute` / `cudaGetDevice` / `cudaDeviceGetAttribute` calls and eliminate hot-path `std::vector` / `std::string` allocations and duplicate `concatenate_rows` |
| **T14** | **P2** | `gated_delta_net/recurrent.cu`, `vq2_kernels.cuh`, `ggml_linear*`, `offloaded_sparse_moe` | Burn down `STACK > 0` local-memory spills across the 255 spilling kernel instantiations in `infernix-serve` and ratchet `tools/cuda_stack/cuda_stack_allowlist.json` down |

---

## 2. Detailed Task Specifications

---

### Task T1 (Priority P0): Fix NVFP4 GDN Conv `T = 4..16` Unfused Materialization Bug, Remove Unused FP8 GDN Partial Workspace, Retune FP8 Input-Proj Residual/122B Tables, and Clamp Small-T MoE L2 Prefetch

#### Affected Files & Symbols
- [`src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.cpp`](file:///opt/etc/Infernix/src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.cpp)
  - [`nvfp4_gdn_conv_resolve_plan`](file:///opt/etc/Infernix/src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.cpp#L43-L55)
- [`src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.cpp`](file:///opt/etc/Infernix/src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.cpp)
  - [`nvfp4_gdn_input_resolve_plan`](file:///opt/etc/Infernix/src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.cpp#L33-L41)
- [`src/ops/gdn_input_proj/fp8/fp8_gdn_input_a8.cu`](file:///opt/etc/Infernix/src/ops/gdn_input_proj/fp8/fp8_gdn_input_a8.cu)
  - [`kRoutes`](file:///opt/etc/Infernix/src/ops/gdn_input_proj/fp8/fp8_gdn_input_a8.cu#L39-L43)
  - [`fp8_gdn_input_partial_capacity_bytes`](file:///opt/etc/Infernix/src/ops/gdn_input_proj/fp8/fp8_gdn_input_a8.cu#L103-L105)
  - [`fp8_gdn_input_a8_residual_launch`](file:///opt/etc/Infernix/src/ops/gdn_input_proj/fp8/fp8_gdn_input_a8.cu#L166-L172)
- [`src/ops/attn_input_proj/fp8/fp8_attn_input_a8.cu`](file:///opt/etc/Infernix/src/ops/attn_input_proj/fp8/fp8_attn_input_a8.cu)
  - [`fp8_attn_input_a8_residual_launch`](file:///opt/etc/Infernix/src/ops/attn_input_proj/fp8/fp8_attn_input_a8.cu#L119-L125)
- [`src/ops/attn_input_proj/fp8/fp8_attn_input_n17408_k3072.cu`](file:///opt/etc/Infernix/src/ops/attn_input_proj/fp8/fp8_attn_input_n17408_k3072.cu)
  - [`fp8_attn_input_n17408_a16_launch`](file:///opt/etc/Infernix/src/ops/attn_input_proj/fp8/fp8_attn_input_n17408_k3072.cu#L67-L86)
- [`src/ops/gdn_input_proj/fp8/fp8_gdn_input_n20480_k3072.cu`](file:///opt/etc/Infernix/src/ops/gdn_input_proj/fp8/fp8_gdn_input_n20480_k3072.cu)
  - [`fp8_gdn_input_n20480_a16_launch`](file:///opt/etc/Infernix/src/ops/gdn_input_proj/fp8/fp8_gdn_input_n20480_k3072.cu#L74-L102)
- [`src/ops/sparse_moe/small_t/sparse_moe_small_t_plan.cpp`](file:///opt/etc/Infernix/src/ops/sparse_moe/small_t/sparse_moe_small_t_plan.cpp)
  - [`kNextWeightPrefetchLimit`](file:///opt/etc/Infernix/src/ops/sparse_moe/small_t/sparse_moe_small_t_plan.cpp#L41-L46)
  - [`resolve_prefetch_bytes`](file:///opt/etc/Infernix/src/ops/sparse_moe/small_t/sparse_moe_small_t_plan.cpp#L48-L55)

#### Root Cause & Hardware Rationale
1. **Unfused Materialization Bug at `T = 4..16` in [`nvfp4_gdn_conv_resolve_plan`](file:///opt/etc/Infernix/src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.cpp#L51-L53):**
   - When `nvfp4_gdn_input_plan.cpp` was updated on GB10 so `AllowA4` uses `A16` for `tokens <= 16` ([`line 36`](file:///opt/etc/Infernix/src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.cpp#L36)), `nvfp4_gdn_conv_resolve_plan` at [`line 52`](file:///opt/etc/Infernix/src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.cpp#L52) was left at the old RTX 5090 threshold (`if (tokens <= 3) return {SmallTFusedA16}; return {Materialized};`).
   - Consequently, for `tokens = 4..16` under `AllowA4`, `gdn_input_proj_conv` allocates a `[10240, T]` BF16 workspace tensor, runs `nvfp4_gdn_input_a16_launch` to write it to global memory, and then launches `nvfp4_gdn_snapshot_post_launch` to read it back from global memory, instead of running the single-pass fused `SmallTFusedA16` kernel (which `A16Only` already uses up to `tokens <= 16` at [`line 47`](file:///opt/etc/Infernix/src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.cpp#L47)).
2. **Unused Split-K Partial Buffer in [`fp8_gdn_input_partial_capacity_bytes`](file:///opt/etc/Infernix/src/ops/gdn_input_proj/fp8/fp8_gdn_input_a8.cu#L103-L105):**
   - `kRoutes` ([`lines 39-43`](file:///opt/etc/Infernix/src/ops/gdn_input_proj/fp8/fp8_gdn_input_a8.cu#L39-L43)) only ever selects `Mma64`, `Mma80`, or `Mma96`; `MidBulk` and `Bulk` are dead. Yet `fp8_gdn_input_partial_capacity_bytes` still returns `Bulk::kPartialBytes` whenever `max_tokens > 256` (whereas `fp8_attn_input_partial_capacity_bytes` at [`fp8_attn_input_a8.cu:64-66`](file:///opt/etc/Infernix/src/ops/attn_input_proj/fp8/fp8_attn_input_a8.cu#L64-L66) already returns `0`).
3. **Stale RTX 5090 Band Tables in FP8 Input Projections:**
   - [`fp8_attn_input_a8_residual_launch`](file:///opt/etc/Infernix/src/ops/attn_input_proj/fp8/fp8_attn_input_a8.cu#L119-L125) (`tokens <= 32`, `65..104`) and [`fp8_gdn_input_a8_residual_launch`](file:///opt/etc/Infernix/src/ops/gdn_input_proj/fp8/fp8_gdn_input_a8.cu#L166-L172) (`tokens <= 32`, `65..96`) still use RTX 5090 tile switches instead of matching the GB10-tuned 1-term `kRoutes` table.
   - [`fp8_attn_input_n17408_a16_launch`](file:///opt/etc/Infernix/src/ops/attn_input_proj/fp8/fp8_attn_input_n17408_k3072.cu#L67-L86) and [`fp8_gdn_input_n20480_a16_launch`](file:///opt/etc/Infernix/src/ops/gdn_input_proj/fp8/fp8_gdn_input_n20480_k3072.cu#L74-L102) retain the 170-SM RTX 5090 band tables.
4. **8 MiB L2 Prefetch Thrashing in [`sparse_moe_small_t_plan.cpp`](file:///opt/etc/Infernix/src/ops/sparse_moe/small_t/sparse_moe_small_t_plan.cpp#L41-L55):**
   - `kNextWeightPrefetchLimit = 8 MiB` (`1/12` of RTX 5090's 96 MiB L2) is applied without querying `cudaDevAttrL2CacheSize`. On GB10 (`24 MiB` L2), prefetching `8 MiB` inside `sparse_moe_decode_d4_small_t_kernel` evicts `33%` of L2 during active MoE execution.

#### Required Implementation Steps
1. In [`nvfp4_gdn_conv_resolve_plan`](file:///opt/etc/Infernix/src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.cpp#L51-L53), change `if (tokens <= 3)` to `if (tokens <= 16)` so `AllowA4` matches `nvfp4_gdn_input_resolve_plan`'s `A16` boundary (`tokens <= 16`) and uses `SmallTFusedA16` across `T = 2..16`.
2. In [`fp8_gdn_input_a8.cu`](file:///opt/etc/Infernix/src/ops/gdn_input_proj/fp8/fp8_gdn_input_a8.cu#L103-L105), return `0` from `fp8_gdn_input_partial_capacity_bytes` and remove the dead `MidBulk` / `Bulk` instantiations.
3. Align [`fp8_attn_input_a8_residual_launch`](file:///opt/etc/Infernix/src/ops/attn_input_proj/fp8/fp8_attn_input_a8.cu#L119-L125) and [`fp8_gdn_input_a8_residual_launch`](file:///opt/etc/Infernix/src/ops/gdn_input_proj/fp8/fp8_gdn_input_a8.cu#L166-L172) (plus `n17408_k3072.cu` and `n20480_k3072.cu`) with the GB10-swept tile boundaries.
4. In [`sparse_moe_small_t_plan.cpp`](file:///opt/etc/Infernix/src/ops/sparse_moe/small_t/sparse_moe_small_t_plan.cpp#L41-L55), clamp `kNextWeightPrefetchLimit` to `static_cast<std::size_t>(l2_bytes) / 12` (`2 MiB` on GB10) using a cached per-device L2 size query, matching [`sparse_moe_decode_plan.cpp`](file:///opt/etc/Infernix/src/ops/sparse_moe/decode/sparse_moe_decode_plan.cpp#L16-L31).

#### Verification & Acceptance Criteria
```bash
ninja -C build-native -j "$(nproc)" infernix-serve infernix_gdn_input_proj_test infernix_attn_input_proj_test infernix_sparse_moe_test
ctest --test-dir build-native -R "gdn_input_proj|attn_input_proj|sparse_moe|infernix_cuda_stack_check" --output-on-failure
```
- **Acceptance Criteria:** All oracle tests and `infernix_cuda_stack_check` pass; `nvfp4_gdn_conv_resolve_plan(T, AllowA4)` selects `SmallTFusedA16` with `0` workspace bytes for `T = 4..16`; `fp8_gdn_input_partial_capacity_bytes` returns `0`.

---

### Task T2 (Priority P0): Right-Size 2,560-CTA Persistent Grids, Eliminate Speculative Dual-Pipeline Launches, and Fuse Q8 Prefill Gather/Combine in `sparse_moe`

#### Affected Files & Symbols
- [`src/ops/sparse_moe/decode/sparse_moe_decode_kernels.cu`](file:///opt/etc/Infernix/src/ops/sparse_moe/decode/sparse_moe_decode_kernels.cu)
  - [`kAdaptiveD3Blocks`, `kAdaptiveD4Blocks`](file:///opt/etc/Infernix/src/ops/sparse_moe/decode/sparse_moe_decode_kernels.cu#L32-L33)
  - Launch sites at [`lines 283, 452, 611, 670`](file:///opt/etc/Infernix/src/ops/sparse_moe/decode/sparse_moe_decode_kernels.cu#L283)
- [`src/ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu`](file:///opt/etc/Infernix/src/ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu)
  - [`sparse_moe_prefill_gather_kernel`](file:///opt/etc/Infernix/src/ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu#L302-L320)
  - [`launch_sparse_moe_prefill_pipeline`](file:///opt/etc/Infernix/src/ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu#L1219-L1386)
  - [`launch_sparse_moe_q8_prefill`](file:///opt/etc/Infernix/src/ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu#L1785-L1859)

#### Root Cause & Hardware Rationale
1. **2,560-CTA Persistent Grids Sized for 170-SM RTX 5090:**
   - [`kAdaptiveD3Blocks = 5 * kIntermediate`](file:///opt/etc/Infernix/src/ops/sparse_moe/decode/sparse_moe_decode_kernels.cu#L32-L33) (`2,560`) and `kAdaptiveD4Blocks = 5 * (kHidden / 4)` (`2,560`) launch 2,560 persistent CTAs that pull work via `atomicAdd` from a global counter (`lines 283, 452, 611, 670`). On 48-SM GB10 (`2–3` CTAs/SM), a persistent work-stealing grid only needs `1` resident wave (`96–144` CTAs) or `min(active_tiles, kAdaptiveBlocks)`; launching 2,560 CTAs schedules ~20 waves of empty tail CTAs competing on `atomicAdd(work_counter, 1)`.
2. **Speculative Dual-Pipeline Launches in [`sparse_moe_prefill_kernels.cu`](file:///opt/etc/Infernix/src/ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu#L1219-L1386):**
   - For the adaptive route (`T = 47..52`), `launch_sparse_moe_prefill_pipeline` launches **both** the small-T pipeline (`2 x 2,560` CTAs) **and** the entire 7-kernel grouped-prefill pipeline on the stream, relying on every kernel checking `plan->use_grouped` in global memory and exiting.
   - For narrow Q4 prefill (`T = 9..46`), [`lines 1353-1386`](file:///opt/etc/Infernix/src/ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu#L1353-L1386) launch **both** `Packed` (`1,536` CTAs) and `Spread` (`1,536` CTAs) gate/up kernels back-to-back.
3. **8x Redundant Gather & 3-Pass Combine in [`launch_sparse_moe_q8_prefill`](file:///opt/etc/Infernix/src/ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu#L1785-L1859):**
   - Launches `sparse_moe_prefill_gather_kernel` to copy each token's BF16 activation `k = 8` times into `expert_input`, then quantizes `k * T` rows in `q8_quantize_x_kernel` (8x redundant quantization over `T` tokens), launches uncompacted `dim3(16, 256) = 4,096` and `dim3(32, 256) = 8,192` grids where inactive experts early-exit, and makes 3 global memory passes over `[H, k*T]` (`down_proj` -> `expert_scale_kernel` -> `moe_combine_launch`).

#### Required Implementation Steps
1. Right-size the persistent work-stealing grid in [`sparse_moe_decode_kernels.cu`](file:///opt/etc/Infernix/src/ops/sparse_moe/decode/sparse_moe_decode_kernels.cu#L32-L33) to the resident SM capacity on GB10 (or bound by `tokens * kIntermediate`) so persistent CTAs loop without launching 20 empty waves.
2. Benchmark and resolve the `T = 47..52` adaptive route and `T = 9..46` `Packed` vs `Spread` route statically on GB10 so only one pipeline is launched per layer.
3. In [`launch_sparse_moe_q8_prefill`](file:///opt/etc/Infernix/src/ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu#L1785-L1859), quantize the `T` input columns **once** (`[K, T]`), have the gate/up kernel gather quantized rows via `row_to_token[row]`, and fuse `expert_scale` into the down-projection epilogue or `moe_combine`.

#### Verification & Acceptance Criteria
```bash
ninja -C build-native -j "$(nproc)" infernix-serve infernix_sparse_moe_test infernix_sparse_moe_bench
ctest --test-dir build-native -R "sparse_moe|infernix_cuda_stack_check" --output-on-failure
```
- **Acceptance Criteria:** Zero regressions in `infernix_sparse_moe_test` and `infernix_cuda_stack_check`; measurable reduction in kernel launch count and latency across `T = 5..52` and Q8 prefill in `infernix_sparse_moe_bench`.

---

### Task T3 (Priority P0): Wire `narrow_acts` Pre-Quantization, Bypass Redundant UMA Staging, and Fix 8x Redundant CPU Scans & Per-Call Syscalls/Allocations in `offloaded_sparse_moe`

#### Affected Files & Symbols
- [`src/ops/offloaded_sparse_moe/cuda/wide_expert.cu`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert.cu)
  - [`carve_workspace`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert.cu#L368-L370)
- [`src/ops/offloaded_sparse_moe/cuda/moe_layer.cu`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/moe_layer.cu)
  - [`gate_up_kernel` / narrow expert launchers](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/moe_layer.cu#L1527-L1744)
  - [`launch_expert_pass`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/moe_layer.cu#L1967)
  - [`stage_kernel` & `pcie_share`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/moe_layer.cu#L276-L277), [`lines 3203-3210`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/moe_layer.cu#L3203-L3210)
- [`src/ops/offloaded_sparse_moe/cuda/ggml_expert_a8.cuh`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/ggml_expert_a8.cuh#L248-L255)
- [`src/ops/offloaded_sparse_moe/cuda/fp8_expert.cuh`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/fp8_expert.cuh#L194-L206)
- [`src/ops/offloaded_sparse_moe/cuda/nvfp4_expert_a8.cuh`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/nvfp4_expert_a8.cuh#L278-L288)
- [`src/ops/offloaded_sparse_moe/cpu/miss_service.h`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cpu/miss_service.h#L43)
- [`src/ops/offloaded_sparse_moe/cpu/expert_team.cpp`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cpu/expert_team.cpp)
  - [`CpuExpertTeam::work_a16`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cpu/expert_team.cpp#L174-L186)
  - [`CpuExpertTeam::work_a4r`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cpu/expert_team.cpp#L244-L256)
  - [`CpuExpertTeam::run`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cpu/expert_team.cpp#L595-L620)
- [`src/ops/offloaded_sparse_moe/cpu/a4r_expert.cpp`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cpu/a4r_expert.cpp#L221-L224), [`w4a16_expert.cpp`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cpu/w4a16_expert.cpp#L145-L148), [`w4a4_expert.cpp`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cpu/w4a4_expert.cpp#L183-L186), [`w4a8_expert.cpp`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cpu/w4a8_expert.cpp#L180-L183)

#### Root Cause & Hardware Rationale
1. **Dead `narrow_acts` Workspace & 40x–80x Redundant Per-CTA Activation Quantization:**
   - [`carve_workspace`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert.cu#L368-L370) already carves `o.narrow_acts` in device workspace, and `moe_layer.cu:1905` passes `layout.narrow_acts` into [`launch_expert_pass`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/moe_layer.cu#L1967), where `int /*columns*/, ActCodes* /*narrow_acts*/` are commented out!
   - For narrow jobs (`count <= 4`), every expert launches `40` to `80` CTAs across output row slices, and **every single CTA** reads the BF16 input columns from global memory and quantizes them from scratch (`40–80x` redundant global reads and quantization per expert job).
2. **Redundant "PCIe" Staging on GB10 UMA (`pcie_divisor = 3`):**
   - [`MissServiceConfig::pcie_divisor = 3`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cpu/miss_service.h#L43) and [`moe_layer.cu:3203-3210`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/moe_layer.cu#L3203-L3210) copy 1/3 of missed experts from host-mapped memory into GPU staging slots via `stage_kernel` before reading them again in the expert kernel. On GB10 UMA (`hostMemoryOrderedAlloc` / unified LPDDR5X), host and device memory share the exact same LPDDR5X controllers, so staging copies data from LPDDR5X to LPDDR5X before reading it a second time.
3. **8x Redundant Full-Column CPU Scans & Per-Call `pthread_setaffinity_np` Syscalls:**
   - In [`CpuExpertTeam::work_a16`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cpu/expert_team.cpp#L174-L186) and [`work_a4r`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cpu/expert_team.cpp#L244-L256), each of the 8 output-row slices (`s = 0..7`) independently scans the entire `hidden`-element input column to compute `a16_column_exponent` or `a4r_row_scale_of`.
   - [`CpuExpertTeam::run`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cpu/expert_team.cpp#L595-L620) calls `pthread_getaffinity_np` and `pthread_setaffinity_np` twice on the calling thread on every single `run()` invocation.
   - The standalone CPU expert paths (`a4r_expert.cpp`, `w4a16_expert.cpp`, `w4a4_expert.cpp`, `w4a8_expert.cpp`) allocate 3 `std::vector`s on the heap on every call.

#### Required Implementation Steps
1. Wire `narrow_acts` in [`moe_layer.cu`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/moe_layer.cu#L1967) (or remove the dead workspace allocation if a codec does not use it): pre-quantize narrow job activations once into `narrow_acts` so all `40–80` CTAs of each narrow expert job load pre-quantized codes+scales via coalesced `uint4` loads.
2. On GB10 / integrated UMA (`deviceProp.integrated == 1` or zero-staging mode), read zero-copy host-mapped expert frames directly without copying through `stage_kernel`.
3. In [`expert_team.cpp`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cpu/expert_team.cpp#L174-L256), compute `a16_column_exponent` / `a4r_row_scale_of` once per column during `prepare_a16` / `prepare_a4r` instead of 8 times across `s = 0..7`, and cache thread affinity once at thread initialization instead of calling `pthread_setaffinity_np` twice per `run()`.
4. Replace per-call `std::vector` allocations in `cpu/*_expert.cpp` with stack/reusable buffers.

#### Verification & Acceptance Criteria
```bash
find src/ops/offloaded_sparse_moe -name '*.cu' -exec touch {} +
ninja -C build-native -j "$(nproc)" infernix-serve infernix_offloaded_sparse_moe_test infernix_offloaded_moe_wide_bench
ctest --test-dir build-native -R "offloaded_sparse_moe|infernix_cuda_stack_check" --output-on-failure
```
- **Acceptance Criteria:** `infernix_offloaded_sparse_moe_test` and `infernix_cuda_stack_check` pass 100%; narrow expert latency and CPU miss-service latency improve measurably without increasing workspace size.

---

### Task T4 (Priority P1): Eliminate 800x Redundant Global `x` Reads, 80x Redundant `h` Reads, and Unfused Combine in `resident_moe`

#### Affected Files & Symbols
- [`src/ops/resident_moe/resident_moe.cu`](file:///opt/etc/Infernix/src/ops/resident_moe/resident_moe.cu)
  - [`gate_up_kernel`](file:///opt/etc/Infernix/src/ops/resident_moe/resident_moe.cu#L179-L219)
  - [`down_kernel`](file:///opt/etc/Infernix/src/ops/resident_moe/resident_moe.cu#L300-L315)
  - [`launch_resident_moe`](file:///opt/etc/Infernix/src/ops/resident_moe/resident_moe.cu#L350-L425)

#### Root Cause & Hardware Rationale
1. In [`gate_up_kernel`](file:///opt/etc/Infernix/src/ops/resident_moe/resident_moe.cu#L179-L219), `gridDim = (intermediate / 8, k * tokens)` launches `100` row-slice CTAs per routed `(token, expert)` entry (`800` CTAs per token for `k = 8, intermediate = 800`), and **every CTA** independently reads `x[token]` from global memory and converts BF16 to FP32 in shared memory (`800x` redundant reads and conversions per token).
2. In [`down_kernel`](file:///opt/etc/Infernix/src/ops/resident_moe/resident_moe.cu#L300-L315), `gridDim = (hidden / 64, k * tokens)` launches `80` CTAs per routed entry, each re-reading `h[entry]` (`80x` redundant reads), and writes uncombined `[H, k * T]` to global memory followed by a separate [`moe_combine`](file:///opt/etc/Infernix/src/ops/resident_moe/resident_moe.cu#L412-L420) kernel pass.

#### Required Implementation Steps
1. Tile `gate_up_kernel` and `down_kernel` across larger row blocks (or group routed tokens per expert at `T > 1`) and use vectorized `uint4` loads of BF16 activations (converting to FP32 in registers rather than storing 4-byte FP32 in shared memory, cutting SMEM usage in half).
2. Fuse weighted accumulation (`weight * sum`) into `down_kernel` (or atomic/in-register combine for `T = 1..4`) to eliminate the intermediate `[H, k * T]` global memory write+read round-trip.

#### Verification & Acceptance Criteria
```bash
ninja -C build-native -j "$(nproc)" infernix-serve infernix_resident_moe_test
ctest --test-dir build-native -R "resident_moe|infernix_cuda_stack_check" --output-on-failure
```
- **Acceptance Criteria:** `infernix_resident_moe_test` and `infernix_cuda_stack_check` pass 100%; global memory traffic and kernel launch count are reduced.

---

### Task T5 (Priority P0): Linear Subsystems (`nvfp4`, `fp8`, `mxfp8`, `bf16`, `q4`, `q6`, `f16acc`, `common/mma.cuh`) — Fix Thresholds, L2 Reversal, Route Workspaces, BF16 FP32-Out Fallback, and 16 Untuned RTX 5090 `/shapes/` Files

#### Affected Files & Symbols
- [`src/ops/linear/nvfp4/shapes/n12800_k2560.cu`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n12800_k2560.cu#L59-L69)
- [`src/ops/linear/nvfp4/nvfp4_a4_tma.cu`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a4_tma.cu#L53-L67)
- [`src/ops/linear/nvfp4/nvfp4_a4_quantize.cuh`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a4_quantize.cuh#L171-L173)
- [`src/ops/linear/bf16/bf16_dispatch.cpp`](file:///opt/etc/Infernix/src/ops/linear/bf16/bf16_dispatch.cpp#L63-L68) & [`bf16_general.cu`](file:///opt/etc/Infernix/src/ops/linear/bf16/bf16_general.cu#L42-L57)
- [`src/ops/linear/mxfp8/mxfp8_dispatch.cpp`](file:///opt/etc/Infernix/src/ops/linear/mxfp8/mxfp8_dispatch.cpp#L41-L56), [`fp8_dispatch.cpp`](file:///opt/etc/Infernix/src/ops/linear/fp8/fp8_dispatch.cpp#L56-L75), [`nvfp4_dispatch.cpp`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_dispatch.cpp#L32-L46)
- [`src/ops/common/mma.cuh`](file:///opt/etc/Infernix/src/ops/common/mma.cuh#L98-L136)
- [`src/ops/linear/f16acc/f16acc_mma.cuh`](file:///opt/etc/Infernix/src/ops/linear/f16acc/f16acc_mma.cuh#L511-L527)
- **Untuned RTX 5090 `/shapes/` files (`16` files):**
  - BF16: [`n48_k2560.cu`](file:///opt/etc/Infernix/src/ops/linear/bf16/shapes/n48_k2560.cu), [`n256_k5120.cu`](file:///opt/etc/Infernix/src/ops/linear/bf16/shapes/n256_k5120.cu), [`n1664_k2560.cu`](file:///opt/etc/Infernix/src/ops/linear/bf16/shapes/n1664_k2560.cu), [`n2560_k2560.cu`](file:///opt/etc/Infernix/src/ops/linear/bf16/shapes/n2560_k2560.cu), [`n5120_k6144.cu`](file:///opt/etc/Infernix/src/ops/linear/bf16/shapes/n5120_k6144.cu), [`n14336_k5120.cu`](file:///opt/etc/Infernix/src/ops/linear/bf16/shapes/n14336_k5120.cu), [`n248320_k2560.cu`](file:///opt/etc/Infernix/src/ops/linear/bf16/shapes/n248320_k2560.cu)
  - Q4 / Q6: [`q4/shapes/n131072_k3072.cu`](file:///opt/etc/Infernix/src/ops/linear/q4/shapes/n131072_k3072.cu), [`q6/shapes/n1152_k1536.cpp`](file:///opt/etc/Infernix/src/ops/linear/q6/shapes/n1152_k1536.cpp), [`q6/shapes/n34816_k5120.cpp`](file:///opt/etc/Infernix/src/ops/linear/q6/shapes/n34816_k5120.cpp)
  - MXFP8: [`n1280_k2560.cu`](file:///opt/etc/Infernix/src/ops/linear/mxfp8/shapes/n1280_k2560.cu), [`n2560_k640.cu`](file:///opt/etc/Infernix/src/ops/linear/mxfp8/shapes/n2560_k640.cu), [`n2560_k6144.cu`](file:///opt/etc/Infernix/src/ops/linear/mxfp8/shapes/n2560_k6144.cu), [`n13952_k2560.cu`](file:///opt/etc/Infernix/src/ops/linear/mxfp8/shapes/n13952_k2560.cu), [`n16384_k2560.cu`](file:///opt/etc/Infernix/src/ops/linear/mxfp8/shapes/n16384_k2560.cu)
  - FP8: [`n1280_k2560.cu`](file:///opt/etc/Infernix/src/ops/linear/fp8/shapes/n1280_k2560.cu), [`n2560_k640.cu`](file:///opt/etc/Infernix/src/ops/linear/fp8/shapes/n2560_k640.cu), [`n2560_k6144.cu`](file:///opt/etc/Infernix/src/ops/linear/fp8/shapes/n2560_k6144.cu), [`n13312_k2560.cu`](file:///opt/etc/Infernix/src/ops/linear/fp8/shapes/n13312_k2560.cu), [`n16384_k2560.cu`](file:///opt/etc/Infernix/src/ops/linear/fp8/shapes/n16384_k2560.cu)

#### Root Cause & Hardware Rationale
1. **Threshold Mismatch in [`nvfp4/shapes/n12800_k2560.cu`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n12800_k2560.cu#L59-L69) & Missing `case 5120:` in [`nvfp4_a4_tma.cu`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a4_tma.cu#L53-L67):**
   - `n12800_k2560.cu` documents that the gate at `65` leaves `T = 48` to `A16` (`100.3 us` vs `119.1 us`), yet `uses_a4` (`line 59`) and `pair_min_tokens` (`line 69`) are set to `33`.
   - `launch_nvfp4_a4_tma_linear_narrow` (`nvfp4_a4_tma.cu:53-67`) handles `K = 2560, 6144, 17408` but omits `case 5120:`.
2. **L2 Reversal Threshold (`total_blocks > 1024`) in [`nvfp4_a4_quantize.cuh`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a4_quantize.cuh#L171-L173):**
   - Sized for 96 MiB L2; on GB10's 24 MiB L2, `T = 512..1024` at `K = 17408` (`11–22 MiB`) already evicts early tokens when quantized in ascending order.
3. **Scalar SIMT Fallback for BF16 Linear with FP32 Output ([`bf16_dispatch.cpp`](file:///opt/etc/Infernix/src/ops/linear/bf16/bf16_dispatch.cpp#L63-L68)):**
   - Any `bf16_dispatch` call with `out.dtype == DType::FP32` and `T > 8` runs [`bf16_general_gemm_kernel`](file:///opt/etc/Infernix/src/ops/linear/bf16/bf16_general.cuh#L24-L60), which loads one scalar `__nv_bfloat16` at a time and computes scalar `fmaf` without Tensor Cores, instead of delegating to `projection_fp32_bf16_mma` / `projection_fp32_bf16_wide_mma`.
4. **Dummy `0x7Fu` Block-Scale Registers in [`mma_fp8_e4m3`](file:///opt/etc/Infernix/src/ops/common/mma.cuh#L98-L136):**
   - Uses `.kind::mxf8f6f4.block_scale.scale_vec::1X` with `kUnitScale = 0x7Fu` to work around RTX 5090's half-rate `.kind::f8f6f4`. On GB10 `sm_121a`, test plain `.kind::f8f6f4` (no scale registers) to save register pressure and instruction overhead.
5. **16 Untuned `/shapes/` Files Across `bf16`, `q4`, `q6`, `mxfp8`, and `fp8`:**
   - Identified via `ckt shapes list` as still carrying RTX 5090 schedules or `kPairMinTokens` thresholds.

#### Required Implementation Steps
1. Reconcile `uses_a4` / `pair_min_tokens` in [`n12800_k2560.cu`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n12800_k2560.cu#L59-L69) via cold-L2 benchmark at `T = 32, 48, 64`, and add `case 5120: launch<5120, 128, 4>(p, output, stream); return;` in [`launch_nvfp4_a4_tma_linear_narrow`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a4_tma.cu#L53-L67).
2. Lower the CTA reversal threshold in [`nvfp4_a4_row_scaled_residual_quantize_kernel`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a4_quantize.cuh#L171-L173) (and `nvfp4_a4_quantize_kernel` / `nvfp4_a4_residual_quantize_kernel`) based on total quantized bytes vs GB10's 24 MiB L2 (`e.g. total_blocks * K >= 2 * 1024 * 1024`).
3. Route `out.dtype == DType::FP32` in [`bf16_dispatch`](file:///opt/etc/Infernix/src/ops/linear/bf16/bf16_dispatch.cpp#L63-L68) through Tensor Core `projection_fp32_bf16_mma` / `projection_fp32_bf16_wide_mma` whenever `N % 64 == 0 && K % 64 == 0`.
4. Overload or update `mxfp8_linear_workspace_capacity_bytes`, `fp8_linear_workspace_capacity_bytes`, and `nvfp4_linear_workspace_capacity_bytes` to accept `LinearRoutes routes` so disabled routes reserve `0` bytes.
5. Sweep and retune the 16 remaining RTX 5090 `/shapes/` files using `ckt shapes bench` / `tools/gb10_shape_sweep.py` with a 32 MiB L2 flush.

#### Verification & Acceptance Criteria
```bash
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt shapes list
ninja -C build-native -j "$(nproc)" infernix-serve infernix_linear_bench infernix_linear_bf16_a16_test infernix_linear_fp8_a16_test infernix_linear_mxfp8_a16_test infernix_linear_nvfp4_a16_test infernix_linear_q4_a16_test infernix_linear_q6_a16_test
ctest --test-dir build-native -R "infernix_linear_|infernix_cuda_stack_check" --output-on-failure
```
- **Acceptance Criteria:** All linear oracle tests and `infernix_cuda_stack_check` pass; `ckt shapes list` shows `0` untuned shapes; measurable latency reductions on swept shapes in `infernix_linear_bench`.

---

### Task T6 (Priority P1): Eliminate Unused Workspace Allocations, Right-Size `kContextQueryMaxSplit = 85`, Remove Redundant SMEM Zeroing/Barriers, and Vectorize `columns_permute_kernel` in `softmax_attention`

#### Affected Files & Symbols
- [`src/ops/softmax_attention/dense/context/context_softmax_attention.cpp`](file:///opt/etc/Infernix/src/ops/softmax_attention/dense/context/context_softmax_attention.cpp#L170-L174)
- [`src/ops/softmax_attention/dense/causal_cache/causal_softmax_attention.cpp`](file:///opt/etc/Infernix/src/ops/softmax_attention/dense/causal_cache/causal_softmax_attention.cpp)
  - [`stage_columns`](file:///opt/etc/Infernix/src/ops/softmax_attention/dense/causal_cache/causal_softmax_attention.cpp#L279-L290)
  - [`causal_softmax_attention_workspace_capacity_bytes`](file:///opt/etc/Infernix/src/ops/softmax_attention/dense/causal_cache/causal_softmax_attention.cpp#L420-L433)
  - [`causal_softmax_attention`](file:///opt/etc/Infernix/src/ops/softmax_attention/dense/causal_cache/causal_softmax_attention.cpp#L485-L534)
- [`src/ops/softmax_attention/common/context_query.cuh`](file:///opt/etc/Infernix/src/ops/softmax_attention/common/context_query.cuh#L20)
- [`src/ops/softmax_attention/sliding_window/kernel.cuh`](file:///opt/etc/Infernix/src/ops/softmax_attention/sliding_window/kernel.cuh#L75)
- [`src/ops/softmax_attention/dense/causal_cache/column_invariant.cu`](file:///opt/etc/Infernix/src/ops/softmax_attention/dense/causal_cache/column_invariant.cu#L31-L58)
- [`src/ops/softmax_attention/dense/causal_cache/fp8/grouped_mma.cuh`](file:///opt/etc/Infernix/src/ops/softmax_attention/dense/causal_cache/fp8/grouped_mma.cuh#L189-L219)
- [`src/ops/softmax_attention/dense/causal_cache/int8/mma.cuh`](file:///opt/etc/Infernix/src/ops/softmax_attention/dense/causal_cache/int8/mma.cuh#L156-L185)
- [`src/ops/softmax_attention/dense/causal_cache/int8/grouped_pipelined.cuh`](file:///opt/etc/Infernix/src/ops/softmax_attention/dense/causal_cache/int8/grouped_pipelined.cuh#L239-L283)

#### Root Cause & Hardware Rationale
1. **Unconditional Partial Workspace Allocation in [`context_softmax_attention.cpp`](file:///opt/etc/Infernix/src/ops/softmax_attention/dense/context/context_softmax_attention.cpp#L170-L174):**
   - Allocates `partial.acc`, `partial.m`, and `partial.l` from `workspace` even when `plan.route == ContextAttentionRoute::Direct` (where `plan.split_capacity == 1` and partials are unused).
2. **Unused `staged.k` and `staged.v` Allocations in [`causal_softmax_attention.cpp`](file:///opt/etc/Infernix/src/ops/softmax_attention/dense/causal_cache/causal_softmax_attention.cpp#L279-L290):**
   - When `verify_rows_attention(cache.storage)` is `true` (all non-VQ formats: BF16, INT8, FP8, NVFP4, K8V4), `k` and `v` are appended in-place from the input tensors (`lines 498-510`), so `staged.k` and `staged.v` allocated by `stage_columns` (`lines 282-283`) are completely untouched.
3. **170-SM `kContextQueryMaxSplit = 85` in [`context_query.cuh`](file:///opt/etc/Infernix/src/ops/softmax_attention/common/context_query.cuh#L20) & [`sliding_window/kernel.cuh`](file:///opt/etc/Infernix/src/ops/softmax_attention/sliding_window/kernel.cuh#L75):**
   - `kContextQueryMaxSplit = 85` sizes shared memory `weights[WarpsPerBlock][MaxSplits]` in `context_query_reduce_body`, even though `context` caps `split_limit` at `64` and `sliding_window` caps `kSlidingWindowMaxSplits` at `32`.
4. **4-Byte Word Copies in [`columns_permute_kernel`](file:///opt/etc/Infernix/src/ops/softmax_attention/dense/causal_cache/column_invariant.cu#L31-L58):**
   - Copies `q`, `k`, `v`, and `out` (`row_bytes >= 1024`, 16-byte aligned) in 4-byte `uint32_t` words instead of 16-byte `uint4` vectors.
5. **Redundant Shared Memory Zeroing & Back-to-Back `__syncthreads()`:**
   - [`fp8/grouped_mma.cuh:189-191`](file:///opt/etc/Infernix/src/ops/softmax_attention/dense/causal_cache/fp8/grouped_mma.cuh#L189-L191), [`int8/mma.cuh:156-158`](file:///opt/etc/Infernix/src/ops/softmax_attention/dense/causal_cache/int8/mma.cuh#L156-L158), and [`int8/grouped_pipelined.cuh:244-246`](file:///opt/etc/Infernix/src/ops/softmax_attention/dense/causal_cache/int8/grouped_pipelined.cuh#L244-L246) zero `q_s` (`Br * D` bytes) and `q_scale_tmp` and call `__syncthreads()` right before overwriting all `0..RowCount-1` rows; `grouped_pipelined.cuh:239-242` also executes two consecutive `__syncthreads()` with no shared memory accesses in between.

#### Required Implementation Steps
1. Guard `allocate_workspace` in [`context_softmax_attention.cpp`](file:///opt/etc/Infernix/src/ops/softmax_attention/dense/context/context_softmax_attention.cpp#L171) on `plan.route == detail::ContextAttentionRoute::Split` (and return `0` in `context_softmax_attention_workspace_capacity_bytes` when `plan.route == Direct`).
2. Pass `bool stage_kv = !verify_rows_attention(cache.storage)` to [`stage_columns`](file:///opt/etc/Infernix/src/ops/softmax_attention/dense/causal_cache/causal_softmax_attention.cpp#L279-L290) so `staged.k` and `staged.v` are not allocated when `verify_rows_attention` is `true`.
3. Lower `kContextQueryMaxSplit` from `85` to `64` in [`context_query.cuh`](file:///opt/etc/Infernix/src/ops/softmax_attention/common/context_query.cuh#L20) and pass `kSlidingWindowMaxSplits` (`32`) instead of `kContextQueryMaxSplit` (`85`) in [`sliding_window/kernel.cuh`](file:///opt/etc/Infernix/src/ops/softmax_attention/sliding_window/kernel.cuh#L75).
4. Vectorize [`columns_permute_kernel`](file:///opt/etc/Infernix/src/ops/softmax_attention/dense/causal_cache/column_invariant.cu#L31-L58) to use `uint4` (16-byte) loads/stores whenever `tensor.row_bytes % 16 == 0` (keeping `uint32_t` for 4-byte `positions`).
5. Remove the redundant `q_s` / `q_scale_tmp` zeroing loops and redundant barriers in `fp8/grouped_mma.cuh`, `int8/mma.cuh`, and `int8/grouped_pipelined.cuh` (only zero padding rows `RowCount..Br-1` if read by `ldmatrix`).

#### Verification & Acceptance Criteria
```bash
find src/ops/softmax_attention -name '*.cu' -exec touch {} +
ninja -C build-native -j "$(nproc)" infernix-serve infernix_ops_attention_test
ctest --test-dir build-native -R "attention|infernix_cuda_stack_check" --output-on-failure
```
- **Acceptance Criteria:** All attention oracle tests and `infernix_cuda_stack_check` pass; workspace bytes drop for `Direct` context attention and `verify_rows_attention` causal attention.

---

### Task T7 (Priority P1): Precompute `args.inverse` for Non-YaRN (`31x` Fewer GPU `powf` Calls), Vectorize Stride-32 Scalar Accesses, Guard `1.05 MiB` `qsa_select` Global Zeroing, and Clean Up `qsa_prompt`

#### Affected Files & Symbols
- [`src/ops/qsa/qsa_prepare.cu`](file:///opt/etc/Infernix/src/ops/qsa/qsa_prepare.cu)
  - [`PrepareArgs`](file:///opt/etc/Infernix/src/ops/qsa/qsa_prepare.cu#L56-L96)
  - [`attention_sincos`, `index_sincos`](file:///opt/etc/Infernix/src/ops/qsa/qsa_prepare.cu#L112-L129)
  - [`load_row`, `store_row`, `store_int8_row`, `store_fp8_row`](file:///opt/etc/Infernix/src/ops/qsa/qsa_prepare.cu#L139-L203)
  - [`qsa_prepare`](file:///opt/etc/Infernix/src/ops/qsa/qsa_prepare.cu#L597-L604)
- [`src/ops/qsa/qsa_select.cu`](file:///opt/etc/Infernix/src/ops/qsa/qsa_select.cu)
  - [`qsa_score_kernel`, `qsa_score_wide_kernel`](file:///opt/etc/Infernix/src/ops/qsa/qsa_select.cu#L240-L447)
  - [`qsa_select`](file:///opt/etc/Infernix/src/ops/qsa/qsa_select.cu#L923-L968)
- [`src/ops/qsa/qsa_prompt.cu`](file:///opt/etc/Infernix/src/ops/qsa/qsa_prompt.cu)
  - [`Stage<Storage>::SplitGather`](file:///opt/etc/Infernix/src/ops/qsa/qsa_prompt.cu#L143-L145)
  - [`qsa_prompt_kernel` query load/store](file:///opt/etc/Infernix/src/ops/qsa/qsa_prompt.cu#L326-L347)
  - [`qsa_prompt_split`, `qsa_prompt_force`, `qsa_prompt_forced`](file:///opt/etc/Infernix/src/ops/qsa/qsa_prompt.cu#L1092-L1111)

#### Root Cause & Hardware Rationale
1. **31x Per-Column GPU `powf` Recomputation in [`qsa_prepare.cu`](file:///opt/etc/Infernix/src/ops/qsa/qsa_prepare.cu#L112-L129):**
   - `PrepareArgs` already contains `double inverse[32]` (`256 B`) in `__grid_constant__` parameter memory, but `qsa_prepare` (`lines 597-604`) only populates `args.inverse` when `args.yarn` is `true`. When `args.yarn` is `false` (the default on Qwen4Exp), every warp calls `powf(a.theta, ...)` inside `attention_sincos` and `index_sincos` (`24` query warps + `4` index-query warps + `2` KV warps + `1` pool warp = **31 warps per column**).
2. **Strided Scalar 2-Byte/1-Byte Global Loads/Stores in `qsa_prepare.cu` & `qsa_prompt.cu`:**
   - `load_row` (`line 141`) and `store_row` (`line 147`) issue 8 strided `bf16` loads/stores (`lane + 32 * j`, stride 64 bytes) instead of loading `uint4` (`8` contiguous `bf16` values at `8 * lane`) and transposing in shared memory or warp shuffles. Similarly, `store_fp8_row` (`line 199`) writes 8 strided single bytes (`lane + 32 * r`) to global memory.
3. **Unconditional `1.05 MiB` Global Zeroing in [`qsa_select.cu`](file:///opt/etc/Infernix/src/ops/qsa/qsa_select.cu#L923-L960):**
   - `clear_words = l.group * (1 + kScoreBins)` (`128 * 2049 = 262,272` words = `1.05 MiB`) is passed to `qsa_score_kernel` / `qsa_score_wide_kernel`, which zero all `262,272` words even when `n < l.group` (e.g. at `T = 1`, `n = 1` only uses `2,049` words = `8 KiB`, so **99.2% of the 1.05 MiB global write is wasted**!) or when every column is in the dense regime (`positions[t] + 1 <= budget`, where `histograms` and `arrivals` are never touched at all).
4. **Dead `SplitGather` Shared-Memory Over-Allocation in [`qsa_prompt.cu`](file:///opt/etc/Infernix/src/ops/qsa/qsa_prompt.cu#L143-L145):**
   - `qsa_prompt.cu:1030` states the split stage was removed, yet `Stage<Storage>::SplitGather` still sets `ScaleSlots = 2` for `Int8Group64` and `Fp8E4M3Row256`, wasting shared memory in every stage.

#### Required Implementation Steps
1. In [`qsa_prepare.cu`](file:///opt/etc/Infernix/src/ops/qsa/qsa_prepare.cu#L597-L604), precompute the 32 RoPE frequencies on the host (or via a `static` cached table keyed by `(theta, yarn_factor, original_positions)`) so `attention_sincos` and `index_sincos` always read `a.inverse[pair]` without calling `powf` on the GPU.
2. In [`qsa_select.cu`](file:///opt/etc/Infernix/src/ops/qsa/qsa_select.cu#L941-L960), set `const int pass_clear_words = n * (1 + kScoreBins);` per group pass (reducing global zeroing at `T = 1` from `1,049 KiB` to `8.2 KiB`, a **128x reduction**!), and inside `qsa_score_kernel` / `qsa_score_wide_kernel`, skip zeroing a column's histogram when `(positions[begin + c] + 1) / ratio <= top_blocks`.
3. In [`qsa_prompt.cu`](file:///opt/etc/Infernix/src/ops/qsa/qsa_prompt.cu#L143-L145), remove the dead `SplitGather` double scale slot (`ScaleSlots = 1`) and clean up unused `kGatherKey` / `kGatherValue` branches.
4. Vectorize `load_row` / `store_row` / `store_fp8_row` in `qsa_prepare.cu` and query loading in `qsa_prompt.cu`.

#### Verification & Acceptance Criteria
```bash
ninja -C build-native -j "$(nproc)" infernix-serve infernix_qsa_test infernix_qsa_attention_bench
ctest --test-dir build-native -R "infernix_qsa_test|infernix_cuda_stack_check" --output-on-failure
./build-native/bench/infernix_qsa_attention_bench --kv-dtype nvfp4,int8,bf16 --context 2048,8192,32768 --width 1,8,16,64
```
- **Acceptance Criteria:** `infernix_qsa_test` (including exactness/invariance checks) and `infernix_cuda_stack_check` pass 100%; `qsa_select` latency at `width = 1..64` improves significantly.

---

### Task T8 (Priority P1): Vectorize Stride-32 Global Stores & Hoist Host `std::pow` Table in `context_kv_materialize`, and Retune `vq_append_split` in `kv_cache`

#### Affected Files & Symbols
- [`src/ops/context_kv_materialize/kernel.cuh`](file:///opt/etc/Infernix/src/ops/context_kv_materialize/kernel.cuh#L294-L361)
- [`src/ops/context_kv_materialize/materialize.cu`](file:///opt/etc/Infernix/src/ops/context_kv_materialize/materialize.cu#L371-L396)
- [`src/ops/kv_cache/append/vq2_launch.cu`](file:///opt/etc/Infernix/src/ops/kv_cache/append/vq2_launch.cu#L49-L69)

#### Root Cause & Hardware Rationale
1. **Stride-32 Scalar Global Stores in [`context_kv_mma_kernel`](file:///opt/etc/Infernix/src/ops/context_kv_materialize/kernel.cuh#L294-L361):**
   - Writes `k_pages` and `v_pages` via scalar 2-byte (`__nv_bfloat16`, `__half`) and 1-byte (`int8_t`, `uint8_t`) stores with stride 32 across lanes (`d = lane + 32 * j`), whereas `context_kv_grouped_kernel` (`lines 421-462`) uses vectorized 16-byte `int4` / 8-byte `uint2` stores.
2. **Per-Call Host `std::pow` Loop & Uncached `cudaFuncSetAttribute` in [`materialize.cu`](file:///opt/etc/Infernix/src/ops/context_kv_materialize/materialize.cu#L371-L396):**
   - Recomputes 64 `std::pow(10000.0, ...)` values on the CPU on every launch and calls `cudaFuncSetAttribute` without a `static` guard whenever `bytes > 48 * 1024`.
3. **RTX 5090 Row-Split Table in [`vq_append_split`](file:///opt/etc/Infernix/src/ops/kv_cache/append/vq2_launch.cu#L49-L69):**
   - Hardcodes RTX 5090 row bands (`rows >= 512 && rows <= 8192`) calibrated on 170 SMs (`128..2048` rows -> `256..4096` CTAs).

#### Required Implementation Steps
1. Stage or shuffle the quantized/converted K and V rows in [`context_kv_mma_kernel`](file:///opt/etc/Infernix/src/ops/context_kv_materialize/kernel.cuh#L294-L361) so global writes to `k_pages` and `v_pages` emit coalesced vectorized stores.
2. Make `PrepareInverse` in [`materialize.cu`](file:///opt/etc/Infernix/src/ops/context_kv_materialize/materialize.cu#L371-L377) `static const` and cache the `cudaFuncSetAttribute` call at [`lines 393-396`](file:///opt/etc/Infernix/src/ops/context_kv_materialize/materialize.cu#L393-L396) per kernel template instantiation.
3. Retune [`vq_append_split`](file:///opt/etc/Infernix/src/ops/kv_cache/append/vq2_launch.cu#L49-L69) on 48-SM GB10 using `infernix_kv_cache_append_bench`.

#### Verification & Acceptance Criteria
```bash
find src/ops/context_kv_materialize src/ops/kv_cache -name '*.cu' -exec touch {} +
ninja -C build-native -j "$(nproc)" infernix-serve infernix_ops_attention_test infernix_context_kv_materialize_bench
ctest --test-dir build-native -R "context_kv_materialize|kv_cache|infernix_cuda_stack_check" --output-on-failure
```
- **Acceptance Criteria:** All oracle tests and `infernix_cuda_stack_check` pass; measurable latency reduction in `infernix_context_kv_materialize_bench`.

---

### Task T9 (Priority P1): Fuse Double `codec_pack_kernel` (`hadamard_i128`) Launch, Right-Size `kTargetCtas = 384`, and Eliminate Unpinned Sync Transfers in `lossless`

#### Affected Files & Symbols
- [`src/ops/lossless/lossless_attention.cu`](file:///opt/etc/Infernix/src/ops/lossless/lossless_attention.cu)
  - [`kTargetCtas`](file:///opt/etc/Infernix/src/ops/lossless/lossless_attention.cu#L69), [`choose_splits`](file:///opt/etc/Infernix/src/ops/lossless/lossless_attention.cu#L1845-L1861), [`line 2090`](file:///opt/etc/Infernix/src/ops/lossless/lossless_attention.cu#L2090)
  - [`codec_pack_kernel`](file:///opt/etc/Infernix/src/ops/lossless/lossless_attention.cu#L1500-L1592), launch sites at [`lines 2035-2038`](file:///opt/etc/Infernix/src/ops/lossless/lossless_attention.cu#L2035-L2038)
- [`src/ops/lossless/lossless_linear.cu`](file:///opt/etc/Infernix/src/ops/lossless/lossless_linear.cu#L158-L181)
- [`src/ops/offloaded_sparse_moe/cuda/lossless_expert.cuh`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/lossless_expert.cuh#L193-L221)

#### Root Cause & Hardware Rationale
1. **Redundant Double-Decode + Double 128-Bit Hadamard Transform in [`lossless_attention.cu`](file:///opt/etc/Infernix/src/ops/lossless/lossless_attention.cu#L1500-L1592):**
   - `codec_pack_kernel` is launched twice (`mode = 0` at line 2035 and `mode = 1` at line 2037). In `mode = 0`, every thread decodes `vplane` from global memory, executes the 8-stage 128-bit Hadamard transform (`hadamard_i128`), writes only `v_max_exp[kv_h * seq_len + s]`, and discards the transformed values. In `mode = 1`, every thread **re-reads and re-decodes `vplane` and recomputes the entire 128-bit Hadamard transform** to pack the mantissas!
   - Because each CTA processes `(s, kv_h)` independently when writing `v_max_exp`, `mode = 0` can write the transformed `I128` (or exact exponent/mantissa intermediate) or compute `v_max_exp` across `s` in a single pass when `seq_len` fits CTA reductions, eliminating the duplicate decode and 128-bit Hadamard transform.
2. **`kTargetCtas = 384` Sized for 170-SM RTX 5090 ([`lossless_attention.cu:69`](file:///opt/etc/Infernix/src/ops/lossless/lossless_attention.cu#L69)):**
   - Launches 8 waves on 48-SM GB10, inflating split-KV partial workspace and `combine_kernel` overhead.
3. **Unpinned Heap `cudaMemcpyAsync` + `cudaStreamSynchronize` in [`LosslessMoE::forward`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/lossless_expert.cuh#L193-L221):**
   - Allocates `std::vector<int32_t> counts` and `plan` on the heap, runs `cudaMemcpyAsync` to/from unpinned heap memory, and calls `cudaStreamSynchronize(stream)` on every layer.

#### Required Implementation Steps
1. Eliminate the duplicate `vplane` decode + `hadamard_i128` computation in [`codec_pack_kernel`](file:///opt/etc/Infernix/src/ops/lossless/lossless_attention.cu#L1500-L1592) while preserving 100% bit-exact lossless outputs.
2. Right-size `kTargetCtas` in [`lossless_attention.cu`](file:///opt/etc/Infernix/src/ops/lossless/lossless_attention.cu#L69) to `2 * device_sm_count()` (`96` on GB10).
3. Compute the expert dispatch plan on the device in [`lossless_expert.cuh`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/lossless_expert.cuh#L193-L221) (or use pre-allocated pinned/device workspace) and replace `std::vector` in [`lossless_linear.cu`](file:///opt/etc/Infernix/src/ops/lossless/lossless_linear.cu#L158-L181) with a fixed `std::array<..., 4>`.

#### Verification & Acceptance Criteria
```bash
ninja -C build-native -j "$(nproc)" infernix-serve infernix_lossless_test
ctest --test-dir build-native -R "lossless|infernix_cuda_stack_check" --output-on-failure
```
- **Acceptance Criteria:** 100% bit-exact match on all `lossless` oracle tests and `infernix_cuda_stack_check`.

---

### Task T10 (Priority P2): Right-Size `kMmaTileTarget` & `kWidePlaneBytes`, Deduplicate Encoders, and Remove Per-Call `std::vector` Allocations in `ggml`

#### Affected Files & Symbols
- [`src/ops/ggml/ggml_linear.cu`](file:///opt/etc/Infernix/src/ops/ggml/ggml_linear.cu#L28), [`lines 46-136`](file:///opt/etc/Infernix/src/ops/ggml/ggml_linear.cu#L46-L136), [`lines 470-575`](file:///opt/etc/Infernix/src/ops/ggml/ggml_linear.cu#L470-L575)
- [`src/ops/ggml/ggml_linear_a8.h`](file:///opt/etc/Infernix/src/ops/ggml/ggml_linear_a8.h#L20)
- [`src/ops/ggml/ggml_linear_a8.cu`](file:///opt/etc/Infernix/src/ops/ggml/ggml_linear_a8.cu#L290)
- [`src/ops/ggml/ggml_linear_a8_wide.cu`](file:///opt/etc/Infernix/src/ops/ggml/ggml_linear_a8_wide.cu#L267-L292)

#### Root Cause & Hardware Rationale
1. `kWidePlaneBytes = 32 MiB` ([`ggml_linear.cu:28`](file:///opt/etc/Infernix/src/ops/ggml/ggml_linear.cu#L28)) and `kMmaTileTarget = 512` ([`ggml_linear.cu:527`](file:///opt/etc/Infernix/src/ops/ggml/ggml_linear.cu#L527)) are sized for RTX 5090 (96 MiB L2, 170 SMs).
2. `partition_slabs` ([`ggml_linear.cu:470-493`](file:///opt/etc/Infernix/src/ops/ggml/ggml_linear.cu#L470-L493)) and `ggml_linear_a8_wide.cu:267-292` allocate `std::vector<std::vector<Slab>>` and `std::vector<bool> taken` on the heap on every call.
3. [`ggml_linear.cu:46-136`](file:///opt/etc/Infernix/src/ops/ggml/ggml_linear.cu#L46-L136) duplicates `quantize_row_q8_1_f16` and `ggml_q8_1_encode_kernel` from [`ggml_encode.cuh`](file:///opt/etc/Infernix/src/ops/ggml/ggml_encode.cuh#L51-L128).

#### Required Implementation Steps
1. Scale `kMmaTileTarget` and `kWidePlaneBytes` to GB10's 48 SMs and 24 MiB L2.
2. Replace `std::vector<std::vector<Slab>>` and `std::vector<bool>` in `partition_slabs` and `ggml_linear_a8_wide.cu` with stack-allocated fixed-capacity arrays (`kMaxSegments = 16`).
3. Include `ggml_encode.cuh` in `ggml_linear.cu` instead of duplicating the Q8_1 encoder kernels.

#### Verification & Acceptance Criteria
```bash
ninja -C build-native -j "$(nproc)" infernix-serve infernix_ggml_linear_test
ctest --test-dir build-native -R "ggml|infernix_cuda_stack_check" --output-on-failure
```
- **Acceptance Criteria:** All `ggml` oracle tests and `infernix_cuda_stack_check` pass with zero heap allocations on the `ggml_linear` dispatch path.

---

### Task T11 (Priority P1): Retune 170-SM & 96-MiB-L2 Launch Heuristics in `rmsnorm_rope`, `launcher/*`, and `common/bf16_vector.cuh` for 48-SM GB10

#### Affected Files & Symbols
- [`src/ops/rmsnorm_rope/launch.cu`](file:///opt/etc/Infernix/src/ops/rmsnorm_rope/launch.cu#L26-L36), [`lines 70-126`](file:///opt/etc/Infernix/src/ops/rmsnorm_rope/launch.cu#L70-L126)
- [`src/ops/launcher/rope.cu`](file:///opt/etc/Infernix/src/ops/launcher/rope.cu#L57-L68), [`lines 98-105`](file:///opt/etc/Infernix/src/ops/launcher/rope.cu#L98-L105)
- [`src/ops/launcher/embed_gather.cu`](file:///opt/etc/Infernix/src/ops/launcher/embed_gather.cu#L77), [`lines 145-166`](file:///opt/etc/Infernix/src/ops/launcher/embed_gather.cu#L145-L166)
- [`src/ops/launcher/argmax.cu`](file:///opt/etc/Infernix/src/ops/launcher/argmax.cu#L18-L34)
- [`src/ops/launcher/vision_pos_embed.cu`](file:///opt/etc/Infernix/src/ops/launcher/vision_pos_embed.cu#L15), [`lines 29-31`](file:///opt/etc/Infernix/src/ops/launcher/vision_pos_embed.cu#L29-L31)
- [`src/ops/launcher/gelu.cu`](file:///opt/etc/Infernix/src/ops/launcher/gelu.cu#L18-L19)
- [`src/ops/common/bf16_vector.cuh`](file:///opt/etc/Infernix/src/ops/common/bf16_vector.cuh#L22-L25)

#### Root Cause & Hardware Rationale
1. **[`rmsnorm_rope/launch.cu`](file:///opt/etc/Infernix/src/ops/rmsnorm_rope/launch.cu#L26-L36):** `kTextHeadsPerBlock = 3` launches only `6 * T` CTAs (`6` CTAs at `T = 1` across 48 SMs = `12.5%` SM utilization). Using 1 head per warp/CTA at small `T` (`tokens <= 8`) increases CTA count to `18 * T` (`18` CTAs at `T=1`, `36` at `T=2`).
2. **[`launcher/rope.cu`](file:///opt/etc/Infernix/src/ops/launcher/rope.cu#L57-L68):** `launch_fixed_text` packs all `QHeads + KHeads` warps into 1 CTA per token for `tokens <= 6` (`1` CTA at `T=1`), and `launch_fixed_pair` (`line 99`) uses `tokens <= 400` (`2,000` CTAs = 170-SM wave threshold).
3. **[`launcher/embed_gather.cu`](file:///opt/etc/Infernix/src/ops/launcher/embed_gather.cu#L77):** Switches `Q8EmbedRoute` from `Grouped` (`64 * T` CTAs) to `Row` (`T` CTAs) at `T > 6`, leaving up to 85% of GB10's 48 SMs idle at `T = 7..47`, and uses 170-SM thresholds `T <= 176` / `T <= 128` (`lines 145-166`).
4. **[`launcher/argmax.cu`](file:///opt/etc/Infernix/src/ops/launcher/argmax.cu#L18-L34):** `kArgmaxBlock = 512` (`8,192` rows/tile) launches only `31` CTAs for 248,077 vocab rows at `T=1`, leaving 17 of 48 SMs idle (`kArgmaxBlock = 256` with `4,096` rows/tile launches `61` CTAs, filling all 48 SMs).
5. **[`launcher/vision_pos_embed.cu`](file:///opt/etc/Infernix/src/ops/launcher/vision_pos_embed.cu#L15):** `kSmallPTargetBlocks = 4096` forces `warps_per_block = 1` for `patches < 820`, causing 18x redundant global reloads of patch indices/weights per CTA.
6. **[`common/bf16_vector.cuh`](file:///opt/etc/Infernix/src/ops/common/bf16_vector.cuh#L22-L25) & [`launcher/gelu.cu`](file:///opt/etc/Infernix/src/ops/launcher/gelu.cu#L18-L19):** `kBf16x8CacheSizedMaxElements = 32M` (`64 MiB` per tensor = `128 MiB` R+W) is 5.3x larger than GB10's 24 MiB L2; clamping to `4LL * 1024LL * 1024LL` (`8 MiB` per tensor = `16 MiB` R+W) prevents streaming elementwise kernels from thrashing L2.

#### Required Implementation Steps
1. Add a 1-head-per-CTA small-T specialization (`tokens <= 8`) in [`rmsnorm_rope/launch.cu`](file:///opt/etc/Infernix/src/ops/rmsnorm_rope/launch.cu#L70-L126) and fix [`launcher/rope.cu`](file:///opt/etc/Infernix/src/ops/launcher/rope.cu#L57-L68) so `T = 1..6` launches across all heads rather than 1 CTA per token.
2. Extend `Grouped` route in [`launcher/embed_gather.cu`](file:///opt/etc/Infernix/src/ops/launcher/embed_gather.cu#L77) up to `T <= 48` (or use a 4–8 CTAs/token split for `T = 7..47`) and adjust `T <= 176` / `T <= 128` to 48-SM wave multiples (`96` / `144`).
3. Tune `kArgmaxBlock` / rows-per-CTA in [`launcher/argmax.cu`](file:///opt/etc/Infernix/src/ops/launcher/argmax.cu#L18-L34) for 48 SMs.
4. Lower `kSmallPTargetBlocks` in [`launcher/vision_pos_embed.cu`](file:///opt/etc/Infernix/src/ops/launcher/vision_pos_embed.cu#L15) to `48 * 8 = 384` (or keep `warps_per_block >= 4`).
5. Lower `kBf16x8CacheSizedMaxElements` in [`bf16_vector.cuh`](file:///opt/etc/Infernix/src/ops/common/bf16_vector.cuh#L22-L25) and `kCacheSizedMaxElements` in [`gelu.cu`](file:///opt/etc/Infernix/src/ops/launcher/gelu.cu#L18-L19) to `4LL * 1024LL * 1024LL` (`8 MiB` per tensor).

#### Verification & Acceptance Criteria
```bash
find src/ops/rmsnorm_rope src/ops/launcher -name '*.cu' -exec touch {} +
ninja -C build-native -j "$(nproc)" infernix-serve infernix_rmsnorm_rope_test infernix_ops_basic_test
ctest --test-dir build-native -R "rmsnorm_rope|basic|infernix_cuda_stack_check" --output-on-failure
```
- **Acceptance Criteria:** All basic/rope oracle tests and `infernix_cuda_stack_check` pass; SM utilization at `T = 1..47` reaches `>= 75–100%` of GB10's 48 SMs.

---

### Task T12 (Priority P1): Vectorize `SwiGluRowMajorMmaEpilogue::finish_tile` via SMEM Staging, Prune Dead `q8_linear_swiglu` Kernels, and Vectorize `int8` Unpacking in `projection_fp32`

#### Affected Files & Symbols
- [`src/ops/linear_swiglu/row_major_mma_epilogue.cuh`](file:///opt/etc/Infernix/src/ops/linear_swiglu/row_major_mma_epilogue.cuh#L75-L96)
- [`src/ops/linear_swiglu/q8/q8_linear_swiglu_gemm_mma.cu`](file:///opt/etc/Infernix/src/ops/linear_swiglu/q8/q8_linear_swiglu_gemm_mma.cu#L22-L98)
- [`src/ops/linear_swiglu/q8/q8_linear_swiglu_plan.cpp`](file:///opt/etc/Infernix/src/ops/linear_swiglu/q8/q8_linear_swiglu_plan.cpp#L45-L50)
- [`src/ops/projection_fp32/projection_fp32.cu`](file:///opt/etc/Infernix/src/ops/projection_fp32/projection_fp32.cu#L230-L310)

#### Root Cause & Hardware Rationale
1. **Uncoalesced Strided Scalar BF16 Stores in [`SwiGluRowMajorMmaEpilogue::finish_tile`](file:///opt/etc/Infernix/src/ops/linear_swiglu/row_major_mma_epilogue.cuh#L75-L96):**
   - Each lane computes `swiglu(gate0, up0)` for column `col0` and `swiglu(gate1, up1)` for `col0 + 8`, and writes them directly to global memory `out[col0 * rows + out_row]` and `out[(col0 + 8) * rows + out_row]`. Because adjacent lanes `lid = 0..3` hold adjacent columns (`col0 = ... + lid`), their writes are strided by `rows * sizeof(bf16)` (`1,280` to `34,816` bytes apart)! Staging the tile through shared memory (`[BlockTokens][BlockRows / 2]`) turns the epilogue into coalesced 16-byte `uint4` writes along `out_row`.
2. **Dead Kernel Instantiations in [`q8_linear_swiglu_gemm_mma.cu`](file:///opt/etc/Infernix/src/ops/linear_swiglu/q8/q8_linear_swiglu_gemm_mma.cu#L22-L98):**
   - `q8_linear_swiglu_mma_r32_c32_launch`, `r32_c48`, `r32_c80`, `r32_c96`, `r32_c128`, `r64_c64`, `r64_c96`, and `dflash2_mma_r64_c64_k128` are compiled into `libinfernix_ops.a` and linked by `nvlink` but never selected by `kCompanionRoutes`, `kDFlash2Routes`, or `kFlashNextSharedRoutes` in [`q8_linear_swiglu_plan.cpp`](file:///opt/etc/Infernix/src/ops/linear_swiglu/q8/q8_linear_swiglu_plan.cpp#L22-L50).
3. **Scalar Bit-Shift Unpacking of `int8` Codes in [`projection_fp32.cu`](file:///opt/etc/Infernix/src/ops/projection_fp32/projection_fp32.cu#L230-L310):**
   - `projection_q8_kernel` and `TallQ8` unpack 16 `int8` bytes from `uint4` using scalar shifts (`static_cast<int8_t>(word >> (8 * b))`) and `int2float` in nested loops instead of vectorized `__byte_perm` / FP16 widening.

#### Required Implementation Steps
1. Stage the SwiGLU output tile in shared memory inside [`SwiGluRowMajorMmaEpilogue::finish_tile`](file:///opt/etc/Infernix/src/ops/linear_swiglu/row_major_mma_epilogue.cuh#L75-L96) (reusing the mainloop activation/weight SMEM after the K-loop barrier) so warps emit coalesced vectorized writes along `out_row`.
2. Remove unselected kernel instantiations in [`q8_linear_swiglu_gemm_mma.cu`](file:///opt/etc/Infernix/src/ops/linear_swiglu/q8/q8_linear_swiglu_gemm_mma.cu#L22-L98) (after verifying which schedules win on `kFlashNextSharedRoutes` at `T = 41..512`).
3. Use vectorized byte unpacking in [`projection_fp32.cu`](file:///opt/etc/Infernix/src/ops/projection_fp32/projection_fp32.cu#L230-L310) while preserving exact FP32 butterfly reduction order.

#### Verification & Acceptance Criteria
```bash
find src/ops/linear_swiglu src/ops/projection_fp32 -name '*.cu' -exec touch {} +
ninja -C build-native -j "$(nproc)" infernix-serve infernix_linear_swiglu_test infernix_projection_fp32_test infernix_q8_linear_swiglu_bench
ctest --test-dir build-native -R "linear_swiglu|projection_fp32|infernix_cuda_stack_check" --output-on-failure
```
- **Acceptance Criteria:** 100% pass on `infernix_linear_swiglu_test`, `infernix_projection_fp32_test`, and `infernix_cuda_stack_check`; measurable speedup in `infernix_q8_linear_swiglu_bench`.

---

### Task T13 (Priority P1): Cache All Hot-Path CUDA Runtime/Driver Queries and Eliminate Per-Call `std::vector` / `std::string` Heap Allocations in `wrapper`, `weight_input.cpp`, and `linear_attention`

#### Affected Files & Symbols
- [`src/ops/linear_attention/gated_delta_net/chunked/recurrence.cu`](file:///opt/etc/Infernix/src/ops/linear_attention/gated_delta_net/chunked/recurrence.cu#L521-L522)
- [`src/ops/linear_attention/kimi_delta_attention/chunked/recurrence.cu`](file:///opt/etc/Infernix/src/ops/linear_attention/kimi_delta_attention/chunked/recurrence.cu#L324-L325)
- [`src/ops/linear_attention/gated_delta_net/replay.cpp`](file:///opt/etc/Infernix/src/ops/linear_attention/gated_delta_net/replay.cpp#L151-L156)
- [`src/ops/softmax_attention/dense/causal_cache/causal_softmax_attention.cpp`](file:///opt/etc/Infernix/src/ops/softmax_attention/dense/causal_cache/causal_softmax_attention.cpp#L371-L372)
- [`src/ops/sparse_moe/decode/sparse_moe_decode_plan.cpp`](file:///opt/etc/Infernix/src/ops/sparse_moe/decode/sparse_moe_decode_plan.cpp#L22)
- [`src/ops/linear/common/sliced_k_pipelined.cuh`](file:///opt/etc/Infernix/src/ops/linear/common/sliced_k_pipelined.cuh#L509-L539)
- [`src/ops/ggml/ggml_linear_a8.cuh`](file:///opt/etc/Infernix/src/ops/ggml/ggml_linear_a8.cuh#L456-L490) & [`ggml_linear_tiled.cu`](file:///opt/etc/Infernix/src/ops/ggml/ggml_linear_tiled.cu#L155-L188)
- [`src/ops/wrapper/sparse_moe.cpp`](file:///opt/etc/Infernix/src/ops/wrapper/sparse_moe.cpp#L38-L160), [`lines 244-248`](file:///opt/etc/Infernix/src/ops/wrapper/sparse_moe.cpp#L244-L248)
- [`src/ops/weight_input.cpp`](file:///opt/etc/Infernix/src/ops/weight_input.cpp#L17-L46), [`lines 140-324`](file:///opt/etc/Infernix/src/ops/weight_input.cpp#L140-L324)

#### Root Cause & Hardware Rationale
1. **Uncached `cudaFuncSetAttribute` on Every Chunked Recurrence Launch:**
   - [`gated_delta_net/chunked/recurrence.cu:521-522`](file:///opt/etc/Infernix/src/ops/linear_attention/gated_delta_net/chunked/recurrence.cu#L521-L522) (`36` calls/pass) and [`kimi_delta_attention/chunked/recurrence.cu:324-325`](file:///opt/etc/Infernix/src/ops/linear_attention/kimi_delta_attention/chunked/recurrence.cu#L324-L325) call `cudaFuncSetAttribute` on every launch without a `static` guard.
2. **Uncached `cudaGetDevice` / `cudaDeviceGetAttribute` & `thread_local std::vector` Scans:**
   - [`causal_softmax_attention_prompt_wave_tokens`](file:///opt/etc/Infernix/src/ops/softmax_attention/dense/causal_cache/causal_softmax_attention.cpp#L371-L372) queries `cudaGetDevice` + `cudaDeviceGetAttribute` without caching.
   - [`sliced_k_resident`](file:///opt/etc/Infernix/src/ops/linear/common/sliced_k_pipelined.cuh#L524-L539), [`fp8_a8_tma_wave`](file:///opt/etc/Infernix/src/ops/linear/fp8/fp8_a8_tma_mma.cuh#L55-L73), [`ggml_linear_a8.cuh`](file:///opt/etc/Infernix/src/ops/ggml/ggml_linear_a8.cuh#L456-L490), and [`ggml_linear_tiled.cu`](file:///opt/etc/Infernix/src/ops/ggml/ggml_linear_tiled.cu#L155-L188) linearly scan a `thread_local std::vector` on every launch instead of using a per-template `static const int` local variable!
3. **Hot-Path `std::vector` and `std::string` Allocations in `wrapper/sparse_moe.cpp` & `weight_input.cpp`:**
   - [`require_disjoint`](file:///opt/etc/Infernix/src/ops/wrapper/sparse_moe.cpp#L38-L58) allocates a `std::vector<AddressRange>` and [`validate_common`](file:///opt/etc/Infernix/src/ops/wrapper/sparse_moe.cpp#L79-L160) constructs ~13 `std::string(op_name) + ...` objects on every MoE layer call (`48` layers/step), plus resolves the plan twice (`lines 244-248`).
   - [`weight_input.cpp`](file:///opt/etc/Infernix/src/ops/weight_input.cpp#L17-L46) allocates thousands of temporary `std::vector<uint64_t>` objects via `matrix()` and calls `concatenate_rows(inputs)` twice when `contiguous(...)` is true (`lines 39-40, 145-146`).

#### Required Implementation Steps
1. Wrap `cudaFuncSetAttribute` in `static const cudaError_t` inside [`gated_delta_net/chunked/recurrence.cu`](file:///opt/etc/Infernix/src/ops/linear_attention/gated_delta_net/chunked/recurrence.cu#L521-L522) and [`kimi_delta_attention/chunked/recurrence.cu`](file:///opt/etc/Infernix/src/ops/linear_attention/kimi_delta_attention/chunked/recurrence.cu#L324-L325).
2. Replace `thread_local std::vector<Entry>` occupancy caches in `sliced_k_pipelined.cuh` (when `shared` is compile-time or bounded by `kMaxStages`), `fp8_a8_tma_mma.cuh`, `ggml_linear_a8.cuh`, and `ggml_linear_tiled.cu` with direct per-template `static` arrays or `static const int` initializers, and cache SM count in `causal_softmax_attention_prompt_wave_tokens`.
3. In [`wrapper/sparse_moe.cpp`](file:///opt/etc/Infernix/src/ops/wrapper/sparse_moe.cpp#L38-L160), replace `std::vector<AddressRange>` with a stack `std::array<AddressRange, 24>`, format error `std::string`s lazily only when a check fails, and pass the already-resolved plan into the launcher.
4. In [`weight_input.cpp`](file:///opt/etc/Infernix/src/ops/weight_input.cpp#L17-L46), change `matrix()` to return `std::array<std::uint64_t, 2>` and compute `concatenate_rows(inputs)` once in `input_projection` and `prepare_gdn_gating_proj_weights`.
5. Replace `std::vector<int32_t>` in [`gated_delta_net/replay.cpp`](file:///opt/etc/Infernix/src/ops/linear_attention/gated_delta_net/replay.cpp#L151-L156) with a stack buffer (`std::array<int32_t, 64>`).

#### Verification & Acceptance Criteria
```bash
ninja -C build-native -j "$(nproc)" infernix-serve infernix_gated_delta_net_test infernix_sparse_moe_test
ctest --test-dir build-native -R "gated_delta_net|sparse_moe|infernix_cuda_stack_check" --output-on-failure
```
- **Acceptance Criteria:** Zero `cudaFuncSetAttribute` / `cudaGetDevice` calls on steady-state kernel launches; zero heap allocations in `wrapper/sparse_moe.cpp` and `weight_input.cpp:matrix()`.

---

### Task T14 (Priority P2): Burn Down `STACK > 0` Local-Memory Spills Across the 255 Spilling Kernel Instantiations and Ratchet `cuda_stack_allowlist.json`

#### Affected Files & Symbols
- [`src/ops/linear_attention/gated_delta_net/recurrent.cu`](file:///opt/etc/Infernix/src/ops/linear_attention/gated_delta_net/recurrent.cu) (`gated_delta_net_recurrent_forward_kernel<true, true, 128, 128, 32>`: `REG:168, STACK:656`)
- [`src/ops/kv_cache/append/vq2_kernels.cuh`](file:///opt/etc/Infernix/src/ops/kv_cache/append/vq2_kernels.cuh) (96 `kv_cache_append_vq_*` instantiations: `STACK:32..144`)
- [`src/ops/lossless/lossless_attention.cu`](file:///opt/etc/Infernix/src/ops/lossless/lossless_attention.cu) (`score_kernel<256, 32>`: `REG:168, STACK:368`)
- [`src/ops/offloaded_sparse_moe/cuda/moe_layer.cu`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/moe_layer.cu) & [`wide_expert.cu`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert.cu) (`ggml_mma_stage_kernel`: `STACK:1312..1568`; `down_select_kernel` / `gate_up_select_kernel`: `STACK:16..432`)
- [`src/ops/ggml/ggml_linear.cu`](file:///opt/etc/Infernix/src/ops/ggml/ggml_linear.cu), [`ggml_linear_a8.cuh`](file:///opt/etc/Infernix/src/ops/ggml/ggml_linear_a8.cuh), [`ggml_linear_q8_0_a8.cu`](file:///opt/etc/Infernix/src/ops/ggml/ggml_linear_q8_0_a8.cu) (`STACK:16..2448`)
- [`tools/cuda_stack/cuda_stack_allowlist.json`](file:///opt/etc/Infernix/tools/cuda_stack/cuda_stack_allowlist.json)

#### Root Cause & Hardware Rationale
Running `PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt static build-native/apps/infernix-serve --spills-only` identifies **255 kernel instantiations with `STACK > 0`**, caused by:
1. Overly restrictive `__launch_bounds__(Threads, MinBlocks)` caps (`e.g. 128` or `168` registers/thread) on kernels whose dynamic shared memory already limits occupancy to 1 or 2 CTAs/SM.
2. Runtime-indexed local arrays (`float state[BV]`, decoded GGML blocks, or VQ2 distance accumulators) that `ptxas` places in local memory (`STL`/`LDL`), thrashing GB10's L1 and 24 MiB L2 caches.

#### Required Implementation Steps
1. Inspect per-kernel SASS (`ckt static --sass --filter <symbol>`) for the highest-impact spilling kernels (`gated_delta_net_recurrent_forward_kernel`, `kv_cache_append_vq_*`, `score_kernel`, `down_select_kernel`, `gate_up_select_kernel`).
2. Replace runtime-indexed local arrays with `#pragma unroll` register loops or shared-memory staging, and relax `__launch_bounds__` `MinBlocks` where shared memory already caps occupancy.
3. Ratchet [`tools/cuda_stack/cuda_stack_allowlist.json`](file:///opt/etc/Infernix/tools/cuda_stack/cuda_stack_allowlist.json) downward as spills are eliminated.

#### Verification & Acceptance Criteria
```bash
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt static build-native/apps/infernix-serve --spills-only
ctest --test-dir build-native -R infernix_cuda_stack_check --output-on-failure
```
- **Acceptance Criteria:** `infernix_cuda_stack_check` passes with reduced allowlist entry count and zero `STACK > 0` on `gated_delta_net_recurrent_forward_kernel` and `kv_cache_append_vq_*`.
