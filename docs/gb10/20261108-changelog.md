# GB10 (`sm_121a`) Changelog — 2026-11-08 (`src/ops` RTX 5090 Artifact Cleanup & Section 1.1 Retune)

## 1. Commit Range & Accounting

- **Base Commit:** `8834c95ff` (`dev-rebased-20261028`, rebased on `origin/main`)
- **Task Reference:** [`docs/tasks/20261108-src-ops-audit-tasks.md`](../tasks/20261108-src-ops-audit-tasks.md) (Section 1.1)
- **Total Files Changed:** 42 source/test files (`173` insertions, `168` deletions) + 2 documentation files (`docs/tasks/20261108-src-ops-audit-tasks.md`, `docs/gb10/20261108-changelog.md`). (`cuda-kernel-toolkit/` is an integrated toolkit and excluded from file-by-file enumeration.)

---

## 2. Architectural Rationale ("Why")

Upstream Infernix targets NVIDIA GeForce RTX 5090 (`sm_120a`, 170 SMs, 96 MiB L2 cache, discrete PCIe 5.0 bus, and half-rate FP32 Tensor Core accumulation on `.kind::f8f6f4`). NVIDIA DGX Spark GB10 (`sm_121a`) differs across four fundamental hardware invariants that required eliminating the remaining RTX 5090 constants in `src/ops`:

1. **48 SMs (`sm_121a`) vs 170 SMs (`sm_120a`):**
   - Wave-sizing constants calibrated for 170 SMs (`kAdaptiveD3Blocks = 2560`, `kTargetCtas = 384`, `kSmallPTargetBlocks = 4096`, `kContextQueryMaxSplit = 85`, `vq_append_split` hardcoded `512..8192` row splits, `rope.cu` `tokens <= 400`, and `embed_gather.cu` `T <= 128/176` crossovers) either over-scheduled empty/tiny waves by `3.3x–3.5x` or under-filled 48 SMs at small token counts (`rmsnorm_rope` launching 6 CTAs at `T=1`, `embed_gather_q8_2048_launch` launching `7..24` CTAs at `T=7..24`).
2. **24 MiB L2 Cache vs 96 MiB L2 Cache (`4x` Smaller L2):**
   - Working-set and prefetch thresholds sized for 96 MiB L2 (`kNextWeightPrefetchLimit = 8 MiB` in `sparse_moe_small_t_plan.cpp`, `kBf16x8CacheSizedMaxElements = 32M` = `64 MiB`/tensor in `bf16_vector.cuh`, `kPlaneBytes = 64 MiB` and `kA8WidePlaneBytes = 64 MiB` in `ggml_linear*`, and `total_blocks > 1024` CTA reversal in `nvfp4_a4_quantize.cuh`) exceeded GB10's entire 24 MiB L2 cache by `2.6x–5.3x`, causing self-eviction of live activations and decoded weight planes.
3. **128 GiB Unified LPDDR5X UMA (`~273 GB/s`) vs Discrete PCIe:**
   - Defaulting `CpuMissService::Options::pcie_divisor` to `3` assumed an independent PCIe DMA engine running alongside CPU miss service, whereas on GB10 UMA a GPU staging copy competes for the same LPDDR5X memory controller as CPU expert execution.
4. **1:1 Full-Rate FP32 Tensor Core Accumulation & Native `QMMA` on `sm_121a`:**
   - On `sm_120a`, `.kind::f8f6f4` FP8 MMA runs at half rate, motivating `mma_fp8_e4m3`'s workaround using `.kind::mxf8f6f4.block_scale` with dummy `0x7f` (`1.0`) UE8M0 scale registers. On `sm_121a`, native `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` (`QMMA`) executes at full rate (`245.88 TFLOP/s` measured in `.ckt_cache/env.json`) without allocating or passing dummy scale registers.

---

## 3. Subsystem Breakdown

### 3.1 `src/ops/common/` & Elementwise / Launcher Kernels
- `M` `src/ops/common/mma.cuh`: Updated `mma_fp8_e4m3` to emit native `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` (`QMMA`) when `__CUDA_ARCH__ == 1210` (`sm_121a`), retaining `mma_fp8_e4m3_scaled(..., 0x7Fu, 0x7Fu)` only for `sm_120a`.
- `M` `src/ops/common/bf16_vector.cuh`: Lowered `kBf16x8CacheSizedMaxElements` from `32LL * 1024LL * 1024LL` (`64 MiB`/tensor) to `8LL * 1024LL * 1024LL` (`16 MiB`/tensor) so `add_bias` and `gelu` switch to `.cs` streaming cache hints when tensors exceed GB10's 24 MiB L2.
- `M` `src/ops/rmsnorm_rope/launch.cu`: Added `tokens <= 4` branch in `launch_text` launching `rmsnorm_rope_d256_text_kernel<QHeads, KHeads, 1>` (`1` head/block = `18 * T` CTAs for `<16, 2>`) instead of `kTextHeadsPerBlock = 3` (`6 * T` CTAs).
- `M` `src/ops/launcher/rope.cu`: Lowered `launch_fixed_pair` `DflashText1D` `<8, 32, 8>` split cutoff from `tokens <= 400` (170 SMs) to `tokens <= 128` (48 SMs).
- `M` `src/ops/launcher/embed_gather.cu`: Raised `embed_gather_q8_2048_launch` `Grouped` threshold from `T <= 6` to `T <= 24` (`8 * T` CTAs across 48 SMs) and lowered `<10, 128>` -> `<5, 128>` crossover in `embed_gather_q8_launch` and `embed_gather_fp8_launch` from `T <= 128` / `176` to `T <= 48`.
- `M` `src/ops/launcher/argmax.cu`: Updated `kFullAggregateBlock` from `128` to `256` and used `kShortlistAggregateBlock` (`256`, `32` CTAs/col) for `131072` rows across all `t_count`.
- `M` `src/ops/launcher/vision_pos_embed.cu`: Lowered `kSmallPTargetBlocks` from `4096` (`24 * 170` SMs) to `1152` (`24 * 48` SMs) and `x.ne[1] < 1024` (`6 * 170` SMs) to `x.ne[1] < 288` (`6 * 48` SMs).
- `M` `src/ops/kv_cache/append/vq2_launch.cu`: Removed the hardcoded RTX 5090 `if (rows >= 512 && rows <= 8192)` override in `vq_append_split` so the `2 * multiprocessors` (`96` CTAs on 48-SM GB10) loop governs all row counts.

### 3.2 Input Projections, Attention, & Lossless Kernels
- `M` `src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.cpp`: Raised `AllowA4` `SmallTFusedA16` threshold in `nvfp4_gdn_conv_resolve_plan` from `tokens <= 3` to `tokens <= 16` to match `nvfp4_gdn_input_plan.cpp:36`, eliminating the unfused `[10240, T]` global memory materialization and `nvfp4_gdn_snapshot_post_launch` pass at `T = 4..16`.
- `M` `src/ops/gdn_input_proj/fp8/fp8_gdn_input_a8.cu`: Updated `fp8_gdn_input_partial_capacity_bytes` to return `0` (matching `fp8_attn_input_partial_capacity_bytes`, as `kRoutes` never selects `MidBulk` or `Bulk` on GB10) and removed the 170-SM `tokens <= 96` `Fp8A8ResidualT32R128` detour in `fp8_gdn_input_a8_residual_launch`.
- `M` `src/ops/attn_input_proj/fp8/fp8_attn_input_a8.cu`: Removed the 170-SM `if (tokens > 64 && tokens <= 104)` `Fp8A8ResidualT32R128` detour in `fp8_attn_input_a8_residual_launch`.
- `M` `src/ops/softmax_attention/common/context_query.cuh`: Lowered `kContextQueryMaxSplit` from `85` (`170 / 2` SMs) to `64`.
- `M` `src/ops/softmax_attention/sliding_window/kernel.cuh`: Replaced `kContextQueryMaxSplit` (`85`) with `32` (`kSlidingWindowMaxSplits`) in `sliding_window_attention_reduce_kernel`.
- `M` `src/ops/lossless/lossless_attention.cu`: Lowered `kTargetCtas` from `384` (170 SMs) to `144` (`3 * 48` SMs).

### 3.3 Sparse MoE, Offloaded Sparse MoE, & GGML Linear
- `M` `src/ops/sparse_moe/small_t/sparse_moe_small_t_plan.cpp`: Added `device_prefetch_limit()` querying `cudaDevAttrL2CacheSize` and clamping `l2 / 12` to `[2 MiB, 8 MiB]` (`2 MiB` on GB10), matching `sparse_moe_decode_plan.cpp`.
- `M` `src/ops/sparse_moe/decode/sparse_moe_decode_kernels.cu`: Lowered `kAdaptiveD3Blocks` and `kAdaptiveD4Blocks` from `5 * 512 = 2560` to `16 * 48 = 768`.
- `M` `src/ops/offloaded_sparse_moe/cpu/miss_service.h`: Changed `CpuMissService::Options::pcie_divisor` default from `3` to `0` (matching `cpu_pcie_divisor = 0` in `expert_offload.h` and `program.h`).
- `M` `src/ops/offloaded_sparse_moe/cuda/wide_expert.h`: Lowered `kGgmlStageDecodeMinColumns` from `512` to `128` (matching `kBf16WideMinColumns = 128`).
- `M` `src/ops/ggml/ggml_linear.cu` & `M` `src/ops/ggml/ggml_linear_a8.h`: Lowered `kPlaneBytes` and `kA8WidePlaneBytes` from `64 MiB` to `16 MiB` (`2/3` of GB10's 24 MiB L2), saving `48 MiB` of workspace per plane.

### 3.4 `src/ops/linear/` Quantizer & `/shapes/` Files
- `M` `src/ops/linear/nvfp4/nvfp4_a4_quantize.cuh`: Lowered CTA order reversal threshold in `nvfp4_a4_row_scaled_residual_quantize_kernel` from `total_blocks > 1024` (96 MiB L2) to `total_blocks > 256` (24 MiB L2).
- `M` `src/ops/linear/mxfp8/shapes/{n1280_k2560,n2560_k640,n2560_k6144,n13952_k2560,n16384_k2560}.cu`: Updated `launch_f16` / `select_f16` from the stale RTX 5090 TMA tile ladders to match the GB10 TMA tiles from `launch_a16` at `T >= kMxfp8F16MinTokens` (`256`).
- `M` `src/ops/linear/fp8/shapes/{n1280_k2560,n2560_k640,n2560_k6144,n13312_k2560,n16384_k2560}.cu`: Cleaned up stale RTX 5090 `kPairMinTokens` comments.
- `M` `src/ops/linear/bf16/shapes/{n48_k2560,n256_k5120,n1664_k2560,n2560_k2560,n5120_k6144,n14336_k5120,n248320_k2560}.cu`: Added GB10 attribution headers, removed unused schedule type aliases (`S0`, `S2..S6`, `S9`), and updated `S6` (`T >= 97`) in `n248320_k2560.cu` to `Bf16MmaRaster::Grouped, 8` for GB10's 24 MiB L2.
- `M` `src/ops/linear/q4/shapes/n131072_k3072.cu`: Updated `T = 17..24` to `launch_q4_a16_sliced_r32_t24_w4_s2` and `T = 33..72` to `launch_q4_a16_mma_r64_t72`, while keeping `launch_q4_a16_mma_r64_t128` for `T > 112` (`1.23x` faster at `T=64`: `1354.50 -> 1100.61 us`; `1.43x` faster at `T=128`: `2411.33 -> 1687.49 us`).
- `M` `src/ops/linear/q6/shapes/{n1152_k1536,n34816_k5120}.cpp`: Replaced fragmented 170-SM RTX 5090 schedules with 48-SM wave-aligned schedules.
- `M` `tests/ops/test_gdn_input_proj_conv_snapshot.cpp`: Updated NVFP4 `AllowA4` route-boundary assertions from `T = 3/4` to `T = 16/17` to match `nvfp4_gdn_snapshot_plan.cpp` and `test_gdn_input_proj.cpp`.

---

## 4. Verification & Benchmark Summary

- **CUDA Stack-Spill Ratchet Gate (`infernix_cuda_stack_check`):** Passed (`0` new `STACK > 0` or `LOCAL > 0` symbols in `infernix-serve`).
- **Static SASS Verification (`ckt static --sass`):** Confirmed `fp8_a8_tma_mma_kernel` emits native `QMMA` instructions (`tc=64 (int=0, flt=0, fp8=64, fp4=0)`) on `sm_121a`.
- **`ckt shapes list` Coverage:** All `111/111` linear shape files in `src/ops/linear/*/shapes/` now report `origin = GB10`.
- **Unit & Oracle Tests (`29/29` CTest suites + targeted shape tests passed):**
  - `infernix_lossless_ops_test`, `infernix_argmax_test`, `infernix_add_bias_test`, `infernix_rope_test`, `infernix_rmsnorm_rope_test`, `infernix_vision_pos_embed_test`, `infernix_gelu_test`, `infernix_kv_cache_append_test`, `infernix_sliding_window_attention_test`, `infernix_kv_cache_vq_test`, `infernix_softmax_attention_fp8_test`, `infernix_softmax_attention_test`
  - `infernix_embedding_test`, `infernix_ggml_linear_a8_wide_test`, `infernix_linear_f16_accumulate_test`, `infernix_offloaded_moe_ggml_test`, `infernix_ggml_linear_a8_test`, `infernix_attn_input_proj_test`, `infernix_offloaded_moe_ggml_a8_test`, `infernix_linear_fp8_a8_test`, `infernix_ggml_linear_test`, `infernix_sparse_moe_test`, `infernix_offloaded_moe_plan_test`, `infernix_gdn_input_proj_test`, `infernix_linear_nvfp4_a4_test`, `infernix_offloaded_moe_layer_test`, `infernix_gdn_input_proj_conv_snapshot_test`, `infernix_linear_add_fp8_test`, `infernix_linear_swiglu_fp8_test`
  - `infernix_ops_linear_test`: `linear_bf16_a16 --shape 248320 2560`, `linear_q4_a16 --shape 131072 3072`, `linear_q6_a16 --shape 1152 1536`, `linear_q6_a16 --shape 34816 5120`, `linear_mxfp8_a16`.

---

## 5. Rebase Conflict Resolution Policy

- **L2 Cache & Wave Constants (`src/ops/**`):** Preserve GB10 24 MiB L2 limits (`device_prefetch_limit()` clamping to `2 MiB` on GB10, `kBf16x8CacheSizedMaxElements = 8M`, `kPlaneBytes = 16 MiB`, `total_blocks > 256` CTA reversal) and 48-SM wave constants (`kAdaptiveD3Blocks = 768`, `kTargetCtas = 144`, `kSmallPTargetBlocks = 1152`, `kContextQueryMaxSplit = 64`) over upstream 170-SM / 96-MiB-L2 constants.
- **FP8 Tensor Core MMA (`src/ops/common/mma.cuh`):** Preserve `__CUDA_ARCH__ == 1210` native `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` (`QMMA`) path on `sm_121a`.
- **Linear `/shapes/` (`src/ops/linear/*/shapes/*`):** Preserve GB10-tuned tile schedules and `Grouped, 8` L2 rasters over upstream RTX 5090 ladders.
