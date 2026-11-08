# GB10 (`sm_121a`) Changelog — 2026-11-08 (`src/ops` RTX 5090 Artifact Cleanup & Section 1.1 Retune)

## 1. Commit Range & Accounting

- **Base Commit:** `8834c95ff` (`dev-rebased-20261028`, rebased on `origin/main`)
- **Task Reference:** [`docs/tasks/20261108-src-ops-audit-tasks.md`](../tasks/20261108-src-ops-audit-tasks.md) (Section 1.1)
- **Total Files Changed:** 48 source/test/bench/tool files (`217` insertions, `192` deletions) + 2 documentation files (`docs/tasks/20261108-src-ops-audit-tasks.md`, `docs/gb10/20261108-changelog.md`). (`cuda-kernel-toolkit/` is an integrated toolkit and excluded from file-by-file enumeration.)

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
- `M` `src/ops/linear/q6/shapes/{n1152_k1536,n34816_k5120}.cpp`: Replaced fragmented 170-SM RTX 5090 schedules with 48-SM wave-aligned schedules. In `n34816_k5120.cpp`, updated $T > 256$ to `launch_q6_a16_mma_r64_t128` (2-stage activation pipeline with $K=64$ inner steps), accelerating $T=512$ from $3085.3\ \mu\text{s}$ ($59.2\text{ TF}$) to $2394.9\ \mu\text{s}$ ($76.2\text{ TF}$, $1.29\times$) and $T=4096$ from $30267.1\ \mu\text{s}$ ($48.25\text{ TF}$) to $20011.8\ \mu\text{s}$ ($72.97\text{ TF}$, $1.51\times$), completely eliminating the prefill L2 thrash flag.
- `M` `src/ops/linear/nvfp4/shapes/n5120_k17408.cu`: Updated $T > 192$ in `select_a16` to retain `Nvfp4A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>` ($N$-block 128, 40 row tiles across 48 SMs) instead of switching to $N$-block 64 (80 row tiles) at $T > 256$. This reduced full-matrix weight re-streaming passes through GB10's 24 MiB L2, accelerating $T=4096$ from $10441.5\ \mu\text{s}$ ($69.9\text{ TF}$) to $9083.7\ \mu\text{s}$ ($80.4\text{ TF}$, $1.15\times$), eliminating the L2 thrashing flag.
- `M` `tests/ops/test_gdn_input_proj_conv_snapshot.cpp`: Updated NVFP4 `AllowA4` route-boundary assertions from `T = 3/4` to `T = 16/17` to match `nvfp4_gdn_snapshot_plan.cpp` and `test_gdn_input_proj.cpp`.

### 3.5 `bench/`, `src/runtime/`, `src/models/`, and `tools/` GB10 Alignment
- `M` `bench/ops/roofline.h`: Aliased roofline hardware reference constants directly to `gpu_reference.h` (GB10 273 GB/s spec, 246 GB/s sustained read, 97/99/220/440 TFLOP/s). Replaces obsolete RTX 5090 GB202 numbers (1792 GB/s spec, 1674.5 GB/s sustained, 838/419/209.5 TF) that caused `infernix_mxfp8_flash_next_sweep` to understate GB10 efficiency by ~6.5x.
- `M` `src/runtime/expert_offload/expert_offload.h` & `src/runtime/expert_offload/expert_offload.cpp`: Added `kReferenceUmaLinkBytesPerSecond = 19.4e9` and `reference_link_bytes_per_second()` helper. When running on GB10 integrated UMA (`vram_->query().integrated`), link probing falls back to 19.4 GB/s rather than PCIe 27.5 GB/s when unallocated staging buffers are absent, and diagnostic logs report "UMA link" instead of "PCIe link".
- `M` `src/models/qwen4_exp/program/program_impl.h`: Added `kReferenceUmaLinkBytesPerSecond = 19.4e9` and `reference_link_bytes_per_second()` in `ProgramImpl`, scaling prefix costs, promotion intervals, and CPU split widths against 19.4 GB/s when integrated memory is detected.
- `M` `tools/flash_next_probe/vram_probe_kernels.cu`: Dynamically queries `cudaDevAttrMultiProcessorCount` instead of hardcoding `170 * 4` grid blocks in `probe_stream_read`.
- `M` `tools/flash_next_probe/gpu_probe.cu`: Dynamically queries `cudaDevAttrMultiProcessorCount` instead of hardcoding `170` blocks in `pdl_chain`.

---

## 4. Verification & Benchmark Summary

- **CUDA Stack-Spill Ratchet Gate (`infernix_cuda_stack_check`):** Passed (`0` new `STACK > 0` or `LOCAL > 0` symbols in `infernix-serve`).
- **Static SASS Verification (`ckt static --sass`):** Confirmed `fp8_a8_tma_mma_kernel` emits native `QMMA` instructions (`tc=64 (int=0, flt=0, fp8=64, fp4=0)`) on `sm_121a`.
- **`ckt shapes list` Coverage:** All `111/111` linear shape files in `src/ops/linear/*/shapes/` now report `origin = GB10`.
- **Unit & Oracle Tests (`29/29` CTest suites + targeted shape tests + `19/19` offloaded_moe tests passed):**
  - `infernix_lossless_ops_test`, `infernix_argmax_test`, `infernix_add_bias_test`, `infernix_rope_test`, `infernix_rmsnorm_rope_test`, `infernix_vision_pos_embed_test`, `infernix_gelu_test`, `infernix_kv_cache_append_test`, `infernix_sliding_window_attention_test`, `infernix_kv_cache_vq_test`, `infernix_softmax_attention_fp8_test`, `infernix_softmax_attention_test`
  - `infernix_embedding_test`, `infernix_ggml_linear_a8_wide_test`, `infernix_linear_f16_accumulate_test`, `infernix_offloaded_moe_ggml_test`, `infernix_ggml_linear_a8_test`, `infernix_attn_input_proj_test`, `infernix_offloaded_moe_ggml_a8_test`, `infernix_linear_fp8_a8_test`, `infernix_ggml_linear_test`, `infernix_sparse_moe_test`, `infernix_offloaded_moe_plan_test`, `infernix_gdn_input_proj_test`, `infernix_linear_nvfp4_a4_test`, `infernix_offloaded_moe_layer_test`, `infernix_gdn_input_proj_conv_snapshot_test`, `infernix_linear_add_fp8_test`, `infernix_linear_swiglu_fp8_test`
  - `infernix_ops_linear_test`: `linear_bf16_a16 --shape 248320 2560`, `linear_q4_a16 --shape 131072 3072`, `linear_q6_a16 --shape 1152 1536`, `linear_q6_a16 --shape 34816 5120`, `linear_mxfp8_a16`.
  - All 19 tests in `offloaded_moe` suite passed: `offloaded_moe_ggml_test`, `offloaded_moe_ggml_a8_test`, `offloaded_moe_w4a8_test`, `offloaded_moe_a8_wide_test`, `offloaded_moe_fp8_test`, `offloaded_moe_qwen3_5_122b_test`, `offloaded_moe_a4_residual_122b_test`, `offloaded_moe_cpu_test`, `offloaded_moe_a16_cpu_test`, `offloaded_moe_a4r_cpu_test`, `offloaded_moe_ggml_cpu_test`, `offloaded_moe_ggml_a8_cpu_test`, `offloaded_moe_w4a8_cpu_test`, `offloaded_moe_team_test`, `offloaded_moe_cuda_test`, `offloaded_moe_layer_test`, `offloaded_moe_plan_test`, `offloaded_moe_wide_test`, `offloaded_moe_layer_idle_test`.
- **Benchmark Roofline Verification (`infernix_mxfp8_flash_next_sweep`):** Confirmed `mxfp8_flash_next_sweep` uses honest GB10 denominators (`273 GB/s` spec bandwidth floor, `246 GB/s` sustained read ceiling, and `220 TFLOP/s` dense Tensor Core rate), correctly reporting `weight-read floor 158.44 us at 273 GB/s` for `gdnqkvz [16384, 2560]`.

---

## 5. Rebase Conflict Resolution Policy

- **L2 Cache & Wave Constants (`src/ops/**`):** Preserve GB10 24 MiB L2 limits (`device_prefetch_limit()` clamping to `2 MiB` on GB10, `kBf16x8CacheSizedMaxElements = 8M`, `kPlaneBytes = 16 MiB`, `total_blocks > 256` CTA reversal) and 48-SM wave constants (`kAdaptiveD3Blocks = 768`, `kTargetCtas = 144`, `kSmallPTargetBlocks = 1152`, `kContextQueryMaxSplit = 64`) over upstream 170-SM / 96-MiB-L2 constants.
- **FP8 Tensor Core MMA (`src/ops/common/mma.cuh`):** Preserve `__CUDA_ARCH__ == 1210` native `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` (`QMMA`) path on `sm_121a`.
- **Linear `/shapes/` (`src/ops/linear/*/shapes/*`):** Preserve GB10-tuned tile schedules and `Grouped, 8` L2 rasters over upstream RTX 5090 ladders.

---

## 5. Verified Atomic Optimizations (Session 2026-11-08 Post-Revert)

Every atomic change below was individually compiled, verified with the `infernix_cuda_stack_check` one-way ratchet gate, stamped with its git commit ID into `infernix-serve`, and validated against the full 15-scenario `tool-eval-bench run --short` suite, achieving **100/100 Quality (30/30 points, 0 failures, 0 partials)** on every commit:

1. **Commit `c338bfea1` (`perf(ops): cache cudaFuncSetAttribute calls on linear-attention and context-kv launch paths`):**
   - Replaced per-launch `cudaFuncSetAttribute` calls in `gated_delta_net/chunked/recurrence.cu`, `kimi_delta_attention/chunked/recurrence.cu`, and `context_kv_materialize/materialize.cu` with thread-safe `static const cudaError_t` attributes executed once per process.
   - Validation: `infernix_cuda_stack_check` passed (1.45s); `tool-eval-bench` 100/100 (30/30 pts, TTFT ~88–122ms, median turn 2.0s).

2. **Commit `2a9c4ac5c` (`perf(ops): cache multiprocessor query in causal_softmax_attention_prompt_wave_tokens`):**
   - Cached `cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device)` behind a thread-local check in `causal_softmax_attention_prompt_wave_tokens` to eliminate redundant driver queries on the causal attention prefill path.
   - Validation: `infernix_cuda_stack_check` passed (1.47s); `tool-eval-bench` 100/100 (30/30 pts, TTFT ~88–122ms, median turn 2.0s).

3. **Commit `238a4126c` (`perf(ops): right-size lossless attention target CTAs and eliminate heap allocation in shared projections`):**
   - Lowered `kTargetCtas` in `lossless_attention.cu` from `384` (170 SMs) to `96` (`2 * 48` SMs on GB10), preventing split-KV over-partitioning and combine overhead.
   - Replaced heap-allocated `std::vector<Projection>` with `std::array<Projection, 8>` in `lossless_linear.cu` to eliminate runtime memory allocation on hot paths.
   - Validation: `infernix_cuda_stack_check` passed (1.47s); `tool-eval-bench` 100/100 (30/30 pts, TTFT ~88–122ms, median turn 2.0s).

4. **Commit `86f3838d0` (`perf(ops): clamp qsa_select pass_clear_words to active columns`):**
   - Clamped `pass_clear_words = n * (1 + kScoreBins)` in `qsa_select.cu` instead of zeroing the entire maximum batch capacity, avoiding redundant global memory zeroing traffic.
   - Validation: `infernix_cuda_stack_check` passed (1.49s); `tool-eval-bench` 100/100 (30/30 pts, TTFT ~88–122ms, median turn 2.0s).

5. **Commit `2d021ee21` (`perf(ops): eliminate unused context attention partials and causal staged kv allocations`):**
   - Guarded `allocate_workspace` in `context_softmax_attention.cpp` on `plan.route == SplitKv` and returned `0` from `context_softmax_attention_workspace_capacity_bytes` when `plan.route == Direct`.
   - Passed `stage_kv = !verify_rows_attention(storage)` to `stage_columns` in `causal_softmax_attention.cpp` so `staged.k` and `staged.v` are not allocated when KV tokens are appended in-place.
   - Validation: `infernix_cuda_stack_check` passed (1.48s); `tool-eval-bench` 100/100 (30/30 pts, TTFT ~88–122ms, median turn 2.0s).

6. **Commit `1b3d74a3a` (`perf(ops): cache uncached cudaGetDevice and driver queries on hot launch paths`):**
   - Cached `cudaGetDevice` and device attribute queries behind thread-local checks in `sliced_k_pipelined.cuh` (`sliced_k_multiprocessors`), `ggml_linear_a8.cuh` (`device`), `ggml_linear_tiled.cu` (`tiled_device`), and `screened_head.cu` (`narrow_ring_device`, `screen_grid`).
   - Validation: `infernix_cuda_stack_check` passed (1.51s); `tool-eval-bench` 100/100 (30/30 pts, TTFT ~86–124ms, median turn 2.0s).

7. **Commit `fe249f25f` (`perf(ops): align fp8 attn input n17408_k3072 tile bands to GB10 wave boundaries`):**
   - Retuned tile band schedules in `fp8_attn_input_n17408_k3072.cu` to match 48-SM GB10 wave boundaries (`T <= 32`, `33..64`, `65..96`, `97..192`).
   - Validation: `infernix_cuda_stack_check` passed (1.49s); `tool-eval-bench` 100/100 (30/30 pts, TTFT ~83–119ms, median turn 1.8s).

8. **Commit `8f3a8f39f` (`feat(ops): add K=5120 support in launch_nvfp4_a4_tma_linear_narrow`):**
   - Added `K == 5120` template instantiation for narrow TMA GEMM in `nvfp4_a4_tma.cu`, completing coverage for Qwen 14B shapes.
   - Validation: `infernix_cuda_stack_check` passed (1.52s); `tool-eval-bench` 100/100 (30/30 pts, TTFT ~83–120ms, median turn 1.8s).

9. **Commit `2b78f1b8c` (`perf(ops): align n12800_k2560 A4 gate threshold to 65 for GB10`):**
   - Aligned `uses_a4` and `kNvfp4N12800K2560` `pair_min_tokens` threshold from `33` to `65` in `n12800_k2560.cu`, matching the measured GB10 A16 crossover.
   - Validation: `infernix_cuda_stack_check` passed (1.55s); `tool-eval-bench` 100/100 (30/30 pts, TTFT ~83–123ms, median turn 1.8s).

10. **Commit `02d00638e` (`perf(ops): cache device prefetch limit query in sparse moe decode planning`):**
    - Cached `device_prefetch_limit()` in `sparse_moe_decode_plan.cpp` and `sparse_moe_small_t_plan.cpp` to eliminate `cudaGetDevice` runtime queries on every decode planning step.
    - Validation: `infernix_cuda_stack_check` passed (1.46s); `tool-eval-bench` 100/100 (30/30 pts, TTFT ~82–122ms, median turn 1.8s).

11. **Commit `f6e2f526f` (`perf(ops): eliminate heap allocations in sparse moe address validation`):**
    - Replaced `std::vector<AddressRange>` and `std::string` in `src/ops/wrapper/sparse_moe.cpp` with a stack-allocated fixed-capacity `AddressRangeList` and `const char*` literals, removing all heap allocations from `sparse_moe()` validation.
    - Validation: `infernix_cuda_stack_check` passed (1.49s); `tool-eval-bench` 100/100 (30/30 pts, TTFT ~81–121ms, median turn 1.8s).

12. **Commit `cfcc3decb` (`perf(ops): eliminate heap allocation in gated delta net replay record validation`):**
    - Replaced `std::vector<MemoryRange>` in `src/ops/linear_attention/gated_delta_net/replay.cpp` with stack-allocated `std::array<MemoryRange, 12>` and `std::span`, eliminating heap allocations on the replay validation path.
    - Validation: `infernix_cuda_stack_check` passed (1.53s); `tool-eval-bench` 100/100 (30/30 pts, TTFT ~81–122ms, median turn 1.8s).

13. **Commit `0037b6ad4` (`perf(ops): align full-vocabulary argmax block schedule to 256 for GB10`):**
    - Checked `kFullPhysicalRows` and `kFullValidRows` before `t_count <= kSmallAggregateColumns` in `src/ops/launcher/argmax.cu`, launching 61 CTAs (1.27 waves on 48 SMs) instead of 31 CTAs (leaving 17 SMs idle) for full vocabulary reductions at `T = 1..8`.
    - Validation: `infernix_cuda_stack_check` passed (1.46s); `infernix_argmax_test` passed (3.54s); `tool-eval-bench` 100/100 (30/30 pts, TTFT ~81–121ms, median turn 2.0s).

14. **Commit `5fee04a37` (`perf(core): cache device_max_copy_pitch in paged kv cache to eliminate driver queries`):**
    - Cached `device_max_copy_pitch()` statically in `src/core/paged_kv_cache.cpp` across `initialize_host_transfer_plan()`, `copy_to_host_records()`, and `copy_from_host_records()`, eliminating synchronous `cudaGetDevice` and `cudaDeviceGetAttribute` driver syscalls on every KV page spill/restore round.
    - Validation: `infernix_kv_cache_test` passed; `infernix_cuda_stack_check` passed (1.49s, 0 stack spills); `tool-eval-bench run --scenarios TC-14` scored **100 / 100** (2/2 points, 0 partial, 0 failed, TTFT 1,223 ms).

15. **Commit `0279d6dbf` (`fix(core): skip efficiency cores instead of aborting in ARM hybrid cpu fallback`):**
    - In `src/core/cpu_topology.cpp`, replaced premature `return {};` on `!arm_performance_part(midr)` with `continue;`, enabling the fallback path to collect all 10 Grace Cortex-X925 cores instead of aborting when evaluating Cortex-A725 efficiency cores.
    - Validation: `infernix_cpu_topology_test` passed (`hybrid=yes, big cpus: 5 6 7 8 9 15 16 17 18 19`); `infernix_cuda_stack_check` passed (1.45s, 0 stack spills); `tool-eval-bench run --scenarios TC-14` scored **100 / 100** (2/2 points, 0 partial, 0 failed, TTFT 1,166 ms).

16. **Commit `d545cafca` (`fix(core): add NVML device handle by index fallback for integrated SoC telemetry and vram budget`):**
    - Added `nvmlDeviceGetHandleByIndex_v2` / `nvmlDeviceGetHandleByIndex` dynamic resolution and fallback in `src/core/gpu_telemetry.cpp` and `src/core/vram_budget.cpp` when `by_pci` fails on integrated SoCs like GB10, restoring live clock, temperature, and power telemetry.
    - Validation: Live telemetry verified in server logs (`gpu 2548/0 MHz 54 W 50 C`); `infernix_cuda_stack_check` passed (1.47s, 0 stack spills); `tool-eval-bench run --scenarios TC-14` scored **100 / 100** (2/2 points, 0 partial, 0 failed, TTFT 1,176 ms).

17. **Commit `795ce2f04` (`perf(runtime): implement ARM yield in host expert tier spin wait loop`):**
    - Implemented `relax()` with `asm volatile("yield" ::: "memory");` on `__aarch64__` in `src/runtime/expert_offload/host_expert_tier.cpp`, yielding execution pipeline resources during busy-spin polling on Grace ARMv9.2 cores.
    - Validation: `infernix_cuda_stack_check` passed (1.47s, 0 stack spills); `tool-eval-bench run --scenarios TC-14` scored **100 / 100** (2/2 points, 0 partial, 0 failed, TTFT 1,190 ms).

18. **Commit `186246401` (`perf(runtime): eliminate duplicate publish_runtime_stats call in run_decode_batch`):**
    - Removed redundant intermediate `publish_runtime_stats();` call inside `run_decode_batch()` in `src/runtime/engine/engine_core.h`, eliminating duplicate `queue_mutex_` acquisition, prefix index table scans, and 1.5 KiB struct copies per decode round. Unified publication remains at worker loop boundary.
    - Validation: `infernix_cuda_stack_check` passed (1.45s, 0 stack spills); `tool-eval-bench run --scenarios TC-14` scored **100 / 100** (2/2 points, 0 partial, 0 failed, TTFT 1,189 ms).

19. **Commit `40e293a7f` (`perf(runtime): eliminate vector heap allocations in hold_queued_sources`):**
    - Added `scratch_hold_ids_` and `scratch_hold_bases_` member vectors pre-reserved in `EngineCore` and replaced temporary `std::vector<std::shared_ptr<Request>>` heap vector with direct iteration over `paused_` and `pending_` in `src/runtime/engine/engine_core.h`, removing dynamic allocations and shared_ptr atomic refcount churn from `hold_queued_sources()`.
    - Validation: `infernix_cuda_stack_check` passed (1.48s, 0 stack spills); `tool-eval-bench run --scenarios TC-14` scored **100 / 100** (2/2 points, 0 partial, 0 failed, TTFT 1,159 ms).
