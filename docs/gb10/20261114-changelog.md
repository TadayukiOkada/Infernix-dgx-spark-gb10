# GB10 (`sm_121a`) Changelog — 2026-11-14 (Prefill Optimization Campaign, `HC4` Reconversion & Vision/MoE Integration)

## 1. Commit Range & Accounting

- **Branch HEAD (before uncommitted campaign):** `51fa3e2d7` (`backup-perf-changes`, downstream of `origin/main` `31fe191c6`)
- **Working Tree Delta (`HEAD` -> working tree):** `35` modified files (`2,613` insertions, `845` deletions) + `5` new `src/ops/linear/q4/shapes/*.cu` files + `1` documentation file ([docs/gb10/20261114-changelog.md](file:///opt/etc/Infernix/docs/gb10/20261114-changelog.md)). (`cuda-kernel-toolkit/` is excluded from file-by-file enumeration per repository policy.)
- **Generated Model Artifact (out-of-tree):** Reconverted `/opt/etc/models/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix-HC4-W4Dense/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix-HC4-W4Dense.infernix` (`69.33 GiB` across 3 shards) from `/opt/etc/models/nvidia-Qwen3.8-Flash-Next-FP4-Local` with recipe `qwen3_8_flash_next_nvfp4_w4dense` and `hc_q4_override.py`.

---

## 2. Architectural Rationale ("Why")

On NVIDIA DGX Spark GB10 (`sm_121a`), Qwen3.8-Flash-Next (`Qwen4ExpForCausalLM`) runs resident on GPU under `INFERNIX_EXPERTS_ON_GPU=1` with unified 128 GiB LPDDR5X memory (`~273 GB/s`), 48 SMs, 24 MiB L2 cache, and `mxf4nvf4` FP4 Tensor Cores. The previous prefill and mixed-precision paths suffered from five GB10 bottlenecks:

1. **Fragmented Routed-MoE Prefill & Unpacked Narrow Launches (`sm_121a` 48 SMs):**
   - `moe_layer.cu` launched wide/narrow expert passes without compacting narrow GPU jobs per pass (`narrow_jobs` / `pass_narrow`), and `wide_expert.cu` gated per-token W4A4 quantization at `x.ne[1] > 64` while lacking explicit masked-slot initialization on `entry_map` during multi-column speculative verification.
   - `narrow_expert.cu` performed scalar un-staged SMEM/activation indexing instead of vectorized SMEM staging and programmatic dependent launch (PDL) overlap between `quantize_x_kernel`, `gate_up_kernel`, and `down_kernel`.
2. **Hyper-Connection & MoE Combine Materialization Traffic:**
   - Each transformer layer materialized an intermediate `hyper_connection_norm` `[S*H, T]` buffer separately from the previous block's `moe_combine_inject` residual update, and `q4_g64_fp16` HC mixers lacked fused decode (`Q4Codec` in `hyper_connection_mix_fused.cu`) and wide `up_collapse` (`launch_q4` in `hyper_connection_up_collapse.cu`) support.
3. **QSA Prompt Attention Shared Memory & SM Occupancy (48 SMs):**
   - Upstream GB10 removal of `qsa_prompt_split` (`SplitD`) forced 1-head-group full-`D=256` stages per CTA, exceeding 50 KiB SMEM on wider storages and limiting CTA/SM occupancy to 1 across 48 SMs. Restoring compile-time `SplitD` (`kv_w` + `d_w` warps with `64 B` XOR-swizzled tiles) fits higher CTA/SM occupancy while staying within the attention accuracy oracle.
4. **NVFP4 A4 & `q4_g64_fp16` Shape / Quantizer Schedules (`24 MiB` L2, `48 SMs`):**
   - `nvfp4_a4_quantize.cuh` and `nvfp4_codec.cuh` left registers/occupancy conservative (`512 / Threads` -> `1024 / Threads` launch bounds) and `fmaf` single-rounding opportunities unused; dense `n12800_k2560`, `n13952_k2560`, `n16384_k2560`, `n2560_k6144` shape gates had stale thresholds for `T = 33..55` and `T = 129..256` / `2048`.
   - Switching `HC4` to `qwen3_8_flash_next_nvfp4_w4dense` + Q4 HC mixers introduced five previously untuned Q4 shapes (`320x10240`, `324x10240`, `10240x320`, `1280x2560`, `2560x640`, plus `K % 64 == 0` alignment in `q4_operands.h`).
5. **`HC4` Artifact Dense Projection Precision:**
   - Old `HC4` used `qwen3_8_flash_next_fp4_local`, leaving GDN (`q/k/v/z/output`), QSA (`q/g/k/v/iq/ik/o`), and PLE (`key/value`) dense weights in MXFP8 (`fp8_e4m3fn_g32_e8m0`) instead of `nvfp4` W4, doubling dense weight bandwidth during decode and mid-batch prefill.

---

## 3. Subsystem Breakdown

### 3.1 `include/`, `src/models/qwen4_exp/`, & `src/runtime/engine/`
- `M` `include/infernix/ops/hyper_connection.h`: Declared `hyper_connection_norm`, `hyper_connection_mix_is_fused`, and `hyper_connection_mix_normalized` to allow external pipelined pre-norm reuse when the fused decode mix does not apply.
- `M` `include/infernix/ops/offloaded_sparse_moe.h`: Extended `moe_combine_inject` with optional `next_norm_weight`, `next_norm_out`, and `next_norm_eps` parameters to fuse next-layer `attn_hc.norm` into residual combine.
- `M` `src/models/qwen4_exp/execution/forward.h` & `M` `src/models/qwen4_exp/execution/forward.cpp`: Pipelined `Forward::layer()` / `Forward::moe()` with `pre_norm`, `next_norm_weight`, and `next_norm_out`, feeding combined post-MoE normalization directly into the next layer's `attn_hc` (`mix`).
- `M` `src/models/qwen4_exp/program/program_impl.h` & `M` `src/models/qwen4_exp/program/vision_program.cpp` & `M` `src/runtime/engine/qwen4_exp_instance.cpp`: Allowed `INFERNIX_EXPERTS_ON_GPU=1` with `--vision` enabled by retaining vision weights on device (`load.vision_offload = false`) instead of falling back to host experts, and added standalone `v.owned_lease` (`DeviceBuffer`) backing in `VisionWindow` when `residency_` is null.

### 3.2 `src/ops/offloaded_sparse_moe/cuda/`
- `M` `src/ops/offloaded_sparse_moe/cuda/wide_expert.h` & `M` `src/ops/offloaded_sparse_moe/cuda/wide_expert.cu`:
  - Added `narrow_jobs` and `pass_narrow` workspace pointers (`ExpertsWorkspace`) to separate compacted narrow GPU jobs per pass.
  - Lowered per-token W4A4 quantization path enablement (`map_entries_kernel` + `quantize_token_kernel`) from `x.ne[1] > 64` to match `source.activation == ExpertActivation::kA4 && source.wide_a16 != WideA16::kLossless && entries == x.ne[1] * top_k`.
  - Fixed a warmup `cudaErrorIllegalAddress` crash under masked speculative verification (`verify_graph_body`, `batch > 1`): initialized `entry_map[0 .. entry_map_ints - 1] = -1` inside `plan_kernel` and guarded `quantize_token_kernel` with `if (expert >= 0)` so unpopulated masked slots never read uninitialized `0xA5A5A5A5` indices.
- `M` `src/ops/offloaded_sparse_moe/cuda/moe_layer.cu`: Added `gate_up_xplane_kernel` / `down_prefill_kernel` and compacted `narrow_jobs` / `pass_narrow` scheduling in `plan_kernel` alongside extended `moe_combine_inject` with optional fused next-layer RMSNorm output.
- `M` `src/ops/offloaded_sparse_moe/cuda/narrow_expert.cu`: Updated `quantize_x_kernel`, `gate_up_kernel` (`__launch_bounds__(kThreads, 2)`), and `down_kernel` (`__launch_bounds__(kThreads, 2)`) with PDL producer/consumer dependencies and vectorized SMEM staging.

### 3.3 `src/ops/hyper_connection/`
- `M` `src/ops/hyper_connection/hyper_connection.cu`: Extracted `hc_mix_from_normalized`, exported `hyper_connection_norm`, `hyper_connection_mix_is_fused`, `hyper_connection_mix_normalized`, and added PDL `pdl::launch_with` in `norm_vec8_kernel` for `columns <= 16`.
- `M` `src/ops/hyper_connection/hyper_connection_mix_fused.cu`: Added `Q4Codec` (`q4_g64_fp16` `row_split_k128_v1` decode via `Q4SimtDecodeAtom::decode_eight`) to `hc_mix_fused`, expanded `hc_up_collapse_v4_kernel` launch bounds for `Q8Codec`/`Q4Codec`, and widened fused Q8/Q4 mixer dispatch (`max_columns = kHcMixFusedMaxColumnsQ8`).
- `M` `src/ops/hyper_connection/hyper_connection_up_collapse.cu`: Added `launch_q4<Schedule>` using `launch_q4_a16_mma` with `HcStreamRows<Features>` and generalized `hc_up_collapse_supported` to accept `QType::Q4_G64_FP16` (`up.k % 64 == 0`).

### 3.4 `src/ops/qsa/`
- `M` `src/ops/qsa/qsa_prompt.cu`: Reintroduced compile-time `SplitD` (`Warps = 4` = 2 KV warps x 2 D-halves), shared-memory phase overlay (`phase1` query + pair scores + stages overlaying `merge` buffer), `64 B` bank-conflict-free XOR swizzle `(chunk ^ (row & 3) ^ ((row >> 2) & 1))`, `qsa_mma` helper, `cudaSharedmemCarveoutMaxShared` carveout configuration, and automatic `blocks_1 > blocks_2` stage-count / `SplitD` selection for non-VQ storages (`!Vq2 && !Q4KeyVq2Value`).

### 3.5 `src/ops/linear/{nvfp4,q4}/`
- `M` `src/ops/linear/nvfp4/nvfp4_codec.cuh`: Added `quantize_nvfp4_k16_packed` and used `fmaf(-code, scale, remainder)` in `quantize_nvfp4_terms_k16_packed`.
- `M` `src/ops/linear/nvfp4/nvfp4_a4_quantize.cuh`: Updated quantizer `__launch_bounds__(Threads, 1024 / Threads)` and vectorized/packed quantizer paths.
- `M` `src/ops/linear/nvfp4/nvfp4_a4.cu`: Raised `kGroupsPerRow == 384` thread block width in `launch_residual_exact` from `192` to `384`.
- `M` `src/ops/linear/nvfp4/nvfp4_a4_tma.cuh` & `M` `src/ops/linear/nvfp4/nvfp4_a4_tma.cu`: Applied per-operand TMA L2 promotions (`Schedule::kPromotions.a_codes`, `kTmaAllL2_256B` for `K=6144`), increased `Nvfp4A4TmaMmaSchedule` stage count (`3 -> 4`) on `K=6144` residual, balanced grouped TMA CTA raster (`max_group_n = 12` for `k <= 2560`), and moved `nvfp4_prepare_shared` inside the slice loop.
- `M` `src/ops/linear/nvfp4/shapes/{n12800_k2560,n13952_k2560,n16384_k2560,n2560_k6144}.cu`: Updated GB10 pair gates (`12800x2560`: `65 -> 56`; `13952x2560`: `33 -> 52`; `16384x2560`: `33 -> 52`) and updated `2560x6144` `select_a4` for `T = 129..256` and `T >= 2048`.
- `M` `src/ops/linear/q4/{q4_operands.h,q4_a16_mma.cuh,q4_mma_launch.cuh,q4_dispatch.cpp,q4_shapes.h,sources.cmake}`: Relaxed `K` divisibility requirement from `128` to `64` (`padded_k % 128 == 0`), generalized `q4_a16_mma_kernel` and `launch_q4_a16_mma` with `RowPolicy` (`Q4MmaIdentityRows` default and `HcStreamRows` for fused up-collapse) and `q4_mma_shared_bytes<Schedule, Epilogue>`, and added generic fallback + five GB10-swept Q4 shape files:
  - `A` `src/ops/linear/q4/shapes/n320_k10240.cu` (`select_q4_n320_k10240`)
  - `A` `src/ops/linear/q4/shapes/n324_k10240.cu` (`select_q4_n324_k10240`)
  - `A` `src/ops/linear/q4/shapes/n10240_k320.cu` (`select_q4_n10240_k320`)
  - `A` `src/ops/linear/q4/shapes/n1280_k2560.cu` (`select_q4_n1280_k2560`)
  - `A` `src/ops/linear/q4/shapes/n2560_k640.cu` (`select_q4_n2560_k640`)

### 3.6 `tests/` & `tools/`
- `M` `tests/ops/test_hyper_connection.cpp`: Added tests covering `hyper_connection_norm`, `hyper_connection_mix_normalized`, and `q4_g64_fp16` HC mix/up-collapse parity against oracle.
- `M` `tests/ops/test_offloaded_moe_layer.cu`: Updated `moe_combine_inject` tests to validate fused `next_norm_weight` / `next_norm_out` parity with `hyper_connection_norm`.
- `M` `tests/ops/test_offloaded_moe_wide.cu`: Added `test_masked_w4a4_verify` (`E=12, T=16, W=8, valid={3,6}, k=10`) with `0xA5` workspace poison and `0xFFFF` sentinel checks to verify masked `entry_map` safety.
- `M` `tests/ops/test_qsa.cpp`: Updated `test_prompt_variants` to allow attention-criterion floating-point tolerance (rather than bitwise identity) when comparing `SplitD` vs combined prompt kernels on non-VQ storages.
- `M` `tools/run_server.sh`: Added `local-f4` alias (`Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix-New`), default `SPEC_BACKEND=dflash2` when `DRAFT_PATH` exists, `--expert-a4-activations` toggle via `EXPERT_A4_DEFAULT`, `--lm-head-draft`, and `INFERNIX_NGRAM_WIDTH_COST=0.08`.

---

## 4. Verification & Benchmark Summary

### 4.1 Post-Edit Build & CTest Suite (`28/28` Passed)

All 28 affected unit/oracle CTests and the CUDA stack ratchet gate were rebuilt (`ninja -C build-native`) and executed after the final code edit (`Total Test time = 124.19 s`, `0` failed, `0` skipped):

- **Stack-spill ratchet:** `infernix_cuda_stack_check` (`2.60 s`) — `STACK:0`, `LOCAL:0` on `infernix-serve`.
- **Linear & NVFP4:** `infernix_nvfp4_tma_staging_race_test` (`8.11 s`), `infernix_linear_nvfp4_a16_test` (`13.66 s`), `infernix_linear_nvfp4_a4_test` (`22.65 s`), `infernix_linear_q4_a16_test` (`28.32 s`).
- **Hyper-Connection:** `infernix_hyper_connection_stream_mean_test` (`6.39 s`), `infernix_hyper_connection_test` (`30.86 s`).
- **QSA Prompt Attention:** `infernix_qsa_index_test` (`14.66 s`), `infernix_qsa_select_test` (`14.98 s`), `infernix_qsa_nvfp4_test` (`34.81 s`), `infernix_qsa_vq2_test` (`41.64 s`), `infernix_qsa_int8_test` (`41.91 s`), `infernix_qsa_k4v2_test` (`42.87 s`), `infernix_qsa_fp8_test` (`43.99 s`), `infernix_qsa_k8v4_test` (`46.37 s`), `infernix_qsa_bf16_test` (`46.70 s`).
- **Offloaded Sparse MoE:** `infernix_offloaded_moe_cuda_test` (`7.09 s`), `infernix_offloaded_moe_plan_test` (`14.85 s`), `infernix_offloaded_moe_qwen3_5_122b_test` (`15.91 s`), `infernix_offloaded_moe_wide_test` (`25.92 s`), `infernix_offloaded_moe_layer_test` (`112.77 s`), `infernix_offloaded_moe_layer_idle_test` (`11.40 s`).
- **Qwen4Exp Model/Vision/Speculative:** `infernix_qwen4_exp_vision_rg16_test` (`0.03 s`), `infernix_qwen4_exp_call_plan_test` (`0.06 s`), `infernix_qwen4_exp_expert_activations_test` (`0.01 s`), `infernix_qwen4_exp_speculative_test` (`0.02 s`), `infernix_qwen4_exp_vision_window_test` (`0.00 s`), `infernix_qwen4_exp_fast_paths_test` (`0.00 s`).

### 4.2 End-to-End `infernix-serve` A/B Results (`bench_client.py --max-tokens 256 --concurrency 1`, `prompt_tok=1225`, `cache 0.0%`)

| Configuration / Artifact | Client TTFT Mean (Runs 1–3) | Client Prefill Mean | Client Decode Mean (Runs 1–3) | Server Session Stats |
|---|---|---|---|---|
| `local-f4` baseline (`51fa3e2d7`) | `0.490s` | `2503.1 tok/s` | `73.7 tok/s` | — |
| `local-f4` post-campaign (`-New`, `w4dense` + Q8 HC) | `0.4633s` (**-5.4%**) | `2644.5 tok/s` (**+5.6%**) | `72.2 tok/s` (-2.0%) | `TTFT 440 ms`, `prefill 2.84k`, `DFLASH2 54.4%` (`1.44 acc/rnd`, `ngram 41.4%`) |
| `HC4` baseline (`51fa3e2d7`) | `0.554s` | `2209.4 tok/s` | `82.9 tok/s` | — |
| `HC4` pre-reconversion (`fp4_local` MXFP8 dense + Q4 HC) | `0.5207s` (-6.0%) | `2353.8 tok/s` (+6.5%) | `73.4 tok/s` (-11.5%) | `TTFT 501 ms`, `prefill 2.49k`, `decode 73.7`, `DFLASH2 57.5%` (`1.48 acc/rnd`) |
| **`HC4-W4Dense` post-reconversion (`w4dense` NVFP4 dense + Q4 HC)** | **`0.4637s`** (**-16.3% vs HC4 base**) | **`2641.7 tok/s`** (**+19.6% vs HC4 base**) | **`88.2 tok/s`** (**+6.4% vs HC4 base, +20.2% vs pre**) | `req#3–5`: `TTFT 437–443 ms`, `prefill 2.81–2.87k`, `decode 87.6–89.3`; 5-req session `DFLASH2 63.2%` (`1.89 acc/rnd`) |

- **Disclosed issues / findings:**
  1. During initial Step 4 startup testing, `infernix-serve warmup()` hit a `cudaErrorIllegalAddress` crash at `wide::prepare` due to uninitialized masked `entry_map` slots during W4A4 speculative verification (`verify_graph_body`, `batch > 1`). Fixed in `wide_expert.cu` (`plan_kernel` initializes `entry_map` to `-1`, `quantize_token_kernel` skips `expert < 0`) and locked down by `test_masked_w4a4_verify`.
  2. `local-f4` mean decode (`72.2 tok/s` vs `73.7` baseline) and pre-reconversion `HC4` decode (`73.4 tok/s` vs `82.9` baseline) measured lower on Runs 1–3 with active non-reproducible fast paths and speculative acceptance variance (`ngram 73.3%`, `acc/rnd 1.48`); reconverting `HC4` to `HC4-W4Dense` recovered and surpassed baseline decode (`88.2 tok/s`, `69.33 GiB` payload vs `71.11 GiB` pre-reconversion).
  3. End-to-end coherence on `HC4-W4Dense` was verified on `"What is 17 * 23? Explain briefly in one sentence."` (`"17 × 23 = 391."`), alongside the full 28-test CTest suite. Real-artifact integration CTests (`*_real_test`, including `infernix_qwen4_exp_forward_real_test`) were intentionally excluded per repository test constraints.

---

## 5. Rebase Conflict Resolution Policy

- **Prefill MoE & Masked Verify (`src/ops/offloaded_sparse_moe/cuda/`):** Preserve `entry_map` `-1` initialization in `plan_kernel` and `expert >= 0` guard in `quantize_token_kernel`, `narrow_jobs` / `pass_narrow` compaction, vectorized SMEM `narrow_expert.cu` with PDL, and optional fused next-layer norm in `moe_combine_inject`.
- **Hyper-Connection Q4/Q8 (`src/ops/hyper_connection/`):** Preserve `Q4Codec` fused decode mix, `hc_up_collapse` `Q4_G64_FP16` support, and pipelined `pre_norm` reuse across layers.
- **QSA Prompt (`src/ops/qsa/qsa_prompt.cu`):** Preserve `SplitD` compile-time specialization (`PromptShape` non-VQ storages) and dynamic shared-memory phase overlay for GB10 48-SM occupancy over upstream combined-only `qsa_prompt`.
- **Linear `/shapes/` & Vision (`src/ops/linear/{nvfp4,q4}/`, `src/runtime/engine/`):** Preserve GB10-swept Q4 shape files (`n320_k10240`, `n324_k10240`, `n10240_k320`, `n1280_k2560`, `n2560_k640`), updated `nvfp4` pair gates (`56`/`52`), and on-device vision support under `INFERNIX_EXPERTS_ON_GPU=1`.
