# NVFP4 Large-Token Prefill & Decoding Performance Optimization Plan (`Qwen3.8-Flash-Next-Local-NVFP4-W4Dense` on GB10 `sm_121a`)

**Date:** 2026-11-07
**Status:** Completed (`T1`–`T8` implemented, oracle-qualified, stack-checked, and benchmarked; see `docs/gb10/20261107-changelog.md`)
**Target Hardware:** NVIDIA DGX Spark GB10 (`sm_121a`, 48 SMs, 99 KiB SMEM/SM, 65,536 32-bit registers/SM, 24 MiB L2, 128 GiB unified LPDDR5X at ~273 GB/s)
**Target Artifact:** `/opt/etc/models/Qwen3.8-Flash-Next-F4-Local-NVPF4-Dense/Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix`
**Target Serving Configuration (`tools/run_server.sh`):**
- `INFERNIX_EXPERTS_ON_GPU=1` (`512` routed experts/layer × `48` layers = `24,576` experts, `68.0 GiB` resident in unified LPDDR5X)
- `--kv-dtype nvfp4` (`KvCacheStorage::Nvfp4Group16`, Hadamard-rotated `D=256` K/V with E2M1 codes and E4M3 group-16 scales)
- `--spec mtp --draft-tokens auto --ngram-draft-tokens 31 --ngram-min-match 4 --lm-head-draft`
- `--max-context 262144 --max-concurrency 4`

---

## 0. Context & Scope: Why a Dedicated Large-Token Campaign

The prior campaign (`docs/tasks/20261107-nvfp4-performance-tasks.md`, documented in `docs/gb10/20261107-changelog.md`) targeted narrow token counts (`T = 1..16`) and uncovered several critical architectural facts about `Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix`:

1. **Exact Model Geometry (`K = 2560`, 48 layers = 36 GDN + 12 QSA):**
   - **Routed MoE (`48x`):** `nvfp4_mul` (`nvfp4_expert_rg16_v1`), `512` experts, `top_k = 10`, `hidden = 2560`, `moe_intermediate = 640`, with `expert_input_scales` (`kA4` activation route). All `24,576 / 24,576` experts have `input_gate == input_up` (`split_input == false`, `src/ops/offloaded_sparse_moe/cuda/wide_expert.h:17-19`).
   - **Shared Expert (`48x`):** `q8_g32_fp16` (`[1280, 2560]` gate/up and `[2560, 640]` down).
   - **GDN Input Projection (`36x`):** `nvfp4_e2m1_g16_e4m3` `[16384, 2560]`, dispatched via `ops::gdn_input_proj` -> `launch_nvfp4_n16384_k2560_split` (`src/models/qwen4_exp/execution/forward.cpp:733`).
   - **QSA Input Projection (`12x`):** `nvfp4_e2m1_g16_e4m3` `[13952, 2560]`, dispatched via `ops::linear_qsa_group` -> `launch_nvfp4_n13952_k2560_split` (`src/models/qwen4_exp/execution/forward.cpp:935`).
   - **Mixer Output Projection (`48x`):** `nvfp4_e2m1_g16_e4m3` `[2560, 6144]`, dispatched via `ops::linear` -> `kNvfp4N2560K6144` (`src/models/qwen4_exp/execution/forward.cpp:781, 954`).
   - **PLE K/V Projection (`1x`, layer 1):** `nvfp4_e2m1_g16_e4m3` `[12800, 2560]`, dispatched via `ops::linear` -> `kNvfp4N12800K2560` (`src/models/qwen4_exp/execution/forward.cpp:667`).
   - **HyperConnection (`100x` up, `98x` down):** `q8_g32_fp16` `[10240, 320]` (`hc/up`) and `[324, 10240]` (`hc/down`).

2. **Two Distinct Large-Token Regimes Need Optimization:**
   - **Regime A — Wide Speculative Verification & Multi-Lane Decoding (`T = 9..128`, `W = 9..32`, Context `4K..256K`):**
     With `--ngram-draft-tokens 31`, `--spec mtp`, and `--max-concurrency 4`, verification steps process `W = 9..32` tokens per sequence (or `T = B * W` up to `128` columns across concurrent lanes). Several fused fast paths currently cut off at `T = 8` or `T = 16` (`kGdnReplayConvMaxWidth = 16`, narrow MoE `gate_up_kernel` 2-column loop, `hc_up_collapse` `T = 17..48` tile cliff, `qsa_prompt_kernel` `Stages = 1` stall at `T >= 24`, and `_split` NVFP4 projections missing `RowTile = 64` at `T = 65..128`).
   - **Regime B — Large-Prompt Prefill (`T = 256..32768+`):**
     At `pp4096..pp32768`, execution time is dominated by:
     1. Wide MoE `grouped_kernel<GateUp/Down>` (`471..1279 ms` at `pp8192..pp32768`) and `quantize_token_kernel` (`104..427 ms`), where `quantize_token_kernel` still runs `170` serial software `fdiv.rn.f32` divisions and `1,120` branches per 16-element block per token, and `map_work_unit<true>` runs two global-memory binary searches per work item on all 257 CTA threads.
     2. QSA prompt attention (`qsa_prompt_kernel`, `484 ms` at `pp32768`), where `KvCacheStorage::Nvfp4Group16` drops from double-buffered `Stages = 2` to unpipelined `Stages = 1` (`cp_wait<0>()`) for all `T >= 24`, even though two `Stages = 2` CTAs (`90 KiB` total) fit inside GB10's `99 KiB` SMEM.
     3. Server prefill chunking (`PREFILL_CHUNK=4096` in `tools/run_server.sh`) and MTP KV prefill chunking (`kMtpKvChunkTokens = 256` in `Forward::finish`), which multiply MoE weight sweeps and MTP launches on long prompts.

---

## 1. Mandatory Engineering & Verification Rules (`AGENTS.md`)

1. **Dead CUDA Depfiles for Header Edits (Mandatory):**
   Ninja's `.cu.o.d` depfiles in `build-native/` do not trigger rebuilds when `.h` / `.cuh` headers change. Before running `ninja` after editing any `.h` or `.cuh` file, **always** `touch` every `.cu` and `.cpp` translation unit that `#include`s the modified header.
2. **Single-Invocation Ninja Builds:**
   `infernix_ops` uses relocatable device code (`-rdc=true`), so `nvlink` takes ~55s. Always build all required targets (tests, microbenchmarks, `infernix_bench`, `infernix-serve`) in a **single** `ninja -C build-native -j "$(nproc)" ...` invocation.
3. **Numerical Qualification & Bit-Exactness:**
   - Every optimization on the default reproducible route (`--reproducible`) must be deterministic, request-invariant, and pass its independent FP32/FP64/exact oracle test (`ctest`).
   - Hardware `cvt.rn.satfinite.e2m1x2.f32` with `prepare_a4_scale` + `scaled_q_value` (4-ULP boundary guard around the seven E2M1 midpoints `{0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0}`) is 100% bit-identical to `canon::quantize_a4_codes` (`exact=yes`).
4. **One-Way CUDA Stack-Spill Ratchet Gate:**
   Run `ctest --test-dir build-native -R infernix_cuda_stack_check --output-on-failure` after any kernel change. Zero new `STACK > 0` or `LOCAL > 0` symbols are permitted.
5. **GB10 Changelog Accounting:**
   Record all completed tasks, hardware rationale, per-task A/B metrics, and any rejected approaches in `docs/gb10/20261107-changelog.md`.

---

## 2. Prioritized Optimization Tasks

### Task 1 (`T1`): Hardware `cvt.rn.satfinite.e2m1x2.f32` + Reciprocal Scaling Across All Wide & Narrow MoE A4 Quantization Paths

- **Priority:** P0 (Critical — Prefill `T > 64`, Wide Verification `T = 9..64`, and Narrow Decode/Verify `T = 1..8` on `Qwen3.8-Flash-Next`)
- **Affected Files & Symbols:**
  - [`src/ops/offloaded_sparse_moe/cuda/wide_expert.cu`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert.cu#L75-L208): `quantize_kernel` (lines 75–119) and `quantize_token_kernel` (lines 144–208)
  - [`src/ops/offloaded_sparse_moe/cuda/wide_expert.cuh`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert.cuh#L540-L592): `GateUpEpilogue::operator()` (lines 540–592)
  - [`src/ops/offloaded_sparse_moe/cuda/moe_layer.cu`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/moe_layer.cu#L1156-L1403): `quantize_column_codes` (lines 1156–1183), `quantize_pair_codes` (lines 1185–1223), and `quantize_a4_block_warp` (lines 1370–1403)
- **Root Cause & Hardware Rationale:**
  - In the previous campaign (`docs/gb10/20261107-changelog.md` T1), replacing `canon::quantize_a4_codes` (`17` serial IEEE-754 `fdiv.rn.f32` instructions + `112` branch comparisons per 16-element block) with `prepare_a4_scale` + `scaled_q_value` (`1` `fdiv.rn.f32` reciprocal + `16` `FMUL`s with exact 4-ULP midpoint fallback) + hardware `cvt.rn.satfinite.e2m1x2.f32` was applied **only** to `quantize_split_codes` and `quantize_split_pair_half` (`split_input == true`).
  - However, on `Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix`, **all 24,576 experts have `input_gate == input_up` (`split_input == false`)** (`wide_expert.h:17-19`). Consequently, the actual model never executes `quantize_split_codes` or `quantize_split_pair_half`!
  - Worse still, at large token counts (`T > 64`), `quantize_token_kernel` (`wide_expert.cu:144-208`, which consumes `104..427 ms` at `pp8192..pp32768`):
    1. Reloads `map[2 * entry]`, `map[2 * entry + 1]`, `dispatch.offsets[expert + 1] - dispatch.offsets[expert]`, and `scales[expert].input_gate` redundantly inside the `for (int m = 0; m < kPerLane; ++m)` loop (`5` times per token on every lane instead of once per token before the `m` loop).
    2. Unpacks `v[0..15]` and recomputes `amax` **inside** `for (int slot = 0; slot < top_k; ++slot)` (`10` times per block!) even though `v[0..15]` and `amax` depend only on `token_x` and `b`, not on `slot`.
    3. Executes `canon::a4_scale_word` (`1` `fdiv.rn.f32`) + `16` calls to `canon::a4_code` (`16` serial `fdiv.rn.f32` + `112` branches) per routed slot (`170` serial FP32 divisions and `1,120` branch comparisons per 16-element block per token).
  - Similarly, `quantize_kernel` (`T = 9..64`), `GateUpEpilogue` (every wide MoE GateUp pass at `T >= 9`), and `quantize_column_codes` / `quantize_pair_codes` / `quantize_a4_block_warp` (every narrow MoE job on `Qwen3.8-Flash-Next`) still use software `fdiv.rn.f32` + `canon::e2m1_rn_satfinite`.
- **Required Implementation:**
  1. Share `scaled_q_value` and `prepare_a4_scale` (currently in `moe_layer.cu:1228-1254`) in a common header (e.g., `src/ops/linear/nvfp4/nvfp4_codec.cuh` or `src/ops/offloaded_sparse_moe/cuda/wide_expert.h`) junto with a single-scale helper `quantize_a4_block_from_f32(const float (&v)[16], float amax, float input_scale, std::uint8_t& scale_word, std::int32_t& scale_scaled, uint2& packed_codes)` that uses `4` `cvt.rn.satfinite.e2m1x2.f32` instructions and `mov.b32` packing.
  2. In `quantize_token_kernel` (`wide_expert.cu:144-208`):
     - Before `for (int m = 0; m < kPerLane; ++m)`, have lanes `lane < top_k` load `entry = t * top_k + lane`, `p = map[2 * entry]`, `expert = map[2 * entry + 1]`, evaluate `wide_route(...)`, and load `input_scale = scales[expert].input_gate` once per token, forming `unsigned wide_mask = __ballot_sync(0xFFFFFFFFU, lane < top_k && is_wide)`.
     - Inside `for (int m = 0; m < kPerLane; ++m)`, unpack `float vf[16]` and compute `amax` **once** per block `b = m * 32 + lane`.
     - Iterate only over set bits of `wide_mask` (using `__ffs`), broadcasting `p_slot` and `scale_slot` via `__shfl_sync`, and quantizing with `quantize_a4_block_from_f32` (`1` reciprocal + `16` `FMUL`s + `4` `cvt.rn.satfinite.e2m1x2.f32`).
  3. Apply the same hardware `cvt.rn.satfinite.e2m1x2.f32` + reciprocal scaling to:
     - `quantize_kernel` (`wide_expert.cu:75-119`),
     - `GateUpEpilogue::operator()` (`wide_expert.cuh:553-589`, using `prepare_a4_scale` + `scaled_q_value` + `2` `cvt.rn.satfinite.e2m1x2.f32` instructions for the 4 rows per thread),
     - `quantize_column_codes`, `quantize_pair_codes`, and `quantize_a4_block_warp` (`moe_layer.cu:1156-1223, 1370-1403`).
- **Verification & Acceptance Criteria:**
  - Touch `src/ops/offloaded_sparse_moe/cuda/wide_expert.cu` and `src/ops/offloaded_sparse_moe/cuda/moe_layer.cu` before building.
  - Run `ctest --test-dir build-native -R "infernix_offloaded_sparse_moe_test|infernix_cuda_stack_check" --output-on-failure` (must remain bit-exact against the CPU reference oracle).
  - Benchmark wide and narrow MoE (`split=0` default, matching `Qwen3.8-Flash-Next`):
    ```bash
    ./build-native/bench/infernix_offloaded_moe_wide_bench --tokens 16,32,64,128,256,512,1024,4096 --placement frames --cold --warmup 5 --repeat 20
    ./build-native/bench/infernix_offloaded_moe_decode_bench --repeat 50
    ```

---

### Task 2 (`T2`): Split-Output NVFP4 Group Projections (`[13952, 2560]` QSA & `[16384, 2560]` GDN) — Add `r64` Schedule at `T = 65..128`, Pair Gate at `T = 33..64`, and Programmatic PDL Across NVFP4 TMA Launches

- **Priority:** P0 (Critical — Medium/Large Batches `T = 33..4096+` across all 48 layers)
- **Affected Files & Symbols:**
  - [`src/ops/linear/nvfp4/shapes/n13952_k2560.cu`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n13952_k2560.cu#L152-L176): `launch_nvfp4_n13952_k2560_split`
  - [`src/ops/linear/nvfp4/shapes/n16384_k2560.cu`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n16384_k2560.cu#L140-L165): `launch_nvfp4_n16384_k2560_split`
  - [`src/ops/linear/nvfp4/nvfp4_launch.cuh`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_launch.cuh#L187-L333): `launch_nvfp4_a4_tma_linear`, `launch_nvfp4_a4_tma_linear_narrow`, `launch_nvfp4_a4_residual_tma_linear`, `launch_nvfp4_a4_residual_tma_linear_r64`
- **Root Cause & Hardware Rationale:**
  - On `Qwen3.8-Flash-Next`, the 12 QSA layers call `ops::linear_qsa_group` -> `launch_nvfp4_n13952_k2560_split` (`forward.cpp:935`) and the 36 GDN layers call `ops::gdn_input_proj` -> `launch_nvfp4_n16384_k2560_split` (`forward.cpp:733`), **not** `kNvfp4N13952K2560` / `kNvfp4N16384K2560`.
  - However, when `kNvfp4N13952K2560` and `kNvfp4N16384K2560` were tuned on GB10 to use `RowTile = 64` (`launch_nvfp4_a4_pair_r64`) for `65 <= T <= 128` and `nvfp4_a4_pair_33_64()` for `33 <= T <= 64`, the `_split` functions in the same files were left behind:
    - `launch_nvfp4_n13952_k2560_split` (`n13952_k2560.cu:167-173`) and `launch_nvfp4_n16384_k2560_split` (`n16384_k2560.cu:156-162`) hardcode `Nvfp4A4TmaMmaSchedule<128, 3, 1, ..., 128>` (`RowTile = 128`) for all `T >= 65`. At `T = 65..128`, `13952 / 128 = 109` CTAs has a 32%-filled 3rd wave on 48 SMs, whereas `RowTile = 64` (`218` CTAs, 4.54 waves) is ~9% faster (`105.0 us` vs `115.0 us`).
  - Furthermore, in both `_split` functions and in `nvfp4_launch.cuh` (`lines 187-333`), `launch_nvfp4_a4_quantize` is immediately followed by `launch_nvfp4_a4_tma_mma` with the default `pdl::Dependency::Serialized` instead of `pdl::Dependency::Programmatic`. Because `nvfp4_a4_tma_kernel` (`nvfp4_a4_tma.cuh:140-152`) initializes mbarriers and acquires TMA tensor maps before calling `pdl::wait_for_dependencies()` (`line 152`), passing `pdl::Dependency::Programmatic` is already supported by the kernel and overlaps CTA startup + mbarrier init with the tail of `nvfp4_a4_quantize_kernel` across all 96 NVFP4 TMA linear calls per forward pass (`36` GDN input + `12` QSA input + `48` mixer output + `1` PLE).
- **Required Implementation:**
  1. In `launch_nvfp4_n13952_k2560_split` (`n13952_k2560.cu:160-175`) and `launch_nvfp4_n16384_k2560_split` (`n16384_k2560.cu:148-164`):
     - Gate the `33 <= T <= 64` TMA branch on `nvfp4_a4_pair_33_64()` (matching `kNvfp4N13952K2560` and `kNvfp4N16384K2560`).
     - Dispatch `Nvfp4A4TmaMmaSchedule<64, 3, 1, 16, 4, 2, 128, 64>` for `65 <= T <= 128`, and `Nvfp4A4TmaMmaSchedule<128, 3, 1, 16, 4, 2, 128, 128>` for `T > 128`.
     - Pass `pdl::Dependency::Programmatic` to `launch_nvfp4_a4_tma_mma`.
  2. In `src/ops/linear/nvfp4/nvfp4_launch.cuh` (`lines 187-333`), pass `pdl::Dependency::Programmatic` to `launch_nvfp4_a4_tma_mma` in `launch_nvfp4_a4_tma_linear`, `launch_nvfp4_a4_tma_linear_narrow`, `launch_nvfp4_a4_residual_tma_linear`, and `launch_nvfp4_a4_residual_tma_linear_r64`.
- **Verification & Acceptance Criteria:**
  - Touch all `.cu` files that include `nvfp4_launch.cuh` (`src/ops/linear/nvfp4/shapes/*.cu`, `src/ops/linear/nvfp4/nvfp4_a4_tma.cu`, `src/ops/attn_input_proj/nvfp4_attn_input_fused_rmsnorm.cu`, `src/ops/gdn_input_proj/nvfp4_gdn_input_fused_rmsnorm.cu`).
  - Run `ctest --test-dir build-native -R "infernix_linear_nvfp4|infernix_gdn_input_proj|infernix_attn_input_proj|infernix_cuda_stack_check" --output-on-failure`.
  - Benchmark GDN input proj and NVFP4 linear shapes across `T = 16..4096`:
    ```bash
    ./build-native/bench/infernix_gdn_input_proj_bench --format nvfp4 --nvfp4-policy a4 --tokens 16,32,64,96,128,256,512,1024,4096 --cache cold --repeat 30
    ./build-native/bench/infernix_linear_bench --qtype NVFP4 --policy a4 --n 2560 --k 6144 --sweep 64:512:64 --repeat 30
    ```

---

### Task 3 (`T3`): Wide MoE `grouped_kernel` $O(1)$ Work-Unit Mapping (`map_work_unit`) & Unstaged Direct Record Resolution

- **Priority:** P1 (High — Prefill `T = 65..32768+` and Wide Verification `T = 9..64`)
- **Affected Files & Symbols:**
  - [`src/ops/offloaded_sparse_moe/cuda/wide_expert.cu`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert.cu#L34-L71): `plan_kernel` (lines 34–71) and `prepare` (lines 779–889)
  - [`src/ops/offloaded_sparse_moe/cuda/wide_expert.cuh`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert.cuh#L241-L266): `map_work_unit` (lines 241–266) and `grouped_kernel` (lines 335–395)
  - [`src/ops/offloaded_sparse_moe/cuda/moe_layer.cu`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/moe_layer.cu#L4242-L4260): `moe_experts` (lines 4242–4260)
- **Root Cause & Hardware Rationale:**
  1. **Two Global-Memory Binary Searches Per Work Unit in `map_work_unit<true>` (`wide_expert.cuh:242-266`):**
     On `Matrix::GateUp` (`kColTileInner == true`), every work unit `w` in `grouped_kernel` runs **two binary searches** over `call.tiles[2 * mid]` in global memory inside **every thread** (the producer thread and all 256 consumer threads) solely to determine the owning job's first tile index `first` and tile count `n = lo - first`. At `pp8192..pp32768` (`~512..6,400` tiles), each work unit executes `~20–26` dependent `__ldg` loads across 257 threads.
     Yet in `plan_kernel` (`wide_expert.cu:59-63`), the thread writing `tiles[2 * (first + i)]` **already has `first`, `i`, and `n` in registers**! Because `probe = tile_begin + w / kRowBlocks = first + i` (where `0 <= i < n` and `column = i * kTileColumns`), `first` is simply `probe - i`! If `tiles[2 * (first + i)]` packs `job | (n << 16)` (since `job <= 512` and `n <= 4096` fit in 16-bit unsigned halves), then a single `int2` read of `(tiles[2 * probe], tiles[2 * probe + 1])` yields `job = raw.x & 0xFFFF`, `n = raw.x >> 16`, `i = raw.y / kTileColumns`, and `first = probe - i` in **$O(1)$ time with zero binary searches**! (Update all readers of `tiles[2 * t]` to mask `& 0xFFFF`, or store `first` and `n` explicitly).
  2. **Redundant `stage_kernel<<<1, 256>>>` Launch on Every Wide Layer (`moe_layer.cu:4246-4256`):**
     When `INFERNIX_EXPERTS_ON_GPU=1` (`source.staging_slots == 0 && source.fetch.request == nullptr`), `unstaged_direct` in `moe_layer.cu:4246` is disabled whenever `use_wide` is true (`&& !use_wide`) only because `grouped_kernel` reads `call.job_records[job]`. Having `plan_kernel` populate `job_records[job] = source.frames + static_cast<std::size_t>(source.slot_of[expert]) * pitch` directly when `source.staging_slots == 0 && source.fetch.request == nullptr` (and `flags == nullptr`) eliminates 48 `stage_kernel` launches per forward step for all `T > 8`.
- **Required Implementation:**
  1. Encode the job's tile count `n` alongside `job` in `plan_kernel` (or in `tiles`) so `map_work_unit<true>` computes `first = probe - (tiles[2 * probe + 1] / kTileColumns)` and `span = n` via a single `__ldg` read at `probe` with zero binary-search loops. Ensure all readers of `tiles[2 * tile]` (`quantize_kernel`, `encode_h_kernel`, `quantize_x_terms_kernel`, `grouped_kernel`, etc.) extract `job = tiles[2 * tile] & 0xFFFF`.
  2. Pass `source` / `job_records` into `plan_kernel` when `unstaged_direct` applies so `plan_kernel` writes `job_records[job]` directly and `moe_experts` skips launching `stage_kernel` when `source.staging_slots == 0 && source.fetch.request == nullptr && !ggml && source.ggml.gate_up_format != QType::FP8_E4M3FN_B128`.
- **Verification & Acceptance Criteria:**
  - Touch all `.cu` files including `wide_expert.h` / `wide_expert.cuh` (`wide_expert.cu`, `moe_layer.cu`).
  - Run `ctest --test-dir build-native -R "infernix_offloaded_sparse_moe_test|infernix_cuda_stack_check" --output-on-failure`.
  - Benchmark `infernix_offloaded_moe_wide_bench` at `T = 16, 64, 256, 1024, 4096` (`--placement frames --cold`).

---

### Task 4 (`T4`): Extend Fused GDN Verification (`gated_delta_net_replay_record_conv`) from `kGdnReplayConvMaxWidth = 16` to `32`

- **Priority:** P1 (High — Speculative Decoding Verification `W = 17..32` with `--ngram-draft-tokens 31`)
- **Affected Files & Symbols:**
  - [`include/infernix/ops/gated_delta_net.h`](file:///opt/etc/Infernix/include/infernix/ops/gated_delta_net.h#L135-L155): `kGdnReplayConvMaxWidth` (line 136)
  - [`src/ops/linear_attention/gated_delta_net/common.h`](file:///opt/etc/Infernix/src/ops/linear_attention/gated_delta_net/common.h#L15-L20): `kRecordConvMaxWidth`
  - [`src/ops/linear_attention/gated_delta_net/recurrent.cuh`](file:///opt/etc/Infernix/src/ops/linear_attention/gated_delta_net/recurrent.cuh#L513-L566): `RecordConvShared`, `RecordConvSharedAccess`, and `recurrent_record_conv_kernel` (lines 880–942)
  - [`src/ops/linear_attention/gated_delta_net/recurrent.cu`](file:///opt/etc/Infernix/src/ops/linear_attention/gated_delta_net/recurrent.cu#L239-L292): `launch_recurrent_record_conv`
- **Root Cause & Hardware Rationale:**
  - `tools/run_server.sh` configures `--ngram-draft-tokens 31`, which produces verification batches up to `W = 32` columns.
  - However, `kGdnReplayConvMaxWidth` is currently `16`. Whenever `batch.width` is `17..32`, `Forward::gdn` (`src/models/qwen4_exp/execution/forward.cpp:744-769`) falls back from the single fused kernel `ops::gated_delta_net_replay_record_conv` to **three separate kernels per GDN layer** (`ops::causal_conv1d_silu_from_states_split`, `ops::gdn_gating_direct`/`ops::gdn_gating`, and `ops::gated_delta_net_replay_record`), launching **72 extra kernels** and round-tripping `q, k, v, g, beta` through global memory across the 36 GDN layers.
  - Inside `recurrent_record_conv_kernel` (`recurrent.cuh:931-938`), the gate-evaluation warp (`warp 3`, 32 lanes) **already supports `width <= 32`** (`gate_column = thread - 96 < width`). At `MaxWidth = 32`, `RecordConvShared<32>` uses `17,664 B` (`17.25 KiB`) of shared memory (`5 * 17.25 KiB = 86.25 KiB <= 99 KiB` SMEM on `sm_121a`).
- **Required Implementation:**
  1. Raise `kGdnReplayConvMaxWidth` (`gated_delta_net.h:136`) and `kRecordConvMaxWidth` (`common.h:17`) from `16` to `32`.
  2. Template `RecordConvShared<int MaxWidth>`, `RecordConvSharedAccess<State, int MaxWidth>`, and `recurrent_record_conv_kernel<bool ADirect, class State, int MaxWidth>` so `launch_recurrent_record_conv` dispatches `MaxWidth = 16` for `width <= 16` (preserving the exact `8,832 B` SMEM footprint for `W = 2..16`) and `MaxWidth = 32` for `17 <= width <= 32`.
- **Verification & Acceptance Criteria:**
  - Touch `src/ops/linear_attention/gated_delta_net/recurrent.cu`, `src/ops/linear_attention/gated_delta_net/replay.cpp`, and `src/models/qwen4_exp/execution/forward.cpp`.
  - Extend `tests/ops/gated_delta_net_replay_record_test.cpp` (or equivalent GDN replay test) to verify `gated_delta_net_replay_record_conv` at `width = 17..32` bit-for-bit against the unfused composition.
  - Run `ctest --test-dir build-native -R "gated_delta_net|gdn_replay|infernix_cuda_stack_check" --output-on-failure`.

---

### Task 5 (`T5`): QSA Prompt Attention (`qsa_prompt.cu`) — Double-Buffered 2-CTA/SM Schedule for `KvCacheStorage::Nvfp4Group16` & Programmatic PDL Launch

- **Priority:** P1 (High — Prefill `T = 24..16384+` and Wide Verification `T = 9..32` with `--kv-dtype nvfp4`)
- **Affected Files & Symbols:**
  - [`src/ops/qsa/qsa_prompt.cu`](file:///opt/etc/Infernix/src/ops/qsa/qsa_prompt.cu#L292-L320): `qsa_prompt_kernel` (lines 292–320, 874–903) and `launch` (lines 996–1038)
- **Root Cause & Hardware Rationale:**
  - When `columns >= 9` on 48 SMs, `qsa_attention` (`src/ops/qsa/qsa.cu:1439-1445`) dispatches `qsa_prompt_attention` (`qsa_prompt_kernel`). At `pp32768`, `qsa_prompt_kernel` takes `484 ms`.
  - For `--kv-dtype nvfp4` (`KvCacheStorage::Nvfp4Group16`), each warp stage is only `S::Bytes = 4,608 B` (`16` tokens × `(128 B K + 128 B V + 16 B K-scale + 16 B V-scale)`).
  - Even with **`Stages = 2` (double-buffered `cp.async`)**, `prompt_smem_bytes<KvCacheStorage::Nvfp4Group16, 2>()` is only `8,192 + 4 * 2 * 4,608 = 45,056 B` (`44 KiB`), so **two CTAs per SM (`2 * (45,056 + 1,024) = 92,160 B <= 100 * 1024`) fit with `Stages = 2`**!
  - Yet `qsa_prompt_kernel` (`line 293`) sets `__launch_bounds__(Warps * 32, Stages == 1 ? 2 : 1)` (`1` CTA/SM for `Stages == 2`), and `launch()` (`line 1031`) forces `Stages = 1` whenever `columns * kv_heads >= 48` (`columns >= 24`). Because `Stage<Nvfp4Group16>::SplitGather` is `false` (`K::Bytes = 128 < 256`), `Stages = 1` executes the unpipelined loop (`lines 874-885`: `issue(i, 0); cp_wait<0>();`), stalling every warp synchronously on `cp_wait<0>()` on every 16-token KV tile!
  - Additionally, `qsa_prompt_kernel` (`line 1008`) is launched with a plain `<<<...>>>` instead of `pdl::launch_consumer` + `pdl::enter()`, failing to overlap CTA launch with `qsa_select`'s `pdl::trigger_dependents()`.
- **Required Implementation:**
  1. In `src/ops/qsa/qsa_prompt.cu`:
     - Define `constexpr int prompt_min_ctas_per_sm() { return 2 * (prompt_smem_bytes<Storage, Stages>() + 1024) <= 100 * 1024 ? 2 : 1; }` and use `__launch_bounds__(PromptShape<Storage>::Warps * 32, prompt_min_ctas_per_sm<Storage, Stages>())` on `qsa_prompt_kernel`.
     - In `launch()` (`lines 1030-1035`), if `2 * (prompt_smem_bytes<Storage, 2>() + 1024) <= 100 * 1024` (true for `KvCacheStorage::Nvfp4Group16`), keep `Stages = 2` at all column widths so `Nvfp4Group16` gets **both** 2 CTAs/SM **and** double-buffered `cp.async` (`cp_wait<1>()`) at `columns >= 24`!
     - Launch `qsa_prompt_kernel` via `pdl::launch_consumer` and call `pdl::enter()` at kernel entry (before global memory reads that depend on `selected`/`counts`, i.e., after computing thread/CTA coordinates and before reading `counts[t]`).
- **Verification & Acceptance Criteria:**
  - Run `ctest --test-dir build-native -R "qsa|infernix_cuda_stack_check" --output-on-failure`.
  - Benchmark `infernix_qsa_attention_bench` on `nvfp4` storage across verification and prefill widths:
    ```bash
    ./build-native/bench/infernix_qsa_attention_bench --storage nvfp4 --contexts 8192,32768,131072 --widths 9,16,24,32,64,256,1024 --warmup 5 --repeat 20
    ```

---

### Task 6 (`T6`): Narrow MoE `gate_up_kernel` 4-Column Quad Loop for Multi-Token Verification (`count = 3..8`)

- **Priority:** P2 (Medium — Multi-Token Speculative Verification `T = 8..64` and Multi-Lane Decode)
- **Affected Files & Symbols:**
  - [`src/ops/offloaded_sparse_moe/cuda/moe_layer.cu`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/moe_layer.cu#L1434-L1585): `gate_up_kernel` (lines 1434–1585)
- **Root Cause & Hardware Rationale:**
  - Routed experts with `count <= 8` (`kMaxColumns = 8`) execute on the narrow MoE kernels (`gate_up_kernel` and `down_kernel`). During speculative verification (`W = 9..32`), multi-lane decode (`B * W = 8..32`), and medium prefills (`T = 32..256`), many active experts receive `count = 3..8` columns.
  - While `down_kernel` (`moe_layer.cu:1719-1750`) processes `3..4` columns per pass via `unit_sums_down_quad_pair_groups` (`for (; c + 2 < count; c += 4)`), `gate_up_kernel` (`moe_layer.cu:1460-1494`) only processes `2` columns per pass (`for (; c + 1 < count; c += 2)`), scanning the 46 KiB staged weight slice in `sm.stage` up to 4 times instead of 2 times.
- **Required Implementation:**
  - Add a 4-column quad pass (`for (; c + 2 < count; c += 4)`) to the `!split_input` branch of `gate_up_kernel` (which covers all 24,576 experts of `Qwen3.8-Flash-Next`), reusing `DownActCodes` / `unit_sums_down_quad_pair_groups` (over `moe::kGateUpBlocks` and `kGateUpGroups = 2`) and `GateUpShared` union storage (without increasing shared memory beyond the 2-CTA/SM limit `sizeof(GateUpShared) + 1024 <= 51,200 B`).
- **Verification & Acceptance Criteria:**
  - Run `ctest --test-dir build-native -R "infernix_offloaded_sparse_moe_test|infernix_cuda_stack_check" --output-on-failure`.
  - Benchmark `infernix_offloaded_moe_decode_bench --repeat 50 --tokens 8` and `infernix_offloaded_moe_wide_bench --tokens 8,16,32,64 --placement frames --cold`.

---

### Task 7 (`T7`): HyperConnection Fused `hc_up_collapse` Tile Schedule Alignment for `T = 17..48`

- **Priority:** P2 (Medium — Wide Verification `T = 17..32` and Medium Batches `T = 17..48`, 100 calls/step)
- **Affected Files & Symbols:**
  - [`src/ops/hyper_connection/hyper_connection_up_collapse.cu`](file:///opt/etc/Infernix/src/ops/hyper_connection/hyper_connection_up_collapse.cu#L130-L157): `hc_up_collapse` (lines 130–157)
- **Root Cause & Hardware Rationale:**
  - In `src/ops/hyper_connection/hyper_connection_up_collapse.cu:140-145`, the comment explicitly states that for `hc/up` `[10240, 320]`, the 32-row `K64` tile (`MmaR32T16` / `MmaR32T24`, unpredicated `K = 320 = 5 * 64` at 4 CTAs/SM) is the argmin across `T = 9..32`, whereas `MmaR64x16T48K128A1` (`K128`, predicated `K` since `320 % 128 != 0`) drops to ~180 GB/s and loses even to the unfused linear + collapse pair.
  - However, line 144 only dispatches `MmaR32T16` for `T <= 16`, leaving `T = 17..48` on `MmaR64x16T48K128A1` (whereas `select_q8_n10240_k320` in `src/ops/linear/q8/shapes/n10240_k320.cu:30-33` dispatches `MmaR32T24` at `17..24`, `MmaR32T16` at `25..32`, and `MmaR32T24` at `33..48`).
- **Required Implementation:**
  - Update `hc_up_collapse` (`src/ops/hyper_connection/hyper_connection_up_collapse.cu:144-146`) to match `select_q8_n10240_k320` across `T = 17..48` (testing `MmaR32T24` and `MmaR32T16` at `T = 17..48` via `infernix_hyper_connection_bench`).
- **Verification & Acceptance Criteria:**
  - Run `ctest --test-dir build-native -R "hyper_connection|infernix_cuda_stack_check" --output-on-failure`.
  - Benchmark `infernix_hyper_connection_bench`:
    ```bash
    ./build-native/bench/infernix_hyper_connection_bench --columns 4,8,12,16,20,24,28,32,40,48,64,128 --codec q8 --warmup 10 --repeat 50
    ```

---

### Task 8 (`T8`): Server Prefill Chunk Sizing (`tools/run_server.sh`) & MTP Prefill KV Chunking (`Forward::finish`)

- **Priority:** P1 (High — End-to-End Large-Prompt Prefill `pp8192..pp32768+`)
- **Affected Files & Symbols:**
  - [`tools/run_server.sh`](file:///opt/etc/Infernix/tools/run_server.sh#L170-L185): `PREFILL_CHUNK` default (line 174)
  - [`src/models/qwen4_exp/execution/forward.cpp`](file:///opt/etc/Infernix/src/models/qwen4_exp/execution/forward.cpp#L47-L50): `kMtpKvChunkTokens` (line 50) and `Forward::finish` (lines 1505–1511)
- **Root Cause & Hardware Rationale:**
  1. **`PREFILL_CHUNK=4096` in `tools/run_server.sh`:**
     `tools/run_server.sh` line 174 hardcodes `PREFILL_CHUNK=4096` (`--prefill-chunk 4096`), overriding the model's native `16384` default (`docs/cli.md:107`). Because all 48 MoE layers stream up to `68 GiB` of expert weights per chunk (`~0.264 s` DRAM floor per chunk at `273 GB/s`), splitting a 16K or 32K prompt into 4,096-token chunks forces 4x as many full-bank weight reads. Prior GB10 measurements (`docs/tasks/20261102-prefill-ceiling-tasks.md`) showed `--prefill-chunk 4096` costs **-18.9% prefill throughput at `pp32768`** (`991.8 tok/s` vs `1,222.5 tok/s` at `--prefill-chunk 16384`) and **-10.0% at `pp8192`**.
  2. **`kMtpKvChunkTokens = 256` in `Forward::finish` (`forward.cpp:50, 1505-1511`):**
     When `--spec mtp` is enabled (the default in `tools/run_server.sh`), `Forward::finish` populates the MTP KV cache during prefill by looping `mtp_block` in chunks of `kMtpKvChunkTokens = 256` tokens (`64` serial MTP passes per `16,384`-token prefill chunk!). Each MTP pass launches its own projections, attention, and resident MoE (`[655360, 2560]` + `[1310720, 640]`). Raising `kMtpKvChunkTokens` from `256` to `512` or `1024` (and sizing workspace accordingly in `Forward::workspace_capacity_bytes`) halves or quarters MTP prefill launch and MoE weight-streaming overhead.
- **Required Implementation:**
  1. Update `PREFILL_CHUNK` in `tools/run_server.sh` (line 174) from `4096` to `16384` (still overridable via CLI `--prefill-chunk` or environment variable `INFERNIX_PREFILL_CHUNK`).
  2. Evaluate raising `kMtpKvChunkTokens` in `src/models/qwen4_exp/execution/forward.cpp:50` from `256` to `512` (or `1024`), verifying transient workspace delta in `Forward::workspace_capacity_bytes` (`forward.cpp:360-369`) and end-to-end prefill speed on `pp4096..pp16384` with `--spec mtp`.
- **Verification & Acceptance Criteria:**
  - Run `ctest --test-dir build-native -R "qwen4_exp|infernix_cuda_stack_check" --output-on-failure`.
  - Benchmark end-to-end prefill and decode on `Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix` using `infernix_bench`:
    ```bash
    INFERNIX_EXPERTS_ON_GPU=1 ./build-native/bench/infernix_bench \
      --model /opt/etc/models/Qwen3.8-Flash-Next-F4-Local-NVPF4-Dense/Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix \
      --kv-dtype nvfp4 --prompt-tokens 4096 --gen-tokens 128 --warmup 1 --runs 3
    ```

---

## 3. Recommended Execution Order & Unified Build Workflow

Execute the tasks in two phases to amortize the ~55s `nvlink` device link:

1. **Phase 1 — Core Kernel & Operator Optimizations (`T1`–`T7`):**
   - Implement `T1` (Wide/Narrow MoE A4 hardware quantization), `T2` (`_split` NVFP4 `r64` + PDL), `T3` (Wide MoE $O(1)$ `map_work_unit` + direct records), `T4` (GDN `kGdnReplayConvMaxWidth = 32`), `T5` (QSA `Nvfp4Group16` 2-CTA `Stages = 2` + PDL), `T6` (Narrow MoE `gate_up_kernel` quad loop), and `T7` (`hc_up_collapse` `T = 17..48` tiles).
   - Touch all dependent `.cu` files and run a single unified build:
     ```bash
     ninja -C build-native -j "$(nproc)" \
       infernix_offloaded_sparse_moe_test infernix_offloaded_moe_wide_bench infernix_offloaded_moe_decode_bench \
       infernix_linear_nvfp4_a4_test infernix_gdn_input_proj_test infernix_gdn_input_proj_bench infernix_linear_bench \
       infernix_gated_delta_net_replay_record_test infernix_gdn_replay_bench \
       infernix_qsa_test infernix_qsa_attention_bench \
       infernix_hyper_connection_test infernix_hyper_connection_bench \
       infernix_bench infernix-serve
     ```
   - Run all oracle tests and the stack-spill ratchet gate:
     ```bash
     ctest --test-dir build-native -R "offloaded_sparse_moe|linear_nvfp4|gdn_input_proj|gated_delta_net|qsa|hyper_connection|infernix_cuda_stack_check" --output-on-failure
     ```
2. **Phase 2 — End-to-End Prefill & Speculative Decoding Qualification (`T8` + Model A/B):**
   - Implement `T8` (`tools/run_server.sh` and `kMtpKvChunkTokens`), run `infernix_bench` across `pp512`, `pp4096`, `pp8192`, `pp16384` and decoding with `--kv-dtype nvfp4 --spec mtp --ngram-draft-tokens 31`, and record full before/after tables in `docs/gb10/20261107-changelog.md`.
