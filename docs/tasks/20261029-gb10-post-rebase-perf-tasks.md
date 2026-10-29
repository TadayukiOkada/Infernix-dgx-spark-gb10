# GB10 (`sm_121a`) Post-Rebase Performance Improvement Tasks (2026-10-29)

This document defines the next actionable batch of performance improvement tasks on **NVIDIA GB10 (`sm_121a`, DGX Spark)** following the `origin/main` (`d8e52a85a`) rebase and full 379-test verification.

---

## 1. Hardware & Baseline Context

### 1.1 Measured GB10 Hardware Ceilings (`.ckt_cache/env.json`)
| Hardware Resource / Execution Pipe | Measured GB10 Ceiling | Architectural Contrast vs. RTX 5090 (`sm_120a`) |
|---|---:|---|
| **SM Count** | **48 SMs** (max 3003 MHz) | `3.54×` fewer SMs than RTX 5090 (`170 SMs`); wave quantum is `48` / `96` CTAs |
| **L2 Cache** | **24 MiB** (`25,165,824 B`) | `4×` smaller than RTX 5090 (`96 MiB`); tensors `> 16 MiB` require 2D `Grouped, 8` CTA swizzling |
| **Unified Memory Read Bandwidth** | **247.6 GB/s** (`273.0 GB/s` spec) | ~121 GiB coherent LPDDR5X (`cudaDevAttrIntegrated = 1`); CPU and GPU share one physical pool |
| **BF16 / FP16 Tensor Core (`HMMA.16816`)** | **125.4 TFLOP/s** (`97.0–100.1 TF` GEMM) | **1:1 throughput parity** between FP32-accumulating (`125.31 TF`) and FP16-accumulating (`125.44 TF`) HMMA |
| **FP8 / MXFP8 Tensor Core (`QMMA.16832`)** | **245.9 TFLOP/s** (`125–160 TF` GEMM) | Block-scaled `m16n8k32` E4M3 |
| **INT8 Tensor Core (`IMMA.16832.S8`)** | **251.2 TOP/s** | `2×` the throughput of `HMMA.16816` (`W8A8` / `W3A8` / `W4A8`) |
| **NVFP4 Tensor Core (`OMMA.16864.E2M1`)** | **501.6 TFLOP/s** (`440.0 TF` dense ref) | `315–350 TFLOP/s` achieved in 1-term `A4` TMA; `245–348 TFLOP/s` hardware (`122–174 TF` useful) in 2-term `Nvfp4A4Pair` |
| **CPU Complex** | **20-core Grace ARMv9.2** | `10×` Cortex-X925 performance cores + `10×` Cortex-A725 efficiency cores |

### 1.2 Current Post-Rebase End-to-End Baselines (`Qwen3.8-Flash-Next`, `chunk = 4096`)
- **`NVFP4-W4Dense` (`fp8` KV, MTP)**: `pp4096 = 3,236.8 tok/s`; `pp4096+tg128 = 3,174.8 tok/s` prefill / `92.8–96.9 tok/s` decode (`acc = 0.949`); `pp4096+tg256 = 3,206.8 tok/s` prefill / `98.9–103.2 tok/s` decode (`acc = 0.920`).
- **`NVFP4-W4Dense` (`bf16` KV, MTP)**: `pp4096 = 3,238.9 tok/s`; `pp4096+tg128 = 97.7 tok/s` decode; `pp4096+tg256 = 101.6 tok/s` decode.
- **`NVFP4-W4Dense` (`nvfp4` KV, Plain)**: `pp4096 = 3,371.4 tok/s`; `tg128 = 46.2 tok/s` decode.
- **`Dense8` (`bf16` KV, MTP)**: `pp4096 = 2,866.2 tok/s`; `pp4096+tg128 = 88.2 tok/s` decode.
- **`IQ3_S` (`bf16` KV, MTP)**: `pp4096 = 1,045.0 tok/s`; `pp4096+tg128 = 98.9 tok/s` decode.
- **Multi-Concurrency (`NVFP4-W4Dense`, `pp2048+tg128`, `fp8` KV, MTP)**:
  - `c = 1`: `93.65 tok/s` (`2,687.6 tok/s` prefill)
  - `c = 2`: `62.49 tok/s/stream` (`124.98 tok/s` aggregate)
  - `c = 3`: `50.41 tok/s/stream` (`151.23 tok/s` aggregate)
  - `c = 4`: `44.86 tok/s/stream` (`179.44 tok/s` aggregate)

---

## 2. Prioritized Task Overview

| Task ID | Priority | Subsystem / Scope | Target Bottleneck | Expected Impact |
|---|---|---|---|---|
| **T1** | **P0** | `MXFP8` Linear Shapes & `F16AccumulatePrefill` Fast Path (`qwen4_exp` `FP4-Local`) | Uncommitted GB10 `mxfp8` `launch_a16` schedules + 5090-only `F16AccumulatePrefill` overriding `Fp8A8PairBlock32` (`QMMA.16832`) | `+15%` to `+89%` op prefill (`1.153×` geomean on `[2560, 640]`, `1.89×` at `T=4096` on `[16384, 2560]`); unlock `QMMA.16832` prefill |
| **T2** | **P0** | Post-Rebase `--a8-activations` (`GgmlA8`, `GgmlA8Wide`, `ggml_expert` A8, `W4A8` MoE) | Upstream `d8652998d` added `IMMA.16832.S8` (`251.2 TOP/s`) kernels with 5090 tile shapes (`kWarpBudget = 1024`, unswizzled wide tiles) | `+35%` to `+75%` `IQ3_S` prefill (`1,045 -> 1,400–1,800+ tok/s`) & `+10–20%` `IQ3_S`/`W4A16` decode |
| **T3** | **P0** | Post-Rebase `Qwen3.5-122B` (`Ornith-1.5-122B`) Per-Row FP8 Shapes (`K = 3072, 8192`) | Upstream `c17e24a89` added 6 `fp8_e4m3fn_row_f32` shapes with 170-SM `WaveCtas = 170` and unswizzled `128×256` `Bulk` tiles | `+20%` to `+34%` large-`T` prefill (`BulkG8` `128×128` `Grouped, 8` avoids 24 MiB L2 thrashing) & `-8%` to `-15%` mid-`T` wave cliffs |
| **T4** | **P1** | Column-Invariant Fixed-Schedule `Q8` `SlicedKMma` (`T = 2..8`) & Small-Weight Single-Launch Split-K | Flash-Next `Q8` hyper-connections (`97×`/pass) & shared experts (`49×`/pass) drop from `230 GB/s` (`T=1..2`) to `117–162 GB/s` (`T=5..8`) on SIMT FMA | Lift `T = 4..8` MTP verify / multi-row bandwidth to `195–225 GB/s` (`+20%` to `+65%` op speedup; `+2–4%` E2E MTP decode) |
| **T5** | **P1** | Hybrid Prefix-Cache `ContextPrefillCost` Calibration on GB10 (`context_cost_defaults.cpp`) | `"nvidia-gb10-sm121"` preset has GB10 `.transfer` costs but copies RTX 5090 `.prefill` costs (`3.5×–6×` too low in ns) | Accurate prefix-cache restore vs. recompute scheduling on multi-turn serving workloads |
| **T6** | **P1** | `ScreenedLmHead` (`screened_head.cu`) & `rg16_linear.cu` 48-SM Occupancy Retune | `screened_head.cu` launches `1` persistent 128-thread CTA/SM (`8.3%` warp occupancy on 48 SMs); `rg16_linear.cu` tuned for 170 SMs | `1.3×–1.6×` faster 2-bit vocabulary screen (`179 MiB` vs `512–1,270 MiB` full head); `+3–7%` decode with `--screen-heads` |
| **T7** | **P1** | `DFlash2` Companion Drafter & `DrafterExpertPool` UMA Optimization (`run_server.sh`, `qwen4_exp`, `qwen3_5`) | `DrafterExpertPool` allocates duplicate VRAM + pinned RAM on UMA; `DFlash2` pair/tree constants (`kDFlashPairMaxDrafts = 2`) tuned on 5090 | Zero duplicate UMA memory for drafter experts; evaluate/tune `DFlash2` vs `MTP` across `c = 1..4` |
| **T8** | **P0** | Eliminate Rebase-Regressed & Newly Added Linear / MoE / GGUF CUDA Stack Spills (`f16acc`, `fp8`, `wide_expert`, `ggml_a8`) | Rebase `3a4b8da40` + `T1` added `151` spilling linear/MoE/GGUF kernels (`8..72 B` `STL`/`LDL`) from unrolled outer loops & pre-staged fragments | Eliminate `151` production kernel stack spills (`STACK = 0, LOCAL = 0`) while preserving `MinBlocksPerSm` and numerical exactness |
| **T9** | **P0** | Eliminate VQ2 / Q4 & INT8 / BF16 Causal-Cache Attention CUDA Stack Spills (`vq2`, `int8`, `bf16`) | Upstream RTX 5090 attention headers overwrote GB10 register-pressure fixes, causing `353` `vq2` (`8..152 B`) + `100` `int8`/`bf16` (`8..56 B`) spills | Eliminate preventable attention stack spills (`453` instances) via `q_scale_shared`, direct FP32 `mma_f16`, and group/step unroll bounds |
| **T10** | **P1** | Eliminate Remaining Quantized Linear (`Q8`/`Q5`/`Q4`), GDN & RoPE `sincosf` Stack Frames (`116` kernels) | `63` RoPE/QSA kernels allocate a `32 B` `sincosf` Payne-Hanek slowpath stack frame; `53` `q8`/`q5`/`q4`/`ggml`/`moe` kernels spill `8..72 B` | Eliminate remaining preventable `STL`/`LDL` stack frames across `q8`, `q5`, `q4`, `gdn`, `ggml`, and `rope`/`qsa` |
| **T11** | **P0** | Audit, Regenerate & Ratchet `tools/cuda_stack/cuda_stack_allowlist.json` on GB10 (`sm_121a`) | Allowlist bloated from `260` (`origin/dev`) to `826` entries (`HEAD`) due to blind `check_cuda_stack.py update` on RTX 5090 / rebased code | Shrink `cuda_stack_allowlist.json` from `826` entries down to only `--lossless` software-oracle kernels and verified occupancy-load-bearing exceptions |
| **T12** | **P2** | Cumulative E2E Verification (`run_kv_matrix_bench.sh` + Multi-Concurrency `llama-benchy` + Stack Ratchet) | Full regression, numerical oracle, zero-spill ratchet, and multi-concurrency serving verification | Verified E2E tables in `docs/gb10_roadmap.md` with `379/379` CTest pass and ratcheted stack allowlist |

---

## 3. Detailed Task Specifications

---

### Task T1 (Priority P0): Complete & Qualify `MXFP8` (`FP4-Local`) Linear Schedules & Re-evaluate `F16AccumulatePrefill` on GB10

#### Affected Files & Symbols
- [`src/ops/linear/mxfp8/shapes/n1280_k2560.cu`](file:///opt/etc/Infernix/src/ops/linear/mxfp8/shapes/n1280_k2560.cu#L18-L40) (`launch_a16`, `launch_f16`)
- [`src/ops/linear/mxfp8/shapes/n13952_k2560.cu`](file:///opt/etc/Infernix/src/ops/linear/mxfp8/shapes/n13952_k2560.cu#L18-L55) (`launch_a16`, `launch_f16`)
- [`src/ops/linear/mxfp8/shapes/n16384_k2560.cu`](file:///opt/etc/Infernix/src/ops/linear/mxfp8/shapes/n16384_k2560.cu#L18-L58) (`launch_a16`, `launch_f16`)
- [`src/ops/linear/mxfp8/shapes/n2560_k6144.cu`](file:///opt/etc/Infernix/src/ops/linear/mxfp8/shapes/n2560_k6144.cu#L18-L48) (`launch_a16`, `launch_f16`)
- [`src/ops/linear/mxfp8/shapes/n2560_k640.cu`](file:///opt/etc/Infernix/src/ops/linear/mxfp8/shapes/n2560_k640.cu#L18-L42) (`launch_a16`, `launch_f16`)
- [`bench/ops/mxfp8_flash_next_sweep.cu`](file:///opt/etc/Infernix/bench/ops/mxfp8_flash_next_sweep.cu#L136-L185) (`Options::flush_mib`)
- [`src/models/qwen4_exp/program/fast_paths.cpp`](file:///opt/etc/Infernix/src/models/qwen4_exp/program/fast_paths.cpp#L448-L455), [`L730-L740`](file:///opt/etc/Infernix/src/models/qwen4_exp/program/fast_paths.cpp#L730-L740), [`L819-L824`](file:///opt/etc/Infernix/src/models/qwen4_exp/program/fast_paths.cpp#L819-L824)

#### Root Cause & Hardware Rationale
1. The 5 `src/ops/linear/mxfp8/shapes/*.cu` files were originally tuned on RTX 5090 (`170` SMs, `256 MiB` L2 flush). A GB10 sweep (`48 MiB` cold-L2 flush, 21 reps) is currently in the working tree for `launch_a16`, yielding **`1.153×` geomean** on `[2560, 640]` (up to `1.58×` at `T = 128`) and **`1.89×` at `T = 4096`** on `[16384, 2560]` (`24,785.6 → 13,083.3 us`) and `[2560, 6144]` (`9,376.0 → 4,839.2 us`).
2. However, `launch_f16` (`FastPath::F16AccumulatePrefill`) in all 5 files still uses the unverified RTX 5090 bands. More importantly, `F16AccumulatePrefill` exists only because FP32-accumulating HMMA runs at half rate on RTX 5090 (`sm_120a`), whereas on GB10 (`sm_121a`) **FP32-accumulating HMMA runs at 1:1 parity with FP16-accumulating HMMA (`125.31` vs `125.44 TFLOP/s`)**!
3. Furthermore, in [`qwen4_exp/program/fast_paths.cpp:454,733`](file:///opt/etc/Infernix/src/models/qwen4_exp/program/fast_paths.cpp#L454), `FastPath::F16AccumulatePrefill` is enabled by default for `{DenseKernels::Mxfp8, false, KvCacheStorage::Int8Group64}` and overrides `Fp8A8PairBlock32` (`QMMA.16832`, `245.9 TFLOP/s` pipe).

#### Required Implementation Steps
1. Build and run `infernix_mxfp8_flash_next_sweep` and `infernix_linear_bench --qtype mxfp8` to verify the working-tree `launch_a16` changes across all 5 shapes (`T = 1..8192`).
2. Benchmark `launch_f16` (`--routes f16-accumulate-prefill`) against exact `launch_a16` (`--routes exact`) and `Fp8A8PairBlock32` (`--routes fp8-a8-pair-block32`) on GB10 (`48 MiB` flush).
3. Update `launch_f16` bands to match the GB10-optimal TMA schedules, and if `F16AccumulatePrefill` is slower than or within noise of exact `launch_a16` on GB10 (because `sm_121a` has 1:1 FP32 HMMA parity), remove `FastPath::F16AccumulatePrefill` from `kQualified` in [`qwen4_exp/program/fast_paths.cpp:454`](file:///opt/etc/Infernix/src/models/qwen4_exp/program/fast_paths.cpp#L454).

#### Verification & Acceptance Criteria
```bash
ninja -C build-native -j "$(nproc)" infernix_mxfp8_flash_next_sweep infernix_linear_bench infernix_linear_mxfp8_a16_test
./build-native/tests/infernix_linear_mxfp8_a16_test
./build-native/bench/infernix_mxfp8_flash_next_sweep --repeat 21
./build-native/bench/infernix_linear_bench --qtype mxfp8 --tokens 1,5,32,128,512,2048,4096
ctest --test-dir build-native -R infernix_cuda_stack_check --output-on-failure
```
- **Acceptance**: `0` test failures, `0` stack spills, and `>= 15%` latency reduction at `T >= 512` across the 5 `MXFP8` shapes vs. the RTX 5090 schedule baseline.

---

### Task T2 (Priority P0): Tune Post-Rebase `--a8-activations` (`GgmlA8`, `GgmlA8Wide`, `ggml_expert` A8 & `W4A8` MoE) for GB10

#### Affected Files & Symbols
- [`src/ops/ggml/ggml_linear_a8.cuh`](file:///opt/etc/Infernix/src/ops/ggml/ggml_linear_a8.cuh#L401-L460) (`kShapes`, `kManyWaves`, `kWarpBudget`, `kMinUnits`)
- [`src/ops/ggml/ggml_a8_tile.cuh`](file:///opt/etc/Infernix/src/ops/ggml/ggml_a8_tile.cuh) (wide `IMMA.16832.S8` prefill tile schedule & CTA rasterization)
- [`src/ops/ggml/ggml_bf16_tile.cuh`](file:///opt/etc/Infernix/src/ops/ggml/ggml_bf16_tile.cuh) (wide `HMMA.16816` prefill tile CTA rasterization)
- [`src/ops/offloaded_sparse_moe/cuda/ggml_expert.cuh`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/ggml_expert.cuh#L281-L520) (GGUF expert narrow/wide `A8` and `BF16` routes)
- [`src/models/qwen4_exp/program/fast_paths.cpp`](file:///opt/etc/Infernix/src/models/qwen4_exp/program/fast_paths.cpp#L775-L825) (`a8_activation_routes`)

#### Root Cause & Hardware Rationale
1. Upstream commit `d8652998d` added `--a8-activations` (`IMMA.16832.S8` 8-bit activation paths for GGUF dense projections, GGUF experts, and NVFP4 W4A16 experts). On GB10, `IMMA.16832.S8` has a measured ceiling of **`251.2 TOP/s`** (`2×` `HMMA.16816`'s `125.4 TFLOP/s`), whereas `IQ3_S` currently achieves only `26–30 TFLOP/s` (`pp4096 = 1,045 tok/s`, vs `2,866 tok/s` on `Dense8` and `3,238 tok/s` on `NVFP4`).
2. However, [`ggml_linear_a8.cuh:405-416`](file:///opt/etc/Infernix/src/ops/ggml/ggml_linear_a8.cuh#L405-L416) explicitly notes that its `kShapes` and `kWarpBudget = 1024` (`170 SMs × 6 warps`) were copied from RTX 5090 ("their own timing is still owed"). On 48 SMs, `kWarpBudget = 1024` is `21.3` warps/SM, causing narrow calls (`T = 1..31`) to over-split K and spill into a 2nd wave (`+25–30%` latency penalty).
3. In [`ggml_a8_tile.cuh`](file:///opt/etc/Infernix/src/ops/ggml/ggml_a8_tile.cuh) and [`ggml_bf16_tile.cuh`](file:///opt/etc/Infernix/src/ops/ggml/ggml_bf16_tile.cuh), wide prefill tiles (`T >= 32`) do not yet apply GB10's 2D `kGroupRows = 8` L2 CTA swizzling when `W > 16 MiB`, causing L2 thrashing at `T = 4096`.
4. In [`qwen4_exp/program/fast_paths.cpp:787-792`](file:///opt/etc/Infernix/src/models/qwen4_exp/program/fast_paths.cpp#L787-L792), `ggml_dense_prefill`, `mxfp8_dense`, and `experts_prefill` are marked `not yet measured end to end`.

#### Required Implementation Steps
1. Benchmark `infernix_ggml_linear_bench` across `T = 1, 2, 4, 5, 8, 16, 31, 32, 64, 128, 512, 4096` for `IQ3_S`, `IQ2_S`, `Q4_K`, `Q6_K` on exact, `bf16-wide`, `a8`, and `a8-wide` routes.
2. Retune `kWarpBudget` (e.g., sweep `288, 384, 512, 768, 1024` for 48 SMs) and `kShapes` in [`ggml_linear_a8.cuh`](file:///opt/etc/Infernix/src/ops/ggml/ggml_linear_a8.cuh#L408-L417).
3. Add 2D `kGroupRows = 8` CTA swizzling to [`ggml_a8_tile.cuh`](file:///opt/etc/Infernix/src/ops/ggml/ggml_a8_tile.cuh) and [`ggml_bf16_tile.cuh`](file:///opt/etc/Infernix/src/ops/ggml/ggml_bf16_tile.cuh) when `T >= 512` and `W > 16 MiB`.
4. Benchmark `infernix_offloaded_moe_decode_bench` and `infernix_offloaded_moe_wide_bench` with `A8` expert activations on `IQ3_S` and `W4A16`, then run an end-to-end A/B on `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix.infernix` with `--a8-activations` and update `a8_activation_routes()` in [`qwen4_exp/program/fast_paths.cpp`](file:///opt/etc/Infernix/src/models/qwen4_exp/program/fast_paths.cpp#L775-L794) so only routes verified faster on GB10 remain active.

#### Verification & Acceptance Criteria
```bash
ninja -C build-native -j "$(nproc)" infernix_ggml_linear_bench infernix_ggml_linear_test infernix_offloaded_moe_wide_bench infernix_offloaded_moe_decode_bench infernix_bench
./build-native/tests/infernix_ggml_linear_test
./build-native/bench/infernix_ggml_linear_bench
ctest --test-dir build-native -R "ggml|offloaded_moe|cuda_stack_check" --output-on-failure
```
- **Acceptance**: All `ggml` and `offloaded_moe` oracle tests pass (`0` stack spills); `IQ3_S` `pp4096` and `tg128` with `--a8-activations` achieve a verified speedup over the exact/BF16 baseline.

---

### Task T3 (Priority P0): Retune `Qwen3.5-122B-A10B` (`Ornith-1.5-122B-NVFP4`) Per-Row FP8 Shapes for 48 SMs & 24 MiB L2

#### Affected Files & Symbols
- [`src/ops/linear/fp8/shapes/n248320_k3072_f32.cu`](file:///opt/etc/Infernix/src/ops/linear/fp8/shapes/n248320_k3072_f32.cu#L34-L86) (`[248320, 3072]` vocabulary head)
- [`src/ops/linear/fp8/shapes/n3072_k1024_f32.cu`](file:///opt/etc/Infernix/src/ops/linear/fp8/shapes/n3072_k1024_f32.cu#L31-L58) (`[3072, 1024]` shared-expert down)
- [`src/ops/linear_swiglu/fp8/fp8_linear_swiglu_n2048_k3072.cu`](file:///opt/etc/Infernix/src/ops/linear_swiglu/fp8/fp8_linear_swiglu_n2048_k3072.cu#L40-L63) (`[2048, 3072]` shared-expert gate/up)
- [`src/ops/attn_input_proj/fp8/fp8_attn_input_n17408_k3072.cu`](file:///opt/etc/Infernix/src/ops/attn_input_proj/fp8/fp8_attn_input_n17408_k3072.cu#L19-L106) (`[17408, 3072]` QKV+gate projection)
- [`src/ops/gdn_input_proj/fp8/fp8_gdn_input_n20480_k3072.cu`](file:///opt/etc/Infernix/src/ops/gdn_input_proj/fp8/fp8_gdn_input_n20480_k3072.cu#L21-L120) (`[20480, 3072]` GDN input projection)
- [`src/ops/linear_add/fp8/fp8_linear_add_n3072_k8192.cu`](file:///opt/etc/Infernix/src/ops/linear_add/fp8/fp8_linear_add_n3072_k8192.cu#L31-L118) (`[3072, 8192]` residual output projection)

#### Root Cause & Hardware Rationale
1. Upstream commit `c17e24a89` added support for `Qwen3.5-122B-A10B` (`Ornith-1.5-122B-UNCENSORED-NVFP4`, `K = 3072`), copying schedule ladders from RTX 5090 (`170 SMs`, `96 MiB` L2):
   - In [`fp8_attn_input_n17408_k3072.cu:22,102-104`](file:///opt/etc/Infernix/src/ops/attn_input_proj/fp8/fp8_attn_input_n17408_k3072.cu#L22), [`fp8_gdn_input_n20480_k3072.cu:24,117-118`](file:///opt/etc/Infernix/src/ops/gdn_input_proj/fp8/fp8_gdn_input_n20480_k3072.cu#L24), and [`fp8_linear_add_n3072_k8192.cu:35,116`](file:///opt/etc/Infernix/src/ops/linear_add/fp8/fp8_linear_add_n3072_k8192.cu#L35), the large-`T` schedule is unswizzled `Bulk = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, 170, 4, 8>`. On GB10's 24 MiB L2 cache, unswizzled `128×256` at `T >= 2048` thrashes L2 (`-21%` to `-25%` slower than `BulkG8 = Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 3, 1, Fp8MmaRaster::Grouped, 8>`, as proven in `P1.3` on `K = 5120`).
   - In [`fp8_attn_input_n17408_k3072.cu:102-103`](file:///opt/etc/Infernix/src/ops/attn_input_proj/fp8/fp8_attn_input_n17408_k3072.cu#L102-L103), `Tma96x256` at `tokens <= 288` was explicitly chosen because `68 row tiles × 3 = 204 CTAs = 1.2 waves on 170 SMs`, which on 48 SMs is `4.25` waves (leaving 36 SMs idle in the 5th wave!).
   - In [`n248320_k3072_f32.cu:47-73`](file:///opt/etc/Infernix/src/ops/linear/fp8/shapes/n248320_k3072_f32.cu#L47-L73), `T > 64` uses unswizzled `Fp8A16MmaSchedule<64, 128, ...>` on a `762 MiB` weight matrix instead of `Grouped, 8` TMA (`Fp8A16TmaT64R128G8` as used in `n248320_k5120.cu`).

#### Required Implementation Steps
1. Add `BulkG8` (`128×128`, `Stages = 3`, `Fp8MmaRaster::Grouped, 8`, `STACK = 0`) for `T > 2048` (`T > 3072` for `K = 3072`) and 48-SM wave-aligned mid-`T` TMA schedules (`T = 129..576`) to `fp8_attn_input_n17408_k3072.cu`, `fp8_gdn_input_n20480_k3072.cu`, and `fp8_linear_add_n3072_k8192.cu` (matching the `P1.3` retune of `fp8_attn_input_a8.cu`, `fp8_gdn_input_a8.cu`, and `fp8_linear_add_a8.cu`).
2. Benchmark and retune `n248320_k3072_f32.cu`, `n3072_k1024_f32.cu`, and `fp8_linear_swiglu_n2048_k3072.cu` on GB10 using `infernix_linear_bench --qtype fp8f32` and `infernix_fp8_linear_swiglu_bench --shared-122b`.

#### Verification & Acceptance Criteria
```bash
ninja -C build-native -j "$(nproc)" infernix_linear_bench infernix_fp8_linear_swiglu_bench infernix_fp8_linear_add_bench infernix_attn_input_proj_bench infernix_gdn_input_proj_bench infernix_linear_fp8_a16_test infernix_linear_fp8_a8_test infernix_linear_swiglu_fp8_test infernix_linear_add_fp8_test
./build-native/bench/infernix_linear_bench --qtype fp8f32 --tokens 1,5,16,64,256,512,2048,4096
./build-native/bench/infernix_fp8_linear_swiglu_bench --shared-122b --t-sweep 1,5,16,64,256,512,2048,4096
ctest --test-dir build-native -R "fp8|cuda_stack_check" --output-on-failure
```
- **Acceptance**: `>= 20%` speedup at `T = 4096` on the `K = 3072` / `K = 8192` `FP8 A8` projections via `BulkG8`, zero route-transition cliffs `>= 15%`, `0` stack spills, and all `fp8` oracle tests passing.

---

### Task T4 (Priority P1): Column-Invariant Fixed-Schedule `Q8` `SlicedKMma` (`T = 2..8`) & Small-Weight Single-Launch Split-K (`Gaps 3 & 4`)

#### Affected Files & Symbols
- [`src/ops/linear/q8/q8_route_catalog.h`](file:///opt/etc/Infernix/src/ops/linear/q8/q8_route_catalog.h#L15-L34) (`q8_a16_column_invariance`)
- [`src/ops/linear/q8/q8_a16_sliced_k_mma.cuh`](file:///opt/etc/Infernix/src/ops/linear/q8/q8_a16_sliced_k_mma.cuh) (`q8_a16_sliced_k_mma_kernel`)
- [`src/ops/linear/q8/shapes/n10240_k320.cu`](file:///opt/etc/Infernix/src/ops/linear/q8/shapes/n10240_k320.cu), [`n320_k10240.cu`](file:///opt/etc/Infernix/src/ops/linear/q8/shapes/n320_k10240.cu), [`n324_k10240.cu`](file:///opt/etc/Infernix/src/ops/linear/q8/shapes/n324_k10240.cu), [`n2560_k640.cu`](file:///opt/etc/Infernix/src/ops/linear/q8/shapes/n2560_k640.cu), [`n1280_k2560.cu`](file:///opt/etc/Infernix/src/ops/linear/q8/shapes/n1280_k2560.cu)
- [`tests/ops/linear/test_q8_a16.cpp`](file:///opt/etc/Infernix/tests/ops/linear/test_q8_a16.cpp)

#### Root Cause & Hardware Rationale
1. In **both `Dense8` and `NVFP4-W4Dense`** (`Qwen3.8-Flash-Next`), every forward pass executes `97×` `Q8` hyper-connection projections (`[324, 10240]` and `[10240, 320]`) and `49×` `Q8` shared-expert projections (`[1280, 2560]` and `[2560, 640]`).
2. Currently, [`q8_a16_column_invariance()`](file:///opt/etc/Infernix/src/ops/linear/q8/q8_route_catalog.h#L15-L34) restricts the exact route on these shapes to SIMT/Stream FP32 FMA (`28.0 TFLOP/s` pipe) up to `kColumns` (`8..64`). On SIMT/Stream, every extra token column `c = 1..8` linearly multiplies `LDG.E.128` activation loads and `fmaf` instructions, causing effective DRAM bandwidth to drop from `191–231 GB/s` at `T = 1..2` down to **`117–162 GB/s` at `T = 5..8`** (the exact width of 1-row and 2-row MTP speculative verification!).
3. In contrast, `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` (`SlicedKMma`, `TileTokens = 8`) computes all `1..8` columns in a single Tensor Core instruction (`200–225 GB/s` flat across `T = 1..8`). Because each output column `col` of `D = A * B` in `m16n8k16` is an independent hardware inner product of row `A[i, :]` and column `B[:, col]`, **a `SlicedKMma` kernel with a fixed `(BlockRows, KSplit)` across `T = 1..8` is strictly bit-invariant across `T = 1..8`**!
4. Additionally, for small-weight shapes (`W = 0.65–3.3 MiB`: `[2560, 640]`, `[1280, 2560]`, `[320, 10240]`), ideal DRAM read time is `2.6–6.5 us`, so launching a separate `reduce_split_k` kernel adds `~3.5 us` of launch/barrier overhead (`Gap 4`).

#### Required Implementation Steps
1. Verify bit-exact column invariance across `T = 1..8` (`test_q8_a16.cpp`) for a fixed-`(BlockRows, KSplit)` `Q8A16SlicedKMma` (`TileTokens = 8`) schedule (or expose under `FastPath::Q8SlicedK` where `Q8SlicedK` is active).
2. Benchmark `T = 1, 2, 4, 5, 8` on `[10240, 320]`, `[320, 10240]`, `[324, 10240]`, `[2560, 640]`, `[2560, 2560]`, `[12800, 2560]`, `[13952, 2560]`, and `[16384, 2560]` using both `infernix_linear_bench --qtype q8` (cold L2) and `infernix_hyper_connection_bench` (hot-L2 composed execution, per Guardrail #7 in `docs/gb10_roadmap.md`).
3. Evaluate an in-kernel L2 arrival-counter Split-K reduction epilogue for `W <= 3.3 MiB` shapes at `T = 1..8` to eliminate the second kernel launch.

#### Verification & Acceptance Criteria
```bash
ninja -C build-native -j "$(nproc)" infernix_linear_bench infernix_hyper_connection_bench infernix_linear_q8_a16_test
./build-native/tests/infernix_linear_q8_a16_test
./build-native/bench/infernix_hyper_connection_bench --columns 1,2,4,5,8,16,32,64
./build-native/bench/infernix_linear_bench --qtype q8 --tokens 1,2,4,5,8,16
ctest --test-dir build-native -R infernix_cuda_stack_check --output-on-failure
```
- **Acceptance**: `test_q8_a16` passes bit-exact column invariance; `T = 5..8` bandwidth on `[10240, 320]`, `[2560, 640]`, and trunk `Q8` shapes increases by `>= 15%` without regressing `infernix_hyper_connection_bench` at `T = 1..8`.

---

### Task T5 (Priority P1): Calibrate Hybrid Prefix-Cache `ContextPrefillCost` on GB10 (`P1.A`)

#### Affected Files & Symbols
- [`src/runtime/engine/context_cache/context_cost_defaults.cpp`](file:///opt/etc/Infernix/src/runtime/engine/context_cache/context_cost_defaults.cpp#L78-L103) (`compiled_context_cost_defaults`, `"nvidia-gb10-sm121"`)
- [`tests/unit/test_context_cost.cpp`](file:///opt/etc/Infernix/tests/unit/test_context_cost.cpp)

#### Root Cause & Hardware Rationale
1. In [`context_cost_defaults.cpp:78-102`](file:///opt/etc/Infernix/src/runtime/engine/context_cache/context_cost_defaults.cpp#L78-L102), the `"nvidia-gb10-sm121"` preset has GB10-calibrated `.transfer` costs (`ns_per_byte_q32 = {183'965'911, 220'829'326, 30'671'379}`), **but its `.prefill` map (`lines 91-101`) still reuses `kGroupwise27bPrefill` (`chunk_ns = 40,803,562`) and `kNvfp4Fp8Prefill` (`chunk_ns = 14,724,132`) verbatim from `"nvidia-geforce-rtx-5090-sm120"`**!
2. On GB10 (`48 SMs`, `247.6 GB/s`), actual prefill recomputation time per chunk/token in nanoseconds is `~3.5×–6×` higher than on RTX 5090 (`170 SMs`, `1792 GB/s`). Using 5090 prefill costs causes the hybrid prefix-cache planner (`hybrid_cache_cost`) to under-price recomputation relative to host/disk context restoration.

#### Required Implementation Steps
1. Build and run `./build-native/bench/infernix_context_cost_bench --suite prefill` on GB10.
2. Define `kGb10Groupwise27bPrefill` and `kGb10Nvfp4Fp8Prefill` in [`context_cost_defaults.cpp`](file:///opt/etc/Infernix/src/runtime/engine/context_cache/context_cost_defaults.cpp#L78-L102) with the GB10-measured `chunk_ns`, `linear_ns_per_token`, and `attention_ns_per_token_step_q32` coefficients, and bind them in the `"nvidia-gb10-sm121"` preset.
3. Update [`tests/unit/test_context_cost.cpp`](file:///opt/etc/Infernix/tests/unit/test_context_cost.cpp) to validate the GB10 preset's `.prefill` and `.transfer` entries.

#### Verification & Acceptance Criteria
```bash
ninja -C build-native -j "$(nproc)" infernix_context_cost_bench infernix_context_cost_test
./build-native/bench/infernix_context_cost_bench --suite prefill
./build-native/tests/infernix_context_cost_test
```
- **Acceptance**: `infernix_context_cost_test` passes and `"nvidia-gb10-sm121"` `.prefill` entries reflect real measured GB10 prefill timings instead of RTX 5090 constants.

---

### Task T6 (Priority P1): Tune `ScreenedLmHead` (`screened_head.cu`) & `rg16_linear.cu` for 48 SMs

#### Affected Files & Symbols
- [`src/ops/screened_head/screened_head.cu`](file:///opt/etc/Infernix/src/ops/screened_head/screened_head.cu#L54-L180) (`kScreenWarps`, `kStages`, `screen_kernel`, persistent grid sizing)
- [`src/ops/offloaded_sparse_moe/cuda/rg16_linear.cu`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/rg16_linear.cu#L51-L160) (`kWarps`, `kWideColumns`, persistent GEMV launch)
- [`src/models/qwen4_exp/program/fast_paths.cpp`](file:///opt/etc/Infernix/src/models/qwen4_exp/program/fast_paths.cpp#L448-L454) (`FastPath::ScreenedLmHead`)

#### Root Cause & Hardware Rationale
1. `screened_head.cu` implements a two-stage vocabulary head: a 2-bit `q2_g64_fp16` screen (`720 B/row` = `179 MiB` for `248,320 × 2560`, vs `512 MiB` for `Q6_K`/`Q8` or `1.27 GB` for `BF16`) using `IMMA.16832.S8`, followed by exact rescoring of at most `256` candidate rows per column.
2. However, `screen_kernel` in [`screened_head.cu`](file:///opt/etc/Infernix/src/ops/screened_head/screened_head.cu) was tuned on RTX 5090 (`170` persistent CTAs, `1` CTA/SM × `128` threads = `4` warps/SM = **`8.3%` SM occupancy**). On GB10 (`48` SMs), `48` CTAs × `4` warps (`192` warps total across the GPU) is not enough concurrent warp-level memory parallelism to saturate LPDDR5X across `15,520` 16-row items. For `Tiles = 1` (`1..8` columns, where dynamic shared memory `screen_shared_bytes<1>()` is `~49.1 KiB`), launching **`2 CTAs/SM` (`96` persistent CTAs, `98.2 KiB <= 99 KiB` SMEM)** doubles active warps and in-flight `cp.async` ring stages per SM!
3. Similarly, [`rg16_linear.cu`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/rg16_linear.cu) (serving Qwen3.5-122B's `[248320, 3072]` head and Qwen3.8-Flash-Next's vision MLP/merger `[4304, 1152]`, `[1152, 4304]`, `[4608, 4608]`, `[2560, 4608]`) was sized for 170 SMs (`3.4` waves on 170 SMs = `12.0` waves on 48 SMs).

#### Required Implementation Steps
1. Inspect `cuobjdump -res-usage` and `ckt static --sass` on `screened_head.cu` and `rg16_linear.cu`.
2. In [`screened_head.cu`](file:///opt/etc/Infernix/src/ops/screened_head/screened_head.cu), test `2 CTAs/SM` (`96` persistent blocks on GB10) for `Tiles = 1` (`columns <= 8`, covering all decode and MTP verification rounds) and measure `screen + select + rescore` latency against the full `[248320, 2560]` head at `T = 1, 2, 4, 5, 8`.
3. Qualified under `FastPath::ScreenedLmHead`, verify whether artifacts with `--screen-heads` gain `+3–7%` decode throughput on GB10 and register the qualified configurations in [`qwen4_exp/program/fast_paths.cpp`](file:///opt/etc/Infernix/src/models/qwen4_exp/program/fast_paths.cpp#L453).

#### Verification & Acceptance Criteria
```bash
ninja -C build-native -j "$(nproc)" infernix_screened_head_test infernix_rg16_linear_test
./build-native/tests/infernix_screened_head_test
./build-native/tests/infernix_rg16_linear_test
ctest --test-dir build-native -R "screened_head|rg16_linear|cuda_stack_check" --output-on-failure
```
- **Acceptance**: `infernix_screened_head_test` and `infernix_rg16_linear_test` pass (`0` stack spills), and `screened_head` latency at `T = 1..8` improves by `>= 15%` on GB10.

---

### Task T7 (Priority P1): Optimize `DFlash2` Companion Drafter & `DrafterExpertPool` on GB10 Unified Memory

#### Affected Files & Symbols
- [`src/models/qwen3_5/program/offload.cpp:98-120`](file:///opt/etc/Infernix/src/models/qwen3_5/program/offload.cpp#L98-L120) (`ProgramOffload::create_drafter_pool`)
- [`src/runtime/expert_offload/drafter_pool.cpp`](file:///opt/etc/Infernix/src/runtime/expert_offload/drafter_pool.cpp) (`DrafterExpertPool`)
- [`src/models/qwen4_exp/program/program_impl.h`](file:///opt/etc/Infernix/src/models/qwen4_exp/program/program_impl.h#L5320-L5360) (`kDFlashPairMaxDrafts`, DFlash2 draft/tree policy)
- [`tools/run_server.sh:35,156-166`](file:///opt/etc/Infernix/tools/run_server.sh#L156-L166) (`DRAFT_PATH`)

#### Root Cause & Hardware Rationale
1. Upstream commit `d8652998d` introduced `DrafterExpertPool` in [`src/runtime/expert_offload/drafter_pool.cpp`](file:///opt/etc/Infernix/src/runtime/expert_offload/drafter_pool.cpp) and [`src/models/qwen3_5/program/offload.cpp:98-120`](file:///opt/etc/Infernix/src/models/qwen3_5/program/offload.cpp#L98-L120) to keep `slots` of the MTP drafter's routed experts in a VRAM pool while pinning the rest in RAM. On GB10 (`integrated == 1`), VRAM and RAM are the same physical LPDDR5X pool (`src/runtime/config.cpp` already disables `mtp_expert_slots` when `load.device_experts` is true, but `qwen3_5/program/offload.cpp` should also guard against creating a split RAM/VRAM drafter pool on integrated UMA devices).
2. On `Qwen3.8-Flash-Next` (`qwen4_exp`), the companion `DFlash2` drafter artifact (`/opt/etc/models/Qwen3.8-Flash-Next-DFlash2-Head/Qwen3.8-Flash-Next-DFlash2-Infernix.infernix`) is installed locally and referenced in [`tools/run_server.sh:35,156`](file:///opt/etc/Infernix/tools/run_server.sh#L35). While `MTP` pair/multi-row drafting was retuned for GB10 (`kPairMaxDrafts = 6`, `kPairWidthCost = 0.15`), `kDFlashPairMaxDrafts = 2` and DFlash2 verification tree widths have not yet been swept on GB10 across `c = 1, 2, 4`.

#### Required Implementation Steps
1. Verify that on integrated GPUs (`device.integrated`), `qwen3_5` offloaded MoE and `DrafterExpertPool` never double-allocate pinned host + device pool memory for drafter experts.
2. Run an interleaved A/B benchmark (`infernix_bench` and `llama-benchy` at `c = 1, 2, 4`) comparing `--spec mtp --lm-head-draft` against `--spec dflash2 --draft-model /opt/etc/models/Qwen3.8-Flash-Next-DFlash2-Head/Qwen3.8-Flash-Next-DFlash2-Infernix.infernix` across `kDFlashPairMaxDrafts = {2, 4, 6}` on GB10.
3. Update `kDFlashPairMaxDrafts` in [`qwen4_exp/program/program_impl.h`](file:///opt/etc/Infernix/src/models/qwen4_exp/program/program_impl.h) and document the winning server configuration in [`tools/run_server.sh`](file:///opt/etc/Infernix/tools/run_server.sh).

#### Verification & Acceptance Criteria
```bash
ninja -C build-native -j "$(nproc)" infernix_bench infernix-serve
./build-native/bench/infernix_bench --model /opt/etc/models/Qwen3.8-Flash-Next-F4-Local-NVPF4-Dense/Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix --draft-model /opt/etc/models/Qwen3.8-Flash-Next-DFlash2-Head/Qwen3.8-Flash-Next-DFlash2-Infernix.infernix --spec dflash2 --kv-dtype fp8 --prompt-tokens 2048 --gen-tokens 128 --concurrency 1,2,4
```
- **Acceptance**: Measured `c = 1, 2, 4` `pp2048+tg128` comparison between `MTP` (`--lm-head-draft`) and `DFlash2` recorded with optimal GB10 draft policy constants.

---

### Task T8 (Priority P0): Eliminate Rebase-Regressed & Newly Added Linear / MoE / GGUF CUDA Stack Spills (`151` Kernels)

#### Affected Files & Symbols
- [`src/ops/linear/f16acc/f16acc_mma.cuh`](file:///opt/etc/Infernix/src/ops/linear/f16acc/f16acc_mma.cuh#L112-L239), [`L325-L372`](file:///opt/etc/Infernix/src/ops/linear/f16acc/f16acc_mma.cuh#L325-L372) (`f16acc_compute_stage`, `f16acc_mma_kernel`: `58` kernels, `STACK = 16..72 B`)
- [`src/ops/linear/fp8/fp8_a8_mma_common.cuh`](file:///opt/etc/Infernix/src/ops/linear/fp8/fp8_a8_mma_common.cuh#L95-L128) (`fp8_mma_compute_stage`, `fp8_a8_tma_mma_kernel`: `44` kernels, `STACK = 16..48 B`)
- [`src/ops/linear/fp8/fp8_a16_ring_mma.cuh`](file:///opt/etc/Infernix/src/ops/linear/fp8/fp8_a16_ring_mma.cuh#L38-L88) (`fp8_a16_ring_compute_stage`, `fp8_a16_tma_mma_kernel`: `8` kernels, `STACK = 16 B`)
- [`src/ops/offloaded_sparse_moe/cuda/wide_expert_a16.cuh`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert_a16.cuh#L285-L350) (`grouped_kernel_a16`: `16` kernels, `STACK = 8..40 B`)
- [`src/ops/offloaded_sparse_moe/cuda/wide_expert.cuh`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert.cuh#L343-L387) (`grouped_kernel`: `3` kernels, `STACK = 8..32 B`)
- [`src/ops/ggml/ggml_linear_a8.cuh`](file:///opt/etc/Infernix/src/ops/ggml/ggml_linear_a8.cuh#L193-L205), [`src/ops/ggml/ggml_linear_a8_rows.cu`](file:///opt/etc/Infernix/src/ops/ggml/ggml_linear_a8_rows.cu), [`src/ops/ggml/ggml_linear_a8_tiled.cu`](file:///opt/etc/Infernix/src/ops/ggml/ggml_linear_a8_tiled.cu) (`unit_terms`, `rows_mma_kernel`, `rows_mma_mixed_kernel`, `tiled_mma_kernel`, `tiled_mma_mixed_kernel`: `23` kernels, `STACK = 32..56 B`)
- [`src/ops/ggml/ggml_a8_tile.cuh`](file:///opt/etc/Infernix/src/ops/ggml/ggml_a8_tile.cuh#L407-L462), [`src/ops/ggml/ggml_bf16_tile.cuh`](file:///opt/etc/Infernix/src/ops/ggml/ggml_bf16_tile.cuh#L214-L265), [`L365-L453`](file:///opt/etc/Infernix/src/ops/ggml/ggml_bf16_tile.cuh#L365-L453) (`compute_decode`, `compute`, `ggml_gate_up_a8_wide<IQ2_S>`, `ggml_gate_up_decode<IQ2_S>`, `ggml_linear_wide`: `28` kernels, `STACK = 8..56 B`)

#### Root Cause & Hardware Rationale
1. During rebase commit `3a4b8da40` (`build: make GB10/Linux the sole supported platform`) and Task `T1` (`2fe67d7be`), `check_cuda_stack.py update` was run across `infernix-serve`, blindly recording `826` spilling kernel symbols into [`tools/cuda_stack/cuda_stack_allowlist.json`](file:///opt/etc/Infernix/tools/cuda_stack/cuda_stack_allowlist.json) (up from `260` on `origin/dev`).
2. SASS inspection (`cuobjdump --dump-sass`) identified the exact register-pressure bugs across the linear, MoE, and GGUF kernels:
   - **`f16acc_mma_kernel` (`58` kernels, `16..72 B`)**: In [`f16acc_mma.cuh:137`](file:///opt/etc/Infernix/src/ops/linear/f16acc/f16acc_mma.cuh#L137), `#pragma unroll` on `for (int k_step = 0; k_step < Schedule::kMmaK; ++k_step)` unrolls all `8` steps of a `128-K` stage simultaneously, keeping `24+` shared-memory address and fragment registers live alongside `carry[2][4][4]` (`32` regs) and `partial[2][4][2]` (`16` regs) under the `96`-register (`17`-warp) / `112`-register (`F16AccSplitRegisters`) cap. Additionally, in `F16AccSplitRegisters` (`USETMAXREG.DEALLOC 32`), `nvcc` unrolls the producer `cp.async` loops (`lines 341, 348, 361`) and hoists `16` address registers across `cta_mbarrier_wait`, spilling `48..64 B` in the producer warps.
   - **`fp8_a8_tma_mma_kernel` (`44` kernels, `16..48 B`) & `fp8_a16_tma_mma_kernel` (`8` kernels, `16 B`)**: In [`fp8_a8_mma_common.cuh:95-128`](file:///opt/etc/Infernix/src/ops/linear/fp8/fp8_a8_mma_common.cuh#L95-L128), `Schedule::kMmaTokens * Schedule::kMmaRows >= 32` (`128×256`, `128` accumulator registers under the `168`-register 9-warp ceiling) unrolls all `kMmaK = 4` steps and `a_fragments[4][4]` (`16` regs), spilling `16 B` (`40..48 B` with Split-K). In [`fp8_a16_ring_mma.cuh:52-87`](file:///opt/etc/Infernix/src/ops/linear/fp8/fp8_a16_ring_mma.cuh#L52-L87), `MinBlocksPerSm = 2` (`96`-reg cap, `64` accumulator regs) allocates `unsigned b[4][2]` (`8` regs) across all row tiles instead of streaming `b0, b1` per `mma_row`.
   - **`grouped_kernel_a16` (`16` kernels, `8..40 B`) & `grouped_kernel` (`3` kernels, `8..32 B`)**: In [`wide_expert_a16.cuh:288`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert_a16.cuh#L288) (`17` warps, `96` regs), `#pragma unroll` on `for (int u = 0; u < 4; ++u)` overlaps `4` units while `int64_t sums[2][2][4]` (`32` regs) + `high`/`low` (`16` regs) + `p[2][2][4]` (`16` regs) + `b`/`scale` (`8` regs) = `72` registers are live. In [`wide_expert.cuh:346`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert.cuh#L346), `ScheduleResidual<3, ...>` (`kTerms = 3`, `96` regs) pre-loads `a_fragments[3][2][4]` (`24` regs) and `a_scales[3][2]` (`6` regs) across all `3` residual terms at once.
   - **`ggml` A8 `rows_mma` / `tiled_mma` (`23` kernels, `32..56 B`) & stage-decode (`2` kernels, `32..56 B`)**: In [`ggml_linear_a8.cuh:198`](file:///opt/etc/Infernix/src/ops/ggml/ggml_linear_a8.cuh#L198) (`unit_terms`), `#pragma unroll` on `for (int j = 0; j < kUnitSlices; ++j)` unrolls all `8` slices of GGUF block decoding (`tile_chunk<Q4_K/Q5_K/Q6_K/IQ4_XS>`) and `32` MMAs under the `128`-register (`2 CTAs/SM`) cap. In [`ggml_a8_tile.cuh:411,444`](file:///opt/etc/Infernix/src/ops/ggml/ggml_a8_tile.cuh#L411) and [`ggml_bf16_tile.cuh:369,418`](file:///opt/etc/Infernix/src/ops/ggml/ggml_bf16_tile.cuh#L369), unrolling `issue_unit` and `for (int k = 0; k < 2; ++k)` pushes `IQ2_S` (`Scaling::kHalf`) past `255` registers.

#### Required Implementation Steps
1. **In [`src/ops/linear/f16acc/f16acc_mma.cuh`](file:///opt/etc/Infernix/src/ops/linear/f16acc/f16acc_mma.cuh)**:
   - Restructure `f16acc_compute_stage` into an outer `#pragma unroll 1` loop over `for (int group = 0; group < kGroupsPerStage; ++group)` (`4` groups of `32` K-elements) and an inner `#pragma unroll` loop over `for (int step = 0; step < kChunkSteps; ++step)` (`2` steps per chunk).
   - Add `#pragma unroll 1` to the producer `cp.async` loops (`lines 341, 348, 358, 361`) so `nvcc` does not hoist `16` address registers across `cta_mbarrier_wait` in the `32`-register producer warps.
2. **In [`src/ops/linear/fp8/fp8_a8_mma_common.cuh`](file:///opt/etc/Infernix/src/ops/linear/fp8/fp8_a8_mma_common.cuh) & [`src/ops/linear/fp8/fp8_a16_ring_mma.cuh`](file:///opt/etc/Infernix/src/ops/linear/fp8/fp8_a16_ring_mma.cuh)**:
   - In `fp8_mma_compute_stage` (`kMmaTokens * kMmaRows >= 32`), use `#pragma unroll 1` on `k_step` and stream `a_fragments` in 2-token pairs (or `#pragma unroll 1` on the Split-K epilogue loop in `fp8_a8_tma_mma.cuh`) so all `44` `128×256` instances have `STACK = 0`.
   - In `fp8_a16_ring_compute_stage` (`kMmaTokens * kMmaRows >= 16 && kMinBlocksPerSm >= 2`), use `#pragma unroll 1` on `k_step` and stream `b0, b1` per `mma_row`, eliminating all `8` `fp8_a16_tma_mma_kernel` spills.
3. **In [`src/ops/offloaded_sparse_moe/cuda/wide_expert_a16.cuh`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert_a16.cuh) & [`wide_expert.cuh`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert.cuh)**:
   - Change `#pragma unroll` on `for (int u = 0; u < S::kUnitsPerKTile; ++u)` (`wide_expert_a16.cuh:288`) to `#pragma unroll 1`.
   - In `wide_expert.cuh:344`, load `b_fragments` and `b_scales` first, then stream `a_fragments[4]` and `a_scale` per `(term, mma_m)` (with `#pragma unroll 1` on `for (int term = 0; term < S::kTerms; ++term)` when `S::kTerms == 3`).
4. **In [`src/ops/ggml/ggml_linear_a8.cuh`](file:///opt/etc/Infernix/src/ops/ggml/ggml_linear_a8.cuh), [`ggml_a8_tile.cuh`](file:///opt/etc/Infernix/src/ops/ggml/ggml_a8_tile.cuh), & [`ggml_bf16_tile.cuh`](file:///opt/etc/Infernix/src/ops/ggml/ggml_bf16_tile.cuh)**:
   - Change `for (int j = 0; j < kUnitSlices; ++j)` in `unit_terms` (`ggml_linear_a8.cuh:198`) to `#pragma unroll 1`.
   - Add `#pragma unroll 1` to `issue_unit` and the 2-task decode loop `for (int k = 0; k < 2; ++k)` in `compute_decode` (`ggml_a8_tile.cuh` and `ggml_bf16_tile.cuh`), and `#pragma unroll 1` on `for (int j = 0; j < kSlices; ++j)` in `ggml_bf16_tile.cuh:215` when `S == Scaling::kMin` (`ggml_linear_wide.cu`).

#### Verification & Acceptance Criteria
```bash
ninja -C build-native -j "$(nproc)" infernix_linear_mxfp8_a16_test infernix_linear_fp8_a16_test infernix_fp8_linear_add_test infernix_ggml_linear_test infernix_offloaded_moe_cuda_test infernix_rg16_linear_test
ctest --test-dir build-native -R "infernix_linear_mxfp8_a16_test|infernix_linear_fp8_a16_test|infernix_fp8_linear_add_test|infernix_ggml_linear_test|infernix_offloaded_moe_cuda_test|infernix_rg16_linear_test" --output-on-failure
```
- **Acceptance**: `STACK: 0` and `LOCAL: 0` verified via `cuobjdump --dump-resource-usage` across all `f16acc_mma_kernel`, `fp8_a8_tma_mma_kernel`, `fp8_a16_tma_mma_kernel`, `grouped_kernel_a16`, `grouped_kernel`, `rows_mma_kernel`, `tiled_mma_kernel`, `ggml_gate_up_a8_wide`, and `ggml_gate_up_decode` instances; all numerical oracle tests pass.

---

### Task T9 (Priority P0): Eliminate VQ2 / Q4 & INT8 / BF16 Causal-Cache Attention CUDA Stack Spills (`453` Kernels)

#### Affected Files & Symbols
- [`src/ops/softmax_attention/dense/causal_cache/vq2/grouped.cuh`](file:///opt/etc/Infernix/src/ops/softmax_attention/dense/causal_cache/vq2/grouped.cuh#L102-L113), [`L440-L685`](file:///opt/etc/Infernix/src/ops/softmax_attention/dense/causal_cache/vq2/grouped.cuh#L440-L685) (`vq_kv_grouped_kernel`: `174` kernels, `STACK = 24..152 B`)
- [`src/ops/softmax_attention/dense/causal_cache/vq2/prompt.cuh`](file:///opt/etc/Infernix/src/ops/softmax_attention/dense/causal_cache/vq2/prompt.cuh#L460-L860) (`vq_prompt_kernel`: `179` kernels, `STACK = 8..72 B`)
- [`src/ops/softmax_attention/dense/causal_cache/int8/grouped_mma.cuh`](file:///opt/etc/Infernix/src/ops/softmax_attention/dense/causal_cache/int8/grouped_mma.cuh) (`int8_kv_grouped_mma_kernel`: `63` kernels, `STACK = 8..56 B`)
- [`src/ops/softmax_attention/dense/causal_cache/bf16/grouped_mma.cuh`](file:///opt/etc/Infernix/src/ops/softmax_attention/dense/causal_cache/bf16/grouped_mma.cuh) (`bf16_kv_grouped_mma_kernel`: `18` kernels, `STACK = 8..24 B`)
- [`src/ops/softmax_attention/dense/causal_cache/int8/fast_tiled_mma.cuh`](file:///opt/etc/Infernix/src/ops/softmax_attention/dense/causal_cache/int8/fast_tiled_mma.cuh) (`causal_attention_prompt_i8_fast_kernel`: `15` kernels, `STACK = 8..40 B`)
- [`src/ops/softmax_attention/dense/causal_cache/int8/grouped_pipelined.cuh`](file:///opt/etc/Infernix/src/ops/softmax_attention/dense/causal_cache/int8/grouped_pipelined.cuh) (`int8_kv_grouped_pipelined_kernel`: `4` kernels, `STACK = 8 B`)

#### Root Cause & Hardware Rationale
1. **VQ2 / Q4 Grouped Decode/Verify Attention (`vq_kv_grouped_kernel`, `174` kernels, `24..152 B`)**:
   - Upstream's `grouped.cuh` dropped `origin/dev`'s `vq_kv_grouped_smem_bytes` check in `__launch_bounds__` (forcing `MinBlocksPerSm = 2` -> `128` regs even when shared memory > `50 KiB` only fits `1` block/SM on `sm_121a`), kept `af[WarpRowTiles][GroupKc][4]` (`32` regs) and `bool keeps_row[WarpRowTiles][2]` in registers across groups, and unrolled `for (int j = 0; j < PVKs; ++j)` across `RowTiles = 3..8`.
   - For the remaining `32` 8-warp instances at `RowTiles = 3, 4` (`WarpRowTiles = 2`), `float q_scale_r[2][2][4]` holds `16` registers in every thread alongside `float acc[RowTiles][2][2][4]` (`48–64` regs). Moving `q_scale` to `q_scale_shared` whenever `Wc == 16 || RowTiles >= 3` frees `16` registers and eliminates the remaining `8..16 B` spills.
2. **VQ2 / Q4 Prompt Attention (`vq_prompt_kernel`, `179` kernels, `8..72 B`)**:
   - Upstream's `prompt.cuh` accumulated PV in FP16 (`unsigned h[PassDBlocks][2][2]`) before unpacking to `float acc[16][2][4]` (`128` regs). Restoring direct FP32 Tensor Core accumulation (`mma_f16` on `acc`) eliminates `91` spills and cuts max stack from `72 B -> 32 B`.
   - The remaining `88` spills (`8..32 B` at `REG = 255`) occur because `acc[16][2][4]` (`128` regs) is live across the tile loop while `finish_tile(v)` unrolls `Bc / Warps = 8` codebook expansions and `qk_softmax(v)` unrolls all `Groups = 4` × `QKNt = 8` × `GroupKc = 4` MMAs alongside `score[8][4]` (`32` regs), `af[4][4]` (`16` regs), and `pa[4][4]` (`16` regs). Using `#pragma unroll 1` on `for (int i = 0; i < Bc / Warps; ++i)` in `finish_tile` and `for (int grp = 0; grp < Groups; ++grp)` in `qk_softmax` (plus streaming `af[4]` per `kk`) keeps live registers below `255`.
3. **INT8 & BF16 Grouped / Fast-Prompt Attention (`100` kernels, `8..56 B`)**:
   - Apply the same `q_scale_shared`, per-group `bf`, `#pragma unroll 1` on outer group/PV loops, and SMEM-aware `__launch_bounds__` (`> 49000 B -> 1 CTA/SM`) to `int8/grouped_mma.cuh`, `bf16/grouped_mma.cuh`, and `int8/fast_tiled_mma.cuh` (preserving `MinBlocksPerSm = 2` only where 2 CTAs physically fit in SMEM and where any residual frame is measured faster on GB10).

#### Required Implementation Steps
1. Update [`src/ops/softmax_attention/dense/causal_cache/vq2/grouped.cuh`](file:///opt/etc/Infernix/src/ops/softmax_attention/dense/causal_cache/vq2/grouped.cuh) to use `q_scale_shared` whenever `Wc == 16 || RowTiles >= 3`, `#pragma unroll 1` on `pv_step` when `RowTiles >= 3`, per-group `bf[GroupKc][2]`, and on-demand `keeps_row`.
2. Update [`src/ops/softmax_attention/dense/causal_cache/vq2/prompt.cuh`](file:///opt/etc/Infernix/src/ops/softmax_attention/dense/causal_cache/vq2/prompt.cuh) to use direct `mma_f16` into `acc`, `#pragma unroll 1` on the `finish_tile` codebook loop and `qk_softmax` `grp` loop, and stream `af[4]` per `kk`.
3. Inspect and apply matching register-pressure fixes to [`int8/grouped_mma.cuh`](file:///opt/etc/Infernix/src/ops/softmax_attention/dense/causal_cache/int8/grouped_mma.cuh), [`bf16/grouped_mma.cuh`](file:///opt/etc/Infernix/src/ops/softmax_attention/dense/causal_cache/bf16/grouped_mma.cuh), and [`int8/fast_tiled_mma.cuh`](file:///opt/etc/Infernix/src/ops/softmax_attention/dense/causal_cache/int8/fast_tiled_mma.cuh).
4. Verify via `nvlink` + `cuobjdump --dump-resource-usage` and run all attention unit/oracle tests.

#### Verification & Acceptance Criteria
```bash
ninja -C build-native -j "$(nproc)" infernix_vq_attention_test infernix_softmax_attention_test
./build-native/tests/infernix_vq_attention_test
./build-native/tests/infernix_softmax_attention_test
```
- **Acceptance**: Zero preventable stack spills in `vq_kv_grouped_kernel` and `vq_prompt_kernel`, reduction of `int8`/`bf16` attention spills to only measured occupancy-load-bearing cases, and `100%` pass on `infernix_vq_attention_test` and `infernix_softmax_attention_test`.

---

### Task T10 (Priority P1): Eliminate Remaining Quantized Linear (`Q8`/`Q5`/`Q4`), GDN & RoPE `sincosf` Stack Frames (`116` Kernels)

#### Affected Files & Symbols
- [`src/ops/kernel/rope.cuh`](file:///opt/etc/Infernix/src/ops/kernel/rope.cuh#L58-L122), [`src/ops/common/dflash_rope.cuh`](file:///opt/etc/Infernix/src/ops/common/dflash_rope.cuh), [`src/ops/qsa/qsa_prepare.cu`](file:///opt/etc/Infernix/src/ops/qsa/qsa_prepare.cu), [`src/ops/qsa/qsa.cu`](file:///opt/etc/Infernix/src/ops/qsa/qsa.cu), [`src/ops/qsa/materialize.cu`](file:///opt/etc/Infernix/src/ops/qsa/materialize.cu) (`rope_fixed_kernel` `42`, `qsa_prepare_kernel` `10`, `rope_fixed_split_kernel` `4`, `materialize` `3`, `rope_generic_kernel` `2`, `qsa` `2`: `63` kernels with `STACK = 32 B`)
- [`src/ops/linear/q8/q8_a16_stream.cuh`](file:///opt/etc/Infernix/src/ops/linear/q8/q8_a16_stream.cuh), [`src/ops/linear/q8/q8_a16_sliced_k_mma.cuh`](file:///opt/etc/Infernix/src/ops/linear/q8/q8_a16_sliced_k_mma.cuh), [`src/ops/linear/q8/q8_a16_mma.cuh`](file:///opt/etc/Infernix/src/ops/linear/q8/q8_a16_mma.cuh) (`18` kernels, `STACK = 8..32 B`)
- [`src/ops/linear/q4/q4_a16_sliced_k_mma.cuh`](file:///opt/etc/Infernix/src/ops/linear/q4/q4_a16_sliced_k_mma.cuh), [`src/ops/linear/q5/q5_a16_direct_simt.cuh`](file:///opt/etc/Infernix/src/ops/linear/q5/q5_a16_direct_simt.cuh), [`src/ops/linear_swiglu/q4/q4_linear_swiglu_w4a8.cu`](file:///opt/etc/Infernix/src/ops/linear_swiglu/q4/q4_linear_swiglu_w4a8.cu) (`7` kernels, `STACK = 8..32 B`)
- [`src/ops/gdn_input_proj/q8/q8_gdn_input_gemm_splitk.cu`](file:///opt/etc/Infernix/src/ops/gdn_input_proj/q8/q8_gdn_input_gemm_splitk.cu), [`src/ops/gdn/gdn_projected_conv.cu`](file:///opt/etc/Infernix/src/ops/gdn/gdn_projected_conv.cu) (`11` kernels, `STACK = 8..16 B`)
- [`src/ops/ggml/ggml_linear.cu`](file:///opt/etc/Infernix/src/ops/ggml/ggml_linear.cu), [`src/ops/ggml/ggml_linear_tiled.cu`](file:///opt/etc/Infernix/src/ops/ggml/ggml_linear_tiled.cu), [`src/ops/offloaded_sparse_moe/cuda/moe_layer.cu`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/moe_layer.cu) (`23` kernels, `STACK = 8..72 B`)

#### Root Cause & Hardware Rationale
1. **CUDA `sincosf()` Payne-Hanek Slowpath (`63` kernels, `STACK = 32 B`)**:
   - SASS inspection of `rope_fixed_kernel` (`REG = 20, STACK = 32`) revealed that CUDA's `sincosf(x, &s, &c)` inlines `__internal_trig_reduction_slowpathf` whenever it cannot statically prove `|x| < 105615.0f` (`FSETP.GE.AND P0, PT, |R9|, 105615`), allocating `int result[7]` (`32 B` local stack array) indexed dynamically by the exponent.
   - In `PreparedRopeCoefficients::sincos_axis` ([`rope.cuh:105-111`](file:///opt/etc/Infernix/src/ops/kernel/rope.cuh#L105-L111)) and `dflash_rope_sincos` ([`dflash_rope.cuh`](file:///opt/etc/Infernix/src/ops/common/dflash_rope.cuh)), `reduced` is **already reduced modulo $2\pi$ in FP64** (`|reduced| <= \pi < 105615.0f`), so the Payne-Hanek slowpath branch is 100% unreachable dead code! Even in `fixed_sincos`, reducing `angle` modulo $2\pi$ or bounding `|reduced| < 105615.0f` before `sincosf` (or using a branch-guarded fast range reduction verified against the FP64 oracle) eliminates the `32 B` local-memory frame across all `63` RoPE/QSA kernels.
2. **Quantized Linear (`Q8`/`Q5`/`Q4`), GDN & GGUF Exact Kernels (`53` kernels, `8..72 B`)**:
   - Small `8..32 B` spills at `REG = 128` (`2 CTAs/SM`) in `q8_a16_stream_kernel` (`Columns = 5, 6`), `q8_a16_sliced_k_mma_kernel`, `q4_a16_sliced_k_mma_kernel`, `q8_gdn_input_gemm_splitk`, `ggml_linear.cu`, `ggml_linear_tiled.cu`, and `moe_layer.cu` come from unrolled inner decode/accumulation loops exceeding the 128-register ceiling by 2–8 registers.

#### Required Implementation Steps
1. Inspect and eliminate the dead Payne-Hanek slowpath stack frame in [`rope.cuh`](file:///opt/etc/Infernix/src/ops/kernel/rope.cuh), [`dflash_rope.cuh`](file:///opt/etc/Infernix/src/ops/common/dflash_rope.cuh), and [`qsa`](file:///opt/etc/Infernix/src/ops/qsa/) while verifying bit-exact / oracle compliance on `infernix_rope_test` and `infernix_qsa_test`.
2. Inspect register pressure and loop unrolling in the remaining `q8`, `q5`, `q4`, `gdn`, `ggml`, and `moe_layer` kernels and eliminate preventable `STL`/`LDL` spills.

#### Verification & Acceptance Criteria
```bash
ninja -C build-native -j "$(nproc)" infernix_rope_test infernix_qsa_test infernix_linear_q8_a16_test infernix_linear_q4_a16_test infernix_linear_q5_a16_test infernix_gdn_test infernix_ggml_linear_test
ctest --test-dir build-native -R "rope|qsa|linear_q8|linear_q4|linear_q5|gdn|ggml" --output-on-failure
```
- **Acceptance**: All `rope`, `qsa`, `q8`, `q4`, `q5`, `gdn`, and `ggml` tests pass with zero preventable stack frames.

---

### Task T11 (Priority P0): Audit, Regenerate & Ratchet `tools/cuda_stack/cuda_stack_allowlist.json` on GB10 (`sm_121a`)

#### Affected Files & Symbols
- [`tools/cuda_stack/cuda_stack_allowlist.json`](file:///opt/etc/Infernix/tools/cuda_stack/cuda_stack_allowlist.json)
- [`tools/cuda_stack/check_cuda_stack.py`](file:///opt/etc/Infernix/tools/cuda_stack/check_cuda_stack.py)
- [`docs/gb10/20261029-changelog.md`](file:///opt/etc/Infernix/docs/gb10/20261029-changelog.md)

#### Required Implementation Steps
1. After completing Tasks `T8`–`T10`, link `infernix-serve` and inspect all remaining symbols with `STACK > 0` or `LOCAL > 0`:
   ```bash
   ninja -C build-native -j "$(nproc)" infernix-serve
   /opt/etc/Infernix/.venv-convert/bin/python3 tools/cuda_stack/check_cuda_stack.py update build-native/apps/infernix-serve
   ```
2. Verify that every remaining entry in [`tools/cuda_stack/cuda_stack_allowlist.json`](file:///opt/etc/Infernix/tools/cuda_stack/cuda_stack_allowlist.json) belongs strictly to:
   - **`--lossless` software-oracle kernels** (`lossless_gemm_kernel`, `lossless_fallback_kernel`, `lossless_flash`, `lossless_attention`, `lossless_moe`, `lossless_ops`, `lossless_gdn`, `lossless_rope`: `~61` kernels that intentionally use multi-word Kulisch / FP64 software accumulator arrays in local memory), or
   - **Documented, measured occupancy-load-bearing exceptions** where `MinBlocksPerSm = 2` is proven faster on GB10 than `MinBlocksPerSm = 1`.
3. Run the one-way ratchet gate (`infernix_cuda_stack_check`) and record the before/after allowlist reduction (`826 -> <100` entries) in [`docs/gb10/20261029-changelog.md`](file:///opt/etc/Infernix/docs/gb10/20261029-changelog.md).

#### Verification & Acceptance Criteria
```bash
ctest --test-dir build-native -R infernix_cuda_stack_check --output-on-failure
git diff --stat tools/cuda_stack/cuda_stack_allowlist.json
```
- **Acceptance**: `infernix_cuda_stack_check` passes with `0` unallowlisted spills and `cuda_stack_allowlist.json` reduced from `826` entries to the minimal audited set.

---

### Task T12 (Priority P2): Cumulative End-to-End KV-Matrix & Multi-Concurrency Serving Gate

#### Affected Files & Symbols
- [`bench/run_kv_matrix_bench.sh`](file:///opt/etc/Infernix/bench/run_kv_matrix_bench.sh)
- [`docs/gb10_roadmap.md`](file:///opt/etc/Infernix/docs/gb10_roadmap.md)

#### Required Verification Steps
After completing Tasks `T1`–`T11`, run the full verification and benchmarking suite on GB10:
```bash
# 1. One-way CUDA stack-spill ratchet gate (must report 0 new STACK>0 or LOCAL>0 kernels)
ctest --test-dir build-native -R infernix_cuda_stack_check --output-on-failure

# 2. Full 379-test CTest suite
ctest --test-dir build-native --output-on-failure --parallel 8

# 3. End-to-end KV-matrix benchmark across Qwen3.8-Flash-Next artifacts (nvfp4, dense8, iq3s)
./bench/run_kv_matrix_bench.sh /tmp/kvmatrix_20261029

# 4. Multi-concurrency serving benchmark (c = 1, 2, 4 at d0, d4096, d8192 via tools/run_server.sh + llama-benchy)
git diff --check
```
- Record all favorable and unfavorable findings, outcome distributions, and worst changes in [`docs/gb10_roadmap.md`](file:///opt/etc/Infernix/docs/gb10_roadmap.md).
