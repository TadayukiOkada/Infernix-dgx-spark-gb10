# GB10 Branch Changelog & Rebase Guide (`origin/main` → `HEAD`, 2026-10-28)

Comparison between `origin/main` (`d8e52a85a`, `test(qwen3_5): nonfinite-withdraw expects the exact route's grid publication`) and `HEAD` (`9614401db`, branch `dev-rebased-20261028`, `build: make GB10/Linux the sole supported platform`).

- **Total repository diff (`origin/main..HEAD`)**: **706 files changed**, **62,792 insertions(+)**, **12,086 deletions(-)**.
- **Tracked in this changelog (excluding `cuda-kernel-toolkit/`)**: **638 files changed** — **157 Added**, **467 Modified**, **14 Deleted**.
- **Intentionally excluded**: `cuda-kernel-toolkit/` (**68 files** across `995ef7448` and subsequent toolkit updates).

---

## Executive Summary: Why These Changes Exist (GB10 Hardware Architecture vs. RTX 5090)

When rebasing onto future `origin/main` commits, keep these six hardware facts about the **NVIDIA GB10 Grace Blackwell Superchip (`sm_121a`)** in mind — nearly every divergence from `origin/main` flows directly from one of them:

1. **48 SMs (`sm_121a`) vs. 170 SMs (`sm_120a` RTX 5090)**:
   - Grid sizes, split-K heuristics, and persistent-kernel waves tuned for 170 SMs severely oversubscribe 48 SMs (causing tail-wave penalties, L2 thrashing, and split-reduction overhead).
   - Register pressure and warps-per-CTA must be tuned for 48 SMs and GB10's 32–48 MiB L2 cache.
2. **128 GiB Coherent Unified LPDDR5X (~256–273 GB/s) vs. Discrete 32 GiB GDDR7 (1,792 GB/s) + PCIe Host RAM**:
   - On GB10 (`cudaDeviceProp::integrated == 1`), CPU and GPU share **one physical LPDDR5X memory pool**.
   - Allocating pinned host memory for offloaded MoE experts *and* a VRAM expert cache double-spends the same physical RAM and wastes DRAM bandwidth copying weights from LPDDR5X to LPDDR5X. Therefore, GB10 defaults to **direct on-device expert residency (`LoadOptions::device_experts = true`)** with zero staging/frame pools and full upfront KV backing.
   - When the pinned-host expert path is explicitly forced (`INFERNIX_EXPERTS_ON_GPU=0`), `VramBudget` and `memory_plan` must account for Linux's reclaimable OS page cache (`MemAvailable - MemFree`) and subtract `shared_host_claim()` so the unified pool is not spent twice.
   - Furthermore, CPU expert offload (`CpuExpertTeam`) reads from the same LPDDR5X pool as the GPU, so splitting decode MoE across GPU + CPU is strictly slower on GB10 (`23.2–26.5 tok/s` vs `35.8 tok/s` GPU-only) and is disabled by default when `device_experts` is active.
3. **Grace 20-Core Asymmetric big.LITTLE ARMv9.2 CPU (10 Cortex-X925 + 10 Cortex-A725)**:
   - Host code cannot use x86 intrinsics (`<immintrin.h>`, `_mm_sfence`, `_mm_pause`, AVX2/AVX-512). It uses AArch64 release fences (`DMB ISHST`), `yield`, and ARMv8.6-A NEON + `FEAT_I8MM` (`vusdotq_s32`).
   - Cortex-A725 efficiency cores have ~28% lower capacity (`718` vs `997`) and ~40% lower sustained throughput; synchronous thread-pool barriers (`CpuExpertTeam`) stall on the slowest core unless pinned to the 10 Cortex-X925 performance cores when `workers <= 10`.
4. **FP16 Tensor Core MMA with FP32 Accumulation (`m16n8k16.f32.f16.f16.f32`) Runs at 1:1 Parity with FP16 Accumulation on `sm_121a`**:
   - On RTX 5090 (`sm_120a`), `f32`-accumulating FP16 HMMA runs at half the throughput of `f16` accumulation, so attention (`mxfp8_tiled_mma.cuh`, `int8/fast_tiled_mma.cuh`) and GDN recurrence (`recurrence.cu`) accumulated into FP16 intermediates (`partial_h`) with per-tile scale scans and exponent shifts (`tile_shift`).
   - On GB10 (`sm_121a`), FP32 accumulation runs at **full rate** (`125.3` vs `125.4 TFLOP/s`). Accumulating directly into FP32 registers eliminates `partial_h`, eliminates per-tile `Nvfp4G16TiledValues::tile_shift` warp reductions (which previously penalized `nvfp4` and `k8v4` KV caches), and improves numerical accuracy.
5. **`sm_121a` SASS Instruction Selection Quirks (`__vsub4` Emulation & FP32 FMA Contraction)**:
   - `sm_121a` has no native hardware packed-byte SIMD adder/subtractor, so `__vsub4` lowers to a ~10-instruction scalar extraction/reassembly sequence. Where lanes are guaranteed non-zero (`q4_lloyd_codec.cuh`), two's-complement negation via carry-free `(mag ^ mask) + (mask & 0x01010101u)` emits 2 instructions (`LOP3.LUT + IADD3`) and is bit-exact. Where lanes can be `0` (`vq2_codec.cuh`), `__vsub4` must be preserved so `0` does not overflow into adjacent bytes.
   - `ptxas` on `sm_121a` contracts paired `__fmul2_rn` + `__fadd2_rn` into `FFMA` (`f32x2`), diverging by 1 ULP from scalar IEEE-754 rounding in `vq2_kernel.cuh`'s `exact_reducible` settle check; scalar `fmul.rn.f32` / `fadd.rn.f32` is kept by default (`INFERNIX_VQ2_PAIRED_FP32=0`).
6. **Memory-Bandwidth-Bound Decode Amortization & Speculative Decoding Economics**:
   - Because GB10 decode is tightly bound by LPDDR5X weight reads, verifying `C = 8` tokens in one forward pass takes only ~`2.6x` the time of `C = 1` (amortizing weight reads across all draft columns). Consequently, speculative decoding (MTP, Pair drafting, Multi-row drafting at `batch >= 3`, and zero-neural-cost N-gram copy proposals) yields much higher net throughput gains when configured with lower width-cost thresholds (`kWidthCost = 0.25`, `kNgramWidthCost = 0.12`, `kPairWidthCost = 0.15`, `kPairMaxDrafts = 6`, max n-gram draft `31`).

---

## 1. Build System, Platform Lock & Toolchain (`CMakeLists.txt`, `CMakePresets.json`)

### Why & Key Mechanisms
- **GB10/Linux (`aarch64` + `sm_121a`) as the sole supported target**: Removed Windows/MSVC (`_WIN32`, DXGI, DXCore, WinSock, IOCP) and x86 (`sm_120a` RTX 5090) build branches so the build graph, compiler flags (`-mcpu=native`), and CUDA architecture (`121a-real`) target GB10 directly without cross-platform shims.
- **Parallelized device linking (`INFERNIX_LINK_JOBS=8`)**: Because `infernix_ops` uses relocatable device code (`-rdc=true`), `nvlink` takes ~56s per consumer binary. Raising `INFERNIX_LINK_JOBS` from `1` to `8` (and splitting multi-instantiation `.cu` files into parallel compilation units) prevents 20 Grace CPU cores from sitting 63% idle during link phases.

### Changed Files
- `CMakeLists.txt` (`M`):
  - Enforces `CMAKE_SYSTEM_NAME STREQUAL "Linux"` and `CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$"`.
  - Sets `CMAKE_CUDA_ARCHITECTURES "121a-real"` unconditionally.
  - Replaces x86 `-march=native` and MSVC flags with `-O3 -mcpu=native` and `--expt-relaxed-constexpr -Xptxas=-v`.
  - Raises default `INFERNIX_LINK_JOBS` from `1` to `8`.
- `CMakePresets.json` (`M`):
  - Removes `windows-base` and all `windows-*` configure/build/test presets.
  - Updates `linux-base` description to `64-bit Linux aarch64, Ninja, CUDA 13, sm_121a (GB10)`.

---

## 2. Public Headers, Core Runtime & OS/Hardware Abstraction (`include/`, `src/core/`, `src/runtime/`)

### Why & Key Mechanisms
- **Unified-memory single-pool accounting (`vram_budget.{cpp,h}`, `host_memory.{cpp,h}`)**:
  - On Linux unified memory (`props_.integrated != 0`), `cudaMemGetInfo` only reports currently unallocated pages (`MemFree`), ignoring tens of GiB of reclaimable OS page cache (`Cached`/`Buffers`/`SReclaimable`) that the kernel frees on demand when CUDA allocates.
  - `reclaimable_pool_bytes()` reads `/proc/meminfo` (`MemAvailable - MemFree`) and clamps to cgroup v2 (`memory.max - memory.current`) so `VramSnapshot::reclaimable` reflects real allocatable capacity without touching swap.
  - `shared_host_claim()` tracks pinned/mapped host allocations (`record_shared_host_claim` / `release_shared_host_claim` in `host_memory.cpp`) so `VramControl` and `size_expert_frames` subtract host claims from the unified pool instead of spending the same LPDDR5X twice.
  - `SpillGuard::end()` returns `0` on `integrated` devices because unified memory does not silently spill device allocations over PCIe into a separate host pool.
  - Auto `--vram-headroom-mib` defaults to `0` on unified memory (vs `1024 MiB` on discrete GPUs).
- **Grace big.LITTLE CPU topology discovery (`cpu_topology.{cpp,h}`)**:
  - Adds `core::hybrid_cpu()` and `core::performance_cpus()`, probing `/sys/devices/system/cpu/cpu*/cpu_capacity`, `cpufreq/cpuinfo_max_freq`, and ARM64 `midr_el1` part numbers (`0xd85` Cortex-X925 vs `0xd87` Cortex-A725) inside the caller's `sched_getaffinity` mask.
- **AArch64 pinned-memory store ordering (`device.cu`)**:
  - Replaces x86 `<immintrin.h>` / `_mm_sfence()` in `publish_pinned_word` with `std::atomic_thread_fence(std::memory_order_release)` (lowering to `DMB ISHST` on AArch64) before writing the ready flag observed by mapped GPU threads.
- **Batched Linux kernel AIO (`read_only_file_posix.cpp`)**:
  - Pre-allocates `blocks`, `submit_batch`, `events`, and `idle` vectors inside `AioContext` (zero per-call heap allocations), submits all queued `iocb`s in a single `SYS_io_submit` syscall, and harvests `min(in_flight, max(1, slots / 4))` completions per `SYS_io_getevents` call.
- **Configurable N-gram cache capacity (`include/infernix/types.h`, `include/infernix/engine.h`, `src/runtime/config.cpp`)**:
  - Removes deprecated `vram_past_budget` from `LoadOptions` and `EngineOptions`.
  - Adds `LoadOptions::ngram_cache_bytes` and `EngineOptions::ngram_cache_bytes` (default `64 MiB`, env `INFERNIX_NGRAM_CACHE_MIB`), and disables the `mtp_expert_slots` pool in `src/runtime/config.cpp` when `device_experts` is enabled.

### Changed Files
- `include/infernix/engine.h` (`M`): Removes `vram_past_budget` from `EngineOptions`; adds `ngram_cache_bytes = 64ULL << 20`.
- `include/infernix/types.h` (`M`): Removes `vram_past_budget` from `LoadOptions`; adds `ngram_cache_bytes = 64ULL << 20`.
- `src/core/CMakeLists.txt` (`M`): Adds `gpu_telemetry.cpp` to `infernix_core`; removes `dxcore_adapter.cpp` and `dxgi` link dependency.
- `src/core/cpu_topology.cpp` (`M`), `src/core/cpu_topology.h` (`M`): Removes Windows `GetLogicalProcessorInformationEx` branches; adds `hybrid_cpu()` and `performance_cpus()` for Grace Cortex-X925 vs Cortex-A725 detection.
- `src/core/device.cu` (`M`): Replaces x86 `_mm_sfence()` with portable `std::atomic_thread_fence(std::memory_order_release)` in `publish_pinned_word`.
- `src/core/direct_read_queue.cpp` (`M`): Strips `#ifdef _WIN32` IOCP / `ReadFileScatter` implementation, retaining the POSIX worker implementation.
- `src/core/dxcore_adapter.cpp` (`D`), `src/core/dxcore_adapter.h` (`D`): Deleted Windows DXCore/WDDM adapter queries.
- `src/core/gpu_telemetry.cpp` (`A`), `src/core/gpu_telemetry.h` (`A`): Dynamic `libnvidia-ml.so.1` NVML telemetry reader (`sample_gpu_telemetry()` returning `power_watts`, `temperature_c`, `sm_clock_mhz`, `mem_clock_mhz`).
- `src/core/host_load.cpp` (`M`): Strips `#ifdef _WIN32` `GetSystemTimes` branch, retaining Linux `/proc/stat` sampling.
- `src/core/host_memory.cpp` (`M`), `src/core/host_memory.h` (`M`): Strips Windows `VirtualAlloc`/`VirtualLock`; adds `record_shared_host_claim()`, `release_shared_host_claim()`, and `shared_host_claim()` atomic ledger for unified-memory host allocations; uses `madvise(..., MADV_HUGEPAGE)` and `/proc/meminfo` (`MemAvailable` + cgroup v2 `memory.max`/`memory.current`).
- `src/core/mapped_file_win.cpp` (`D`), `src/core/read_only_file_win.cpp` (`D`): Deleted Windows file I/O implementations.
- `src/core/read_only_file_posix.cpp` (`M`): Pre-allocated `AioContext` buffers, batched `SYS_io_submit`, and coalesced `SYS_io_getevents` harvesting.
- `src/core/vram_budget.cpp` (`M`), `src/core/vram_budget.h` (`M`): Adds `VramSnapshot::reclaimable`, `VramBudget::integrated()`, `reclaimable_pool_bytes()`, unified-memory accounting in `VramControl`, `SpillGuard` no-op on `integrated`, and dynamic `dlopen("libnvidia-ml.so.1")` display/memory queries.
- `src/runtime/config.cpp` (`M`): Plumbs `load.ngram_cache_bytes = options.ngram_cache_bytes`; disables `mtp_expert_slots` when `load.device_experts` is active.
- `src/runtime/engine/context_cache/context_cost.cpp` (`M`): Removes `#ifdef _WIN32` `_getpid()` branch in `write_document_atomic`.
- `src/runtime/engine/context_cache/context_cost_defaults.cpp` (`M`): Adds the `"nvidia-gb10-sm121"` `ContextCostMachinePreset` with measured GB10 transfer costs (`ns_per_byte_q32 = {183'965'911, 220'829'326, 30'671'379}`) and 4 prefill signatures.
- `src/runtime/engine/context_cache/disk_store.cpp` (`M`): Removes `#ifdef _WIN32` file I/O branches, keeping POSIX `open`/`pread`/`pwrite`/`fsync`.

---

## 3. CLI & Serving Applications (`apps/`, `src/serve/`)

### Why & Key Mechanisms
- Exposes `--ngram-cache-mib` across CLI, server, and bench binaries; raises the maximum `--ngram-draft-tokens` limit from `15` to `31` (matching `kMaxDraftTokens` on GB10 where wider copy drafts are cheap to verify); removes deprecated `--vram-past-mib` and Windows socket/console shims.

### Changed Files
- `apps/common/app_utils.h` (`M`), `apps/common/cli_options.cpp` (`M`), `apps/common/cli_options.h` (`M`): Removes Windows `<io.h>`/`_isatty` and `--vram-past-mib`; adds `--ngram-cache-mib <n>` (`INFERNIX_NGRAM_CACHE_MIB`) and raises `--ngram-draft-tokens` max from `15` to `31`.
- `apps/infernix-cli/main.cpp` (`M`), `apps/infernix-perplexity/main.cpp` (`M`), `apps/infernix-serve/main.cpp` (`M`): Removes Windows console codepage setup; plumbs `ngram_cache_bytes` and removes `vram_past_budget`.
- `src/serve/CMakeLists.txt` (`M`), `src/serve/http_server.cpp` (`M`), `src/serve/openai_parsing.cpp` (`M`), `src/serve/serve_app.h` (`M`), `src/serve/generation_service.cpp` (`M`):
  - Drops `ws2_32` link dependency and Winsock initialization in `http_server.cpp`.
  - Raises OpenAI `ngram_draft_tokens` API validation limit from `15` to `31` in `openai_parsing.cpp`.
  - Adds `ngram_cache_bytes` to `ServeOptions` (`serve_app.h`) and wires `engine_options.ngram_cache_bytes = options_.ngram_cache_bytes` in `generation_service.cpp`.

---

## 4. CUDA Kernel Launch Infrastructure & Register/Bounds Tuning (`src/ops/kernel/`, `src/ops/launcher/`, `src/ops/common/`)

### Why & Key Mechanisms
- **Zero-spill enforcement & explicit launch bounds (`kernel/launch.cuh`)**: Adds `INFERNIX_LAUNCH_BOUNDS(max_threads, min_blocks)` and `INFERNIX_LAUNCH_BOUNDS_MAX(max_threads)` macros so kernels can constrain `ptxas` register allocation on `sm_121a` without local-memory stack spills (`infernix_cuda_stack_check` enforces zero `STACK > 0` or `LOCAL > 0` across `infernix-serve`).
- **Eliminating redundant runtime `cudaFuncSetAttribute` overhead (`launcher/kernel_launcher.h`)**: Skips calling `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)` when `smem_bytes <= 48 KiB` (already covered by the default static hardware carveout), calling it only once per kernel when opting into `> 48 KiB` dynamic shared memory.

### Changed Files
- `src/ops/kernel/launch.cuh` (`M`): Adds `INFERNIX_LAUNCH_BOUNDS` and `INFERNIX_LAUNCH_BOUNDS_MAX`.
- `src/ops/launcher/kernel_launcher.h` (`M`): Guards `cudaFuncSetAttribute` behind `smem_bytes > kDefaultSharedMemBytes` (`48 * 1024`) with per-kernel cache tracking.
- `src/ops/common/elemwise.cu` (`M`), `src/ops/common/norm.cu` (`M`), `src/ops/common/rope.cu` (`M`), `src/ops/common/vision.cu` (`M`): Adds explicit launch bounds and vectorizes/unrolls elementwise, RMSNorm, RoPE, and vision helper kernels for `sm_121a`.
- `src/ops/rows/rows.cu` (`M`): Optimizes row gather/scatter/split indexing and adds launch bounds.

---

## 5. Dense & Quantized Linear Projections (`src/ops/linear/`, `src/ops/ggml/`)

### Why & Key Mechanisms
- **GB10 (`sm_121a`, 48 SMs, 32–48 MiB L2) schedule & tile tuning**:
  - Retuned tile shapes (`M/N/K` warps, pipeline stages, split-K thresholds, and token route boundaries `max_tokens`) across all 14 linear quantization families (`bf16`, `dense8`, `fp8`, `int4`, `iq2kl`, `iq2s`, `iq2xs`, `iq2xxs`, `iq3s`, `iq4xs`, `mxfp4`, `nvfp4`, `q6k`, `q8k`) specifically for 48 SMs and LPDDR5X bandwidth.
  - Added shape-specialized translation units (`src/ops/linear/*/shapes/n<N>_k<K>.{cu,cpp}`) for real Qwen3.5 / Qwen3.8-Flash-Next (`qwen4_exp`) model dimensions (`n512_k2048`, `n1024_k2048`, `n1536_k2048`, `n2048_k2048`, `n3072_k2048`, `n4096_k2048`, `n6144_k2048`, `n8192_k2048`, `n10240_k2048`, `n12288_k2048`, `n16384_k2048`, `n17408_k2048`, `n32768_k2048`, `n2048_k1536`, `n2048_k4096`, `n2048_k6144`, `n2048_k8192`, `n2048_k16384`, `n248320_k2048`, etc.).
- **New native `sm_121a` Tensor Core kernels**:
  - `nvfp4/kernels/a4_gemm_64x128.cuh` (`A`): 4-warp (`128`-thread) `64x128` NVFP4 block-scaled MMA (`m16n8k64.e2m1`) GEMM kernel with 2-stage cp.async pipeline for medium token batches where `128x128` under-fills 48 SMs.
  - `nvfp4/kernels/a8_gemm_128x128.cuh` (`A`): `128x128` MXFP8×NVFP4 block-scaled GEMM kernel (`a8_gemm_kernel_128x128_s2`).
  - `mxfp4/kernels/a8_gemm_16x64.cuh` (`A`): Low-M (`16x64`) MXFP8×MXFP4 Tensor Core kernel for small prefill / speculative verification batches (`M <= 16`) on 48 SMs.
  - `iq3s/kernels/dp4a_gemv.cuh` (`A`) and `iq3s/kernels/a8_gemm.cuh` (`M`): Dedicated Q8_1×IQ3_S `dp4a` GEMV and `16x64` / `64x64` MXFP8 MMA schedules (`a8_gemm_kernel_16x64_w2_s2`, `a8_gemm_kernel_64x64_w4_s2`) for IQ3_S weights (such as the `248320 x 2048` LM head and dense layers).
- **Parallel compilation of `ggml` IQ3_S/IQ2_S/IQ2_KL (`src/ops/ggml/`)**:
  - Split monolithic `iq3_s.cu` into `iq3_s_gemv.cu`, `iq3_s_mma.cu`, and `iq3_s_mma_q8.cu`, and replaced recursive template unrolling in `iq2_kl.cu`, `iq2_s.cu`, and `iq3_s.cuh` with compact loops and explicit launch bounds to eliminate register spills and cut compile time.

### Changed Files
- `src/ops/CMakeLists.txt` (`M`): Registers all new linear shape files, split `ggml/iq3_s_*.cu` files, and split `softmax_attention/causal_cache/int8/{tiled_launch,fast_prompt_launch}.cu` files.
- `src/ops/ggml/iq2_kl.cu` (`M`), `src/ops/ggml/iq2_s.cu` (`M`), `src/ops/ggml/iq3_s.cu` (`M`), `src/ops/ggml/iq3_s.cuh` (`A`), `src/ops/ggml/iq3_s_gemv.cu` (`A`), `src/ops/ggml/iq3_s_mma.cu` (`A`), `src/ops/ggml/iq3_s_mma_q8.cu` (`A`).
- `src/ops/linear/common/`:
  - Modified (`M`): `activation_q8.cu`, `activation_q8.cuh`, `epilogue.cuh`, `features.h`, `gemv_dp4a.cuh`, `layout.cuh`, `mma_e2m1.cuh`, `mma_f16.cuh`, `mma_s8.cuh`, `pipeline.cuh`, `quantize_mxfp8.cu`, `quantize_nvfp4.cu`, `splitkreduce.cu`, `splitkreduce.cuh`, `warp_reduce.cuh`.
- `src/ops/linear/bf16/`:
  - Modified (`M`): `dispatch.cpp`, `kernels/a16_gemm.cuh`, `kernels/a16_gemv.cuh`, and 19 files in `shapes/`.
  - Added (`A`): 8 new shape files (`n1024_k2048.cu`, `n1536_k2048.cu`, `n2048_k1536.cu`, `n2048_k2048.cu`, `n2048_k4096.cu`, `n3072_k2048.cu`, `n4096_k2048.cu`, `n6144_k2048.cu`).
- `src/ops/linear/dense8/`:
  - Modified (`M`): `dispatch.cpp`, `kernels/a16_gemm.cuh`, `kernels/a8_gemm.cuh`, `kernels/dp4a_gemm.cuh`, `kernels/dp4a_gemv.cuh`, and 27 files in `shapes/`.
  - Added (`A`): 8 new shape files (`n1024_k2048.cu`, `n1536_k2048.cu`, `n2048_k1536.cu`, `n2048_k2048.cu`, `n2048_k4096.cu`, `n3072_k2048.cu`, `n4096_k2048.cu`, `n6144_k2048.cu`).
- `src/ops/linear/fp8/`:
  - Modified (`M`): `dispatch.cpp`, `kernels/a16_gemm.cuh`, `kernels/a16_gemv.cuh`, `kernels/a8_gemm.cuh`, `kernels/a8_gemv.cuh`, `kernels/dp4a_gemv.cuh`, and 21 files in `shapes/`.
  - Added (`A`): 8 new shape files (`n1024_k2048.cu`, `n1536_k2048.cu`, `n2048_k1536.cu`, `n2048_k2048.cu`, `n2048_k4096.cu`, `n3072_k2048.cu`, `n4096_k2048.cu`, `n6144_k2048.cu`).
- `src/ops/linear/int4/`:
  - Modified (`M`): `dispatch.cpp`, `kernels/a16_gemm.cuh`, `kernels/a8_gemm.cuh`, `kernels/dp4a_gemm.cuh`, `kernels/dp4a_gemv.cuh`, and 21 files in `shapes/`.
  - Added (`A`): 8 new shape files (`n1024_k2048.cu`, `n1536_k2048.cu`, `n2048_k1536.cu`, `n2048_k2048.cu`, `n2048_k4096.cu`, `n3072_k2048.cu`, `n4096_k2048.cu`, `n6144_k2048.cu`).
- `src/ops/linear/iq2kl/`:
  - Modified (`M`): `dispatch.cpp`, `kernels/a16_gemm.cuh`, `kernels/a16_gemv.cuh`, `kernels/dp4a_gemv.cuh`, and 20 files in `shapes/`.
  - Added (`A`): 8 new shape files (`n1024_k2048.cu`, `n1536_k2048.cu`, `n2048_k1536.cu`, `n2048_k2048.cu`, `n2048_k4096.cu`, `n3072_k2048.cu`, `n4096_k2048.cu`, `n6144_k2048.cu`).
- `src/ops/linear/iq2s/`:
  - Modified (`M`): `dispatch.cpp`, `kernels/a16_gemm.cuh`, `kernels/dp4a_gemm.cuh`, `kernels/dp4a_gemv.cuh`, and 20 files in `shapes/`.
  - Added (`A`): 8 new shape files (`n1024_k2048.cu`, `n1536_k2048.cu`, `n2048_k1536.cu`, `n2048_k2048.cu`, `n2048_k4096.cu`, `n3072_k2048.cu`, `n4096_k2048.cu`, `n6144_k2048.cu`).
- `src/ops/linear/iq2xs/`:
  - Modified (`M`): `dispatch.cpp`, `kernels/a16_gemm.cuh`, `kernels/dp4a_gemm.cuh`, `kernels/dp4a_gemv.cuh`, and 23 files in `shapes/`.
  - Added (`A`): 8 new shape files (`n1024_k2048.cu`, `n1536_k2048.cu`, `n2048_k1536.cu`, `n2048_k2048.cu`, `n2048_k4096.cu`, `n3072_k2048.cu`, `n4096_k2048.cu`, `n6144_k2048.cu`).
- `src/ops/linear/iq2xxs/`:
  - Modified (`M`): `dispatch.cpp`, `kernels/a16_gemm.cuh`, `kernels/dp4a_gemm.cuh`, `kernels/dp4a_gemv.cuh`, and 20 files in `shapes/`.
  - Added (`A`): 8 new shape files (`n1024_k2048.cu`, `n1536_k2048.cu`, `n2048_k1536.cu`, `n2048_k2048.cu`, `n2048_k4096.cu`, `n3072_k2048.cu`, `n4096_k2048.cu`, `n6144_k2048.cu`).
- `src/ops/linear/iq3s/`:
  - Modified (`M`): `dispatch.cpp`, `kernels/a16_gemm.cuh`, `kernels/a8_gemm.cuh`, `kernels/dp4a_gemm.cuh`, `shapes/shapes.h`, and 27 files in `shapes/`.
  - Added (`A`): `kernels/dp4a_gemv.cuh` and 8 new shape files (`n1024_k2048.cu`, `n1536_k2048.cu`, `n2048_k1536.cu`, `n2048_k2048.cu`, `n2048_k4096.cu`, `n3072_k2048.cu`, `n4096_k2048.cu`, `n6144_k2048.cu`).
- `src/ops/linear/iq4xs/`:
  - Modified (`M`): `dispatch.cpp`, `kernels/a16_gemm.cuh`, `kernels/dp4a_gemm.cuh`, `kernels/dp4a_gemv.cuh`, and 22 files in `shapes/`.
  - Added (`A`): 8 new shape files (`n1024_k2048.cu`, `n1536_k2048.cu`, `n2048_k1536.cu`, `n2048_k2048.cu`, `n2048_k4096.cu`, `n3072_k2048.cu`, `n4096_k2048.cu`, `n6144_k2048.cu`).
- `src/ops/linear/mxfp4/`:
  - Modified (`M`): `dispatch.cpp`, `kernels/a8_gemm_128x128.cuh`, `kernels/a8_gemm_64x64.cuh`, `kernels/a8_gemv.cuh`, `kernels/dp4a_gemm.cuh`, `kernels/dp4a_gemv.cuh`, `shapes/shapes.h`, and 21 files in `shapes/`.
  - Added (`A`): `kernels/a8_gemm_16x64.cuh` and 8 new shape files (`n1024_k2048.cu`, `n1536_k2048.cu`, `n2048_k1536.cu`, `n2048_k2048.cu`, `n2048_k4096.cu`, `n3072_k2048.cu`, `n4096_k2048.cu`, `n6144_k2048.cu`).
- `src/ops/linear/nvfp4/`:
  - Modified (`M`): `dispatch.cpp`, `kernels/a16_gemm.cuh`, `kernels/a4_gemm_128x128.cuh`, `kernels/a4_gemm_64x64.cuh`, `kernels/a4_gemv.cuh`, `kernels/a8_gemm.cuh`, `kernels/a8_gemv.cuh`, `kernels/dp4a_gemv.cuh`, `shapes/shapes.h`, and 29 files in `shapes/`.
  - Added (`A`): `kernels/a4_gemm_64x128.cuh`, `kernels/a8_gemm_128x128.cuh`, and 8 new shape files (`n1024_k2048.cu`, `n1536_k2048.cu`, `n2048_k1536.cu`, `n2048_k2048.cu`, `n2048_k4096.cu`, `n3072_k2048.cu`, `n4096_k2048.cu`, `n6144_k2048.cu`).
- `src/ops/linear/q6k/`:
  - Modified (`M`): `dispatch.cpp`, `kernels/a16_gemm.cuh`, `kernels/dp4a_gemm.cuh`, `kernels/dp4a_gemv.cuh`, and 20 files in `shapes/`.
  - Added (`A`): 8 new shape files (`n1024_k2048.cu`, `n1536_k2048.cu`, `n2048_k1536.cu`, `n2048_k2048.cu`, `n2048_k4096.cu`, `n3072_k2048.cu`, `n4096_k2048.cu`, `n6144_k2048.cu`).
- `src/ops/linear/q8k/`:
  - Modified (`M`): `dispatch.cpp`, `kernels/a16_gemm.cuh`, `kernels/a8_gemm.cuh`, `kernels/dp4a_gemm.cuh`, `kernels/dp4a_gemv.cuh`, and 21 files in `shapes/`.
  - Added (`A`): 8 new shape files (`n1024_k2048.cu`, `n1536_k2048.cu`, `n2048_k1536.cu`, `n2048_k2048.cu`, `n2048_k4096.cu`, `n3072_k2048.cu`, `n4096_k2048.cu`, `n6144_k2048.cu`).

---

## 6. Attention, KV Cache & Linear Attention (`src/ops/softmax_attention/`, `src/ops/kv_cache/`, `src/ops/qsa/`, `src/ops/linear_attention/`)

### Why & Key Mechanisms
- **Direct FP32 PV accumulation in tiled attention (`mxfp8_tiled_mma.cuh`, `int8/fast_tiled_mma.cuh`, `nvfp4_tiled_values.cuh`, `fp8/tile_io.cuh`)**:
  - Previously, `nvfp4` and `k8v4` KV caches were slower than `fp8` on GB10 because `mxfp8_accumulate_pv_tile` accumulated 64 keys into FP16 registers (`partial_h`) to work around RTX 5090's half-rate FP32 HMMA, and `Nvfp4G16TiledValues::tile_shift` performed two `uint4` scale loads + a 5-step butterfly warp reduction (`__shfl_xor_sync`) + `ldexpf` per 64-key tile to prevent FP16 overflow.
  - Because `sm_121a` runs `mma.sync.aligned.m16n8k16.f32.f16.f16.f32` at 1:1 parity with FP16 accumulation:
    1. `mxfp8_accumulate_pv_tile` and `i8_fast_accumulate_pv_tile` now accumulate directly into the FP32 `acc` registers via `mma_f16` in k-major order (`k = li / PVHalf`, `n2 = (li % PVHalf) * 2`), eliminating `partial_h` and the FP16-to-FP32 unpack/FMA loop.
    2. `Nvfp4G16TiledValues::tile_shift` is now a constant `return 0;` (since individual decoded V values `<= 6 * 448 = 2688` always fit in FP16 without shifting), completely removing the per-tile scale scan and shift for `nvfp4` and `k8v4` KV caches!
    3. `Fp8KvTiledValues::tile_shift` threshold was raised from `maximum < 2.0f` to `maximum < 128.0f` (`ilogbf(maximum) - 6`), and `kCausalPromptI8FastF16PartialScaleLimit` was raised from `8.0f` to `512.0f`.
- **Prompt split-K & fast-prompt threshold retuning for 48 SMs (`mxfp8_tiled_plan.h`, `int8/fast_tiled_plan.h`, `nvfp4/plan.cpp`, `bf16/plan.cpp`, `int8/plan.cpp`)**:
  - Raised the split-reduction penalty in `mxfp8_tiled_active_splits` and `i8_fast_tiled_active_splits` from `0.5 / 24.0` to `3.5 / 24.0` per column to avoid expensive global FP32 split-partial writes/merges on 48 SMs.
  - Lowered `kGroupedPrefillMaxWidth` from `256` to `192` in `bf16/plan.cpp` and `int8/plan.cpp`.
  - Updated `nvfp4_fast_prompt_applies(max_visible_keys, width)` in `nvfp4/plan.cpp` (`max_visible_keys >= width + 512 || max_visible_keys > 3072`) so short-prefix prompts (`ctx < 512`, `W <= 2048`) use the 64-row tiled kernel instead of wasting half the 128-row fast-prompt tile in the causal triangle.
- **`sm_121a` codec arithmetic (`q4_lloyd_codec.cuh`, `vq2_codec.cuh`, `vq2_kernel.cuh`)**:
  - In `q4_lloyd_codec.cuh` (`k4v2`), replaced emulated `__vsub4(0, mag & mask)` with carry-free `((mag ^ mask) + (mask & 0x01010101u))` (`LOP3 + IADD3`), which is bit-exact because Q4 Lloyd magnitudes are strictly non-zero (`>= 4`).
  - Documented in `vq2_codec.cuh` why `__vsub4` must remain there (VQ2 codebooks contain `0` magnitudes where `+ 1` would carry into adjacent bytes), and set `INFERNIX_VQ2_PAIRED_FP32` default to `0` in `vq2_kernel.cuh` to preserve exact IEEE-754 rounding in `exact_reducible`.
- **Eliminating local-memory spills in `column_invariant.cu`**:
  - Replaced dynamic array indexing `batch_tensors.tensors[blockIdx.y]` in `causal_decode_column_invariant_4_kernel` with a 4-way compile-time ternary expression so `ptxas` keeps `ColumnInvariantTensorParams` in registers (`0 B` stack spill vs `96 B` previously).
- **Sliding-window attention (`sliding_window/launch.cu`)**:
  - Unified `window == 4096` with `window == 2048` (`reduce_warps = 4` and token-throttled `split_limit` instead of hardcoding `reduce_warps = 1, split_limit = 32` at `window == 4096`).
- **Quantized Sparse Attention (`src/ops/qsa/`)**:
  - `qsa_prepare.cu`: Added support for K/V-only preparation when `q_out` and `index_q_out` are empty (used by speculative 1-row catch-up `mtp_catch_up` for rejected draft columns).
  - `qsa.cu`: `attention_splits()` now returns `1` (unsplit Tensor Core prompt route) whenever `splits < 4` (`columns >= 9` on 48 SMs), avoiding SIMT split-attention fallback; `merge_kernel<rotate_values>` templated on compile-time bool; `e2m1_at` uses native `__nv_fp4_e2m1`.
  - `qsa_prompt.cu`: Target block count reduced from `256` to `96` (`2 * 48 SMs`) to cut L2 KV traffic in half on GB10.
- **Gated Delta Net (`src/ops/linear_attention/gated_delta_net/`)**:
  - `chunked/recurrence.cu`: Replaced FP16 partial carry (`mma_f16_partial`, `kFp16Carry = 2`) with direct FP32 Tensor Core accumulation (`mma_f16`) and retuned `value_tile()` for 48 SMs using wave efficiency.
  - `recurrent.cuh`: Prefetches `next_key`, `next_gate`, and `next_value` before `apply_gdn_transition` to hide LPDDR5X latency across token steps; unions `TreeWalkPending` with `saved` in shared memory (`SmemStorage`); packs `replay_root_path[16]` into two 64-bit registers (`path_lo`, `path_hi`) to eliminate local-memory spills.

### Changed Files
- `src/ops/kv_cache/`:
  - `kv_cache.cu` (`M`), `kv_cache_codecs.cuh` (`M`), `q4_lloyd_codec.cuh` (`M`), `vq2_codec.cuh` (`M`).
- `src/ops/softmax_attention/`:
  - `common/paged_kv_address.cuh` (`M`): Adds `PagedKVPromptMetadata` and `load_prompt_metadata()`.
  - `common/vq2_kernel.cuh` (`M`): Sets `INFERNIX_VQ2_PAIRED_FP32=0` default on `sm_121a`.
  - `sliding_window/launch.cu` (`M`): Unifies `window == 4096` with `2048` (`reduce_warps = 4`, token-throttled `split_limit`).
  - `causal_cache/common/column_invariant.cu` (`M`): Eliminates 96-byte local-memory spill in `causal_decode_column_invariant_4_kernel`.
  - `causal_cache/common/mxfp8_tiled_kernel.cuh` (`M`), `mxfp8_tiled_mma.cuh` (`M`), `mxfp8_tiled_plan.h` (`M`), `nvfp4_tiled_values.cuh` (`M`), `prompt_grouped.cuh` (`M`), `prompt_launch.cuh` (`M`), `prompt_single.cuh` (`M`), `split_merge.cuh` (`M`), `tiled_prompt.cuh` (`M`).
  - `causal_cache/bf16/launch.cu` (`M`), `causal_cache/bf16/plan.cpp` (`M`).
  - `causal_cache/fp8/launch.cu` (`M`), `causal_cache/fp8/tile_io.cuh` (`M`).
  - `causal_cache/int8/`: `fast_prompt_launch.cu` (`A`), `fast_prompt_launch.h` (`A`), `tiled_launch.cu` (`A`), `tiled_launch.h` (`A`), `fast_tiled_kernel.cuh` (`M`), `fast_tiled_mma.cuh` (`M`), `fast_tiled_plan.h` (`M`), `launch.cu` (`M`), `plan.cpp` (`M`), `tile_io.cuh` (`M`).
  - `causal_cache/k4v2/launch.cu` (`M`), `causal_cache/k8v4/launch.cu` (`M`), `causal_cache/vq2/launch.cu` (`M`).
  - `causal_cache/nvfp4/`: `fast_prompt.cuh` (`M`), `fast_prompt_kernel.cuh` (`M`), `fast_prompt_mma.cuh` (`M`), `fast_prompt_state.cuh` (`M`), `launch.cu` (`M`), `plan.cpp` (`M`).
- `src/ops/qsa/`:
  - `qsa.cu` (`M`), `qsa_prepare.cu` (`M`), `qsa_prompt.cu` (`M`).
- `src/ops/linear_attention/`:
  - `conv1d.cu` (`M`), `gated_delta_net/chunked/chunk_local.cu` (`M`), `gated_delta_net/chunked/prepare_wy.cu` (`M`), `gated_delta_net/chunked/recurrence.cu` (`M`), `gated_delta_net/chunked/solve_tril.cu` (`M`), `gated_delta_net/decode.cu` (`M`), `gated_delta_net/gdn_exact.cu` (`M`), `gated_delta_net/recurrent.cuh` (`M`).

---

## 7. Sparse MoE & Offloaded Sparse MoE (`src/ops/sparse_moe/`, `src/ops/offloaded_sparse_moe/`)

### Why & Key Mechanisms
- **ARMv8.6-A NEON & `FEAT_I8MM` CPU expert kernels (`neon_mad.h`, `cpu/*_expert.cpp`)**:
  - Replaced `x86_mad.h` (`_mm256_maddubs_epi16` / AVX-512 VNNI) with `neon_mad.h` providing `AsimdMad` (`vmlal_s16`) and `I8mmMad` (`vusdotq_s32` compiled with `__attribute__((target("arch=armv8.6-a+i8mm")))`, selected at runtime via `getauxval(AT_HWCAP2) & HWCAP2_I8MM`).
- **Grace big.LITTLE CPU pinning (`expert_team.{cpp,h}`)**:
  - `CpuExpertTeam::placement_cpus(workers)` pins worker threads (and temporarily pins the calling thread during `CpuExpertTeam::run()`) to the 10 Cortex-X925 performance cores (`core::performance_cpus()`) whenever `workers <= big.size()`, preventing Cortex-A725 stragglers from stalling spin barriers while avoiding oversubscription when `workers > 10`.
  - Replaced `<immintrin.h>` `_mm_pause()` in spin loops with AArch64 `asm volatile("yield" ::: "memory")`.
- **Direct on-device expert residency & GPU MoE tuning (`offloaded_sparse_moe.cu`, `offloaded_sparse_moe_gpu.cuh`, `sparse_moe/*.cu`)**:
  - Supports `device_experts` where all expert records live in device memory (`resident_records` points directly into the device expert bank, `miss_count = 0`, zero D2D staging/eviction).
  - Retuned GPU expert tile schedules, shared-memory layouts, and launch bounds across `sparse_moe` (`bf16`, `dense8`, `fp8`, `int4`, `iq2kl`, `iq2s`, `iq2xs`, `iq2xxs`, `iq3s`, `iq4xs`, `mxfp4`, `nvfp4`, `q6k`, `q8k`) and `offloaded_sparse_moe` for 48 SMs.

### Changed Files
- `src/ops/offloaded_sparse_moe/`:
  - `cpu/neon_mad.h` (`A`), `cpu/x86_mad.h` (`D`).
  - `cpu/dense8_expert.cpp` (`M`), `cpu/expert_common.h` (`M`), `cpu/expert_team.cpp` (`M`), `cpu/expert_team.h` (`M`), `cpu/iq2_kl_expert.cpp` (`M`), `cpu/iq2_s_expert.cpp` (`M`), `cpu/iq3_s_expert.cpp` (`M`), `cpu/mxfp4_expert.cpp` (`M`), `cpu/nvfp4_expert.cpp` (`M`), `cpu/q8_k_expert.cpp` (`M`).
  - `offloaded_sparse_moe.cu` (`M`), `offloaded_sparse_moe_gpu.cuh` (`M`), `offloaded_sparse_moe_iq2_kl.cu` (`M`), `offloaded_sparse_moe_iq2_s.cu` (`M`), `offloaded_sparse_moe_iq3_s.cu` (`M`), `offloaded_sparse_moe_mxfp4.cu` (`M`), `offloaded_sparse_moe_nvfp4.cu` (`M`).
- `src/ops/sparse_moe/`:
  - `common.cuh` (`M`), `routing.cu` (`M`), `sparse_moe.cu` (`M`), `sparse_moe_bf16.cu` (`M`), `sparse_moe_dense8.cu` (`M`), `sparse_moe_fp8.cu` (`M`), `sparse_moe_int4.cu` (`M`), `sparse_moe_iq2kl.cu` (`M`), `sparse_moe_iq2s.cu` (`M`), `sparse_moe_iq2xs.cu` (`M`), `sparse_moe_iq2xxs.cu` (`M`), `sparse_moe_iq3s.cu` (`M`), `sparse_moe_iq4xs.cu` (`M`), `sparse_moe_mxfp4.cu` (`M`), `sparse_moe_nvfp4.cu` (`M`), `sparse_moe_q6k.cu` (`M`), `sparse_moe_q8k.cu` (`M`).

---

## 8. Sampling, Penalty, PLE & Hyper-Connection Ops (`src/ops/sampling/`, `src/ops/penalty/`, `src/ops/ple/`, `src/ops/hyper_connection/`)

### Why & Key Mechanisms
- **Single-pass fused argmax + logprob in `top_logprobs.cuh` (`src/ops/sampling/`)**:
  - `top_logprobs_1_kernel` and `top_logprobs_kernel` now track `local_max`, online `local_sum` (rescaled on new max), and `first_key` in a single pass over the vocabulary, allowing `Forward::mtp_block` to get both the greedy draft token (`top_ids`) and its log-probability (`top_values`) from one `ops::top_logprobs(..., k=1)` launch instead of launching both `ops::argmax_sample` and `ops::top_logprobs`.
- **Strided 2D `ple_gate` & FP32-scaled NVFP4 `ple_embed` (`src/ops/ple/ple.cu`, `src/ops/ple/ple.h`)**:
  - Added `column_strided_2d(TensorView)` plus `key_stride` and `value_stride` parameters to `gate_kernel` and `gate_exact_kernel`. In `Forward::ple`, `kv.slice(1, ...)` views are passed directly into `ops::ple_gate`, eliminating the intermediate `ops::split_rows` launch and its temporary workspace buffers (`gate_k`, `gate_v`)!
  - Wired `embed_nvfp4_kernel` (reading `QuantType::NVFP4` tables with FP32 group scales at the tail of each row) into `ple_embed()`, validated by `tests/ops/test_ple_embed.cu`.

### Changed Files
- `src/ops/hyper_connection/hyper_connection.cu` (`M`): Adds launch bounds and vectorizes hyper-connection width mixing.
- `src/ops/penalty/penalty.cu` (`M`): Adds launch bounds and optimizes penalty token scan.
- `src/ops/ple/ple.cu` (`M`), `src/ops/ple/ple.h` (`M`): Adds strided 2D `key`/`value` support in `ple_gate()` and `QuantType::NVFP4` support in `ple_embed()`.
- `src/ops/sampling/sampling.cu` (`M`), `src/ops/sampling/top_logprobs.cuh` (`M`): Single-pass online softmax + argmax in `top_logprobs.cuh` and launch bounds across sampling kernels.

---

## 9. Models: `qwen3_5` & `qwen4_exp` (`src/models/`)

### Why & Key Mechanisms
1. **Default On-Device Expert Residency on Unified Memory (`qwen4_exp/load_options.h`, `load.cpp`, `memory_plan.{cpp,h}`, `program_impl.h`, `qwen4_exp_instance.cpp`)**:
   - `LoadOptions::device_experts` defaults to `true` on integrated GPUs (`props.integrated != 0`), overrideable via `INFERNIX_EXPERTS_ON_GPU=0|1`.
   - When `device_experts` is true:
     - Routed expert banks bind with `Residency::Device` (`load.cpp` temporarily copies the scale tail to host during `materialize_model` to compute `expert_scales`, then restores the device pointer).
     - `staging_slots = 0`, `max_frames = 0`, `kv_base_pages = kv_pages` (100% of KV pages backed upfront since there are no expert frames to borrow from), and `static_columns = max(lanes * W, chunk)` (no per-chunk fallback arena alloc/free).
     - `cpu_workers` defaults to `0` (CPU miss-service disabled) and `track_routes()` skips per-round route D2H downloads unless `--route-trace` is active.
   - When `device_experts` is false on an integrated GPU (`INFERNIX_EXPERTS_ON_GPU=0`), `size_expert_frames` adds `snap.reclaimable`, subtracts `shared_host_bytes`, and caps the expert frame pool at `kIntegratedFramePoolCapBytes = 32 GiB`.
2. **N-gram Speculative Proposer & Frontend Index Handoff (`qwen3_5/program/ngram_proposer.h`, `frontend.cpp`, `prepared_prompt.h`, `program_impl.h`)**:
   - **Zero-rebuild prompt admission (`take_ngram_index`)**: `PreparedPromptAccess::take_ngram_index` moves the `NgramProposer` already built during `Frontend::prepare_prompt()` directly into the execution lane on single-request prefill instead of clearing and re-ingesting the entire prompt.
   - **Fast in-place `NgramProposer::propose()`**: Hoists `live_begin()`, deduplicates candidate `entry.end` positions via a stack `seen_ends` array (skipping redundant backward extensions when shorter n-gram orders hit the same history match), counts `avail` in-place without allocating a `std::vector<Token>` per candidate, and exits early via `maximal()`.
   - **Multi-request & wider n-gram drafting**: Caps the history slice passed to `propose()` at `kNgramMatchHistoryTokens`, enables n-gram proposals when `batch > 1`, scores n-gram copy proposals with `kNgramWidthCost = 0.12` (`INFERNIX_NGRAM_WIDTH_COST`) vs neural MTP's `kWidthCost = 0.25` (`INFERNIX_WIDTH_COST`), and clamps the online `width_cost()` regression floor to `0.14` (`[0.14, kCostMax]`).
3. **Pair Drafting, Multi-Row Speculative Decoding & K/V-Only Catch-Up (`qwen4_exp/program/program_impl.h`, `mtp_round.{cu,cuh}`, `forward.{cpp,h}`, `parameters.{cpp,h}`)**:
   - Lowered `kPairWidthCost` from `0.47` to `0.15` (`INFERNIX_PAIR_WIDTH_COST`) and raised `kPairMaxDrafts` from `2` to `6` (`INFERNIX_PAIR_MAX_DRAFTS`, keeping `kDFlashPairMaxDrafts = 2` for DFlash2/Exact).
   - Added `choose_multirow_draft_length()` (`kMultiRowDrafts = 1.0`, `kMultiRowMaxDrafts = 5`, env `INFERNIX_MULTIROW_DRAFTS` / `INFERNIX_MULTIROW_MAX_DRAFTS`) so `batch >= 3` MTP rounds continue drafting instead of dropping to plain 1-column decode.
   - Added `AttentionParameters::kv_projection` and `index_key_projection` (`parameters.{cpp,h}`) plus `kv_only` in `Forward::qsa` (`forward.cpp`) and `U = min(W, max_k)` width trimming in `mtp_catch_up()` so 1-row catch-up after partial draft rejection only computes K/V projections on committed columns (skipping Q and index-Q projections and unused columns).
   - Added `need_mtp_logprobs()` to skip computing/copying MTP logprobs when neither `draft_min_p` nor `mtp_cumulative_base` is active, fused argmax + logprob into a single `ops::top_logprobs` call in `Forward::mtp_block`, and removed redundant D2D `cudaMemcpyAsync` of `residuals` and `residual_out` in `Forward::finish` / `Forward::run_mtp`.
4. **4-Way Set-Associative PLE N-gram Volume Cache & Persistent Reader Pool (`qwen4_exp/ngram_volume.{cpp,h}`, `ngram_hash.cpp`, `program_impl.h`)**:
   - Replaced the direct-mapped PLE n-gram row cache with a **4-way set-associative cache** (`kCacheWays = 4`, round-robin per-set `hand_`, configurable via `--ngram-cache-mib` from `2^16` to `2^28` slots).
   - Replaced per-call `std::thread` creation with a persistent 8-thread `NgramVolume::ReaderPool` (`kReadThreads = 8`), lowered `kParallelBlocks` from `512` to `16`, enabled `parallel_gather_` and coprime-strided `scatter_` (`stride = 97`) by default, parallelized `NgramHash::row_ids` across 8 threads when `count >= 512`, batched `stage_decode_rows` and `verify` `read_rows` across all active lanes in multi-row rounds, and enabled `ngram_first_span_async_` by default when reading from an SSD volume.
5. **GB10 Prefix-Cost Calibration, DFlash2 Auto-Discovery & `--no-cuda-graph` Fix (`qwen4_exp_instance.cpp`, `program.h`, `program_impl.h`)**:
   - Calibrated `prefix_cost` in `qwen4_exp_instance.cpp` for integrated GPUs (`chunk_seconds = 0.56`, `token_seconds = 9.0e-5`, `attention_pair_seconds = 4.0e-9`, `h2d_bytes_per_second = 19.4e9`, `prefix_span_seconds = 0.0` when `device_experts`), raised the auto host prefix-cache floor to `kDefaultHybridHostCacheBytes`, and added `default_dflash2_model()` companion artifact discovery.
   - Plumbed `ProgramOptions::use_cuda_graph` into `ProgramImpl::replay()` (`qwen3_5` and `qwen4_exp`) so `--no-cuda-graph` actually disables CUDA Graph capture/replay during decode rounds.

### Changed Files
- `src/models/qwen3_5/`:
  - `frontend.cpp` (`M`), `prepared_prompt.h` (`M`): Adds `PreparedPromptAccess::take_ngram_index` to transfer the prepared `NgramProposer` into the execution lane.
  - `program.h` (`M`): Adds `use_cuda_graph = true` to `ProgramOptions`.
  - `program/forward.cpp` (`M`), `program/workspace.cpp` (`M`): Eliminates intermediate `split_rows` buffers (`gate_k`, `gate_v`) when calling strided `ops::ple_gate`.
  - `program/ngram_proposer.h` (`M`): Adds `seen_ends` deduplication, hoisted `live_begin()`, in-place match length counting, and `maximal()` early exit.
  - `program/program_impl.h` (`M`): Plumbs `use_cuda_graph`, takes prepared `NgramProposer` on single-request prefill, caps history slice at `kNgramMatchHistoryTokens`, enables `batch > 1` n-gram proposals, and uses `kNgramWidthCost = 0.12` / `kWidthCost = 0.25`.
- `src/models/qwen4_exp/`:
  - `load.cpp` (`M`), `load_options.h` (`M`): Adds `device_experts` (`Residency::Device` expert bank binding with temporary host scale-tail read) and `ngram_cache_bytes`.
  - `memory_plan.cpp` (`M`), `memory_plan.h` (`M`): Unified-memory `reclaimable` + `shared_host_bytes` accounting, `kIntegratedFramePoolCapBytes = 32 GiB`, and `device_experts` zero-frame / full-KV-base planning.
  - `ngram_hash.cpp` (`M`): Parallelizes `NgramHash::row_ids` across 8 threads when `count >= 512`.
  - `ngram_volume.cpp` (`M`), `ngram_volume.h` (`M`): 4-way set-associative cache (`kCacheWays = 4`), configurable `cache_bytes`, persistent 8-thread `ReaderPool`, `kParallelBlocks = 16`, and default-on `parallel_gather_` / `scatter_`.
  - `parameters.cpp` (`M`), `parameters.h` (`M`): Adds `kv_projection` and `index_key_projection` sub-slices on `AttentionParameters` for K/V-only catch-up.
  - `program.h` (`M`): Adds `use_cuda_graph = true` to `ProgramOptions`.
  - `program/forward.cpp` (`M`), `program/forward.h` (`M`), `program/workspace.cpp` (`M`), `program/workspace.h` (`M`): Adds `kv_only` in `Forward::qsa`, strided `ops::ple_gate` without `split_rows`, single-call `ops::top_logprobs` in `mtp_block`, conditional `need_mtp_logprobs`, and removes redundant D2D residual copies.
  - `program/mtp_round.cu` (`M`), `program/mtp_round.cuh` (`M`): Updates MTP round kernels for optional logprobs and multi-row/pair drafting.
  - `program/program_impl.h` (`M`): Implements `device_experts` execution path, `kWidthCost = 0.25`, `kNgramWidthCost = 0.12`, `kPairWidthCost = 0.15`, `kPairMaxDrafts = 6`, `choose_multirow_draft_length()`, K/V-only `mtp_catch_up` with `U = min(W, max_k)` trimming, batched multi-lane PLE volume reads, and `ngram_first_span_async_`.
  - `qwen4_exp_instance.cpp` (`M`): Sets `device_experts` default on integrated GPUs, calibrates GB10 `prefix_cost`, configures `ngram_cache_bytes`, and adds `default_dflash2_model()` auto-discovery.

---

## 10. Benchmarks & Profiling Harnesses (`bench/`)

### Why & Key Mechanisms
- **GB10-right-sized L2 cache flush (`32–48 MiB` vs `256 MiB`)**: RTX 5090's 256 MiB L2 flush buffer wasted 5x–8x wall time per iteration on GB10's LPDDR5X; `infernix_linear_bench` now queries `cudaDevAttrL2CacheSize` and uses `max(48 MiB, 2 * l2)` when `--l2-flush-mb` is not overridden, plus adds `--max-tokens` for route-boundary probing (`[B, B+1]`).
- **End-to-end KV-matrix A/B benchmark (`bench/run_kv_matrix_bench.sh`)**: Standardizes full-model `pp4096` and `tg128` verification across all KV dtypes (`bf16`, `fp8`, `int8`, `k8v4`, `nvfp4`, `k4v2`, `vq2`) and `plain`/`mtp` modes on real Qwen3.8-Flash-Next artifacts (`iq3s`, `dense8`, `nvfp4`).

### Changed Files
- `bench/CMakeLists.txt` (`M`), `bench/README.md` (`M`).
- `bench/attention_bench.cpp` (`M`), `bench/bench_main.cpp` (`M`), `bench/context_cost_bench.cpp` (`M`), `bench/gdn_bench.cpp` (`M`), `bench/kv_cache_bench.cpp` (`M`), `bench/linear_bench.cpp` (`M`), `bench/moe_bench.cpp` (`M`), `bench/offloaded_moe_bench.cu` (`M`), `bench/ple_bench.cpp` (`M`), `bench/qsa_bench.cpp` (`M`), `bench/sampling_bench.cpp` (`M`), `bench/shared_prefix_bench.cpp` (`M`), `bench/vision_bench.cpp` (`M`).
- `bench/run_kv_matrix_bench.sh` (`A`): End-to-end KV-cache & plain/MTP benchmark matrix runner.

---

## 11. Tests (`tests/`)

### Why & Key Mechanisms
- **One-way CUDA stack-spill ratchet gate (`tests/ops/cuda_stack_check.cmake`, `cuda_stack_baseline.txt`)**: Runs `cuobjdump -res-usage` on `infernix-serve` in CTest (`infernix_cuda_stack_check`) and fails if any kernel symbol has `STACK > 0` or `LOCAL > 0`.
- **Shape-targeted linear oracle qualification (`--shape <N> <K>`)**: Added `--shape` filtering to `test_linear_common.cuh` and all `test_linear_*_a16.cu` suites so individual modified shapes can be qualified against the FP32 oracle in ~2.1s (instead of running the full 118s suite).
- **PLE NVFP4 embedding oracle test (`tests/ops/test_ple_embed.cu`)**: Validates `ops::ple_embed` across `BF16`, `Q4_0`, `Q2_K`, and `NVFP4` against an independent CPU reference.
- **GB10 hardware contract updates**: Updated `test_vram_budget.cpp` (unified-memory reclaimable pool, zero auto headroom, `SpillGuard` no-op on integrated GPUs), `test_qwen4_exp_memory_plan.cpp` (`kIntegratedFramePoolCapBytes = 32 GiB`, `reclaimable`, `shared_host_bytes`), `test_qwen4_exp_cpu_expert.cpp` (ARMv8.6-A NEON/`FEAT_I8MM`), and `test_cli_options.cpp` (`--ngram-cache-mib`, max `--ngram-draft-tokens` 31).

### Changed Files
- `tests/CMakeLists.txt` (`M`), `tests/README.md` (`M`).
- `tests/ops/cuda_stack_baseline.txt` (`A`), `tests/ops/cuda_stack_check.cmake` (`A`): Zero-spill CTest gate (`0` stack/local symbols).
- `tests/ops/test_ple_embed.cu` (`A`): Independent oracle test for `ops::ple_embed` including `QuantType::NVFP4`.
- `tests/ops/test_linear_common.cuh` (`M`), `tests/ops/test_linear_bf16_a16.cu` (`M`), `tests/ops/test_linear_dense8_a16.cu` (`M`), `tests/ops/test_linear_fp8_a16.cu` (`M`), `tests/ops/test_linear_int4_a16.cu` (`M`), `tests/ops/test_linear_iq2kl_a16.cu` (`M`), `tests/ops/test_linear_iq2s_a16.cu` (`M`), `tests/ops/test_linear_iq2xs_a16.cu` (`M`), `tests/ops/test_linear_iq2xxs_a16.cu` (`M`), `tests/ops/test_linear_iq3s_a16.cu` (`M`), `tests/ops/test_linear_iq4xs_a16.cu` (`M`), `tests/ops/test_linear_mxfp4_a16.cu` (`M`), `tests/ops/test_linear_nvfp4_a16.cu` (`M`), `tests/ops/test_linear_q6k_a16.cu` (`M`), `tests/ops/test_linear_q8k_a16.cu` (`M`).
- `tests/ops/test_kv_cache.cu` (`M`), `tests/ops/test_offloaded_sparse_moe.cu` (`M`).
- `tests/integration/test_cli_options.cpp` (`M`), `tests/integration/test_qwen4_exp_cpu_expert.cpp` (`M`), `tests/integration/test_qwen4_exp_memory_plan.cpp` (`M`), `tests/integration/test_qwen4_exp_p5_gate.cpp` (`M`), `tests/integration/test_vram_budget.cpp` (`M`).
- `tests/unit/test_context_cost.cpp` (`M`), `tests/unit/test_host_load.cpp` (`M`).

---

## 12. Tools & Hardware Sweeps (`tools/`)

### Why & Key Mechanisms
- Added GB10 linear shape sweep utilities (`gb10_shape_sweep.py`, `gb10_nvfp4_a4_sweep.py`, `run_all_gb10_sweeps.sh`) and stripped `#ifdef _WIN32` branches from `tools/flash_next_probe/*` (`expert_read_probe.cpp`, `vram_probe.cpp`, `gpu_probe.cu`, `probes.cmake`, `run_m0.sh`).

### Changed Files
- `tools/gb10_nvfp4_a4_sweep.py` (`A`), `tools/gb10_shape_sweep.py` (`A`), `tools/run_all_gb10_sweeps.sh` (`A`).
- `tools/flash_next_probe/expert_read_probe.cpp` (`M`), `tools/flash_next_probe/gpu_probe.cu` (`M`), `tools/flash_next_probe/probes.cmake` (`M`), `tools/flash_next_probe/run_m0.sh` (`M`), `tools/flash_next_probe/vram_probe.cpp` (`M`).

---

## 13. Documentation (`README.md`, `docs/`)

### Changed Files
- Modified (`M`): `README.md`, `docs/README.md`, `docs/cli.md`, `docs/conversion.md`, `docs/model-support-matrix.md`, `docs/performance.md`, `docs/perplexity.md`, `docs/serving.md`, `docs/maintainer/benchmark-methodology.md`, `docs/maintainer/coding-standards.md`, `docs/maintainer/engine-architecture.md`, `docs/maintainer/hybrid-prefix-cache-spec.md`, `docs/maintainer/model-quality.md`, `docs/maintainer/op-development.md`, `docs/maintainer/paged-kv-cache.md`, `docs/maintainer/qwen3_5-reference.md`, `docs/maintainer/qwen3_8-flash-next-design.md`, `docs/maintainer/resource-scheduling-and-context-cache.md`.
- Added (`A`): `docs/gb10_roadmap.md`, `docs/gb10/20261026-optimization-report.md`, `docs/gb10/20261027-optimization-report.md`.
- Deleted (`D` — superseded RTX 5090 / Windows phase-gate reports): `docs/maintainer/qwen3_8-flash-next-m0-measurements.md`, `docs/maintainer/qwen3_8-flash-next-p1-gate.md`, `docs/maintainer/qwen3_8-flash-next-p2-gate.md`, `docs/maintainer/qwen3_8-flash-next-p3-gate.md`, `docs/maintainer/qwen3_8-flash-next-p4-gate.md`, `docs/maintainer/qwen3_8-flash-next-p5-gate.md`, `docs/maintainer/qwen3_8-flash-next-p6-gate.md`, `docs/maintainer/qwen3_8-flash-next-p7-gate.md`, `docs/maintainer/qwen3_8-flash-next-p8-gate.md`.

---

## 14. Agent Instructions, Multi-Agent Orchestration & Skills (`AGENTS.md`, `.github/`)

### Changed Files
- `AGENTS.md` (`M`), `.github/copilot-instructions.md` (`A`).
- `.github/agents/`:
  - `cuda-gpu-benchmarker.agent.md` (`A`), `cuda-kernel-implementer.agent.md` (`A`), `cuda-kernel-optimizer.agent.md` (`A`), `cuda-kernel-orchestrator.agent.md` (`A`), `cuda-source-auditor.agent.md` (`A`).
- `.github/skills/`:
  - Modified (`M`): `cuda-kernel-optimization/SKILL.md`, `cuda-profiling/SKILL.md`.
  - Added (`A`): `cuda-architecture-discovery/SKILL.md`, `cuda-attention-kv-design/SKILL.md`, `cuda-collective-overlap-design/SKILL.md`, `cuda-decode-smallm-design/SKILL.md`, `cuda-elementwise-reduction-norm-design/SKILL.md`, `cuda-experiment-memory/SKILL.md`, `cuda-gemm-epilogue-design/SKILL.md`, `cuda-kernel-repair/SKILL.md`, `cuda-moe-grouped-gemm-design/SKILL.md`, `cuda-quantized-lowbit-design/SKILL.md`, `cuda-runtime-graph-launch-design/SKILL.md`, `cuda-validation-benchmarking/SKILL.md`.

---

## 15. Rebase Guide & Conflict Resolution Reference (For Future Rebases onto `origin/main`)

When rebasing this branch onto newer `origin/main` commits, resolve conflicts according to the following subsystem rules:

| Subsystem / File Pattern | Conflict Resolution Policy | Critical Invariants to Preserve |
|---|---|---|
| `CMakeLists.txt`, `CMakePresets.json`, `src/core/*`, `src/serve/*`, `apps/*` | **Keep ours (GB10 Linux/AArch64)** for platform checks, flags (`121a-real`, `-mcpu=native`, `INFERNIX_LINK_JOBS=8`), and POSIX/NVML implementations; **integrate upstream** if `origin/main` adds new source files, new CLI flags, or new HTTP endpoints. | Never re-introduce `_WIN32`, `<immintrin.h>`, `_mm_sfence`, `_mm_pause`, `dxcore_adapter`, or `120a-real`. Keep `--ngram-cache-mib` and max `--ngram-draft-tokens` = `31`. |
| `src/core/vram_budget.{cpp,h}`, `src/core/host_memory.{cpp,h}`, `src/models/qwen4_exp/memory_plan.{cpp,h}` | **Keep ours** for unified-memory accounting (`integrated()`, `reclaimable`, `shared_host_claim()`, `kIntegratedFramePoolCapBytes = 32 GiB`, `device_experts` zero-frame / full-KV-base planning). | On `integrated` GPUs, host and device share one pool: never drop `shared_host_claim()` subtraction or `reclaimable_pool_bytes()`, and keep `SpillGuard::end() == 0`. |
| `src/models/qwen4_exp/{load.cpp,load_options.h,qwen4_exp_instance.cpp}` | **Keep ours** for `device_experts` default on integrated GPUs, GB10 `prefix_cost` calibration, and `ngram_cache_bytes`; **merge upstream** if new model tensors or config fields are added. | Ensure `load.cpp` still temporarily copies the expert scale tail to host during `materialize_model` when `options.device_experts` is true, then restores the device pointer. |
| `src/models/qwen3_5/program/{ngram_proposer.h,program_impl.h}`, `src/models/qwen4_exp/program/{program_impl.h,forward.cpp,mtp_round.*}` | **Keep ours** for `take_ngram_index`, `NgramProposer::propose()` deduplication/in-place match, `kWidthCost = 0.25`, `kNgramWidthCost = 0.12`, `kPairWidthCost = 0.15`, `kPairMaxDrafts = 6`, `choose_multirow_draft_length()`, K/V-only `mtp_catch_up()`, and `use_cuda_graph`. | If upstream changes `ProgramImpl` scheduling or transaction hooks, merge those structural changes while preserving our GB10 speculative decoding policies and K/V-only catch-up slice projections. |
| `src/models/qwen4_exp/{ngram_volume.cpp,ngram_volume.h,ngram_hash.cpp}` | **Keep ours** (`kCacheWays = 4` set-associative cache, `ReaderPool`, `kParallelBlocks = 16`, parallel `NgramHash::row_ids`, batched multi-lane `read_rows`). | Do not revert to the direct-mapped 1-way cache or per-call `std::thread` spawning. |
| `src/ops/softmax_attention/causal_cache/**` | **Keep ours** (`mxfp8_tiled_mma.cuh` and `int8/fast_tiled_mma.cuh` direct FP32 accumulation, `Nvfp4G16TiledValues::tile_shift() == 0`, `mxfp8_tiled_plan.h` split penalty `3.5 / 24.0`, `nvfp4/plan.cpp` fast-prompt gate, `column_invariant.cu` 4-way ternary, split `int8/{tiled_launch,fast_prompt_launch}.cu`). | Do not re-introduce `partial_h` FP16 accumulation or per-tile scale reductions in `Nvfp4G16TiledValues::tile_shift` — that directly regresses `nvfp4` and `k8v4` decode throughput by ~15–20%. |
| `src/ops/kv_cache/{q4_lloyd_codec.cuh,vq2_codec.cuh}` & `src/ops/softmax_attention/common/vq2_kernel.cuh` | **Keep ours**: carry-free `(mag ^ mask) + (mask & 0x01010101u)` in `q4_lloyd_codec.cuh`, `__vsub4` in `vq2_codec.cuh`, and `INFERNIX_VQ2_PAIRED_FP32=0` in `vq2_kernel.cuh`. | Changing `vq2_codec.cuh` to byte-add corrupts zero lanes; enabling paired FP32 in `vq2_kernel.cuh` breaks `--lossless` / exact settle parity on `sm_121a`. |
| `src/ops/linear/**`, `src/ops/sparse_moe/**`, `src/ops/offloaded_sparse_moe/**` | **Keep ours** for all `/shapes/n<N>_k<K>.{cu,cpp}` files, `a4_gemm_64x128.cuh`, `a8_gemm_128x128.cuh`, `a8_gemm_16x64.cuh`, `iq3s/kernels/dp4a_gemv.cuh`, `neon_mad.h`, and `expert_team.cpp` Cortex-X925 pinning. | After resolving any kernel conflict, always run `ctest --test-dir build-native -R infernix_cuda_stack_check --output-on-failure` to verify zero new stack spills on `sm_121a`. |
