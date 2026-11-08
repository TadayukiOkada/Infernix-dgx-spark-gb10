# GB10 (`sm_121a`) `src/runtime` & `src/core` Package Audit & Optimization Task Roadmap (2026-11-08)

- **Date:** 2026-11-08
- **Target Architecture:** NVIDIA DGX Spark GB10 (`sm_121a`, 48 SMs, 99 KiB L1/SMEM per SM, 65,536 32-bit registers per SM, 24 MiB L2 cache, 128 GiB unified LPDDR5X UMA at ~273 GB/s peak, 20-core Grace ARMv9.2 CPU: 10 Cortex-X925 + 10 Cortex-A725, 1:1 FP32/FP16 Tensor Core accumulation parity)
- **Baseline Commit:** `c0643d404` (`dev-rebased-20261028`)
- **Scope Audited:** Entire `src/runtime/` (62 files) and `src/core/` (66 files) packages
- **Focus Areas:**
  1. **Specific RTX 5090 (`sm_120a`, 170 SMs, 96 MiB L2, discrete PCIe, GDDR7 1792 GB/s) Constants & Heuristics Left in Active Code**
  2. **Unnecessary Reads/Writes, Device Workspace Over-Allocations, Host Heap Allocations, and Memory Transfers**
  3. **Inefficient or Redundant Operations, Uncached CUDA Runtime/Driver Queries, Synchronous Pipeline Stalls, and CPU Topology / Telemetry Bugs**

---

## Mandatory GB10 (`sm_121a`) Workflow & Architectural Invariants

1. **Strict 1-Change-At-A-Time Protocol**:
   - Every single task must be executed independently in isolation: edit -> build -> test -> commit.
   - Revert any non-working change immediately before attempting another design.
2. **Binary Build & nvlink Amortization**:
   - Always shut down any running `infernix-serve` process (`pkill -f infernix-serve`) before rebuilding targets to prevent file-locking write collisions.
   - Build binaries using:
     ```bash
     ninja -C build-native -j "$(nproc)" infernix-serve infernix_tests
     ```
3. **Quality & Functional Verification Ratchet**:
   - Run the server in the background:
     ```bash
     ./tools/run_server.sh &
     ```
   - Execute the short tool evaluation suite:
     ```bash
     tool-eval-bench run --short --base-url http://127.0.0.1:8000/v1
     ```
   - Ensure the quality score is strictly **100 / 100** with **0 partial or failed** test cases.
4. **One-Way CUDA Stack-Spill Ratchet Gate (`infernix_cuda_stack_check`)**:
   - Every task must pass:
     ```bash
     ctest --test-dir build-native -R infernix_cuda_stack_check --output-on-failure
     ```
   - Zero new `STACK > 0` or `LOCAL > 0` symbols are permitted.
5. **Mandatory GB10 Changelog Entry (`docs/gb10/YYYYMMDD-changelog.md`)**:
   - Record accounting, commit hash, architectural rationale, subsystem breakdown, and verification evidence for each completed milestone.

---

## 1. Executive Summary & Audit Inventory Tables

### 1.1 Specific RTX 5090 (`sm_120a`) Assumptions & Cost Constants

| ID | Subsystem / File | Symbol / Constant | RTX 5090 Assumption | Impact on GB10 (`48 SMs`, `24 MiB L2`, UMA, Grace CPU) | Task ID |
|---|---|---|---|---|---|
| **C1** | [`context_cost_defaults.cpp:89`](file:///opt/etc/Infernix/src/runtime/prefix_cache/context_cost_defaults.cpp#L89) | `preset.prefill = {}` | RTX 5090 `kGroupwise27bPrefill` profile | Evaluates GB10 prefill as 3.5x–6x cheaper than reality, distorting hybrid prefix cache retention/eviction decisions | **T2.1** |
| **C2** | [`model_instance.cpp:373`](file:///opt/etc/Infernix/src/runtime/engine/model_instance.cpp#L373), [`qwen4_exp_instance.cpp:430`](file:///opt/etc/Infernix/src/runtime/engine/qwen4_exp_instance.cpp#L430) | `kDefaultHostCacheBytes = 8 GiB` | Discrete GPU with separate CPU RAM | Reserves 8 GiB pinned host cache in the same physical 128 GiB LPDDR5X UMA, causing double-allocation and bandwidth-sapping H2D copies | **T3.1** |
| **C3** | [`model_instance.cpp:710`](file:///opt/etc/Infernix/src/runtime/engine/model_instance.cpp#L710) | `default_headroom = 0ULL` (integrated) | Integrated SoC with no host cache | Sizing KV capacity with 0 headroom before allocating the 8 GiB pinned host cache triggers out-of-memory risks under concurrency | **T3.2** |
| **C4** | [`model_instance.cpp:304`](file:///opt/etc/Infernix/src/runtime/engine/model_instance.cpp#L304) | `prefill_chunk = 16384U` | PCIe 5.0 streaming amortization | Forces massive 16K prefill activation reservations and coarse decode interleaving on unified device-resident MoE | **T3.3** |
| **C5** | [`link_probe.cu:25`](file:///opt/etc/Infernix/src/core/link_probe.cu#L25) | `kWarmMilliseconds = 200.0F` | Discrete PCIe link power ramp | Spins all SMs for 200 ms to wake PCIe Gen5 link; completely unnecessary on UMA | **T3.4** |
| **C6** | [`block_copy.cu:17`](file:///opt/etc/Infernix/src/core/block_copy.cu#L17) | `kSlices = 16` | 170-SM wave divisor | On GB10 (48 SMs), launching 1-block copies with 16 slices leaves 32 SMs (67%) idle | **T3.5** |
| **C7** | [`device.cu:425`](file:///opt/etc/Infernix/src/core/device.cu#L425) | `blocks = std::min(..., 64)` | 64/72 SM legacy GPUs | Clips upload grid size below GB10 2-wave sizing (96 CTAs) | **T3.6** |

---

### 1.2 Inefficient / Redundant Operations, Uncached Queries & Stalls

| ID | Subsystem / File | Symbol / Operation | Overhead Mechanism | Impact on GB10 Engine Throughput | Task ID |
|---|---|---|---|---|---|
| **O1** | [`paged_kv_cache.cpp:884-889, 922, 969`](file:///opt/etc/Infernix/src/core/paged_kv_cache.cpp#L884-L889) | `device_max_copy_pitch()` | Calls `cudaGetDevice` & `cudaDeviceGetAttribute` on every KV spill/restore | Hot-path synchronous CUDA driver syscall latency during context swapping | **T1.1** |
| **O2** | [`cpu_topology.cpp:240-244`](file:///opt/etc/Infernix/src/core/cpu_topology.cpp#L240-L244) | `performance_cpus()` | `if (!arm_performance_part(midr)) return {};` | Aborts MIDR fallback on first efficiency core (A725), returning empty vector and disabling big-core pinning | **T1.2** |
| **O3** | [`gpu_telemetry.cpp:105`](file:///opt/etc/Infernix/src/core/gpu_telemetry.cpp#L105), [`vram_budget.cpp:47`](file:///opt/etc/Infernix/src/core/vram_budget.cpp#L47) | `nvmlDeviceGetHandleByPciBusId_v2` | Formats PCIe bus string on integrated SoC | Handle lookup fails on integrated GB10, disabling all telemetry (clocks, temps, power) | **T1.3** |
| **O4** | [`host_expert_tier.cpp:48-51`](file:///opt/etc/Infernix/src/runtime/expert_offload/host_expert_tier.cpp#L48-L51) | `relax()` | Empty stub on ARM | Hot spin-wait loop burns execution slots without yielding pipeline resources | **T1.4** |
| **O5** | [`engine_core.h:2240, 2902`](file:///opt/etc/Infernix/src/runtime/engine/engine_core.h#L2240) | `publish_runtime_stats()` | Double invocation per step | Acquires `queue_mutex_`, runs prefix cache scans, and copies 1.5 KiB struct twice per token/step | **T2.2** |
| **O6** | [`engine_core.h:1845`](file:///opt/etc/Infernix/src/runtime/engine/engine_core.h#L1845) | `hold_queued_sources()` | 3 heap `std::vector` allocations per iteration | Constant allocator churn in worker thread loop even when queue is unchanged | **T2.3** |
| **O7** | [`prefix_index.cpp:1787`](file:///opt/etc/Infernix/src/runtime/prefix_cache/prefix_index.cpp#L1787) | `PrefixCacheIndex::stats()` | Linear scans over `holds_` and slabs | Scans up to 4,096 entries on every stats query | **T2.4** |
| **O8** | [`arena.cu:98, 114`](file:///opt/etc/Infernix/src/core/arena.cu#L98) | `DeviceBuffer::fill`, `copy_from_host` | `cudaStreamSynchronize(nullptr)` | Serializes legacy default stream against all concurrent streams in process | **T3.7** |
| **O9** | [`expert_offload.cpp:605-621`](file:///opt/etc/Infernix/src/runtime/expert_offload/expert_offload.cpp#L605-L621) | `measure_cpu_rate_settled()` | 150 background H2D copies & sleep loop | 4s–8s synchronous blocking delay during startup | **T3.8** |
| **O10** | [`copy_batch.cpp:20-25`](file:///opt/etc/Infernix/src/core/copy_batch.cpp#L20-L25) | `CopyBatch::add()` | `std::map<uintptr, uintptr> ranges_` | Heap node allocations and RB-tree lookups during batch copy assembly | **T3.9** |

---

## 2. Work Packages & Actionable Tasks

### Work Package 1: Tier 1 - High-Impact Driver, Topology & Telemetry Repairs (Low Risk)

#### Task 1.1: Cache `device_max_copy_pitch()` in [`src/core/paged_kv_cache.cpp`](file:///opt/etc/Infernix/src/core/paged_kv_cache.cpp) [COMPLETED]
- **Problem**: `device_max_copy_pitch()` is called on every `copy_to_host_records` and `copy_from_host_records` invocation. It calls `cudaGetDevice` and `cudaDeviceGetAttribute(&pitch, cudaDevAttrMaxPitch, device)` synchronously during active KV swapping.
- **Action**: Cached the maximum pitch statically in the anonymous namespace via thread-safe static initialization, shared between `initialize_host_transfer_plan()`, `copy_to_host_records()`, and `copy_from_host_records()`.
- **Commit**: `5fee04a37` (`perf(core): cache device_max_copy_pitch in paged kv cache to eliminate driver queries`)
- **Verification**: `infernix_kv_cache_test` passed; `infernix_cuda_stack_check` passed (1.49s, 0 stack spills); `tool-eval-bench run --scenarios TC-14` scored **100 / 100** (2/2 points, 0 partial, 0 failed, TTFT 1,223 ms).

#### Task 1.2: Fix Grace Hybrid CPU Fallback Bug in [`src/core/cpu_topology.cpp`](file:///opt/etc/Infernix/src/core/cpu_topology.cpp) [COMPLETED]
- **Problem**: In `performance_cpus()` ([`lines 240-244`](file:///opt/etc/Infernix/src/core/cpu_topology.cpp#L240-L244)), the loop iterates over all CPUs. When it encounters the first efficiency core (A725), `!arm_performance_part(midr)` triggers `return {};`, immediately aborting and returning an empty list instead of skipping non-performance cores.
- **Action**: Replaced the combined error/check with `if (!arm_performance_part(midr)) { continue; }`, allowing all Cortex-X925 performance cores to be discovered and registered.
- **Commit**: `0279d6dbf` (`fix(core): skip efficiency cores instead of aborting in ARM hybrid cpu fallback`)
- **Verification**: `infernix_cpu_topology_test` passed (reported `hybrid=yes, big cpus: 5 6 7 8 9 15 16 17 18 19`); `infernix_cuda_stack_check` passed (1.45s, 0 spills); `tool-eval-bench run --scenarios TC-14` scored **100 / 100** (2/2 points, 0 partial, 0 failed, TTFT 1,166 ms).

#### Task 1.3: Add NVML Handle Fallback for Integrated SoC in [`src/core/gpu_telemetry.cpp`](file:///opt/etc/Infernix/src/core/gpu_telemetry.cpp) & [`src/core/vram_budget.cpp`](file:///opt/etc/Infernix/src/core/vram_budget.cpp) [COMPLETED]
- **Problem**: Calling `nvmlDeviceGetHandleByPciBusId_v2` with formatted PCIe bus string fails on integrated GB10 SoC. This sets `device_ = nullptr` and causes telemetry to report all zeros or invalid data.
- **Action**: In `gpu_telemetry.cpp` and `vram_budget.cpp`, added fallback to `nvmlDeviceGetHandleByIndex_v2` (and legacy `nvmlDeviceGetHandleByIndex`) whenever PCIe bus ID handle acquisition fails or returns null.
- **Commit**: `d545cafca` (`fix(core): add NVML device handle by index fallback for integrated SoC telemetry and vram budget`)
- **Verification**: Server correctly reports live GPU telemetry (`gpu 2548/0 MHz 54 W 50 C`); `infernix_cuda_stack_check` passed (1.47s, 0 spills); `tool-eval-bench run --scenarios TC-14` scored **100 / 100** (2/2 points, 0 partial, 0 failed, TTFT 1,176 ms).

#### Task 1.4: Implement ARM `yield` in Spin Loop in [`src/runtime/expert_offload/host_expert_tier.cpp`](file:///opt/etc/Infernix/src/runtime/expert_offload/host_expert_tier.cpp)
- **Problem**: `relax()` on non-x86 is currently an empty inline function. In busy-spin loops waiting for worker completions, this burns pipeline execution resources on Grace ARMv9.2 cores.
- **Action**: Implement `relax()` for `__aarch64__` using `asm volatile("yield" ::: "memory");` or `__builtin_arm_yield()`.
- **Verification**: Build `infernix-serve`; verify `tool-eval-bench run --short` maintains 100/100 score.

---

### Work Package 2: Tier 2 - Prefill Cost Modeling & Engine Runtime Hot-Path Optimizations

#### Task 2.1: Populate GB10 Prefill Cost Table in [`src/runtime/prefix_cache/context_cost_defaults.cpp`](file:///opt/etc/Infernix/src/runtime/prefix_cache/context_cost_defaults.cpp)
- **Problem**: `"nvidia-gb10-sm121"` preset has empty `.prefill = {}`. Fallback to RTX 5090 table values prices prefill ~4× cheaper than reality, distorting retention thresholds in `hybrid_cache_arbiter.cpp`.
- **Action**: Populate `.prefill` in `kGb10Defaults` with measured GB10 prefill costs (calibrated against GB10's 273 GB/s LPDDR5X UMA vs RTX 5090's 1792 GB/s GDDR7).
- **Verification**: Run `build-native/tests/infernix_context_cost_test`; verify hybrid prefix cache keeps high-value prefixes in device memory.

#### Task 2.2: Eliminate Duplicate `publish_runtime_stats()` in [`src/runtime/engine/engine_core.h`](file:///opt/etc/Infernix/src/runtime/engine/engine_core.h)
- **Problem**: `publish_runtime_stats()` is invoked twice at the end of each decode/prefill round ([`lines 2240 and 2902`](file:///opt/etc/Infernix/src/runtime/engine/engine_core.h#L2240)), acquiring `queue_mutex_`, running prefix index scans, and copying 1.5 KiB structures redundantly.
- **Action**: Remove the redundant second consecutive call in worker step loop.
- **Verification**: Run `tool-eval-bench run --short --base-url http://127.0.0.1:8000/v1` and ensure 100/100 score.

#### Task 2.3: Eliminate Per-Iteration Heap Vector Allocations in `hold_queued_sources()` in [`src/runtime/engine/engine_core.h`](file:///opt/etc/Infernix/src/runtime/engine/engine_core.h)
- **Problem**: `hold_queued_sources()` allocates 3 `std::vector`s on every worker iteration even when the queue is idle or unchanged.
- **Action**: Keep reusable scratch vectors in `EngineCore` member state and `.clear()` them between iterations.
- **Verification**: Verify zero regression on generation tests and benchmarks.

#### Task 2.4: Optimize `PrefixCacheIndex::stats()` in [`src/runtime/prefix_cache/prefix_index.cpp`](file:///opt/etc/Infernix/src/runtime/prefix_cache/prefix_index.cpp)
- **Problem**: `PrefixCacheIndex::stats()` performs full linear scans over `holds_` and slabs (up to 4,096 entries) on every stats publication.
- **Action**: Maintain running incremental counters for active holds and slab residency instead of re-scanning arrays.
- **Verification**: Run `ctest -R infernix_prefix_cache_test` and verify stats consistency.

---

### Work Package 3: Tier 3 - GB10 UMA Memory Sizing, PCIe Bypass & Grid Sizing

#### Task 3.1: Right-Size / Bypass 8 GiB Pinned Host Cache on Integrated UMA in [`src/runtime/engine/model_instance.cpp`](file:///opt/etc/Infernix/src/runtime/engine/model_instance.cpp)
- **Problem**: On GB10 UMA, allocating an 8 GiB pinned host cache wastes 8 GiB of the shared 128 GiB physical LPDDR5X pool. Moving pages between "host" and "device" transfers across the same memory bus.
- **Action**: When `props.integrated != 0` and device MoE is active, reduce host cache allocation to minimal staging or bypass host KV copy staging.
- **Verification**: Run `infernix-serve` on large models; observe available device VRAM increase.

#### Task 3.2: Fix Headroom Calculation for Integrated SoC in [`src/runtime/engine/model_instance.cpp`](file:///opt/etc/Infernix/src/runtime/engine/model_instance.cpp)
- **Problem**: Line 710 sets `default_headroom = 0ULL` on integrated GPUs. If host memory allocations occur subsequently, high concurrency triggers out-of-memory errors.
- **Action**: Enforce a safe minimum headroom (e.g. 512 MiB - 1 GiB) on integrated systems to protect against activation memory spikes.
- **Verification**: Run concurrency stress test with multiple parallel streams.

#### Task 3.3: Bypass 200 ms PCIe Link Ramp-Up on Integrated GPU in [`src/core/link_probe.cu`](file:///opt/etc/Infernix/src/core/link_probe.cu)
- **Problem**: `kWarmMilliseconds = 200.0F` spins all SMs for 200 ms to wake PCIe Gen5 link from low-power state. On GB10 UMA, there is no PCIe link to warm up.
- **Action**: Guard the warm-up spin: if `device_is_integrated`, set warm-up time to 0 ms.
- **Verification**: Measure engine startup latency reduction (~200 ms faster).

#### Task 3.4: Right-Size Block Copy Slice Count for 48 SMs in [`src/core/block_copy.cu`](file:///opt/etc/Infernix/src/core/block_copy.cu)
- **Problem**: `kSlices = 16` leaves 32 of 48 SMs idle when copying a single block.
- **Action**: Size `kSlices` dynamically based on `device_sms` or set `kSlices = 24` or `48` to saturate all 48 SMs on GB10.
- **Verification**: Run `ctest -R infernix_block_copy_test`; verify copy throughput.

#### Task 3.5: Eliminate Synchronous Device Waits in `arena.cu` in [`src/core/arena.cu`](file:///opt/etc/Infernix/src/core/arena.cu)
- **Problem**: `DeviceBuffer::fill` and `copy_from_host` call `cudaStreamSynchronize(nullptr)` to settle the legacy default stream, blocking all streams on the GPU.
- **Action**: Use explicit caller stream or replace with stream-ordered asynchronous memset/memcpy.
- **Verification**: Verify all arena unit tests pass.

#### Task 3.6: Reduce 8-Second Settled CPU Rate Measurement Delay in [`src/runtime/expert_offload/expert_offload.cpp`](file:///opt/etc/Infernix/src/runtime/expert_offload/expert_offload.cpp)
- **Problem**: `measure_cpu_rate_settled()` runs a 4s–8s sleep loop with 150 background copies.
- **Action**: Reduce repetition count and sampling interval when UMA is detected.
- **Verification**: Measure startup latency improvement.

#### Task 3.7: Optimize `CopyBatch::add()` Overlap Checking in [`src/core/copy_batch.cpp`](file:///opt/etc/Infernix/src/core/copy_batch.cpp)
- **Problem**: `ranges_` is a `std::map` performing dynamic tree allocations per add.
- **Action**: Use a flat contiguous vector of intervals sorted on flush, avoiding node allocations.
- **Verification**: Verify batch copy correctness with `infernix_tests`.

---

## 3. Execution Priority & Acceptance Criteria

| Phase | Tasks Included | Focus & Outcome | Risk Level |
|---|---|---|---|
| **Phase 1** | Tasks 1.1, 1.2, 1.3, 1.4 | Core driver syscalls, ARM CPU topology fix, NVML telemetry restoration | Low |
| **Phase 2** | Tasks 2.1, 2.2, 2.3, 2.4 | Prefill cost modeling, duplicate stats publishing, heap churn elimination | Low / Medium |
| **Phase 3** | Tasks 3.1, 3.2, 3.3, 3.4, 3.5 | UMA memory budgeting, PCIe warmup bypass, 48-SM block copy wave sizing | Medium |

### Acceptance Criteria
1. **Quality Ratchet**: 100/100 score on `tool-eval-bench run --short --base-url http://127.0.0.1:8000/v1` with zero failed/partial test cases.
2. **CUDA Stack Spill Ratchet**: 0 new `STACK > 0` or `LOCAL > 0` symbols on `ctest -R infernix_cuda_stack_check`.
3. **Hardware Alignment**: Proper Grace Cortex-X925 core utilization, live NVML telemetry on GB10, and zero PCIe warmup latency on integrated SoC.
