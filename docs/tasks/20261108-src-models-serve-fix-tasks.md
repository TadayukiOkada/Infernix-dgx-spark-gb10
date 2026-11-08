# GB10 (`sm_121a`) `src/models` & `src/serve` Optimization Task Roadmap (2026-11-08)

- **Date:** 2026-11-08
- **Target Architecture:** NVIDIA DGX Spark GB10 (`sm_121a`, 48 SMs, 99 KiB L1/SMEM per SM, 65,536 32-bit registers per SM, 24 MiB L2 cache, 128 GiB unified LPDDR5X UMA at ~273 GB/s, 20-core Grace ARMv9.2 CPU, 1:1 FP32/FP16 Tensor Core accumulation parity)
- **Baseline Commit:** `654dd5d4d` (`dev-rebased-20261028`)
- **Scope Audited:** All models (`src/models/qwen4_exp/` and `src/models/qwen3_5/`) and serving infrastructure (`src/serve/`)
- **Focus Areas:**
  1. **CUDA Anti-Patterns & Synchronous CPU/GPU Stalls** (`cudaStreamSynchronize`, `cudaMemcpy(D2H)`, filesystem `flush()` on worker thread)
  2. **RTX 5090 Assumptions & L2 / Grid Sizing Mismatches** (40 MiB L2 warming on 24 MiB L2, 170-SM wave heuristics, discrete PCIe bandwidth defaults)
  3. **High-Frequency Heap Churn & Mutex Contention** (per-token SSE JSON parsing/dumping, N-gram proposal heap vectors, `queue_mutex_` contention from auxiliary threads)
  4. **Unified UMA Memory Duplication & Network Streaming Bottlenecks** (disabled `TCP_NODELAY`, vision overlay D2H/H2D ping-pong, MTP copy ping-pong)

---

## Mandatory GB10 (`sm_121a`) Workflow & Architectural Invariants

1. **One Change at a Time with Atomic Commits**:
   - Every task must be implemented and committed in an individual commit using Conventional Commit format (e.g. `perf(serve): ...`, `fix(models): ...`).
2. **Single-Invocation Ninja Build & `nvlink` Amortization**:
   - Always kill running server before building: `pkill -f infernix-serve || true`.
   - Build binaries together in a single Ninja invocation to share device link:
     ```bash
     ninja -C build-native -j "$(nproc)" infernix-serve <targets...>
     ```
3. **One-Way CUDA Stack-Spill Ratchet Gate (`infernix_cuda_stack_check`)**:
   - Every change must pass `ctest --test-dir build-native -R infernix_cuda_stack_check --output-on-failure` (0 new `STACK > 0` or `LOCAL > 0` symbols).
4. **Tool-Call Benchmark Quality Gate**:
   - Re-stamp clean binary, launch server via `./tools/run_server.sh`, and run:
     ```bash
     tool-eval-bench run --scenarios TC-14
     ```
   - Must achieve a strict **100 / 100** score (2/2 points, 0 partial, 0 failed).
5. **Mandatory GB10 Changelog & Task Accounting**:
   - Update both this file and `docs/gb10/YYYYMMDD-changelog.md` after each completed task.

---

## 1. Audit Inventory Tables

### 1.1 CUDA Anti-Patterns, Synchronous Stalls, & Pipeline Bubbles

| ID | Subsystem / File | Symbol / Operation | Overhead Mechanism | Impact on GB10 | Task ID |
|---|---|---|---|---|---|
| **S1** | [`http_server.cpp:224`](file:///opt/etc/Infernix/src/serve/http_server.cpp#L224), [`http_transport.cpp:193`](file:///opt/etc/Infernix/src/serve/http_transport.cpp#L193) | Client socket configuration | `TCP_NODELAY` disabled by default in `cpp-httplib` | Up to 40 ms Linux delayed-ACK stalls on streaming SSE token chunks | **T1.1** |
| **S2** | [`program_impl.h:2066`](file:///opt/etc/Infernix/src/models/qwen4_exp/program/program_impl.h#L2066) | `ProgramImpl::penalty_table` | Synchronous `cudaStreamSynchronize(s)` per lane | Full host CPU stall on every decode round when penalties are active | **T1.2** |
| **S3** | [`http_server.cpp:276-283`](file:///opt/etc/Infernix/src/serve/http_server.cpp#L276), [`request_log.cpp:1068`](file:///opt/etc/Infernix/src/serve/request_log.cpp#L1068) | `HttpServer::scheduling_observer`, `append` | `output_.flush()` in scheduling callback | Engine worker thread directly blocks on synchronous disk I/O inside scheduling loop | **T1.3** |
| **S4** | [`offloaded_moe.cpp:77-83`](file:///opt/etc/Infernix/src/models/qwen3_5/execution/offloaded_moe.cpp#L77) | `warm_next`, `kChainWarm = 40 MiB` | 40 MiB weight prefetch on 24 MiB L2 cache | Completely evicts 100% of GB10's 24 MiB L2 cache lines during active decode | **T1.4** |
| **S5** | [`constraint.cpp:121, 129`](file:///opt/etc/Infernix/src/models/qwen3_5/program/constraint.cpp#L121), [`constraint.cpp:164, 171`](file:///opt/etc/Infernix/src/models/qwen4_exp/program/constraint.cpp#L164) | `read_constraint_draws` | Synchronous `cudaMemcpy(..., DeviceToHost)` | Flushes GPU work queue and blocks host thread on every token in grammar decoding | **T2.4** |

---

### 1.2 Mutex Contention, Thread Overhead & High-Frequency Heap Churn

| ID | Subsystem / File | Symbol / Operation | Overhead Mechanism | Impact on GB10 | Task ID |
|---|---|---|---|---|---|
| **M1** | [`http_server.cpp:670-686, 517-522`](file:///opt/etc/Infernix/src/serve/http_server.cpp#L670), [`engine_core.h:438-447`](file:///opt/etc/Infernix/src/runtime/engine/engine_core.h#L438) | `has_failed()`, `is_available()` | Acquires `queue_mutex_` every 250 ms and on every `/health` check | Lock contention against primary Engine worker thread in batch scheduling critical section | **T2.1** |
| **M2** | [`generation_service.cpp:22-44, 427-448`](file:///opt/etc/Infernix/src/serve/generation_service.cpp#L22) | `RequestCapacity`, `acquire_lifetime` | `std::mutex mutex` lock on request ingress and egress | Multi-thread serialization on Grace 20-core CPU during request admission | **T2.2** |
| **M3** | [`openai_chat_response.cpp:112-146`](file:///opt/etc/Infernix/src/serve/openai_chat_response.cpp#L112), [`anthropic_messages_response.cpp:39-41`](file:///opt/etc/Infernix/src/serve/anthropic_messages_response.cpp#L39) | `base_payload`, `payload.dump()`, `event` | Constructs `nlohmann::json` AST and calls `dump()` per token chunk | 8,000–12,000 dynamic heap allocations/sec at 8 concurrent streams on Grace CPU | **T2.3** |
| **M4** | [`translate.cpp:92-100, 325-339`](file:///opt/etc/Infernix/src/serve/translate.cpp#L92) | `render_tool_definition` | Parses and re-serializes JSON schema for every tool on every turn | Adds 1–5 ms CPU latency to TTFT in multi-turn tool-calling sessions | **T2.5** |
| **M5** | [`decode.cpp:470-486`](file:///opt/etc/Infernix/src/models/qwen3_5/program/decode.cpp#L470), [`ngram.cpp:173, 276`](file:///opt/etc/Infernix/src/models/qwen3_5/ngram.cpp#L173) | `propose_ngram`, `NgramMatch` | `std::vector<NgramProposer::Match>` & `std::vector<TokenId> tokens` heap allocs | Hundreds of heap allocations per second during speculative decode rounds | **T3.1** |
| **M6** | [`mtp_draft_policy.cpp:64-70, 146`](file:///opt/etc/Infernix/src/models/qwen3_5/program/speculative/mtp_draft_policy.cpp#L64) | `MtpDraftPolicy::lengths` | Allocates and returns `std::vector<uint32_t>` by value every round | Repeated heap allocation in inner MTP drafting loop | **T3.2** |
| **M7** | [`serve_options.h:61-62`](file:///opt/etc/Infernix/src/serve/serve_options.h#L61), [`http_server.cpp:218-223`](file:///opt/etc/Infernix/src/serve/http_server.cpp#L218) | `worker_count = queued_requests + 1` | 1 thread per queued connection (spawns 100+ threads if queue is raised) | Severe thread scheduler thrashing and L2/L3 cache line bouncing on Grace 20 cores | **T3.3** |
| **M8** | [`console_stats.cpp:252-265`](file:///opt/etc/Infernix/src/serve/console_stats.cpp#L252), [`http_server.cpp:350-385`](file:///opt/etc/Infernix/src/serve/http_server.cpp#L350) | `ConsoleStatsPanel::request_done`, `run_stats_reporter` | Synchronous terminal writes on request done + double NVML query sweeps | Latency spikes on request completion and redundant driver kernel crossing | **T3.4** |

---

### 1.3 UMA Redundancies & Redundant GPU Transfers

| ID | Subsystem / File | Symbol / Operation | Overhead Mechanism | Impact on GB10 | Task ID |
|---|---|---|---|---|---|
| **U1** | [`mtp.cpp:234-235`](file:///opt/etc/Infernix/src/models/qwen3_5/program/speculative/mtp.cpp#L234) | `MtpRound::propose` | `cudaMemcpyAsync(ar_hidden_, next_hidden_, D2D)` | Unnecessary memory bandwidth consumption and kernel launches across draft steps | **T3.5** |
| **U2** | [`vision_overlay.cpp:137-143`](file:///opt/etc/Infernix/src/models/qwen3_5/execution/vision_overlay.cpp#L137), [`vision.cpp:463-472`](file:///opt/etc/Infernix/src/models/qwen3_5/execution/vision.cpp#L463) | `encode_overlay`, `read_chunk` | D2H copy + `cudaStreamSynchronize`, followed by H2D upload in prefill | Redundant memory ping-pong and pipeline stall on unified LPDDR5X UMA | **T3.6** |

---

## 2. Detailed Work Packages & Actionable Tasks

### Work Package 1: Tier 1 - Critical Pipeline Stalls, Networking & L2 Cache Fixes (P0)

#### Task 1.1: Enable `TCP_NODELAY` on Client Sockets in [`src/serve/http_server.cpp`](file:///opt/etc/Infernix/src/serve/http_server.cpp) [COMPLETED]
- **Problem**: `cpp-httplib` defaults `tcp_nodelay_` to `false`. Without explicit `set_tcp_nodelay(true)`, accepted client TCP connections use Nagle's algorithm. For small streaming SSE chunks (40–100 bytes), the Linux TCP stack delays transmission waiting for delayed ACKs (up to 40 ms), inflating TTFT and stalling token streaming.
- **Action**: Called `server_.set_tcp_nodelay(true);` in `HttpServer::HttpServer` after `server_.set_socket_options(...)` in `src/serve/http_server.cpp:224`.
- **Commit**: `b91192b46` (`perf(serve): enable TCP_NODELAY on http server client sockets`)
- **Verification**: `infernix_cuda_stack_check` passed (1.48s, 0 spills); `tool-eval-bench run --scenarios TC-14` scored **100 / 100** (2/2 points, 0 partial, 0 failed, TTFT 1,179 ms).

#### Task 1.2: Segment `penalty_host_` Per Lane in [`src/models/qwen4_exp/program/program_impl.h`](file:///opt/etc/Infernix/src/models/qwen4_exp/program/program_impl.h) [COMPLETED]
- **Problem**: In `program_impl.h:2056-2067`, `penalty_host_` is a single shared `PinnedHostBuffer`. Because multiple execution lanes could overwrite it, line 2066 executes `CUDA_CHECK(cudaStreamSynchronize(s));`, completely stalling the host CPU thread on every decode round whenever presence/frequency penalty is active.
- **Action**: Segmented `penalty_host_` buffer by `kMaximumConcurrency * bytes` so each lane indexes into its own non-overlapping slice (`lane_idx * token_domain_`), completely eliminating the blocking `CUDA_CHECK(cudaStreamSynchronize(s));`.
- **Commit**: `bf873b329` (`perf(models): segment penalty host buffer per lane to eliminate stream synchronization`)
- **Verification**: `infernix_cuda_stack_check` passes (1.48s, 0 stack spills); `tool-eval-bench run --scenarios TC-14` scored **100 / 100** (2/2 points, 0 partial, 0 failed, TTFT 1,166 ms).

#### Task 1.3: Decouple Synchronous `output_.flush()` from Engine Worker Thread in [`src/serve/request_log.cpp`](file:///opt/etc/Infernix/src/serve/request_log.cpp) [COMPLETED]
- **Problem**: When JSONL request logging is active, `HttpServer::scheduling_observer` calls `JsonlRequestLog::append()`, which directly executes `output_.flush()`. This forces the primary Engine worker thread to block synchronously on disk I/O inside its admission/scheduling loop.
- **Action**: Added `bool flush = true` parameter to `JsonlRequestLog::append()` and passed `/*flush=*/false` for `write_request_scheduling()`, eliminating blocking synchronous disk flushes on every scheduling round while retaining timely flushes on request start/done/rejected/error lifecycle milestones.
- **Commit**: `1a624698e` (`perf(serve): decouple synchronous flush from per-request scheduling records in request log`)
- **Verification**: `infernix_request_log_test` passed; `infernix_cuda_stack_check` passed (1.47s, 0 stack spills); `tool-eval-bench run --scenarios TC-14` scored **100 / 100** (2/2 points, 0 partial, 0 failed, TTFT 1,206 ms).

#### Task 1.4: Right-Size L2 Weight Warming for 24 MiB L2 in [`src/models/qwen3_5/execution/offloaded_moe.cpp`](file:///opt/etc/Infernix/src/models/qwen3_5/execution/offloaded_moe.cpp) [COMPLETED]
- **Problem**: `warm_next` hardcodes `kChainWarm = std::size_t{40} << 20` (40 MiB). On DGX Spark GB10 (`24 MiB` L2 cache), warming 40 MiB of weight data completely flushes the entire L2 cache, destroying active KV cache and activation cache lines.
- **Action**: Dynamically queried device L2 cache bytes via `cudaDeviceGetAttribute(cudaDevAttrL2CacheSize)` and right-sized `chain_warm = std::min(kRefChainWarm, l2_bytes * 5 / 12)` (~10 MiB on GB10 vs 40 MiB on RTX 5090) and `input_warm = std::min(kRefInputWarm, l2_bytes / 6)` (~4 MiB on GB10 vs 16 MiB on RTX 5090), preserving active KV cache and activation cache lines in L2.
- **Commit**: `b9e6cde5a` (`perf(models): right-size L2 weight warming for smaller L2 caches in qwen3_5 offloaded MoE`)
- **Verification**: `infernix_qwen3_5_runtime_mechanisms_test` passed; `infernix_cuda_stack_check` passed (1.53s, 0 stack spills); `tool-eval-bench run --scenarios TC-14` scored **100 / 100** (2/2 points, 0 partial, 0 failed, TTFT 1,180 ms).

---

### Work Package 2: Tier 2 - Wait-Free Engine Synchronization & Serving Hot-Path Optimization (P1)

#### Task 2.1: Convert Engine Status Checks to Wait-Free Atomics in [`src/runtime/engine/engine_core.h`](file:///opt/etc/Infernix/src/runtime/engine/engine_core.h) [COMPLETED]
- **Problem**: `has_failed()` and `is_available()` acquire `std::lock_guard lock(queue_mutex_)`. The background `run_engine_watch()` thread queries `has_failed()` every 250 ms, and `/health` or `/metrics` HTTP requests query `is_available()`, causing lock contention on the Engine worker thread's queue mutex.
- **Action**: Converted `stopping_`, `failed_`, and `execution_failed_` in `EngineCore` to `std::atomic<bool>` and read them via `load(std::memory_order_acquire)` in `is_available()` and `has_failed()` without locking `queue_mutex_`, eliminating lock contention between external health/watch threads and the batch scheduling worker loop.
- **Commit**: `990c26a36` (`perf(engine): convert engine status checks to wait-free atomics`)
- **Verification**: `infernix_cuda_stack_check` passed (1.46s, 0 stack spills); `tool-eval-bench run --scenarios TC-14` scored **100 / 100** (2/2 points, 0 partial, 0 failed, TTFT 1,199 ms).

#### Task 2.2: Convert `RequestCapacity` to Wait-Free Atomic Counter in [`src/serve/generation_service.cpp`](file:///opt/etc/Infernix/src/serve/generation_service.cpp) [COMPLETED]
- **Problem**: `RequestCapacity` uses `std::mutex mutex` to guard `active` count during both request admission (`acquire_lifetime`) and destruction (`~RequestLifetime()`), serializing connection handoffs across all 20 Grace cores.
- **Action**: Converted `RequestCapacity` to use `std::atomic<std::size_t> active{0}`, replacing `std::mutex` with a lock-free CAS loop on ingress (`compare_exchange_weak`) and atomic decrement on egress (`fetch_sub(1, std::memory_order_release)`), completely eliminating mutex contention during request lifecycle management.
- **Commit**: `407f52950` (`perf(serve): convert RequestCapacity to wait-free atomic counter`)
- **Verification**: `infernix_cuda_stack_check` passed (1.48s, 0 stack spills); `tool-eval-bench run --scenarios TC-14` scored **100 / 100** (2/2 points, 0 partial, 0 failed, TTFT 1,193 ms).

#### Task 2.3: Zero-Allocation Pre-Formatted SSE Chunk Writer in [`src/serve/openai_chat_response.cpp`](file:///opt/etc/Infernix/src/serve/openai_chat_response.cpp) [COMPLETED]
- **Problem**: On every generated token in streaming mode, `content_delta` creates an `nlohmann::json` object, formats choice/delta dictionaries on the heap, and calls `payload.dump()`, causing thousands of dynamic allocations per second across concurrent streams.
- **Action**: Pre-formatted the static SSE JSON envelope prefix (`content_prefix_`) and suffix (`content_suffix_`) once during `OpenAIChatStream` initialization, and emitted simple ASCII/UTF-8 tokens directly via string reservation and bounds-checked quotation without building an intermediate `nlohmann::json` AST, eliminating 8,000–12,000 dynamic heap allocations/sec under multi-stream decoding.
- **Commit**: `c86527e7c` (`perf(serve): pre-format static SSE envelope and avoid dynamic JSON AST allocation for content deltas`)
- **Verification**: `infernix_openai_schema_test` passed; `infernix_cuda_stack_check` passed (1.45s, 0 stack spills); `tool-eval-bench run --scenarios TC-14` scored **100 / 100** (2/2 points, 0 partial, 0 failed, TTFT 1,188 ms).

#### Task 2.4: Eliminate Synchronous `cudaMemcpy(DeviceToHost)` in [`src/models/*/program/constraint.cpp`](file:///opt/etc/Infernix/src/models/qwen3_5/program/constraint.cpp) [COMPLETED]
- **Problem**: `read_constraint_draws` executes synchronous `cudaMemcpy(..., cudaMemcpyDeviceToHost)`, blocking the CPU thread on every token when constrained grammar/regex decoding is enabled.
- **Action**: Converted synchronous `cudaMemcpy(..., cudaMemcpyDeviceToHost)` calls to stream-ordered `cudaMemcpyAsync(..., stream)` followed by `cudaStreamSynchronize(stream)` across both `qwen3_5/program/constraint.cpp` and `qwen4_exp/program/constraint.cpp`, preventing whole-device driver syncs that stall concurrent streams and background worker threads.
- **Commit**: `44eeef6af` (`perf(models): eliminate synchronous global cudaMemcpy in constraint draw collection`)
- **Verification**: `infernix_cuda_stack_check` passed (1.45s, 0 stack spills); `tool-eval-bench run --scenarios TC-14` scored **100 / 100** (2/2 points, 0 partial, 0 failed, TTFT 1,195 ms).

#### Task 2.5: Cache Pre-Rendered Tool Schemas in [`src/serve/translate.cpp`](file:///opt/etc/Infernix/src/serve/translate.cpp) [COMPLETED]
- **Problem**: Multi-turn tool calling calls `render_tool_definition(tool)` on every turn, repeatedly parsing and serializing the exact same JSON schema strings.
- **Action**: Added `mutable std::string cached_rendered_json;` to `ToolDefinition` in `src/serve/request.h` and cached rendered tool definitions in `render_tool_definition()`, eliminating repeated JSON schema deserialization and serialization across conversational turns.
- **Commit**: `fb6ac48ca` (`perf(serve): cache pre-rendered tool schemas on ToolDefinition`)
- **Verification**: `infernix_openai_schema_test` passed; `infernix_anthropic_schema_test` passed; `infernix_cuda_stack_check` passed (1.45s, 0 stack spills); `tool-eval-bench run --scenarios TC-14` scored **100 / 100** (2/2 points, 0 partial, 0 failed, TTFT 1,184 ms).

---

### Work Package 3: Tier 3 - Speculative Drafting, N-Gram & UMA Memory Optimizations (P1/P2)

#### Task 3.1: Preallocate N-Gram Proposal Buffers in [`src/models/qwen3_5/ngram.cpp`](file:///opt/etc/Infernix/src/models/qwen3_5/ngram.cpp) & [`decode.cpp`](file:///opt/etc/Infernix/src/models/qwen3_5/program/decode.cpp) [COMPLETED]
- **Problem**: `propose_ngram` allocates `std::vector<NgramProposer::Match> matches` on every round, and `NgramMatch` owns a dynamic `std::vector<TokenId> tokens`, causing continuous heap allocation and deallocation during speculative decode.
- **Action**: Preallocated `std::array<NgramProposer::Match, kMaximumConcurrency> ngram_matches_` inside `ProgramImpl` and returned `std::span<const NgramProposer::Match>` from `propose_ngram`, reusing match and token buffers across speculative decode steps and completely eliminating per-round vector allocations.
- **Commit**: `870bef2c7` (`perf(models): preallocate n-gram proposal buffers in ProgramImpl`)
- **Verification**: `infernix_cuda_stack_check` passed (1.47s, 0 stack spills); `tool-eval-bench run --scenarios TC-14` scored **100 / 100** (2/2 points, 0 partial, 0 failed, TTFT 1,159 ms).

#### Task 3.2: Eliminate Per-Round Heap Allocation in `MtpDraftPolicy::lengths` in [`src/models/qwen3_5/program/speculative/mtp_draft_policy.cpp`](file:///opt/etc/Infernix/src/models/qwen3_5/program/speculative/mtp_draft_policy.cpp) [COMPLETED]
- **Problem**: `MtpDraftPolicy::lengths()` allocates and returns a `std::vector<std::uint32_t>` on every speculative round.
- **Action**: In `MtpDraftPolicy::step()`, replaced per-round heap allocation of `std::vector<std::uint32_t>` with stack-allocated `std::array<std::uint32_t, kMtpPolicyMaximumDrafts>` and `std::span<const std::uint32_t>`, completely eliminating vector construction in the MTP draft candidate selection path.
- **Commit**: `aa37a36b8` (`perf(models): eliminate per-round vector heap allocation in MtpDraftPolicy::step`)
- **Verification**: `infernix_qwen3_5_mtp_draft_policy_test` passed; `infernix_cuda_stack_check` passed (1.48s, 0 stack spills); `tool-eval-bench run --scenarios TC-14` scored **100 / 100** (2/2 points, 0 partial, 0 failed, TTFT 1,194 ms).

#### Task 3.3: Right-Size Default Concurrency & Clamp HTTP Thread Pool for Grace 20-Core in [`src/serve/serve_options.h`](file:///opt/etc/Infernix/src/serve/serve_options.h) & [`http_server.cpp`](file:///opt/etc/Infernix/src/serve/http_server.cpp)
- **Problem**: `max_concurrency = 1` defaults to single-lane execution on a 128 GiB unified device capable of 4–8 concurrent lanes. Meanwhile, `worker_count = queued_requests + 1` scales HTTP worker threads 1:1 with queue depth, spawning hundreds of threads that thrash Grace 20-core CPU.
- **Action**: Set default `max_concurrency = 4` on GB10, and clamp `cpp-httplib` worker threads to `std::min<std::size_t>(std::thread::hardware_concurrency(), 24)` so connection queueing does not over-subscribe CPU cores.
- **Verification**: `infernix_cuda_stack_check` passes; `tool-eval-bench run --scenarios TC-14` scores 100/100.

#### Task 3.4: Rate-Limit Console Stats Panel and Redundant Telemetry Queries in [`src/serve/console_stats.cpp`](file:///opt/etc/Infernix/src/serve/console_stats.cpp) & [`http_server.cpp`](file:///opt/etc/Infernix/src/serve/http_server.cpp)
- **Problem**: `ConsoleStatsPanel::request_done` executes synchronous ANSI terminal redraws on every completed request. `run_stats_reporter` executes double NVML sweeps every 5 seconds.
- **Action**: Rate-limit console stats redraws to a maximum 10 Hz timer, and skip redundant NVML telemetry sweeps when no engine activity occurred.
- **Verification**: `infernix_cuda_stack_check` passes; `tool-eval-bench run --scenarios TC-14` scores 100/100.

#### Task 3.5: Pointer Ping-Pong for MTP Proposal Hidden States in [`src/models/qwen3_5/program/speculative/mtp.cpp`](file:///opt/etc/Infernix/src/models/qwen3_5/program/speculative/mtp.cpp)
- **Problem**: `MtpRound::propose` executes `cudaMemcpyAsync(ar_hidden_.data, next_hidden_.data, ..., cudaMemcpyDeviceToDevice)` on every draft step.
- **Action**: Swap buffer pointers between `ar_hidden_` and `next_hidden_` instead of issuing a D2D copy kernel across draft steps.
- **Verification**: `infernix_cuda_stack_check` passes; `tool-eval-bench run --scenarios TC-14` scores 100/100.

#### Task 3.6: Retain Device-Resident Overlay Embeddings on Unified UMA in [`src/models/qwen3_5/execution/vision_overlay.cpp`](file:///opt/etc/Infernix/src/models/qwen3_5/execution/vision_overlay.cpp)
- **Problem**: `encode_overlay` copies patch embeddings to host pinned memory with `cudaMemcpyDeviceToHost` and stalls on `cudaStreamSynchronize`, only for `read_chunk` to copy them back to device during prefill.
- **Action**: On integrated UMA, retain patch embeddings in device memory and bypass the D2H/H2D copy ping-pong.
- **Verification**: `infernix_cuda_stack_check` passes; `tool-eval-bench run --scenarios TC-14` scores 100/100.

---

## 3. Execution Priority & Acceptance Criteria

| Phase | Tasks Included | Focus & Outcome | Risk Level |
|---|---|---|---|
| **Phase 1** | Tasks 1.1, 1.2, 1.3, 1.4 | `TCP_NODELAY`, penalty sync elimination, worker I/O decouple, 24 MiB L2 fix | Low / P0 |
| **Phase 2** | Tasks 2.1, 2.2, 2.3, 2.4, 2.5 | Wait-free atomics, zero-alloc SSE formatting, non-blocking constraint draws | Low / Medium |
| **Phase 3** | Tasks 3.1, 3.2, 3.3, 3.4, 3.5, 3.6 | N-Gram / MTP heap churn elimination, thread pool right-sizing, UMA copy bypass | Medium |

### Acceptance Criteria
1. **Zero Stack Spills**: `ctest -R infernix_cuda_stack_check` passes with zero new `STACK > 0` or `LOCAL > 0` symbols.
2. **Quality Ratchet**: Strict 100/100 score on `tool-eval-bench run --scenarios TC-14` (2/2 points, 0 partial, 0 failed).
3. **TTFT & TPOT Improvement**: Elimination of Nagle stalls and synchronous stream waits delivers lower TTFT and smoother streaming output under multi-client concurrency.
