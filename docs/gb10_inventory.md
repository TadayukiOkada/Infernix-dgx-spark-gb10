# GB10 Inference Library — Component Inventory (Phase 1 Audit)

Scope: read-only audit of the Infernix engine as it exists today, measured against the
target device. This document establishes *what exists and where the tuning lives* before any
rewrite. It deliberately separates **measured facts** (this session, on this box) from
**hypotheses** (inherited provenance that still needs a GB10 measurement).

Target device (measured, this session):

| Property | Value | How measured |
|---|---|---|
| GPU | NVIDIA GB10 | `nvidia-smi` |
| Compute capability | sm_121a (CC 12.1) | `deviceQuery` / CMake gate |
| SM count | 48 | `cudaDevAttrMultiProcessorCount` |
| Max SM clock | 3003 MHz | `nvidia-smi` |
| Memory | ~121 GiB unified LPDDR5x (`cudaDevAttrIntegrated = 1`) | device attributes |
| Swap | ~15 GiB (`/swap.img`) | `swapon` |
| CPU | Grace, 20 cores (10× Cortex-X925 + 10× Cortex-A725) | `lscpu` |
| L3 | 24 MiB × 2, NUMA 1 node | `lscpu` |
| Driver / CUDA | 580.178.04 / CUDA 13.0 | `nvidia-smi`, toolkit |
| Profilers | `nsys`, `ncu` present | `/usr/local/cuda/bin` |

Hard constraint honoured throughout: **no GPU clock or system-level GPU setting was changed.**

Build gate (measured): `CMakeLists.txt` sets `CMAKE_CUDA_ARCHITECTURES = 121a` and hard-errors on
any other value. `device_info.cu` fallback `kReferenceSmCount = 48` ("GB10 sm_121a, 48 SMs").
The 5090 (sm_120a) target is no longer compiled.

---

## 1. Op / kernel inventory (`src/ops/`)

Per family: quant variants, what it computes, and whether selection is token-banded.

| Family | Variants | Computes | Token-banded route? |
|---|---|---|---|
| `attn_input_proj` | bf16, fp8, nvfp4, q4_q5, q8 | Fused QKV(+gate) projection (5120→6144+1024), incl. fused-RMSNorm variant | yes (`q4_q5_attn_input_plan.cpp` bands 12/32/64/104/128/193) |
| `candidate_selector` | bf16 | Speculative candidate selection path | path plan |
| `context_kv_materialize` | format-agnostic | Materialize KV from prefix/context store into paged cache | no |
| `dynamic_grouped_conv` | bf16, q8 | Grouped depthwise conv prepare/add (GDN conv path) | prepare/add plans |
| `gdn_gating_proj` | bf16 | GDN gating projection w/ SplitK halving policy | SplitK policy (**170-SM-shaped**, see §5) |
| `gdn_input_proj` | fp8, nvfp4, q4_q5, q8 | Gated-delta-net input projection (+ conv snapshot variants) | q4_q5 crossover 11–12 tokens |
| `hyper_connection` | — | Hyper-connection mixing (fused) | no |
| `kernel` | — | ~30 elementwise/epilogue device kernels (rmsnorm, rope, causal_conv1d, silu_and_mul, speculative_round, mtp_pack/round, paged_kv_address, sampling…) | launched via `launcher/` |
| `kv_cache` | fp8_e4m3, int8_g64, nvfp4_group16, q4_lloyd, vq2 (+ Hadamard d256) | KV-codec defs + append kernels | codec-select |
| `launcher` | — | Host-side launchers for every `kernel/` primitive | no |
| `linear` | bf16, fp8(A16+A8), nvfp4(A16+A4), q4, q5, q6, q8 | Core GEMM `[K,T]×[N,K]→[N,T]`; gemv/simt/mma/sliced-k/TMA-MMA families | **yes — 61 per-shape `select_*` tables** (see §4) |
| `linear_add` | bf16, fp8, nvfp4, q4, q5, q8 | GEMM + residual add fused | yes (q8 20-band K4096 / 17-band K6144; fp8 A16/A8 crossover T=40; nvfp4 retuned this session) |
| `linear_attention` | gated_delta_net, kimi_delta_attention (chunked+recurrent) | Linear-attention state recurrence | chunk/recurrent choice |
| `linear_pair` | q8 | Paired dual-GEMM (two weights, one pass) | interval band table |
| `linear_swiglu` | bf16, fp8, nvfp4, q4, q8 | FFN gate/up + SiLU-mul fused epilogues | yes (fp8 A8≥34; nvfp4 retuned this session) |
| `linear_topk` | fp8, q4, q8 (+m64, merge) | Router-logit batched GEMM + top-k | grouped k-split |
| `offloaded_sparse_moe` | W4A4 | Experts outside device memory: narrow/wide routes + CPU W4A4 engine + SSD tier | residency tiers |
| `ple` | — | Post-LN/embedding fused op | no |
| `projection_fp32` | fp32 | FP32 router logits (dot order fixed by H) | route header (tall-vs-wide) |
| `qsa` | — (VQ window) | Quantized sparse attention: page spaces, selection, prompt path | SM-count-driven splits (live) |
| `resident_moe` | — | Device-resident MoE layer | no |
| `rmsnorm_rope` | d128/d256 | Fused RMSNorm+RoPE | launch comment |
| `sparse_moe` | decode/prefill/small_t × Q4/Q5/Q6 | 256-expert top-8 routing | codec-specific crossovers (Q5 T≤17, Q6 T≤11) |
| `softmax_attention` | dense/causal_cache × {bf16,fp8,int8,k8v4,nvfp4,vq2}, sliding_window, mxfp8 tiled | Causal attention w/ quantized KV | tile count vs **live** `multiprocessor_count` |
| `speculative_tree` | — | Verification tree shape validation/compaction | no |
| `rows`, `wrapper`, `common` | — | Row-split MMA helpers, public-op wrappers → plan dispatchers, shared MMA/TMA primitives | wrappers only |

Op families with token-count route/band tables: `linear` (61 shape tables), `linear_add`,
`linear_swiglu`, `attn_input_proj`, `gdn_input_proj`, `sparse_moe`, `linear_pair`,
`softmax_attention` (SM-driven), `projection_fp32`. Counts measured this session:
**61** `select_*` shape tables, **28** `resolve_route`/`_resolve_plan` files, **9** `RouteSpec`/
`routes_are_closed` band arrays.

---

## 2. Dispatch / scheduler mechanism

Flow: public op (`include/infernix/ops/*.h`) → thin wrapper (`src/ops/wrapper/*.cpp`) → per-variant
`*_plan.cpp::resolve_route(...)` / `*_resolve_plan(problem)` returning a schedule id → `execute_plan` /
`launch_*`. `linear.cpp::dispatch_linear` switches on `w.qtype` into per-QType dispatchers.

Selection mechanisms observed:
- **Compile-time token-band tables** — `RouteSpec{first,last,schedule}` arrays closed by
  `static_assert(routes_are_closed(...))` (e.g. q8 linear_add).
- **Explicit crossover thresholds** re-fitted on GB10 (fp8 linear_add `first_a8 = 40`; nvfp4 swiglu
  128→136 TMA gate; nvfp4 add TMA gate 1024→288 — both changed this session).
- **Live SM-count plans** — attention tiles and qsa splits computed against `device_sm_count()` at
  runtime (`causal_query_tiles_underfill_sms`), not hardcoded.

CUDA graph capture/replay: `src/core/decode_graph.{h,cpp}` (`DecodeGraphDefinition::capture`,
`DecodeGraphExecutable::instantiate/update/upload/launch`). Graph families keyed per token-frontier
interval in `src/models/qwen3_5/program/graphs.cpp` (gap-free coverage required). Prefill is EAGER;
decode/verify run under captured graphs, so `resolve_plan` runs once at capture time.

Concurrency / streams: `src/runtime/engine/scheduler.h` lane-based `Scheduler<Request>`
(`kMaximumConcurrency = 8`). Streams in `src/core/device.cu`; expert-stream pipelining in
`qwen4_exp/execution/expert_stream.cpp`; host CPU workers in `core/host_worker_pool.cpp` +
`direct_read_queue.cpp`. Dependency ordering is program-structured (graph replay + Programmatic
Dependent Launch via `core/pdl.cuh`, gated on live `sm_count()`), **not** a generic dependency DAG.

---

## 3. Memory / offload subsystem

Allocators (`src/core/`):
- `arena.{h,cu}` — `DeviceBuffer` (one `cudaMalloc` per long-lived buffer), `DeviceArena` suballocation.
- `WorkspaceArena` (in `include/infernix/types.h`, used by ~20 op headers) carves per-op scratch via
  `allocate_*_workspace_capacity_bytes` (fp8 linear_add, fp8_a8, linear_topk, …).
- `vmm_arena.{h,cu}` — CUDA VMM arena: virtual range reserved, physical chunks mapped/unmapped at the
  top; addresses stay stable under captured graphs. `VmmRange` elastic pool, per-plane chunk mapping.
- `evictable_weight_pool.{h,cu}` — weight arena whose 16 MiB-chunk tail is unmapped/remapped into an
  overlay range (device memory reclaimed with zero new allocations), restored from a pinned host mirror.
- `host_context_arena.{h,cu}` + `host_kv_arena.{h,cpp}` — pinned host arenas for KV blocks / state
  snapshots (hybrid prefix-cache tier).

KV cache: `paged_kv_cache.{h,cpp}` (page size 64, block tables, quantized planes k_codes/k_scales/tags),
`cyclic_kv_cache.{h,cpp}`, storage formats in `paged_kv_storage.h` / `tests/kv_cache_storage.h`; codecs
under `src/ops/kv_cache/*`. Selected via `--kv-dtype bf16|int8|fp8|nvfp4|k8v4|vq2|k4v2`.

Offload model — **explicit copies, NOT unified-memory demand paging.** Measured fact: there is **no**
`cudaMallocManaged`, `cudaMemAdvise`, or `cudaPrefetchHostToDevice` anywhere in `src/` or `include/`.
Movement is explicit `cudaMemcpyAsync` (`arena.cu`, `copy_batch.{h,cpp}`, `paged_kv_cache.cpp`, program
commit/replay transactions), plus:
- **Expert offload tiers** (`offloaded_sparse_moe/`): experts resident in device frames (`frames[e] >= 0`)
  or a pinned host bank; non-resident records staged to a few device staging slots first (scattered
  pinned reads run "<half the copy rate"); further miss tier = **CPU W4A4 compute** (`cpu/miss_service.cpp`,
  heartbeat-word completion); SSD tier from the artifact (`--expert-ram-mib`).
- **Unified-memory awareness is sizing-only**: `vram_budget.h` `VramSnapshot.integrated`
  (`cudaDevAttrIntegrated`) flags GB10-style unified LPDDR5x where allocations share one pool; the
  elastic expert-frame pool is capped on integrated devices.

Buffer categories: weights → `DeviceBuffer`/`EvictableWeightPool`/VMM; activations & op scratch →
`WorkspaceArena` over `DeviceArena`; KV cache → paged device pages + host KV arena; GDN/linear-attention
state → `core/linear_attention_state.*`, `gdn_replay_records.*`; prefix/context snapshots →
`HostContextArena` + device snapshot slots (`--device-snapshot-slots`); vision overlay →
`qwen3_5/execution/vision_overlay.cpp`.

---

## 4. Runtime / inference path

- `src/runtime/engine/` — round-based admission, decode rounds, commit dispositions, speculative stats;
  fairness scheduler (≤8 lanes); `kv_capacity.*`, `generation_budget.h`, model instances.
- `src/runtime/engine/context_cache/` — cost-model-driven context materialization
  (`context_cost.{h,cpp}`, `context_cost_defaults.cpp` — see §5).
- `src/runtime/prefix_cache/` — content-addressed 64-token block hashing, prefix index, tap planner.
- `src/models/qwen3_5/` — Qwen3.5/3.8 family: load (artifact binder/materializer), prefill/decode,
  graph families, request planning, hybrid prefix program, transactions (commit/replay/pause/binding),
  execution (text/vision/attention/ffn/gdn/mtp/draft).
- Speculative decoding: ngram draft source (2²⁰-row hashed cache ~170 MB), MTP draft, verification trees
  (max 32 nodes / 8 paths / depth ≤15).
- `src/models/qwen4_exp/` — experimental MoE-heavy program: memory plan (device/host/SSD expert budgeting),
  expert residency/cache, prefill walk, route trace, VRAM monitor.
- `src/serve/` — HTTP front-end (OpenAI chat/responses, Anthropic messages), generation service, request
  lanes → engine concurrency.
- `src/product/` — CLI/product surface (speculative options, YaRN, prompt input, logging).

---

## 5. RTX 5090 / architecture assumptions — comment vs behavioral

84 files match `5090|sm_120|120a|RTX`. Classified:

**Behavioral — hardcoded 170-SM constants (FP8 A8 TMA split-K wave), runtime-mitigated:**
`Fp8A8TmaSplitKSchedule<…, 170, 4, 8>` appears in ~10 files
(`linear_add/fp8/fp8_linear_add_a8.cu:20-30`, `linear_swiglu/fp8/…_a8.cu:10`,
`linear/fp8/shapes/n5120_k17408.cu`, `n5120_k6144.cu`, `n14336_k5120.cu`, `n16384_k5120.cu`,
`n34816_k5120.cu`, `attn_input_proj/fp8/…_a8.cu:11`, `gdn_input_proj/fp8/…_a8.cu:10-11`).
Mitigation verified this session in `linear/fp8/fp8_a8_tma_mma.cuh::fp8_tma_split_k_plan`: at runtime both
the tuned 170 wave **and** a device-capped wave (`min(kSplitWaveCtas, device_sm_count())`) are evaluated and
the candidate leaving fewer idle SMs in the final resident wave is chosen (ties prefer capped). So 170 is a
candidate, not an absolute assumption — but the *allocation upper bound* still uses `kSplitWaveCtas`, and the
selection logic itself should be re-measured on GB10 rather than assumed adequate.

**Behavioral but already GB10-refitted (no 170 constant, values re-measured on 48 SMs):**
q8 linear_add band tables (5090 bands explicitly flagged as 14–47% regressions); fp8 linear_add `first_a8=40`;
fp8 swiglu A8≥34; nvfp4 swiglu 128→136 and nvfp4 add TMA gate 1024→288 (both this session); attention and qsa
plans use live `multiprocessor_count`.

**Behavioral, needs a GB10 preset (currently falls back to generic 5090-era estimates):**
`context_cost_defaults.cpp` — the **only** compiled machine preset is `hardware_class =
"nvidia-geforce-rtx-5090-sm120"` with measured transfer costs (`batch_ns 48'655`, `ns_per_byte_q32 91'692'315`, …)
and four model-signature prefill costs. On GB10 **no preset matches**, so the engine uses
`generic_context_transfer_cost()` / `generic_context_prefill_cost()` (conservative 5090-era numbers). A GB10-class
preset is absent. Measuring machinery exists (`bench/context_cost/`, `test_context_cost_measure.cpp`) — generating
a real GB10 preset is a concrete Phase-2 experiment.

**Behavioral policy still labeled 170-SM (flag for re-fit):**
`gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.cpp:33,42` — SplitK halving progression annotated as an
"RTX 5090/170-SM performance policy". Candidate for a GB10 re-fit.

**Provenance comments only (no behavioral effect) — safe to relabel to GB10 later:**
Most `linear/{q4,q5,bf16}/shapes/*.cu` headers ("Selected with complete-Op cold CUDA Graph measurements on
RTX 5090"); `common/mma.cuh`, `bf16_vector.cuh`, `mxfp8_tiled_mma.cuh`, `vq2/grouped.cuh` (smem limits identical
on 120a/121a → benign); `launcher/rope.cu`, `rmsnorm_rope/launch.cu`, `gdn_gating_proj_kernels.cu` qualification
notes; `sparse_moe/prefill` provenance; `fp8_a16_codec.cuh` PTX-availability note (120a→121a compatible);
`vram_budget.h` budget example (comment).

**Watch-list (provenance comment attached to band *values* that may still be 5090-fitted):**
Some `linear/bf16/shapes/*.cu` vision-shape routes documented as derived from the 5090 sweep without a GB10
re-fit note. These need a verification pass (they are among the 61 `select_*` tables in §4).

---

## 6. Benchmark + test infrastructure

`bench/ops/` — **57** per-op benches built by `bench/ops/benchmarks.cmake` via `infernix_add_op_bench(...)`.
Common harness `bench/ops/infernix_bench_common.h`; fixtures `bench/common/fixture_data.cuh`; helper headers
`direct_bf16_weight.cuh`, `quantized_weight.cuh`.

`--variants` mode (present on nvfp4/fp8 add+swiglu benches): times **every internal A16/A4 schedule candidate
independently per token count** — this is how production band tables get fitted, bypassing policy gates.
Standard flags: `--warmup/--repeat/--t-sweep/--policy a16|a4`, `--csv-out PATH` (~20 benches). Timing method:
L2 flushed before each warmup and timed sample; `measure_cold_launch_prepared(...)` / `measure_cold_launch(...)`.

Other bench dirs: `bench/inference/` (`infernix_bench.cpp` end-to-end), `bench/ab/` (`ab_runner.py`,
`ab_workloads.py`), `bench/agentic_ab/`, `bench/gb10/` + `gb10_baseline/` + `gb10_prof/` (GB10 matrix/profiling
sweeps; baseline in `bench/gb10/matrix_table.txt`), `bench/context_cost/`, `bench/decode_quality/`,
`bench/models/`, `bench/fixtures/`.

`tests/` — **37** `test_*.cpp` unit/integration tests plus subdirs `tests/{ops,models,runtime,artifact,convert}`
and Python (`test_bench_matrix.py`, `test_serve_corpus.py`, `test_expert_cache_host_tier.py`, …).

Correctness validation: bit-exactness asserted structurally (route swaps "bit-identical", CPU-vs-GPU W4A4 experts
identical bits, `bf16_flash_next_sweep.cu` route-vs-reference bit invariance); `test_perplexity_evaluation.cpp` +
`eval/` (Python, `run_qwen3_8_27b_nvfp4_*` GPQA/reasoning) validate quality; corpus serve tests.
Numerical caveat carried forward: same-K-order MMA tiles are bit-identical, but split-K swaps or route changes alter
accumulation order → perplexity must be re-run after any retune.

---

## 7. Retune status & initial bottleneck hypotheses

Measured & validated this session (nvfp4): `linear_add` A4 ladder + TMA gate 1024→288; `linear_swiglu` A4 ladder
confirmed optimal + TMA gate 256→136 + small-T A16 band ≤4→==1. Production dispatch matches measured winners except
run-to-run noise.

Derived GB10 winners (median µs) captured in `/tmp/gb10_more/` (add K6144/K17408, swiglu variants/fine). Key finding:
resident M128N128 falls off a cliff past its wave capacity (swiglu 914 µs at T=136 vs 467 µs at T=128) — hence the
tight TMA gates.

Priority hypotheses to test next (not yet proven):
1. FP8 A8 `WaveCtas=170` selection logic could be simplified/re-measured on 48 SMs (currently runtime-mitigated, not re-derived).
2. Missing GB10 context-cost preset → engine runs hybrid-prefix/materialization decisions on 5090-era estimates.
3. `gdn_gating_proj` bf16 SplitK progression is 170-SM-shaped.
4. 61 `linear/*/shapes/*.cu` `select_*` tables (bf16/q4/q5/q6/q8/fp8-a16) are largely unfitted for GB10.
5. q5 bands derivable from existing `/tmp/gb10_more/q5cand_*.csv` but not yet applied.

Ordered experiment plan (Phase 2 onward, honoring "audit first"): build the formal benchmark+correctness harness
(`docs/benchmark_methodology.md`) → derive q5 bands + extend `--variants` candidate harness to linear_pair,
q8/q4 swiglu, sparse_moe, gdn_gating_proj → retune the 61 shape tables → replace the behavioral 5090 assumptions
with GB10-measured values → re-run perplexity + end-to-end `infernix_bench` vs baseline.
