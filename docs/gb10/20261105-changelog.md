# GB10 changelog 2026-11-05

Scope: Qwen4Exp (`Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix`) end-to-end `infernix-serve` and
`llama-benchy` (`pp2048 tg128 @ d0, d4096, d8192` across `c = 1, 2, 4`) prefill and decode optimizations:
unsplit device-resident MoE prefill chunks and `rows_behind` PLE overlap, OpenAI `"cache_prompt"` protocol
support to eliminate structural tap splitting and wasted host snapshot copies on `--no-cache` workloads,
concurrent batch prefill completion before decode interleaving, optimal N-gram copy prefix truncation
(`best_copy_length`) and tile-aware MTP draft length scoring, `-rdc=true` static `.shared` memory bloat
elimination across `infernix_ops`, single-stage tall LM-head projection (`SingleStage` + wide router MMA),
and removal of redundant MTP verification argmax calls on NVIDIA GB10 (`sm_121a`).

## Commit range & accounting

- **Base commit (`origin/main`)**: `68c54356fd490ab329bd1475d48957f886bb7dd1`
- **Previous HEAD**: `fbbc1c763` (`perf(gb10): optimize Qwen4Exp MoE staging, multi-row N-gram/MTP speculation, and 32-col LM head`)
- **Delta in this milestone**: 16 code files changed (+1 changelog file = 17 files total), 226 code insertions(+), 99 code deletions(-) (excluding `cuda-kernel-toolkit/`, which is an integrated toolkit and intentionally excluded from file-by-file enumeration).

## Why (GB10 hardware invariants)

1. **128 GiB unified LPDDR5X UMA (`device_experts == true`) & zero PCIe staging**:
   - Upstream `ProgramImpl::call_cost()` hardcoded `.served_columns = 8U` unconditionally and disabled `rows_behind` prefill PLE overlap unless `expert_service != nullptr`. On GB10, all 512 routed experts reside directly in device-mapped unified LPDDR5X (`options_.device_experts == true`), so prefill has zero host miss-service traffic. Setting `.served_columns = 0U` when `options_.device_experts` is true prevents `EngineCore::split_prefill_call` from halving 2048-token prefill chunks into two sub-calls (`u = 2 -> u = 1`) whenever another request is active, and enabling `rows_behind` overlaps PLE loads during prefill.
2. **Unified LPDDR5X memory bandwidth conservation on `--no-cache` (`"cache_prompt": false`) requests**:
   - Because GB10 shares a single ~246 GB/s LPDDR5X memory pool between CPU and GPU, writing ~500 MB per request of host prefix-cache snapshots and splitting `System+User` prompts at structural system-turn boundaries (`d4096` and `d8192`) directly competes with GPU prefill bandwidth and doubles kernel launch / tail-wave overhead (`u = 2..3` prefill updates per request). Honoring `"cache_prompt": false` on `/v1/chat/completions` by setting `CacheParticipation::Disabled` keeps every request in a single contiguous prefill update (`u = 1`) and eliminates 100% of wasted host snapshot writes and evictions.
3. **4-column Tensor Core verification tiles & optimal N-gram copy prefix truncation**:
   - On GB10 (`sm_121a`), `WideExpert` and `projection_fp32` verify 1–3 MTP draft tokens (`W = 2..4` columns per row) inside the same 4-column hardware tile with ~0.10 marginal cost per column (`tile_cost = 0.10` for $k \le 3$), whereas wider N-gram copy verification (`W = 6..16`) spans multiple tiles (`kWidthCost = 0.18`, `kPairWidthCost = kMultiRowWidthCost = 0.16`).
   - Charging `tile_cost = 0.10` for $k \le 3$ inside `draft_length_at` prevents stochastic `temperature = 1.0` rejections at position 2 from trapping `draft_length_at` at `K = 2` (which otherwise starves `lane.mtp_accept[2]` of observations).
   - Furthermore, when `--ngram-draft-tokens 15` is configured, a 4–7 token prompt match (`bucket = 0`, $p = 0.75$) previously proposed all 15 tokens; because $0.75^j$ decays to near zero past $j = 7$ while the denominator $1 + 0.10 \times 15 = 2.50$ grows linearly, `copy_score(15, 0.75) = 1.584 < chain_score (1.775)`, causing `c = 1` and `c = 2` to reject 100% of Bucket-0 N-gram matches. Truncating N-gram proposals via `best_copy_length(copies, copy_accept, copy_cost)` to the prefix $k \in [1, \text{copies}]$ maximizing $\frac{E[\text{copy}(k)]}{1 + c_{\text{copy}} k}$ (`k = 5` for Bucket 0, `k = 9` for Bucket 1, `k = 12` for Bucket 2, `k = 15` for Bucket 3) restores Bucket-0 N-gram speculation at `c = 1, 2`.
4. **Relocatable device code (`-rdc=true`) `.weak .shared` deduplication & 48-SM CTA occupancy**:
   - Because `infernix_ops` compiles with `-rdc=true`, `nvlink` merges any `static __shared__` buffers from inline header templates (`cub::BlockReduce` in `argmax.cuh`, `target_logprobs.cuh`, `token_constraint.cuh`, `top_logprobs.cuh`) into every translation unit's static `.shared` reservation (`50,240..66,809` B/CTA), throttling occupancy across unrelated kernels. Moving those reductions to dynamic shared memory (`extern __shared__ __align__(16) char`) drops static shared memory to `1,132..17,493` B/CTA.
   - Switching `projection_tall_kernel` from `MultiStage` (`32,768` B shared, 3 CTAs/SM) to `SingleStage` (`16,384` B shared, 6 CTAs/SM on GB10's 100 KiB/SM budget) doubles CTA residency on the `248,320 × 2,560` Q6_K LM head.

## Subsystem breakdown

### 1. Core runtime & serving (`src/runtime/`, `src/serve/`)

| Action | File | Mechanism |
|---|---|---|
| M | `src/runtime/engine/engine_core.h` | Updates `has_tail_prefill_ready()` so concurrent batch arrivals (`c = 2, 4`) complete prefill before starting decode rounds when `scheduler_.decode_share().share == 0.0`, preventing later requests' TTFT from stalling behind interleaved single-lane decode rounds. |
| M | `src/serve/request.h`, `src/serve/openai_chat_request.cpp`, `src/serve/generation_service.cpp` | Parses optional boolean `"cache_prompt"` on OpenAI `/v1/chat/completions` requests (`llama-benchy --no-cache`) and sets `EngineRequest::cache_participation = CacheParticipation::Disabled` when `"cache_prompt": false`, eliminating structural system-turn tap splitting (`u = 2 -> u = 1`) and ~500 MB/request of unused host prefix-cache snapshot copies. |

### 2. Model & speculative decoding (`src/models/qwen4_exp/`)

| Action | File | Mechanism |
|---|---|---|
| M | `src/models/qwen4_exp/program/mtp_gating.h` | Adds `best_copy_length(copies, copy_accept, copy_cost)` to truncate N-gram copy proposals to the expected-speedup-maximizing prefix length $k$; charges `tile_cost = min(width_cost, 0.10f)` for $k \le 3$ inside `draft_length_at` so 1–3 MTP drafts share the 4-column hardware tile cost without getting trapped at `K = 2` under sampled decode. |
| M | `src/models/qwen4_exp/program/program_impl.h` | Sets `.served_columns = options_.device_experts ? 0U : 8U` in `ProgramImpl::call_cost()` to stop `split_prefill_call` from splitting device-resident MoE prefill chunks; enables `rows_behind` prefill PLE overlap when `options_.device_experts == true`; applies `best_copy_length` to single-row and multi-row N-gram proposals under `--draft-tokens auto`; tunes GB10 width costs (`kWidthCost = 0.18`, `kPairWidthCost = 0.16`, `kMultiRowWidthCost = 0.16`, `kPairMaxDrafts = 3`). |
| M | `src/models/qwen4_exp/execution/forward.cpp` | Removes redundant `ops::argmax` launch in `Forward::step_mtp` when `draft_out != nullptr` is already populated by `sample_mtp_draft_column` or greedy selection, and uses `Scratch::top_logprobs_bytes(1, 1)` for single-token argmax workspace. |

### 3. Ops & kernels (`src/ops/`)

| Action | File | Mechanism |
|---|---|---|
| M | `src/ops/kernel/argmax.cuh`, `src/ops/kernel/target_logprobs.cuh`, `src/ops/kernel/token_constraint.cuh`, `src/ops/kernel/top_logprobs.cuh`, `src/ops/launcher/top_logprobs.cu` | Replaces `static __shared__ typename BlockReduce::TempStorage` in header-defined kernels with dynamic shared memory (`extern __shared__ __align__(16) char`), reducing `-rdc=true` static `.shared` reservations across `infernix_ops` from `50–66 KiB` to `1.1–17.5 KiB`; adds `top_logprobs_one_kernel` fast path for `k == 1` greedy selection. |
| M | `src/ops/projection_fp32/projection_fp32.cu`, `src/ops/projection_fp32/projection_fp32_mma.h`, `src/ops/projection_fp32/projection_fp32_mma.cu` | Switches `projection_tall_kernel` (`kCols = 8, 12, 16, 24, 32`) from `MultiStage` (`32 KiB` shared, 3 CTAs/SM) to `SingleStage` (`16 KiB` shared, 6 CTAs/SM); dispatches `columns = 9..16` router projections (`rows = 512, K = 2560`) through `launch_router_mma_wide<16>` (8 CTAs × 4 warps = 32 warps vs 16 warps). |

### 4. Tools (`tools/`)

| Action | File | Mechanism |
|---|---|---|
| M | `tools/run_server.sh` | Sets default `--ngram-draft-tokens 15` on `infernix-serve` launcher for GB10 single-pass 16-column verify tile alignment. |

## Verification & benchmark summary (`llama-benchy`, `pp2048 tg128 @ d0, d4096, d8192`, `c = 1, 2, 4`)

- **Unit & oracle tests**: 19/19 speculative and stack-ratchet CTest targets pass (`ctest --test-dir build-native -R "speculative|cuda_stack_check"`), with 0 new stack spills.
- **End-to-end `llama-benchy` (`Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix`, `--no-cache --pp 2048 --tg 128 --depth 0 4096 8192 --concurrency 1 2 4 --runs 3`)**:

| Test | `c` | `pp t/s` (Base A → Base B → **Now**) | `pp` Gain (vs A / vs B) | `tg t/s` (Base A → Base B → **Now**) | `tg` Gain (vs A / vs B) | `TTFT ms` (Base A → Base B → **Now**) | `Total ms` (Base A → Base B → **Now**) | Total Time Change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `pp2048 tg128 @ d0` | `c1` | `2,648*` → `2,763*` → **`3,051*`** | **+15.2%** / **+10.4%** | `71.6` → `77.1` → **`81.8`** | **+14.2%** / **+6.1%** | `773` → `741` → **`671`** | `2,546` → `2,388` → **`2,237`** | **-12.1%** / **-6.3%** |
| `pp2048 tg128 @ d0` | `c2` | `2,458` → `2,633` → **`2,885`** | **+17.4%** / **+9.6%** | `90.3` → `106.0` → **`115.3`** | **+27.7%** / **+8.8%** | `1,544` → `1,476` → **`1,299`** | `4,264` → `3,701` → **`3,520`** | **-17.4%** / **-4.9%** |
| `pp2048 tg128 @ d0` | `c4` | `2,516` → `2,716` → **`2,866`** | **+13.9%** / **+5.5%** | `139.4` → `144.7` → **`144.1`** | **+3.4%** / **-0.4%** | `3,019` → `2,821` → **`2,618`** | `6,437` → `6,184` → **`6,172`** | **-4.1%** / **-0.2%** |
| `pp2048 tg128 @ d4096` | `c1` | `3,172*` → `3,316*` → **`3,877*`** | **+22.2%** / **+16.9%** | `61.8` → `79.7` → **`75.0`** | **+21.4%** / **-5.9%** | `1,937` → `1,853` → **`1,585`** | `3,993` → `3,446` → **`3,291`** | **-17.6%** / **-4.5%** |
| `pp2048 tg128 @ d4096` | `c2` | `2,892` → `3,093` → **`3,547`** | **+22.6%** / **+14.7%** | `91.1` → `99.8` → **`107.1`** | **+17.6%** / **+7.3%** | `3,934` → `3,690` → **`3,133`** | `6,588` → `6,101` → **`5,523`** | **-16.2%** / **-9.5%** |
| `pp2048 tg128 @ d4096` | `c4` | `2,937` → `3,125` → **`3,611`** | **+22.9%** / **+15.6%** | `133.0` → `136.1` → **`145.3`** | **+9.2%** / **+6.8%** | `7,676` → `7,202` → **`6,194`** | `11,328` → `10,640` → **`9,717`** | **-14.2%** / **-8.7%** |
| `pp2048 tg128 @ d8192` | `c1` | `3,613*` → `3,722*` → **`4,379*`** | **+21.2%** / **+17.7%** | `69.3` → `71.0` → **`74.7`** | **+7.8%** / **+5.2%** | `2,834` → `2,751` → **`2,338`** | `4,668` → `4,541` → **`4,052`** | **-13.2%** / **-10.8%** |
| `pp2048 tg128 @ d8192` | `c2` | `3,298` → `3,434` → **`3,962`** | **+20.1%** / **+15.4%** | `90.5` → `98.5` → **`113.1`** | **+25.0%** / **+14.8%** | `5,641` → `5,455` → **`4,651`** | `8,309` → `7,879` → **`6,915`** | **-16.8%** / **-12.2%** |
| `pp2048 tg128 @ d8192` | `c4` | `3,325` → `3,453` → **`3,986`** | **+19.9%** / **+15.4%** | `138.3` → `150.2` → **`142.9`** | **+3.3%** / **-4.9%** | `11,241` → `10,841` → **`9,286`** | `14,792` → `13,985` → **`12,868`** | **-13.0%** / **-8.0%** |

## Rebase conflict resolution policy

- **Runtime & serving (`engine_core.h`, `request.h`, `openai_chat_request.cpp`, `generation_service.cpp`)**: Preserve `"cache_prompt"` support (`CacheParticipation::Disabled` when false) and `has_tail_prefill_ready()` prefill-completion scheduling when `decode_share == 0.0`.
- **Model & speculative decoding (`mtp_gating.h`, `program_impl.h`, `forward.cpp`)**: Preserve `.served_columns = options_.device_experts ? 0U : 8U` in `call_cost()`, `rows_behind` prefill overlap on `device_experts`, `best_copy_length` N-gram prefix truncation, and `tile_cost = 0.10` for $k \le 3$ in `draft_length_at`.
- **Ops (`argmax.cuh`, `target_logprobs.cuh`, `token_constraint.cuh`, `top_logprobs.cuh`, `projection_fp32.cu`, `projection_fp32_mma.cu`)**: Preserve dynamic shared memory in header CUB reductions (avoiding `-rdc=true` static `.shared` bloat), `top_logprobs_one_kernel`, `SingleStage` tall LM-head projections, and `launch_router_mma_wide<16>` for `columns = 9..16`.
