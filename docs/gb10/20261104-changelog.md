# GB10 changelog 2026-11-04

Scope: Qwen4Exp (`Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix`) end-to-end single-stream and
multi-concurrency (`c=1, 2, 4`) prefill and decode optimizations, N-gram volume asynchronous prefetch
and multi-row copy speculation, narrow/wide routed-expert MoE staging and quantization, NVFP4
direct-register unpack and router fast path, tall LM-head projection up to 32 columns, fused hyper-connection
branch normalization, and GDN recurrent state vectorization on NVIDIA GB10 (`sm_121a`).

## Commit range & accounting

- **Base commit (`origin/main`)**: `68c54356fd490ab329bd1475d48957f886bb7dd1`
- **Previous HEAD**: `3475bfdc5` (`feat: Implement narrow-token TMA linear launch for improved performance based on geometry`)
- **Delta in this milestone**: 29 files changed, 1,050 insertions(+), 639 deletions(-) (excluding `cuda-kernel-toolkit/`, which is an integrated toolkit and intentionally excluded from file-by-file enumeration).
- **Cumulative branch delta vs `origin/main` (excluding `cuda-kernel-toolkit/`)**: 1,209 files changed (including this changelog), ~66,926 insertions(+), 82,311 deletions(-).

## Why (GB10 hardware invariants)

1. **48 SMs and ~246 GB/s sustained LPDDR5X unified memory (`sm_121a`)**:
   - Narrow routed-expert MoE decode kernels (`batch * W <= 8`, 40 CTAs × 4 warps = 160 warps across 10 active experts/token) are memory-bandwidth-bound. Skipping the external `stage_kernel` launch (`unstaged_direct`, saving 80 launches/round) while retaining per-CTA `cp.async` shared-memory activation staging (`GateUpShared = 1.28 KiB`, `DownShared = 2.56 KiB`) prevents a 10x L2 read amplification (`+3.4 ms/round` penalty when 160 warps pull activations uncached from L2).
   - Because all 512 routed experts reside directly in unified LPDDR5X (`device_resident() == true`) without PCIe host-staging stalls, and `WideExpert` MMA handles `batch * W > 8`, the marginal verify column cost on GB10 is `~0.11` of a single-token step at `batch = 2` and `~0.20` at `batch = 1, 4` (versus upstream's PCIe-offloaded `0.25–0.30` assumption). Lowering `kWidthCost` (`0.22`, floor `0.14`), `kPairWidthCost` (`0.12`), and `kMultiRowWidthCost` (`0.22`) unlocks wider MTP verification (`W=4..6`) and multi-row N-gram copy speculation (`W <= 16` at `c=2`, `W <= 8` at `c=4`).
2. **Single-pass LM head bandwidth amortization up to `W = 32`**:
   - The `248,320 × 2,560` Q6_K LM head weighs `675 MiB` (`~2.7 ms` per full LPDDR5X read). Extending `projection_tall_kernel` from `kTallMaxColumns = 24` to `32` (`kTallPassLaneChunks = 3`, `48 KiB` dynamic shared memory, 2 CTAs/SM on GB10's 100 KiB/SM budget, 0 stack spills) allows 31-draft N-gram copy rounds (`W = 32` verify columns) to stream the LM head once instead of four `kCols = 8` passes (`2.7 GiB` saved per 32-column round).
3. **Grace 20-core big.LITTLE ARMv9.2 & NVMe `O_DIRECT` AIO concurrency**:
   - Prefill suffix indexing (`ingest_prefix` over 2,048 tokens) and multi-block NVMe `O_DIRECT` reads parallelize across the host thread pool (`kParallelBlocks = 64`). Overlapping N-gram volume suffix lookup with the GPU verification forward pass via an asynchronous worker (`NgramAsyncPrefetch`) removes NVMe latency from the critical GPU decode timeline.
4. **Tensor Core 1:1 FP32 accumulation & register-level NVFP4 unpack**:
   - Unpacking four E2M1 nibbles into two `__half2` pairs via register bit-shifts/masks (`unpack_e2m1x4_to_half2x2`) avoids per-byte constant/shared LUT gather instructions in narrow sliced-K MMA loops while preserving exact IEEE FP16 operand bits into `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`.

## Subsystem breakdown

### 1. Core runtime (`src/runtime/`)

| Action | File | Mechanism |
|---|---|---|
| M | `src/runtime/engine/engine_core.h` | Prioritizes decode batch consolidation (`has_ready_decode`) before launching a new prefill round when other requests are mid-flight, preventing staggered concurrent arrivals (`c=2, 4`) from starving ready decode rows behind serial prefill waves. |

### 2. Ops & quantization (`src/ops/`)

| Action | File | Mechanism |
|---|---|---|
| M | `src/ops/offloaded_sparse_moe/cuda/moe_layer.cu` | Restores `GateUpShared` (`1.28 KiB`) and `DownShared` (`2.56 KiB`) `cp.async` shared-memory activation staging in `moe_w4a8_grouped_gemv_Launcher` and `moe_w4a8_down_gemv_Launcher`; skips `stage_kernel` (`unstaged_direct`) when `!use_wide` on device-resident weights; replaces single-thread `quantize_a4_block` with warp-parallel `quantize_a4_block_warp` and 4-bit `c2` packing (`uint8_t` instead of `uint16_t`). |
| M | `src/ops/offloaded_sparse_moe/cuda/wide_expert.cu`, `wide_expert.cuh` | Adds `unstaged_direct` parameter to `WideExpert::forward` to read unpermuted `x` / `gate_up` directly on narrow-only fallback; routes `x.ne[1] <= 64` directly through `quantize_kernel` (1 launch instead of `map_entries_kernel` + `quantize_token_kernel`) so masked multi-row verification columns (`valid_columns != nullptr`) only iterate over live scheduled tiles (`tile < pass_tiles[passes]`). |
| M | `src/ops/projection_fp32/projection_fp32.cu` | Raises `kTallMaxColumns` from `24` to `32` (`kTallPassLaneChunks = 3`, `48 KiB` dynamic shared memory, 2 CTAs/SM, 0 stack spills); routes `columns` in `6..8` through `launch_pass<8>()` (`REG=64, STACK=0`) so `infernix_cuda_stack_check` passes with zero spills. |
| M | `src/ops/linear/nvfp4/nvfp4_codec.cuh`, `nvfp4_a16_sliced_k_mma.cuh` | Adds `unpack_e2m1x4_to_half2x2` bitwise register unpack and applies it in `nvfp4_a16_sliced_k_mma` to eliminate constant/LUT table indexing overhead on narrow token counts. |
| M | `src/ops/linear/nvfp4/shapes/n13952_k2560.cu` | Routes `m <= 16` on the `13952 × 2560` GDN fused projection to the narrow-token TMA kernel (`launch_narrow_tma<32, 64, 2, 2, 0>`) to avoid multi-wave tail underutilization on 48 SMs. |
| M | `src/ops/linear/q8/shapes/n10240_k320.cu`, `n320_k10240.cu`, `n324_k10240.cu` | Adds `__launch_bounds__` and retunes GB10 Q8 stream/pair dispatch thresholds on the hyper-connection / low-rank shapes (`10240 × 320`, `320 × 10240`, `324 × 10240`). |
| M | `src/ops/linear_swiglu/q8/q8_linear_swiglu_plan.cpp`, `q8_linear_swiglu_stream_pair.cu`, `q8_dflash2_linear_swiglu.cu` | Retunes Q8 SwiGLU stream-pair geometry and launch bounds for GB10 48-SM occupancy with zero stack spills. |
| M | `src/ops/hyper_connection/hyper_connection.cu` | Adds `branch_PrepareFused` (`hyper_connection_branch_prepare_fused_kernel`) to fuse the 4-stream hyper-connection weighted sum (`branch_Prepare`) with its subsequent RMSNorm (`branch_norm`) into a single kernel launch (saving 80 launches per forward pass). |
| M | `src/ops/linear_attention/gated_delta_net/recurrent.cuh` | Vectorizes GDN recurrent state load/update/store in `gated_delta_net_recurrent_step_kernel` using `float4` (`kVecWidth = 4`) across the 128-element state dimension, cutting instruction and memory transaction counts by 4x during decode and draft verification. |
| M | `src/ops/kernel/top_logprobs.cuh`, `src/ops/launcher/top_logprobs.cu` | Increases `kTopLogprobsSerialBlock` from `256` to `1024` (32 warps/CTA) to accelerate per-column top-K/top-P vocabulary reduction over `V = 248,320`. |

### 3. Model & speculation (`src/models/qwen4_exp/`)

| Action | File | Mechanism |
|---|---|---|
| M | `src/models/qwen4_exp/program/program_impl.h` | Unlocks multi-row N-gram copy speculation (`kPairNgramMaxDrafts = 15` for `batch == 2`, `kMultiRowNgramMaxDrafts = 7` for `batch >= 3`); retunes GB10 MTP width costs (`kWidthCost = 0.22` with adaptive floor `0.14`, `kPairWidthCost = 0.12`, `kMultiRowWidthCost = 0.22`); adds `NgramAsyncPrefetch` background worker to overlap N-gram volume suffix lookup with GPU verification; adds `select_multirow_ngram_drafts` and adaptive N-gram cooldown gating (`ngram_miss_streak`, `ngram_cooldown`). |
| M | `src/models/qwen4_exp/program/graph_set.cpp` | Expands startup CUDA graph pre-capture (`kMaxCapturedMultiRowVerifyCols = 16`) to cover multi-row N-gram verify widths (`W = 8, 16` at `batch = 2` and `W = 8` at `batch = 3..4`) so zero runtime graph captures occur during multi-concurrency serving. |
| M | `src/models/qwen4_exp/program/ngram_volume.h`, `ngram_volume.cpp` | Parallelizes `ingest_prefix` suffix-table construction across `pool_->run` for prompts `>= 256` tokens and tunes `kParallelBlocks = 64` for multi-block `O_DIRECT` AIO reads. |
| M | `src/models/qwen4_exp/execution/forward.h`, `forward.cpp` | Calls `hyper_connection::branch_PrepareFused` across all 80 layer steps (`run_layer_attention_half` and `run_layer_moe_half`); gathers `logit_columns` before `final_mixer` in `Forward::finish` when `logit_columns.size() < hidden.ne[1]` so prefill (`T = 2048`) normalizes 1 column instead of 2,048. |
| M | `src/models/qwen4_exp/program/fast_paths.cpp` | Qualifies `FastPath::RouterMma` for `DenseKernels::Nvfp4` (`W4Dense`) alongside `DenseKernels::Default`. |
| M | `src/models/qwen4_exp/program/prefix/state_image.cpp` | Uses `cudaMemcpyAsync` on the caller stream instead of synchronous `cudaMemcpy` when capturing/restoring recurrent prefix state images. |

### 4. Tools & tests (`tools/`, `tests/`)

| Action | File | Mechanism |
|---|---|---|
| M | `tools/run_server.sh` | Updates default server launch flags for GB10 N-gram volume and MTP configuration. |
| M | `tests/models/qwen4_exp/test_fast_paths.cpp` | Verifies `FastPath::RouterMma` qualification on `DenseKernels::Nvfp4`. |
| M | `tests/ops/linear/test_projection_fp32.cpp` | Extends `projection_fp32` oracle coverage up to 32 columns (`kTallMaxColumns = 32`). |

## Verification & benchmark summary (`infernix_bench`, `pp2048+tg128`)

- **Unit & oracle tests**: 31/31 affected suites pass (`ctest -R "moe|projection_fp32|top_logprobs|fast_paths|cuda_stack_check"`), including `infernix_cuda_stack_check` (0 stack spills across 7,492 kernels).
- **`infernix_bench` (`Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix`, `--kv-dtype int8 --spec mtp --draft-tokens auto --ngram-draft-tokens 31 --ngram-min-match 4 --lm-head-draft`)**:
  - **`c=1`**: Prefill `2,743.5 -> 2,916.2 tok/s` (**`+6.3%`**); Decode `116.3 -> 227.8 tok/s` (**`+95.9%`**, rounds `16 -> 7`).
  - **`c=2`**: Prefill `2,763.0 -> 2,899.9 tok/s` (**`+5.0%`**); Decode `168.4 -> 280.6 tok/s` aggregate (`84.2 -> 140.3 tok/s/row`, **`+66.6%`**, rounds `48 -> 18`).
  - **`c=4`**: Prefill `2,760.2 -> 2,917.0 tok/s` (**`+5.7%`**); Decode `195.7 -> 285.7 tok/s` aggregate (`48.9 -> 71.4 tok/s/row`, **`+46.0%`**, rounds `110 -> 63`).
  - **Pure MTP (`--ngram-draft-tokens 0`, 0% N-gram copy hits)**: `92.8 tok/s` at `c=1`, `130.7 tok/s` aggregate (`65.4 tok/s/row`) at `c=2`.

## Rebase conflict resolution policy

- **MoE (`moe_layer.cu`, `wide_expert.cu`)**: Preserve `unstaged_direct` (skipping `stage_kernel` on narrow device-resident calls) while keeping `GateUpShared` / `DownShared` `cp.async` shared-memory staging and `quantize_a4_block_warp`. Preserve routing `x.ne[1] <= 64` through `quantize_kernel` in `WideExpert::forward` for masked multi-row safety.
- **Model & speculation (`program_impl.h`, `graph_set.cpp`, `forward.cpp`, `fast_paths.cpp`)**: Preserve GB10 MTP width costs (`0.22 / 0.12 / 0.22`), multi-row N-gram copy speculation (`kPairNgramMaxDrafts = 15`, `kMultiRowNgramMaxDrafts = 7`), `NgramAsyncPrefetch`, pre-`final_mixer` column gather in `Forward::finish`, and `FastPath::RouterMma` on `DenseKernels::Nvfp4`.
- **Ops (`projection_fp32.cu`, `hyper_connection.cu`, `recurrent.cuh`, `nvfp4_codec.cuh`)**: Preserve `kTallMaxColumns = 32` (`kTallPassLaneChunks = 3`), `branch_PrepareFused`, `float4` GDN recurrent vectorization, and bitwise register E2M1 unpack.
