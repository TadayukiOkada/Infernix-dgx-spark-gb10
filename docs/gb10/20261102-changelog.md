# GB10 changelog 2026-11-02

Scope: concurrency fixes for Qwen4Exp MTP rounds and the prefill items of
`docs/tasks/20261102-prefill-ceiling-tasks.md` (T1 wide x quantization, T5 prompt attention,
T6 GDN chunked prefill, T7 extension: per-T route sweeps, the a4 live-route correction and the
QSA selection pipeline fix). `cuda-kernel-toolkit/` is an integrated toolkit and is
not enumerated.

## Why (GB10 invariants)

- 48 SMs and ~246 GB/s sustained LPDDR5X: narrow routed-expert decode kernels already stream at ~86% of
  the read ceiling, so concurrency gains come from fewer verified columns and fewer runtime graph captures.
- Wide-path x quantization is latency-bound at 156 registers; three CTAs/SM (74 registers) fit the
  65,536-register file.
- GDN chunk recurrence CTA count is value_heads x (128 / tile), so the two-wave-fit tile rule is a
  48-SM constant: tile 64 fills two waves at Hv=48 (measured 13-19% faster than tile 16); the earlier
  ceil(Hv/tile) scoring ignored the state-dimension slices and always chose 16.
- QSA prompt attention is latency-bound at 2 warps/scheduler with a 26% tensor pipe on GB10: Stages=2
  (one 100 KB-smem CTA/SM) regresses +57% against the 100 KB smem/SM budget, so the two-CTAs-per-SM
  gate is a hardware invariant, not a tunable.

## Changes

| Action | File | Mechanism |
|---|---|---|
| M | `src/models/qwen4_exp/program/program_impl.h` | `mtp_confident_` truncation applies to `batch == 1` only; DFlash2 vs MTP `pair_max` choice; multi-row drafts honour `kMultiRowDrafts = 0`; separate `kMultiRowWidthCost` (0.35), `kPairWidthCost` 0.25 |
| M | `src/models/qwen4_exp/program/graph_set.cpp` | startup capture of multi-row (>= 3) verify/commit/draft graphs so no graph is captured while decoding |
| M | `src/ops/offloaded_sparse_moe/cuda/wide_expert.cu`, `wide_expert.h` | `map_entries_kernel` + `quantize_token_kernel` replace the entry-major wide x quantization when the dispatch has no shared job; workspace gains `entry_map` ([entries][2]) |
| M | `src/ops/qsa/qsa_prompt.cu` | skip the per-tile 128-FMAL o-rescale when `__all_sync` shows every row max held (alpha == 1 exactly, `exp2_approx(0)` = 1, bits unchanged) |
| M | `src/ops/linear_attention/gated_delta_net/chunked/recurrence.cu` | `value_tile` picks the widest stripe filling two 48-SM waves (`value_heads * (kStateDim / tile) >= 2 * sms`) instead of scoring ceil(value_heads / tile) waves, which always returned tile 16 |
| M | `src/ops/qsa/qsa_select.cu` | wide-row score kernel key staging as uint4 pair rows: `store_tile` stages a thread's two blocks (g, g + 8) as consecutive rows of one uint4 lane, so each leaf's key read is a single LDS.128 (192 -> 160 LDS per thread per key tile); arithmetic and bit contract unchanged |
| M | `bench/ops/nvfp4_route_sweep.cu` | five 2-term pair candidates (r128_s3_m1 promotions ws/all/none + baked-K 2560/6144) so the sweep covers the route the model actually runs |

## Results (summary)

- `tg512`, c = 1 / 2 / 4, int8 KV: per-stream 77.7 / 40.7 / 24.9 tok/s (before: 83.8 / 37.7 / 23.6; the c = 1 row is
  run-to-run variation, no change was made to that path). `pp8192+tg512` aggregate 112 / 142 / 175 tok/s.
- Wide x quantization: kernel -50%, `pp8192` prefill +1.9% (paired A/B, 3 rounds, all positive); outputs bit-identical.
- QSA prompt (T5): roofline 4.96 ms/layer vs 15.15 ms (33%); rescale-skip shipped (bench 15.31 -> 15.20 ms,
  bit-identical, oracle pass); forced Stages=2 measured +57% — gate kept; closing the 3x needs a
  warp-specialized TMA/mbarrier redesign.
- GDN (T6): recurrence 1.90 -> 1.53 ms at T=8192 (tile 64 default), conv at its byte floor, prepare at
  81% of its DRAM floor. T5 + T6 together: `pp8192` paired A/B +1.71% mean (conservative median pair
  +0.57%; post-removal confirmation pair vs the shipped binary +0.44% / +0.73%), acceptance unchanged.
  Oracle tests (`qsa`, `qsa_select`, `gated_delta_net` fp16acc profiles)
  and the stack ratchet pass; `chunk_recurrence_kernel<64, 1, float>` REG=128 STACK=0 LOCAL=0.
- a4 T=32768 correction: the earlier "ladder 2.9-3.4% ahead" sweep result compared 1-term arms against a
  production arm the model does not run — nsys demangled names show every dense NVFP4 GEMM at prefill on the
  2-term baked-K pair TMA (~1018 ms of the 8321 ms round), which the extended pair ladder confirms is its own
  argmin (promotions within ±1%; baked-K 2.5-4.3% ahead of runtime-K and already live). Nothing adoptable.
- QSA selection pipeline: score_wide issue-bound at 33% occupancy (2 CTAs/SM by REG 108 and 42.2 KiB smem —
  a GB10 invariant here, every wider staging variant drops to 1 CTA), select_global latency-bound
  (long_sb 12.4 + barrier 7.2 of 33.1 cyc/instr). The uint4 pair-row key staging shipped at select_us
  −0.94% (bf16) / −0.68% (int8) at 32768 columns (paired vs HEAD-built binary, bit-exact oracle pass,
  REG 107 STACK 0). Rejected measured: depth-2 key prefetch +7%, selection slices 2→4 +6%,
  div-free q staging neutral; kWalkBatch=8 rejected pre-measure (REG 63→~79 halves CTAs/SM).
  E2E gate (kv-matrix nvfp4/int8/mtp, pp8192) 3710.3 / 43.6 tok/s, acceptance 0.950 — unchanged.

## Rebase conflict policy

Keep the `batch == 1` truncation and multi-row graph capture over upstream's multi-row cut; keep
`quantize_token_kernel` unless upstream changes the wide x plane layout (`kXRowBytes`, `plane_block`).
Keep the two-wave `value_tile` rule (GB10 48-SM measurement) over upstream's ceil(Hv/tile) scoring, and
keep the QSA rescale-skip guard (bit-exact under the exact int8-widen-to-FP16 MMA semantics).
