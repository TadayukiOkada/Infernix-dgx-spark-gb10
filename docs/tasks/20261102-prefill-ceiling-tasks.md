# Prefill ceiling tasks (2026-11-02)

Deliverable: find and remove prefill time on GB10 that is not explained by a memory or tensor-pipe
floor, on the active `Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix` artifact.

## Baseline (fixed; do not change after seeing results)

- Workload: `infernix_bench -p 8192 --concurrency 1 --kv-dtype int8 --spec mtp --draft-tokens auto
  --lm-head-draft`, `INFERNIX_EXPERTS_ON_GPU=1`, n-gram volume, `bench/fixtures/bench_corpus.ids`,
  `-r 2 --warmup 1` (`run_kv_matrix_bench.sh` shape `pp8192`). Prefill 3,625 / 3,588 / 3,586 t/s
  at c = 1 / 2 / 4 (2026-11-02).
- Trace: `nsys profile --cuda-graph-trace=node` of `pp8192+tg128`, c = 1. Prefill window 2,220 ms,
  GPU busy 2,198 ms (99%), so host gaps are not a factor.
- Ceilings: sustained read 241-246 GB/s (`ckt env`); NVFP4 tensor peak 440 TFLOP/s (bench
  profile; the older audit used 501); BF16 F32-acc ~97.
- Acceptance for any change: output bit-exact where the Op is exact (existing op tests), the
  affected kernel faster in its own bench, and end-to-end `pp8192` prefill t/s measured against
  the baseline above. Adverse results stay recorded in the table below.

## Trace breakdown (ms of the 2,198 ms prefill)

| Kernel group | ms | % | Floor estimate | Task |
|---|---:|---:|---|---|
| Wide expert GEMM `grouped_kernel` | 457 | 20.8 | 5.8 ms/layer weight stream (1.416 GB) vs 9.7 actual | T1 |
| HC Q8 projections (10240x320, 324x10240) | ~256 | 11.6 | ~0.7 ms/call IO vs 1.47 / 1.16 | T2 |
| Elementwise (`norm`, `quantize`, `inject`, `combine_inject`, `rmsnorm`) | ~516 | 23 | per-kernel bytes | T3 |
| Router `projection_wide_kernel` | 139 | 6.3 | FP32 GEMM 21 GFLOP | T4 |
| `qsa_prompt_kernel` | 182 | 8.3 | not yet derived | T5 |
| GDN chunk recurrence / conv / prepare | ~200 | 9 | not yet derived | T6 |
| NVFP4 dense (`a4` pair GEMM + quantize) | 295 | 13.4 | at ceiling (GEMM ~84% of peak, quantize 98% of BW) | closed |

## Tasks (executed one by one; update Status/Result when done)

| ID | Task | Status | Result |
|---|---|---|---|
| T0 | Resolve the 88-96% DRAM claim for the wide expert path against the ~59% measured here | done | Claim used 1.71-2.06 GB per layer; the exact record is 3*2560*640/256 units * 144 B = 2,764,800 B, x512 = 1.4155 GB. Roadmap's own timings (4.83 + 2.99 ms at T=4096) then give 181 GB/s = 73% of 247.6, not 88-96%. In-model T=8192: 96 launches avg 4.86 ms (min 3.80 ms) = 9.7 ms/layer = 146 GB/s (59%). |
| T1 | Wide expert GEMM below its weight-stream floor: bench T = 2048 / 4096 / 8192, NCU on synthetic weights, fix what the evidence shows | partial | Clean per-layer `moe_experts` at T = 2048 / 4096 / 8192: 7.9 / 9.2 / 12.8 ms (floor 5.75 ms weight stream). `WarpTile64` at 64 columns/expert: no change (reverted). Wide x quantization made token-major and low-register: kernel 2040 -> 1000 us at T=8192, call 12.76 -> 11.2-11.7 ms, outputs bit-identical. GEMM kernels (`grouped_kernel` ~9.7 ms/layer at T=8192) still open. |
| T3a | `quantize_kernel` in the wide path (part of T3) | done | See log. |
| T7 | Prefill chunk size (4096 / 8192 / 16384) on `pp8192`, MoE weight re-stream per chunk | done | `pp8192`: chunk 2048 / 4096 / 8192 / 16384 = 2710 / 3284 / 3612 / 3625 t/s. Fit: ~0.264 s fixed per chunk (= 68 GB expert weights at 246 GB/s) + ~0.24 ms per token. `pp32768`: chunk 8192 / 16384 / 32768 = 3556 / 3747 / 3839 t/s (+5.4% / +8.0% over 8192). Documented default for this model is already 16384; raising it is a latency trade-off (decode stalls between chunks), not changed. |
| T8 | In-model vs isolated kernel clocks (`nvidia-smi` during prefill; same shape isolated vs traced) | done | `pp32768` prefill, 0.5 s samples: SM 2405-2476 MHz (median 2463), power 50-85 W (median 84), 60-74 C, throttle reasons 0x0 in all 49 busy samples. No clock or power throttling; in-model vs isolated differences are not clocks. |
| T3 | `norm_kernel` against its byte floor (the wide-path `quantize_kernel` is T3a) | done, no change | `norm_kernel` (grid 4 x 8192, 1.37 ms) reads and writes ~336 MB at T=8192: 245 GB/s, at the sustained ceiling. |
| T2 | HC Q8 projections at T = 8192 against IO floor | done, no change | Isolated T=8192: 10240x320 1244 us (141 GB/s of I/O), 324x10240 1036 us (171 GB/s). Seven MMA launches tried for each at T = 2048 / 4096 / 8192: the current choice is fastest in all six cases (next best +4% to +9% for up at 8192, +9% for down at 8192). Temporary override reverted. |
| T4 | Router projection at T = 8192 | measured, not changed | Exact FP32 SIMT `projection_wide_kernel`: 2.9 ms per call (139 ms per `pp8192` prefill), ~7 TFLOP/s. Its accumulation order is a contract shared with the narrow mapping (same bits), so it cannot be retiled freely. The registered opt-in fast path `--fast-path router-mma` (BF16 tensor cores, routing then depends on call start) measured 3716 vs 3642 t/s (+2.0%, one unpaired pair); it needs model-quality qualification and is not enabled by default. |
| T5 | Prompt attention against a roofline | measured, micro-fix shipped | Roofline 4.96 ms/layer (14.69M attended pairs/column-set, 481 GFLOP issued incl. the 25% 16-row MMA padding, 97 TF/s) vs 15.15 ms measured = 33%. NCU: latency-bound (2 warps/scheduler, 66% no-eligible, tensor 26.2%, L2 hit 98.7% so cold-L2 is not it). Forced Stages=2 regressed +57% — the two-CTA gate is correct. Shipped the warp-uniform o-rescale skip (bit-identical; bench 15.31 -> 15.20 ms). Remaining lever: a warp-specialized TMA/mbarrier redesign, not a schedule retune. |
| T6 | GDN prefill kernels against a floor | shipped | `value_tile` picked tile 16 for every shape (wave formula used ceil(Hv/tile), not Hv x kStateDim/tile); fixed to the widest stripe filling two SM waves: recurrence 1.90 -> 1.53 ms at T=8192 (-19%), 122.6 -> 106.2 us at T=512 (-13%). conv is at its byte floor (1.39 vs 1.37 ms); prepare is at 81% of its ~0.78 ms DRAM floor (follow-up: share Q/K per group). Recurrence still ~3x its ~0.5 ms FP16-MMA floor. E2E paired `pp8192` +1.71% mean (median pair +0.57%) for T5+T6 together. |

Order is by evidence size and cost of the check, not numeric ID.

## Log

(append dated results here per task: command, target, workload, summarized result, adverse findings)

### 2026-11-02 T0

- Source: `docs/gb10_roadmap.md` Gap 6B (lines 364, 390, 481). Byte count inflated by 21-45% versus `nvfp4_record_bytes(kFlashNextShape)`. The roadmap wording "88-96% of DRAM read ceiling" should be corrected when T1 lands.
- Measurement note: GPU benches must not overlap other runs; a concurrent `run_kv_matrix_bench.sh` at 06:43-06:46 invalidated one wide-bench run (discarded).

### 2026-11-02 T1 / T3a: wide-path x quantization

- Cause: `quantize_kernel` read each token row once per routed entry (420 MB read for 118 MB written at
  T=8192) but NCU showed it latency-bound, not bandwidth-bound: 156 registers, ~2 warps per scheduler,
  27% issue slots, ~42 instructions per element. Token-major alone made it slower (2190 us); the gain
  came from processing one 16-element block per lane at a time (74 registers, `__launch_bounds__(256, 3)`,
  no stack) so three CTAs fit per SM.
- Change: `map_entries_kernel` + `quantize_token_kernel` in `wide_expert.cu` (workspace gains `entry_map`);
  used when the dispatch has no shared job, otherwise the entry-major kernel runs as before.
- Kernel: 2040 -> ~1000 us at T=8192, ~930 -> ~505 us at T=4096, ~460 -> ~257 us at T=2048.
- Output: fnv1a of the wide bench identical to the baseline at T = 1024 / 2048 / 4096 / 8192.
- Tests (target GB10 sm_121, CUDA 13): `infernix_offloaded_moe_{wide,layer,plan,cuda,layer_idle,a4_residual_122b,qwen3_5_122b,a8_wide}_test`,
  `infernix_sparse_moe_wide_test`, `infernix_cuda_stack_check` pass. A first version failed the stack check
  (48 then 24 bytes of stack at 4 CTAs/SM); fixed by an explicit unroll and 3 CTAs/SM.
- End to end, `pp8192`, c = 1, `-r 3 --warmup 1`, interleaved A/B of two binaries: baseline 3623 / 3610 / 3620 t/s,
  new 3692 / 3690 / 3678 t/s (+1.9% / +2.2% / +1.6%, mean +1.9%). An earlier unpaired run (3632 t/s)
  looked flat; the paired runs are the evidence. Single-point `pp8192` at c = 2 / 4 (unpaired): 3647 / 3675 t/s.
- Rejected: lowering `kWarpTile64MinColumns` to 64 (T=4096 9.18 -> 9.15 ms, T=8192 12.72 -> 12.70 ms; no change).
- Not done: `grouped_kernel` itself (~59% of its weight-stream floor in the model), roadmap Gap 6B wording
  (88-96% claim) still to be corrected.

### 2026-11-02 T7: prefill chunk size

- Command: `infernix_bench -p 8192|32768 -r 3|2 --warmup 1 --concurrency 1 --prefill-chunk N --kv-dtype int8 --spec mtp --draft-tokens auto --lm-head-draft`, W4Dense artifact, int8 KV, idle GPU.
- `pp8192` (single chunk at 8192+): 2710 / 3284 / 3612 / 3625 t/s at chunk 2048 / 4096 / 8192 / 16384.
- `pp32768`: 3556 / 3747 / 3839 t/s at chunk 8192 / 16384 / 32768. Attention cost grows with context, so the long-prompt asymptote is lower than the ~4.2k t/s the MoE-only fit predicts.
- Decision: no default change; the 16384 default is documented in `docs/cli.md`. Users with long prompts can pass `--prefill-chunk 32768` for +2.4%.

### 2026-11-02 T8: clocks during prefill

- `nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active` every 0.5 s during `infernix_bench -p 32768` (prefill 3712 t/s): no throttle reason ever set, SM clock stable at ~2.46 GHz.

### 2026-11-02 T2 / T3

- T3: `norm_kernel` moves ~336 MB (read + write of [8192, 4 x 2560] BF16) in 1.37 ms = ~245 GB/s; nothing to gain.
- T2: sweep of `launch_q8_a16_mma_*` variants (r64_t128 current for up, r128x64_t128_k64_a1 for down) via a temporary env override at T = 2048 / 4096 / 8192; the current launches won every case. Both shapes stay as they are.

### 2026-11-02 T4: router projection

- `projection_wide_kernel` (grid 17 x 512, 512 threads, FP32 FMA) is the router at T=8192: 21.5 GFLOP in 2.89 ms.
- `infernix_bench -p 8192 -r 3 --warmup 1 --kv-dtype int8 --spec mtp ...` with and without `--fast-path router-mma`: 3642.3 vs 3715.6 t/s (+2.0%).
- Not enabled by default: `docs/maintainer/model-quality.md` section 6 requires qualification, and the path changes the routing of a prompt with its call boundaries.

### 2026-11-02 T5: prompt attention roofline

- Roofline derivation: the selective prompt route attends, per query column, the union of selected
  blocks (counts >= 0 at the `pp8192` prompt, budget 2048 + 3 cap): sum of attended key positions
  ~14.69M per column-set. Each pair costs 2 x 128 FLOP over 12 query heads, issued on 16 MMA rows
  (m16n8k16 padding, 25% structural waste): 481 GFLOP per layer -> 4.96 ms per layer at the
  measured 97 TF/s BF16 F32-acc ceiling. The kernel runs 15.15 ms per layer (182 ms / 12) = 33%
  of that floor.
- NCU (synthetic operands, `/tmp/ncu_qsa_prompt_base.ncu-rep`): 2 warps per scheduler, 66% of
  cycles with no eligible warp, tensor pipe 26.2%, L2 hit 98.7% — latency-bound on the
  gather -> widen -> MMA -> softmax chain, not DRAM- or L2-bound.
- A/B (`INFERNIX_TMP_QSA_PROMPT_STAGES`, since removed): forcing Stages=2 (one 100 KB-smem CTA per
  SM, double-buffered ring) measured 24.1 ms vs 15.3 ms at 8192 columns (+57%). The existing
  gate (two CTAs per SM must fit, else Stages=1) is correct; adverse result recorded.
- Shipped: skip the per-tile 128-FMAL o-rescale when every row max held (`__all_sync` on
  alpha == 1; `exp2_approx(0)` is exactly 1, so the outputs are bit-identical).
  `infernix_qsa_attention_bench` prompt route at 8192 columns: 15.31 -> 15.20 ms median.
  Small win because the max rarely holds across a whole warp for a whole tile; the loop is kept
  because removing it outright would change bits.
- Open lever: the remaining 3x is the serialized per-tile dependency (int8 KV gather -> exact FP16
  widen -> QK MMA -> online softmax -> PV MMA -> rescale in the same 4 warps). Closing it needs
  warp specialization with TMA loads and an mbarrier pipeline (producer warps stream KV, MMA warps
  never wait on the gather), not another schedule constant. That is a kernel redesign, out of this
  campaign's scope.

### 2026-11-02 T6: GDN chunked prefill

- Floors (Hv=48, Hqk=16, D=128, T=8192): conv moves its byte minimum at 1.37 ms and runs 1.39 ms —
  closed. `prepare_kernel` floor ~0.78 ms DRAM, measured ~0.97 ms (81%): the follow-up is one CTA
  per (group, chunk) sharing the Q/K read across the 3 value heads instead of one per
  (value head, chunk). `chunk_recurrence_kernel` floor ~0.45-0.6 ms at the FP16-MMA rate; it ran
  1.89 ms before this change and 1.53 ms after — still ~3x above its floor.
- Cause of the recurrence gap: `value_tile` scored SM waves as ceil(value_heads / tile) instead of
  value_heads x (kStateDim / tile), so every tile looked like a sub-one-wave launch and tile 16
  always won — the opposite of its own comment. Fixed to pick the widest stripe that still fills
  two waves (`src/ops/linear_attention/gated_delta_net/chunked/recurrence.cu`).
- A/B (`INFERNIX_TMP_GDN_TILE`, since removed; `infernix_gated_delta_net_bench --force-chunked
  --breakdown --fp16-acc --csv`, interleaved medians at Hv=48): tile 16 1901.5 us vs tile 64
  1529.6 us at T=8192 (-19%); 122.6 vs 106.2 us at T=512 (-13%). Tile 32 sits between; the
  two-wave rule picks 64 at Hv=48 and keeps 32/16 for the underfilling head counts.
- Verification: `infernix_ops_recurrent_test gated_delta_net` passes (chunked fp16acc out/state
  profiles incl. H=16/48 T=2049); `infernix_ops_attention_test qsa`/`qsa_select` pass;
  `infernix_cuda_stack_check` passes; `chunk_recurrence_kernel<64, 1, float>` is REG=128 STACK=0
  LOCAL=0 (2 CTAs/SM). Measurement note: an earlier probe read a stale linked bench (default still
  tile 16); after relink the default matches forced tile 64 (1528.5 us).
- End to end, `pp8192` c = 1, `-r 3 --warmup 1`, paired interleaved A/B of the two binaries
  (T5 + T6 changes together): baseline 3648.2 mean, candidate 3710.5 mean (+1.71%); per-pair
  +0.07% / +4.59% / +0.57% — the +4.59% pair contains a 3549.5 t/s baseline outlier, so the
  conservative read is the median pair, +0.57%. Confirmation pair after the switch removal
  (baseline binary vs the shipped tree binary, 2 rounds): +0.44% / +0.73%, mean +0.58% — matches
  the conservative read. `pp8192+tg512` +0.23%; `tg512` decode +1.36%
  (within noise, both kernels are prefill-only); MTP acceptance unchanged (0.974-0.981).
- Temporary switches `INFERNIX_TMP_QSA_PROMPT_STAGES` and `INFERNIX_TMP_GDN_TILE` were removed with
  their readers and forced arms once the A/Bs decided, per the `temporary_switches` policy.

### 2026-11-02 T7 extension: prefill kernel inventory and per-T sweeps at T > 4096

- Chunk grid (`infernix_bench -p 4096,8192,16384,32768 -r 2 --warmup 1` per chunk process,
  int8 KV, MTP, W4Dense artifact), `prefill_tok_s_median`:

  | chunk | pp4096 | pp8192 | pp16384 | pp32768 |
  |---|---:|---:|---:|---:|
  | 4096 | 3365 | 3335 | 3300 | 3261 |
  | 8192 | 3337 | 3676 | 3625 | 3593 |
  | 16384 | 3341 | 3686 | 3872 | 3811 |
  | 32768 | 3342 | 3676 | 3858 | 3877 |

  The parameter's whole effect is the chunk count: each halving of the count adds ~5-8%
  (pp32768 3261 -> 3877, +18.9%; per-extra-chunk fixed cost ~0.23-0.28 s = the 68 GB expert
  weight re-stream, T7's fit). Once chunk >= prompt the value is irrelevant: paired A/B at
  pp32768 of 32768/40960/65536 read 3972/3966/3963 and 3955/3954/3954 (+-0.2%). A single-shot
  65536-vs-32768 gap of +2.4% seen in the first grid did not reproduce and is recorded as
  between-process noise. Default 16384 stays (decode-stall trade-off, T7 decision);
  `--prefill-chunk 32768` is worth +1.7% on pp32768.
- Inventory: pure-prefill nsys traces (`--cuda-graph-trace=node`, -r 2 --warmup 1, chunk = T) at
  T = 4096/8192/16384/32768. Per-round GPU ms (3 rounds averaged; 46 kernels, top ones):

  | kernel | 4096 | 8192 | 16384 | 32768 | us/tok 32K/4K |
  |---|---:|---:|---:|---:|---:|
  | grouped_kernel (wide experts) | 328 | 471 | 747 | 1279 | 0.49 |
  | q8_a16_mma (HC projections) | 162 | 343 | 674 | 1340 | 1.03 |
  | nvfp4_a4_tma (dense FFN + head) | 130 | 254 | 525 | 1030 | 0.99 |
  | qsa_prompt_kernel | 80 | 183 | 388 | 805 | 1.26 |
  | combine_inject + inject | 124 | 250 | 504 | 1004 | 1.01 |
  | projection_wide (router) | 69 | 142 | 282 | 556 | 1.00 |
  | norm + rmsnorm_warp | 90 | 179 | 359 | 717 | 1.00 |
  | quantize_token + a4 residual quantize | 50 | 104 | 211 | 427 | 1.05 |
  | GDN recurrence/conv-pairs/prepare | 72 | 145 | 305 | 610 | 1.05 |
  | gate_up + down (dense FFN pair) | 71 | 53 | 41 | 30 | 0.05 |
  | qsa score/select/prepare pipeline | 25 | 55 | 145 | 484 | 4.8 |
  | TOTAL per round | 1280 | 2274 | 4286 | 8321 | 0.81 |

  Weight-stream kernels amortize (grouped 0.49, gate_up/down ~0.05), elementwise and GEMM
  kernels are per-token flat, and the QSA selection pipeline grows ~5x per token with context —
  at T=32768 it costs 484 ms/round, 60% of qsa_prompt itself. `spin_kernel` (65 ms/round
  constant) is the host idle wait, not prefill work.
- Per-T candidate sweeps (route ladders, cold L2, this artifact's shapes):
  - Q8 HC shapes 10240x320 and 324x10240 (149 routes): production is the best bit-exact
    T-invariant route at every T (no adoptable candidate printed at 4096..32768).
  - NVFP4 a4 2560x6144 / 16384x2560 / 13952x2560: production within 1% of the ladder argmin at
    T <= 16384; at T = 32768 the ladder is 2.9-3.4% ahead (`a4_split_1q1g` on 13952x2560,
    `a4_tma_t256_r128_s3_m1_l2` on 16384x2560, `a4_tma_t256_r128_s3_m1_all` on 2560x6144).
    At 12.4% of the T=32768 round that is ~0.4% E2E — a candidate for a T-band extension, not
    yet adopted (needs in-route paired A/B and bit-exactness qualification of the split variant).
  - `--routes fast` vs exact on all five linear shapes: flat (+0.7% to -1.5%, noise) — no
    fast-path win at these widths.
  - QSA prompt kernel variant 0 (combined gather) vs 1 (split K/V stage): identical (+-0.15%).
  - GDN auto vs `--force-chunked`: identical (chunked is already the production route at
    every T; recurrence/prepare stay linear: 775/1536/3038/6061 us per call).
  - Wide-expert activation ladder a4 vs a8 vs a16 at T=4096..32768: a4 (production) is 1.8-4x
    faster everywhere (T=32768: 31.5 vs 111.8 vs 129.7 ms per layer-call, 102 vs 28.8 vs 24.8
    TF/s); a8/a16 also change bits (not adoptable).
- Conclusion: at T > 4096 every prefill kernel family's production choice is the measured argmin
  except the NVFP4 a4 T=32768 band (-3%; superseded — see the T7 extension 2 section below, the
  live route is the 2-term pair and is already its own argmin); the prefill t/s headroom above
  the current default is the chunk-count effect (measured, parameter-covered), the router's
  FP32 SIMT cost (T4, fast-path qualification), grouped_kernel's in-model weight-stream gap
  (T1, open) and the QSA prompt/selection redesigns (T5).

## 2026-11-02 (T7 extension 2) — a4 live-route correction and the QSA selection pipeline

Follow-up to the per-T sweeps above, implementing the two outstanding findings. Both were
investigated to a measured verdict; one code change shipped (QSA score kernel).

### NVFP4 a4 T=32768 band — corrected, not adoptable

The earlier sweep bullet ("at T = 32768 the ladder is 2.9-3.4% ahead") compared 1-term TMA
ladder arms against a production arm the model does not execute. Evidence:

- The dense shapes' `select_a4` 1-term route (T >= 257) is dead at model prefill: the
  artifact's a4-residual policy plus the pair gates (33 / 65) and the fused split launchers
  (`launch_nvfp4_n16384_k2560_split`, `launch_nvfp4_n13952_k2560_split`, gate 65) route every
  dense NVFP4 GEMM to the 2-term residual pair TMA. The nsys demangled names at T = 32768
  confirm it: all launches are
  `nvfp4_a4_tma_kernel<Nvfp4ScheduleInstance<Nvfp4A4TmaMmaSchedule<128, 3, 1, WeightCodeL2_128B, 2, 128>, K>>`
  — per round 36x 16384x2560 (avg 15677 us), 12x 13952x2560 (avg 13572 us), 48x 2560x6144
  (avg 6054 us), 1x lm-head (12361 us): ~1018 ms of the 8321 ms round.
- The sweep gained five pair candidates (r128_s3_m1 promotions ws/all/none, baked-K 2560/6144).
  At T = 32768/40960 (cold 32 MiB L2): promotions within ±1% (noise, direction flips between
  shapes); baked-K beats runtime-K by 2.5-4.3% (13952x2560 14409 vs 14948 us, 16384x2560
  16855 vs 17311 us, 2560x6144 8399 vs 8774 us). The live launch is the baked-K WeightCodeL2
  pair schedule, i.e. the live route is the measured argmin of its own ladder.

No production change; the sweep candidates stay as measurement infrastructure.

### QSA selection pipeline (score / select / prepare)

Per round at T = 32768 (trace): `qsa_score_wide_kernel` 163 ms (3072 launches, avg 52 us),
`qsa_select_global_kernel` 59 ms, `qsa_prepare_kernel` 51 ms (76 launches) — 273 ms = 3.3% of
the round; bench `select_us` = 17.5 ms/layer. NCU: score_wide is issue-bound — SM 60.2%,
occupancy 33.3% (2 CTAs/SM capped by REG 108 and 42.2 KiB smem), 4 warps/scheduler, FFMA 58%
of issued instructions; per thread per key tile 1024 FFMA + ~248 FADD (butterfly stack) +
192 LDS (128 q float4 + 64 key uint2) + ~256 widen SHF/LOP3 + 32 strided STS.U16.
select_global is pure latency: long_scoreboard 12.43 + barrier 7.17 of 33.09 cyc/instr,
SM 18%. prepare is a warp-per-task streaming kernel (normalize/rotate/codec).

- Shipped: key staging as uint4 pair rows. `store_tile` stages a thread's two blocks
  (g, g + 8) as consecutive rows of one uint4 lane, so each leaf's key read is one LDS.128
  instead of two LDS.64 (192 -> 160 LDS per thread per key tile). Pure data movement — the
  `qsa_select` and `qsa` oracle suites pass bit-exact, stack ratchet passes, REG 108 -> 107
  STACK 0 LOCAL 0. Interleaved A/B vs the HEAD-built binary at 32768 columns
  (--warmup 2 --repeat 15, 6 rounds): select_us bf16 −0.94%, int8 −0.68%. E2E gate
  (kv-matrix nvfp4/int8/mtp, pp8192): 3710.3 prefill t/s, 43.6 decode t/s, acceptance 0.950 —
  unchanged (the pipeline is ~15 ms/round at pp8192, so no E2E delta is expected or claimed).
- Rejected, measured: depth-2 key-tile prefetch +6.8/+7.3% (select_us bf16/int8); selection
  slices 2 -> 4 (kSliceBlocks 4096 -> 2048, grid factor 16x sms) +5.9/+6.2% — histogram
  parallelism is not the constraint, the extra wave costs; div-free float4 q staging neutral
  (±0.1%) and dropped. Rejected pre-measure: kWalkBatch 4 -> 8 (REG 63 -> ~79 would halve
  select CTAs/SM); FP32-key staging, double-buffered keys and 64-block tiles all exceed the
  49.5 KiB second-CTA budget (1 CTA/SM, the file's own 2-CTA rule); q register hoist needs
  512 registers; tensor-core dots violate the fixed butterfly bit contract.
- Remaining structure (open, T5-class redesigns): score_wide needs warp-specialized TMA
  producer/consumer or a 3-CTA staging layout to go materially above ~55% FFMA issue;
  select_global's last-CTA serial filter+radix could be spread across slices with a
  published-bin spin (est. <= 30% of 59 ms/round) at real complexity cost.
