# GB10 changelog 2026-11-06

Scope: `context_kv_materialize` (DFlash2 drafter five-layer context K/V ring materialization)
task T1 from `docs/tasks/20261106-performance-optimization-tasks.md`: eliminate the FP32
`key_scratch` global round-trip and the separate `context_kv_key_post_kernel` launch on every
column band where a `Rows = kHeadDim` fused-MMA route measures faster, and select the measured-best
raw-code staging grouping per band, on NVIDIA GB10 (`sm_121a`).

## 1. Accounting

- **Base commit (`origin/main`)**: `68c54356fd490ab329bd1475d48957f886bb7dd1`
- **Previous HEAD**: `b7ff28f87` (`docs: next tasks`)
- **Delta in this milestone**: 3 files changed, +131 / -38. `cuda-kernel-toolkit/` intentionally
  excluded from file-by-file enumeration (integrated toolkit, untouched).

## 2. Task-premise corrections / Architectural rationale ("Why" on GB10 `sm_121a`)

1. **The dual `codes` SMEM plane was never a fitting constraint (falsified premise)**:
   the task doc states `codes[128][128]` "wastes 16,384 bytes" and blocks `Rows = 128` from fitting
   in GB10's SMEM. The shipped baseline `Fused64 <128, 64, 128, 2>` already ran `Rows = 128` with
   the codes plane at 66,560 B dynamic SMEM ≤ 101,376 B limit. Codes removal is therefore a
   performance option, not a prerequisite for key fusion.
2. **The doc's "stage codes into the upper half of `code_values`" scheme is race-prone**: the upper
   half of `code_values` *is* the decoded destination rows the current tile's `ldmatrix` reads;
   `cp.async` of tile k+1 landing there while MMA(k) reads it is a WAR hazard. A correct single-
   buffer scheme must either keep a separate staging buffer (the baseline design) or keep raw codes
   out of SMEM entirely (register staging, implemented here as `LdgCodes`).
3. **Register staging (`LdgCodes`) is band-dependent, not uniformly better**: LDG→register→decode
   keeps the next tile's weight loads in flight across the MMA, but at 1 CTA/SM the cp.async group
   hides the global latency better at two-to-four column blocks per grid. Measured 3-round
   interleaved cold-L2 A/B: LDG +8..11% at 129..192 columns, LDG −8..−9% at ≥512 columns versus the
   cp.async instantiation of the same tile. Both mechanisms ship, selected per band.
4. **Scale-plane grouping is band-dependent too (`TailScales`)**: grouping `stage_scales(next)`
   with the activation group (instead of the baseline's codes group) measures −1.5..−4.4% at two
   column blocks and +3..5% from three up. Both groupings ship, selected per band.
5. **The doc's baseline route table was misdescribed**: shipped baseline was KSplit16 `≤16`,
   KSplit24 `17..24`, KSplit16 `25..32`, Mma32 (Rows=64, scratch) `33..64`, Mma80/Mma96
   (Rows=64, scratch) `65..96`, Fused64 `97..256`, Mma64 (Rows=64, scratch) `≥257` — not
   "KSplit16 for 17..20, KSplit24 for 21..32".
6. **`T = 17..64` was "unfused" but already at its measured best (falsified premise)**: those bands
   win CTAs from split-K (`context_kv_grouped_kernel`) and half-height `Rows = 64` schedules
   (16 row-blocks × grid.y vs 8 at `Rows = 128`). Promoting them to `Rows = 128` fused MMA measured
   **+5..11.5% slower** (3-round interleaved A/B); the promotion was reverted and the baseline
   routes kept. The doc's cited T=32 (339.7 µs) > T=48 (311.1 µs) inversion did not reproduce in
   this session's thermal state (T=32: 268.3 µs < T=48: 309.0 µs).
7. **Mma80/Mma96 promotion needs `BlockK = 64`**: `Rows = 128, Columns = 80/96, BlockK = 128`
   spills (STACK 8/16 at 512/768-thread launch-bounds register caps); `ColumnWarps = 1` at 256
   threads is worse (REG 255, STACK 24). `<128, 80, 64, 2>` / `<128, 96, 64, 2>` with LDG staging
   are spill-free (REG 118/122) and replace the 640/768-thread Rows=64 scratch schedules.
8. **Correctness lesson (UB-visible in SASS)**: the first LDG decode indexed
   `decoded[pair >> 1].pair[pair & 1]` on a 2-element `Bf16x8` array for 16-code chunks; the
   out-of-bounds `decoded[2]/decoded[3]` stores are UB, and ptxas legally deleted the loads feeding
   them (SASS showed 1 uint4 loaded per tile where PTX had 2), producing NaN caches. The correct
   mapping is `decoded[pair >> 2].pair[pair & 3]`. Verified per-symbol: 2×`LDG.E.128` prologue +
   loop, 16×`F2FP.BF16` per tile for `kChunksPerThread = 2`.

## 3. Subsystem breakdown

|  | File | Change |
|---|---|---|
| M | `src/ops/context_kv_materialize/materialize.cu` | `context_kv_mma_kernel` gains `LdgCodes`/`TailScales` template knobs (LDG→register codes path with `decode_registered_codes`; cp.async path keeps the `codes` plane as async staging); epilogue fuses per-head Key RMSNorm + RoPE + BF16 cache store (`store_key_head`) whenever `Rows == kHeadDim` and writes the FP32 scratch only for the half-height Mma32 route; `launch_projection`: restore KSplit24 + KSplit16(25..32) + Mma32 `<64,32,128,2>` scratch routes, promote Mma80/Mma96 to `<128,{80,96},64,2,true,true>`, split Fused64 into `Fused64 <128,64,128,2,false,true>` (≤128 cols), `Fused64Mid <128,64,128,2,false,false>` (129..256, baseline grouping), `Fused64Wide <128,64,128,2,true,true>` (≥257); `Mma64` instantiation deleted. |
| M | `src/ops/context_kv_materialize/launch.h` | Route enum `{KSplit16, KSplit24, Mma32, Mma80, Mma96, Fused64, Fused64Mid, Fused64Wide}`; route table `≤16 KSplit16, ≤24 KSplit24, ≤32 KSplit16, ≤64 Mma32, ≤80 Mma80, ≤96 Mma96, ≤128 Fused64, ≤256 Fused64Mid, else Fused64Wide`; `uses_scratch` = KSplit16/KSplit24/Mma32. |
| M | `tests/ops/test_context_kv_materialize.cpp` | Batched sweep gains width 21 so the capacity-interval equality check reaches the new largest scratch envelope (63 columns at B=3). |

## 4. Verification

- Build: single `ninja -C build-native -j $(nproc) infernix-serve infernix_ops_attention_test
  infernix_context_kv_materialize_bench` after `find src/ops/context_kv_materialize -name '*.cu' -exec touch {} +`.
- `ctest -R "context_kv_materialize|infernix_cuda_stack_check"` → **2/2 PASS** (oracle: 3 real
  artifact geometries × all sweep widths/batches, graph replays, guards, capacity==observed peak).
- Spill ratchet: all 18 `context_kv_mma_kernel` instantiations + grouped/post kernels on the linked
  `infernix-serve` are `STACK:0 LOCAL:0` (max REG 128). `tools/cuda_stack/cuda_stack_allowlist.json`
  has no context_kv entries and is unchanged (nothing to ratchet down; zero new spills).
- A/B: `/tmp/t1_base_bench` (HEAD `b7ff28f87` binary) vs candidate, 3 interleaved rounds per sweep,
  cold L2 (32 MiB flush), graph execution, NVIDIA GB10. Medians, per band:
  - columns ≤ 64 (routes unchanged): −2.6..+1.4%, all within run-to-run noise; scratch/nodes unchanged.
  - **65..96** (new fused LDG Mma80/Mma96): **−3.4..−15.5%**, graph nodes 2→1, workspace peak → 0
    (e.g. 65: 403.3→340.8 µs, 80: 411.3→347.8 µs, 9,8=72: 405.2→343.9 µs).
  - **97..128** (Fused64, tail scales): **−1.4..−5.1%** (e.g. 97: 423.6→405.2, 16,8=128: 435.9→417.5).
  - 129..256 (Fused64Mid, baseline grouping): −0.4..+0.4% (unchanged by design).
  - **257..2048** (Fused64Wide replaces Rows=64 scratch Mma64): **−5.3..−12.2%**, nodes 2→1,
    workspace peak 41.9 MiB→0 (257: 795.6→753.5, 512: 1341.1→1220.4, 1024: 2786.1→2446.0,
    2048: 5669.6→5105.3 µs).

## 5. Regressions & adverse results

1. **First promotion attempt regressed 17..64 columns +5..11.5%** (Rows=128 halves CTA count vs the
   split-K/half-height baseline routes). Detected by the same interleaved A/B; the 17..64 baseline
   routes (KSplit24, KSplit16, Mma32 scratch) were restored. Net final effect on 17..64: unchanged.
2. **Uniform LDG staging regressed 97..128 (+3.4..6.8%) and 129..256 (+8..11%)**; fixed by the
   per-band `LdgCodes` selection above. Net final effect: no band is slower than baseline beyond
   the ±2.6% noise band.
3. **Tail-grouped scales regressed 129..256 by +3..5%**; fixed by the `TailScales` knob (the
   129..256 route keeps the baseline codes-group staging = `Fused64Mid`).
4. **The T1 acceptance bar "≥10–25% on formerly unfused T=17..96" is met only for 65..96.**
   17..64 cannot meet it: its unfused baseline is measured-faster than any fused Rows=128 variant
   (premise 6). The 129..256 band is likewise already fused in the baseline and stays ≈ flat.
5. A standalone pattern repro of the LDG load loop passed while the real kernel produced NaN; the
   repro had omitted the decode consumer, which is where the OOB index (§2.8) lived — repro
   harnesses must exercise the full load+consume+store pattern.
6. The doc's cited baseline latencies (e.g. T=32 339.7 µs) did not reproduce in this session's
   thermal state; all accept/reject decisions were taken against the freshly HEAD-built baseline
   binary in interleaved rounds, never against the doc's absolute bars.

## 6. Accepted gaps / follow-ups

- **T ≤ 16 (KSplit16) and 17..24 (KSplit24) keep the FP32 scratch + `context_kv_key_post_kernel`**
  (2 graph nodes, scratch peak intact). The grouped split-K kernel splits K across CTAs, so no CTA
  owns a full 128-row Key head; fusing RMSNorm+RoPE there needs a cross-CTA reduction or a
  whole-head grouped schedule. Measured: the grouped route beats every fused variant tried at these
  widths, so the scratch round-trip is the cheaper side of that trade.
- 25..32 keeps KSplit16 and 33..64 keeps the Rows=64 scratch Mma32 for the same CTA-count reason.
- The 128/256-column staging boundaries are grid-wave-based empirical choices measured at the bench's
  full-count envelopes; a differently-shaped batch mix reaching the same column totals exercises
  the same routes, but re-measure before moving the boundaries.
- `Fused64Mid` is structurally the baseline mainloop; if the LDG pipeline improves (e.g. a
  register-budget change lowers REG below the 2-CTA threshold at 512 threads), retest it as a
  single-LDG band and drop the extra instantiation.

---

# GB10 (`sm_121a`) Branch Changelog — 2026-11-06 — T2: Transpose `dynamic_grouped_conv_prepare` FP32 scratch to `[split][row][token]`

Scope: `bf16_dynamic_grouped_conv_prepare` (DFlash2 drafter RMSNorm→projection→conv-coefficient
prepare) task T2 from `docs/tasks/20261106-performance-optimization-tasks.md`: transpose the FP32
split-K partial scratch from `[split][token][coefficient_row]` to
`[split][coefficient_row][token]` so the MMA epilogue stores are contiguous (float2 across even
`tokens`) and the split-K reduction reads are contiguous +4 B per lane, on GB10 (`sm_121a`).

## 1. Accounting

- **Base commit (`origin/main`)**: `68c54356fd490ab329bd1475d48957f886bb7dd1`; **previous HEAD**:
  `1ebbf71a1` (T1).
- **Files changed**: `src/ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_partial.cu` (M),
  `src/ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_reduce.cu` (M).
  `bf16_dynamic_grouped_conv_prepare_plan.cpp` (reviewed, unchanged — capacity formula is
  layout-invariant and route bands re-validated, §2.5). `cuda-kernel-toolkit/` untouched.
- **Allowlist delta**: none — both kernels were and remain `STACK:0 LOCAL:0`; no `conv_prepare`
  entry exists in `tools/cuda_stack/cuda_stack_allowlist.json` and none was added.
- **Rebuild count**: 2 single-invocation Ninja builds (fresh HEAD baseline binary, candidate),
  shared device link.

## 2. Task-premise corrections / Architectural rationale ("Why" on GB10 `sm_121a`)

1. **The layout premise held**: the old `put(r, c)` wrote `(split·tokens + c)·kCoefficientRows + r`,
   so a warp's four lanes sharing accumulator row `r` stored 2,560 B (H=2560) / 5,120 B (H=5120)
   apart and the reduce's adjacent lanes read the same strides — 1 float per 128 B line. The
   transpose makes both contiguous; every `(coefficient, position)` still sums the same split values
   in the same ascending order, so the change is bit-exact by construction.
2. **The doc's live-evidence symbol is stale**: `partial_kernel<5120, 64, 64, 64, 2, 2>` (`REG:103`)
   is not instantiated in this tree; live routes are R16/C{8,16,32,48}/S8 (T≤4) and
   R32/C{32,64}/S4 (T≥5) at `REG:40..58`. Baseline live reduce was `REG:38`, not 32.
3. **The W14B6 (T=84) "103.8 µs spike" did not reproduce as a spike**: the freshly HEAD-built
   baseline measured 89.8 µs there, on an already smooth T-curve (T=80: 91.9, T=88: 90.2 medians).
   The doc's absolute bars came from a different thermal state; all decisions were taken against the
   freshly built baseline binary in interleaved rounds.
4. **The improvement is real but smaller than the 15–30 % bar**: the Op is dominated by streaming the
   13.1 MB BF16 projection weight (~48 µs floor at ~273 GB/s UMA); the scratch is only
   0.26–2.6 MB, so removing its 32× line amplification moves the total monotonically with T from
   ≈ −2.7 % (T=13..31) to −13.3 % (T=97..128), mean −7.8 % over T=32..128 (§4).
5. **Route bands need no retune**: both C32 and C64 R32 bands carry identical split-4 scratch cost,
   and the candidate T-curve shows no cliff at the 4/5, 12/13, 64/65, 96/97 edges (≤ 1 timer tick).
   PDL overlap (doc step 3) was already shipped (`pdl::enter_streaming`/`enter` +
   `pdl::launch_consumer` on both kernels); no additional change.

## 3. Subsystem breakdown

- `bf16_dynamic_grouped_conv_prepare_partial.cu` — `dynamic_grouped_conv_prepare_partial_kernel`
  epilogue: index `partial` as `(split·kCoefficientRows + r)·tokens + c`; when `tokens` is even and
  `c + 1 < tokens` (all production widths; `col` provably even → 8 B-aligned) store
  `make_float2(acc[m][0], acc[m][1])` / `(acc[m][2], acc[m][3])`, else adjacent scalar stores for the
  odd-token case and the straddling tail column. SASS: 4 × `STG.E.64` in the hot path of
  `<5120,32,64,4>` (baseline: 8 scalar `STG.E`).
- `bf16_dynamic_grouped_conv_prepare_reduce.cu` — `dynamic_grouped_conv_prepare_reduce_kernel`
  phase 1: hoist `row = coefficient·kGroups + group`, read
  `partial[row·tokens + batch·width + position + split·kCoefficientRows·tokens]` — contiguous +4 B
  per lane within a `Capacity` group instead of 2.5–5.0 KiB-strided. Sum order unchanged.

## 4. Verification

- **Build**: `find src/ops/dynamic_grouped_conv -name '*.cu' -exec touch {} +` then one
  `ninja -C build-native -j $(nproc) infernix-serve infernix_ops_recurrent_test
  infernix_dynamic_grouped_conv_prepare_bench` (the doc's `infernix_ops_conv_state_test` target does
  not exist; the prepare test builds into `infernix_ops_recurrent_test`).
- **CTest** (`-R "dynamic_grouped_conv|infernix_cuda_stack_check"`, serial): 3/3 passed —
  `infernix_cuda_stack_check`, `infernix_dynamic_grouped_conv_prepare_test` (all 120 W×B geometries
  × H ∈ {5120, 2560, 3072}, incl. odd-token scalar-fallback and CUDA-graph cases),
  `infernix_linear_dynamic_grouped_conv_add_test`.
- **Resources** (`cuobjdump --dump-resource-usage`, candidate vs baseline): partial `<5120,32,64,4>`
  REG 58 vs 56, STACK 0 / LOCAL 0 both; reduce REG 48 vs 38 (+10 from 64-bit row/stride math),
  STACK 0 / LOCAL 0 both; `__launch_bounds__(…, 4)` still admits 4 CTAs/SM
  (48·128·4 = 24,576 ≤ 65,536 regs) — occupancy unchanged.
- **A/B** (`infernix_dynamic_grouped_conv_prepare_bench`, cold-L2 graph medians; 7 baseline ×
  6 candidate interleaved full sweeps of 120 points, per-point medians of medians):

  | T band | n | mean Δ | worst point |
  |---|---|---|---|
  | 2..4 (R16/S8) | 4 | −0.41 % | +0.02 % |
  | 5..12 (C32) | 19 | −0.96 % | −0.02 % |
  | 13..31 (C64) | 35 | −2.66 % | +0.11 % |
  | 32..64 (C64) | 37 | −5.72 % | −2.34 % |
  | 65..96 (C32) | 18 | −10.00 % | −7.02 % |
  | 97..128 (C64) | 7 | −13.31 % | −9.07 % |
  | 32..128 | 62 | **−7.82 %** | −2.34 % (all 62 improved) |

  Key points: W14B6 T=84 89.8 → 78.9 µs (−12.2 %, spike gone — curve monotone-smooth),
  W16B8 T=128 102.1 → 86.8 µs (−15.0 %), W16B1 T=16 70.7 → 68.4 µs (−3.3 %).
  Pass-to-pass spread reached 13.7 % (base) / 33.0 % (cand, one small-T point), so single passes are
  not decision-grade; the medians above are.

## 5. Regressions & adverse results

Compared: all 120 bench points (W∈2..16 × B∈1..8) baseline-vs-candidate on interleaved freshly
built binaries, plus per-symbol REG/STACK/LOCAL/SMEM, CTest results, and route-band edges.

- **No latency regression at any point.** Largest positive delta +0.11 % (W2B8 T=16, 70.46 → 70.54 µs)
  and +0.02 % (W2B1 T=2) — both inside the measured pass-to-pass spread and ≈ 0.05–0.1 timer ticks;
  disposition: noise, accepted.
- **Reduce kernel REG 38 → 48.** Cause: 64-bit `row_base`/`split_stride` hoisting. STACK/LOCAL stayed
  0, launch-bounds occupancy unchanged, and no small-T latency cost is measurable (T=2..12 mean
  −0.96 %). Disposition: accepted tradeoff.
- **Acceptance bar not met**: the doc's "≥ 15–30 % at T=32..128" is not reached (mean −7.8 %; only
  T=128 individually approaches −15 %). The premise correction in §2.4 explains the size; the bar is
  disclosed as unmet rather than re-tuned.

## 6. Accepted gaps / follow-ups

- Bit-exactness is by construction (identical MMA mainloop, identical stored values, identical
  split-ascending sum order) and the 360-geometry oracle test passes, but the bench does not dump
  outputs, so no direct old-vs-new binary output-bit diff was run.
- The E2E drafter routes (DFlash2/MTP decode via `qwen4_exp`/`qwen3_5` `dflash2.cpp`) were not
  re-benchmarked: outputs are bit-identical by construction and the op-level win is 0.5–2.5 µs per
  prepare call; E2E effect is expected-positive but unmeasured here.
- Odd `tokens` (e.g. W3B1) keeps scalar stores and gains nothing measurable (−0.0…0 %); a
  shuffle-packed float2 path for odd T would trade instructions for sectors at T where the scratch
  is noise — not worth it at these widths.

# GB10 (`sm_121a`) Branch Changelog — 2026-11-06 — T3: Multi-CTA occupancy + single-pass vectorized split-KV merge in `sliding_window_attention` & `context_softmax_attention`

Scope: task T3 from `docs/tasks/20261106-performance-optimization-tasks.md` on GB10 (`sm_121a`):
raise split-KV CTA residency in the two `context_query` attention kernels, drop the dead query-position
register state, and replace the 128-thread two-pass shared-memory split-KV merge with a warp-per-row
single-pass vectorized merge.

## 1. Accounting

- **Previous HEAD**: `b5e1d89f3` (`dev-rebased-20261028`, T2). Worktree was clean at session start.
- **Files changed** (all `M`):
  `src/ops/softmax_attention/common/context_query.cuh`,
  `src/ops/softmax_attention/sliding_window/kernel.cuh`,
  `src/ops/softmax_attention/dense/context/kernel.cuh`,
  `src/ops/softmax_attention/dense/context/launch.cu`.
- **Allowlist delta**: none. Every `*_split_partial_kernel` / `*_reduce_kernel` instantiation was
  `STACK:0 LOCAL:0` before and stays `STACK:0 LOCAL:0`; no entry for these kernels exists in
  `tools/cuda_stack/cuda_stack_allowlist.json` and none was added or changed. The one-way ratchet was
  never run (`check_cuda_stack.py update` was not needed because nothing was reduced).
- **Rebuild count**: 3 single-invocation Ninja builds (fresh HEAD baseline, candidate, final
  comment/indent-only rebuild), each sharing the `cmake_device_link.o` device link; headers under
  `src/ops/softmax_attention` were touched into their consuming `.cu` files before every build.

## 2. Task-premise corrections / Architectural rationale ("Why" on GB10 `sm_121a`)

1. **`__launch_bounds__` was already `MinBlocks = 2`, not `1`** (`git log -S
   '__launch_bounds__(WarpsPerCta * 32, 1)' -- src/ops/softmax_attention/` returns no commit). The
   measured base occupancy was therefore already 3 CTAs/SM at `WarpsPerCta = 3`, 4 at 2, and 6
   (SMEM-bound) at 1; only the 4-warp shapes (`T = 13..16`, 128 threads, 215–222 regs) sat at
   2 CTAs/SM. The doc's "1–2 CTAs/SM despite 16 KiB SMEM" holds only for that 4-warp class.
2. **The dead `q_positions[Tokens]` / `min_position` / `max_position` state and `prime()` loop never
   existed in this tree** (`git log -S min_position -- src/ops/softmax_attention/` returns no commit).
   `SlidingWindowAttentionPolicy` already stores `const std::int32_t* positions` and reads
   `positions[token]` on demand, and its `prime()` is an empty hook — the hook has to stay because
   `ContextAttentionPolicy::prime` is *live* (it preloads the 4-lane block-table group). So this item
   was already satisfied at base and nothing was removed.
3. **Register clamp works at `KeyBlock = 32` and spills at `KeyBlock = 64`.** A `KeyBlock = 32` CTA
   carries 16 KiB dynamic SMEM, so residency is register-bound at 65,536 regs/SM: clamping
   `MinBlocks` to 3 (4-warp) / 4 (≤3-warp) lands `REG` at 166–168 with `STACK:0` and gives
   3 and 4 CTAs/SM. The first candidate also clamped the `KeyBlock = 64` route to 3 and
   `cuobjdump` reported `STACK:40–72` on `context_attention_split_partial_kernel<9..16,3|4,64,0>`;
   3 CTAs/SM there needs ≤ 170 regs from 238–240, which ptxas only reaches by spilling, and 32 KiB
   SMEM caps that route at 3 CTAs/SM anyway (2 register-bound is within one CTA of the SMEM limit).
   The clamp is therefore gated on `KeyBlock` (§3).
4. **The merge rewrite matches the doc's description of the context path** (15 `__syncthreads()` in
   source; 17 block-barrier instructions in SASS) and now executes **0** block barriers, 20 shuffles,
   `max_load_width` 4 → 8 B (BF16 `uint2`; the FP32 sliding-window path gets 16 B `float4`), and
   `mufu` (exp) ops 15 → 9. The 4 dimensions per lane cost static instructions
   (+38.9 % per CTA) but each CTA now merges 4 rows, so the Op's total merge work drops.
5. **Wave arithmetic the clamp targets**: `T = 16`, `L = 1024`, `B = 1` is `splits = 16` →
   `8 KV heads × 16 = 128` CTAs; at 2 CTAs/SM the 48-SM wave holds 96 CTAs (2 waves, second at 33 %
   fill), at 3 CTAs/SM it holds 144 (1 wave). §4 measures that band.

## 3. Subsystem breakdown

- `src/ops/softmax_attention/common/context_query.cuh`
  - `kContextQueryReduceWarps = 4` — one warp merges one `(q_head, token)` row, so a reduce CTA runs
    4 rows.
  - `kContextQueryMinBlocksPerSm<WarpsPerCta, KeyBlock>` — 3 (4-warp) / 4 (≤3-warp) at
    `KeyBlock = 32`, 2 at 64 (§2.3).
  - `context_query_reduce_body<Partial, Tokens, KeyBlock, WarpsPerBlock, MaxSplits>`: row index comes
    from `blockIdx.x * WarpsPerBlock + warp`; `global_m` / `global_l` are `warp_max<32>` /
    `warp_sum<32>` over the same `split = lane; split += 32` mapping the old code used, the per-split
    rescale weight is computed once and parked in `weights[warp][split]`, and the numerator loop reads
    4 consecutive dimensions per lane (`float4` for FP32 partials, `load_bf16x4` → one 64-bit load for
    BF16 partials) in ascending split order, then writes `pack_bf16x2` × 2 as one `uint2`.
- `src/ops/softmax_attention/sliding_window/kernel.cuh` — `sliding_window_attention_split_partial_kernel`
  takes the computed `MinBlocks`; `sliding_window_attention_reduce_kernel` is now a wrapper that reads
  `positions[Tokens * batch]`, passes `min(length, window_mask)` as the tile count and calls the shared
  body. The FP32 sum order, the `split → lane` mapping and the final `numerator / global_l` division are
  unchanged, so this route is bit-identical by construction.
- `src/ops/softmax_attention/dense/context/kernel.cuh` — same `MinBlocks` treatment;
  `context_attention_reduce_kernel` calls the shared body with `Partial = __nv_bfloat16` and
  `MaxSplits = kContextQueryMaxSplit` (85, the plan's own cap).
- `src/ops/softmax_attention/dense/context/launch.cu` — reduce grid `(32, T, B)` × 128 threads →
  `(8T, 1, B)` × 128 (4 rows/CTA); the split-partial launch, plan and workspace are untouched.

## 4. Verification

**Build** (final, after the comment/indent-only edits; headers touched into their consuming `.cu`
files first, single Ninja invocation so the `-rdc` device link runs once):

```text
find src/ops/softmax_attention -name '*.cu' -exec touch {} + && \
ninja -C build-native -j "$(nproc)" infernix-serve infernix_ops_attention_test \
      infernix_sliding_window_attention_bench infernix_context_softmax_attention_bench
[29/29] Linking CXX executable apps/infernix-serve        # 0 warnings, 0 errors
```

**Register / SMEM / stack resources** — `cuobjdump --dump-resource-usage` on
`build-native/apps/infernix-serve`
(`sm_121a`), HEAD baseline vs candidate; CTAs/SM is the register-bound residency
(65,536 regs/SM), which is the binding limit at `KeyBlock = 32` (16 KiB dynamic SMEM allows 6):

| kernel class | REG base → new | STACK | SMEM (dyn) | CTAs/SM base → new |
|---|---|---|---|---|
| `sliding_window_attention_split_partial_kernel` KB=32, W=4 (`T=13..16`) | 216 / 208 → **168** | 0 → 0 | 16 KiB | **2 → 3** |
| `sliding_window_attention_split_partial_kernel` KB=32, W=3 (`T=9..12`) | 215 / 207 → **167 / 168** | 0 → 0 | 16 KiB | **3 → 4** |
| `sliding_window_attention_split_partial_kernel` KB=32, W≤2 (`T≤8`) | 223–233 (unchanged) | 0 | 16 KiB | 4 (already; 4 CTAs need ≤ 256 regs) |
| `context_attention_split_partial_kernel` KB=32, W=4 / W=3 | 222 / 214 → **166–168**; 220 / 212 → **168** | 0 → 0 | 16 KiB | **2 → 3** / **3 → 4** |
| both `*_split_partial_kernel` KB=64 | 238–240 (unchanged, clamp gated off) | 0 | 32 KiB | 2 (SMEM caps at 3) |
| `sliding_window_attention_reduce_kernel` (W=1 / W=4 rows-per-CTA configs) | 48–56 → 61 / 64 | 0 → 0 | 1152 → 1364 / 1536 → 2384 B static | — |
| `context_attention_reduce_kernel` | 47–48 → 48 | 0 → 0 | 1536 → 2384 B static | — |

All 128 touched instantiations are `STACK:0 LOCAL:0` before and after;
`infernix_cuda_stack_check` (the one-way ratchet) passes and
`tools/cuda_stack/cuda_stack_allowlist.json` is unchanged (nothing was reduced, so `update` was
never run).

**SASS** — `ckt compare --exe-a <HEAD infernix-serve> --exe-b <candidate> --symbol …`:

| symbol | base → new |
|---|---|
| `sliding_window_attention_split_partial_kernel<16,4,32,0>` | total instr 1280 → 1360 (+6.25 %), `uniform_ops` 3 → 43 (the new row-index math), tensor / load / store counts unchanged |
| `context_attention_reduce_kernel<16,32>` | block barriers 17 → **0**, shuffles 0 → 20, `max_load_width` 4 → 8 B, `shared_ld` 30 → 15, `shared_st` 16 → 5, `mufu` (exp) 15 → 9, total 576 → 800 with 4 rows per CTA |

**Correctness** — `ctest --test-dir build-native -R
"infernix_cuda_stack_check|infernix_softmax_attention_test$|infernix_softmax_attention_extended_test$|infernix_sliding_window_attention_test$"
--output-on-failure`: **4/4 passed, 48.0 s** (`#228` 2.49 s, `#234` 32.45 s, `#28` 1.49 s,
`#235` 11.90 s). Note the task doc's own regex
(`-R "sliding_window_attention|context_softmax_attention|infernix_cuda_stack_check"`) resolves to
only 2 registered tests — the context route's oracle tests are registered as
`infernix_softmax_attention[_extended]_test`, so both were run explicitly.
The FP32 sliding-window merge is bit-identical by construction (same `split → lane` mapping, same
ascending-split accumulation, same final `numerator / global_l`); the BF16 context merge changes only
the `global_l` summation order and passes the FP32 oracle at test tolerance.

**Op latency** — per-shape interleaved A/B against the HEAD-built bench binaries (`--repeat 60`,
4 pairs per shape, order alternated per shape, median of pairs; `spread` = max−min across pairs).
`sw` = `infernix_sliding_window_attention_bench`, `cx` = `infernix_context_softmax_attention_bench`.

| W | T | L | base µs | new µs | Δ | spread b/n |
|---|---|---|---|---|---|---|
| 4096 | 2 | 128 | 13.06 | 12.02 | −7.97 | 1.98/0.13 |
| 4096 | 2 | 512 | 19.17 | 19.17 | +0.00 | 0.86/1.15 |
| 4096 | 2 | 1024 | 29.31 | 28.90 | −1.42 | 1.89/2.59 |
| 4096 | 2 | 2048 | 47.86 | 47.39 | −0.97 | 2.05/1.02 |
| 4096 | 8 | 128 | 14.05 | 12.19 | −13.21 | 0.03/0.51 |
| 4096 | 8 | 512 | 20.42 | 20.82 | +1.96 | 0.29/1.02 |
| 4096 | 8 | 1024 | 32.08 | 31.52 | −1.75 | 1.98/1.18 |
| 4096 | 8 | 2048 | 50.03 | 49.41 | −1.25 | 3.01/1.06 |
| 4096 | 16 | 128 | 14.19 | 14.34 | +1.01 | 0.16/2.05 |
| 4096 | 16 | 512 | 24.27 | 24.37 | +0.40 | 1.02/1.89 |
| 4096 | 16 | 1024 | 32.99 | 32.72 | −0.82 | 1.06/2.05 |
| 4096 | 16 | 2048 | 50.00 | 49.98 | −0.03 | 3.07/0.48 |
| 2048 | 2 | 128 | 13.15 | 12.03 | −8.52 | 1.70/0.10 |
| 2048 | 2 | 512 | 19.18 | 19.14 | −0.25 | 2.18/2.05 |
| 2048 | 2 | 1024 | 29.54 | 28.03 | −5.09 | 0.96/2.72 |
| 2048 | 2 | 2048 | 48.98 | 47.74 | −2.52 | 2.02/0.90 |
| 2048 | 8 | 128 | 14.05 | 14.03 | −0.11 | 1.66/1.82 |
| 2048 | 8 | 512 | 20.78 | 20.83 | +0.23 | 1.92/1.15 |
| 2048 | 8 | 1024 | 31.20 | 31.47 | +0.87 | 3.07/0.80 |
| 2048 | 8 | 2048 | 49.39 | 49.74 | +0.71 | 0.99/1.15 |
| 2048 | 16 | 128 | 14.10 | 14.19 | +0.68 | 0.19/0.35 |
| 2048 | 16 | 512 | 24.29 | 24.29 | +0.00 | 0.03/1.76 |
| 2048 | 16 | 1024 | 32.51 | 33.62 | **+3.40** | 0.83/1.86 |
| 2048 | 16 | 2048 | 50.91 | 50.43 | −0.94 | 1.22/2.02 |

| T | L | base µs | new µs | Δ | spread b/n |
|---|---|---|---|---|---|
| 2 | 128 | 12.27 | 12.22 | −0.39 | 2.05/0.45 |
| 2 | 512 | 19.70 | 20.13 | +2.19 | 0.83/1.89 |
| 2 | 2048 | 47.82 | 46.78 | −2.17 | 0.96/2.02 |
| 8 | 128 | 14.11 | 14.05 | −0.45 | 0.06/1.95 |
| 8 | 512 | 21.78 | 20.67 | −5.07 | 1.98/1.63 |
| 8 | 2048 | 50.86 | 47.89 | −5.85 | 0.45/1.89 |
| 16 | 128 | 18.13 | 15.31 | **−15.53** | 1.95/1.86 |
| 16 | 512 | 27.60 | 25.30 | −8.35 | 1.76/1.06 |
| 16 | 2048 | 56.53 | 56.53 | +0.00 | 4.03/3.04 |
| 1 | 32768 | 592.03 | 591.76 | −0.05 | 0.99/2.24 |
| 1 | 131072 | 2322.64 | 2322.37 | −0.01 | 8.48/93.70 |
| 2 | 32768 | 593.78 | 592.19 | −0.27 | 5.25/2.05 |
| 2 | 131072 | 2323.23 | 2321.17 | −0.09 | 58.85/16.06 |

Batched `sw` (`W = 4096`, `--batch`):

| T | B | L | base µs | new µs | Δ | spread b/n |
|---|---|---|---|---|---|---|
| 8 | 2 | 1024 | 48.03 | 48.83 | +1.67 | 3.20/2.98 |
| 8 | 2 | 2048 | 82.66 | 83.78 | +1.36 | 2.11/0.42 |
| 16 | 2 | 1024 | 58.08 | 59.42 | **+2.31** | 2.66/2.02 |
| 16 | 2 | 2048 | 94.27 | 95.04 | +0.81 | 0.74/2.85 |
| 8 | 4 | 1024 | 84.67 | 84.70 | +0.04 | 1.98/2.11 |
| 8 | 4 | 2048 | 155.62 | 153.41 | −1.42 | 4.22/3.23 |
| 16 | 4 | 1024 | 97.12 | 96.93 | −0.20 | 4.93/0.99 |
| 16 | 4 | 2048 | 173.76 | 173.76 | +0.00 | 2.05/3.01 |
| 8 | 8 | 1024 | 178.27 | 174.85 | −1.92 | 6.27/1.86 |
| 8 | 8 | 2048 | 327.65 | 330.34 | +0.82 | 3.90/2.72 |
| 16 | 8 | 1024 | 176.86 | 173.76 | −1.76 | 2.05/4.93 |
| 16 | 8 | 2048 | 335.55 | 331.62 | −1.17 | 4.54/1.92 |

Aggregates (geometric mean of the per-shape ratios): `sw` −1.55 % (24 shapes, −13.21…+3.40),
`cx` `L < 32768` −4.10 % (9 shapes, −15.53…+2.19), `cx` `L ≥ 32768` −0.11 % (4 shapes),
batched `sw` +0.04 % (12 shapes, −1.92…+2.31), all 49 shapes −1.52 %. Only 4 shapes have
`|Δ|` larger than their own observed run-to-run spread, and all four are wins
(`cx T=16 L=128/512`, `cx T=8 L=2048`, `sw T=8 L=128`).

**End-to-end (real artifact)** — the changed ops are reached from
`src/models/qwen3_5/execution/draft.cpp`, i.e. the DFlash2/MTP draft route, so `plain` mode does not
exercise them. `Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix` (+ its `.ngram`,
`--ngram-residency volume`, `INFERNIX_EXPERTS_ON_GPU=1`), bf16 KV,
`-p 8192 -n 512 -pg '8192,512' -r 2 --warmup 1 --concurrency 1`,
`--spec mtp --draft-tokens auto --lm-head-draft`; 3 rounds base-first + 3 rounds new-first against
the HEAD-built `infernix_bench`:

| test | prefill t/s base → new | decode t/s base → new | acceptance base → new |
|---|---|---|---|
| `pp8192` | 3829.3 → 3833.5 (+0.11 %) | — | — |
| `pp8192+tg512` | 3828.7 → 3824.2 (−0.12 %) | 121.2 → 121.2 (+0.05 %) | 0.984 → 0.984 (identical in all 12 arms) |
| `tg512` | — | 93.1 → 93.2 (+0.03 %) | 0.831 → 0.831 (identical in all 12 arms) |

**Acceptance criteria**

| Item | Verdict |
|---|---|
| `REG ≤ 168`, `STACK: 0` on both `*_split_partial_kernel` | **met** for every `KeyBlock = 32` instantiation (166–168, `STACK:0`); `KeyBlock = 64` deliberately left at 238–240 / 2 CTAs per §2.3 |
| Dead `q_positions` / `min_position` / `max_position` + `prime()` loop removed | **already satisfied at base** — that state never existed in this tree (§2.2); nothing to remove |
| Split-KV reduce vectorized, single pass | **met** — 0 block barriers, `uint2`/`float4` loads and stores, one warp per row |
| ≥ 10–20 % latency reduction across `T = 2..16`, `L = 128..2048` | **not met as a blanket bar** — see §5.2 |
| CTests + stack ratchet | **met** — 4/4 |

## 5. Regressions & adverse results

1. **First candidate spilled at `KeyBlock = 64`.** Clamping every split-partial instantiation to
   `MinBlocks = 3` produced new `STACK:40–72` on
   `context_attention_split_partial_kernel<9..16,3|4,64,0>` — a one-way-ratchet violation caught by
   `cuobjdump` before any timing was taken, so it never reached a measurement or the tree. Fixed by
   gating the clamp on `KeyBlock` (§2.3); final `STACK:0` on all 128 instantiations.
2. **The doc's ≥ 10–20 % bar is not met.** It holds only where the merge dominates the Op:
   `cx T=16 L=128` −15.53 %, `sw T=8 L=128` −13.21 %, `sw T=2 L=128` −7.97/−8.52 %,
   `cx T=8/16 L=512…2048` −5.1…−8.4 %. Group means are −1.55 % (`sw`), −4.10 % (`cx` short),
   −0.11 % (`cx` long context) and +0.04 % (batched). At `L ≥ 32768` the merge is amortized and the
   route is unchanged (−0.01…−0.27 %), which is the expected null result, not a win.
3. **The predicted 1-wave band did not win.** `T = 16`, `L = 1024` is the grid the residency change
   targets (§2.5): measured −0.82 % (`B=1`), −0.20 % (`B=4`), −1.76 % (`B=8`) and **+2.31 %**
   (`B=2`) — all inside their own spreads. Extra residency therefore did not translate into a
   measurable gain there; the band stays unconfirmed rather than refuted (§6).
4. **Worst individual positives**: `sw W=2048 T=16 L=1024` **+3.40 %**, batched `B=2` band
   +0.81…+2.31 %, `cx T=2 L=512` +2.19 %, `sw T=8 L=512` +1.96 %. None exceeds its measured
   run-to-run spread, and the batched group mean is +0.04 %, so there is no group-level regression;
   the `B=2` band is the weakest region found and is disclosed rather than dismissed as noise.
5. **The reduce kernel is larger per CTA**: `REG` 47→48 (context) and 48–56→61–64 (sliding-window),
   static `SHARED` 1152→1364 / 1536→2384 B (`weights[warp][split]`), and +38.9 % static instructions
   for 4 dimensions per lane. This pays off because each CTA now merges 4 rows with 0 barriers, but
   a grid whose last CTA merges fewer than 4 rows does relatively more work per row.
6. **An earlier sweep-ordered A/B reported +15–17 % "regressions"** at `W = 2048`, `L = 2048`.
   Isolated interleaved re-runs of the same shapes measure ≈ 0 % (base 47.9–49.9 µs vs new
   48.9–50.1 µs). The cause was running each binary as a whole sweep (ordering + thermal drift) and
   taking a median over only 2 passes; that harness was discarded and all reported numbers come
   from the per-shape interleaved driver.
7. **A one-off `cudaErrorIllegalAddress`** aborted `infernix_context_softmax_attention_bench`
   (`bench/ops/context_softmax_attention_bench.cu:384`, `cudaStreamSynchronize`) on the **baseline**
   binary immediately after linking. Three identical reruns passed, and no later run reproduced it;
   it was on the HEAD binary, so it is not attributable to this change, but it is not explained
   either.
8. **E2E prefill initially looked like a regression**: base-first ordering gave −0.41 % / −0.47 %
   prefill t/s. Reversing the order per round flipped the sign (new-first: new 3836–3853 vs base
   3789–3827). Pooled across both orderings the effect is +0.11 % / −0.12 % — i.e. run-order/thermal
   bias, not a change. The single-order cut is reported here because it was measured first.
9. **Noise floor**: run-to-run spread is 1–3 µs on the 12–50 µs shapes and up to 93.7 µs on the
   2.32 ms `T=1 L=131072` shape (58.8 µs on `T=2 L=131072`), so readings below ~1 % at those shapes
   are not resolvable and are treated as neutral, not as wins or losses.
10. **Two task-doc premises were already false at base** (§2.1, §2.2): `__launch_bounds__` was
    already `MinBlocks = 2` (not 1), and the dead position-register state never existed. The
    occupancy premise only held for the 4-warp class, and the "dead state removed" acceptance line
    is vacuous, which is part of why the measured gain is smaller than the doc predicts.

## 6. Accepted gaps / follow-ups

- **No full KV-dtype matrix.** The E2E evidence is one artifact (`F4-Local NVFP4-W4Dense`), one KV
  dtype (bf16), `plain` + `mtp`, both arm orderings. The `dense8` arm could not run as scripted:
  `bench/run_kv_matrix_bench.sh`'s `ND` path
  (`…NVIDIA-NVFP4-Dense8-Infernix.ngram`) does not exist on this host — the file is
  `…Dense8-Infernix-00001-of-00003.infernix.ngram`. That stale path is a bench-script bug outside
  T3's contract and was left for its owner rather than patched mid-measurement.
- **No old-vs-new output bit diff.** These benches time the Op and do not dump activations, so
  equivalence rests on construction (sliding-window FP32 path: identical `split → lane` mapping,
  accumulation order and final division) plus the FP32-oracle tolerance tests for the context route,
  whose only numeric change is the `global_l` summation order (128-thread tree → `warp_sum<32>`).
  The MTP acceptance rate being identical in all 12 E2E arms is indirect support, not a bit diff.
- **No NCU trace.** The SASS comparison plus interleaved A/B answered the kernel question; the
  synthetic-input-only rule for `ncu` was respected by not profiling at all.
- **`KeyBlock = 64` stays at 2 CTAs/SM.** 3 CTAs there needs ≤ 170 registers from 238–240, which
  ptxas only reaches by spilling (§5.1); raising that band needs a register-cost cut in the MMA body
  (accumulator staging), not a launch bound.
- **The `T = 16`, `L = 1024` one-wave hypothesis is unresolved.** Deciding whether that band is
  residency-bound or dependency-latency-bound needs an `nsys` wave-fill trace on the real route;
  the op bench alone cannot separate them.
- **Empty `SlidingWindowContextPolicy::prime()` hook retained**: the interface requires it because
  `ContextAttentionPolicy::prime` is live (block-table preload), so there is no dead hook to drop.

---

# GB10 (`sm_121a`) Branch Changelog — 2026-11-06 — T4: Remove the FP32 head buffer and full-vocabulary cast from `Forward::mtp_block`; route the unscreened MTP draft head through `ops::linear` BF16

Scope: task T4 from `docs/tasks/20261106-performance-optimization-tasks.md` on GB10 (`sm_121a`):
stop the unscreened `Forward::mtp_block` draft head from always allocating FP32 `[rows, T]` logits and
casting the whole vocabulary down to BF16, and consider `ops::linear_topk` (or a direct BF16
`Forward::project()`) as the fused replacement.

## 1. Accounting

- **Base commit**: `34751c38e` (`dev-rebased-20261028`, T3). Worktree was clean at session start.
- **Files changed** (all `M`, 3 files, +63 / −19):
  `src/models/qwen4_exp/execution/forward.cpp` (+47 / −6),
  `src/models/qwen4_exp/execution/forward.h` (+4 / −0),
  `src/models/qwen4_exp/execution/forward_lossless.cpp` (+12 / −13, a move, no body change).
  `src/models/qwen4_exp/execution/dflash2.cpp` was audited and **not** changed (§2.3).
- **Allowlist delta**: none. No `.cu`/`.cuh` was touched, so no stored symbol changed its
  `REG`/`STACK`/`LOCAL`; `infernix_cuda_stack_check` passes against the unchanged
  `tools/cuda_stack/cuda_stack_allowlist.json` and the one-way ratchet was never run.
- **Rebuild count**: 2 single-invocation Ninja builds (candidate with the fused route, final after the
  fused route was removed), each sharing the `-rdc` device link; plus the 4 test-binary targets the
  gate runs, built in one further invocation.

## 2. Task-premise corrections / Architectural rationale ("Why" on GB10 `sm_121a`)

1. **The names in the task text are not this tree's.** `Forward::mtp_block` has no `p_.lossless` and
   no `out.logits`; the live flags are `config_.lossless` and the `MtpCall` buffers. The doc's
   `sampled` / `greedy` booleans are `call.sample_configs != nullptr` and
   `call.draft_logprobs.data != nullptr`. The gate below uses the real names.
2. **There is no FP32 → BF16 → FP32 round-trip.** The base code formed FP32 logits
   (`ops::projection_fp32` over the head's rows) and cast them once to BF16
   (`ops::cast_fp32_to_bf16`), and every consumer read BF16 — nothing went back to FP32. What was
   actually redundant is the `[rows, T]` FP32 buffer plus the whole-vocabulary cast kernel: the draft
   head's consumers (`ops::argmax`, `ops::top_logprobs`, `ops::speculative_sample_sparse_drafts`)
   only *select* from these rows, so the FP32 resolution was destroyed by the cast before its first
   read. `Forward::logits` was also not part of this — it still forms FP32 logits on every route.
3. **`dflash2.cpp` has no such round-trip to remove.** Its non-fused `head_candidates` runs
   `ops::projection_fp32` and hands the FP32 logits straight to `ops::top_logprobs` (no BF16 consumer),
   and its fused route already calls `ops::linear_topk` whenever `config_.fused_head` is planned. No
   change was needed or made there.
4. **`ops::linear_topk` structurally cannot serve most of the MTP head routes.** Its scores are the raw
   top-K logits; `call.sample_configs` and `call.draft_logprobs` need log-probabilities
   (`logit − logsumexp` over the full row), which the Op never forms. Only the pure-argmax route
   (`call.draft_logprobs.data == nullptr`, `call.sample_configs == nullptr`) was a candidate.
5. **The Flash-Next proposal head has no `ops::linear` BF16 schedule.** `src/ops/linear/q4/q4_dispatch.cpp`
   registers `n = 131072` only at the `k` it was tuned at (2048, 3072, 5120); the Q4_G64_FP16
   `[131072, 2560]` proposal head is at none of them, so it cannot take the direct-BF16 route either.
   `ops::linear_column_invariant_route(qtype, n, k)` is used as the public capability predicate, so the
   gate cannot drift from the dispatch tables it asks about.
6. **Why the saving is invisible end to end on this host.** At the text head's
   `V = 248 320` rows and `T = 1`, the removed path is 4 V B written + 4 V B read + 2 V B written =
   2.5 MB of extra traffic per column ≈ 9 µs at the GB10 UMA rate, inside a draft round that measures
   41 ms mean (`execution.decode.mean_ms`, `tg384`, `auto` arm) — ≤ 0.03 % of the round, below this
   host's interleaved-A/B resolution (§5.3). The draft step is CUDA-graph-captured, so the removed
   kernel launch was not a launch-cost saving either.
7. **The doc's `infernix_qwen4_exp_mtp_test` / `infernix_linear_topk_test` / `infernix_mtp_pack_test`
   are CTest case names, not Ninja targets** (the recurring task-doc defect). The real binaries are
   `infernix_ops_sampling_test`, `infernix_qwen3_5_mtp_draft_policy_test`,
   `infernix_qwen3_5_mtp_confidence_policy_test` and `infernix_qwen3_5_dflash2_tree_greedy_real_test`;
   the doc's CTest regex was run as given.

## 3. Subsystem breakdown

- `src/models/qwen4_exp/execution/forward.h` (`M`) — declares `bf16_rows(const Tensor&)` at namespace
  scope: a BF16 `[K, N]` tensor of `N` rows as a native `Weight [N, K]` view of the same memory.
- `src/models/qwen4_exp/execution/forward_lossless.cpp` (`M`) — that helper's definition moves out of
  the anonymous namespace to namespace scope (byte-identical body); the anonymous namespace now opens
  after it. No lossless behaviour, arithmetic or route changes.
- `src/models/qwen4_exp/execution/forward.cpp` (`M`)
  - `text_head_bf16()` (`A`) — the text head's rows projected straight to BF16: `ops::ggml_linear` for
    a GGUF import, `ops::linear` with the Q8 head's own policy for a Q8 head, `ops::linear` with
    `A16Only` over `bf16_rows(output_head)` for a BF16 head. Mirrors `text_head_logits`'s three-way
    head selection, `dense_routes()` passed through.
  - `linear_projects_head_rows()` (`A`) — whether `ops::linear` has a schedule for a draft head's rows
    (§2.5); a GGUF-import head is not a Linear route and stays on the FP32 projection.
  - `Forward::mtp_block`'s unscreened head section (`M`) — the FP32 buffer + cast now run only when
    `config_.lossless` or the head is one Linear cannot project; otherwise the `[rows_out, T]` BF16
    tensor is filled by `project()` (proposal head) or `text_head_bf16()` (text head) in one kernel.
    The screened branch, the sampled-draft chain, the `draft_logprobs` chain, both `argmax` arms and
    the whole lossless route are untouched.

## 4. Verification

**Build** (final, one Ninja invocation so the `-rdc` device link runs once):

```text
ninja -C build-native -j "$(nproc)" infernix-serve infernix_ops_linear_test infernix_bench   # EXIT=0
ninja -C build-native -j "$(nproc)" infernix_qwen3_5_mtp_draft_policy_test \
      infernix_qwen3_5_mtp_confidence_policy_test infernix_qwen3_5_dflash2_tree_greedy_real_test \
      infernix_ops_sampling_test                                                             # EXIT=0
```

**Required CTest gate** — `ctest --test-dir build-native -R "mtp|linear_topk|infernix_cuda_stack_check"
--output-on-failure` (serial; `--parallel` produces numerical failures on this single-GPU host):

```text
100% tests passed, 0 tests failed out of 12   (57.5 s)
The following tests did not run:
        126 - infernix_qwen3_5_mtp_tree_greedy_real_test       (Skipped)
        127 - infernix_qwen3_5_mtp_tree_batch_greedy_real_test (Skipped)
```

`infernix_cuda_stack_check`, `infernix_linear_topk_test`, `infernix_mtp_pack_test`,
`infernix_mtp_round_test`, both `speculative_mtp_*` one-hot/distribution pairs (FP32 and BF16 logits)
and the two draft/confidence policy tests all pass. The two real-artifact tree tests skip because their
artifact environment variables are unset in this shell.

**Real-artifact draft-distribution oracle** — `infernix_qwen4_exp_speculative_sampling_real_test
--trials 48 --tokens 8` (IQ3_S artifact, `INFERNIX_QWEN4_KV=bf16`, `INFERNIX_EXPERTS_ON_GPU=1`): every
MTP check passes — one-hot drafts draft (581 drafts / 48 requests) and match plain sampling; sampled
drafts draft (660 / 48), are both accepted and rejected, match plain sampling, and both greedy prompts
give the one-hot tokens. The run's only 2 FAILs are the test's own power self-check
(`plain T 0.8 differs from plain sampling (power)`, least p 0.00238 against
`alpha = 1e-3/(tokens+1) = 1.11e-4`), which needs the default 2 000 trials and compares two **plain**
decode arms that never enter `mtp_block`; the same 2 FAILs appeared in the pre-change probe run. The
process exits 1 on them, so the CTest registration (default trials) was not overridden here.

**End-to-end A/B** — `infernix_bench` HEAD vs patched, round-robin interleaved, 2 passes with the arm
order reversed in the second, IQ3_S artifact, bf16 KV, `-p 2048 -n 384 -pg '2048,384' -r 2 --warmup 1
-c 1`, all 16 runs exit 0. `auto` = `--spec mtp --draft-tokens auto --lm-head-draft`,
`autotext` = `auto` without `--lm-head-draft`, `lmhead4` = `--draft-tokens 4 --lm-head-draft`,
`texthead` = `--draft-tokens 4`. Note the flag reads backwards from what it does: `--lm-head-draft`
binds the *optimized proposal head* (`config.proposal_head = optimized`, the Q4 `[131072, 2560]` rows),
while omitting it drafts with the text head (`proposal_head = full`) — verified from each report's
`config.proposal_head`. Only `autotext` and `texthead` run the new BF16-direct head; `auto` and
`lmhead4` run the unchanged FP32 route on the Q4 head and serve as controls.

| arm | decode t/s p1 | decode t/s p2 | decode mean | total t/s mean | rounds / drafted / accepted (both binaries, both passes) |
|---|---|---|---|---|---|
| `auto` (control) | 72.55 → 72.57 (+0.03 %) | 72.25 → 72.29 (+0.06 %) | 72.40 → 72.43 (**+0.05 %**) | 72.11 → 72.16 | 224 / 646 / 510 — identical |
| `autotext` (new) | 67.25 → 67.51 (+0.39 %) | 67.28 → 67.21 (−0.11 %) | 67.26 → 67.36 (**+0.14 %**) | 67.05 → 67.14 | 212 / 640 / 520 — identical |
| `lmhead4` (control) | 64.53 → 64.73 (+0.32 %) | 64.59 → 64.40 (−0.28 %) | 64.56 → 64.57 (**+0.02 %**) | 64.35 → 64.35 | 250 / 998 / 518 — identical |
| `texthead` (new) | 59.85 → 60.13 (+0.47 %) | 60.02 → 59.88 (−0.24 %) | 59.94 → 60.00 (**+0.12 %**) | 59.78 → 59.84 | 240 / 958 / 528 — identical |

`pp2048+tg384` decode: `auto` 84.53 → 84.48 (−0.05 %), `autotext` 76.77 → 76.97 (+0.26 %),
`lmhead4` 79.65 → 79.65 (+0.01 %), `texthead` 73.10 → 73.17 (+0.10 %). Acceptance rate is identical in
every arm and pass (0.7895 / 0.8125 / 0.5190 / 0.5511 decode, 0.8841 / 0.8957 / 0.7625 / 0.7844 pp+tg).
`git diff --check` clean.

## 5. Regressions & adverse results

1. **The fused `ops::linear_topk` route was implemented, measured slower, and removed.** On the only
   route it can serve on this artifact (the Q4_G64_FP16 `[131072, 2560]` head, plain argmax drafts,
   the `lmhead4`/`fused` arm), the 20-run interleaved A/B of the earlier candidate gave: decode tg384
   64.55 → 64.30 (−0.40 %; passes −0.14 % and −0.66 %), `pp2048+tg384` decode 79.66 → 78.45
   (**−1.52 %**; passes −1.36 % and −1.69 %, both passes agreeing), total t/s −1.02 %. Its acceptance
   also moved on the long-context arm: 0.7625 → 0.7520, drafted 758 → 766, accepted 578 → 576, rounds
   190 → 192 — the fused Op's top-1 differs from `argmax` of the BF16-cast logits on near-ties, so it
   is not merely slower but changes the draft. AGENTS.md registers a fast path "only where it is
   measured faster", so the branch, its `kFusedHeadCandidates` capacity constant and the
   `linear_topk` include were deleted; the surviving code comments that decision where the route lives.
   The DFlash2 fused route that already ships (`config_.fused_head`) is untouched and was not measured.
2. **A real crash the candidate introduced and the smoke test caught.** With the fused branch in place,
   `--spec mtp --draft-tokens 4 --lm-head-draft` aborted with
   `qwen4_exp/mtp: argmax: row_ids must be contiguous I32 [physical_rows]` where HEAD succeeded: the
   fused case wrote `call.drafts` but control still fell through to the downstream
   `ops::argmax(narrow, head.token_ids, …)` on an unallocated `narrow`. Fixed by making the fused case
   the head of the selection chain; after the branch was removed the arm was re-run on the final binary
   and matches HEAD exactly (exit 0, identical rounds/drafted/accepted in both passes).
3. **The shipped direct-BF16 route is not measurably faster.** Its two arms move +0.14 % and +0.12 %
   mean decode with the sign flipping between the interleaved passes (+0.39/+0.47 % in pass 1,
   −0.11/−0.24 % in pass 2), and the two control arms that run *unchanged* head code show the same
   pass-order pattern (+0.05 %, +0.02 %). The honest reading is: within noise, no measured gain, for the
   reason quantified in §2.6. It is kept because it is strictly less work (one 2.5 MB-per-column buffer
   and one whole-vocabulary kernel out of the captured graph), not because it was measured faster.
4. **Prefill moved −0.38 % … +0.14 % mean across the four arms** (`pp2048`: `auto` −0.33 %,
   `autotext` −0.17 %, `lmhead4` +0.14 %, `texthead` −0.38 %), with mixed signs across passes and
   including arms whose head code is byte-identical to HEAD. Prefill never executes `mtp_block`, so
   this is unattributed host/thermal drift rather than a regression from the change; it is disclosed
   because it is not explained either.
5. **The oracle test exits 1 at `--trials 48`** on its own two power self-checks (§4). Not a regression
   from this change — plain-decode arms only, and the same 2 FAILs appear in the pre-change probe run —
   but it does mean the fast probe is not a green signal, and the default-trials run (~40× the cost)
   was not executed here.

## 6. Accepted gaps / follow-ups

- **The Q4_G64_FP16 `[131072, 2560]` proposal head still runs FP32 + cast.** No `ops::linear` BF16
  schedule exists at that `K` (§2.5); admitting one is a `q4_dispatch` registration/qualification task
  with its own oracle and A/B, outside this route's contract. Registering `linear_topk` as the MTP
  replacement is likewise not admissible at the measured result (§5.1).
- **`config_.lossless` keeps the exact FP32 head rows**, and a GGUF-import head keeps
  `ops::projection_fp32` (it is not a Linear route). Both were re-verified as live branches, not just
  asserted.
- **The sampled / log-probability draft routes on a BF16-projectable head now normalise over BF16
  logits instead of FP32.** That is a fast path in the numerical-acceptance sense: it changes the
  softmax normalizer's operand precision, is covered by the real-artifact distribution oracle (all MTP
  checks pass, greedy drafts bit-identical to the one-hot arm) and by the byte-identical draft counts
  in both A/B arms, but was not qualified against an FP64 oracle at the Op level.
- **`Forward::mtp_workspace_bytes` still reserves 6 bytes per vocabulary row** (FP32 + BF16). The FP32
  route stays live on the same lanes (lossless, Q4, GGUF), so the upper bound is correct; the
  direct-BF16 lanes over-reserve by 4 bytes/row.
- **No `nsys`/`ncu` trace of the draft step.** The A/B plus the arithmetic bound in §2.6 answered the
  question (the removed traffic is ≤ 0.03 % of a draft round); a trace would not change any decision.
- **`--spec mtp` on a `screen/*` artifact (`screened == true`) is unexercised locally.** No delivered
  artifact here carries a head screen, so that branch is only shown unchanged by reading, not run.
