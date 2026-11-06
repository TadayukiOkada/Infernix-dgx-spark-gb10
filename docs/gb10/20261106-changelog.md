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
