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
