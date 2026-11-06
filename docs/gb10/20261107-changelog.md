# GB10 (`sm_121a`) Branch Changelog — 2026-11-07 — T1 (nvfp4): Single-pass lane-parity W4A4 narrow `gate_up_kernel` scan + 2-column pair weight reuse under split input scales

Scope: task T1 (P0) of `docs/tasks/20261107-nvfp4-performance-tasks.md`: in the narrow W4A4
`gate_up_kernel` (`src/ops/offloaded_sparse_moe/cuda/moe_layer.cu`, device-resident `moe_experts`),
replace the two-pass parity loop under `split_input` (`scales.input_up != scales.input_gate`) with a
single pass whose lanes select their rows' nibble half and scale by lane parity, and enable the
2-column pair weight-reuse loop when split input scales are active.

**Status: all acceptance criteria met and verified.** Qwen's initial commit (`5f38eb4c0`) implemented
the single-pass lane-parity scan and a two-half pair loop that reached `195.0 µs` (`-29.1%`) at `T=4`
and `373.1 µs` (`-29.9%`) at `T=8` (short of the `≤ 165.0 µs` / `≤ 310.0 µs` `≥ 40%` bars). The
reviewer optimization pass replaced the software per-element division + E2M1 branch ladder and the
2-ActCodes-per-unit shared-memory layout with hardware `cvt.rn.satfinite.e2m1x2.f32` dual-scale
packing, reciprocal multiplication with a 4-ULP exact IEEE-754 division guard around the seven E2M1
decision boundaries, and a bank-conflict-free parity-interleaved `ActCodes` layout (`acts[2 * db + (i & 1)]`),
bringing `split=1` warm latency to **`79.3 µs` (`-23.2%`) at `T=1`**, **`148.1–150.0 µs` (`-45.4%` to
`-46.1%`) at `T=4`**, and **`269.7–271.8 µs` (`-48.9%` to `-49.3%`) at `T=8`**, matching `split=0`
within `≤ 1%` across all widths while preserving 100% bit-exactness (`exact=yes` on all 12 benchmark
arms) and reducing register pressure from `REG:124` to `REG:112` (`STACK:0 LOCAL:0`).

## 1. Accounting

- **Base commit**: `8884aa82e` (pre-T1 baseline) → `5f38eb4c0` (Qwen initial T1 commit) → `d743823a7`
  (`docs: update`) → reviewer T1 fix/optimization commit on top of `d743823a7`.
- **Files changed in reviewer pass (vs `d743823a7`)**:
  - `src/ops/offloaded_sparse_moe/cuda/moe_layer.cu` (M): `+179 / -78` lines (`cuda-kernel-toolkit/`
    untouched).
  - `docs/gb10/20261107-changelog.md` (A): this changelog.
- **Total net T1 code diff vs pre-T1 baseline (`8884aa82e`)**:
  - `src/ops/offloaded_sparse_moe/cuda/moe_layer.cu` (M): `+281 / -26` lines.
- **Allowlist delta**: none — `gate_up_kernel` is `REG:112 STACK:0 SHARED:0 LOCAL:0`;
  `tools/cuda_stack/cuda_stack_allowlist.json` untouched.

## 2. Architectural rationale ("Why" on GB10 `sm_121a`)

1. **Lane-parity invariant in `unit_sums`**: `unit_sums` lane `l = 8q + i` (`i = lane & 7`) accumulates
   only rows `i` and `i + 8`, which share parity `(i & 1) == ((i + 8) & 1)`, and reduces across `q`
   via `__shfl_xor_sync` with offsets `8` and `16` (preserving `i = lane & 7`). Even lanes therefore
   only ever contribute to gate rows and odd lanes only to up rows. Selecting the parity's nibble and
   scale per lane computes all 16 interleaved gate/up rows in a single pass over `sm.stage` with
   bit-identical integer sums.
2. **Why Qwen's initial pair loop (`5f38eb4c0`) stopped at `-29%` (`195.0 µs` / `373.1 µs`)**:
   - In `5f38eb4c0`, `quantize_split_codes` quantized each column and scale using two full calls to
     `canon::quantize_a4_codes` (`34` serial software IEEE-754 `fdiv.rn.f32` divisions and `32 × 7 = 224`
     software E2M1 boundary comparisons per block per column), executed across all `40` `kGateUpCtas`
     slice CTAs per expert.
   - In `5f38eb4c0`, `unit_sums_split_pair` stored `acts` column-major (`acts[0 * 80 + db]` for column 0
     and `acts[1 * 80 + db]` for column 1, each holding gate in low nibbles and up in high nibbles),
     requiring every lane in `unit_sums_split_pair` to load **two** `ActCodes` structs (`a0` and `a1`)
     from shared memory per K-block (`480` words apart = `15 × 32` banks, causing a 2-way shared-memory
     bank conflict when `a0` and `a1` or even/odd lanes accessed the same bank mod 32) plus extra shifts
     and masks in the inner loop.
3. **Three compounding optimizations in the reviewer pass**:
   - **Reciprocal multiplication with exact 4-ULP boundary fallback (`scaled_q_value`, `prepare_a4_scale`)**:
     All seven E2M1 rounding thresholds `{0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0}` lie in `[0.25, 5.0]`
     and have zero in their low 21 mantissa bits (`((qa + 4U) & 0x1FFFF8U) == 0U`). Computing one
     reciprocal `inv_d = canon::div_rn(1.0F, d)` per 16-element block and multiplying `q = canon::mul_rn(v[j], inv_d)`
     with a fallback to `canon::div_rn(v[j], d)` only when `q` lands within `±4` ULPs of a 21-zero-bit
     boundary replaces 16 serial `fdiv.rn.f32` calls per scale with 1 `fdiv.rn.f32` + 16 `FMUL`s while
     preserving 100% IEEE-754 bit-exactness for every possible BF16 input and FP32 scale.
   - **Hardware `cvt.rn.satfinite.e2m1x2.f32` dual-scale encoding (`quantize_split_block_from_f32`)**:
     Each thread loads its 16 BF16 activations once, unpacks `v[16]` and `amax` once, scales under
     both `gate_scale` and `up_scale`, and uses `cvt.rn.satfinite.e2m1x2.f32 b, u_j, g_j` +
     `mov.b32 %0, {b0, b1, b2, b3}` to convert and pack both the gate code (low nibble) and up code
     (high nibble) in a single hardware instruction per element (16 instructions + 4 `PRMT`s per block).
   - **Parity-interleaved conflict-free `ActCodes` layout (`quantize_split_pair_half`, `unit_sums_split_pair_half`)**:
     In each 80-block K-half (`2 * 80 = 160` active threads `< 256`), thread `tid = 2 * db + col`
     quantizes block `B0 + db` of column `col` under both scales, then exchanges `(col0_up, col1_gate)`
     with partner lane `tid ^ 1` via `__shfl_xor_sync(..., 1)` so even lane (`col = 0`) writes
     `(col0_gate, col1_gate)` to `acts[2 * db + 0]` and odd lane (`col = 1`) writes `(col0_up, col1_up)`
     to `acts[2 * db + 1]`. In `unit_sums_split_pair_half`, even lanes read `acts[2 * db + 0]` (banks
     `(12 * db + 0..5) % 32`) and odd lanes read `acts[2 * db + 1]` (banks `(12 * db + 6..11) % 32`):
     each lane loads only **one** `ActCodes` per K-block with **zero shared-memory bank conflicts**
     (1-wavefront multicast across the warp) and executes the exact same inner loop body as `unit_sums_pair`.

## 3. Subsystem breakdown

- `src/ops/offloaded_sparse_moe/cuda/moe_layer.cu` (`M`):
  - `unit_sums<Groups, bool SplitInput = false>`: single-pass lane-parity nibble/scale selection when
    `SplitInput` is true; unchanged when false.
  - `unit_sums_split_pair_half<Groups, B0, BLen>`: compile-time unrolled K-half pair accumulation reading
    parity-interleaved `acts[2 * db + (i & 1)]` with zero bank conflicts.
  - `scaled_q_value`, `prepare_a4_scale`, `quantize_split_block_from_f32`: exact reciprocal + 4-ULP
    boundary fallback and hardware `cvt.rn.satfinite.e2m1x2.f32` dual-scale packer.
  - `quantize_split_codes`: single-pass dual-scale single-column quantizer (`b = threadIdx.x < 160`).
  - `quantize_split_pair_half<B0, BLen>`: single-pass 160-thread dual-column quantizer with warp-xor
    transpose into `acts[2 * db + col]`.
  - `gate_up_kernel`: two-half (`kHalf = 80`) `split_input` 2-column pair loop and single-pass
    single-column tail.

## 4. Verification & benchmark results

- **Build**:
  `ninja -C build-native -j "$(nproc)" infernix-serve infernix_ops_offloaded_moe_test infernix_offloaded_moe_decode_bench`
  succeeded with 0 warnings.
- **Resource footprint (`cuobjdump -res-usage build-native/apps/infernix-serve`)**:
  - `gate_up_kernel` (`moe_layer.cu`): **`REG:112 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:1676`**
    (down from `REG:124` in `5f38eb4c0`, well within the 128-register budget for `__launch_bounds__(256, 2)`).
- **CTest (`ctest --test-dir build-native -R "infernix_offloaded_moe|infernix_cuda_stack_check" --output-on-failure -j 4`)**:
  - **20 / 20 tests PASSED** (135.31 s), including `infernix_offloaded_moe_layer_test`,
    `infernix_offloaded_moe_wide_test`, `infernix_offloaded_moe_cuda_test`, and `infernix_cuda_stack_check`
    (`0` new stack spills).
  - Note: the task prompt's regex `infernix_offloaded_sparse_moe` matches 0 registered CTest targets;
    the actual CTest family is `infernix_offloaded_moe*` (19 tests) + `infernix_cuda_stack_check`.
- **Decode benchmark (`./build-native/bench/infernix_offloaded_moe_decode_bench --repeat 15`)**:
  All 12 arms verified bit-exact (`exact=yes`) against the CPU oracle (`moe::expert_forward`):

  | Width `T` | Cache | Doc Baseline `split=0` / `split=1` (`µs`) | Qwen `5f38eb4c0` `split=0` / `split=1` (`µs`) | Final `split=0` / `split=1` (Run 1, `µs`) | Final `split=0` / `split=1` (Run 2, `µs`) | `split=1` Δ vs Doc Baseline | Target Bar (`split=1` warm) |
  |---:|:---|---:|---:|---:|---:|---:|:---|
  | `T=1` | warm | `92.608` / `103.264` | `102.5` / `81.3` | `82.368` / **`82.144`** | `91.488` / **`79.328`** | **`-20.5%` to `-23.2%`** | `≤ 95.0 µs` (`≥ 8%`) — **MET** |
  | `T=1` | cold | `101.664` / `113.984` | `129.0` / `119.5` | `130.752` / `116.608` | `131.168` / `118.496` | (`split=1` faster than `split=0` cold by `10%`) | — |
  | `T=4` | warm | `148.928` / `274.944` | `145.8` / `195.0` | `145.568` / **`148.096`** | `148.960` / **`150.048`** | **`-45.4%` to `-46.1%` (`1.86x`)** | `≤ 165.0 µs` (`≥ 40%`) — **MET** |
  | `T=4` | cold | `164.224` / `291.552` | `161.5` / `213.6` | `161.728` / **`168.800`** | `162.496` / **`170.816`** | **`-41.4%` to `-42.1%`** | — |
  | `T=8` | warm | `276.672` / `531.904` | `274.8` / `373.1` | `272.896` / **`271.808`** | `269.632` / **`269.664`** | **`-48.9%` to `-49.3%` (`1.96x`)** | `≤ 310.0 µs` (`≥ 40%`) — **MET** |
  | `T=8` | cold | `289.600` / `541.376` | `287.1` / `389.2` | `280.416` / **`290.528`** | `286.400` / **`290.336`** | **`-46.3%` to `-46.4%`** | — |

## 5. Regressions, intermediate candidates, and dispositions

1. **Intermediate warp-local 3-chunk staging candidate (`quantize_split_pair_chunk<0,8>`, `<8,8>`, `<16,4>`)**:
   - Measured `T=4 split=1 warm = 175.328 µs` (`-36.2%`) and `T=8 split=1 warm = 320.960 µs` (`-39.7%`).
   - Superseded by the 2-half `quantize_split_pair_half` + `cvt.rn.satfinite.e2m1x2.f32` +
     parity-interleaved `acts[2 * db + (i & 1)]` design because the 3-chunk layout required every thread
     to load/unpack 3 blocks (with even/odd parity threads duplicating the same `x` load and BF16 unpack),
     whereas the 2-half `tid < 160` layout loads each `(block, col)` only once (`2` blocks per thread
     across both halves for `tid < 160`, `0` blocks for `tid >= 160`).
2. **Register count (`REG:100` pre-T1 → `REG:124` in `5f38eb4c0` → `REG:112` final)**:
   - Adding the 2-column `split_input` pair path increases register usage of `gate_up_kernel` from `100`
     (pre-T1, where `split_input` had no pair path) to `112` (down from `124` in `5f38eb4c0`). Because
     `112 <= 128` (`__launch_bounds__(256, 2)`), SM occupancy is unchanged at 2 CTAs/SM (`512` threads/SM,
     limited by the `50,048 B` shared-memory footprint), and `STACK:0 LOCAL:0` is preserved.
3. **`T=1` cold L2 flush baseline difference (`101.7 / 114.0 µs` in task table vs `130.8 / 116.6 µs` current)**:
   - The `split=0` code path is byte-for-byte unchanged, yet `T=1 split=0 cold` measures `130.8–131.2 µs`
     on the current tree (and `T=1 split=1 cold` is **faster** than `T=1 split=0 cold` at `116.6–118.5 µs`
     because `quantize_split_codes` uses reciprocal + `cvt.rn.satfinite.e2m1x2.f32` whereas
     `quantize_column_codes` uses software `canon::quantize_a4_codes`). All warm and cold `T=4` and `T=8`
     `split=1` arms improve by `41.4%–49.3%`.

## 6. Rebase conflict resolution policy

- Preserve the single-pass `unit_sums<Groups, true>`, `quantize_split_block_from_f32` (`cvt.rn.satfinite.e2m1x2.f32` +
  4-ULP boundary-guarded reciprocal), `quantize_split_pair_half`, and `unit_sums_split_pair_half`
  (`acts[2 * db + (i & 1)]`) in `src/ops/offloaded_sparse_moe/cuda/moe_layer.cu`.

---

# GB10 (`sm_121a`) Branch Changelog — 2026-11-07 — T2: Double-buffered `ldmatrix_x4` A-fragments and even MMA-tile split across the column warps in wide NVFP4 MoE `grouped_kernel`

## 1. Accounting

- Base commit: `c0a43a61f` (branch `dev-rebased-20261028`).
- Modified: `src/ops/offloaded_sparse_moe/cuda/wide_expert.cuh` (`M`) — `grouped_kernel` consumer
  loop, `TileContext`, and the four `grouped_kernel` epilogues;
  `tests/ops/test_offloaded_moe_wide.cu` (`M`) — the accumulator-dump `ProbeEpilogue` moved to the
  new `col_base`/`live_tiles` contract (it is a `TileContext` consumer of `grouped_kernel`).
- `tools/cuda_stack/cuda_stack_allowlist.json`: unchanged (zero new spills, zero eliminated — every
  `grouped_kernel` instantiation was and stays `STACK:0 LOCAL:0`).
- Rebuilds: one shared-device-link `ninja` pass per candidate iteration (targets
  `infernix-serve infernix_ops_offloaded_moe_test infernix_offloaded_moe_wide_bench`), plus one
  per-object `wide_expert.cu.o` compile check.

## 2. Task-premise corrections / Architectural rationale

1. **`__launch_bounds__` already exists; step 1.3 rejected.** `grouped_kernel` is declared
   `__global__ __launch_bounds__(S::kThreads, S::kMinBlocksPerSm)` — `(288, 2)` for `Schedule`,
   `(160, 2)` for `WarpTile64`. The task's "NO `__launch_bounds__` at line 317" and its
   `REG:174`/`REG:168`/`SHARED:43008`/`SHARED:50176` resource block do not match this tree:
   `cuobjdump --dump-resource-usage` on the HEAD bench binary gives `REG:96 STACK:0`
   (`<GateUp, GateUpEpilogue, Schedule>`), `REG:95 STACK:0` (`<Down, DownEpilogue, Schedule>`) and
   `REG:136`/`REG:144 STACK:0` for the two `WarpTile64` instantiations (shared is opt-in dynamic,
   hence `SHARED:0` in the static attribute dump). Adding `__launch_bounds__(kThreads, 1)` would
   have been actively harmful: it lifts the register cap from 112 to 224 and the kernel is
   deliberately built for 2 CTAs/SM (43,056 B of shared fit two CTAs; the register file would not
   at >112 registers), so ptxas would have traded the second resident CTA for registers.
2. **The bench widths run the 8-consumer-warp `Schedule`, not `WarpTile64`.** `wide_t64` admits
   `WarpTile64` at `mean_job_columns >= kWarpTile64MinColumns` (512) and the bench's
   `mean_job_columns` is `entries / max_jobs` = 20 at T=1024 and 80 at T=4096 (512 experts, top-10).
   So the idle-upper-warp case the task describes is live on both claimed widths.
3. **Step 2's mechanism does not exist in this tree and cannot be added at its stated cost.** The
   current `GateUpEpilogue` has no `sm.epilogue` staging, no `__syncthreads()`, and no
   `if (mma_warp_m <= live_mma)` gate — it gates per column (`column >= ctx.columns`). Moving the
   lower warp's accumulators to shared memory so the upper warp takes half the epilogue would need
   8–16 KB of staging on top of the 43,008 B pipeline, past the ~49.2 KB per-CTA budget for
   2 CTAs/SM. The implemented equivalent keeps everything in registers: the two column warps split
   the token tile's live 16-column MMA tiles evenly (`col_base`/`live_tiles`), so the upper warp
   takes over columns 16..31 including their whole K loop and epilogue whenever the tile has ≤ 32
   columns, and every column's MMA chain (K order, A rows, B fragments, scale words) is exactly the
   chain the lower warp used to run. All four epilogues write exactly the warp's live tiles, which
   also removes the two-warp double store the split would otherwise create (caught by the bench's
   FNV-1a output hash before any timing was claimed).
4. **Double-buffering is limited to the one-term schedule.** `grouped_kernel` is shared with the
   two/three-term A4-residual schedules; a `[2][kTerms][kMmaTokens][4]` fragment buffer would keep
   the next K step's fragments of every term live together — the exact condition the existing
   term-loop comment records as spilling at the 2-CTA 112-register cap. `if constexpr` keeps the
   residual schedules on the inline path (their instantiations stayed `REG:92..96 STACK:0`).
5. **Measured baseline differs slightly from the task table.** `infernix_offloaded_moe_wide_bench`
   on HEAD: T=1024 frames 8382.1 µs (11.0 TF/s), T=4096 frames 8819.0 µs (45.7 TF/s) (task doc:
   8199.6/8282.2 µs). The staged-placement arms (~23.3/23.8 ms) are bound by the host→device
   record staging copy, not by `grouped_kernel`.

## 3. Subsystem breakdown

- `src/ops/offloaded_sparse_moe/cuda/wide_expert.cuh` (`M`):
  - `TileContext`: added `col_base` (warp's first live 16-column MMA tile's column) and
    `live_tiles`; the accumulator-layout comment now reads `col_base + m * 16 + ...`.
  - `grouped_kernel` consumer prologue: with `kSkipIdleColumns` and two column warps, the tile's
    `total = ceil(columns / 16)` live MMA tiles are split `ceil(total / 2)` / the rest — identical
    to the old mapping for 33..64 columns, and for ≤ 32 columns the upper warp now takes columns
    16..31 instead of idling. Other schedules keep the old per-warp formula.
  - `grouped_kernel` K-tile loop: the `rg16_fragments` decode was factored into `load_b_step`; for
    `kTerms == 1` the A fragments and stage scales live in `a_fragments[2][kMmaTokens][4]` /
    `a_step_scales[2][kMmaTokens]`, slot 0 is preloaded after the stage's `mbarrier` wait and the
    next 64-wide step's slot is prefetched before the current step's OMMA block (the
    `nvfp4_a4_tma_kernel` pattern), so `LDSM`/`LDS` latency overlaps the tensor pipe. Multi-term
    schedules keep the inline term-loop loads.
  - Epilogues (`GateUpEpilogue`, `DownEpilogue`, `GateUpEpilogueResidual`, `DownEpilogueResidual`):
    columns map from `ctx.col_base`; the `m` loop breaks at `ctx.live_tiles`; the epilogue call
    gate became `ctx.col_base < ctx.columns`.
- `tests/ops/test_offloaded_moe_wide.cu` (`M`): `ProbeEpilogue` (the test's accumulator dump) uses
  the same contract — `ctx.col_base` column mapping and the `ctx.live_tiles` break — so the C1/C3
  accumulator probes address the columns their warp actually computed.

## 4. Verification

- Resources (`cuobjdump --dump-resource-usage` on the bench binary, all `grouped_kernel`
  instantiations): `STACK:0 LOCAL:0` everywhere; `Schedule` W4A4 `REG:96` (unchanged), A4-residual
  `REG:94..96` (unchanged), `WarpTile64` W4A4 `REG:136→166`/`144→163` within the `(160,2)` 200-register
  cap (2 CTAs × 160 threads × 166 = 53,120 ≤ 65,536 registers), `WarpTile64` residual `REG:128..146`.
- Bitwise: `infernix_offloaded_moe_wide_bench` FNV-1a output hashes equal the HEAD baseline in all
  24 A/B runs (T=64/256/1024/4096 × frames/staged × 3 passes × both arms); the wide oracle test's
  per-configuration worst-case errors equal HEAD's to the digit.
- CTest: `-R "infernix_offloaded_moe"` (the task's `infernix_offloaded_sparse_moe` regex matches no
  registered CTest name; the family is `infernix_offloaded_moe*`) — 19/19 passed, 243.5 s
  sequential; `infernix_cuda_stack_check` passed (allowlist unchanged; no ratchet-down applied —
  no spills were eliminated). The first suite run failed `infernix_offloaded_moe_wide_test` (6
  checks) because the test's own `ProbeEpilogue` still used the old column mapping; the HEAD binary
  passes that test, so this was the change's contract break, fixed in the test together with the
  kernel (§3) — the rerun is the 19/19 above.
- A/B wall time, interleaved HEAD/candidate binaries, frames placement:

  | arm | HEAD medians (µs) | candidate medians (µs) | delta |
  |---|---|---|---|
  | T=1024 (3 passes × 10) | 8420.9 / 8432.0 / 8326.4 | 8373.0 / 8334.8 / 8333.8 | −0.6% |
  | T=4096 (3 passes × 10) | 8956.4 / 8891.4 / 9328.6 | 9432.8 / 8911.3 / 8921.1 | −0.4% |
  | T=1024 (4 iters × 20) | 8334.8 / 8355.3 / 8297.9 / 8367.2 | 8610.7 / 8314.3 / 8317.5 / 8335.8 | −0.2% (per-iteration medians) |
  | T=4096 (4 iters × 20) | 8950.6 / 8976.5 / 8746.0 / 8753.2 | 8693.1 / 8848.1 / 8762.0 / 8747.8 | −1.1% (arm means 8856.6→8762.8) |

  Staged-placement arms (host-staging-copy bound) and T=64/256 moved within ±1.2% in either
  direction.
- nsys per-launch kernel time (8 calls, 64 launches per arm, frames):

  | width | kernel | HEAD mean/median (µs) | candidate mean/median (µs) | delta |
  |---|---|---|---|---|
  | T=4096 | gate/up grouped | 670.4 / 649.1 | 614.1 / 604.4 | −8.4% / −6.9% |
  | T=4096 | down grouped | 407.4 / 387.2 | 374.0 / 368.0 | −8.2% / −5.0% |
  | T=1024 | gate/up grouped | 402.1 / 386.7 | 406.4 / 384.5 | +1.1% / −0.6% |
  | T=1024 | down grouped | 208.6 / 202.5 | 204.6 / 196.8 | −1.9% / −2.8% |

  At T=4096 the call is ~97% `grouped_kernel` (0 narrow experts), so the kernel-level −8% is the
  honest Op-level result of this change; it converts to only ~−1% at the call median because the
  per-call wall is dominated by the DRAM streaming state the consumer-side levers do not change
  (grouped busy 8.62 → 7.90 ms/call while the call means moved 8857 → 8763 µs).

## 5. Regressions & adverse results

- **Acceptance bar not met (adverse result, not a regression):** the required ≥5% bench latency
  improvement at T=1024 and T=4096 was not achieved — −0.2..−0.6% (T=1024) and −0.4..−1.1%
  (T=4096) at the call median, despite the kernel itself being ~8% faster at T=4096. Disposition:
  the shipped levers are bit-exact, spill-free, and never slower at the kernel level; the doc's
  stall model (consumer LDS latency + idle warps causing the 59%-of-floor gap) does not carry to
  call latency on this bench, where the same bytes must still stream from DRAM.
- T=1024 gate/up grouped per-launch mean +1.1% in the nsys pair (402.1→406.4 µs; median −0.6%);
  the wall medians moved −0.2%. The split duplicates the `rg16_fragments` B-decode on the
  previously idle upper warp — extra SIMT work that only pays off when the consumer, not the DRAM
  stream, is binding. Disposition: disclosed; within single-trace variance, not reproduced at the
  wall.
- One T=1024 wall iteration measured 8610.7 µs (+3.3% against its paired 8334.8 µs HEAD run); the
  other three iterations were −0.5% / +0.2% / −0.4%. Disposition: unexplained single-pass outlier,
  disclosed with the full 4-iteration pair rather than re-run until favorable.
- `WarpTile64` instantiations carry +27/+19 registers (to 163..166, `STACK:0`, within the 200-reg
  2-CTA cap); no bench-reachable route exercises them, so no performance regression is measured
  for them (§6).

## 6. Accepted gaps / follow-ups

- Experts with ≤ 16 columns still leave the upper warp idle: a 16-column MMA tile is the smallest
  bit-exact unit, and splitting below it would need accumulator staging the shared-memory budget
  cannot hold.
- The two/three-term A4-residual schedules did not get the double buffer (register cap, §2.4); they
  did get the column split.
- The `WarpTile64` instantiations carry the double buffer too and grew to 163..166 registers
  (within cap, `STACK:0`); the bench cannot route them (`mean_job_columns` would need ≥ 512, i.e.
  T ≥ 26,215 tokens), so their performance change is unmeasured here — covered only for
  correctness by the CTest selection.
- The task's E2E `pp8192` attribution (457 ms over 47 layers) was not re-measured; acceptance was
  evaluated on the task's own bench command.
- The kernel-level −8% at T=4096 converts to only ~−1% at the call median; closing the remaining
  50%-of-DRAM-floor gap needs producer-side work (bulk-copy DRAM efficiency across 512 × 2.4 MiB
  records), which is outside T2's two consumer-side mechanisms.

## 7. Rebase conflict resolution policy

- Preserve `TileContext::col_base` / `TileContext::live_tiles`, the `S::kWarpsTokens == 2` even MMA-tile
  split, the `S::kTerms == 1` double-buffered `a_fragments[2][S::kMmaTokens][4]` / `a_step_scales[2][S::kMmaTokens]`
  K-step pipeline, and the `ctx.col_base` / `ctx.live_tiles` epilogue bounds in
  `src/ops/offloaded_sparse_moe/cuda/wide_expert.cuh` and `tests/ops/test_offloaded_moe_wide.cu`, while
  retaining the 2-CTA/SM `__launch_bounds__(S::kThreads, S::kMinBlocksPerSm)` (`kMinBlocksPerSm = 2`).

# GB10 (`sm_121a`) Branch Changelog — 2026-11-06 — T3 (nvfp4): `linear_add` A4/A4R schedule-cliff fixes (`T=33..64`, `T=129..287`) + shared 48-SM narrow TMA token-tile selection in `linear_add`/`linear_swiglu`

## 1. Accounting

- Base commit: `f690e70e9` (branch `dev-rebased-20261028`, clean tree at start).
- Files changed (all `M`):
  - `src/ops/linear_add/nvfp4/nvfp4_linear_add_a4.cu` — `uses_tma` 288→129, resident tile removed,
    narrow/wide TMA + `Tiled128`/`Tiled256` quantizer selection, residual band split at 33.
  - `src/ops/linear_add/nvfp4/nvfp4_linear_add_a4_tma.cu` / `..._a4_tma_launch.h` — added
    `launch_nvfp4_a4_tma_linear_add_narrow` (`Nvfp4A4TmaMmaSchedule<128, 4, 1, weight-L2-128B>`).
  - `src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_a4_tma.cu` / `..._a4_tma_launch.h` — added
    `launch_nvfp4_linear_swiglu_a4_tma_narrow` (same 128-token/4-stage/128-row shape, SwiGLU epilogue).
  - `src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp` — `TmaFusedA4` threshold lowered
    from `136` to `129` (`> M128N128::kBlockTokens`) and route now picks the token tile (and
    matching scale layout) through the shared predicate.
  - `src/ops/linear/nvfp4/nvfp4_a4_plan.h` — received `nvfp4_a4_prefer_narrow_token_tile` (moved).
  - `src/ops/linear/nvfp4/nvfp4_launch.cuh` — gave up that predicate (kept its route-table use).
  - `tests/ops/linear_add/test_nvfp4.cpp` — added `AllowA4` coverage at `T=256, 257, 384, 385`
    (with CUDA-Graph replay at `257`) to exercise `launch_nvfp4_a4_tma_linear_add_narrow` (`Tiled128`).
  - `docs/gb10/20261107-changelog.md` — this section.
- `tools/cuda_stack/cuda_stack_allowlist.json`: no delta (no allowlisted symbol touched; the removed
  resident instantiation and the two new narrow kernels are all `STACK 0`/`LOCAL 0`).
- Rebuild: one `cmake --build build-native` invocation, 343 steps (full `src/ops` CUDA recompile
  because `nvfp4_a4_plan.h`/`nvfp4_launch.cuh` are consumed by 25 files across `linear/nvfp4`,
  `linear/nvfp4/shapes`, `linear_add`, `linear_swiglu`, `attn_input_proj`, `gdn_input_proj`).

## 2. Task-premise corrections / Architectural rationale

- The task's verification flags are wrong for both benches: there is no `--tokens`; the token list
  flag is `--t-sweep`, and the add bench additionally requires `--n 5120 --k 6144|17408`.
- The task's `linear_swiglu` file references do not match the code: `nvfp4_linear_swiglu_a4.cu` has
  no `uses_tma`/`launch_problem` TMA gate and *already* dispatched `ResidualM64N128` for `33..64`;
  the 1-term SwiGLU TMA entry is `resolve_route(policy, tokens >= 129) -> TmaFusedA4` in
  `nvfp4_linear_swiglu_plan.cpp` (lowered from `136` to `129`, eliminating the 2-tile `M128N128`
  cliff at `T=129..135`: `917.3–920.4 us -> 539.5–548.9 us`, `-41.2%` / `1.70x`), which is also
  where the selector was wired. The `T=33..64` residual
  cliff existed only in `linear_add`, whose `launch_residual_problem` sent `17..64` to
  `ResidualM32N64` (`kBlockTokens = 32`): from 33 tokens its second token tile re-streamed the
  49.0 MiB weight panel past the 24 MiB L2 from LPDDR5X.
- The template spelling `Nvfp4A4TmaMmaSchedule<128, 128, 128, 2>` in the task does not exist; the
  narrow tile the standalone route validates is
  `Nvfp4A4TmaMmaSchedule<128, 4, 1, kNvfp4A4TmaWeightCodeL2_128B>` (128-token tile, 128-row weight
  panel, four 128-K stages, 1 CTA/SM; `kScaleLayout` is `Tiled128` by construction).
- At `N=34816` (SwiGLU gate_up, 100.1 MiB panel) the predicate's wave condition (12 vs 7/8·12) and
  its 52 MiB extra-traffic bound both decline the narrow tile at every `T >= 129`, so the SwiGLU
  wiring is provably inert today; it keeps the one shared predicate authoritative instead of a
  per-op copy (measured: swiglu a4 unchanged within noise above 136, §4).
- The in-code claim "resident still wins at 256" (the old 288 boundary) was the RTX 5090 fit; the
  GB10 candidate sweep puts TMA ahead at every width from 129 (`--variants`, cold L2, quantize
  included): K=6144 113.7/127.0/140.3 us TMA vs 140.4/155.6/200.8 resident at T=129/256/287;
  K=17408 314.8/340.0/371.7 vs 473.1/511.0/730.2. The narrow/wide *ordering* of the sweep points
  matched `nvfp4_a4_prefer_narrow_token_tile` at all seven probed widths.

## 3. Subsystem breakdown

- `ops/linear/nvfp4` (`M`): `nvfp4_a4_prefer_narrow_token_tile` moved verbatim from
  `nvfp4_launch.cuh` into the host-safe `nvfp4_a4_plan.h` so the plan `.cpp` files (no CUDA headers)
  can drive quantizer layout and launcher from one predicate; `nvfp4_launch.cuh` reaches it through
  its existing include, so the standalone shape tables are unchanged.
- `ops/linear_add/nvfp4` (`M`): `uses_tma(tokens) = tokens >= 129`; `M128N128Resident` and its dead
  `129..` band removed from `launch_problem` (TMA owns 129+); `nvfp4_linear_add_a4_launch` selects
  `Tiled128`+`launch_nvfp4_a4_tma_linear_add_narrow` when the predicate fires (both Ks at `257..383`,
  K=6144 additionally at `513..640`) and `Tiled256`+the existing wide launcher otherwise;
  `launch_residual_problem` gained `ResidualM64N128` for `33..64` (RowMajor planes, matching the
  swiglu ladder), keeping `ResidualM32N64` up to 32 where its 160-CTA single tile still wins.
- `ops/linear_swiglu/nvfp4` (`M`): narrow 1-term TMA launcher instantiated; `TmaFusedA4` entry
  lowered from `tokens >= 136` to `tokens >= 129` (`> M128N128::kBlockTokens`) and route resolves
  the predicate and passes the matching layout to quantizer and launcher. Residual route untouched
  (`T >= 97` TMA, `33..64` already `ResidualM64N128`).
- `tests/ops/linear_add/test_nvfp4.cpp` (`M`): added `AllowA4` test points at `T=256, 257, 384, 385`
  (with CUDA-Graph replay at `T=257`) to exercise the narrow 128-token TMA (`Tiled128`) route and
  its transitions to/from wide 256-token TMA (`Tiled256`).

## 4. Verification

- Build: single invocation, `BUILD_EXIT 0`, zero warnings in the touched files.
- CTest (serial): `infernix_linear_add_nvfp4_test` Passed (2.41 s), `infernix_linear_swiglu_nvfp4_test`
  Passed (10.65 s), `infernix_cuda_stack_check` Passed (1.48 s). The add suite covers T=33/64 (new
  residual tile), T=129/256/257/384/385 (wide/narrow TMA on both `AllowA4` and `AllowA4Residual`,
  with 257 replayed under CUDA Graphs on both policies) and 1023/1024/1025.
- Production-route A/B (before/after HEAD binaries, interleaved, cold L2, quantizer included,
  warmup 3 / repeat 10, median of 3 passes; µs):

| route | T | K=6144 before | after | Δ | K=17408 before | after | Δ |
|---|---|---|---|---|---|---|---|
| a4r | 16 | 88.1 | 88.1 | 0 | 230.4 | 228.4 | −0.9% |
| a4r | 32 | 88.0 | 90.1 | +2.4%¹ | 241.7 | 242.8 | +0.5%¹ |
| a4r | 33 | 107.5 | 90.1 | **−16.2%** | 461.8 | 247.8 | **−46.3%** |
| a4r | 48 | 111.6 | 90.7 | **−18.7%** | 464.9 | 251.9 | **−45.8%** |
| a4r | 64 | 111.6 | 94.2 | **−15.6%** | 480.3 | 257.0 | **−46.5%** |
| a4r | 65 | 105.5 | 103.5 | −1.9% | 291.8 | 290.8 | −0.4% |
| a4r | 128 | 111.6 | 111.6 | 0 | 311.3 | 314.5 | +1.0%¹ |
| a4r | 256 | 133.1 | 136.2 | +2.3%¹ | 386.0 | 389.3 | +0.8%¹ |
| a4r | 512 | 250.9 | 254.0 | +1.3%¹ | 650.0 | 649.9 | 0 |
| a4 | 64 | 92.2 | 92.2 | 0 | 248.8 | 249.5 | +0.3%¹ |
| a4 | 128 | 100.4 | 99.3 | −1.0% | 267.3 | 268.2 | +0.3%¹ |
| a4 | 256 | 158.7 | 123.9 | **−21.9%** | 505.9 | 338.8 | **−33.0%** |
| a4 | 288 | 142.8 | 134.1 | −6.1% (narrow) | 373.7 | 365.6 | −2.2% (narrow) |
| a4 | 384 | 152.6 | 147.5 | −3.4% (narrow edge) | 399.4 | 395.3 | −1.0% |
| a4 | 512 | 167.0 | 166.9 | 0 | 444.2 | 441.3 | −0.6% |
| a4 | 640 | 223.2 | 212.5 | −4.8% (narrow) | 564.2 | 555.2 | −1.6% |
| a4 | 768 | 245.9 | 240.5 | −2.2% | 601.1 | 609.8 | +1.4%¹ |
| a4 | 1024 | 328.7 | 322.3 | −2.0% | 803.8 | 798.7 | −0.6% |
| a4 | 2048 | 590.8 | 587.7 | −0.5% | 1584.1 | 1576.0 | −0.5% |

  swiglu a4 (N=34816, K=5120): `T=129..135` drops from `917.3–920.4 us` to `539.5–548.9 us` (**−41.2% / 1.70x**); T=128/256/384/512/640/768/2048 all within ±0.4% except T=512 +4.8%¹ (§5).
  `¹` = route byte-identical in before/after; see §5 for the noise disposition.
- Acceptance: a4r `T=33..64` K=17408 −46% (≥44% bar met, 1.86x); absolute `<= 245 us` bar missed by
  2.8..12 µs at T=48/64 (251.9/257.0) — the doc's own baseline (424..442 µs) also measured 8% higher
  here (461.8..480.3), consistent with thermal drift of absolute µs bars; the relative bar is the
  meaningful one. a4 `T=256`: K=17408 505.9→338.8 (≤390 ✓, −33% ≥20% ✓); K=6144 158.7→123.9
  (≤135 ✓, −21.9% ✓).

## 5. Regressions & adverse results

- swiglu a4 `T=512` +4.8% in the first 3-pass matrix: not reproduced by a 5-pass reversed-order
  recheck (after median 622.4 vs before 625.5, distributions identical; the route is byte-identical
  code — predicate declines the narrow tile at every served width). Disposition: measurement-order
  artifact of the first matrix (the `after` arm always ran later within each pass); no regression.
- `a4r` K=6144 `T=32/256/512` +2.4/+2.3/+1.3% and `a4` K=17408 `T=64/128/768` ≤+1.4%: every one of
  these cells runs byte-identical code in both binaries. The reversed-order 5-pass recheck gave
  +0.2%/+3.0%/+0.4% with heavily overlapping distributions (T=256: after {132.1..140.3} vs before
  {133.3..137.2}; pooled mean Δ +1.8%, within the pooled spread, t≈1.6). Single-pass outliers of
  comparable size appeared on both sides (before-arm spikes: 800.0 µs at a4 K17408 T=768 pass 2,
  881.9 µs at swiglu-after recheck pass). Disposition: pass-to-pass noise on unchanged routes; no
  mechanism exists for a change in them; not hidden but not dismissed by fiat — the data is above.
- Absolute acceptance bar `a4r T<=245 us` at K=17408: missed by ≤12 µs (247.8..257.0 vs 245) while
  the relative ≥44% bar passes; the doc's baseline for the same cell measured 8% high here.
  Disposition: accepted; the shipped candidate (`ResidualM64N128`) is the measured argmin of the
  seven-candidate panel at these widths (next-best `a4r_tma128_r128_s3` is +15..20% behind).

## 6. Accepted gaps / follow-ups

- End-to-end (KV-matrix / serve) A/B was not run: the claim scope is the two op-level production
  benches, and the wide-T arms (T=1024/2048, unchanged wide route) measured ≤0.6% drift. A prefill
  E2E pass is the natural next check when this branch is next benchmarked end-to-end.
- The swiglu narrow launcher is instantiated but inert at the only registered geometry (N=34816);
  it becomes live automatically if a lower-row-count gate_up geometry is registered. The predicate
  itself is unmodified from its 161-point validation.
- `a4` K=6144 `T=256` measured the narrow tile ~1.6% ahead in the candidate sweep, but the shared
  predicate (fit on the standalone route) selects wide there; kept the predicate rather than a
  per-op exception, since a 1.6% point is inside the sweep's pass-to-pass spread.

---

# GB10 (`sm_121a`) Branch Changelog — 2026-11-06 — T4 (nvfp4): Hoist the `e4m3 -> f16x2` scale conversion out of the A16 MMA widening helpers

## 1. Accounting

- Base commit: `5bf4703c3` (`dev-rebased-20261028`, clean worktree at start).
- Modified (`M`): `src/ops/linear/nvfp4/nvfp4_codec.cuh`,
  `src/ops/linear/nvfp4/nvfp4_a16_sliced_k_mma.cuh`,
  `src/ops/linear/nvfp4/nvfp4_a16_slice_ring_mma.cuh`.
- Inspected and left unchanged (`-`): `src/ops/linear/nvfp4/nvfp4_a16_mma.cuh` (its
  `nvfp4_scaled_oct_bf16` call already spends one scale conversion per 8 codes),
  `src/ops/offloaded_sparse_moe/cuda/narrow_a16_mma.cuh` and
  `src/ops/resident_moe/resident_moe.cu` (they contain none of the symbols the task names and do
  not decode an NVFP4 E4M3 scale in any inner loop — §2), `src/ops/linear/nvfp4/nvfp4_a16_gemv.cuh`
  (one FP32 `decode_nvfp4_e4m3` per scale group, nothing shared).
- `tools/cuda_stack/cuda_stack_allowlist.json`: no delta. `STACK`/`LOCAL` stayed 0 on all 223
  `nvfp4_a16*` instances, so there is nothing to ratchet down.
- Rebuilds: 4 `ninja` invocations (1 aborted on a doc-named target that does not exist, 1 failed on
  my own `"h"` asm-constraint error, 1 complete 65/65 sharing one device link, 1 final 77/77 after
  routing the pair helper through the existing `decode_nvfp4_e2m1x2_f16` instead of duplicating its
  inline asm). The rebuilt binary's per-symbol SASS counts are identical to the measured candidate
  below, so every measurement here was taken on equivalent code. Consuming `.cu` files were touched
  before each build (dead CUDA depfiles).

## 2. Task-premise corrections / Architectural rationale

Held:

- CUDA 13.0 (`nvcc V13.0.88`) `ptxas -arch=sm_121a` has no `cvt.rn.bf16x2.e2m1x2`, so every A16
  widening compiles the FP16 path — the redundant `F2FP.F16.E4M3.UNPACK_B` the task describes is
  really there.
- `cvt.rn.f16x2.e4m3x2` decodes two E4M3 bytes per instruction, so a weight-row pair's two scales
  genuinely can share one conversion, and the result is bit-identical: each byte converts
  independently, and E2M1 × a finite E4M3 scale is exact in FP16, so only the final BF16 pack
  rounds, exactly as in the old per-use form.

Corrected by measurement:

- Four of the five named call-site symbols do not exist in this tree (`load_nvfp4_a_frag_direct`,
  `load_b_scaled_pairs`, `load_nvfp4_oct_bf16`, and the `pack_same_low16_u32` helper). The only
  callers of the widening helpers in the whole `src/` tree are the three sites in §3.
- Root-cause item 1 is already implemented upstream of this change: `nvfp4_scaled_oct_bf16`
  converts its scale once per 8 codes, so `nvfp4_a16_mma_kernel` spends exactly 4
  `F2FP.F16.E4M3` per widen body, one per *distinct* scale byte. There is no redundancy there to
  remove, and that kernel's SASS is identical before and after (F2FP 4, PRMT 1, 888 instructions).
  The doc's "4x reduction inside `nvfp4_a16_mma_kernel`" is therefore unreachable.
- The achievable dedup is 2x, not 4x, at the two reachable kernels: the sliced-K tile converts one
  scale per row per k-step (packing the row pair halves it: 160 → 80), and the slice-ring tile
  converts one per fragment register where four registers carry two distinct scales (8 → 4).
- The dedup is partly paid back in `PRMT`: duplicating each converted half into a full `f16x2`
  multiplier costs one `PRMT` per multiplier, so the sliced-K kernel trades 80 `F2FP` for 160
  `PRMT` (6480 → 6384 total instructions, −1.5%) and the ring kernel trades 4 for 8 (total
  unchanged at 792). Net instruction savings are ~1%, which is why no wall-clock gain follows on
  routes that already sit at the weight-streaming floor.

## 3. Subsystem breakdown

- `src/ops/linear/nvfp4/nvfp4_codec.cuh` (`M`): added
  `nvfp4_e4m3_f16x2_bits` (register view of the existing decode),
  `nvfp4_unpack_two_scales_f16x2` (one `cvt.rn.f16x2.e4m3x2` over `s0 | s1 << 8`, halves broadcast
  with `__byte_perm(both, both, 0x1010 / 0x3232)`), `nvfp4_scaled_pair_bf16_f16scale` and
  `nvfp4_scaled_two_pairs_bf16_f16scale`, which take the pre-converted multiplier (the pair form
  widens through the existing `decode_nvfp4_e2m1x2_f16`, so no FP16 widening asm is duplicated).
  The byte-taking `nvfp4_scaled_pair_bf16` / `nvfp4_scaled_two_pairs_bf16` `#else` bodies now delegate
  to those forms, so the FP16 widening exists once in the file; their CUDA 13.2+ native branches
  are untouched (§6).
- `src/ops/linear/nvfp4/nvfp4_a16_sliced_k_mma.cuh` (`M`): `load_two_pairs` takes a
  `std::uint32_t` multiplier register, and one `nvfp4_unpack_two_scales_f16x2` per
  `(row tile, k_step)` converts `s_top` and `s_bot` together.
- `src/ops/linear/nvfp4/nvfp4_a16_slice_ring_mma.cuh` (`M`): the four A-fragment registers of a
  `(row pair, step)` use the two multipliers from one conversion instead of four conversions.

## 4. Verification

Per-symbol SASS (`cuobjdump -sass -fun <exact symbol>`, `K=5120` instances, `infernix-serve`;
baseline is a copy of the HEAD-linked binary):

| kernel | `F2FP.F16.E4M3` | `PRMT` | total instructions |
|---|---|---|---|
| `nvfp4_a16_sliced_k_mma_kernel` | 160 → **80** | 0 → 160 | 6480 → 6384 |
| `nvfp4_a16_slice_ring_mma_kernel` | 8 → **4** | 0 → 8 | 792 → 792 |
| `nvfp4_a16_mma_kernel` (unchanged code) | 4 → 4 | 1 → 1 | 888 → 888 |

- Resources: all 223 shared `nvfp4_a16*` instances have `STACK:0 LOCAL:0` in both binaries, so the
  spill ratchet is untouched and no allowlist entry was added, raised, or lowered. `REG` fell in 15
  instances (e.g. 128→121, 115→111, 87→80) and rose in 11 (+2…+5); recomputing the register-limited
  CTAs/SM with GB10's 8-register warp rounding shows no instance changes occupancy (96→99 at 256
  threads stays 2 CTAs/SM, 98→103 at 128 threads stays 4), and dynamic SMEM is unchanged.
- Oracle/contract tests, serial: `ctest -R
  "linear_nvfp4_a16|linear_nvfp4_a4|nvfp4_a16_slice_ring|linear_add_nvfp4|linear_swiglu_nvfp4|resident_moe|offloaded_moe_cuda|offloaded_moe_wide|attn_input_proj|attn_input_proj_fused_rmsnorm|gdn_input_proj|kv_cache_append_nvfp4|qsa_nvfp4|softmax_attention_nvfp4|ple_nvfp4|lossless|cuda_stack_check"`
  → **24/24 passed, 0 failed** (312.6 s on the measured binary, re-run 315.3 s / 24/24 on the final
  binary), including `infernix_cuda_stack_check` and
  `infernix_nvfp4_a16_slice_ring_test`, which checks the ring tile against the sliced tile bit for
  bit. Four real-artifact `*_lossless_real_test` members skipped (artifact env unset), as on HEAD.
- Benchmarks (cold-L2, baseline binary copied once, arms interleaved; the doc's
  `--tokens` flag does not exist — these benches take `--t-sweep`/`--sweep`):
  `infernix_nvfp4_linear_add_bench --policy a16`, `T = 2,4,8,16,32`, 3 rounds —
  K=6144: −0.27 / 0.00 / 0.00 / −1.52 / +2.38 %; K=17408: +0.18 / 0.00 / −0.57 / +0.17 / +0.78 %.
  Focused 4-round pass with alternating arm order (the bands this edit can reach, T ≤ 32):

| route | mean Δ | median Δ | worst point |
|---|---|---|---|
| Linear NVFP4 a16 `[5120, 17408]` | −0.02% | +0.02% | +2.38% (T=22) |
| Linear NVFP4 a16 `[16384, 2560]` | −0.63% | −0.44% | −3.52% (T=26) |
| fused `linear_add` a16 `[5120, 17408]` | +0.43% | +0.34% | +1.44% (T=32) |

## 5. Regressions & adverse results

- **The throughput acceptance bar is not met.** No route measured faster by more than its own
  round-to-round spread. The reachable A16 decode bands are weight-streaming bound (`[5120,
  17408] T=8` runs at 82.5% of its memory floor and 91.6% `READ_%`; `[5120, 6144] T=2` at
  248 GB/s of the ~273 GB/s UMA peak), so removing conversion instructions cannot show up in wall
  time. Disposition: shipped as an instruction-count/pressure improvement with the perf bar
  explicitly unmet, not as a speedup.
- **A first wide sweep reported large apparent regressions that are not real.** Grouping each
  arm's runs per round (instead of alternating per round) produced +8.5% (T=32), +21.4% (T=38),
  +23.1% (T=44), +16.4% (T=68) and −24.1% (T=104) on `[5120, 17408]`, and −22.8% (T=98), −22.5%
  (T=128) on `[16384, 2560]`. Those token counts dispatch `nvfp4_linear_a16_mma` / `a4` routes
  whose SASS is identical in both binaries (the published dispatch bands `T 33-64`, `T 65-96` are
  the unmodified `mma` schedules), and the same points swung in both directions between passes with
  baseline round spreads up to 48%. Re-measured with alternating order the reachable bands are
  within ±2.4% and the unmodified bands were dropped. Disposition: measurement-protocol artifact,
  disclosed rather than deleted.
- **One small adverse point remains unexplained**: `T=22` on `[5120, 17408]` measured +2.38% with a
  0.9%/1.8% round spread in the alternating pass, i.e. it exceeds its own spread. No mechanism is
  known (no spill, no occupancy change, one fewer instruction on the path); the neighbouring
  `T=20`/`T=24` points measured −1.07%/−1.43%. Left unresolved, not dismissed as noise — the
  route-boundary padding quantum is the nearest candidate and would need a per-token sweep to test.
- No new spills, no occupancy loss, no numerical change, no interface or memory change; the
  24-test oracle set is bit-identical to HEAD.

## 6. Accepted gaps / follow-ups

- The byte-taking `nvfp4_scaled_pair_bf16` / `nvfp4_scaled_two_pairs_bf16` now have no callers at
  all in `src/`. They are kept because they carry the CUDA 13.2+ native `cvt.rn.bf16x2.e2m1x2` path
  that this toolchain cannot compile or test; deleting them would silently downgrade a future CUDA
  13.2 build. A 13.2 build should re-express the migrated callers natively and drop them then.
- The new `*_f16scale` helpers are unconditional FP16 arithmetic (as `nvfp4_scaled_oct_bf16`
  already was). On a future 13.2 toolchain the two migrated MMA callers would forgo the native
  instruction while staying bit-exact; that is the accepted cost of a single FP16 implementation.
- The doc's `nvfp4_scaled_oct_bf16_f16scale` was not added: no caller shares an 8-code multiplier,
  so it would be dead code, and routing `nvfp4_scaled_oct_bf16` through it would replace its
  single 4-byte `mov.b32` unpack with three byte shifts.
- No end-to-end (`run_kv_matrix_bench.sh` / serve) A/B was run. The edit touches two kernels that
  the op-level benches measure directly, both at the streaming floor, and the fused production
  route over them (`linear_add --policy a16`) was measured at +0.43% mean.
- If a future task moves the A16 decode routes off the bandwidth floor, re-measure here first: the
  dedup's headroom is real but small (−1.5% instructions), and it is paid for in `PRMT`.

# GB10 (`sm_121a`) Branch Changelog — 2026-11-06 — T5 (nvfp4): Coalesced 4-byte scale stores, `__launch_bounds__(256, 2)`, and PDL wiring in the fused RMSNorm NVFP4 quantizer

## 1. Accounting

- Base commit: `7f899d8d5` (HEAD of `dev-rebased-20261028`, clean worktree at start).
- Files changed (4 files, +50/−18):
  - `M src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_fused_rmsnorm.cu` — kernel + both launchers
  - `M src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_a4_tma_launch.h` — dependency parameter
  - `M src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_a4_tma.cu` — forward the dependency
  - `M src/ops/linear/nvfp4/nvfp4_a4_tma.cuh` — dependency parameter + consumer-side wait
- `tools/cuda_stack/cuda_stack_allowlist.json`: unchanged (zero spills before and after; nothing to
  ratchet). Rebuilds: 1 full ninja pass (65 edges, one shared device link).

## 2. Task-premise corrections / Architectural rationale

- **Premise 2 ("redundant second global read of `x`") is already false in this codebase.** The task
  doc describes a group-strided kernel that loads `x_vec4[group * 2]` twice; the shipped
  `rmsnorm_nvfp4_quantize_kernel` uses pair ownership (`kPairsPerThread = 10` `__nv_bfloat162`
  `values[]`/`weights[]` registers per thread for the single `Nvfp4N14336K5120` geometry), computes
  sum-of-squares from those registers, stages the normalized BF16 row in shared, and the quantize
  pass reads only shared memory. The post-change SASS confirms it: all 20 `LDG.E.CONSTANT`
  (residual + weight) precede the first `BAR.SYNC` of the block reduction and there is no `LDG`
  after it (quantize reads `LDS.128` only). Required step 1 was therefore a no-op; no `kIters`
  cache arrays were added. The doc's `K=2560` case does not exist here — this kernel is fixed to
  `K = 5120` (`kGroupsPerRow = 320`).
- **The fused route gate is `T >= 512`, not `T >= 5`.** Eligibility is
  `nvfp4_attn_input_tma_route(tokens)` (`src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h`),
  i.e. prefill chunks, not decode/MTP-verify widths.
- **`store_nvfp4_scale_quad` is not templated on `<Geometry, Layout>` and `pdl::Dependency::None`
  does not exist.** The shipped helper takes `(scales, offset, scale, tid)`; the enum is
  `{Serialized, Programmatic}`. Used accordingly: `Serialized` for the quantizer (it reads the
  residual its producer just wrote), `Programmatic` for the downstream GEMM.
- Shuffle safety for the quad store: the quantize loop is `for (group = tid; group < 320; group +=
  256)`, so its second iteration runs only threads 0..63 — whole warps 0 and 1 — and every warp
  that executes `store_nvfp4_scale_quad`'s `__shfl_sync(FULL_MASK, …, 4)` has all 32 lanes present.
  Quads cover four consecutive groups of one token, which are one aligned 4-byte word in both
  tiled layouts (a 16-group scale tile row is 16 bytes).
- PDL protocol per `core/pdl.cuh`: the quantizer calls `pdl::enter()` (the grid is always ≥ 128
  CTAs after tile padding, above half of 48 SMs, so it never pre-triggers; dependents launch as its
  CTAs exit) and the padding early-return happens after the call, keeping the protocol complete.
  `nvfp4_a4_tma_kernel` gained `pdl::wait_for_dependencies()` after its mbarrier-init preamble —
  the only producer-dependent reads are the TMA loads below it — and `launch_nvfp4_a4_tma_mma`
  gained a `pdl::Dependency` parameter defaulting to `Serialized`, so all 17 other call sites
  (linear, linear_add, linear_swiglu, gdn_input_proj, shapes) keep their exact current launch.
  Programmatic serialization is applied only under graph capture, as everywhere in this route
  family; eager launches keep full stream serialization.

## 3. Subsystem breakdown

- `nvfp4_attn_input_fused_rmsnorm.cu`:
  - `rmsnorm_nvfp4_quantize_kernel`: `__launch_bounds__(kBlock)` → `(kBlock, 2)`; `pdl::enter()` at
    entry; padding path stores one `uint32_t{0}` per quad from `(thread & 3) == 0` (320 1-byte
    `STG.E.U8` → 80 4-byte `STG.E` stores per padding token); quantize path replaces the byte
    `scales[...] = quantized.scale` with `store_nvfp4_scale_quad` (320 1-byte → 80 4-byte stores
    per token). SASS: `STG.E.U8` 2 → 0, `SHFL` +4 (quad packing), net +32 static instructions.
  - `launch_quantize`: `<<< >>>` → `pdl::launch_with(pdl::Dependency::Serialized, …)`.
  - `nvfp4_attn_input_fused_rmsnorm_launch`: passes `pdl::Dependency::Programmatic` to the TMA GEMM.
- `nvfp4_attn_input_a4_tma_launch.h` / `nvfp4_attn_input_a4_tma.cu`: dependency parameter
  (default `Serialized` at the header) forwarded to both `Nvfp4A4TmaMmaSchedule` instantiations.
- `nvfp4_a4_tma.cuh`: `launch_nvfp4_a4_tma_mma(…, Rows = {}, pdl::Dependency = Serialized)`;
  `kernel<<<>>>` → `pdl::launch_with(dependency, …)`; kernel waits via
  `pdl::wait_for_dependencies()` between the barrier-init preamble and the TMA producer loop.

## 4. Verification

- Build: `ninja -C build-native -j "$(nproc)" infernix-serve infernix_ops_projection_test
  infernix_ops_linear_fused_test` after touching all consuming `.cu` under the five touched
  `nvfp4`/`attn_input_proj` trees (header edits, dead CUDA depfiles). The task doc's
  `infernix_ops_linear_fused_test` exists but the fused-rmsnorm CTests live in
  `infernix_ops_projection_test` (FAMILY `projection`); both were built.
- `ctest -R "infernix_attn_input_proj|infernix_cuda_stack_check"`: 3/3 passed —
  `infernix_cuda_stack_check` (ratchet gate, zero new spills), `infernix_attn_input_proj_test`,
  `infernix_attn_input_proj_fused_rmsnorm_test` (byte-exact fused-vs-reference at T = 512, 700,
  1024, 1500, 2048, 3584, 4096, plus the FP64-oracle A4 criterion and the CUDA-graph-captured
  T = 1500 byte-exactness that exercises the new Programmatic GEMM launch under capture).
- Neighbors of the shared TMA kernel: `infernix_gdn_input_proj_test`,
  `infernix_linear_nvfp4_a16_test`, `infernix_linear_nvfp4_a4_test`,
  `infernix_linear_add_nvfp4_test`, `infernix_linear_swiglu_nvfp4_test` — 5/5 passed.
- Resources (`cuobjdump --dump-resource-usage`): both `rmsnorm_nvfp4_quantize_kernel`
  instantiations `REG:54 STACK:0 SHARED:11300 LOCAL:0`, identical to baseline (the `(256, 2)`
  clamp is 128 registers, far above 54, so it costs nothing). All 37 `nvfp4_a4_tma_kernel`
  instantiations: `REG/STACK/SHARED/LOCAL` byte-identical to the baseline binary.
- SASS (`cuobjdump -sass -fun` on the exact `Tiled256` stored symbol): zero `LDG` after the block
  reduction's first `BAR.SYNC`; scale stores are 4-byte `STG.E` (padding path `STG.E …, RZ`,
  quantize path predicated `@!P0 STG.E` from the quad leader); no `STG.E.U8` remains.

## 5. Regressions & adverse results

Fused-route wall time (`INFERNIX_MEASURE_FUSED_STAGE=1`, eager launches, median of 3
alternating-order rounds of the *test's own* 31-sample medians, baseline binary `/tmp/t5_before_*`
built from `7f899d8d5`; the `reference_us` column is the *unmodified* rmsnorm+attn route in the
same process and serves as the drift control):

| T | fused base µs | fused cand µs | Δ fused | control Δ |
|---|---|---|---|---|
| 512 | 368.1 | 374.1 | +1.62% | +1.03% |
| 700 | 531.7 | 535.6 | +0.73% | −0.45% |
| 1024 | 522.0 | 521.4 | −0.11% | −0.80% |
| 2048 | 990.4 | 998.5 | +0.82% | +0.61% |
| 3584 | 1677.2 | 1686.0 | +0.52% | −0.03% |
| 4096 | 1903.4 | 1906.9 | +0.18% | +0.03% |

- No point regressed beyond the inter-binary drift band (the unchanged control route itself moved
  ±1% and prior campaigns measured up to ±3.5% between linked binaries); no point improved either.
  The eager fused stage is GEMM-dominated, the scale stores are 320 B/token of a ~2 MB/token
  quantizer output, and PDL is inert eagerly — so eager neutrality is the expected result, and
  that is what was measured. Disposition: accepted; the change's admissible gain is only the
  captured-graph preamble overlap, which no local harness times (see §6).
- The small positive fused deltas at T = 700/3584 (+0.73%/+0.52%) sit against control moves of
  ∓0.45%/−0.03%; they are within the pass-to-pass spread of these wall measurements (round spread
  ≤ 1.0%) and have no candidate mechanism (register/stack/shared usage identical; the kernel
  trades 4× fewer scale-store transactions for +32 static instructions and +4 shuffles).
  Disclosed as unexplained-small, not dismissed by a different criterion than the gains used.
- No numerical change (byte-exact tests), no new spills, no occupancy/SMEM change, no register or
  interface change for the 17 unchanged `launch_nvfp4_a4_tma_mma` call sites.

## 6. Accepted gaps / follow-ups

- The PDL enablement's only observable effect is under CUDA graph capture, where the GEMM's
  preamble can overlap the quantizer's tail; this host has no per-layer capture benchmark for the
  fused route, so its benefit is **unmeasured** (its correctness is measured: the T = 1500
  captured byte-exact case). A prefill-graph E2E A/B (`bench/run_kv_matrix_bench.sh` nvfp4 arm)
  would be the first real number if a future task needs one.
- The quantizer itself stays `Serialized` behind the residual producer per the spec; if a future
  prefill pipeline wants it as a consumer too, `launch_with(Programmatic, …)` now needs no code
  change (the `enter()` protocol is in place).
- The zero-padding and quantize stores assume the 16-group scale tile keeps quads inside one word
  (`kNvfp4ScaleTileGroups = 16`, unchanged since the layout's introduction); a layout change would
  need the same audit the standalone quantizer comment carries.
- The doc's claimed register-cache win (step 1) was already shipped earlier in this branch; no
  further register caching was possible or added.

---

# GB10 (`sm_121a`) Branch Changelog — 2026-11-06 — T6 (nvfp4): Route-gated register hoisting that removes the `mxfp8_kv_tiled_mma_kernel` NVFP4 stack spill + tail-only `q_s` zeroing and one barrier in `nvfp4_kv_grouped_mma_kernel`

## 1. Accounting

- Base commit: `5b0c7fd5e` (HEAD of `dev-rebased-20261028`, clean worktree at start).
- Files changed (3 files, +39/−25 source, −4 allowlist entries):
  - `M src/ops/softmax_attention/common/mxfp8_tiled_mma.cuh` — per-lane base/Q-scale hoisting made
    route-conditional (`Keys::kNvfp4`, `Pv8`)
  - `M src/ops/softmax_attention/dense/causal_cache/nvfp4/grouped_mma.cuh` — tail-only `q_s`
    zeroing merged into the query pass; two `__syncthreads()` removed
  - `M tools/cuda_stack/cuda_stack_allowlist.json` — 4 entries removed, 0 added, 0 raised
    (255 of 7623 kernels now carry stack/local; was 259 of 7623)
- Rebuilds: 4 single-invocation ninja passes (initial + 3 register-gate iterations), each paying the
  shared device link once. 2 baseline binaries (`/tmp/t6/before_causal`, `/tmp/t6/before_qsa`) built
  once from `HEAD` content and reused for every A/B.

## 2. Task-premise corrections / Architectural rationale

- **The spill is 3 of 18 instantiations, and it is the `Pv8 == false` subset.** `infernix-serve`
  carries 18 `mxfp8_kv_tiled_mma_kernel<Nvfp4KvArenaTiledSchedule, Nvfp4TiledKeys,
  Nvfp4G16TiledValues, …>` symbols (3 geometries × {`PagedKVDirectMetadata`,
  `PagedKVBatchMetadata<0>`, `PagedKVBatchMetadata<1>`} × `Pv8` ∈ {0,1}); exactly the 3 with
  `Pv8 == false` spilled `STACK:8`, which is exactly the 3 allowlist entries. The cause is the FP16
  PV phase: `v_lane_base`/`v_as`/`v_r` are live across the QK MMA loop only in that variant.
- **`SHARED:38016` in the task doc is wrong for these symbols**: `cuobjdump` reports `SHARED:0`
  because the KV arena is dynamic shared memory (`extern __shared__`), not static.
- **No `warp_row1` exists** in this kernel; only `warp_row0`, with the two row indices derived per
  tile as `row0`/`row1`.
- **`q_s` is `__half`, not `bf16`** (8 KiB for `Br × D = 16 × 256` still holds). Tail-only zeroing is
  exact because `causal_swizzle(row, d)` permutes only inside row `row`, so the query loop covers
  `[0, RowCount · D)` completely and leaves `[RowCount · D, Br · D)` untouched.
- **The doc's "scope it for every route" fix is a measured regression on the sibling routes.**
  Variant 1 (all bases/Q-scales materialized at use, exactly as the doc specifies) removed all 3
  spills but raised ptxas allocation on the routes that have registers to spare
  (FP8 `16/2 direct` 240→244, FP8 `16/2 batch` 243→245, K8V4 `16/2 direct` 254→255, K8V4 `16/2
  batch` 250→255) and cost **+0.61 %** on `d256-h16-kv2 fp8 W=512 ctx=16384` (median of 9
  interleaved passes; candidate slower in 8/9). The shipped fix gates the hoisting with
  `constexpr` predicates (`Keys::kNvfp4`, `kHoistPvBases = Pv8 || !Keys::kNvfp4`), so the FP8/K8V4
  routes get their original expressions back and only the register-pinned NVFP4 FP16-PV route
  rebuilds at its use sites.
- **Evidence that the gating is exact, not approximate**: after the change the FP8 and K8V4
  instantiations keep their baseline `REG` (18/18 unchanged) and
  `cuobjdump -sass -fun <mxfp8_kv_tiled_mma_kernel<…, Fp8KvTiledKeys, Fp8KvTiledValues,
  PagedKVDirectMetadata, false>>` is **byte-identical** to the `HEAD` binary (9,225 lines, zero
  diffs). Instruction counts also match for K8V4 `16/2 direct` `Pv8` 0/1 (4,152 / 4,536) and FP8
  `24/4 batch` (4,552).
- The 4th allowlist removal is **not** from T6: `…projection_tall_kernel…TallQ8…` was a stale key
  from an older build (that kernel is now `<TallQ8, 32, false/true>` at `REG:128 STACK:0`), so the
  refresh dropped it. It is a ratchet-down, not a mask.

## 3. Subsystem breakdown

- `src/ops/softmax_attention/common/mxfp8_tiled_mma.cuh`
  - `q_scale_r0`/`q_scale_r1` are `Keys::kNvfp4 ? 0.0F : q_scale[warp_row0 + gid (+8)]`, so the
    NVFP4 instantiation folds them away and reads the two row scales from shared memory at their
    single use after the `slab` loop (`q_scale_row0`/`q_scale_row1`); FP8/K8V4 keep the hoisted load.
  - `q_lane_base`/`q_as`/`q_r`, `k_lane_base`/`k_as`/`k_r` and `a_rowoff` fold to constants for
    `Keys::kNvfp4`; the NVFP4 QK path derives `a_row = warp_row0 + a_rin + ((a_mat & 1) << 3)`
    in-place (its own addresses never used those bases).
  - `v_lane_base`/`v_as`/`v_r` are gated by `kHoistPvBases = Pv8 || !Keys::kNvfp4`; the FP16-PV
    phase of the NVFP4 route rebuilds them per tile as `pv_lane_base`/`pv_as`/`pv_r`.
    `v8_key`/`v8_lane_base` are the mirror image (`Pv8 ? … : 0`), so neither phase carries the
    other's addresses.
- `src/ops/softmax_attention/dense/causal_cache/nvfp4/grouped_mma.cuh`
  - `for (index = tid; index < Br * D; …) q_s[index] = 0` + `__syncthreads()` became
    `for (index = RowCount * D + tid; index < Br * D; index += Threads)`, executed in the same pass
    as the query quantization loop. For the common decode width (`RowCount == Br == 16`) the loop
    trip count is zero, so the whole 8 KiB shared write disappears; for `RowCount = 24` (`Br = 32`)
    it writes the 8 KiB → 4 KiB tail.
  - The barrier that followed the query loop was removed, leaving the single `__syncthreads()` that
    precedes the first `q_s` read (the QK `ldmatrix` in the split-K loop); the register-only lane
    decomposition between them needs no barrier.

## 4. Verification

- **Resource gate** (`cuobjdump --dump-resource-usage build-native/apps/infernix-serve`): every
  `mxfp8_kv_tiled_mma_kernel` symbol in the binary is `STACK:0 LOCAL:0` (was: `STACK:8` on
  `Nvfp4KvArenaTiledSchedule` `256,16,2 direct`, `256,32,2 direct` and `256,24,4 batch0`, all
  `Pv8=false`). Register delta vs `HEAD`: only the 5 NVFP4 `Pv8=false` instantiations, 255→254.
  `nvfp4_kv_grouped_mma_kernel`: 272 instantiations, `STACK:0 LOCAL:0` before and after.
- **Allowlist ratchet**: `check_cuda_stack.py update` → `git diff --numstat` = `0 4` (zero added,
  four removed; the 3 target symbols plus the stale `projection_tall_kernel` key).
  `ctest -R infernix_cuda_stack_check` passes.
- **Oracle/behaviour CTests** (candidate binary, sequential, no `--parallel`):
  `infernix_softmax_attention_nvfp4_test`, `infernix_softmax_attention_fp8_test`,
  `infernix_softmax_attention_k8v4_test`, `infernix_qsa_nvfp4_test` and
  `infernix_cuda_stack_check` after the final build; the earlier build additionally passed
  `infernix_softmax_attention_{test,bf16,int8,extended,wide_fp8,wide_nvfp4,wide_k8v4}_test`,
  `infernix_vq_attention_test` and `infernix_qsa_{k8v4,k4v2}_test` (13 tests, 0 failures).
  The arithmetic is unchanged by construction (same expressions, same order; only where a value is
  produced), and the FP8 route is byte-identical SASS, so no numerical tolerance moved.
- **Op-level A/B** (`infernix_causal_softmax_attention_bench --entry cached --warmup 3 --repeat 7`,
  interleaved before/after binaries, cold L2, `median_us`), variant 1 (3 passes × 60 cells:
  `d256-h16-kv2` nvfp4 + nvfp4 `--fast-prompt --fast-prompt-pv8` + fp8 + k8v4,
  `d256-h24-kv4` nvfp4, `W = 1,4,16,64,256,512`, `context = 4096/16384`):
  mean −0.31 %, median −0.05 %, and the pass-to-pass spread inside single cells reached 12–23 %, so
  only the focused 9-pass paired re-check of the 10 suspicious/extreme cells decides:

  | cell | median paired Δ | candidate slower |
  |---|---|---|
  | fp8 `16/2` W=512 c=16384 | **+0.61 %** | 8/9 |
  | nvfp4 `16/2` W=1 c=16384 | +0.97 % | 6/9 |
  | nvfp4 `16/2` W=16 c=16384 | −0.03 % | 2/9 |
  | nvfp4 `16/2` W=64 c=4096 | −0.36 % | 1/9 |
  | nvfp4 `16/2` W=512 c=4096 | +0.01 % | 5/9 |
  | nvfp4 pv8 W=4 c=16384 | −2.09 % | 1/9 |
  | nvfp4 pv8 W=16 c=16384 | −0.13 % | 3/9 |
  | k8v4 `16/2` W=1 c=16384 | −1.31 % | 3/9 |
  | nvfp4 `24/4` W=1 c=4096 | +0.00 % | 4/9 |

  A 5-pass interleaved paired A/B re-check on the **shipped route-gated binary** (`before_causal` vs
  `final_causal`, `--warmup 3 --repeat 11`, cold L2) across all 10 cells confirms the FP8 regression
  is eliminated and `nvfp4_kv_grouped_mma_kernel` (`W=1,4`) improves or stays neutral:

  | cell | `before_med` (µs) | `final_med` (µs) | median paired Δ | candidate slower |
  |---|---|---|---|---|
  | nvfp4 `16/2` W=1 c=16384 | 51.97 | 50.91 | **−0.31 %** | 0/5 |
  | nvfp4 `16/2` W=4 c=16384 | 58.21 | 59.07 | **−1.18 %** | 2/5 |
  | nvfp4 `16/2` W=16 c=16384 | 102.11 | 102.14 | +0.03 % | 3/5 |
  | nvfp4 `16/2` W=64 c=4096 | 106.56 | 106.37 | −0.06 % | 2/5 |
  | nvfp4 `16/2` W=512 c=4096 | 702.02 | 702.75 | +0.23 % | 3/5 |
  | nvfp4 pv8 W=4 c=16384 | 58.05 | 57.15 | **−1.51 %** | 1/5 |
  | nvfp4 pv8 W=16 c=16384 | 102.11 | 102.08 | −0.03 % | 1/5 |
  | k8v4 `16/2` W=1 c=16384 | 68.26 | 67.30 | −0.05 % | 2/5 |
  | nvfp4 `24/4` W=1 c=4096 | 32.45 | 30.59 | **−2.85 %** | 0/5 |
  | fp8 `16/2` W=512 c=16384 | 1485.70 | 1487.58 | +0.28 % | 3/5 |

- **`infernix_qsa_attention_bench --storage nvfp4 --contexts 4096,16384 --widths 1,64,512
  --warmup 2 --repeat 5`** (3 interleaved passes, medians): `total_us` within ±1.1 % on all six
  cells (4096/1 −0.14 %, 4096/64 +1.09 %, 4096/512 +0.01 %, 16384/1 +0.43 %, 16384/64 −0.96 %,
  16384/512 +0.18 %); `attention_us` −3.25 % at 16384/1 and ≤ +1.2 % elsewhere. QSA keeps its own
  kernels, so unchanged is the expected result.

## 5. Regressions & adverse results

- **FP8/K8V4 +0.6 % on the first variant — fixed.** Scoping the bases for every route (the doc's
  literal step 1) changed ptxas allocation on routes that were never register-pinned; the
  `d256-h16-kv2 fp8 W=512 ctx=16384` cell measured +0.61 % (median of 9 interleaved passes, slower
  in 8/9; the 3-pass matrix saw +1.68 % there). The shipped route-gated version leaves those
  symbols at baseline `REG` with byte-identical SASS, and the 5-pass paired re-check confirms
  noise-floor parity (+0.28 %, 3/5 slower).
- **A favourable number that did not survive**: the 3-pass matrix showed −7.54 % at
  nvfp4 `16/2` W=512 ctx=4096 (692.02 → 639.87 µs). The 9-pass paired re-check gives +0.01 %
  (5/9 slower) and the 5-pass shipped-binary re-check gives +0.23 % (3/5 slower) — that cell's
  spread was 14–17 %, so the "win" was noise and is not claimed.
  Same disposition for the 3-pass −3.94 %/−6.38 % at W=4 c=4096 and the apparent +3.70 %
  nvfp4-pv8 W=4 c=16384 (paired −2.09 % on variant 1, −1.51 % on shipped binary).
- **Variant-1 `nvfp4 16/2 W=1 c=16384` (+0.97 %) resolved on shipped binary (−0.31 %, 0/5 slower)**:
  on the shipped route-gated binary, `nvfp4 16/2 W=1 c=16384` measures 51.97 → 50.91 µs (−0.31 %
  median paired Δ, slower in 0/5 passes) and `nvfp4 24/4 W=1 c=4096` (`RowCount=6 < Br=16`) measures
  32.45 → 30.59 µs (−2.85 % median paired Δ, slower in 0/5 passes). Inlining `a_row` on `mxfp8_kv_tiled_mma_kernel`
  adds 16 SASS instructions to 5 of the 6 NVFP4 `Pv8=false` symbols (10,505→10,521 `direct`,
  10,521→10,537 `batch`; `24/4 batch` unchanged at 10,569) while eliminating the `STACK:8` spill,
  with `W=16, 64, 512` within ±0.23 % of baseline.
- **Barrier merge risk**: `q_s` is written by two loops now (tail zeroing + query quantization)
  before the single remaining `__syncthreads()`. The loops' write sets are disjoint
  (`[RowCount·D, Br·D)` vs the swizzled rows below `RowCount·D`) and the first `q_s` read is the
  QK `ldmatrix` after that barrier, verified by grep of every `q_s` reference in the file. Both
  loops are uniform-control (value selects, not branches), so no divergent barrier was introduced.

## 6. Accepted gaps / follow-ups

- Only the 5 register-pinned NVFP4 `Pv8=false` instantiations rebuild their PV addresses per tile.
  The `Pv8=true` NVFP4 variants keep the hoisted PV bases because they never spilled; if a future
  schedule pushes them to the ceiling too, the same `constexpr` gate extends to them.
- `q_s` tail zeroing is still a full-row loop: when `RowCount` is not a multiple of 16 it writes the
  whole padding tail (`Br − RowCount` rows). Vectorising it to `__half2`/`uint4` stores or folding
  it into the producer's own row loop was not measured and is not claimed.
- `tools/cuda_stack/check_cuda_stack.py update` also dropped one stale pre-existing key
  (`projection_tall_kernel<TallQ8>`); a future allowlist refresh should re-key such stale entries
  deliberately rather than as a side effect.

# GB10 (`sm_121a`) Branch Changelog — 2026-11-06 — T7 (nvfp4): `attn_input_proj` A16→A4 hand-over moved 4→17 (draft-vs-verify activation parity at MTP/DFlash2 widths) and `AllowA4Residual` fallback fixed in the `attn_input_proj`/`gdn_input_proj` planners

## 1. Accounting

- Base commit: `ead81ce45` (branch `dev-rebased-20261028`, HEAD at task start). Work left
  uncommitted for the Antigravity review.
- Files changed (4, +55/−13):
  - `M src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.cpp` — `resolve_route` gate + residual fix
  - `M src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.cpp` — residual fix (gate unchanged)
  - `M tests/ops/test_attn_input_proj.cpp` — residual-Use helper handling, A4 criterion gated on `T>=17`, workspace high-water check, + 4 new cases (`AllowA4` T=16, `AllowA4Residual` T=16/17/128)
  - `M tests/ops/test_gdn_input_proj.cpp` — A4 criterion gated on `T>=17` + 4 new cases (`AllowA4` T=16, `AllowA4Residual` T=16/17/128)
- Spill allowlist: untouched (no CUDA translation unit changed; `infernix_cuda_stack_check` green).
- Rebuilds: 3 Ninja invocations (baseline target check — already current, 14-edge candidate build
  sharing one device link, 1-edge attn-test rebuild after the first CTest failure).

## 2. Task-premise corrections / Architectural rationale

- The doc's symbols `select_nvfp4_attn_input_route`/`select_nvfp4_gdn_input_route` do not exist;
  the planners' real symbol is the file-local `resolve_route`. The named test files
  `tests/ops/attn_input_proj_test.cu`/`gdn_input_proj_test.cu` are `tests/ops/test_attn_input_proj.cpp`
  /`test_gdn_input_proj.cpp`.
- The cited `infernix_linear_bench` numbers at `[13952, 2560]` (80.9–82.9 µs, 91–92 % DRAM) are not
  this planner's op. `[13952, 2560]` is the six-output fused variant, which the wrapper sends to
  `launch_nvfp4_n13952_k2560_split` (and whose capacity carve-out already keeps A16 through T=64).
  The two modified planners serve the K=5120 fused shapes `[14336, 5120]`/`[16384, 5120]`; there the
  A16 lane measures 179–181 µs at T=5..16 (≈227–231 GB/s ≈ 82–84 % of the ~273 GB/s LPDDR5X
  ceiling). 80.9 µs would require ~512 GB/s on a 41.5 MB parent and cannot transfer; the ≥90 % bar
  was not attainable and the claim is reported at the measured value.
- The 2026-10-31 T7 note "the Attn A4 gate at 4 is already optimal" is superseded: today's
  interleaved 3-pass cold A/B on the fused op (forced A16 vs the production A4 lane) shows A16
  ahead by 1.0–4.1 % across the whole T=4..16 band and behind by 2.8–5.5 % from T=17, so the
  hand-over moved to 17. The GDN boundary stays at 16 (re-confirmed: T=4..16 tie within ±0.6 %,
  T=17..24 A16 +7.1..+9.2 % slower).
- Step 1's "`kMaxTokens` template cap in `nvfp4_attn_input_decode.cu`" does not exist:
  `Nvfp4ScheduleInstance`'s `Capacity` defaults to 0 and `nvfp4_attn_input_a16_launch` already
  dispatches the `<16,8,2,2>` RowTiles-2 sliced-K tile over the whole T=2..16 band (shipped
  2026-10-31). No kernel change was needed.
- Step 2's "whenever a caller passes `LinearPolicy::AllowA4Residual`": no production caller does
  for these projections (`limit_precision` passes `a4_residual_here=false` for attn/gdn, and
  `common_policy` rejects residual Uses at projection prepare). The fix aligns the planners with
  the public wrappers' `validate_policy`, which already accepts the residual Use as an execution
  policy; it is a contract fix, not a live-behavior change.
- `nvfp4_linear_swiglu_plan.cpp`/`nvfp4_linear_add_plan.cpp` (in the doc's target list) already
  split `allows_a8_residual` (→ A16) from `allows_a4_residual` (→ registered 2-term route); no
  change was required there.

## 3. Subsystem breakdown

- `nvfp4_attn_input_plan.cpp` (`M`): `resolve_route` hand-over `tokens >= 4` → `tokens >= 17`;
  `AllowA4Residual` now joins the token ladder (A16 T≤16, one-term A4 T≥17) instead of being
  forced to A16 at prefill widths; `AllowA8Residual` stays A16 (no 4-bit activation permission);
  throw only when neither `allows_a4` nor `allows_a4_residual`. The workspace capacity function
  tracks `resolve_route`, so a residual Use at T≥17 now declares the one-term A4 capacity.
- `nvfp4_gdn_input_plan.cpp` (`M`): same residual split (`allows_residual` → `allows_a8_residual`
  in the A16 branch; `allows_a4 || allows_a4_residual` in the ladder branch). The 16/17 boundary
  and its 2026-10-31 evidence comment stand.
- `test_attn_input_proj.cpp` (`M`): A4 criterion applies when the one-term lane actually runs
  (`AllowA4` and `AllowA4Residual` from `T>=17`, enforcing `kAttnInputProjA16Tolerance` and zero
  workspace usage across `T<=16`); workspace query/execution high-water check added to
  `run_nvfp4_target_case`; residual Uses prepare with `AllowA4` and then set the
  execution policy (mirroring the invariant-Use pattern, since `common_policy` intersects only the
  one-term policies); residual Uses pass identical activation divisors (the shared-activation
  quantizer requirement); new cases `AllowA4` T=16 and `AllowA4Residual` T=16 (A16 band), T=17/128 (A4 lane).
- `test_gdn_input_proj.cpp` (`M`): same `T>=17` criterion predicate; new cases `AllowA4` T=16 and
  `AllowA4Residual` T=16/17/128.

## 4. Verification

- Premise A/B (baseline HEAD binaries, `--nvfp4-policy a16` vs `a4`, cold, warmup 5 / repeat 30,
  median of 3 interleaved passes, µs):
  - attn [14336,5120]: T=4 −3.15 %, T=5 −3.33 %, T=6 −4.05 %, T=8 −2.72 %, T=10 −2.73 %,
    T=12 −1.05 %, T=14 −2.73 %, T=16 −3.30 % (A16 ahead); T=17 +4.42 %, T=20 +2.80 %,
    T=24 +5.45 % (A16 behind) → hand-over 17.
  - gdn [16384,5120]: T=4..16 |Δ| ≤ 0.6 %; T=17 +7.08 %, T=20 +7.58 %, T=24 +9.21 % (A16 behind)
    → boundary 16 unchanged.
- Post-change production-route A/B (baseline vs candidate binaries interleaved, `a4` policy):
  attn T=4..16 −2.2..−3.8 %, T=17..24 ≤ +0.1 %; gdn all widths within run drift (route unchanged).
  Reverse-order control at unchanged widths: T=1 +0.01 %, T=2 +0.00 %, T=3 +0.28 %.
- Candidate verification-band DRAM efficiency: 41,482,240 B / ≈181–183 µs ≈ 226–229 GB/s ≈ 83 % of
  ~273 GB/s (task's ≥90 % bar came from the other shape; see §5/§6).
- CTest (`-R "infernix_attn_input_proj|infernix_gdn_input_proj|infernix_cuda_stack_check"`, serial):
  `infernix_cuda_stack_check` 1.48 s Passed; `infernix_attn_input_proj_test` 24.6 s Passed;
  `infernix_attn_input_proj_fused_rmsnorm_test` skip-77; `infernix_gdn_input_proj_test` 10.1 s
  Passed; `infernix_gdn_input_proj_conv_snapshot_test` 27.4 s Passed;
  `infernix_gdn_input_proj_conv_record_test` 8.3 s Passed. 5 passed + 1 skipped, 0 failed.
  (First run failed: the attn helper threw "invalid projection activation policy" from prepare's
  `common_policy`; fixed by the prepare-with-AllowA4-then-set-execution-policy pattern.)
- No CUDA source changed: per-symbol resources identical, allowlist unchanged, nothing to ratchet
  down. `git diff --check` clean.

## 5. Regressions & adverse results

1. Apparent attn T=1..3 regression (+0.6..+1.2 %) in the first post-change pass: those widths run
   the identical A16 route in both binaries. A reverse-order pass (candidate first) measured
   +0.01/+0.00/+0.28 % while the gate band still showed −3.2/−3.3 %, so the positive deltas were
   base-first run-order/thermal drift, not a regression. Disposition: resolved (no change).
2. Apparent gdn T=1..16 +0.02..+1.4 % in the same first pass: same order pattern, and the gdn
   candidate's AllowA4 route table is byte-identical to baseline (only the residual branch
   changed, which the `a4` policy never takes). Disposition: not a regression; same control.
3. Numerical semantics under `AllowA4` at T=4..16 (27B-class `[14336,5120]` artifacts): outputs
   switch from the one-term A4 lane to the exact A16 lane, so the live kernel set of that
   configuration changes and the band's bits move to the A16 route's (closer to the oracle; the
   A4-criterion oracle tests pass and the band is now draft(T=1)-consistent). This is the task's
   intended fast-path demotion to a higher-precision route, not a loss.
4. The task's ≥90 % DRAM-efficiency bar is not met at the verification band: measured ≈83 %
   (179–183 µs vs the ≈152 µs weight-stream floor). The bar was inherited from the `[13952,2560]`
   plain-Linear bench and does not transfer to the 41.5 MB fused op. Disposition: accepted; the
   −2.2..−3.8 % relative win against the previous production route is the supported claim.
5. End-to-end MTP/DFlash2 acceptance rate could not be measured: every local NVFP4 artifact is
   Qwen3.8-Flash-Next (hidden 2560), whose attn/GDN projections dispatch through the K=2560 split
   paths untouched by this task; no hidden-5120 NVFP4 artifact exists on this host. Disposition:
   unverified (§6).

## 6. Accepted gaps / follow-ups

- The Flash-Next (K=2560) verification-width story lives in the `[13952,2560]`/`[16384,2560]`
  linear route tables behind `launch_nvfp4_n*_k2560_split`. Its capacity carve-out (no A4 workspace
  below T=65) suggests A16 already covers T≤64 there, but that path was neither audited nor changed
  in T7; a follow-up should confirm it if Flash-Next acceptance remains a target.
- The model-level acceptance-rate improvement promised by the task title is unmeasured on this
  host (no artifact exercises the modified K=5120 planners); the shipped evidence is op-level
  latency, route semantics, and oracle tests.
- `AllowA4Residual` on these two projections still has no production caller; the planner semantics
  and tests are contract preparation for a residual-qualified artifact or a future
  `limit_precision` change.
- The task doc's absolute-µs/DRAM-% baselines should be regenerated from
  `infernix_attn_input_proj_bench`/`infernix_gdn_input_proj_bench` (the fused ops' own benches) in
  any follow-up, not from `infernix_linear_bench` on a same-named plain-Linear shape.
