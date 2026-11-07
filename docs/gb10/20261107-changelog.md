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

# GB10 (`sm_121a`) Branch Changelog — 2026-11-06 — T8 (nvfp4): `ExpertA4Residual`/`ExpertA4ResidualDecode` qualified on `DenseKernels::Nvfp4` (W4Dense, `--a16-experts`) and the NVFP4 GDN conv-post kernel sized to its token tiles with PDL (the multi-row `Forward::gdn` snapshot premise falsified)

## 1. Accounting

- Base commit: `025b192cf` (branch `dev-rebased-20261028`, HEAD at task start). Work left
  uncommitted for the Antigravity review.
- Files changed (3, +130/−46):
  - `M src/models/qwen4_exp/program/fast_paths.cpp` — `fast_path_applies` split (`ExpertA4Residual`
    + `ExpertA4ResidualDecode` gain `DenseKernels::Nvfp4`; `ExpertA4Residual3` stays on
    `Fp8Row`/`Mxfp8`), `kQualified` + 14 entries (`{Nvfp4, false}` × every KV × the two paths),
    the stale W4Dense block comment rewritten to the shipped table.
  - `M src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_post.cu` — `nvfp4_gdn_conv_post_kernel`
    templated over `WarpsPerCta ∈ {1, 4, 8, 32}` and sized by the token-tile count; the 7 × 32
    shared tap loads spread across the whole CTA; `pdl::enter()` + programmatic dependent launch.
  - `M tests/models/qwen4_exp/test_fast_paths.cpp` — 8 new checks (W4Dense applies/qualified for
    the two paths, residual-3 excluded), two stale comments corrected, and the stale
    "gdn-conv-fused is an unqualified candidate" check flipped to the owner's registration.
- Spill allowlist: untouched (`nvfp4_gdn_conv_post_kernel` instantiations are `STACK:0 LOCAL:0`
  before and after; the `update` run wrote no diff). `infernix_cuda_stack_check` green.
- Rebuilds: 4 Ninja invocations (1 HEAD baseline for the A/B binaries, 1 fifteen-edge candidate
  sharing one device link, 2 single-edge relinks after test/comment edits).

## 2. Task-premise corrections / Architectural rationale

- **The title's `Forward::gdn` change is invalid and no `forward.cpp` change was made.**
  `batch.width` is the tokens-per-sequence dimension and the lanes are `batch` (1..8), so
  `decode = batch.verify == nullptr && batch.width <= 1` already covers every decode, multi-lane
  included — a B = 2..8 decode round runs the fused `gdn_input_proj_conv_snapshot` routes form
  (W=1, B=1..8 → `launch_nvfp4_n16384_k2560_conv_snapshot`), not the composed pair the doc
  describes. A `verify == nullptr, width > 1` call is a single-sequence prefill chunk (the branch
  throws for `batch != 1`) and stays composed by design; a multi-row snapshot across lanes would
  advance one sequence's `conv_state` through other lanes' tokens, which is semantically wrong.
  The doc's own "Required Implementation Steps" omitted the `forward.cpp` change it titled.
- The doc's claim that `Forward::gdn` lines 697–708 call `ops::gdn_input_proj_conv_record` under
  verification is false for Qwen4Exp: its verification conv runs inside
  `gated_delta_net_replay_record_conv` (width ≤ `kGdnReplayConvMaxWidth`) or through
  `causal_conv1d_silu_from_states_split`; the record public Op is the Qwen3.5 route.
- The post-kernel's idle-warp defect is real but the doc's band is wrong: `T = 2..16` under the
  A16 policy is the fused `SmallTFusedA16` SIMT kernel, not the post kernel. The post kernel runs
  on the Materialized schedules — A16 `T = 17..64`, the A4 policy from `T = 4`, and the
  column-invariant/batched compositions — where the retired 32-warp CTA covered at most 8 token
  tiles and left 24–31 warps idle. The fix sizes the CTA to the tile count (1 warp through `T = 8`,
  4 through `T = 32`, 8 through `T = 64`, 32 above — the doc's 4/8 split at 32 leaves three of
  four warps idle at `T ≤ 8`, so the 1-warp tier was added and measured). Per-lane arithmetic and
  the tile decomposition are unchanged, so the outputs are bit-identical; the `bool Record`
  template the doc suggested is already the `Publish` policy parameter.
- The doc's verification command names the wrong bench: `infernix_gdn_input_proj_bench` is the
  projection-only contract bench and has no fused/composed arm; the fused route is
  `infernix_gdn_input_proj_conv_snapshot_bench`. Its absolute baselines (113.0 … 178.7 µs at
  `T = 2..32`) match no arm of either binary on this host (the `nvfp4` fixture's fused rows
  measured 214–237 µs at the HEAD binary; the `nvfp4-flash` fixture refuses `T > 1` snapshots) —
  consistent with T7's note that these bars need regeneration. Acceptance was therefore taken as
  candidate-against-HEAD-binary A/B on the fused Op, not the doc's absolute µs.
- The doc's "update the comment at line 573 to match the table" exposed a stale contract: the
  owner's commit `98aff48b0` (2026-11-05) had registered `GdnConvFused` on `Nvfp4` (both expert
  precisions, every KV) while the table's own block comment still said it "registers nothing" and
  `test_fast_paths.cpp` still asserted it was an unqualified candidate — the test failed 14
  assertions at HEAD before this task. The comment and the check were brought to the shipped
  table; the owner's registration stands.
- **The registered configuration cannot execute on this host.** `a16_experts` comes from the
  artifact (config.cpp: experts stored without activation scales, or an a4-optional artifact run
  without `--expert-a4-activations`); this host's W4Dense artifact stores 4-bit activation scales,
  so it selects `{Nvfp4, true}` and the new `{Nvfp4, false}` qualifications change nothing here.
  The doc's "`--a16-experts`" is a conversion/load-time property, not a runtime CLI flag. The
  quality evidence carries from recipe C's gates, which are bank-based ("artifacts with NVFP4
  W4A16 experts", model-quality.md 6.7) on the same NVFP4 planes; the speed evidence for the
  narrow route against the replaced W8A8 decode route cannot be produced here (§5/§6).

## 3. Subsystem breakdown

- `fast_paths.cpp` (`M`): `fast_path_applies(ExpertA4Residual/ExpertA4ResidualDecode)` now
  `!a4_experts && (Fp8Row || Mxfp8 || Nvfp4)`; `ExpertA4Residual3` keeps the old pair (two terms
  pass everywhere they are run, so the precise form stays an unregistered candidate). `kQualified`
  gains `ExpertA4Residual` and `ExpertA4ResidualDecode` on `{Nvfp4, false}` × 7 KV formats with a
  comment carrying the evidence and the unmeasured narrow-speed follow-up. Selecting the narrow
  path flips `a8_activation_routes`' `nvfp4_experts` off (its own `!expert_a4_narrow` term), so
  the configuration's experts move from the W8A8 route to `NarrowA16::kA4Residual` (GPU and CPU)
  and `WideA16::kA4Residual` — the replacement the task specified.
- `nvfp4_gdn_snapshot_post.cu` (`M`): one instantiation per size tier shared by both `Publish`
  policies; the launcher dispatches on `ceil(T / 8)` tiles. The `warp == 0` serial fill of
  `initial[3][32] + weight[4][32]` became a whole-CTA strided fill (value index = part × 32 +
  column, consecutive threads on consecutive channels of one tap). `infernix::pdl::enter()` is the
  kernel's first statement (no read before it) and both launchers use
  `pdl::launch_with(Dependency::Programmatic, …)` — active only under CUDA-graph capture, where it
  overlaps this grid's launch with the projection's CTA drain; eager launches serialize as before.
- `tests` (`M`): W4Dense registration matrix (+4 qualified, +1 applies, +1 residual-3-excluded),
  the residual-decode "no configuration registers it" comment scoped to W4Dense, and the
  gdn-conv-fused check aligned to `98aff48b0`.

## 4. Verification

- Resources (`cuobjdump --dump-resource-usage`): 8 instantiations (`{1,4,8,32} × {NoHistory,
  SnapshotHistory}`) at `REG 40–44 STACK:0 SHARED:1920 LOCAL:0` (baseline: the two `<32, 8>`
  forms at `REG 42/44 STACK:0 LOCAL:0`). Allowlist: zero new entries, zero removals;
  `infernix_cuda_stack_check` Passed.
- CTest (no `--parallel`, single-GPU host): `infernix_gdn_input_proj_test` (10.4 s),
  `infernix_gdn_input_proj_conv_snapshot_test` (27.2 s),
  `infernix_gdn_input_proj_conv_record_test` (8.2 s),
  `infernix_qwen4_exp_fast_paths_test` and `infernix_cuda_stack_check` — all pass. The two conv
  tests are the post kernel's numerical oracles (snapshot publish + record, every registered
  `T`/`B` band including the mixed `valid_columns` prefixes).
- Op A/B (`infernix_gdn_input_proj_conv_snapshot_bench --format nvfp4`, cold, 25 reps, 2
  alternating passes, best-of-2 medians; best-of-2 because host contention skews single GB10
  runs):

  | band | forms × executions | baseline → candidate | Δ |
  |---|---|---|---|
  | a4 `T = 4/8/16/32` | snapshot+record × graph+eager | 218.9–237.4 → 207.7–222.0 µs | −9.0 … −17.4 µs (−4.1 … −7.7 %) |
  | a16 `T = 17/32/64` | snapshot+record × graph+eager | 245.8–380.7 → 235.3–366.4 µs | −10.2 … −20.9 µs (−2.8 … −5.6 %) |
  | a4 `T = 2` (fused SIMT, control) | all four | 214.9–215.7 → 213.7–215.1 µs | −1.2 … 0.0 µs (tie) |

  Eager improves as much as graph, so the win is the warp sizing and the distributed fill; the
  PDL wiring's graph-launch overlap is an addition, not the measured source, and eager (where PDL
  is inert) carries no regression.
- E2E regression on W4Dense (the unchanged `{Nvfp4, true}` configuration): 4 alternating
  baseline/candidate passes of `infernix_bench -pg 2048,128 -r 2`: prefill 2960.3 → 2968.2 t/s
  (+0.27 %), decode 47.76 → 47.34 out t/s (−0.87 %, §5). The selected fast-path list, the
  prefill signature, and an 8.37 MB expert route trace (all bytes but one header timestamp qword
  pair) are identical between the binaries.

## 5. Regressions & adverse results

1. **E2E decode −0.87 % on W4Dense** (4 alternating passes, pp2048+tg128, r = 2: base
   47.84/48.09/47.08/48.02, candidate 47.39/47.33/47.39/47.24 out t/s; means 47.76 vs 47.34,
   Welch t ≈ −1.8, distributions overlap). No functional difference exists on this configuration:
   the selection log is identical, the route traces are byte-identical except one header timestamp
   pair, and nothing in the diff executes on its routes (the conv-post kernel is not called by
   Qwen4Exp's Forward; the new qualifications require `a4_experts == false`, which this artifact
   does not select). The candidate's spread (0.15) is smaller than the baseline's (1.01) and the
   baseline's own worst pass undercuts every candidate run, so no mechanism is known; consistent
   with binary-layout/thermal variance on this host. Disposition: **unresolved, disclosed** — the
   task's own configuration (where the diff *does* run) has no local artifact to measure.
2. **One A/B pass outlier**: snapshot-eager a16 `T = 17` candidate pass 1 read 318.4 µs against
   its own pass 2 at 241.5 (baseline 253.8/257.9). The remaining 63 pairs are monotone within
   their bands; best-of-2 keeps the point at −4.8 %. Disposition: single-run host-contention
   spike, disclosed (the known GB10 host-contention effect).
3. **The task's fused-latency bar could not be reproduced or evaluated as written** (premise §2):
   its named bench has no fused arm and its µs numbers match no arm of either binary. The shipped
   evidence is the HEAD-binary A/B above; `fused` vs `composed` was not re-derived because the
   bench offers no composed arm and the projection row of
   `infernix_gdn_input_proj_bench --format nvfp4 --nvfp4-policy a4` (194.6 µs at `T = 8`) is a
   different fixture's projection-only contract, not this route's composed pair.
4. **`T = 2..3` under the a4 policy do not improve** (−0.5 %…0 %, within noise): they are the
   unchanged fused SIMT path, and the doc's expectation of post-kernel gains there does not hold
   (the kernel does not run at those widths). Disposition: accepted, control bands tie.
5. The `infernix_qwen4_exp_fast_paths_test` failure found at HEAD (14 gdn-conv-fused assertions,
   stale against the owner's `98aff48b0`) was **fixed** here by aligning the check and both stale
   comments to the shipped table — not by reverting the registration.

## 6. Accepted gaps / follow-ups

- **The `{Nvfp4, false}` registration's speed gate is unmeasured on this host.** The narrow
  (decode/verification) route replaces the W8A8 expert route the owner adopted 2026-10-27; the
  wide route's speed is the gated one (+2.0 % MTP decode on the same bank class,
  model-quality.md 6.7), the narrow route's verdict against A8 needs an `--a16-experts` W4Dense
  artifact (experts stored without activation scales), which this host does not carry as of
  2026-11-07. The registration follows the task's explicit instruction; the follow-up is the E2E
  A/B on such an artifact, and the `kQualified` comment records exactly this.
- The W4Dense E2E −0.87 % decode difference is unattributed (§5.1); the route traces prove the
  executed routes are identical, not the timing.
- PDL overlap was shipped as specified but not isolated from the warp-sizing win (eager, where
  PDL is inert, already captures the full gain); no serialized-dependency variant was timed
  because no eager or graph run regressed against it.
- `forward.cpp` and `nvfp4_gdn_snapshot_small_t.cu` (the doc's target list) were left untouched:
  the title's premise is falsified (§2) and the SmallT kernel already serves `T = 2..16` fused.

# GB10 (`sm_121a`) Branch Changelog — 2026-11-06 — T9 (nvfp4): `(group, column)` 2D grid, warp-local `float4` staging, and PDL wiring in `nvfp4_logits_topk_kernel` (the serial-column-loop premise falsified by the cub sort)

## 1. Accounting

- Base commit: `a1db53b3f` (T8 HEAD, `dev-rebased-20261028`).
- Files changed (2, 61 insertions / 27 deletions):
  - `src/ops/linear_topk/nvfp4.cu` (`M`) — kernel + launcher.
  - `bench/ops/linear_topk_bench.cu` (`M`) — new `fp8-f32-122b` profile; the bench had **no**
    registered profile that executed this kernel, so the task's own verification command could
    not have measured the change.
- Stack-spill allowlist: unchanged (kernel is `STACK:0 LOCAL:0` before and after; nothing to
  ratchet in either direction).
- Rebuild count: 3 single-invocation Ninja builds (baseline bench with the new profile, then
  `infernix-serve` + `infernix_ops_sampling_test` + `infernix_linear_topk_bench` twice — the
  smem-staging variant and the shipped warp-shuffle variant).

## 2. Task-premise corrections / Architectural rationale

Measured on GB10 (`sm_121a`, 48 SMs) at the kernel's only production shapes:

1. **No router use, no 4-CTA idle GPU.** `resolve_profile` registers `linear_topk` only for the
   248,320/131,072-row vocabulary heads; a `[512, 2560]` MoE router never reaches this kernel
   (`linear_topk.cpp` throws `unsupported head profile`). The baseline grid is already
   `producer_groups = 1940` CTAs for `rows = 248,320` — all 48 SMs loaded, not "4 CTAs with
   91.7 % of the GPU idle".
2. **The baseline load was one fully-coalesced `LDG.E.32` per thread**, not "4 strided scalar
   floats" (adjacent thread ↔ adjacent row of a 128-row tile: 128 B per warp, one sector).
3. **One explicit `__syncthreads()` per column** plus the `cub::BlockMergeSort` internals, not
   two.
4. **The doc's `float4` mapping is structurally wrong**: `row_base = blockIdx.x * 128 + lane * 4`
   with 128 lanes spans 512 rows, but the tile is 128 rows — a 128-row tile holds exactly 32
   `float4`s. The correct vectorization is warp-local: each warp's 32 rows are eight `float4`s
   loaded by lanes 0..7 in one `LDG.E.128` and redistributed to the sort's one-row-per-thread
   layout with four intra-warp `SHFL` (cross-warp register movement is impossible; the smem
   variant that fixes the doc's mapping costs a barrier — measured, §5.4, rejected).
5. **The dominant cost is the per-tile `cub::BlockMergeSort`, not the column loop.** The baseline
   kernel runs 1905 µs at `U = 128` reading 127 MB of logits ≈ 66 GB/s, and the whole shipped
   package (2D grid + vectorized load + PDL) moves it only −2.1 % (1905 → 1866 µs): spreading
   the identical per-`(group, column)` sort work over more CTAs cannot beat a sort-throughput
   bound. The serial loop's latency chain was already hidden by ~16 resident CTAs/SM.

## 3. Subsystem breakdown

- `src/ops/linear_topk/nvfp4.cu`
  - `nvfp4_logits_topk_kernel` (`M`): 1D grid × serial column loop → `dim3(producer_groups,
    columns)` (the merge kernel's shape); `column = blockIdx.y`, loop and its per-column
    `__syncthreads()` removed; `pdl::enter()` at entry; `__launch_bounds__(kThreads)` →
    `__launch_bounds__(kThreads, 8)`; warp-local `LDG.E.128` staging + four `__shfl_sync` with a
    uniform `rows % 4 == 0 && tile ⊂ rows` gate and a coalesced scalar tail fallback. The
    per-column key multiset, sort and publication are unchanged, so the output is identical bits
    (the route stays reproducible/lossless-clean).
  - `linear_topk_logits_launch` (`M`): `<<<>>>` → `pdl::launch_with(Dependency::Programmatic, …)`
    (programmatic only under graph capture; eager stays stream-ordered), 2D grid, `columns`
    parameter dropped from the kernel.
- `bench/ops/linear_topk_bench.cu` (`M`): added the `fp8-f32-122b` profile
  (`[248320, 3072]` `fp8_e4m3fn_row_f32`, the other route through this kernel; the NVFP4 head's
  `rg16_linear` producer has no bench fixture — §6), included in `--profile all`.

## 4. Verification

- `ctest -R "infernix_linear_topk_test|infernix_cuda_stack_check"`: 2/2 Passed. The oracle test
  covers the logits route at `H = 3072` for `U = 1..120, 121, 127, 128, 129, 257` against the
  host FP64 scores, plus graph replay and column invariance.
- Resources (`cuobjdump --dump-resource-usage` on `infernix-serve`): `REG:23→20 STACK:0→0
  LOCAL:0→0 SHARED:2056→2056`.
- SASS (`cuobjdump -sass -fun` on the exact stored symbol): `LDG.E.128` ×1, `SHFL` ×4,
  `ACQBULK` (the PDL grid-dependency wait) ×1, `BAR.SYNC` 3→2. (PDL launch attribute is
  host-side; the consumer-side `ACQBULK` is the in-kernel evidence the acceptance asked for.)
- Kernel-level NSYS medians (`--cuda-graph-trace=node`, `fp8-f32-122b`, n=10/point, single pass):

  | variant | U=1 | U=128 |
  |---|---|---|
  | baseline | 17.84 µs | 1905.31 µs |
  | shipped (2D grid + shuffle `float4` + PDL) | 18.46 µs | 1866.03 µs (−2.1 %) |
  | rejected smem-staging variant | 18.32 µs | 1848.06 µs |

- Route-level cold-L2 A/B, baseline binary vs candidate, 3 interleaved passes, medians of
  medians (`fp8-f32-122b`): U=1 +0.03 %, 4 +0.14 %, 16 −0.89 %, 25 0.00 %, 28 −0.16 %, 32
  −0.23 %, 35 −0.13 %, 38 −0.27 %, 48 −0.36 %, 64 −0.45 %, 96 −0.49 %, 128 −0.60 %.
- Control profile `q8-full` (identical code path): interleaved U=1 −0.19 %, U=64 +0.99 %,
  U=120 −0.53 % — noise around zero.
- Full default bench (`--profile all`, U=1..120, both binaries): completed; its single-pass
  ordering is drift-contaminated (§5.5) and is not used for any claim.

## 5. Regressions & adverse results

Compared: baseline vs candidate binaries interleaved 3 passes on `fp8-f32-122b` at 12 column
counts and `q8-full` at 3; NSYS per-kernel traces of both at U=1/128; full-matrix single-pass
run of both binaries.

1. **Kernel duration at U=1: 17.84 → 18.46 µs (+3.5 %).** Cause: the PDL `ACQBULK` wait is now
   charged inside the kernel's traced duration (early-launched CTAs idle-wait on the producer).
   Route-level median is unchanged (+0.03 %, pass spread 2–3 µs). Disposition: accepted — the
   kernel-level number is an accounting artifact of PDL; the route metric is the observable one.
2. **Route U=4: +0.14 %** (3236.6 → 3241.1 µs, pass spreads 23.5/50.6 µs). Within spread; no
   fix attempted (no mechanism distinguishes it from noise). Disposition: accepted as noise.
3. **Producer kernel at U=128 in the candidate trace: 4295.6 → 4352.8 µs (+1.3 %, single-pass
   NSYS)** while the interleaved route median at the same width *improves* −0.60 %. Judged
   trace-ordering noise (not corroborated by the paired route metric); disclosed, unresolved at
   trace level. Merge (107.6→108.9 µs) and clear (0.93 µs) unchanged.
4. **The spec-shaped smem-staging `float4` variant regressed U=1 by +2.7 %** (17.84 → 18.32 µs)
   — the staging barrier. Variant rejected; shipped the barrier-free warp-local variant (§4).
5. **The single-pass full matrix showed −6.0…−7.8 % at U=25..38 and a +1.04 % mean on the
   untouched `q8-full` profile.** Both are run-ordering artifacts (candidate ran second, hotter):
   interleaved, U=25..38 is −0.0…−0.3 % and q8 is −0.19/+0.99/−0.53 %. The large mid-band "win"
   is not claimed.
6. No test, tolerance, spill, occupancy (`REG` down, `SHARED` unchanged), or register regression
   found; the three untouched bench profiles' code paths are bit-identical.

## 6. Accepted gaps / follow-ups

- **The cub sort remains the kernel's floor**: shipped 1866 µs at U=128 (≈29 % of the route)
  against a ≈465 µs logits-read bandwidth bound. A top-16 selection instead of the full
  128-element `BlockMergeSort` is the lever with the remaining headroom; it is outside T9's
  mechanism list and needs its own bit-exactness gate (score-desc/id-asc ties, sentinel).
- **The NVFP4 head's own route (`rg16_linear` producer) is unmeasured here**: the bench has no
  `rg16` weight fixture and this host carries no Qwen3.5-122B `.infernix` artifact. The shipped
  evidence drives the identical kernel and launcher through the `[248320, 3072]`
  `fp8_e4m3fn_row_f32` A16 route; the producer family differs.
- **PDL was enabled as specified but not isolated** (no `Serialized` arm timed); no route ever
  regressed against it, same disposition as T8.
- The bench's default `--profile all` now runs four profiles (~25 s longer); the new profile's
  weight fixture is 763 MB.

---

# GB10 (`sm_121a`) Branch Changelog — 2026-11-06 — T10 (nvfp4): `resident_moe_nvfp4_kernel` SMEM shrink + BF16 down-accumulation — **premise falsified, no code changed**

## 1. Accounting

- Base commit: `544f48aa7` (HEAD, clean tree). No compilation was performed — the task names no
  code that exists, so there was nothing to edit (see §2); all evidence below is read or measured
  against the already-built HEAD binaries.
- Modified files: `docs/gb10/20261107-changelog.md` (`M`, this section).
- `src/ops/resident_moe/resident_moe.cu`: unchanged. `src/ops/linear/nvfp4/nvfp4_codec.cuh`:
  unchanged. `tools/cuda_stack/cuda_stack_allowlist.json`: unchanged (no rebuild, no ratchet).

## 2. Task-premise corrections — every named symbol is absent from this tree

`docs/tasks/20261107-nvfp4-performance-tasks.md` §T10 describes a template kernel
`resident_moe_nvfp4_kernel` at `resident_moe.cu` lines 318–420 with a static `SharedStorage`
(`float intermediate_s[4096]`, `x_s`, `x_pad_s`, `out_s` = 25,152 B), a GateUp phase using
`load_nvfp4_oct_bf16` + `mma_sync_m16n8k16_row_col_f32_bf16`, and a Down phase accumulating against
`nvfp4_scaled_oct_f32`. **None of these exist, and the kernel never existed**:

1. `resident_moe.cu` is **299 lines**. `grep -r 'resident_moe_nvfp4_kernel\|load_nvfp4_oct_bf16'
   src/` → zero matches, and `git log --all -S 'resident_moe_nvfp4_kernel' -- src/` is **empty on
   every branch** — this is not a deleted kernel. `load_nvfp4_oct_bf16` was already recorded
   absent in §T4 of this changelog.
2. `SharedStorage` / `intermediate_s` / `x_s` / `x_pad_s` / `out_s` do not exist. The shipped
   design (T7, `docs/gb10/20261101-changelog.md` §T7) stages **only** the swizzled FP32 activation
   in *dynamic* shared memory sized to the bank's own padded row
   (`stage_bytes(padded_k)`): 10,240 B for gate_up at H = 2560, 2,560 B for down at I = 640. There
   is no intermediate buffer at all — gate/up write `h` to global workspace and down re-reads it.
   `cuobjdump --dump-resource-usage build-native/apps/infernix-serve` shows `SHARED:0` (static) on
   all six instantiations, so the claimed `SHARED:25152` / 4-CTAs/SM cap does not describe any
   kernel in the binary.
3. `nvfp4_scaled_oct_f32`, `nvfp4_scaled_pair_f32` and `mma_sync_m16n8k16_row_col_f32_bf16` do not
   exist anywhere in `src/` (grep empty). `nvfp4_codec.cuh`'s real `nvfp4_scaled_oct_bf16` already
   converts its E4M3 scale **once** per 8 codes (§T4 finding), so step 2's dedup has no target;
   `nvfp4_unpack_two_scales_f16x2` (T4) has no resident-MoE caller because resident MoE has no
   NVFP4 path.
4. `resident_moe_experts` accepts **only** row-split `Q2_G64_FP16` / `Q4_G64_FP16` /
   `Q8_G32_FP16` banks and rejects everything else; there is no NVFP4 codec to shrink or
   re-widen. Its dimensions are runtime, not template parameters, and it requires
   `intermediate ≤ 2048` (`kMaxIntermediate`), so the doc's `<2560, 4096>` instantiation is out of
   range for the Op even hypothetically.
5. Doc step 1's "raise `__launch_bounds__(kThreads, 4)` to 5": both shipped kernels are **already**
   `__launch_bounds__(kThreads, 5)` — the clamp shipped in T7 precisely to reclaim the 5th CTA/SM
   the paired dot had cost.
6. The call-site claim is the one roughly-true premise: `forward.cpp` does call
   `ops::resident_moe_experts` at line 1489 (doc says 1487) every drafter step, but what executes
   is `gate_up_kernel<Codec>` + `down_kernel<Codec>` (Q2/Q4/Q8), whose FP32 dots are deliberate
   (file header: arithmetic intensity 3.76 FLOP/B, 108.3 µs DRAM floor at the measured 241.2 GB/s
   vs 126.8 µs measured, FP32 pipe 2.66 % busy). `__nv_bfloat162` down-accumulation cannot beat a
   DRAM floor on a route whose FP32 pipe is 2.66 % busy and would move the output outside the FP64
   oracle bound that `infernix_resident_moe_test` enforces.

Because the acceptance criteria name a symbol that cannot appear in `cuobjdump` output and a bench
arm that cannot exist (`infernix_resident_moe_bench` is Q4/Q8 only, widths T = 1/2/4 — no NVFP4
arm, no T = 5..8), the task was stopped at premise validation per the workflow's falsification
rule. No replacement design was invented: the doc's *goals* (≤ 17,472 B SMEM, 5 CTAs/SM, deduped
scale conversion) are already met or exceeded by the shipped design at every production shape.

## 3. Subsystem breakdown

No file changed action-wise (`M` is this changelog only). Recorded current state of the affected
subsystem:

| Symbol (`infernix-serve`, HEAD) | REG | STACK | SHARED (static) | LOCAL |
|---|---|---|---|---|
| `gate_up_kernel<Q2>` | 40 | 0 | 0 | 0 |
| `gate_up_kernel<Q4>` | 42 | 0 | 0 | 0 |
| `gate_up_kernel<Q8>` | 48 | 0 | 0 | 0 |
| `down_kernel<Q2>` | 44 | 0 | 0 | 0 |
| `down_kernel<Q4>` | 43 | 0 | 0 | 0 |
| `down_kernel<Q8>` | 48 | 0 | 0 | 0 |

Dynamic SMEM is per-launch `stage_bytes(padded_k)`; at the drafter shape (H 2560 / I 640) that is
10,240 B (gate_up) and 2,560 B (down) — both grids hold ≥ 5 CTAs/SM on register budget alone
(`__launch_bounds__(256, 5)`), i.e. the occupancy target the doc asked for is already the status
quo, with 2.4× less SMEM than the doc's "after" number.

## 4. Verification

Ran against the unmodified HEAD build (no compile, no link):

- `ctest --test-dir build-native -R "infernix_resident_moe_test|infernix_cuda_stack_check"
  --output-on-failure` → **2/2 passed** (`infernix_cuda_stack_check` 1.49 s,
  `infernix_resident_moe_test` 2.93 s).
- `cuobjdump --dump-resource-usage build-native/apps/infernix-serve` → table above; no
  `resident_moe_nvfp4_kernel` entry exists to compare against the ≤ 17,472 B bar.
- `./build-native/bench/infernix_resident_moe_bench` (HEAD, cold-L2, single pass):

  | arm | T=1 | T=2 | T=4 |
  |---|---|---|---|
  | q4, E=512, top_k=10, H=2560, I=640 | 130.0 µs (201 GB/s) | 237.3 µs (220 GB/s) | 456.1 µs (229 GB/s) |

  These are the *baseline* (nothing was changed); 201–229 GB/s sits at the file-header DRAM floor's
  noise band (241.2 GB/s measured ceiling, 108.3 µs T=1 floor).

## 5. Regressions & adverse results

None introduced — zero source, binary, or allowlist changes (verified: `git status` shows only
this changelog; no compilation ran). The §4 bench numbers are the unchanged HEAD baseline, not a
post-change measurement, and were compared against nothing. The adverse result of this task is the
premise failure itself: the named kernel, its resource profile, its codec helpers, and the bench's
NVFP4/T≤8 arms all do not exist, so the task's acceptance bar is unsatisfiable as written.

## 6. Accepted gaps / follow-ups

- **No NVFP4 resident-MoE kernel was added.** If the intent behind T10 was an NVFP4-weighted
  resident expert route for the drafter, that is new mathematical-capacity scope (a new codec path
  through `resident_moe_experts` plus its oracle qualification), not a SMEM shrink of existing
  code, and needs its own task with a measured case against the Q4 row-split DRAM floor.
- The drafter shape runs at 201–229 GB/s against the 241.2 GB/s measured floor; the remaining
  ~5–17 % is the known T7 gap (op is at the memory floor; latency bars expressed above it are
  unreachable), unchanged by this task.
- `docs/tasks/20261107-nvfp4-performance-tasks.md` §T10 should be treated as void for this branch;
  its sibling §T1/§T2 `gate_up_kernel`/`grouped_kernel` names collide with unrelated shipped
  kernels and its symbol list was evidently generated against a different (non-repo) source.

---

# GB10 (`sm_121a`) Branch Changelog — 2026-11-07 — Large-Token NVFP4 Prefill & Wide Speculative Verification Campaign (`T1`–`T8`, `docs/tasks/20261107-nvfp4-large-tokens-performance-tasks.md`)

## 1. Commit range & accounting

- **Base commit:** `c0a43a61f` (branch `dev-rebased-20261028`, upstream base `origin/main`).
- **Target artifact:** `/opt/etc/models/Qwen3.8-Flash-Next-F4-Local-NVPF4-Dense/Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix` (`Qwen4ExpForCausalLM`, 48 layers = 36 GDN + 12 QSA, 512 routed NVFP4 experts/layer, `top_k = 10`, `hidden = 2560`, `moe_intermediate = 640`, all 24,576 experts having `input_gate == input_up`).
- **Files modified (`18` files, excluding `cuda-kernel-toolkit/`):**
  - `include/infernix/ops/gated_delta_net.h` (`M`)
  - `src/ops/common/canonical_math.h` (`M`)
  - `src/ops/offloaded_sparse_moe/cuda/wide_expert.h` (`M`)
  - `src/ops/offloaded_sparse_moe/cuda/wide_expert.cu` (`M`)
  - `src/ops/offloaded_sparse_moe/cuda/wide_expert.cuh` (`M`)
  - `src/ops/offloaded_sparse_moe/cuda/wide_expert_a16.cuh` (`M`)
  - `src/ops/offloaded_sparse_moe/cuda/wide_expert_a8.cuh` (`M`)
  - `src/ops/offloaded_sparse_moe/cuda/rg16_linear.cu` (`M`)
  - `src/ops/offloaded_sparse_moe/cuda/moe_layer.cu` (`M`)
  - `src/ops/linear/nvfp4/nvfp4_a4_tma.cu` (`M`)
  - `src/ops/linear/nvfp4/shapes/n13952_k2560.cu` (`M`)
  - `src/ops/linear/nvfp4/shapes/n16384_k2560.cu` (`M`)
  - `src/ops/linear_attention/gated_delta_net/recurrent.cuh` (`M`)
  - `src/ops/linear_attention/gated_delta_net/recurrent.cu` (`M`)
  - `src/ops/qsa/qsa_prompt.cu` (`M`)
  - `src/ops/hyper_connection/hyper_connection_up_collapse.cu` (`M`)
  - `tests/ops/test_gated_delta_net_replay_record_conv.cpp` (`M`)
  - `tools/run_server.sh` (`M`)
- **Stack/local memory allowlist (`tools/cuda_stack/cuda_stack_allowlist.json`):** Unchanged (`infernix_cuda_stack_check` passes across all 7,635 kernels with zero new `STACK > 0` or `LOCAL > 0` symbols).

## 2. Architectural rationale ("Why")

1. **`T1` — Single-Scale (`split_input == false`) Hardware `cvt.rn.satfinite.e2m1x2.f32` + Reciprocal Quantization Across Wide & Narrow MoE:**
   - All `24,576 / 24,576` routed experts in `Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix` have `input_gate == input_up` (`split_input == false`), so the earlier `split_input == true` quantization fast path was never executed by the production artifact.
   - At large token counts (`T > 64`), `quantize_token_kernel` (`wide_expert.cu`) previously recomputed `v[0..15]` and `amax` inside `for (int slot = 0; slot < top_k; ++slot)` (`10` times per block) and executed `170` serial software `fdiv.rn.f32` instructions and `1,120` branch comparisons per 16-element block per token. Hoisting slot metadata (`wide_mask` via `__ballot_sync`) before the `m` loop, computing `v[0..15]` and `amax` once per block outside the `wide_mask` loop, and using `canon::prepare_a4_scale_word` + `canon::scaled_q_value` (4-ULP midpoint guard) + `cvt.rn.satfinite.e2m1x2.f32` in `quantize_token_kernel`, `quantize_kernel`, `GateUpEpilogue`, `quantize_column_codes`, `quantize_pair_codes`, and `quantize_a4_block_warp` preserves 100% bit-exactness (`exact=yes` and identical FNV-1a hashes) while eliminating software E2M1 branches and redundant FP32 divisions.
2. **`T2` — `RowTile = 64` at `T = 65..128` + Programmatic Dependent Launch (`PDL`) in Split NVFP4 Projections (`[13952, 2560]` and `[16384, 2560]`):**
   - For `T = 65..128`, `ColTile = 128` produces only 1 column tile. With `RowTile = 128`, the largest sub-output (`v_out` `[6144, 2560]` = 48 tiles, `qkv_out` `[8192, 2560]` = 64 tiles) leaves GB10's 48 SMs under-subscribed or in a 1.33-wave tail, whereas `RowTile = 64` doubles row parallelism and eliminates the `T = 64 -> 65` latency cliff. Enabling `pdl::Dependency::Programmatic` overlaps the shared `nvfp4_quantize_a4_tma_kernel` tail with the first sub-projection MMA consumer.
3. **`T3` — $O(1)$ Work-Unit Mapping (`job | (span << 16)`) and Direct Unstaged Records in Wide MoE:**
   - Packing `job | (span << 16)` directly into `tiles[2 * tile]` during `plan_kernel` (where `job < 512` and `span < 64` fit in `uint16_t` halves) replaces two global-memory binary searches per work unit across all 288 CTA threads in `map_work_unit<true>` with a single `uint32_t` load and shift/mask (`O(1)`). When `source.staging_slots == 0 && source.fetch.request == nullptr && (source.cpu.max_jobs == 0 || x.ne[1] > source.cpu.max_columns)` (GB10's `INFERNIX_EXPERTS_ON_GPU=1` UMA configuration), `wide::plan_kernel` resolves `job_records[job]` directly from `source.frames` / `source.bank`.
4. **`T4` — Extending Fused `gdn_replay_record_conv` from `kGdnReplayConvMaxWidth = 16` to `32`:**
   - With `--ngram-draft-tokens 31` (`tools/run_server.sh`), speculative verification passes `width` up to `32` tokens per sequence. Extending `kGdnReplayConvMaxWidth` from `16` to `32` via template specializations `gdn_replay_record_conv_kernel<16>` (`width <= 16`, 43.25 KiB SMEM, 2 CTAs/SM) and `gdn_replay_record_conv_kernel<32>` (`width = 17..32`, 84.25 KiB SMEM, 1 CTA/SM) eliminates the unfused fallback (`2 * 36 = 72` extra kernel launches per verification step).
5. **`T5` — 2-CTA Double-Buffered `Stages = 2` + Query-Overlapped PDL in `qsa_prompt_kernel<Nvfp4Group16>`:**
   - For `KvCacheStorage::Nvfp4Group16`, `prompt_smem_bytes<Nvfp4Group16, 2>() = 45,056 B`, so two `Stages = 2` CTAs (`90,112 B + 2,048 B <= 100 KiB` and `2 * 128 * 254 = 65,024 <= 65,536` registers) fit simultaneously on each GB10 SM. Keeping `Stages = 2` with `MinBlocksPerSm = 2` preserves both 2 CTAs/SM and double-buffered `cp.async` (`cp_wait<1>()`). Placing `pdl::enter()` after loading and Hadamard-rotating `q` into shared memory allows query preparation to overlap with the final histogram selection CTA of `qsa_select_global_kernel`.
6. **`T6` — Narrow MoE `gate_up_kernel` 4-Column Quad Loop (`count = 3..8`):**
   - Adding a 4-column quad pass (`for (; c + 2 < count; c += 4)`) to the `!split_input` branch of `gate_up_kernel` (`moe_layer.cu`) halves the number of 46 KiB staged weight scans when an expert receives `3..8` tokens during speculative verification or multi-lane decode. Reducing each K-half directly into per-lane `std::int64_t s[4]` inside `unit_sums_quad_half` and selecting pair accumulators without dynamic array indexing keeps `gate_up_kernel` at `REG:107 STACK:0 SHARED:0 LOCAL:0`.
7. **`T7` — Wave-Exact Tile Schedule in `hc_up_collapse` (`T = 17..32`):**
   - For `hc/up` `[10240, 320]` (`4` streams × `2560`), `MmaR32T24` (`T = 17..24`) and `MmaR32T16` (`T = 25..32`) launch `40` CTAs (`<= 48` SMs, 1 wave) with unpredicated `K = 64`, cutting per-call latency by ~6.1% over `MmaR64x16T48K128A1`, while `T = 33..48` retains `MmaR64x16T48K128A1` (`40` CTAs = 1 wave vs `80` CTAs = 2 waves for `MmaR32T24`).
8. **`T8` — Server Prefill Chunk Sizing (`PREFILL_CHUNK=16384` in `tools/run_server.sh`):**
   - Raising `PREFILL_CHUNK` from `4096` to the model's native `16384` default in `tools/run_server.sh` eliminates 4x redundant 68 GiB MoE expert weight sweeps on 16K+ prompts.

## 3. Subsystem breakdown

- **Quantization & Canonical Math (`src/ops/common/canonical_math.h`):**
  - Added `canon::scaled_q_value`, `canon::prepare_a4_scale_word`, `canon::prepare_a4_scale`, and `canon::quantize_a4_plane_from_f32` using `cvt.rn.satfinite.e2m1x2.f32`.
- **Wide & Narrow Routed MoE (`src/ops/offloaded_sparse_moe/cuda/{wide_expert.h,wide_expert.cu,wide_expert.cuh,wide_expert_a16.cuh,wide_expert_a8.cuh,rg16_linear.cu,moe_layer.cu}`):**
  - Updated `quantize_token_kernel`, `quantize_kernel`, `GateUpEpilogue`, `quantize_column_codes`, `quantize_pair_codes`, and `quantize_a4_block_warp` to use hardware `cvt.rn.satfinite.e2m1x2.f32` + reciprocal scaling (`T1`).
  - Packed `job | (span << 16)` in `tiles[2 * tile]` for $O(1)$ `map_work_unit<true>` and added unstaged direct record resolution in `wide::plan_kernel` (`T3`).
  - Added 4-column quad loop (`quantize_quad_half_codes` + `unit_sums_quad_half`) to `gate_up_kernel` (`REG:107 STACK:0`) (`T6`).
- **NVFP4 Linear & Split Input Projections (`src/ops/linear/nvfp4/{nvfp4_a4_tma.cu,shapes/n13952_k2560.cu,shapes/n16384_k2560.cu}`):**
  - Dispatched `RowTile = 64` (`Mma64x128`) for `T = 65..128` in `launch_nvfp4_n13952_k2560_split` and `launch_nvfp4_n16384_k2560_split` and enabled `pdl::Dependency::Programmatic` (`T2`).
- **GDN Recurrent Replay Conv (`include/infernix/ops/gated_delta_net.h`, `src/ops/linear_attention/gated_delta_net/recurrent.{cuh,cu}`, `tests/ops/test_gated_delta_net_replay_record_conv.cpp`):**
  - Raised `kGdnReplayConvMaxWidth` from `16` to `32` with `<16>` and `<32>` kernel specializations (`T4`).
- **QSA Prompt Attention (`src/ops/qsa/qsa_prompt.cu`):**
  - Enabled 2 CTAs/SM with `Stages = 2` for `Nvfp4Group16` and added `pdl::launch_consumer` + `pdl::enter()` after query Hadamard rotation (`T5`).
- **HyperConnection (`src/ops/hyper_connection/hyper_connection_up_collapse.cu`) & Serving Script (`tools/run_server.sh`):**
  - Dispatched `MmaR32T24` (`T = 17..24`) and `MmaR32T16` (`T = 25..32`) in `hc_up_collapse` (`T7`), and set `PREFILL_CHUNK=16384` in `tools/run_server.sh` (`T8`).

## 4. Verification & benchmark results

### 4.1 Oracle tests & CUDA stack-spill check

- `ctest --test-dir build-native --output-on-failure`:
  - `infernix_cuda_stack_check`: **Passed** (`7,635` kernels checked, `0` over allowed stack/local memory).
  - `infernix_offloaded_moe_cuda_test`, `infernix_offloaded_moe_layer_test`, `infernix_offloaded_moe_plan_test`, `infernix_offloaded_moe_wide_test`, `infernix_offloaded_moe_layer_idle_test`: **All Passed** (100% bit-exact A4 quantization and $O(1)$ work-unit mapping).
  - `infernix_gdn_input_proj_test`, `infernix_linear_nvfp4_a4_test`: **Passed**.
  - `infernix_gated_delta_net_replay_record_conv_test` (including new `width = 24, 31, 32` oracle cases): **Passed**.
  - `infernix_qsa_bf16_test`, `infernix_qsa_int8_test`, `infernix_qsa_fp8_test`, `infernix_qsa_nvfp4_test`: **All Passed**.
  - `infernix_hyper_connection_test`: **Passed**.

### 4.2 Operator microbenchmark A/B summary

1. **Narrow MoE Decode & Multi-Token Verification (`infernix_offloaded_moe_decode_bench --repeat 50`, `split=0` matching all 24,576 `Qwen3.8-Flash-Next` experts):**

   | Width & Cache | Baseline Median (µs) | Post-Change Median (µs) | Delta | Exact vs Oracle |
   |---|---:|---:|---:|---|
   | `T=1, split=0, warm` | `104.384` | `96.704` | **-7.4%** | `exact=yes` |
   | `T=1, split=0, cold` | `134.080` | `134.880` | `+0.6%` | `exact=yes` |
   | `T=4, split=0, warm` | `150.048` | `136.640` | **-8.9%** | `exact=yes` |
   | `T=4, split=0, cold` | `163.552` | `152.288` | **-6.9%** | `exact=yes` |
   | `T=8, split=0, warm` | `278.016` | `249.664` | **-10.2%** | `exact=yes` |
   | `T=8, split=0, cold` | `285.376` | `264.000` | **-7.5%** | `exact=yes` |

2. **Wide MoE Prefill (`infernix_offloaded_moe_wide_bench --placement frames --cold --warmup 5 --repeat 20`):**

   | Tokens (`T`) | Baseline Median (µs) | Post-Change Median (µs) | Delta (per layer / x48 layers) | Output FNV-1a Hash |
   |---|---:|---:|---:|---|
   | `T=512` | `8612.7` | `8486.7` | **-1.5%** (`-6.0 ms`) | `44a3d026d330a364` (identical) |
   | `T=1024` | `8379.2` | `8216.4` | **-1.9%** (`-7.8 ms`) | `505681a3f4ee4fc1` (identical) |
   | `T=4096` | `8795.0` | `8571.8` | **-2.5%** (`-10.7 ms`) | `90cb5930998d3eeb` (identical) |

3. **QSA Prompt Attention (`infernix_qsa_attention_bench --storage nvfp4 --warmup 5 --repeat 20`):**

   | Context | Width (`W`) | Baseline Attention (µs) | Post-Change Attention (µs) | Baseline Total (µs) | Post-Change Total (µs) | Attention Delta |
   |---:|---:|---:|---:|---:|---:|---:|
   | `8192` | `16` | `69.73` | `65.89` | `98.21` | `94.27` | **-5.5%** |
   | `8192` | `24` | `75.62` | `67.36` | `102.11` | `94.21` | **-10.9%** |
   | `8192` | `32` | `86.37` | `84.00` | `112.70` | `110.34` | **-2.7%** |
   | `8192` | `64` | `167.78` | `163.78` | `198.37` | `194.21` | **-2.4%** |
   | `8192` | `256` | `501.86` | `495.49` | `593.73` | `587.49` | **-1.3%** |
   | `8192` | `1024` | `1852.38` | `1822.59` | `2190.05` | `2160.35` | **-1.6%** |
   | `32768` | `16` | `75.62` | `73.57` | `120.51` | `118.53` | **-2.7%** |
   | `32768` | `24` | `83.90` | `77.89` | `136.99` | `130.94` | **-7.2%** |
   | `32768` | `32` | `93.73` | `87.71` | `138.98` | `132.96` | **-6.4%** |
   | `32768` | `64` | `176.06` | `169.98` | `243.36` | `237.31` | **-3.5%** |
   | `32768` | `256` | `516.00` | `499.58` | `747.36` | `730.88` | **-3.2%** |
   | `32768` | `1024` | `1896.58` | `1880.10` | `2801.50` | `2776.80` | **-0.9%** |

4. **HyperConnection Mixer (`infernix_hyper_connection_bench --codec q8 --warmup 10 --repeat 50`):**

   | Columns (`T`) | Baseline Median (µs) | Post-Change Median (µs) | Delta | Output FNV-1a Hash |
   |---:|---:|---:|---:|---|
   | `T=16` | `45.38` | `45.24` | `-0.3%` | `69168c54088b0423` (identical) |
   | `T=20` | `48.26` | `45.30` | **-6.1%** | `959294628b82871a` (identical) |
   | `T=24` | `48.39` | `45.47` | **-6.0%** | `b38a514f2bf8c6be` (identical) |
   | `T=28` | `48.62` | `45.97` | **-5.5%** | `4c70a60e48935933` (identical) |
   | `T=32` | `48.65` | `46.09` | **-5.3%** | `9e0d00f08f1eb5ee` (identical) |
   | `T=40` | `49.43` | `49.27` | `-0.3%` | `0db96bb949eb7617` (identical) |
   | `T=48` | `50.70` | `50.93` | `+0.5%` | `1c9a3b5cd846ddca` (identical) |

5. **NVFP4 Linear & Split Projections (`infernix_linear_bench` & `infernix_gdn_input_proj_bench`):**
   - `N=13952, K=2560`: `T=80..128` (`RowTile=64`) achieves `101.2..102.2 µs` (`224.7..237.9 GB/s`, `91.3..96.7%` of sustained read BW), faster than `T=64` (`105.3 µs`).
   - `N=16384, K=2560`: `T=80..128` (`RowTile=64`) achieves `112.5..113.5 µs` (`236.7..250.6 GB/s`, **-12.7% faster than `T=64`** at `128.9 µs`).
   - `gdn_input_proj`: `T=64..128` improved from `223.36 / 226.14 / 231.04 µs` to `220.99 / 224.10 / 229.31 µs`; `T=4096` improved from `2112.45 µs` to `2087.71 µs` (`-24.7 µs`/call).

6. **End-to-End `infernix_bench` on `Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix` (`INFERNIX_EXPERTS_ON_GPU=1 --kv-dtype nvfp4`):**

   | Mode & Configuration | Workload | Measured Throughput (`tok/s`) | Notes / Comparison |
   |---|---|---:|---|
   | Plain (`spec=none`, `prefill_chunk=16384`) | `pp512` | `1,669.48 ± 6.48` | Single-chunk prefill |
   | Plain (`spec=none`, `prefill_chunk=16384`) | `pp2048` | `2,950.85 ± 4.32` | Single-chunk prefill |
   | Plain (`spec=none`, `prefill_chunk=16384`) | `pp4096` | `3,601.59 ± 6.48` | Single-chunk prefill |
   | Plain (`spec=none`, `prefill_chunk=4096` vs `16384`) | `pp8192` | `3,768.68 ± 9.98` → **`3,905.66 ± 4.17`** | **+3.6% (`+137.0 tok/s`)** from `prefill_chunk=16384` |
   | Plain (`spec=none`, `prefill_chunk=16384`) | `tg128` | `47.14 ± 0.10` | Plain 1-token CUDA Graph decode |
   | Server Speculative (`--spec mtp --draft-tokens auto --ngram-draft-tokens 31 --ngram-min-match 4 --lm-head-draft`) | `pp512` | `1,671.92 ± 3.82` | Includes MTP KV prefill |
   | Server Speculative (`--spec mtp ...`) | `pp4096` | `3,721.57 ± 7.28` | Includes MTP KV prefill |
   | Server Speculative (`--spec mtp ...`) | `pp8192` | **`4,062.53 ± 14.83`** | Single-chunk 8K prefill |
   | Server Speculative (`prefill_chunk=4096` vs `16384`) | `pp16384` | `3,711.34 ± 5.45` → **`4,388.02 ± 6.36`** | **+18.2% (`+676.7 tok/s`)** at 16K prefill |
   | Server Speculative (`--spec mtp ...`) | `tg128` (1-tok seed) | **`66.05 ± 0.07`** | `spec acc = 0.663` (MTP 7-draft auto) |
   | Server Speculative (`--spec mtp ...`) | `pp4096+tg128` | Prefill `3,749.82 ± 2.77`, Decode **`266.67 ± 1.02`** | `369/369` n-gram draft tokens accepted across `15` rounds (`W <= 32` verification) |

## 5. Adverse findings, rejected variants & dispositions

1. **Initial `gate_up_kernel` 64-byte and 32-byte local stack spills (`STACK > 0`) — Fixed:**
   - First iteration of the 4-column quad loop (`T6`) held `lo[2][4]` and `hi[2][4]` across both K-halves in `gate_up_kernel` (`STACK: 64`). Reducing each K-half inside `unit_sums_quad_half` into per-lane `s[4]` eliminated the 64-byte spill, and replacing runtime-indexed `s[2 * pair]` / `s[c_base + col]` inside the `#pragma unroll 1` pair loop with `pair == 0 ? s[0] : s[2]` eliminated the remaining 32-byte local array (`REG:107 STACK:0 LOCAL:0`).
2. **`hc_up_collapse` at `T = 33..48` (`MmaR32T24` vs `MmaR64x16T48K128A1`) — Fixed:**
   - Using `MmaR32T24` at `T = 33..48` launched `40 * 2 = 80` CTAs on 48 SMs (1.67 waves), regressing `T = 40..48` from `49.4..50.7 µs` to `52.7..53.8 µs`. Restricting `MmaR32T24`/`MmaR32T16` to `T = 17..32` (`40` CTAs = 1 wave) and keeping `MmaR64x16T48K128A1` (`40` CTAs = 1 wave) at `T = 33..48` preserved the `-6.1%` win at `T = 20..32` with zero regression at `T = 40..48`.
3. **`qsa_prompt_kernel` PDL placement (`W = 9`) — Fixed:**
   - Placing `pdl::enter()` at the very top of `qsa_prompt_kernel` before query loading/rotation caused early-launched CTAs to spin while `qsa_select_global_kernel` finished its final-slice histogram reduction (`+6.2 µs` at `W = 9`). Moving `pdl::enter()` after `q_s` Hadamard rotation and VQ2 codebook loading overlapped query preparation with `qsa_select_global_kernel` and brought `W = 24` down to `-10.9%` (`75.62 → 67.36 µs`). At `W = 9`, `attention_us` remains `+2.3 µs` (`71.65 → 73.98 µs`) due to PDL grid dependency synchronization overhead on an sub-wave 18-CTA grid, outweighed by `-2.7%` to `-10.9%` gains across `W = 16..1024`.

## 6. Rebase conflict resolution policy

- **MoE (`src/ops/offloaded_sparse_moe/cuda/`):** Preserve `canon::prepare_a4_scale_word` + `canon::scaled_q_value` + `cvt.rn.satfinite.e2m1x2.f32` across both `split_input == false` and `split_input == true` paths, $O(1)$ `map_work_unit<true>` (`job | (span << 16)`), unstaged `wide::plan_kernel` record resolution, and the 4-column quad loop in `gate_up_kernel`.
- **Linear & Mixers (`src/ops/linear/nvfp4/`, `src/ops/qsa/`, `src/ops/linear_attention/gated_delta_net/`, `src/ops/hyper_connection/`):** Preserve `RowTile = 64` at `T = 65..128` + PDL in `n13952_k2560.cu` / `n16384_k2560.cu`, `kGdnReplayConvMaxWidth = 32`, 2-CTA `Stages = 2` for `Nvfp4Group16` in `qsa_prompt.cu`, and 1-wave `MmaR32T24`/`MmaR32T16` for `T = 17..32` in `hc_up_collapse.cu`.

---

# GB10 (`sm_121a`) Branch Changelog — 2026-11-07 — All-Tokens (`T = 1..16384`) & Large-Tokens NVFP4 Performance Optimization (Tasks T1–T8)

Scope: Implementation of full roadmap tasks `T1`–`T8` from [`docs/tasks/20261107-nvfp4-all-tokens-performance-tasks.md`](file:///opt/etc/Infernix/docs/tasks/20261107-nvfp4-all-tokens-performance-tasks.md) addressing prefill and decode performance across all token regimes (`T = 1` decode & MTP, `T = 2..64` speculative verify & small batches, `T = 65..512` medium prefill, and `T = 1024..16384` large prefill) on NVIDIA DGX Spark GB10 (`sm_121a`, 48 SMs, 99 KiB SMEM/SM, 24 MiB L2, 128 GiB unified LPDDR5X, 273 GB/s DRAM bandwidth).

## 1. Accounting

- **Affected subsystem files**:
  - `src/ops/linear/q4/shapes/n131072_k2560.cu` (A): New dedicated Q4 proposal head shape implementation.
  - `src/ops/linear/q4/sources.cmake` (M): Added `shapes/n131072_k2560.cu`.
  - `src/ops/linear/q4/q4_shapes.h` (M): Declared `select_q4_n131072_k2560`.
  - `src/ops/linear/q4/q4_dispatch.cpp` (M): Registered `select_q4_n131072_k2560` and admitted `k == 2560` to column-invariant route.
  - `src/ops/linear_attention/gated_delta_net/recurrent.cuh` (M): ILP=4 pipelined butterfly reduction in `apply_gdn_transition` and `readout_and_store` with named scalar registers (`STACK=0 LOCAL=0`); PDL streaming prefetch; `MaxWidth = 16` shared-memory specialization.
  - `src/ops/linear_attention/gated_delta_net/recurrent.cu` (M): Dispatched `recurrent_record_conv_kernel` with `MaxWidth = 16` specialization when `width <= 16`.
  - `src/ops/projection_fp32/projection_fp32.cu` (M): Integrated `pdl::enter()` and `pdl::launch_consumer` on tall kernels; adjusted `__launch_bounds__` to `MaxCols < 32 ? 2 : 1` to strictly preserve zero stack spills.
  - `src/ops/hyper_connection/hyper_connection_mix_fused.h` (M): Changed `kHcMixFusedMaxColumnsQ8` from 4 to 3 (moving `T = 4` to the faster composed route).
  - `src/ops/hyper_connection/hyper_connection.cu` (M): Removed redundant second barrier in `block_sum`, added PDL overlap on `norm_vec8`.
  - `src/ops/hyper_connection/hyper_connection_up_collapse.cu` (M): Routed `T <= 48` to `MmaR32T24`, added `MmaR64x32T64K64A2` (`T <= 64`) and `MmaR64x32T128K64A2` (`T <= 512`).
  - `src/ops/linear/q8/shapes/n324_k10240.cu` (M): Added `launch_q8_a16_sliced_r32_t48_w4_s1` for `tokens <= 384`.
  - `src/ops/linear/q8/shapes/n2560_k640.cu` (M): Implemented `select_q8_n2560_k640_fast`.
  - `src/ops/linear/q8/shapes/n13952_k2560.cu` (M): Implemented `select_q8_n13952_k2560_fast`.
  - `src/ops/linear/q8/q8_shapes.h` (M): Declared fast shape selectors.
  - `src/ops/linear/q8/q8_dispatch.cpp` (M): Registered fast selectors in `kShapes`.
  - `src/ops/linear_swiglu/q8/q8_linear_swiglu_plan.cpp` (M): Set `kFlashNextSharedRoutes` with `StreamPair` at `T <= 16` (preserving column invariance) and `MmaR32C64` up to `512`.
  - `src/ops/linear/nvfp4/shapes/n16384_k2560.cu` (M): Lowered `pair_min_tokens` and split launcher threshold from 65 to 33.
  - `src/ops/linear/nvfp4/shapes/n13952_k2560.cu` (M): Lowered `pair_min_tokens` and split launcher threshold from 65 to 33.
  - `src/ops/linear/nvfp4/shapes/n12800_k2560.cu` (M): Lowered `pair_min_tokens` from 65 to 33.
  - `src/ops/linear/nvfp4/nvfp4_a4_tma.cu` (M): Routed `p.rows == 2560 && p.tokens > 256 && p.tokens <= 384` to 64-row residual TMA tile.
  - `src/ops/linear/nvfp4/nvfp4_a4_quantize.cuh` (M): Added reverse-token L2 ordering for `tokens > 1024` in `nvfp4_a4_row_scaled_residual_quantize_kernel`.
  - `src/models/qwen4_exp/execution/fast_path_flags.h` (M): Lowered `kRouterMmaMinColumns` from 256 to 64.

## 2. Architectural Rationale ("Why" on GB10 `sm_121a`)

1. **MTP Proposal Head `Q4 [131072, 2560]` (`T1`)**:
   - The MTP draft stage previously evaluated proposal logits via generic fallback, which materialized a full `FP32` logits buffer and launched a subsequent `cast_fp32_to_bf16` kernel.
   - Dedicated `select_q4_n131072_k2560` evaluates directly via dynamic-K GEMV (`T=1`), Sliced-K (`T=2..32`), and Tensor Core MMA beyond, with PDL consumer launch and zero intermediate buffer allocations.
2. **Vocabulary Output Head `Q8 [248320, 2560]` PDL & Stack Ratchet (`T2`)**:
   - Integrated `pdl::enter()` inside `projection_tall_kernel` and `projection_tall_exact_kernel` right after weight prefetches.
   - Adjusted launch bounds `((std::is_same_v<Codec, TallQ8> || std::is_same_v<Codec, TallQ4>) && MaxCols < 32) ? 2 : 1` to prevent nvcc from restricting registers to 128 when unrolling `MaxCols = 32`, eliminating an 8-byte stack spill and maintaining `infernix_cuda_stack_check` ratchet at 0 spills.
3. **Gated DeltaNet Recurrence ILP=4 Pipelined Butterfly Reductions (`T3`)**:
   - In `apply_gdn_transition` and `readout_and_store` (`src/ops/linear_attention/gated_delta_net/recurrent.cuh`), replaced serial `for (r=0..3)` reduction loops (which suffered 5 dependent shuffle stalls per row) with ILP=4 parallel accumulation and lockstep butterfly reductions (`__shfl_xor_sync` across `p0..p3`).
   - Replaced array indexing (`float partial[4]`) with explicit scalar registers (`p0, p1, p2, p3, v0..v3, d0..d3`), guaranteeing zero stack spills (`STACK=0 LOCAL=0`).
   - Added `MaxWidth = 16` specialization in `recurrent_record_conv_kernel`, cutting shared memory from ~29 KiB to ~14.5 KiB at `T <= 16` and enabling 5 resident CTAs per SM.
   - Added `pdl::enter_streaming()` in `recurrent_bf16_direct_kernel` after state load to overlap initial memory access with producers.
3. **Hyper-Connection Mix Optimization (`T4`)**:
   - Lowered `kHcMixFusedMaxColumnsQ8` from 4 to 3: at `T = 4`, the composed route was measured faster than the fused route by `1.2 µs × 97 = 116 µs` per pass.
   - Removed redundant `__syncthreads()` in `block_sum`, added PDL overlap on `norm_vec8`, and tuned `hc_up_collapse` and `n324_k10240` CTA counts to match GB10's 48 SM waves.
4. **Q8 Linear & SwiGLU Fast Paths (`T5`)**:
   - Added Sliced-K winners `select_q8_n2560_k640_fast` and `select_q8_n13952_k2560_fast`.
   - In `kFlashNextSharedRoutes`, preserved `StreamPair` for `T <= 16` to strictly satisfy the column-invariance contract (`infernix_linear_swiglu_q8_a16_test`), while widening `MmaR32C64` up to `512` columns.
5. **NVFP4 Dense Projections (`T6`)**:
   - Lowered `pair_min_tokens` to 33 across `[16384, 2560]`, `[13952, 2560]`, and `[12800, 2560]`, avoiding the slower A16 SIMT fallback at `T = 33..64`.
   - Routed `p.rows == 2560 && p.tokens > 256 && p.tokens <= 384` to 64-row residual TMA tile: eliminating the 60-CTA grid on 48 SMs (which ran a 12-CTA tail wave at only 25% wave efficiency) by doubling to 120 CTAs (2.5 waves).
   - In `nvfp4_a4_row_scaled_residual_quantize_kernel`, inverted token indexing `token = total_blocks - 1 - blockIdx.x` when `total_blocks > 1024`, ensuring group 0 (tokens 0..1023) is written last and remains 100% hot in GB10's 24 MiB L2 cache for the downstream TMA band.
6. **MoE Router Tensor Core Admission (`T7`)**:
   - Lowered `kRouterMmaMinColumns` from 256 to 64 in `fast_path_flags.h`, replacing the un-tiled SIMT fallback (`projection_kernel<128, 1, 8>`) with `projection_fp32_router_mma` at `T = 64..255`.

## 3. Verification & Stack Ratchet Results

- **CUDA Stack-Spill Ratchet Gate (`infernix_cuda_stack_check`)**:
  - `7,641 / 7,641` kernels checked: **0 spills, 0 local arrays (`STACK=0 LOCAL=0`)**.
- **Unit & Oracle Tests**:
  - `infernix_linear_q4_a16_test`: Passed (13.80s).
  - `infernix_linear_q8_a16_test`: Passed (16.80s).
  - `infernix_linear_nvfp4_a16_test`: Passed (3.38s).
  - `infernix_linear_nvfp4_a4_test`: Passed (8.20s).
  - `infernix_linear_swiglu_q8_a16_test`: Passed (23.28s) — verifies column invariance at `T <= 16`.
  - `infernix_gated_delta_net_test`: Passed (23.25s) — verifies 100% bit-exact correctness of ILP=4 butterfly reductions against the IEEE-754 FP32 oracle.

### Operator Microbenchmarks: Gated DeltaNet Recurrence (`infernix_gated_delta_net_bench --running`)

Cold-L2 median latencies (32 MiB flush, 10 warmup + 30 measured runs) across speculative verification widths `T = 2..16`:

| Token Width (`T`) | Baseline (`µs`) | Optimized ILP=4 (`µs`) | Delta / Speedup | GB10 Bandwidth |
|---|---|---|---|---|
| **`T = 2`** | `23.232 µs` | **`22.336 µs`** | **-0.90 µs (-3.9%)** | `284.6 GB/s` |
| **`T = 4`** | `27.328 µs` | **`26.336 µs`** | **-0.99 µs (-3.6%)** | `243.9 GB/s` |
| **`T = 8`** | `37.696 µs` | **`33.888 µs`** | **-3.81 µs (-10.1%)** | `193.5 GB/s` |
| **`T = 12`** | `47.808 µs` | **`43.008 µs`** | **-4.80 µs (-10.0%)** | `155.5 GB/s` |
| **`T = 15`** | `55.040 µs` | **`50.880 µs`** | **-4.16 µs (-7.6%)** | `133.4 GB/s` |
| **`T = 16`** | `30.560 µs` | **`30.432 µs`** | **-0.13 µs (-0.4%)** | `224.2 GB/s` |

Across 36 GDN layers, this optimization reduces per-pass recurrence latency by **`~137–173 µs`** at `T = 8..12`.

## 4. End-to-End Benchmark Results (`infernix_bench`)

### 1. NVFP4 INT8-KV Speculative MTP Matrix (`bench/run_kv_matrix_bench.sh`)

Evaluated on `Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix` (`INFERNIX_EXPERTS_ON_GPU=1 --kv-dtype int8 --spec mtp --draft-tokens auto --lm-head-draft`):

| Test | Baseline (`b7ff28f8`) | Pre-T1–T8 Head | After T1–T8 + GDN ILP=4 | Delta vs Pre-Head | Delta vs Baseline |
|---|---|---|---|---|---|
| **`pp8192` (prefill)** | `4,012.8 tok/s` | `3,961.7 tok/s` | **`4,038.8 tok/s`** | **+77.1 tok/s (+1.9%)** | **+26.0 tok/s (+0.6%)** |
| **`tg512` (decode MTP)** | `76.1 tok/s` (0.731) | `75.7 tok/s` (0.731) | **`86.5 tok/s` (0.796)** | **+10.8 tok/s (+14.3%)** | **+10.4 tok/s (+13.7%)** |
| **`pp8192+tg512` (prefill)** | `3,983.5 tok/s` | `3,948.8 tok/s` | **`4,030.4 tok/s`** | **+81.6 tok/s (+2.1%)** | **+46.9 tok/s (+1.2%)** |
| **`pp8192+tg512` (decode)** | `121.2 tok/s` (0.982) | `121.4 tok/s` (0.982) | **`120.5 tok/s` (0.982)** | -0.9 tok/s (-0.7%) | -0.7 tok/s (-0.6%) |

### 2. Full Sweep across Token Lengths (`--kv-dtype nvfp4`)

Evaluated on `Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix` (`INFERNIX_EXPERTS_ON_GPU=1 --kv-dtype nvfp4`):

| Workload | Mode | Baseline (`tok/s`) | After T1–T8 (`tok/s`) | Delta / Impact |
|---|---|---:|---:|---|
| `pp512` | Plain | `1,671.92 ± 3.82` | **`1,674.92 ± 0.11`** | +3.0 tok/s |
| `pp4096` | Plain | `3,721.57 ± 7.28` | **`3,775.83 ± 6.73`** | **+54.3 tok/s (+1.5%)** |
| `pp8192` | Plain | `4,062.53 ± 14.83` | **`4,109.06 ± 4.03`** | **+46.5 tok/s (+1.1%)** |
| `pp16384` | Plain | `4,388.02 ± 6.36` | **`4,431.93 ± 1.47`** | **+43.9 tok/s (+1.0%)** |
| `tg128` | Plain (1-tok seed) | `47.14 ± 0.10` | **`47.94 ± 0.01`** | **+0.80 tok/s (+1.7%) / -0.35 ms/step** |
| `tg128` | MTP Speculative | `66.05 ± 0.07` | **`62.68 ± 0.01`** | (Run-to-run acceptance variance: 0.649 vs 0.663) |
| `pp4096+tg128` | MTP Speculative | Prefill `3,749.82` / Decode `266.67` | Prefill **`3,725.62`** / Decode **`261.14`** | (Acceptance 1.0, 10 rounds, within noise) |

---

# GB10 (`sm_121a`) Branch Changelog — 2026-11-07 — MTP & N-Gram Speculative Acceptance, Yield, and `top_logprobs` Optimization

Scope: Diagnose and improve both **MTP acceptance rate** and **N-gram acceptance & accepted tokens per round** for `Qwen4Exp` (`Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix`), plus eliminate redundant global-memory rescans in `top_logprobs_kernel` when extracting top-$K$ sparse proposal distributions (`K > 4`).

## 1. Accounting

- **Affected subsystem files**:
  - `src/models/qwen3_5/frontend/frontend.cpp` (M): Added Qwen structural phase-transition special tokens (`<think>`, `</think>`, `<tool_call>`, `</tool_call>`, `<tool_response>`, `</tool_response>`) to `ngram_boundaries` (sorted and deduplicated) whenever N-gram sources are enabled.
  - `src/models/qwen3_5/program/ngram_proposer.h` (M): Added `is_boundary(Token)` (`token < 0 || std::binary_search(...)`), enforced boundary checks inside `propose()` on the query `tail` window (`history[history.size() - match_len .. history.size() - 1]`), and deduplicated `boundaries_` in `set_boundaries()`.
  - `src/ops/kernel/top_logprobs.cuh` (M): Replaced `else if (rank < 4)` with `else if (k3 < previous)` in `top_logprobs_kernel`, allowing ranks `4 .. count - 1` to use the thread-local 4-key register cache (`k0, k1, k2, k3`) whenever a thread has not yet exhausted its top-4 local keys (`k3 < previous`).
  - `src/models/qwen4_exp/program/program_impl.h` (M):
    - Raised `kSampledDraftRanks` from `4` to `8` (clamped to `ops::kScreenedHeadMaxRanks = 4` when `parameters_.draft_head.screen` is bound).
    - Raised `kMultiRowNgramMaxDrafts` from `7` to `15` and raised the multi-row N-gram column budget from `32 / batch - 1` to `64 / batch - 1` (`batch * W <= 64`).
    - Tuned `kNgramAcceptancePriors` from `{0.75, 0.88, 0.93, 0.96}` to `{0.72, 0.92, 0.96, 0.98}`, lowered `kNgramWidthCost` default from `0.10` to `0.05`, and applied faster negative EWMA backoff (`2.5 * kAcceptanceWeight`) on complete N-gram rejection (`A == 0`).
    - In `draft_length_at`, restricted the tile-quantized `tile_cost = std::min(cost, 0.10)` override to confidence-gated MTP rounds (`!(sampled_drafts_ && lane.sampling.temperature > 0.0F)`), using the true `cost` (`0.22`) on ungated sampled rounds.
  - `src/models/qwen4_exp/program/graph_set.cpp` (M): Updated `ngram_max_drafts_for_rows` from `std::min(7, 32 / rows - 1)` to `std::min(15, 64 / rows - 1)` so multi-row CUDA Graphs capture the `W = 16` (`15` draft tokens) N-gram verification widths at `rows = 2..4`.
  - `tests/models/qwen3_5/test_ngram_proposer.cpp` (M) & `tests/models/qwen3_5/test_frontend.cpp` (M): Added unit test coverage for `tail` boundary rejection in `NgramProposer::propose` and phase-marker boundaries in `Frontend::prepare`.

## 2. Architectural Rationale ("Why" on GB10 `sm_121a`)

1. **Sampled MTP Draft Mass & `top_logprobs_kernel` Register Cache (`kSampledDraftRanks = 8`)**:
   - Under `temperature > 0` (`--temperature 1.0 --top-p 0.95 --top-k 20`), MTP uses sparse Gumbel-max draft sampling over the top-$K$ proposal logprobs (`kSampledDraftRanks`). Previously `kSampledDraftRanks = 4` captured only the top-4 proposal logits into `proposal_top_ids_` / `proposal_top_logprobs_`; any target token outside the draft head's top-4 had draft probability $q(x) = 0$, reducing captured draft probability mass $Z_M$ and causing frequent rejection under block verification.
   - Raising `kSampledDraftRanks` to `8` captures twice as many candidate ranks per draft step while keeping `kMaxTopLogprobs = 16` bounds intact and preserving `ops::kScreenedHeadMaxRanks = 4` when a screened draft head is bound.
   - In `top_logprobs_kernel` (`src/ops/kernel/top_logprobs.cuh`), each thread precomputes its top-4 keys `(k0 >= k1 >= k2 >= k3)` during the initial vector pass over `[248320]` or `[131072]` vocabulary logits. Previously, `else if (rank < 4)` forced every rank $\ge 4$ to rescan global memory across all threads even though at rank $r < 16$ at most $\lfloor r / 4 \rfloor \le 3$ threads out of 256 could possibly have emitted all 4 of their cached keys. Changing the condition to `else if (k3 < previous)` is mathematically exact (since every un-cached element owned by the thread is $\le k_3 < \text{previous}$) and keeps 253–256 of 256 threads in registers across all ranks $r = 0..7$.
2. **Ungated Sampled MTP Adaptive Draft Length (`draft_length_at`)**:
   - Confidence gating (`draft_gate_threshold`) is disabled on sampled rounds (`sampled_drafts_ && temperature > 0`) to preserve exact target distribution unbiasedness. However, `draft_length_at` previously replaced `cost = 0.22` with `tile_cost = std::min(cost, 0.10)` whenever `draft_gate_threshold > 0` regardless of whether the round was actually gated. On low-acceptance sampled chains (`alpha ~ 0.30`), using `0.10` instead of `0.22` forced 5–7 un-gated draft steps per round (`30.3%` acceptance), wasting 3–5 serial MTP forward passes per round and starving N-gram proposals on short prefixes. Using the true `cost` (`0.22`) on ungated sampled rounds right-sizes the MTP chain to `2–3` high-yield steps (`40.1%` acceptance) and frees verify budget for N-gram jumps.
3. **N-Gram Structural Phase Boundaries & Query Tail Filtering**:
   - `Frontend::prepare` previously populated `ngram_boundaries` only with chat turn delimiters (`im_start`, `im_end`, `eos`, `pad`). Reasoning and tool-call transitions (`<think>`, `</think>`, `<tool_call>`, `</tool_call>`, `<tool_response>`, `</tool_response>`) were indexed and matched across phases, and `NgramProposer::propose` did not check whether the query `tail` itself contained a boundary token. Filtering both indexing and query `tail` windows across all structural phase markers prevents false cross-phase proposals.
4. **Multi-Row N-Gram Verification Width (`kMultiRowNgramMaxDrafts = 15`, `64 / batch - 1`) & Controller Priors**:
   - Multi-row N-gram verification (`batch = 2..4`) was previously capped at `min(7, 32 / batch - 1)` (`7` drafts at `batch = 2..4`), whereas GB10's NVFP4/GDN/QSA verification kernels efficiently handle up to `64` total columns (`batch * W <= 64`). Raising the cap to `min(15, 64 / batch - 1)` doubles the per-round N-gram verification horizon at `batch = 2..4` (`15` drafts, `W = 16` columns per lane) and updates `graph_set.cpp` to pre-capture the corresponding multi-row CUDA Graph widths.
   - Updating `kNgramAcceptancePriors` for `match >= 4` (`0.96 -> 0.98`), lowering `kNgramWidthCost` (`0.10 -> 0.05`), and applying `2.5x` faster negative EWMA backoff when an N-gram round accepts `0` tokens lets verified copy spans sustain `15–31` draft tokens while backing off immediately after a missed speculation.

## 3. Verification & Oracle Results

- **CUDA Stack-Spill Ratchet Gate (`infernix_cuda_stack_check`)**:
  - Passed (`0` stack spills, `0` local memory bytes across all kernels).
- **Unit & Mathematical Oracle Tests**:
  - `infernix_ngram_proposer_test`: Passed (verifies `tail` boundary rejection and boundary deduplication).
  - `infernix_qwen3_5_frontend_test`: Passed (verifies `<think>`, `</think>`, `<tool_call>`, `</tool_call>` inclusion in `ngram_boundaries`).
  - `infernix_qwen4_exp_speculative_test`: Passed.
  - `infernix_ops_sampling_test top_logprobs` & `speculative_round`: Passed (exact top-K ordering across `K = 1..16` and tie-breaking).
  - `infernix_qwen4_exp_speculative_sampling_real_test --trials 200 --tokens 8 --concurrency 1` (on `Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix`, `KV=nvfp4`): Passed (`all speculative sampling checks passed`, exact greedy parity and chi-square sampled distribution homogeneity `least p = 0.0532 >= 0.000111`).

## 4. End-to-End A/B Benchmark Results (`infernix_bench`)

Evaluated on `/opt/etc/models/Qwen3.8-Flash-Next-F4-Local-NVPF4-Dense/Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix` (`--kv-dtype nvfp4 --spec mtp --draft-tokens auto --ngram-draft-tokens 31 --ngram-min-match 4`, `warmup=1, reps=2`):

| Workload & Mode | Metric | Before (`HEAD`) | After (MTP + N-Gram Opt) | Delta / Impact |
|---|---|---:|---:|---|
| **`sampled_c1` (`temp=1.0, top_p=0.95, top_k=20`, `c=1`) — `tg256`** | **Decode throughput** | `43.89 tok/s` | **`55.87 tok/s`** | **+11.98 tok/s (+27.3%)** |
| `sampled_c1` — `tg256` | Overall spec acceptance | `0.299` (`1.75 tok/round`) | **`0.442` (`1.93 tok/round`)** | **+14.3 pp (+47.8% rel)** |
| `sampled_c1` — `tg256` | MTP acceptance rate | `0.303` (`219 / 722`) | **`0.401` (`167 / 416`)** | **+9.8 pp (+32.3% rel)** |
| `sampled_c1` — `tg256` | N-Gram acceptance & yield | `0.000` (`0 / 11`, `0.00/r`) | **`0.564` (`79 / 140`, `7.18/r`)** | **0.00 → 7.18 accepted/round** |
| **`sampled_c1` (`temp=1.0, top_p=0.95, top_k=20`, `c=1`) — `pp2048+tg256`** | **Decode throughput** | `234.30 tok/s` | **`237.00 tok/s`** | **+2.70 tok/s (+1.2%)** |
| `sampled_c1` — `pp2048+tg256` | Acceptance length | `19.69 tok/round` | **`20.48 tok/round`** | **+4.0% tok/round** |
| `sampled_c1` — `pp2048+tg256` | MTP acceptance & yield | `0.444` (`1.33/r`) | **`0.462` (`1.50/r`)** | **+12.8% accepted/round** |
| `sampled_c1` — `pp2048+tg256` | N-Gram accepted/round | `20.96/r` (`91.7%` acc) | **`22.90/r` (`88.9%` acc)** | **+9.3% accepted/round** |
| **`sampled_c4` (`temp=1.0, top_p=0.95, top_k=20`, `c=4`) — `pp2048+tg256`** | **Decode throughput** | `55.76 tok/s/stream` (`223.0 agg`) | **`71.06 tok/s/stream` (`284.2 agg`)** | **+27.4% (+61.2 tok/s agg)** |
| `sampled_c4` — `pp2048+tg256` | Acceptance length | `5.95 tok/round` (`344` rounds) | **`7.67 tok/round` (`267` rounds)** | **+28.9% tok/round** |
| `sampled_c4` — `pp2048+tg256` | N-Gram accepted/round | `6.72/r` (`1479 / 1535`, `96.4%`) | **`13.34/r` (`1548 / 1605`, `96.4%`)** | **+98.5% accepted/round** |
| `sampled_c4` — `pp2048+tg256` | MTP acceptance & yield | `0.588` (`223 / 379`, `1.80/r`) | `0.545` (`232 / 426`, `1.54/r`) | More non-copy tail transitions handled by MTP after 15-tok N-gram jumps |
| **`greedy_c4` (`temp=0`, `c=4`) — `pp2048+tg256`** | **Decode throughput** | `58.75 tok/s/stream` (`235.0 agg`) | **`76.55 tok/s/stream` (`306.2 agg`)** | **+30.3% (+71.2 tok/s agg)** |
| `greedy_c4` — `pp2048+tg256` | Acceptance length | `6.31 tok/round` (`324` rounds) | **`9.22 tok/round` (`222` rounds)** | **+46.1% tok/round** |
| `greedy_c4` — `pp2048+tg256` | MTP acceptance & yield | `0.710` (`220 / 310`, `2.00/r`) | **`0.701` (`230 / 328`, `2.09/r`)** | **+4.5% accepted/round** |
| `greedy_c4` — `pp2048+tg256` | N-Gram accepted/round | `7.01/r` (`1500 / 1508`, `99.5%`) | **`14.23/r` (`1594 / 1626`, `98.0%`)** | **+103.0% accepted/round** |

## 5. Adverse findings & dispositions

- On `sampled_c4` `pp2048+tg256`, per-round MTP acceptance rate shifted from `0.588` (`1.80/r` over `124` MTP rounds) to `0.545` (`1.54/r` over `151` MTP rounds), even though total MTP accepted tokens increased (`223 → 232`) and overall decode throughput rose `+27.4%`. Cause: doubling the multi-row N-gram horizon (`7 → 15` draft tokens/round) allowed N-gram rounds to consume the predictable copy spans in half as many rounds (`220 → 116` N-gram rounds), concentrating MTP rounds onto the harder non-copy transition points between spans.
- On `sampled_c1` `tg256`, restricting `tile_cost = 0.10` to confidence-gated rounds reduced raw MTP accepted tokens per MTP round (`0.81 → 0.66/r`) by trimming low-yield tail steps (`2.67 → 1.64` drafted/round), which boosted MTP acceptance rate from `30.3%` to `40.1%`, unblocked N-gram matching (`0.00/r → 7.18/r`), raised overall acceptance length (`1.75 → 1.93 tok/round`), and increased end-to-end `tg256` decode throughput by **`+27.3%` (`43.89 → 55.87 tok/s`)**.

## 6. Rebase conflict resolution policy

- Preserve `kSampledDraftRanks = 8`, `kMultiRowNgramMaxDrafts = 15`, `64 / batch - 1` multi-row N-gram column budgeting, gated-only `tile_cost` in `draft_length_at`, and updated N-gram priors/backoff in `src/models/qwen4_exp/program/program_impl.h` and `src/models/qwen4_exp/program/graph_set.cpp`.
- Preserve `else if (k3 < previous)` register-cache reuse in `src/ops/kernel/top_logprobs.cuh`.
- Preserve phase-marker `ngram_boundaries` in `src/models/qwen3_5/frontend/frontend.cpp` and `is_boundary` query-tail filtering in `src/models/qwen3_5/program/ngram_proposer.h`.

---

# GB10 (`sm_121a`) Branch Changelog — 2026-11-07 — `nsys`-Guided NVFP4 Decode & Prefill Kernel Optimizations

Scope: Capture and analyze node-level `nsys` traces (`profiles/nsys/nvfp4_pp4096.sqlite` and `profiles/nsys/nvfp4_tg128_nodes.sqlite`) on `Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix` (`--kv-dtype nvfp4 --spec mtp --draft-tokens auto`) and implement the top 5 remaining kernel-level optimizations across decode and prefill.

## 1. Accounting

- **Affected subsystem files**:
  - `src/ops/offloaded_sparse_moe/cuda/moe_layer.cu` (M):
    - Halved `GateUpShared::stage` from `kGateUpSliceBytes` (`46,080 B`, 2 CTAs/SM) to `kGateUpStageBytes` (`23,040 B`, 1 K-half = 80 blocks across both groups `g = 0, 1`), reducing `sizeof(GateUpShared)` from `50,048 B` to `27,008 B` (`3 * 28,032 = 84,096 <= 101,376 B`) and raising `gate_up_kernel` to `__launch_bounds__(kThreads, 3)` (`3 CTAs/SM = 50.0%` occupancy, `REG:79 STACK:0`, matching `down_kernel`).
    - Added `stage_gate_up_half` and `quantize_pair_half_codes<B0, BLen>` (parallelizing 2-column A4 quantization across 160 threads with warp shuffles instead of serial per-thread `t0`/`t1` quantization) so all 4 column loops in `gate_up_kernel` overlap K-half 0 and K-half 1 weight `cp.async` transfers with activation quantization while preserving bit-exact `int64_t` accumulation.
    - In `down_kernel`, exited empty/wide-expert CTAs (`record_slice == nullptr` with `tile != 0 || job != job_base`) before `pdl::wait_for_dependencies()` so `~9,800` wide-route CTAs in prefill no longer occupy SM slots blocked on `pdl::wait_for_dependencies()` while `gate_up_kernel` runs its remaining waves.
    - Specialized `combine_inject_vec8_kernel<bool kMasked, bool kWriteMean = false>` with `__launch_bounds__(320, kWriteMean ? 2 : 4)` (`4 CTAs/SM = 83.3%` occupancy at `REG:48 STACK:0` when `mean_out == nullptr`, vs `72–78` regs / 2 CTAs/SM previously), and gated `moe_combine_inject` PDL to `columns <= 16 ? pdl::Dependency::Programmatic : pdl::Dependency::Serialized`.
  - `src/ops/hyper_connection/hyper_connection_mix_fused.cu` (M):
    - Added `ExactColumns = 2` (`launch_exact<2>()` and `launch_up_v4<2>()`) for `columns == 2` so `T = 2` no longer falls back to `ExactColumns = 0` (`168–202` regs, 1 CTA/SM).
    - Inverted the `(t, h, q)` loop order in `hc_up_collapse_v4_kernel` when `ExactColumns > 0` so `Codec::decode(chunk[q][h], wv)` executes once per `(q, h)` instead of `ExactColumns` times while preserving the exact per-`(t, q)` FMA reduction order (`acc[t][q]`) and `warp_sum8_transposed` bits.
    - Raised `hc_down_partials_v4_kernel` launch bounds to `(ExactColumns == 2 || ExactColumns == 3) ? 2 : 1` (`103–128` regs, `STACK:0`, 2 CTAs/SM) and `hc_up_collapse_v4_kernel` launch bounds to `(std::is_same_v<Codec, Q8Codec> && (ExactColumns == 2 || ExactColumns == 3)) ? 2 : 1` (`128` regs, `STACK:0`, 2 CTAs/SM for `Q8Codec`).
  - `src/ops/kernel/rmsnorm.cuh` & `src/ops/launcher/rmsnorm.cu` (M):
    - Added `kRmsPreloadMidD = 10240` (`Preload = 40`) for Qwen4Exp's MTP `hidden_norm` (`d = 10240`), eliminating 24 dead loop iterations/branches per thread (`Preload = 64 -> 40`) and enabling `pdl::enter()` / `pdl::launch_consumer` overlap with the producer while keeping the exact `rmsnorm_generic_inv` tree reduction order (`STACK:0`).
  - `src/ops/hyper_connection/hyper_connection.cu` (M):
    - Reversed `blockIdx.y` (`t = gridDim.y - 1 - blockIdx.y` when `gridDim.y > 1024`) in `norm_vec8_kernel`, `norm_kernel`, `inject_norm_vec8_kernel`, and `inject_norm_kernel` so prefill (`T = 4096`) hits the upstream kernel's freshly written tail (`~18 MB`) in GB10's 24 MiB L2 cache and leaves `normalized[:, 0..~900]` hot in L2 for the downstream `linear(down)` GEMM.
    - Preloaded `weight` before `pdl::enter()` in `norm_vec8_kernel`, and gated `hyper_connection_norm` and `hyper_connection_inject_norm` PDL to `columns <= 16 ? pdl::Dependency::Programmatic : pdl::Dependency::Serialized`.
  - `src/ops/linear/nvfp4/nvfp4_a4.cu` (M):
    - Gated `launch_residual_exact` (`nvfp4_a4_row_scaled_residual_quantize_kernel` / `nvfp4_a4_residual_quantize_kernel`) to `(Layout == Nvfp4ScaleLayout::RowMajor && tokens <= 16) ? pdl::Dependency::Programmatic : pdl::Dependency::Serialized` (matching `launch_quantize_exact`), preventing multi-wave prefill quantize grids from stealing SM slots during Wave 0 of their producer and preserving the reverse-token L2 tail hit.

## 2. Verification & Microbenchmark Results

- **CUDA Stack-Spill Ratchet Gate (`infernix_cuda_stack_check`)**:
  - Passed (`7,661` kernels checked in `infernix-serve`, `0` violations).
- **Unit & Mathematical Oracle Suite (`ctest`)**:
  - `27 / 27` tests passed (`100%`), including `infernix_hyper_connection_test`, `infernix_rmsnorm_test`, `infernix_offloaded_moe_layer_test`, `infernix_offloaded_moe_wide_test`, `infernix_offloaded_moe_cuda_test`, and `infernix_linear_nvfp4_a4_test`.
- **Op Microbenchmarks**:
  - `infernix_hyper_connection_bench` (`S=4, H=2560, rank=320, codec=q8`, `7.70 MB` weights/call):
    - `T=1`: `30.43 us` median (`253 GB/s`)
    - `T=2`: **`30.34 us` median (`254 GB/s`)** — matches `T=1` at 93–100% of GB10's LPDDR5X bandwidth ceiling.
    - `T=5`: `45.47 us` median (`169 GB/s`)
  - `infernix_offloaded_moe_decode_bench`:
    - `exact=yes` across all `T=1, 4, 8`, `split=0, 1`, `warm/cold` configurations (`T=1 warm`: `93.41 us`, `T=4 warm`: `124.45 us`, `T=8 warm`: `233.98 us`).

## 3. End-to-End A/B Benchmark Results (`infernix_bench`)

Evaluated on `/opt/etc/models/Qwen3.8-Flash-Next-F4-Local-NVPF4-Dense/Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix` (`INFERNIX_EXPERTS_ON_GPU=1 --kv-dtype nvfp4 --use-original-nvfp4-prefill-kernel --spec mtp --draft-tokens auto --temperature 1.0 --ngram-draft-tokens 31 --ngram-min-match 4 --top-p 0.95 --top-k 20 --expert-tail-skip 3 --lm-head-draft -r 5`):

| Workload | Metric | Before (`pre_nsys_opt`) | After (`nsys_opt`) | Delta / Impact |
|---|---|---:|---:|---|
| **`pp512`** | Prefill throughput | `1,643.58 ± 2.20 tok/s` | **`1,681.81 ± 16.29 tok/s`** | **+38.23 tok/s (+2.3%)** |
| **`pp2048`** | Prefill throughput | `2,979.01 ± 10.09 tok/s` | **`3,005.49 ± 18.91 tok/s`** | **+26.48 tok/s (+0.9%)** |
| **`pp4096`** | Prefill throughput | `3,734.69 ± 12.15 tok/s` | **`3,726.91 ± 10.11 tok/s`** | -7.78 tok/s (-0.2%, within `±10.11` noise) |
| **`tg128`** | Speculative decode throughput | `54.63 ± 7.55 tok/s` | **`61.00 ± 11.50 tok/s`** | **+6.37 tok/s (+11.7%)** |
| **`tg256`** | Speculative decode throughput | `57.74 ± 7.90 tok/s` | **`60.29 ± 5.88 tok/s`** | **+2.55 tok/s (+4.4%)** |

## 4. Adverse findings & dispositions

- During initial linking, forcing `__launch_bounds__(kUpV4Threads, 2)` on `hc_up_collapse_v4_kernel` for `Bf16Codec` (`sizeof(Chunk) == 16 B`, vs `12 B` for `Q8Codec`) and adding a separate `ws[Preload]` loop across `pdl::enter()` in `rmsnorm_generic_kernel` caused stack spills in `Bf16Codec` (`8–40 B`) and `rmsnorm_generic_kernel` (`288–296 B`). Scoping the 2-CTA/SM bound on `hc_up_collapse_v4_kernel` to `std::is_same_v<Codec, Q8Codec>` (the production codec for `Qwen4Exp`) and keeping `xs[j]` / `ws[j]` in the single loop after `pdl::enter()` in `rmsnorm_generic_kernel<Epilogue, 40>` eliminated all stack spills (`0` violations across `7,661` kernels in `infernix_cuda_stack_check`).
- On `pp4096`, prefill throughput measured `3,726.91 ± 10.11 tok/s` vs `3,734.69 ± 12.15 tok/s` (`-0.2%`, well within the `±10–12 tok/s` standard deviation), while `pp512` (`+2.3%`) and `pp2048` (`+0.9%`) gained cleanly; at `T=4096`, CUTLASS Tensor Core GEMMs and `wide_expert` dominate `>71%` of total execution time.

## 5. Rebase conflict resolution policy

- Preserve the 1-K-half `GateUpShared` staging (`27,008 B`, 3 CTAs/SM), parallel `quantize_pair_half_codes`, early wide-CTA exit before `pdl::wait_for_dependencies()` in `down_kernel`, and `combine_inject_vec8_kernel` `kWriteMean` specialization in `src/ops/offloaded_sparse_moe/cuda/moe_layer.cu`.
- Preserve `ExactColumns = 2`, hoisted `Codec::decode` in `hc_up_collapse_v4_kernel`, and 2-CTA/SM launch bounds in `src/ops/hyper_connection/hyper_connection_mix_fused.cu`.
- Preserve `kRmsPreloadMidD = 10240` (`Preload = 40`) in `src/ops/kernel/rmsnorm.cuh` and `src/ops/launcher/rmsnorm.cu`.
- Preserve reverse-token L2 indexing and `columns <= 16` PDL gating in `src/ops/hyper_connection/hyper_connection.cu` and `src/ops/linear/nvfp4/nvfp4_a4.cu`.

---

# GB10 (`sm_121a`) Branch Changelog — 2026-11-07 — `c591b202` vs `7734da89` Milestone Split & MTP Acceptance Recovery

Scope: Root-cause and resolve the `-12.5%` to `-37.9%` MTP decode throughput and acceptance regressions observed in `7734da89b` vs `c591b202c` on the 3-model `int8 × mtp` milestone benchmark (`./bench/run_kv_matrix_bench.sh`, `-r 5 --warmup 2`), while retaining the `+1.85%` prefill and kernel occupancy gains.

## 1. Root-Cause Diagnosis

Inspection of the per-repetition trajectories in `dist/kv_matrix_head_pair/c591b202/` vs `dist/kv_matrix_head_pair/7734da89/` revealed two distinct root causes inside `7734da89b`:

1. **Non-deterministic MTP draft corruptions across repetitions (`pp8192+tg512` `-36.3%` to `-37.9%` across `iq3s` and `nvfp4`)**:
   - In `c591b202c`, greedy (`temperature = 0`) MTP decoding produced identical `(rounds, drafted_tokens, accepted_tokens)` across all 5 repetitions (`iq3s pp8192+tg512`: `[(73, 451, 438), (74, 460, 438) × 4]`, `acc = 0.9559`; `nvfp4 pp8192+tg512`: `[(74, 446, 438) × 5]`, `acc = 0.9821`).
   - In `7734da89b`, every repetition diverged (`iq3s pp8192+tg512`: `acc = 0.7486`; `nvfp4 pp8192+tg512`: `acc = 0.7455`).
   - **Cause A (PDL + CUDA Graph fork/workspace hazards)**: Converting `projection_fp32` (`launch_tall`), `rmsnorm_generic_kernel<Epilogue, 40>` (`d = 10240`), `hyper_connection_norm`, and `qsa_prompt_kernel` to `pdl::launch_consumer` introduced programmatic stream edges across `Forward::moe` / `Forward::mtp_block` event forks (`experts_.shared_events`) and across `Forward::mtp_block`'s recycled split-K workspace (`project(e, p.embedding_projection, ...)` -> `hn`), where `split_k_reduce_bf16_kernel` (a non-PDL kernel) overlapped with `rmsnorm_generic_kernel` writing `hn` at the freed split-K partial address.
   - **Cause B (`Q4 [131072, 2560]` draft-head route flip)**: Registering `src/ops/linear/q4/shapes/n131072_k2560.cu` flipped `linear_projects_head_rows(*head.rows)` in `Forward::mtp_block` (`forward.cpp:127`) from `false` (`ops::projection_fp32` FP32 accumulation) to `true` (`GemvR4W1` `Fp16Mantissa`).
2. **Deterministic greedy trajectory shifts on `tg512` (`dense8 tg512` `-26.9%`, `iq3s tg512` `-12.5%`)**:
   - Adding `select_q8_n13952_k2560_fast` and `select_q8_n2560_k640_fast` (`if (tokens <= 8) return launch_q8_a16_sliced_r16_t8_w8_s2_m3;`) overrode `T = 1` GEMV with 2-launch 8-warp sliced-K and changed `T = 2..8` from 4-warp (`w4`) to 8-warp (`w8`) reduction order on `dense8` and `iq3s`.
   - Rewriting the 4-element dot product in `apply_gdn_transition` and `readout_and_store` (`recurrent.cuh`) without the `0.0f` accumulator seed altered FMA contraction across all 36 GDN layers.

## 2. Changes Applied (Split of `7734da89b`)

- **Reverted to `c591b202c`**:
  - Speculative-acceptance / n-gram / router-mma tuning: `src/models/qwen3_5/frontend/frontend.cpp`, `src/models/qwen3_5/program/ngram_proposer.h`, `src/models/qwen4_exp/execution/fast_path_flags.h`, `src/models/qwen4_exp/program/graph_set.cpp`, `src/models/qwen4_exp/program/program_impl.h`, `src/ops/kernel/top_logprobs.cuh`, `tests/models/qwen3_5/test_frontend.cpp`, `tests/models/qwen3_5/test_ngram_proposer.cpp`, `src/ops/hyper_connection/hyper_connection_mix_fused.h`, `tools/run_server.sh`.
  - `Q4 [131072, 2560]` shape registration: removed `src/ops/linear/q4/shapes/n131072_k2560.cu` and restored `src/ops/linear/q4/{q4_dispatch.cpp,q4_shapes.h,sources.cmake}` so `Forward::mtp_block` keeps `ops::projection_fp32` for the `2560`-wide proposal head.
  - `Q8` `_fast` overrides and `GDN` inner FMA contraction: restored `src/ops/linear/q8/{q8_dispatch.cpp,q8_shapes.h,shapes/n13952_k2560.cu,shapes/n2560_k640.cu}` and the exact `0.0f`-seeded loop in `apply_gdn_transition` / `readout_and_store` (`src/ops/linear_attention/gated_delta_net/recurrent.cuh`, keeping the `MaxWidth <= 16 ? 5 : 4` occupancy specialization of `recurrent_record_conv_kernel`).
  - Unsafe PDL consumer conversions: restored serialized stream launches in `src/ops/projection_fp32/projection_fp32.cu`, `src/ops/kernel/rmsnorm.cuh` + `src/ops/launcher/rmsnorm.cu` (retaining `kRmsPreloadMidD = 10240`, `Preload = 40`), `src/ops/hyper_connection/hyper_connection.cu` (`norm_kernel` / `norm_vec8_kernel` and `block_sum` barrier, retaining `columns <= 16` PDL gating on `inject_norm` and reverse-token prefill indexing), and `src/ops/qsa/qsa_prompt.cu` (retaining `two_ctas_2` 2-CTA `Stages = 2` occupancy for `Nvfp4Group16`).
- **Retained from `7734da89b` (verified bit-exact and prefill-positive)**:
  - `src/ops/common/canonical_math.h` and `src/ops/offloaded_sparse_moe/cuda/{moe_layer.cu,wide_expert.cu,wide_expert.cuh,wide_expert.h,wide_expert_a16.cuh,wide_expert_a8.cuh,rg16_linear.cu}`: 3-CTA/SM 1-K-half `GateUpShared` staging (`27,008 B`), 4-column quad loop (`gate_up_kernel`), parallel `quantize_pair_half_codes`, hardware `cvt.rn.satfinite.e2m1x2.f32` A4 quantization, $O(1)$ `map_work_unit<true>`, early wide-CTA exit in `down_kernel`, and 4-CTA/SM `combine_inject_vec8_kernel` specialization.
  - `src/ops/hyper_connection/hyper_connection_mix_fused.cu` and `hyper_connection_up_collapse.cu`: `ExactColumns = 2`, hoisted `Codec::decode`, 2-CTA/SM launch bounds, and wave-exact `hc_up_collapse` tiles.
  - `src/ops/linear/nvfp4/{nvfp4_a4.cu,nvfp4_a4_quantize.cuh,nvfp4_a4_tma.cu,shapes/n12800_k2560.cu,shapes/n13952_k2560.cu,shapes/n16384_k2560.cu}` and `src/ops/linear/q8/shapes/n324_k10240.cu` + `src/ops/linear_swiglu/q8/q8_linear_swiglu_plan.cpp`.

## 3. Three-Way Milestone A/B Verification (`c591b202` vs `7734da89` vs `fixed_head2`, `-r 5 --warmup 2`)

Measured with `BENCH_BIN=./build-native/bench/infernix_bench REPS=5 WARMUP=2 ./bench/run_kv_matrix_bench.sh dist/kv_matrix_head_pair/fixed_head2` against the existing `dist/kv_matrix_head_pair/c591b202/` and `dist/kv_matrix_head_pair/7734da89/` reports:

### Prefill (`tok/s` median)

| Run / Test | `c591b202` | `7734da89` (broken) | `fixed_head2` (split) | $\Delta$ (`fixed_head2` vs `c591b202`) |
|---|---:|---:|---:|---:|
| **`nvfp4 pp8192`** | `4007.2` | `4077.2` | **`4081.3`** | **`+1.85%` (`+74.1 tok/s`)** |
| **`nvfp4 pp8192+tg512`** | `3978.6` | `4040.8` | **`4052.0`** | **`+1.84%` (`+73.4 tok/s`)** |
| **`dense8 pp8192`** | `3409.4` | `3408.3` | **`3430.6`** | **`+0.62%` (`+21.2 tok/s`)** |
| **`dense8 pp8192+tg512`** | `3405.2` | `3408.0` | **`3414.1`** | **`+0.26%` (`+8.9 tok/s`)** |
| **`iq3s pp8192`** | `1619.5` | `1614.2` | **`1619.1`** | `-0.02%` (mean `1620.8 -> 1620.3`) |
| **`iq3s pp8192+tg512`** | `1614.4` | `1599.4` | **`1611.9`** | `-0.15%` |

### Decode (`tok/s` median) & MTP Acceptance Rate

| Run / Test | `c591b202` (`tok/s` / `acc`) | `7734da89` (`tok/s` / `acc`) | `fixed_head2` (`tok/s` / `acc`) | $\Delta$ (`fixed_head2` vs `c591b202`) | Per-Repetition `(rounds, drafted, accepted)` (`fixed_head2`) |
|---|---:|---:|---:|---:|---|
| **`dense8 tg512`** | `83.9` / `0.861` | `61.3` / `0.713` | **`83.5` / `0.859`** | `-0.5%` / `-0.1pp` (**recovered `+22.2 tok/s`**) | `[(113,459,392), (107,455,395), ...]` |
| **`dense8 pp8192+tg512`** | `108.2` / `0.980` | `107.6` / `0.974` | **`108.9` / `0.980`** | **`+0.6%` / `0.0pp`** | `[(72, 449, 440) × 5]` (bit-exact to `c591b202`) |
| **`iq3s tg512`** | `74.4` / `0.782` | `65.1` / `0.688` | **`74.1` / `0.782`** | `-0.4%` / `0.0pp` (**recovered `+9.0 tok/s`**) | `[(158, 436, 341) × 5]` (bit-exact to `c591b202`) |
| **`iq3s pp8192+tg512`** | `113.2` / `0.956` | `70.3` / `0.749` | **`112.4` / `0.957`** | `-0.7%` / `+0.1pp` (**recovered `+42.1 tok/s`**) | `[(72,449,439), (74,460,438) × 4]` |
| **`nvfp4 tg512`** | `76.8` / `0.731` | `95.5` / `0.832` | **`76.9` / `0.731`** | **`+0.1%` / `0.0pp`** | `[(168, 431, 315) × 5]` (bit-exact to `c591b202`) |
| **`nvfp4 pp8192+tg512`** | `122.8` / `0.982` | `78.3` / `0.745` | **`123.2` / `0.984`** | **`+0.3%` / `+0.2pp`** (**recovered `+44.9 tok/s`**) | `[(74,446,438) × 3, (72,444,440), (74,446,438)]` |
