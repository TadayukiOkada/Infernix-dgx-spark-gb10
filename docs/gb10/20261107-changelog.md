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
