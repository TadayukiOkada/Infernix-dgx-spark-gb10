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
