# Autonomous Agent Playbook: Q5 (`q5`) Linear Kernel Analysis, Manual Optimization & Autotuner Overhaul on NVIDIA GB10 (`sm_121a`)

This playbook gives an LLM agent concrete, step-by-step instructions to execute the 7-step GB10 kernel optimization and `cuda-kernel-toolkit` (`ckt`) overhaul for the **Q5 (`q5`)** linear family, including applying the exact architectural fixes proven on Q4 (`Pair32` scale deduplication, `BlockK=128` early weight prefetch, 8-weight vectorized decode, and `50,176`-byte 2-blocks/SM shared-memory tuning).

> [!IMPORTANT]
> **Hard Gate — Reusing Existing Schedules on a Few Shapes Is Insufficient**:
> You MUST NOT treat `src/ops/linear/q5/*.cuh` (`q5_schedule.cuh`, `q5_a16_mma.cuh`, `q5_a16_sliced_k_mma.cuh`, `q5_a16_simt.cuh`, `q5_a16_gemv.cuh`, `q5_instances.cuh`) as read-only black boxes, and you MUST NOT stop after tweaking shape files with pre-existing schedules.
> You must:
> 1. Run `$PY -m ckt static build-native/bench/infernix_linear_bench --kernel q5 --sass` to verify `STACK = 0, LOCAL = 0` and check SM occupancy limits.
> 2. Implement the kernel-level fixes in `q5_schedule.cuh`, `q5_a16_mma.cuh`, `q5_mma_launch.cuh`, `q5_a16_sliced_k_mma.cuh`, and `q5_instances.cuh` documented in Steps 3 & 4 below.
> 3. Benchmark and optimize **all 7 shape files** in `src/ops/linear/q5/shapes/`.

---

## 0. Target Hardware & Family Surface Area

### Hardware Constants (NVIDIA GB10, `sm_121a`)
- **SM Count**: 48 SMs (`maxThreadsPerMultiProcessor = 1536`, `regsPerMultiprocessor = 65536`).
- **L2 Cache**: 24 MiB (`--flush-mib 32` is mandatory; never use the 256 MiB RTX 5090 default).
- **Sustained DRAM Read Ceiling**: ~246 GB/s.
- **Shared Memory Ceiling**: 48 KiB static default, 99 KiB (`101,376` B) opt-in dynamic per block (`50,176` B max for **2 blocks/SM**).

### Q5 Source Files & Targets
- **Kernel & Schedule Headers**:
  - [`src/ops/linear/q5/q5_schedule.cuh`](file:///opt/etc/Infernix/src/ops/linear/q5/q5_schedule.cuh): Defines `Q5A16SimtSchedule`, `Q5A16MmaSchedule`, `Q5A16SlicedKMmaSchedule`, `Q5A16GemvSchedule`, `Q5A16DirectSimtSchedule`.
  - [`src/ops/linear/q5/q5_a16_gemv.cuh`](file:///opt/etc/Infernix/src/ops/linear/q5/q5_a16_gemv.cuh) & [`src/ops/linear/q5/q5_a16_direct_simt.cuh`](file:///opt/etc/Infernix/src/ops/linear/q5/q5_a16_direct_simt.cuh): $T=1$ and small-$T$ direct/staged SIMT.
  - [`src/ops/linear/q5/q5_a16_simt.cuh`](file:///opt/etc/Infernix/src/ops/linear/q5/q5_a16_simt.cuh) & [`src/ops/linear/q5/q5_a16_simt.cu`](file:///opt/etc/Infernix/src/ops/linear/q5/q5_a16_simt.cu).
  - [`src/ops/linear/q5/q5_a16_sliced_k_mma.cuh`](file:///opt/etc/Infernix/src/ops/linear/q5/q5_a16_sliced_k_mma.cuh) & [`src/ops/linear/q5/q5_a16_sliced_k_mma.cu`](file:///opt/etc/Infernix/src/ops/linear/q5/q5_a16_sliced_k_mma.cu).
  - [`src/ops/linear/q5/q5_a16_mma.cuh`](file:///opt/etc/Infernix/src/ops/linear/q5/q5_a16_mma.cuh), [`src/ops/linear/q5/q5_mma_launch.cuh`](file:///opt/etc/Infernix/src/ops/linear/q5/q5_mma_launch.cuh), & [`src/ops/linear/q5/q5_a16_mma.cu`](file:///opt/etc/Infernix/src/ops/linear/q5/q5_a16_mma.cu).
  - [`src/ops/linear/q5/q5_instances.cuh`](file:///opt/etc/Infernix/src/ops/linear/q5/q5_instances.cuh) & [`src/ops/linear/q5/q5_launch.h`](file:///opt/etc/Infernix/src/ops/linear/q5/q5_launch.h).
- **Shape Dispatch Directory**: `src/ops/linear/q5/shapes/` (7 files: `n1024_k5120.cpp`, `n1152_k1152.cpp`, `n1152_k4304.cpp`, `n5120_k6144.cu`, `n5120_k17408.cu`, `n17408_k5120.cu`, `n34816_k5120.cu`).
- **Q5 Physical Storage (`Q5RowSplitStorage`)**:
  - Group size `kGroupK = 64`: `32` low-nibble code bytes (`kCodeBytesPerGroup = 32`) + `8` high-bit bytes (`kHighBytesPerGroup = 8`) + `2` FP16 scale bytes (`kScaleBytesPerGroup = 2`) = `42` bytes per 64 weights (`0.65625` bytes/weight).

---

## Step 1 & Step 2: Audit Autotuner Coverage & Capture Cold-L2 GB10 Baselines

```bash
export PYTHONPATH=cuda-kernel-toolkit
PY=/opt/etc/Infernix/.venv-convert/bin/python3

$PY -m ckt shapes list --qtype q5

for f in src/ops/linear/q5/shapes/n*.*; do
  nk=$(basename "$f" | sed -E 's/n([0-9]+)_k([0-9]+)\..*/\1 \2/')
  set -- $nk
  $PY -m ckt shapes bench --qtype q5 --n $1 --k $2 --mode boundaries --flush-mib 32
done
```

### Why `ckt autotune` Currently Fails on Q5
1. `CandidateRegistry._generate_family_template_candidates()` in [`registry.py`](file:///opt/etc/Infernix/cuda-kernel-toolkit/ckt/autotune/registry.py) has **no `qtype == "q5"` branch** (synthesizes 0 candidates).
2. `BenchmarkRunner` in [`runner.py`](file:///opt/etc/Infernix/cuda-kernel-toolkit/ckt/autotune/runner.py) has **no `infernix_q5_route_sweep` binary**, so it falls back to `infernix_linear_bench` (testing only the already-compiled dispatch).

---

## Step 3 & Step 4: Known Kernel Bugs & GB10 Bottlenecks in `src/ops/linear/q5/`

Inspect [`src/ops/linear/q5/q5_schedule.cuh`](file:///opt/etc/Infernix/src/ops/linear/q5/q5_schedule.cuh), [`q5_a16_mma.cuh`](file:///opt/etc/Infernix/src/ops/linear/q5/q5_a16_mma.cuh), and [`q5_a16_sliced_k_mma.cuh`](file:///opt/etc/Infernix/src/ops/linear/q5/q5_a16_sliced_k_mma.cuh) — they share the **exact same three bottlenecks** we diagnosed and fixed in Q4:

1. **Bug 1: Duplicate `Pair32` Scale Storage & Hardcoded `48 KiB` Limit in [`q5_schedule.cuh:79-103`](file:///opt/etc/Infernix/src/ops/linear/q5/q5_schedule.cuh#L79-L103)**:
   - Currently:
     ```cpp
     static constexpr int kScaleBytes =
         kScaleLoadMode == Q5ScaleLoad::Pair32 ? 4 : Q5RowSplitStorage::kScaleBytesPerGroup;
     static constexpr int kSharedBytes =
         kBlockRows * kBlockK * static_cast<int>(sizeof(__nv_bfloat16)) +
         kActivationStages * kBlockTokens * kBlockK * static_cast<int>(sizeof(__nv_bfloat16)) +
         kStages * kBlockRows * kGroupsPerK * Q5RowSplitStorage::kCodeBytesPerGroup +
         kStages * kBlockRows * kGroupsPerK * (kScaleBytes + Q5RowSplitStorage::kHighBytesPerGroup);
     static_assert(kSharedBytes <= 48 * 1024, ...);
     ```
   - When `BlockK = 128` (`kGroupsPerK = 2`) and `Pair32` is enabled, two adjacent groups share ONE 4-byte scale pair, so `kScaleRowBytes` should be `(kGroupsPerK / 2) * 4 = 4` bytes per row, NOT `2 * 4 = 8` bytes per row!
   - And the `48 * 1024` `static_assert` must be raised to `99 * 1024` with dynamic shared memory support in `q5_a16_mma.cuh` and `q5_mma_launch.cuh` (mirroring [`q4_a16_mma.cuh`](file:///opt/etc/Infernix/src/ops/linear/q4/q4_a16_mma.cuh) and [`q4_mma_launch.cuh`](file:///opt/etc/Infernix/src/ops/linear/q4/q4_mma_launch.cuh)).
2. **Bug 2: `BlockK = 128`, Early Weight Prefetch (`S=1/2, BS=1`), and `50,176`-Byte 2-Blocks/SM Occupancy in [`q5_a16_mma.cuh`](file:///opt/etc/Infernix/src/ops/linear/q5/q5_a16_mma.cuh)**:
   - Upgrade `q5_a16_mma.cuh` to:
     - Use vectorized 8-weight decoding (`uint32_t` low nibbles + `uint8_t` high bits $\to$ `Q5Bf16x8Bits` $\to$ 16-byte `STS.128` `store_vec`).
     - Separate `stage_quant` and `stage_activation` async groups when `BS == 1` (`cp_wait<1>()` before `decode_weight`, `cp_wait<0>()` before MMA, and `stage_quant` prefetch before the 8-step MMA loop).
     - Use `BlockK = 128, Stages = 1, ActivationStages = 1, MinBlocksPerSm = 2` for `MmaR64T80`, `MmaR64T96`, `MmaR64T112`:
       - With `S = 1, BS = 1, BK = 128, BM = 64`:
         - `As = 16,384` B
         - `Cr + Hr + Sr = 1 * 64 * (64 + 16 + 4) = 5,376` B
         - `Bs = BN * 256` B:
           - `BN = 80`: `20,480` B $\to$ Total = `42,240` B (`<= 48 KiB` static, **2 blocks/SM**!).
           - `BN = 96`: `24,576` B $\to$ Total = `46,336` B (`<= 48 KiB` static, **2 blocks/SM**!).
           - `BN = 112`: `28,672` B $\to$ Total = `50,432` B (or with `MmaR32T128` `<32, 128, 128, 16, 32, 2, 2, ..., 1>`: `8192 + 32768 + 5376 = 46,336` B $\le 48$ KiB static, **2 blocks/SM**!).
3. **Bug 3: Redundant `ldmatrix_x2` in [`q5_a16_sliced_k_mma.cuh`](file:///opt/etc/Infernix/src/ops/linear/q5/q5_a16_sliced_k_mma.cuh)**:
   - For `MT == 2` (`BlockRows = 32`), hoist `ldmatrix_x2(b0, b1, ...)` outside the `mi` loop (`0..MT-1`) while keeping `MT == 1` (`BlockRows = 16`) specialized so `REG <= 64` (`8 blocks/SM`).

---

## Step 4b: Diagnostic Decision Matrix (How to Interpret Results & What to Do Next)

| Observed Signal in `ckt shapes bench` / `ckt static` | Root Cause Diagnosis | Exact Action to Take Next |
|---|---|---|
| **`[UPWARD CLIFF]` at $T=64 \to 65$ or $T=96 \to 97$ (`read_%` drops from $>75\%$ to $<55\%$)** | Schedule switched from `BlockK=128, BS=1` (2 blocks/SM) to `BlockK=64, BS=2` (1 block/SM, 4 warps). | Upgrade `q5_instances.cuh` schedules to `BlockK=128, Stages=1, ActivationStages=1, MinBlocksPerSm=2` (8 warps = 256 threads). |
| **`[INVERTED CLIFF]` at $T=32 \to 33$ or $T=63 \to 64$** | `sliced_r32_t32` held too high in $T$ where `mma_r32_t64_k128_s2_a1` or `mma_r64_t64_k128_s2_a1` is already 25–45% faster. | Switch $T=33..64$ to `mma_r32_t64` (for $N \le 34816$) or `mma_r64_t64` (for $N > 34816$). |
| **Vision shape (`n1152_k1152.cpp`, `n1152_k4304.cpp`) fails on $T \bmod 4 \ne 0$** | Vision shapes require $T$ to be a multiple of 4 (`if (tokens % 4 != 0) throw ...`). | Always pass multiples of 4 (`--tokens 4,8,16,32,64,96,128,256`) when sweeping vision shapes and preserve the `tokens % 4 != 0` guard in `n1152_k*.cpp`! |

---

## Steps 5, 6 & 7: Build `infernix_q5_route_sweep`, Update `ckt` & Verify

1. Create `bench/ops/q5_route_sweep.cu` (mirroring [`bench/ops/q4_route_sweep.cu`](file:///opt/etc/Infernix/bench/ops/q4_route_sweep.cu)) and register `infernix_q5_route_sweep` in `bench/ops/benchmarks.cmake`.
2. Add `"q5": "infernix_q5_route_sweep"` to `runner.py` and `elif qtype == "q5":` to `registry.py`.
3. Build and qualify against the independent FP64 oracle:
   ```bash
   ninja -C build-native -j "$(nproc)" infernix_linear_bench infernix_q5_route_sweep infernix_linear_q5_a16_test
   ./build-native/tests/infernix_linear_q5_a16_test
   ```
4. Re-benchmark all 7 Q5 shapes with `--flush-mib 32`, verify `git diff` preserves all `tokens % 4 != 0` guards, and run `$PY -m pytest cuda-kernel-toolkit/tests -q` and `git diff --check`.
