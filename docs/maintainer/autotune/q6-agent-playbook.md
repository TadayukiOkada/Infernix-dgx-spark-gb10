# Autonomous Agent Playbook: Q6 (`q6`) Linear Kernel Analysis, Manual Optimization & Autotuner Overhaul on NVIDIA GB10 (`sm_121a`)

This playbook gives an LLM agent concrete, step-by-step instructions to execute the 7-step GB10 kernel optimization and `cuda-kernel-toolkit` (`ckt`) overhaul for the **Q6 (`q6`)** linear family, including applying the exact architectural fixes proven on Q4 (`Pair32` scale deduplication, `BlockK=128` early weight prefetch, vectorized weight decode, and `50,176`-byte 2-blocks/SM shared-memory tuning).

> [!IMPORTANT]
> **Hard Gate — Reusing Existing Schedules on a Few Shapes Is Insufficient**:
> You MUST NOT treat `src/ops/linear/q6/*.cuh` (`q6_schedule.cuh`, `q6_a16_mma.cuh`, `q6_a16_sliced_k_mma.cuh`, `q6_a16_simt.cuh`, `q6_instances.cuh`) as read-only black boxes, and you MUST NOT stop after tweaking shape files with pre-existing schedules.
> You must:
> 1. Run `$PY -m ckt static build-native/bench/infernix_linear_bench --kernel q6 --sass` to verify `STACK = 0, LOCAL = 0` and check SM occupancy limits.
> 2. Implement the kernel-level fixes in `q6_schedule.cuh`, `q6_a16_mma.cuh`, `q6_mma_launch.cuh`, `q6_a16_sliced_k_mma.cuh`, and `q6_instances.cuh` documented in Steps 3 & 4 below.
> 3. Benchmark and optimize **all 4 shape files** in `src/ops/linear/q6/shapes/`.

---

## 0. Target Hardware & Family Surface Area

### Hardware Constants (NVIDIA GB10, `sm_121a`)
- **SM Count**: 48 SMs (`maxThreadsPerMultiProcessor = 1536`, `regsPerMultiprocessor = 65536`).
- **L2 Cache**: 24 MiB (`--flush-mib 32` is mandatory; never use the 256 MiB RTX 5090 default).
- **Sustained DRAM Read Ceiling**: ~246 GB/s.
- **Shared Memory Ceiling**: 48 KiB static default, 99 KiB (`101,376` B) opt-in dynamic per block (`50,176` B max for **2 blocks/SM**).

### Q6 Source Files & Targets
- **Kernel & Schedule Headers**:
  - [`src/ops/linear/q6/q6_schedule.cuh`](file:///opt/etc/Infernix/src/ops/linear/q6/q6_schedule.cuh): Defines `Q6A16SimtSchedule`, `Q6A16GemvSchedule`, `Q6A16MmaSchedule`, and `Q6A16SlicedKMmaSchedule`.
  - [`src/ops/linear/q6/q6_a16_simt.cuh`](file:///opt/etc/Infernix/src/ops/linear/q6/q6_a16_simt.cuh) & [`src/ops/linear/q6/q6_a16_simt.cu`](file:///opt/etc/Infernix/src/ops/linear/q6/q6_a16_simt.cu).
  - [`src/ops/linear/q6/q6_a16_sliced_k_mma.cuh`](file:///opt/etc/Infernix/src/ops/linear/q6/q6_a16_sliced_k_mma.cuh) & [`src/ops/linear/q6/q6_a16_sliced_k_mma.cu`](file:///opt/etc/Infernix/src/ops/linear/q6/q6_a16_sliced_k_mma.cu).
  - [`src/ops/linear/q6/q6_a16_mma.cuh`](file:///opt/etc/Infernix/src/ops/linear/q6/q6_a16_mma.cuh), [`src/ops/linear/q6/q6_mma_launch.cuh`](file:///opt/etc/Infernix/src/ops/linear/q6/q6_mma_launch.cuh), & [`src/ops/linear/q6/q6_a16_mma.cu`](file:///opt/etc/Infernix/src/ops/linear/q6/q6_a16_mma.cu).
  - [`src/ops/linear/q6/q6_instances.cuh`](file:///opt/etc/Infernix/src/ops/linear/q6/q6_instances.cuh) & [`src/ops/linear/q6/q6_launch.h`](file:///opt/etc/Infernix/src/ops/linear/q6/q6_launch.h).
- **Shape Dispatch Directory**: `src/ops/linear/q6/shapes/` (4 files: `n1152_k1536.cpp`, `n34816_k5120.cpp`, `n248320_k2048.cpp`, `n248320_k5120.cpp` — note that `248320` is the LM head!).
- **Q6 Physical Storage (`Q6RowSplitStorage`)**:
  - Group size `kGroupK = 64`: `32` low-4-bit code bytes (`kCodeBytesPerGroup = 32`) + `16` high-2-bit bytes (`kHighBytesPerGroup = 16`) + `2` FP16 scale bytes (`kScaleBytesPerGroup = 2`) = `50` bytes per 64 weights (`0.78125` bytes/weight).

---

## Step 1 & Step 2: Audit Autotuner Coverage & Capture Cold-L2 GB10 Baselines

```bash
export PYTHONPATH=cuda-kernel-toolkit
PY=/opt/etc/Infernix/.venv-convert/bin/python3

$PY -m ckt shapes list --qtype q6

for f in src/ops/linear/q6/shapes/n*.cpp; do
  nk=$(basename "$f" .cpp | sed -E 's/n([0-9]+)_k([0-9]+)/\1 \2/')
  set -- $nk
  $PY -m ckt shapes bench --qtype q6 --n $1 --k $2 --mode boundaries --flush-mib 32
done
```

### Why `ckt autotune` Currently Fails on Q6
1. `CandidateRegistry._generate_family_template_candidates()` in [`registry.py`](file:///opt/etc/Infernix/cuda-kernel-toolkit/ckt/autotune/registry.py) has **no `qtype == "q6"` branch** (synthesizes 0 candidates).
2. `BenchmarkRunner` in [`runner.py`](file:///opt/etc/Infernix/cuda-kernel-toolkit/ckt/autotune/runner.py) has **no `infernix_q6_route_sweep` binary**, so it falls back to `infernix_linear_bench`.

---

## Step 3 & Step 4: Known Kernel Bugs & GB10 Bottlenecks in `src/ops/linear/q6/`

Inspect [`src/ops/linear/q6/q6_schedule.cuh`](file:///opt/etc/Infernix/src/ops/linear/q6/q6_schedule.cuh), [`q6_a16_mma.cuh`](file:///opt/etc/Infernix/src/ops/linear/q6/q6_a16_mma.cuh), and [`q6_a16_sliced_k_mma.cuh`](file:///opt/etc/Infernix/src/ops/linear/q6/q6_a16_sliced_k_mma.cuh):

1. **Bug 1: Duplicate `Pair32` Scale Storage & Hardcoded `48 KiB` Limit in [`q6_schedule.cuh:83-106`](file:///opt/etc/Infernix/src/ops/linear/q6/q6_schedule.cuh#L83-L106)**:
   - Deduplicate `Pair32` scale storage when `kGroupsPerK % 2 == 0` (`kScaleRowBytes = (kGroupsPerK / 2) * 4` instead of `kGroupsPerK * 4`).
   - Raise the `static_assert(kSharedBytes <= 48 * 1024)` to `99 * 1024` and enable dynamic shared memory in `q6_a16_mma.cuh` and `q6_mma_launch.cuh` (mirroring [`q4_a16_mma.cuh`](file:///opt/etc/Infernix/src/ops/linear/q4/q4_a16_mma.cuh)).
2. **Bug 2: `BlockK = 128`, Early Weight Prefetch (`S=1, BS=1`), and `50,176`-Byte 2-Blocks/SM Occupancy in [`q6_a16_mma.cuh`](file:///opt/etc/Infernix/src/ops/linear/q6/q6_a16_mma.cuh)**:
   - Separate `stage_quant` and `stage_activation` async groups when `BS == 1` (`cp_wait<1>()` before `decode_weight`, `cp_wait<0>()` before MMA, and `stage_quant` prefetch before the MMA loop).
   - With `S = 1, BS = 1, BK = 128, BM = 64`:
     - `As = 16,384` B
     - `Cr + Hr + Sr = 1 * 64 * (64 + 32 + 4) = 6,400` B (`6.25 KiB`)
     - `Bs = BN * 256` B:
       - `BN = 64`: `16,384` B $\to$ Total = `39,168` B (`38.25 KiB` $\le 48$ KiB static, **2 blocks/SM**).
       - `BN = 80`: `20,480` B $\to$ Total = `43,264` B (`42.25 KiB` $\le 48$ KiB static, **2 blocks/SM**).
       - `BN = 96`: `24,576` B $\to$ Total = `47,360` B (`46.25 KiB` $\le 48$ KiB static, **2 blocks/SM**).
       - For `BN = 112` or `128`, use `BlockRows = 32` (`MmaR32T112K128S2A1` / `MmaR32T128K128S2A1`: `As = 8,192` B, `Cr+Hr+Sr = 6,400` B, `Bs = 32,768` B $\to$ Total = `47,360` B $\le 48$ KiB static, **2 blocks/SM**!).
3. **Bug 3: Redundant `ldmatrix_x2` in [`q6_a16_sliced_k_mma.cuh`](file:///opt/etc/Infernix/src/ops/linear/q6/q6_a16_sliced_k_mma.cuh)**:
   - Hoist `ldmatrix_x2(b0, b1, ...)` outside the `mi` loop for `MT == 2` (`BlockRows = 32`) while keeping `MT == 1` (`BlockRows = 16`) at `REG <= 64` (`8 blocks/SM`).

---

## Step 4b: Diagnostic Decision Matrix (How to Interpret Results & What to Do Next)

| Observed Signal in `ckt shapes bench` / `ckt static` | Root Cause Diagnosis | Exact Action to Take Next |
|---|---|---|
| **LM head (`n248320_k2048`, `n248320_k5120`) drops from ~210 GB/s at $T=64$ to ~135 GB/s at $T=65..128$** | `MmaR64T80..T128` using `BlockK=64, BS=2` or exceeding `50,176` B (1 block/SM). | Upgrade `MmaR64T80` and `MmaR64T96` to `BlockK=128, Stages=1, ActivationStages=1` (`<= 47,360` B $\to$ **2 blocks/SM**) and use `MmaR32T128K128S2A1` (`47,360` B $\to$ **2 blocks/SM**) for $T > 96$. |
| **`[INVERTED CLIFF]` at $T=32 \to 33$ or $T=63 \to 64$ on `n34816_k5120.cpp`** | `sliced_r32_t32` held up to $T=63$ where `mma_r32_t64` or `mma_r64_t64` is already ~40% faster. | Switch $T=33..64$ to `mma_r32_t64_k128_s2_a1` or `mma_r64_t64_k128_s2_a1`. |

---

## Steps 5, 6 & 7: Build `infernix_q6_route_sweep`, Update `ckt` & Verify

1. Create `bench/ops/q6_route_sweep.cu` (mirroring [`bench/ops/q4_route_sweep.cu`](file:///opt/etc/Infernix/bench/ops/q4_route_sweep.cu)) and register `infernix_q6_route_sweep` in `bench/ops/benchmarks.cmake`.
2. Add `"q6": "infernix_q6_route_sweep"` to `runner.py` and `elif qtype == "q6":` to `registry.py`.
3. Build and qualify against the independent FP64 oracle:
   ```bash
   ninja -C build-native -j "$(nproc)" infernix_linear_bench infernix_q6_route_sweep infernix_linear_q6_a16_test
   ./build-native/tests/infernix_linear_q6_a16_test
   ```
4. Re-benchmark all 4 Q6 shapes with `--flush-mib 32`, verify `git diff` preserves all `tokens % 4 != 0` guards (in `n1152_k1536.cpp`), and run `$PY -m pytest cuda-kernel-toolkit/tests -q` and `git diff --check`.
