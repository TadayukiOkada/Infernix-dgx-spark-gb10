# Autonomous Agent Playbook: Q8 (`q8`) Linear Kernel Analysis, Manual Optimization & Autotuner Overhaul on NVIDIA GB10 (`sm_121a`)

This playbook gives an LLM agent concrete, step-by-step instructions to execute the 7-step GB10 kernel optimization and `cuda-kernel-toolkit` (`ckt`) verification for the **Q8 (`q8`)** linear family, interpret every benchmark/compiler/profiler signal, and choose the exact next action for each outcome.

> [!IMPORTANT]
> **Hard Gate — Reusing Existing Schedules on a Few Shapes Is Insufficient**:
> You MUST NOT treat kernel headers (`q8_a16_gemv.cuh`, `q8_a16_simt.cuh`, `q8_a16_stream.cuh`, `q8_a16_sliced_k_mma.cuh`, `q8_a16_mma.cuh`, `q8_schedule.cuh`, `q8_instances.cuh`) as read-only black boxes, and you MUST NOT stop after tweaking a subset of shape files with pre-existing schedules.
> A complete family optimization on NVIDIA GB10 (`sm_121a`) requires:
> 1. **Static SASS & Register-Spill Audit (`$PY -m ckt static build-native/bench/infernix_linear_bench --kernel q8 --sass`)**: Verify `STACK = 0, LOCAL = 0` across all Q8 kernel instances and check SM occupancy limits (`<= 50,176` B SMEM and `<= 128` regs at 256 threads for 2 blocks/SM on GB10).
> 2. **GB10-Specific Schedule & Kernel Engineering**: Audit and engineer kernel/schedule improvements in `src/ops/linear/q8/` (such as `Stages = 1, ActivationStages = 1` async group separation in `q8_a16_mma.cuh` to fit `BlockRows = 64, BlockTokens = 64..96` inside `<= 50,176` B for 2 blocks/SM, hoisted `ldmatrix_x2` for `MT == 2` in `q8_a16_sliced_k_mma.cuh`, and L2-aware tile ordering at large $T$).
> 3. **Correct Interpretation of Numerical & Column-Invariance Contracts**: Read `tests/ops/linear/test_q8_a16.cpp` first. Distinguish true column-invariance contracts from false "must match old incumbent's `KWarps` reduction tree at $T \ge 9$" assumptions.
> 4. **100% Shape Coverage**: Benchmark and optimize **all 31 shape files** in `src/ops/linear/q8/shapes/` across decode ($T=1..8$), mid-band ($T=9..256$), and large prefill ($T=257..8192$).

---

## 0. Target Hardware & Family Surface Area

### Hardware Constants (NVIDIA GB10, `sm_121a`)
- **SM Count**: 48 SMs (`maxThreadsPerMultiProcessor = 1536`, `regsPerMultiprocessor = 65536`).
- **L2 Cache**: 24 MiB (`--flush-mib 32` is mandatory; never use the 256 MiB RTX 5090 default).
- **Sustained DRAM Read Ceiling**: ~246 GB/s.
- **Shared Memory Ceiling**: 48 KiB static default, 99 KiB (`101,376` B) opt-in dynamic per block (`102,400` B per SM with `1,024` B driver overhead per CTA).
  - **2-Blocks/SM Limit**: `kSharedBytes <= 50,176` bytes (`49.0 KiB`).
  - **3-Blocks/SM Limit**: `kSharedBytes <= 33,152` bytes (`32.375 KiB`).

### Q8 Source Files & Targets
- **Kernel & Schedule Headers**:
  - [`src/ops/linear/q8/q8_schedule.cuh`](file:///opt/etc/Infernix/src/ops/linear/q8/q8_schedule.cuh): Defines `Q8A16GemvSchedule`, `Q8A16SimtSchedule`, `Q8A16StreamSchedule`, `Q8A16SlicedKMmaSchedule`, `Q8A16GroupedSlicedKMmaSchedule`, and `Q8A16MmaSchedule`.
  - [`src/ops/linear/q8/q8_a16_gemv.cuh`](file:///opt/etc/Infernix/src/ops/linear/q8/q8_a16_gemv.cuh): $T=1$ GEMV (`WarpsPerRow = 1, 2, 4, 8`).
  - [`src/ops/linear/q8/q8_a16_simt.cuh`](file:///opt/etc/Infernix/src/ops/linear/q8/q8_a16_simt.cuh): Small-$T$ staged SIMT (`BlockRows = 4, 8`, `BlockTokens = 1..8`).
  - [`src/ops/linear/q8/q8_a16_stream.cuh`](file:///opt/etc/Infernix/src/ops/linear/q8/q8_a16_stream.cuh): Register-streamed multi-pass decode ($T \le 16$, weights read once into registers and applied across passes of `ColsPerPass` tokens; `Q8StreamX::Shared` vs `Q8StreamX::Global`).
  - [`src/ops/linear/q8/q8_a16_sliced_k_mma.cuh`](file:///opt/etc/Infernix/src/ops/linear/q8/q8_a16_sliced_k_mma.cuh) & [`src/ops/linear/q8/q8_a16_grouped_sliced_k_mma.cuh`](file:///opt/etc/Infernix/src/ops/linear/q8/q8_a16_grouped_sliced_k_mma.cuh): Inner-K split Tensor Core MMA (`BlockRows = 16, 32`, `KWarps = 2, 4, 8`).
  - [`src/ops/linear/q8/q8_a16_mma.cuh`](file:///opt/etc/Infernix/src/ops/linear/q8/q8_a16_mma.cuh): Full-K Tensor Core MMA (`BlockK = 64` or `128`, `ActivationStages = 1` or `2`, `kStages = 2` code prefetch while MMA runs).
  - [`src/ops/linear/q8/q8_instances.cuh`](file:///opt/etc/Infernix/src/ops/linear/q8/q8_instances.cuh) & [`src/ops/linear/q8/q8_launch.h`](file:///opt/etc/Infernix/src/ops/linear/q8/q8_launch.h).
- **Sweep & Test Binaries**:
  - Sweep binary: `build-native/bench/infernix_q8_route_sweep` ([`bench/ops/q8_route_sweep.cu`](file:///opt/etc/Infernix/bench/ops/q8_route_sweep.cu), 74 routes).
  - Oracle test: `build-native/tests/infernix_linear_q8_a16_test`.
- **Shape Dispatch Directory**: `src/ops/linear/q8/shapes/` (31 `.cpp`/`.cu` files).
- **Q8 Physical Storage (`Q8RowSplitStorage`)**:
  - Group size `kGroupK = 32`: 32 `int8` codes (`32` B) + 1 FP16 scale (`2` B) = `34` B per 32 weights (`1.0625` bytes/weight).

---

## Step 1: Audit Autotuner Speed & Q8 Candidate Coverage

```bash
export PYTHONPATH=cuda-kernel-toolkit
PY=/opt/etc/Infernix/.venv-convert/bin/python3

# Verify Q8 shape inventory and L2 residency
$PY -m ckt shapes list --qtype q8
```

Check [`cuda-kernel-toolkit/ckt/autotune/runner.py`](file:///opt/etc/Infernix/cuda-kernel-toolkit/ckt/autotune/runner.py) and [`registry.py`](file:///opt/etc/Infernix/cuda-kernel-toolkit/ckt/autotune/registry.py):
1. Confirm `prewarm_tokens()` batches all token grid points into a single `infernix_q8_route_sweep --n <N> --k <K> --tokens <t1,t2,...> --flush-mib 32` call (~10–25s per shape instead of spawning per candidate).
2. Check if any shape file in `src/ops/linear/q8/shapes/` uses inline template instantiations (e.g., `q8_a16_stream_launch<...>` or `q8_a16_mma_launch<...>`) that are not yet registered in `bench/ops/q8_route_sweep.cu`.

---

## Step 2: Run Baseline Boundary Benchmarks & Stash Any Raw Autotuner Output

1. Run boundary cliff detection across all 31 Q8 shapes:
   ```bash
   for f in src/ops/linear/q8/shapes/n*.*; do
     nk=$(basename "$f" | sed -E 's/n([0-9]+)_k([0-9]+)\..*/\1 \2/')
     set -- $nk
     $PY -m ckt shapes bench --qtype q8 --n $1 --k $2 --mode boundaries --flush-mib 32
   done
   ```
2. Flag every shape that exhibits:
   - `[INVERTED CLIFF / DOMINATED BAND]` (latency drops when $T$ increases across a route switch).
   - `[UPWARD CLIFF]` ($>15\%$ latency jump across $T \to T+1$).
   - Low DRAM bandwidth utilization (`read_% < 70%` at $T \le 64$).

---

## Step 3: Manual Audit of Q8 Kernel Families (`GEMV`, `SIMT`, `Stream`, `Sliced-K`, `MMA`)

Inspect kernel register, shared-memory, and instruction usage:

```bash
$PY -m ckt static build-native/bench/infernix_linear_bench --kernel q8 --sass
```

Check these specific Q8 mechanisms in `src/ops/linear/q8/`:

1. **`q8_a16_mma.cuh` & `Q8A16MmaSchedule` (`q8_schedule.cuh:100-170`)**:
   - Look at `kSharedBytes` in `Q8A16MmaSchedule`:
     - `As` (`BlockRows * BlockK * 2` B) + `Bs` (`ActivationStages * BlockTokens * BlockK * 2` B) + `Cr` (`2 * BlockRows * BlockK` B) + `Sr` (`2 * BlockRows * (BlockK / 32) * 4` B).
     - Notice that for `BlockRows = 64, BlockK = 128, ActivationStages = 1`:
       - `As = 16,384` B, `Cr = 16,384` B (`2 * 64 * 128`), `Sr = 2,048` B (`2 * 64 * 16`).
       - Weight shared memory alone is **`34,816` bytes (`34.0 KiB`)**!
       - Adding `Bs` (`BlockTokens * 256` B):
         - At `BlockTokens = 48`: `Bs = 12,288` B $\to$ Total = `47,104` B (`<= 50,176` B $\to$ **2 blocks/SM**!).
         - At `BlockTokens = 64`: `Bs = 16,384` B $\to$ Total = `51,200` B (`> 50,176` B $\to$ drops to **1 block/SM**!).
     - **Key Insight**: Because `decode_weight` in `q8_a16_mma.cuh` consumes `Cr` into `As` *before* the next weight tile is prefetched during the MMA loop, if `Cr` uses `kStages = 1` (`8,192` B instead of `16,384` B) with separate async groups (`cp_wait<1>()` before `decode_weight`, `cp_wait<0>()` before MMA), it saves **8,192 bytes (`8.0 KiB`)** of shared memory!
     - Saving `8,192` bytes reduces `BlockRows = 64, BlockTokens = 64, BlockK = 128, ActivationStages = 1` from `51,200` B to **`43,008` B (`<= 48 KiB` static shared memory, 2 blocks/SM!)**, and allows `BlockTokens = 80` (`47,104` B) and `BlockTokens = 96` (`51,200` B, or `49,408` B with packed scale pairs) to run at **2 blocks/SM**!
2. **`q8_a16_sliced_k_mma.cuh`**:
   - Check if `MT == 2` (`BlockRows = 32`) hoists `ldmatrix_x2` for activation fragments `b0, b1` outside the `mi` loop (`0..MT-1`). If `ldmatrix_x2` is inside the `mi` loop, hoisting it cuts activation `ldmatrix_x2` instructions in half!
3. **`q8_a16_stream.cuh` (`Q8A16StreamSchedule`) & Stack Spill Elimination**:
   - Check $T=2..16$ on medium/small $K$ shapes (`K = 640, 2048, 2560, 4096`): `Q8A16StreamSchedule` keeps weights in registers across multiple token passes (`ColsPerPass`), avoiding repeated DRAM weight reads when $T > 1$.
   - **Stack Spill Root Cause**: Setting `MinBlocksPerSm = 2` on 8-warp (256 threads) kernels imposes `__launch_bounds__(256, 2)`, restricting registers to $\le 128$ regs/thread (`65536 / (256 * 2) = 128`), which forces 8–24 bytes of stack spills (`STACK: 8..24, LOCAL: 8..24`). Setting `MinBlocksPerSm = 1` lets `nvcc` use ~200–224 registers and achieves **`STACK: 0, LOCAL: 0` across all stream instances**.

---

## Step 4: Diagnostic Decision Matrix (How to Interpret Results & What to Do Next)

| Observed Signal in `ckt shapes bench` / `ckt static` | Root Cause Diagnosis | Exact Action to Take Next |
|---|---|---|
| **`mma_r64x16_t48_k128_a1` beats `mma_r64x32_t64_k128_a1` even at $T=64$ or $T=96$** | `mma_r64x16_t48_k128_a1` uses `47,104` B ($\le 50,176$ B $\to$ **2 blocks/SM**), whereas `mma_r64x32_t64_k128_a1` uses `51,200` B ($> 50,176$ B $\to$ **1 block/SM**)! | Shrink `Cr`/`Sr` staging in `Q8A16MmaSchedule` (`q8_schedule.cuh` / `q8_a16_mma.cuh`) so `BlockTokens = 64` and `80` fit in $\le 50,176$ B (2 blocks/SM), or use `mma_r64x16_t48_k128_a1` / `mma_r64_t96` where wave quantization matches 48 SMs. |
| **`[INVERTED CLIFF]` at $T=24 \to 25$ or $T=54 \to 55$** | `sliced_r16_t16` / `sliced_r16_t32` is held too high in $T$, or `mma` is entered too late. | Run `./build-native/bench/infernix_q8_route_sweep --n <N> --k <K> --tokens 16,24,25,32,48,54,55,64 --flush-mib 32` and move the crossover to the exact token where the destination route becomes faster. |
| **`infernix_linear_q8_a16_test` fails with "outputs differ from the columns computed alone"** | Speculative verification column-invariance contract violated (`test_q8_a16.cpp:111-125`). | Flash-Next shapes require column invariance up to their specific extent: `(324, 10240)` and `(320, 10240)` up to 64 cols; `(1280, 2560)` up to 48 cols; `(2560, 640)` up to 16 cols; `(10240, 320)`, `(16384, 2560)`, `(2560, 6144)`, `(12800, 2560)`, `(13952, 2560)`, `(2560, 2560)`, `(2560, 4096)` up to 8 cols. Maintain `Stream` up to these exact column bounds before switching to MMA. |
| **`[UPWARD CLIFF]` at $T=128 \to 129$ or $T=192 \to 193$** | Switching from a 2-blocks/SM schedule to a 1-block/SM schedule (`mma_r32_t128` or `mma_r64_t128`) or bad tile-to-token divisibility (`ceil(129 / 128) = 2` tiles with 50% empty columns). | At $T=129..144$, a `t48` (3 tiles = 144) or `t72` (2 tiles = 144) schedule avoids wasting 49% of the second `t128` tile. Check `mma_r64x16_t48_k128_a1`, `mma_r64_t96`, and `mma_r128_t80`. |
| **Autotuner strips `if (tokens > ...)` or GDN conv snapshot logic** | `generator.py` replaced the function body of a shape file that had custom prologue logic (e.g., `n16384_k2560.cu`). | Always inspect `git diff` after `--generate`. Restore any deleted precondition guards or specialized prologue calls before compiling. |

---

## Steps 5, 6 & 7: Implement Improvements, Verify Against Oracle & Update `ckt`

1. Build `infernix_linear_bench`, `infernix_q8_route_sweep`, and `infernix_linear_q8_a16_test` in a single Ninja invocation:
   ```bash
   ninja -C build-native -j "$(nproc)" infernix_linear_bench infernix_q8_route_sweep infernix_linear_q8_a16_test
   ```
2. Qualify every modified kernel or shape against the independent FP64 oracle:
   ```bash
   ./build-native/tests/infernix_linear_q8_a16_test --shape <N> <K>
   ```
3. Re-run boundary benchmarks (`--flush-mib 32`) to confirm all upward/inverted cliffs are resolved and record before/after latencies.
4. If a newly added `q8_instances` schedule wins, add it to [`bench/ops/q8_route_sweep.cu`](file:///opt/etc/Infernix/bench/ops/q8_route_sweep.cu) and [`cuda-kernel-toolkit/ckt/autotune/registry.py`](file:///opt/etc/Infernix/cuda-kernel-toolkit/ckt/autotune/registry.py), then run `$PY -m pytest cuda-kernel-toolkit/tests -q`.
