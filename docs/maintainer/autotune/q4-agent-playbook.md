# Autonomous Agent Playbook: Q4 (`q4`) Linear Kernel Analysis, Manual Optimization & Autotuner Overhaul on NVIDIA GB10 (`sm_121a`)

This playbook serves as the **reference implementation and case study** for the 7-step GB10 kernel optimization and `cuda-kernel-toolkit` (`ckt`) overhaul completed on the **Q4 (`q4`)** family. Other family playbooks ([`nvfp4`](file:///opt/etc/Infernix/docs/maintainer/autotune/nvfp4-agent-playbook.md), [`q8`](file:///opt/etc/Infernix/docs/maintainer/autotune/q8-agent-playbook.md), [`bf16`](file:///opt/etc/Infernix/docs/maintainer/autotune/bf16-agent-playbook.md), [`fp8`](file:///opt/etc/Infernix/docs/maintainer/autotune/fp8-agent-playbook.md), [`q5`](file:///opt/etc/Infernix/docs/maintainer/autotune/q5-agent-playbook.md), [`q6`](file:///opt/etc/Infernix/docs/maintainer/autotune/q6-agent-playbook.md)) follow this exact methodology.

> [!IMPORTANT]
> **Why Both Kernel/Schedule Engineering AND Full-Family Shape Tuning Are Mandatory**:
> As demonstrated on both Q4 and FP8, merely re-sorting existing schedules in 2–3 shape files leaves the largest GB10 speedups on the table (such as 2-blocks/SM `<= 50,176` B SMEM schedules, 9-warp TMA register-spill elimination, `RowTiles = 2` Sliced-K, `Simt` `TokenTile == Capacity` matching, and 2D L2-swizzled `Grouped` / `RowFast` TMA schedules for GB10's 24 MiB L2 cache). Every family optimization must audit and improve the kernel headers (`*.cuh`), schedule instances (`*_instances.cuh`), sweep binary, and 100% of shape files (`shapes/*`).

---

## 1. Exact 7-Step Sequence Executed on Q4

### Step 1: Review Autotuner Slowdowns (`runner.py`, `cli.py`)
- **Symptom**: `ckt autotune tune` took ~25 minutes per shape.
- **Root Cause**: `BenchmarkRunner.measure()` in [`cuda-kernel-toolkit/ckt/autotune/runner.py`](file:///opt/etc/Infernix/cuda-kernel-toolkit/ckt/autotune/runner.py) spawned a separate route-sweep process for every `(candidate, token)` pair instead of batching all token grid points (`--tokens t1,t2,...`) into one process per shape.
- **Fix**: Added `prewarm_tokens()` and `_sweep_batch_cache` in `runner.py`, lowered default sweep iterations to `--warmup 2 --iters 5`, and skipped redundant `infernix_linear_bench` rechecks in [`cli.py`](file:///opt/etc/Infernix/cuda-kernel-toolkit/ckt/autotune/cli.py), cutting per-shape autotuning to **~10–37 seconds**.

### Step 2: Run Autotuner (Stash Only) & Capture Cold-L2 Baseline
- Ran `ckt autotune tune --family q4` before any kernel edits: all 10 shapes reported `ALREADY OPTIMAL` with zero edits.
- Captured exact cold-L2 (`--flush-mib 32`) route-boundary baselines across all Q4 shapes using:
  ```bash
  PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt shapes bench \
      --qtype q4 --n <N> --k <K> --mode boundaries --flush-mib 32
  ```

### Steps 3 & 4: Manual Kernel Review & GB10 Hardware Optimization
1. **Deduplicating `Pair32` Scale Storage & Unlocking `BlockK = 128` (`q4_schedule.cuh`)**:
   - In [`src/ops/linear/q4/q4_schedule.cuh`](file:///opt/etc/Infernix/src/ops/linear/q4/q4_schedule.cuh), `kScaleRowBytes` allocated `kGroupsPerK * 4 = 8` bytes per row when `kGroupsPerK == 2` (`BlockK = 128`) even though two adjacent 64-element groups share a single 4-byte `Pair32` load.
   - Deduplicating `kScaleRowBytes = (kGroupsPerK / 2) * 4` and eliminating the duplicate `cp_async<4>` load in [`q4_a16_mma.cuh`](file:///opt/etc/Infernix/src/ops/linear/q4/q4_a16_mma.cuh) saved `512` B of shared memory and halved scale `cp.async` instructions.
2. **Vectorized 8-Weight Shared Decode (`q4_a16_mma.cuh`)**:
   - Replaced byte-by-byte (`LDS.U8` + `STS.32`, 2 weights/thread) decoding in `decode_weight` with 4-byte `LDS.U32` $\to$ `Q4Bf16x8Bits` $\to$ 16-byte `STS.128` `store_vec` (8 weights/thread), reducing shared-memory decode instructions by $4\times$.
3. **Async Group Separation & `50,176`-Byte 2-Blocks/SM Occupancy (`q4_a16_mma.cuh`, `q4_instances.cuh`)**:
   - Separated weight (`stage_quant`) and activation (`stage_activation`) `cp.async` groups when `ActivationStages == 1`:
     - `cp_wait<1>()` before `decode_weight` (overlapping `decode_weight` with `stage_activation`).
     - `cp_wait<0>()` before MMA, followed immediately by `stage_quant` prefetch *before* the 8-step MMA loop (overlapping cold DRAM weight reads with Tensor Core compute).
   - Configured `MmaR64T96K128S2A1` and `MmaR64T112K128S2A1` with `Stages = 1, ActivationStages = 1` (`45,312` B and `49,408` B $\le 50,176$ B GB10 2-blocks/SM ceiling), keeping **2 blocks/SM (50% occupancy)** resident up to $T=112$ and raising sustained bandwidth from `130–145 GB/s` to **183–215 GB/s (`74%–88%` of GB10 DRAM peak, up to `71.0 TFLOP/s`)**.
4. **Sliced-K MMA `ldmatrix_x2` Hoisting (`q4_a16_sliced_k_mma.cuh`)**:
   - Hoisted `ldmatrix_x2` out of the `mi` loop for `MT == 2` (`BlockRows = 32`) while keeping `MT == 1` (`BlockRows = 16`) at `REG = 64` (`8 blocks/SM`).

### Steps 5, 6 & 7: Root-Causing `ckt` Blind Spots, Creating `infernix_q4_route_sweep` & Retuning
- Created [`bench/ops/q4_route_sweep.cu`](file:///opt/etc/Infernix/bench/ops/q4_route_sweep.cu) (`infernix_q4_route_sweep`) and wired `q4` into [`runner.py`](file:///opt/etc/Infernix/cuda-kernel-toolkit/ckt/autotune/runner.py), [`registry.py`](file:///opt/etc/Infernix/cuda-kernel-toolkit/ckt/autotune/registry.py), and [`generator.py`](file:///opt/etc/Infernix/cuda-kernel-toolkit/ckt/autotune/generator.py) (gating by `numeric_ok` unless `requires_byte_identity` is set, and preserving `tokens % 4 != 0` guards).
- Retuned all Q4 shapes in `src/ops/linear/q4/shapes/`, achieving **1.27x to 1.87x speedups** across $T=8..192$ with 100% FP64 oracle qualification (`./build-native/tests/infernix_linear_q4_a16_test`).
