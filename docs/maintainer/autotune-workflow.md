# Offline Linear Autotuning and A/B Benchmarking Runbook

This document defines the exact sequence of commands, options, and operational procedures used to discover, baseline, autotune, compile, gate, and verify CUDA linear dispatch shapes across quantization families (`q8`, `bf16`, `fp8`, `nvfp4`, etc.) on NVIDIA GPUs (specifically NVIDIA GB10, `sm_121a`).

---

## 1. Environment & Prerequisites

Execute all commands from the repository root `/opt/etc/Infernix` using the maintainer Python virtual environment and Ninja:

```bash
# 1. Export in-tree cuda-kernel-toolkit to PYTHONPATH
export PYTHONPATH=cuda-kernel-toolkit

# 2. Select maintainer virtual environment Python interpreter
PY=/opt/etc/Infernix/.venv-convert/bin/python3

# 3. High-throughput parallel build jobs
NPROC=$(nproc)

# 4. Check hardware detection and pipe ceilings
$PY -m ckt env
```

`$PY -m ckt env` queries `cudaDeviceProp`, validates the compute capability (`sm_121a` on GB10), and caches precision ceilings and bandwidth numbers in `.ckt_cache/env.json`.

---

## 2. L2 Cache Sizing: 32 MiB vs 256 MiB Default

### Why 256 MiB was the historical default
In earlier versions, the default cold-cache eviction buffer was configured for NVIDIA GeForce RTX 5090 (Blackwell desktop), which features a large 96 MiB–128 MiB L2 cache. To guarantee complete cache line eviction across the widest bus configurations without cache residency leakage, a conservative 256 MiB payload was allocated.

### Why 32 MiB is right-sized for GB10 (DGX Spark)
On the NVIDIA GB10 (`sm_121a`, 48 SMs):
- **Hardware L2 Cache Size**: Exactly **24 MiB** (25,165,824 bytes).
- Evicting **32 MiB** (`1.33x` L2 cache size) guarantees 100% eviction of dirty lines before timing begins.
- Evicting 256 MiB on every cold timing sample wastes ~87% of device memory bandwidth on unneeded cache flushes.
- Measuring with `--flush-mib 32` accelerates sweep and autotune runs by **5x to 8x** without compromising cold-cache measurement validity.

---

## 3. Autotune Engine Architecture & Gating Rules

The autotuner (`ckt autotune`) executes offline candidate evaluations and generates production C++ dispatch ladders based on the following architectural mechanisms:

1. **Dynamic Route-Matrix Sweep Execution**:
   - `runner.py` automatically links to dedicated route sweep binaries:
     - `infernix_q8_route_sweep`: Evaluates all 74 Q8 kernel routes dynamically for any arbitrary $[N, K]$ across tokens, passing `--flush-mib 32`.
     - `infernix_bf16_flash_next_sweep`: Evaluates BF16 candidate schedules for Flash-Next shapes.
     - `infernix_fp8_flash_next_sweep`: Evaluates FP8 candidate schedules for Flash-Next shapes.
     - Fallback to `infernix_linear_bench` for hardcoded shapes.
2. **Candidate-Aware Measurement Caching**:
   - `_meas_cache` in `runner.py` keys on `(candidate.id, candidate.launch_expr, token_count, flush_mib)`. Different candidates evaluated at the same token count are measured independently.
3. **Strict Baseline-Favoring Gate**:
   - `NoiseAwareSelector` strictly retains the incumbent baseline route unless an alternative candidate outperforms it by at least `min_improvement_pct` (default 2.0% or 3.0%).
4. **No-Improvement Protection**:
   - If no candidate beats the baseline by `min_improvement_pct`, `cli.py` prints `[OK] Shape is ALREADY OPTIMAL` and leaves the existing shape file completely untouched (`git status` remains clean). Zero-functional-change or comment-only diffs are prohibited.
5. **Tolerance-Based Span Merging**:
   - `PolicyCompressor.compress` merges adjacent token intervals when candidates remain within `tolerance_pct`, eliminating micro-interval fragmentation.

---

## 4. Exact Step-by-Step Command Sequence per Shape

To autotune and verify an individual shape $[N, K]$ for quantization family `<qtype>`:

### Step 1: Run Autotune Sweep
```bash
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt autotune tune \
    --family <qtype> -n <N> -k <K> \
    --flush-mib 32 \
    --generate src/ops/linear/<qtype>/shapes/n<N>_k<K>.cpp \
    --validate
```
*(Note: Use `.cpp` or `.cu` depending on the file extension used by the shape in `src/ops/linear/<qtype>/shapes/`)*

### Step 2: Check Git Status & Gating
Inspect whether the autotuner found actionable performance improvements:
```bash
git status
```

- **Case A: Clean Working Tree**:
  ```text
  [OK] Shape Q8 [1024, 2048] is ALREADY OPTIMAL.
  No candidate beat the incumbent dispatch by >= 2.0%.
  Preserving existing shape file without modification.
  ```
  The shape is already optimal on hardware. **Do not force a change or commit.** Proceed to the next shape.

- **Case B: Modified File**:
  ```text
  [PERF] Shape Q8 [1024, 2048] improved over baseline at 17 token widths:
    T= 140: launch_q8_a16_mma_r32_t128 (31.46 us) -> launch_q8_a16_mma_r64x16_t48_k128_a1 (27.23 us, +13.4%)
    T= 256: launch_q8_a16_mma_r32_t128 (58.62 us) -> launch_q8_a16_mma_r64x16_t48_k128_a1 (30.56 us, +47.8%)
    T= 512: launch_q8_a16_mma_r32_t128 (63.17 us) -> launch_q8_a16_mma_r64_t96 (44.86 us, +29.0%)
  ```
  Inspect the diff to verify functional dispatch changes and ensure bounds checks are intact:
  ```bash
  git diff src/ops/linear/<qtype>/shapes/n<N>_k<K>.cpp
  ```

### Critical Step 2b: Manual Review & Reversion of Stripped Invariants

> [!WARNING]
> **Autotuner Overwrite Risk: Accidental Removal of Critical Preconditions & Bounds Checks**
> When the autotuner updates or rewrites the dispatch ladder in `n<N>_k<K>.{cpp,cu}`, its AST/regex code generator substitutes the inner dispatch body. In doing so, it can inadvertently strip important function-level preconditions, token extent limits, or invariants that were present in the handwritten source, such as:
> ```cpp
> if (tokens > 32768) {
>     throw std::invalid_argument("tokens exceeds maximum supported static dispatch extent");
> }
> ```
> or zero/negative token guards (`if (tokens <= 0) return;`), custom epilogue hooks, or shape-specific assertions.
>
> **Mandatory Review Rule**:
> 1. Run `git diff src/ops/linear/<qtype>/shapes/n<N>_k<K>.{cpp,cu}`.
> 2. Carefully inspect all deleted lines (prefixed with `-`).
> 3. If any precondition, invariant check, or exception throw was removed by the generator, **manually restore those lines** into the modified file before proceeding to compilation and testing.
> 4. Ensure the diff contains **only** genuine route and schedule dispatch replacements.

### Step 3: Unified Binary Compilation
Build the test and benchmark targets in a single Ninja invocation to amortize the relocatable device link (`nvlink`/`cmake_device_link.o` ~55s):
```bash
ninja -C build-native -j "$NPROC" infernix_linear_bench infernix_linear_<qtype>_a16_test
```

### Step 4: Independent Mathematical Oracle Qualification
Validate the modified shape against the naive FP32/FP64 mathematical oracle and column invariance across all token widths:
```bash
./build-native/tests/infernix_linear_<qtype>_a16_test --shape <N> <K>
```
The test must exit code 0 with:
```text
OK <QTYPE>_A16 Linear
```

### Step 5: Hardware Latency & A/B Benchmark Verification
Verify the actual latency of the compiled dispatch using `infernix_linear_bench` at the improved token widths:
```bash
# Measure cold-L2 latency at target token width T
./build-native/bench/infernix_linear_bench \
    --qtype <qtype> --n <N> --k <K> --t <T> \
    --flush-mib 32 --warmup 3 --repeat 10
```

#### Optional: Interleaved Round-Robin A/B Benchmark (`ckt ab`)
To compare candidate $A$ vs candidate $B$ free of thermal drift:
```bash
$PY -m ckt ab \
    --cmd-a "./build-native/bench/infernix_linear_bench --qtype <qtype> --n <N> --k <K> --t <T> --flush-mib 32" \
    --cmd-b "./build-native/bench/infernix_linear_bench --qtype <qtype> --n <N> --k <K> --t <T> --flush-mib 32" \
    --rounds 6
```

### Step 6: Atomic Conventional Commit
Only commit when:
1. `git status` shows genuine functional dispatch modifications.
2. Unit tests pass 100%.
3. Measured hardware latency confirms the speedup.

```bash
git add src/ops/linear/<qtype>/shapes/n<N>_k<K>.cpp
git commit -m "perf(<qtype>): retune n<N>_k<K> dispatch on GB10 (<speedup> on <token-range>)"
```

---

## 5. Full-Family Batch Tuning Script

To batch autotune an entire quantization family across all registered shapes:

```bash
$PY -c "
import glob, os, re, subprocess

qtype = '<qtype>'  # e.g. 'q8', 'bf16', 'fp8'
py = '/opt/etc/Infernix/.venv-convert/bin/python3'
env = os.environ.copy()
env['PYTHONPATH'] = 'cuda-kernel-toolkit'

shapes = []
for p in sorted(glob.glob(f'src/ops/linear/{qtype}/shapes/*.*')):
    m = re.search(r'n(\d+)_k(\d+)\.(cpp|cu)$', p)
    if m:
        shapes.append((int(m.group(1)), int(m.group(2)), m.group(0), p))

print(f'Discovered {len(shapes)} shapes for {qtype}.')

for n, k, fname, fpath in shapes:
    print(f'\n========================================')
    print(f'== Tuning {qtype} shape [{n}, {k}]: {fname}')
    print(f'========================================')

    # 1. Run autotuner
    cmd_tune = [
        py, '-m', 'ckt', 'autotune', 'tune',
        '--family', qtype, '-n', str(n), '-k', str(k),
        '--flush-mib', '32',
        '--generate', fpath,
        '--validate'
    ]
    subprocess.run(cmd_tune, env=env)

    # 2. Check if file was modified
    res_status = subprocess.run(['git', 'status', '--porcelain', fpath], capture_output=True, text=True)
    if not res_status.stdout.strip():
        print(f'== Shape [{n}, {k}]: ALREADY OPTIMAL. Skipping.')
        continue

    print(f'== Shape [{n}, {k}]: IMPROVEMENT DETECTED. Building targets...')

    # 3. Build test and bench binaries
    test_target = f'infernix_linear_{qtype}_a16_test'
    cmd_build = ['ninja', '-C', 'build-native', '-j', str(os.cpu_count()), 'infernix_linear_bench', test_target]
    res_build = subprocess.run(cmd_build)
    if res_build.returncode != 0:
        print(f'ERROR: Build failed for [{n}, {k}]. Reverting.')
        subprocess.run(['git', 'checkout', '--', fpath])
        continue

    # 4. Run test against oracle
    cmd_test = [f'./build-native/tests/{test_target}', '--shape', str(n), str(k)]
    res_test = subprocess.run(cmd_test)
    if res_test.returncode != 0:
        print(f'ERROR: Test failed for [{n}, {k}]. Reverting.')
        subprocess.run(['git', 'checkout', '--', fpath])
        continue

    print(f'== Shape [{n}, {k}]: All tests PASSED. Ready for benchmark/commit.')
"
```

---

## 6. Exploratory Route Inspection & Boundary Cliff Detection

For deep manual investigation or diagnosing performance cliffs across token boundaries:

### Shape Discovery & Schedule Inspection
```bash
# List all registered shapes, token bands, and L2 residency
$PY -m ckt shapes list --qtype <qtype>

# Inspect an individual shape file's schedules and token dispatch bands
$PY -m ckt shapes inspect --file src/ops/linear/<qtype>/shapes/n<N>_k<K>.{cu,cpp}
```

### Boundary Cliff Probing
```bash
# Probe route boundaries ([B, B+1]) with cold L2 flushes
$PY -m ckt shapes bench \
    --qtype <qtype> --n <N> --k <K> --flush-mib 32 \
    --csv-out cuda-kernel-toolkit/results/<qtype>_base_n<N>_k<K>.csv
```

### SASS & Register Resource Audit
Inspect kernel registers, stack usage, shared memory, and instructions without whole-binary dumps:
```bash
# Inspect resources (REG, STACK, SHARED, LOCAL)
cuobjdump -res-usage build-native/bench/infernix_linear_bench

# Inspect SASS for a specific kernel symbol
cuobjdump -sass -fun <exact_symbol_name> build-native/bench/infernix_linear_bench
```

---

## 7. Summary of Q8 Shapes Retuned on NVIDIA GB10 (`sm_121a`)

All 30 retuned shapes below passed 100% mathematical oracle tests (`infernix_linear_q8_a16_test`), satisfied request and column invariance, and were committed after cold-L2 hardware verification:

| Shape `[N, K]` | File | Optimal Token Ranges & Routes | Measured Speedup | Tests | Status | Commit |
|---|---|---|---|---|---|---|
| `[1024, 2048]` | `n1024_k2048.cpp` | $T \le 128$: SIMT/Sliced-K; $T \le 256$: `mma_r64x16_t48_k128_a1`; $T > 384$: `mma_r64_t96` | **1.92x (T=256)**, **1.41x (T=512)** | PASS | Committed | `0989ada5` |
| `[1024, 5120]` | `n1024_k5120.cpp` | $T \le 128$: SIMT/Sliced-K; $T \le 281$: `mma_r64x16_t48_k128_a1`; $T \le 459$: `mma_r64_t96`; $T > 459$: `mma_r64_t128` | **1.35x–1.68x (T=140..512)** | PASS | Committed | `c06e16ad` |
| `[1280, 2560]` | `n1280_k2560.cu` | $T \le 48$: Stream; $T \le 192$: `mma_r64x16_t48_k128_a1`; $T \le 256$: `mma_r64x32_t64_k128_a1`; $T > 384$: `mma_r128_t64` | **1.10x–1.20x (T=70..384)** | PASS | Committed | `fcf19a34` |
| `[2048, 4096]` | `n2048_k4096.cu` | $T \le 8$: SIMT; $T \le 128$: `mma_r32_t64`; $T \le 281$: `mma_r64_t96`; $T \le 459$: `mma_r96_t96` | **1.17x–1.22x (T=140..192)** | PASS | Committed | `1b187833` |
| `[2048, 4608]` | `n2048_k4608.cpp` | $T \le 32$: SIMT; $T \le 114$: `mma_r64x32_t64_k128_a1`; $T \le 192$: `mma_r32_t64`; $T \le 281$: `mma_r64_t96` | **1.18x (T=192)** | PASS | Committed | `320d8530` |
| `[2048, 16384]` | `n2048_k16384.cu` | $T \le 32$: Sliced-K; $T \le 192$: `mma_r64_t96`; $T \le 281$: `mma_r96_t96`; $T \le 459$: `mma_r128_t80` | **1.38x–1.69x (T=64..384)** | PASS | Committed | `37a251cf` |
| `[2560, 640]` | `n2560_k640.cu` | $T \le 16$: Stream; $T \le 44$: `mma_r32_t64/t128`; $T \le 96$: `mma_r64x16_t48_k128_a1`; $T \le 192$: `mma_r64_t96` | **1.15x–1.30x (T=18..192)** | PASS | Committed | `fbe6517c` |
| `[2560, 2560]` | `n2560_k2560.cu` | $T \le 2$: Stream; $T \le 16$: `mma_r64x16_t48_k128_a1`; $T \le 128$: `mma_r64x32_t64_k128_a1`; $T \le 384$: `mma_r64_t96` | **1.18x–1.44x (T=54..512)** | PASS | Committed | `1e43414a` |
| `[2560, 6144]` | `n2560_k6144.cu` | $T \le 8$: SIMT; $T \le 76$: `mma_r64x16_t48_k128_a1`; $T \le 128$: `mma_r64x32_t64_k128_a1`; $T \le 459$: `mma_r96_t96` | **1.15x–1.85x (T=15..459)** | PASS | Committed | `9d70b1e2` |
| `[320, 10240]` | `n320_k10240.cu` | $T \le 64$: Stream/SIMT; $T > 64$: `mma_r64x16_t48_k128_a1` | **1.25x–1.33x (T=70..512)** | PASS | Committed | `f5662a54` |
| `[324, 10240]` | `n324_k10240.cu` | $T \le 64$: Stream/SIMT; $T > 64$: `mma_r64x16_t48_k128_a1` | **1.10x–1.32x (T=70..512)** | PASS | Committed | `e1031a73` |
| `[4608, 4608]` | `n4608_k4608.cpp` | $T \le 8$: SIMT; $T \le 48$: `mma_r64x16_t48_k128_a1`; $T \le 192$: `mma_r96_t96`; $T \le 306$: `mma_r128_t64` | **1.50x (T=16)**, **1.48x (T=192)** | PASS | Committed | `709d97e2` |
| `[5120, 4608]` | `n5120_k4608.cpp` | $T \le 5$: SIMT; $T \le 48$: `mma_r64x16_t48_k128_a1`; $T \le 152$: `mma_r128_t80`; $T \le 192$: `mma_r96_t96` | **1.27x (T=16)**, **1.34x (T=140)** | PASS | Committed | `eae7a0cc` |
| `[5120, 6144]` | `n5120_k6144.cu` | $T \le 32$: Sliced-K; $T \le 64$: `sliced_r16_t32_w4_s2`; $T \le 96$: `mma_r64_t96`; $T \le 152$: `mma_r128_t80` | **1.45x–1.83x (T=35..192)** | PASS | Committed | `bd1068e2` |
| `[5120, 10240]` | `n5120_k10240.cu` | $T \le 24$: Sliced-K; $T \le 48$: `sliced_r16_t32_w4_s2`; $T \le 64$: `mma_r64x32_t64_k128_a1`; $T \le 152$: `mma_r128_t80` | **1.58x–1.86x (T=25..152)** | PASS | Committed | `02004d3b` |
| `[5120, 17408]` | `n5120_k17408.cu` | $T \le 24$: Sliced-K; $T \le 64$: `sliced_r16_t32_w4_s2`; $T \le 152$: `mma_r128_t80`; $T \le 192$: `mma_r96_t96` | **1.62x–2.04x (T=25..152)** | PASS | Committed | `d434a7d9` |
| `[5120, 25600]` | `n5120_k25600.cu` | $T \le 32$: Sliced-K; $T \le 54$: `sliced_r16_t32_w4_s2`; $T \le 64$: `mma_r64x32_t64_k128_a1`; $T \le 152$: `mma_r128_t80` | **1.62x–1.80x (T=35..152)** | PASS | Committed | `ab8d425f` |
| `[6144, 2560]` | `n6144_k2560.cu` | $T \le 16$: `sliced_r16_t16_w4_s2`; $T \le 24$: `sliced_r16_t32_w4_s2`; $T \le 128$: `mma_r64_t128`; $T \le 152$: `mma_r128_t80` | **1.56x–1.82x (T=82..512)** | PASS | Committed | `47b8e1a4` |
| `[6144, 5120]` | `n6144_k5120.cu` | $T \le 16$: Sliced-K; $T \le 54$: `sliced_r16_t32_w4_s2`; $T \le 64$: `mma_r64x32_t64_k128_a1`; $T \le 152$: `mma_r128_t80` | **1.59x–1.83x (T=17..152)** | PASS | Committed | `c75d6606` |
| `[9216, 2048]` | `n9216_k2048.cpp` | $T \le 8$: SIMT; $T \le 32$: `mma_r64x32_t64_k128_a1`; $T \le 96$: `mma_r96_t96`; $T \le 192$: `mma_r64_t96` | **1.21x (T=24)**, **1.08x (T=64)** | PASS | Committed | `c86186af` |
| `[10240, 320]` | `n10240_k320.cu` | $T \le 2$: `simt_r8_t4`; $T \le 4$: `simt_r8_t8`; $T \le 12$: `simt_r8_t4`; $T \le 16$: `simt_r8_t8`; $T \le 64$: Stream | **1.15x–1.47x (T=2..140)** | PASS | Committed | `a11fbc9b` |
| `[10240, 2560]` | `n10240_k2560.cu` | $T \le 8$: SIMT; $T \le 24$: `mma_r64x16_t48_k128_a1`; $T \le 64$: `mma_r64x32_t64_k128_a1`; $T \le 281$: `mma_r96_t96` | **1.53x–1.86x (T=24..512)** | PASS | Committed | `1af6c84a` |
| `[12288, 2560]` | `n12288_k2560.cu` | $T \le 8$: SIMT; $T \le 32$: `mma_r64x16_t48_k128_a1`; $T \le 96$: `mma_r64x32_t64_k128_a1`; $T \le 281$: `mma_r96_t96` | **1.50x–2.57x (T=35..512)** | PASS | Committed | `90f84b42` |
| `[12800, 2560]` | `n12800_k2560.cu` | $T \le 8$: `simt_r8_t4`; $T \le 32$: `mma_r64x16_t48_k128_a1`; $T \le 128$: `mma_r64x32_t64_k128_a1` | **1.15x–1.82x (T=4..140)** | PASS | Committed | `ac162908` |
| `[13952, 2560]` | `n13952_k2560.cu` | $T \le 8$: `simt_r8_t4`; $T \le 32$: `mma_r64x16_t48_k128_a1`; $T \le 108$: `mma_r64x32_t64_k128_a1` | **1.20x–1.92x (T=4..192)** | PASS | Committed | `5e9a4db2` |
| `[14336, 5120]` | `n14336_k5120.cu` | $T \le 24$: Sliced-K; $T \le 54$: `sliced_r16_t32_w4_s2`; $T \le 96$: `mma_r96_t96`; $T > 96$: `mma_r64_t128` | **1.92x–1.97x (T=28..54)** | PASS | Committed | `f8e0b887` |
| `[16384, 2560]` | `n16384_k2560.cu` | GDN convolution snapshot & Stream/MMA dispatch | **Optimal (0% delta)** | PASS | Retained | - |
| `[17408, 5120]` | `n17408_k5120.cu` | $T \le 24$: Sliced-K; $T \le 54$: `sliced_r16_t32_w4_s2`; $T \le 96$: `mma_r96_t96`; $T > 96$: `mma_r64_t128` | **1.94x–1.98x (T=28..54)** | PASS | Committed | `f046df62` |
| `[34816, 5120]` | `n34816_k5120.cu` | $T \le 24$: Sliced-K; $T \le 54$: `sliced_r16_t32_w4_s2`; $T \le 64$: `mma_r64x32_t64_k128_a1`; $T \le 96$: `mma_r96_t96` | **1.98x–2.22x (T=25..54)** | PASS | Committed | `c3670c94` |
| `[248320, 5120]` | `n248320_k5120.cu` | $T \le 32$: Sliced-K; $T \le 54$: `mma_r64x32_t64_k128_a1`; $T \le 76$: `mma_r128_t80`; $T \le 96$: `mma_r96_t96` | **1.05x (T=64)** | PASS | Committed | `29a24f3e` |

---

## 8. Q4 Family Manual Kernel Optimization & Autotuner Overhaul Methodology on GB10

When initial end-to-end benchmarks after automated tuning showed no improvement on Q4 models, we executed a 7-step root-cause investigation, kernel redesign, and autotuner overhaul on NVIDIA GB10 (`sm_121a`, 48 SMs, 24 MiB L2 cache, 246 GB/s sustained DRAM read bandwidth, 99 KiB opt-in shared memory per SM).

### Step 1: Diagnosing Why `ckt autotune` Became Very Slow (~25 Minutes/Shape)
Reviewing recent changes in `cuda-kernel-toolkit/ckt/autotune/runner.py` and `cli.py` exposed four compounding bottlenecks:
1. **Per-Token Subprocess Spawning**: `BenchmarkRunner.measure()` spawned a new sweep subprocess for *every single `(candidate, token)` pair*. Because sweep binaries (`infernix_q8_route_sweep`, `infernix_q4_route_sweep`) evaluate all ~36–74 candidates at a token count $T$ in one invocation, spawning per candidate multiplied GPU invocations by $36\times\text{–}74\times$.
   - *Fix*: Added `_sweep_batch_cache` keyed by `(sweep_exe, qtype, n, k, flush_mib)` and pre-warmed all token grid points in a **single batched `--tokens t1,t2,...` invocation** (`prewarm_tokens()`), cutting per-shape sweep runtime from ~25 minutes to **~8–12 seconds**.
2. **Excessive Iterations per Sweep**: Lowered default sweep iterations from `--warmup 3 --iters 10` to `--warmup 2 --iters 5` (matching `ckt shapes bench`).
3. **Redundant `recheck_tokens` Re-Benchmarking**: `cli.py` was re-running `infernix_linear_bench` across up to 8 tokens after the sweep had already measured the exact same cold-L2 CUDA graphs.
4. **Denormal Subnormal Payloads in `infernix_linear_bench`**: Raw `0x3c00` byte fills created FP32 subnormals when interpreted as FP32/E4M3 scales; fixed in `bench/ops/linear_bench.cu`.

### Step 2: Why the Old Autotuner Failed to Improve Q4 (0% Change)
Running the unmodified autotuner on all 10 Q4 shapes produced `ALREADY OPTIMAL` with zero edits because:
1. **Zero Q4 Candidate Synthesis**: `CandidateRegistry._generate_family_template_candidates()` in `registry.py` only had branches for `fp8` and `q8`. For `q4`, it returned `[]`, so the autotuner only knew about routes already written in the target shape file.
2. **No Q4 Route Sweep Binary**: `BenchmarkRunner` had no `infernix_q4_route_sweep` executable and fell back to `infernix_linear_bench`, which only runs the already-compiled production dispatch function (`ops::linear`) and ignores the candidate route being tested.
3. **False Rejection via Bitwise `memcmp == 0` Against RTX 5090 Incumbents**: `runner.py` required `bit_exact == 1` (`memcmp == 0` against the current RTX 5090 route) even when `workload.requires_byte_identity` was `False`. On Q4, switching between `KWarps=4` (`sliced_r32_t32_w4_s2`), `KWarps=2` (`sliced_r32_t64_w2_s1`), and `KWarps=1` (`mma_r64_t64_k128_s2_a1`) changes the FP32 K-reduction tree order, producing 1 BF16 ULP differences while passing the FP64 mathematical oracle (`numeric_ok == 1`).

### Steps 3 & 4: Manual Audit of Q4 GEMV, SIMT, Sliced-K MMA, and Full MMA Kernels on GB10
Inspecting `src/ops/linear/q4/` (`q4_schedule.cuh`, `q4_a16_mma.cuh`, `q4_a16_sliced_k_mma.cuh`, `q4_a16_simt.cuh`, `q4_a16_gemv.cuh`, `q4_instances.cuh`) and profiling with `ckt static` and `ckt shapes bench` revealed why Q4 inference was bottlenecked on GB10:

1. **The `T >= 65` Full-MMA Bandwidth Collapse (`200 GB/s` -> `130 GB/s`)**:
   - At $T=64$, `MmaR64T64K128S2A1` (`BlockK=128, Stages=2, ActivationStages=1`, 8 warps = 256 threads) achieved **184–200 GB/s (75–81% of GB10's 246 GB/s DRAM bandwidth)** because `BlockK=128` (`GPB=2`) utilizes all 256 threads during 16-byte `cp.async` weight code loads, halves `__syncthreads()` barriers, and prefetches the next weight tile *before* the 8-step MMA loop.
   - However, at $T \ge 65$ (`MmaR64T72`, `T80`, `T96`, `T112`, `T120`, `T128`), all schedules had been forced down to `BlockK=64, ActivationStages=2` and 4 warps (`128` threads, `REG=152`, **1 block/SM = 25% occupancy**) due to:
     - Duplicate `Pair32` scale storage in `Q4A16MmaSchedule` (`kScaleRowBytes` allocated 4 bytes per group instead of 4 bytes per *pair* of groups when `kGroupsPerK == 2`).
     - A static `48 KiB` shared-memory `static_assert` in `q4_schedule.cuh` and `q4_a16_mma.cuh`.
   - Furthermore, in `q4_a16_mma.cuh`, `cp_wait<0>()` was called before `decode_weight`, stalling on the activation load instead of overlapping `decode_weight` with `stage_activation`, and `decode_weight` decoded weights 2 at a time (`LDS.U8` + `STS.32`) instead of 8 at a time (`LDS.U32` -> `Q4Bf16x8Bits` -> 16-byte `STS.128`).
2. **The `50,176`-Byte Shared-Memory Occupancy Cliff on GB10 (`2 blocks/SM` -> `1 block/SM`)**:
   - GB10 provides `102,400` bytes (`100 KiB`) of shared memory per SM with `1,024` bytes of driver-reserved shared memory per CTA. Sustaining **2 blocks/SM (`512` threads = `50%` occupancy)** requires `kSharedBytes <= 50,176` bytes (`49.0 KiB`).
   - With `S=2` weight stages, `MmaR64T112` used `53,760` bytes (`> 50,176` bytes), dropping occupancy from 2 blocks/SM to 1 block/SM (`190 GB/s -> 132 GB/s`).
   - Because `decode_weight` consumes `Cr[0]` and `Sr[0]` into `As` *before* `stage_quant` prefetches the next weight tile prior to the MMA loop, setting `Stages=1, ActivationStages=1` (`S=1, BS=1`) with `cp_wait<1>()` before `decode_weight` and `cp_wait<0>()` before MMA preserves full async weight prefetch during MMA AND async activation overlap during `decode_weight` while saving `4,352` bytes of shared memory:
     - `MmaR64T96` drops to `45,312` bytes (`44.25 KiB <= 48 KiB` static shared memory, **2 blocks/SM**).
     - `MmaR64T112` drops to `49,408` bytes (`<= 50,176` bytes, **2 blocks/SM**).
     - `MmaR32T128K64S2A2` upgraded to `<32, 128, 128, 16, 32, 2, 2, ..., 1>` (`45,312` bytes `<= 48 KiB`, 8 warps, **2 blocks/SM**).
3. **Sliced-K MMA `ldmatrix_x2` Duplication for `BlockRows = 32` (`MT == 2`)**:
   - In `q4_a16_sliced_k_mma.cuh`, `ldmatrix_x2` for activation fragments `b0, b1` was inside the `mi` loop (`0..MT-1`), executing redundant shared-memory matrix loads for `BlockRows = 32`.
   - Hoisting `ldmatrix_x2` out of the `mi` loop for `MT == 2` and using `q4_sliced_k_integer_quad` (`__byte_perm` + `__hsub2` 4-weight decode) while keeping `MT == 1` (`BlockRows = 16`) specialized at `REG = 64` (`8 blocks/SM`) cut `ldmatrix_x2` traffic in half without reducing `BlockRows = 16` occupancy.

### Steps 5, 6 & 7: Rebuilding the Q4 Sweep Binary, Autotuner, and GB10 Dispatch Ladders
1. Created [`bench/ops/q4_route_sweep.cu`](file:///opt/etc/Infernix/bench/ops/q4_route_sweep.cu) (`infernix_q4_route_sweep`) to benchmark all 31+ registered Q4 routes across arbitrary $[N, K]$ and token lists `--tokens t1,t2,...` with `--flush-mib 32` and FP64/reference validation.
2. Connected `infernix_q4_route_sweep` and all Q4 candidate templates in `cuda-kernel-toolkit/ckt/autotune/runner.py`, `registry.py`, and `generator.py` (including preserving leading `if (tokens % 4 != 0)` precondition guards).
3. Swept all Q4 shapes on GB10, updated `src/ops/linear/q4/shapes/`, verified 100% mathematical oracle pass (`./build-native/tests/infernix_linear_q4_a16_test`), and measured cold-L2 (`--flush-mib 32`) boundary performance:

| Shape `[N, K]` | File | Key GB10 Kernel & Dispatch Changes | Baseline Latency (`us`) | Optimized Latency (`us`) | Speedup / BW | Oracle Test |
|---|---|---|---|---|---|---|
| `[1024, 5120]` | `n1024_k5120.cu` | $T=3..16$: `sliced_r16_t16_w4_s2`; $T=33..48$: `sliced_r16_t16_w2_s2`; $T=49..128$: `mma_r32_t32_k128_s2_a2` | $T=8$: `30.53`<br>$T=48$: `34.66`<br>$T=128$: `43.20` | $T=8$: **`20.32`**<br>$T=48$: **`28.48`**<br>$T=128$: **`38.78`** | **1.50x ($T=8$)**<br>**1.22x ($T=48$)** | PASS |
| `[4096, 5120]` | `n4096_k5120.cu` | $T=2..4$: `sliced_k5120_t4`; $T=33..64$: `mma_r32_t64_k128_s2_a1`; $T=65..96$: `mma_r64_t80/t96`; $T=97..128$: `mma_r64_t64_k64` | $T=64$: `98.18`<br>$T=65$: `128.86`<br>$T=81$: `129.02`<br>$T=128$: `137.09` | $T=64$: **`71.55`**<br>$T=65$: **`81.73`**<br>$T=81$: **`89.92`**<br>$T=128$: **`102.27`** | **1.37x ($T=64$)**<br>**1.58x ($T=65$)**<br>**1.43x ($T=81$)**<br>**1.34x ($T=128$)** | PASS |
| `[5120, 6144]` | `n5120_k6144.cu` | $T=2..4$: `simt_r4_t4`; $T=33..64$: `sliced_r32_t64_w2_s1`; $T=65..96$: `mma_r64_t80/t96`; $T=129..192$: `mma_r64_t96` | $T=64$: `145.82`<br>$T=80$: `195.14`<br>$T=96$: `197.82`<br>$T=192$: `375.49` | $T=64$: **`114.53`**<br>$T=80$: **`114.50`**<br>$T=96$: **`120.67`**<br>$T=192$: **`225.15`** | **1.27x ($T=64$)**<br>**1.70x ($T=80$)**<br>**1.64x ($T=96$)**<br>**1.67x ($T=192$)** | PASS |
| `[6144, 5120]` | `n6144_k5120.cu` | $T=2..4$: `sliced_r16_t16`; $T=17..24$: `sliced_r32_t24`; $T=33..64$: `sliced_r32_t64_w2_s1`; $T=65..96$: `mma_r64_t80/t96` | $T=48$: `128.86`<br>$T=80$: `198.46`<br>$T=96$: `198.46`<br>$T=192$: `324.32` | $T=48$: **`100.19`**<br>$T=80$: **`106.34`**<br>$T=96$: **`116.74`**<br>$T=192$: **`206.66`** | **1.29x ($T=48$)**<br>**1.87x ($T=80$)**<br>**1.70x ($T=96$)**<br>**1.57x ($T=192$)** | PASS |
| `[7168, 5120]` | `n7168_k5120.cu` | $T=2..3$: `simt_r4_t4`; $T=4..16$: `sliced_k5120_t16`; $T=17..32$: `sliced_r32_t32_w4_s2`; $T=33..64$: `mma_r32_t64_k128_s2_a1`; $T=65..96$: `mma_r64_t80/t96` | $T=32$: `145.28`<br>$T=64$: `148.29`<br>$T=80$: `218.72`<br>$T=112$: `253.79` | $T=32$: **`100.19`**<br>$T=64$: **`128.74`**<br>$T=80$: **`155.49`**<br>$T=112$: **`200.54`** | **1.45x ($T=32$)**<br>**1.41x ($T=80$)**<br>**1.27x ($T=112$)** | PASS |
| `[34816, 5120]` | `n34816_k5120.cu` | $T=2..4$: `simt_r4_t4`; $T=5..16$: `sliced_k5120_t16`; $T=33..64$: `mma_r32_t64_k128_s2_a1`; $T=65..112$: `mma_r64_t80/t96/t112` (`K128S1A1`) | $T=63$: `890.85`<br>$T=64$: `542.43`<br>$T=80$: `762.88`<br>$T=96$: `784.45`<br>$T=112$: `973.86` | $T=63$: **`505.70`**<br>$T=64$: **`505.70`**<br>$T=80$: **`526.08`**<br>$T=96$: **`544.61`**<br>$T=112$: **`612.16`** | **1.76x ($T=63$)**<br>**1.45x ($T=80$)**<br>**1.44x ($T=96$)**<br>**1.59x ($T=112$)** | PASS |
| `[131072, 2048]` | `n131072_k2048.cu` | $T=17..24$: `sliced_r32_t24`; $T=33..64$: `mma_r64_t64_k128_s2_a1`; $T=65..112$: `mma_r64_t80/t96/t112` (`215 GB/s = 87.5%` DRAM BW) | $T=64$: `793.57`<br>$T=80$: `1132.42`<br>$T=96$: `1145.66`<br>$T=112$: `1160.16` | $T=64$: **`736.10`**<br>$T=80$: **`761.73`**<br>$T=96$: **`794.46`**<br>$T=112$: **`870.24`** | **1.49x ($T=80$)**<br>**1.44x ($T=96$)**<br>**1.33x ($T=112$)** | PASS |
| `[131072, 5120]` | `n131072_k5120.cu` | $T=5..8$: `sliced_k5120_t8`; $T=17..24$: `sliced_r32_t24`; $T=65..112$: `mma_r64_t80/t96/t112` (`207 GB/s = 84.2%` DRAM BW, `71.0 TFLOP/s`) | $T=65$: `2801.79`<br>$T=80$: `2810.50`<br>$T=81$: `2908.86`<br>$T=112$: `2800.16` | $T=65$: **`1806.18`**<br>$T=80$: **`1831.94`**<br>$T=81$: **`1857.38`**<br>$T=112$: **`2116.42`** | **1.55x ($T=65$)**<br>**1.53x ($T=80$)**<br>**1.57x ($T=81$)**<br>**1.32x ($T=112$)** | PASS |

---

## 9. Per-Family Autonomous Agent Playbooks (`docs/maintainer/autotune/`)

Each quantization family has a dedicated, self-contained agent instruction playbook with its exact source files, schedule structs, physical storage layout, known GB10 bottlenecks, and diagnostic decision matrix (how to interpret benchmark/static-analysis results and what action to take in each outcome):

- **NVFP4 (`nvfp4`, 15 shapes)**: [`docs/maintainer/autotune/nvfp4-agent-playbook.md`](file:///opt/etc/Infernix/docs/maintainer/autotune/nvfp4-agent-playbook.md)
- **Q8 (`q8`, 31 shapes)**: [`docs/maintainer/autotune/q8-agent-playbook.md`](file:///opt/etc/Infernix/docs/maintainer/autotune/q8-agent-playbook.md)
- **BF16 (`bf16`, 25 shapes)**: [`docs/maintainer/autotune/bf16-agent-playbook.md`](file:///opt/etc/Infernix/docs/maintainer/autotune/bf16-agent-playbook.md)
- **FP8 (`fp8`, 11 shapes)**: [`docs/maintainer/autotune/fp8-agent-playbook.md`](file:///opt/etc/Infernix/docs/maintainer/autotune/fp8-agent-playbook.md)
- **Q5 (`q5`, 7 shapes)**: [`docs/maintainer/autotune/q5-agent-playbook.md`](file:///opt/etc/Infernix/docs/maintainer/autotune/q5-agent-playbook.md)
- **Q6 (`q6`, 4 shapes)**: [`docs/maintainer/autotune/q6-agent-playbook.md`](file:///opt/etc/Infernix/docs/maintainer/autotune/q6-agent-playbook.md)
- **Q4 (`q4`, 10 shapes — Reference Case Study)**: [`docs/maintainer/autotune/q4-agent-playbook.md`](file:///opt/etc/Infernix/docs/maintainer/autotune/q4-agent-playbook.md)

---

## 10. Mandatory Agent Guardrails & Common Failure Modes Across All Families

When an agent executes any family playbook, it MUST enforce these five non-negotiable engineering rules (derived from post-audit findings on FP8 commits `b26d04f5`, `31975b76`, and `f758eb0a`):

1. **Never Treat Kernel Headers (`*.cuh`) or Schedule Instances (`*_instances.cuh`) as Read-Only Black Boxes**:
   - Re-sorting existing schedules in 2–3 shape files is NOT a complete family optimization.
   - Always run `$PY -m ckt static build-native/bench/infernix_linear_bench --kernel <family> --sass` to inspect `REG`, `STACK`, `SHARED`, `LOCAL`, and SASS instructions across every compiled kernel instance.
   - Watch for the **9-Warp TMA Register Ceiling on GB10**: A warp-specialized TMA kernel with 1 producer warp + 8 consumer warps = 9 warps (288 threads) places 3 warps on one SM warp scheduler, capping hardware registers at $\lfloor 512 / 3 \rfloor_8 = \mathbf{168\text{ registers/thread}}$. In `fp8_a8_mma_common.cuh`, `PingPong` fragment staging (`64` fragment regs + `128` accumulator regs) spilled 104 bytes/thread (`STACK=104`, 57 `LDL`/`STL` instructions in the inner loop) on `128x256` A8 TMA kernels until `fp8_mma_compute_stage` was updated to stream B fragments on one plane when `kMmaTokens * kMmaRows >= 32` (`STACK=0, LOCAL=0`).
2. **Engineer 2-Blocks/SM (`<= 50,176` B SMEM) and L2-Swizzled (`Grouped` / `RowFast`) TMA Schedules for GB10**:
   - On GB10 (`48 SMs`, `24 MiB L2 cache`, `100 KiB SMEM/SM`), 1-block/SM TMA schedules (`98 KiB` SMEM) at $T=384..3072$ leave half of the SM's 65,536 registers idle; adding 2-blocks/SM `BlockK=64` TMA schedules (`Fp8A16TmaT128R128S2B2K64`, `Fp8A16TmaT64R128S3B2K64`, `49.2 KiB` SMEM $\to$ 96 CTAs/wave) increases throughput from ~75–85 TFLOP/s to **100–105.5 TFLOP/s** (up to **1.35x faster**).
   - At large $T \ge 1536$, `TokenFast` CTA rasterization streams 30–80 MiB of activations per weight row tile and thrashes GB10's 24 MiB L2 cache. Never fall back to non-TMA `Fp8A16MmaSchedule` (`MmaR128T256K64` / `MmaR64T128K64`, ~68 TFLOP/s, which caused +20% to +47% upward cliffs across 7 shapes); instead, expose `RasterGroupRows = 8` (`Fp8MmaRaster::Grouped, 8`) or `Fp8MmaRaster::RowFast` on TMA schedules (`Fp8A16TmaT128R128G8`, `Fp8A16TmaT64R256G8`, `Fp8A16TmaT64R256RF`), sustaining **88–97 TFLOP/s** with **0 ULP bit-identity**.
3. **Never Confuse Incumbent Reduction-Tree Hash Identity at $T \ge 9$ with Column-Invariance Contracts**:
   - Always check `tests/ops/linear/test_<family>_*.cpp` (`fp8_a16_column_invariance()`, etc.) to see which shapes and token windows actually require column invariance across $T$.
   - For example, in FP8 (`test_fp8_a16.cpp`), default-route column invariance is tested ONLY on the 5 Flash-Next shapes (`16384x2560`, `13312x2560`, `2560x6144`, `1280x2560` for $T \in [1..8]$, and `2560x640` for $T \in [1..48]$). Holding slower `KWarps=2` or `KWarps=4` Sliced-K schedules at $T=9..32$ (or $T=2..32$ on Qwen3.5 shapes) out of fear of changing a `KWarps=4` reduction hash leaves 8–65% mid-band speedups on the table.
4. **Match `Simt` `TokenTile` to Dispatch `Capacity` at Small $T$**:
   - If a shape dispatches `simt<Geometry, 8, Simt>` for $T \le 8$ (as in `n1280_k2560.cu` and `n2560_k6144.cu`), `Simt::kTokenTile` MUST be `8`, not `4`! Using `TokenTile = 4` for $T=5..8$ launches `token_tiles = 2` and reads the entire weight matrix from DRAM twice. Widening `TokenTile` to `8` with `Streaming` cache policy preserves exact `ValuesPerLane=16, AccumulatorChains=1` column-invariance bits while cutting $T=5..8$ latency by 15–20%.
5. **100% Shape & Policy Coverage Is Mandatory**:
   - Every shape returned by `$PY -m ckt shapes list --qtype <family>` must be benchmarked with `--mode boundaries --flush-mib 32` across all supported policies (`a16`, `a8`, `a4`), including specialized shapes like `n248320_k5120.cu` (`lm_head`).
