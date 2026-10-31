# GB10 (`sm_121a`) Post-Changelog Decoding & Prefill 90%+ Ceiling Optimization Roadmap (2026-10-31)

This document defines **10 concrete, self-contained, high-impact engineering tasks** (`T1`–`T10`) targeting decoding (`T = 1..64`) and prefill (`T = 128..4096`) performance across the **`Q8` (`Q8_G32_FP16`)** and **`NVFP4` (`NVFP4_G16_E4M3`)** kernel families on **NVIDIA DGX Spark GB10 (`sm_121a`)**.

## Hardware Ceilings & Baseline Audit Summary (`ckt env` & Live Sweeps on GB10)

Live hardware capability and microbenchmark ceiling measurements from `PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt env`:

| Hardware Resource / Pipe | `ckt env` Measured Ceiling | `infernix_linear_bench` Reference | `90%+` Target Floor |
|---|---|---|---|
| **SM Configuration** | `48 SMs`, `1,536 threads/SM` (`48 warps/SM`), `65,536 regs/SM`, `99 KiB` (`101,376 B`) opt-in SMEM/CTA | — | Zero `STACK>0` / `LOCAL>0` spills |
| **L2 Cache & Unified Memory** | `24 MiB` (`25,165,824 B`) L2; `128 GiB` LPDDR5X UMA (`device.integrated = true`) | `32–48 MiB` cold-L2 flush | Zero redundant HostPinned+Device copies |
| **Sustained Read Bandwidth** | `238.8 GB/s` (`ckt env` stream read); `273.0 GB/s` theoretical LPDDR5X peak | `246.0 GB/s` (`READ_%` ref) | **`>= 221.4 GB/s` (`>= 90.0%` `READ_%`)** |
| **Shared Memory Bandwidth** | `3,970.8 GB/s` (`128 B/clk/SM`) | — | Bank-conflict-free `LDS.128` / `ldmatrix.x4` |
| **BF16 / FP16 Tensor Core (`HMMA.16816`, FP32 acc)** | `123.74 TFLOP/s` (`bf16`), `123.60 TFLOP/s` (`fp16_f32acc`) | `97.0 TFLOP/s` (`BF16_F32ACC` ref) | **`>= 87.3 TFLOP/s` (`>= 90.0%` of `97.0 TF` ref)** |
| **INT8 Tensor Core (`IMMA.16832.S8`, INT32 acc)** | `247.43 TOP/s` (`int8`) | `200.0 TOP/s` (`INT8_DP` ref) | **`>= 135–180+ TOP/s` (`1.4×–1.8×` BF16 ceiling)** |
| **NVFP4 Tensor Core (`OMMA.16864`, FP32 acc)** | `494.36 TFLOP/s` (`nvfp4`), `490.41 TFLOP/s` (`mxfp4`) | `440.0 TFLOP/s` (`NVFP4_F32ACC` ref) | **`>= 350–396+ TFLOP/s` (`1-term` / `2-term` hardware pipe)** |

---

### Task T1 (Priority P0): Double-Buffer `q8_a16_mma_kernel` Code/Scale Staging, Vectorize `dequant_w`, and Promote Sub-Ceiling Q8 Prefill Shapes to `128x128` Grouped Schedules (`90%+` of `BF16_F32ACC` Ceiling)

#### Affected Files & Symbols
- [q8_a16_mma.cuh](file:///opt/etc/Infernix/src/ops/linear/q8/q8_a16_mma.cuh) (`q8_a16_mma_kernel`, `stage_w`, `dequant_w`)
- [q8_instances.cuh](file:///opt/etc/Infernix/src/ops/linear/q8/q8_instances.cuh) (`MmaR128x64T128K64A1`, `MmaR64T128K64A1`, `MmaR64x16T256K64A1`)
- [n2560_k12800.cpp](file:///opt/etc/Infernix/src/ops/linear/q8/shapes/n2560_k12800.cpp) (`q8_n2560_k12800_select`)
- [n4608_k4608.cpp](file:///opt/etc/Infernix/src/ops/linear/q8/shapes/n4608_k4608.cpp) (`q8_n4608_k4608_select`)
- [n5120_k4608.cpp](file:///opt/etc/Infernix/src/ops/linear/q8/shapes/n5120_k4608.cpp) (`q8_n5120_k4608_select`)
- [n12288_k2048.cpp](file:///opt/etc/Infernix/src/ops/linear/q8/shapes/n12288_k2048.cpp) (`q8_n12288_k2048_select`)
- [n12288_k2560.cpp](file:///opt/etc/Infernix/src/ops/linear/q8/shapes/n12288_k2560.cpp) (`q8_n12288_k2560_select`)
- [n2048_k16384.cpp](file:///opt/etc/Infernix/src/ops/linear/q8/shapes/n2048_k16384.cpp) (`q8_n2048_k16384_select`)
- [n6144_k2560.cpp](file:///opt/etc/Infernix/src/ops/linear/q8/shapes/n6144_k2560.cpp) (`q8_n6144_k2560_select`)
- [n6144_k5120.cpp](file:///opt/etc/Infernix/src/ops/linear/q8/shapes/n6144_k5120.cpp) (`q8_n6144_k5120_select`)
- [n10240_k5120.cpp](file:///opt/etc/Infernix/src/ops/linear/q8/shapes/n10240_k5120.cpp) (`q8_n10240_k5120_select`)

#### Root Cause & Hardware Rationale
- **Evidence**: A full 31-shape cold-L2 sweep of `infernix_linear_bench --qtype Q8_G32_FP16 --routes fast` across `T = 1..4096` shows that **not a single Q8 shape reaches `90%` (`87.3 TFLOP/s`) of the `97.0 TFLOP/s` `BF16_F32ACC` reference ceiling** at `T = 512..4096`:
  - Large shapes using `MmaR128x64T128K64A1` plateau at **`80.72–84.33 TFLOP/s` (`83.22–86.94%` of `97.0 TF`)**:
    - `[16384, 2560]`: `83.35 TFLOP/s` (`85.92%` at `T=4096`, `78.57 TFLOP/s` / `81.00%` at `T=512`)
    - `[17408, 5120]`: `83.62 TFLOP/s` (`86.20%` at `T=4096`, `78.92 TFLOP/s` / `81.36%` at `T=512`)
    - `[34816, 5120]`: `80.72 TFLOP/s` (`83.22%` at `T=4096`, `79.68 TFLOP/s` / `82.14%` at `T=512`)
    - `[25600, 5120]`: `82.61 TFLOP/s` (`85.16%` at `T=4096`, `79.96 TFLOP/s` / `82.43%` at `T=512`)
    - `[14336, 5120]`: `82.77 TFLOP/s` (`85.33%` at `T=4096`, `78.28 TFLOP/s` / `80.70%` at `T=512`)
  - Nine medium/large shapes drop even further to **`59.94–78.79 TFLOP/s` (`61.79–81.23%` of `97.0 TF`)**:
    - `[2560, 12800]`: `60.66 TFLOP/s` (`62.54%`) at `T=4096` vs `73.83 TFLOP/s` (`76.11%`) at `T=512` (`n2560_k12800.cpp` switches to `r64x16_t256` at `T > 512`)
    - `[4608, 4608]`: `59.94 TFLOP/s` (`61.79%`) at `T=4096`, `56.67 TFLOP/s` (`58.43%`) at `T=512`
    - `[5120, 4608]`: `60.21 TFLOP/s` (`62.08%`) at `T=4096`, `57.16 TFLOP/s` (`58.92%`) at `T=512`
    - `[2048, 16384]`: `61.23 TFLOP/s` (`63.13%`) at `T=4096`, `57.22 TFLOP/s` (`58.99%`) at `T=512`
    - `[12288, 2048]`: `64.35 TFLOP/s` (`66.34%`) at `T=4096`, `59.97 TFLOP/s` (`61.82%`) at `T=512`
    - `[12288, 2560]`: `64.37 TFLOP/s` (`66.37%`) at `T=4096`, `60.67 TFLOP/s` (`62.54%`) at `T=512`
    - `[6144, 2560]`: `76.96 TFLOP/s` (`79.34%`) at `T=4096`, `72.37 TFLOP/s` (`74.61%`) at `T=512`
    - `[6144, 5120]`: `77.78 TFLOP/s` (`80.19%`) at `T=4096`, `74.47 TFLOP/s` (`76.78%`) at `T=512`
    - `[10240, 5120]`: `78.79 TFLOP/s` (`81.23%`) at `T=4096`, `76.09 TFLOP/s` (`78.44%`) at `T=512`
- **Root Causes**:
  1. **Single-Buffered `Cr`/`Sr` and Serialized `dequant_w` Before Next-Stage `cp.async` Issue in [`q8_a16_mma.cuh`](file:///opt/etc/Infernix/src/ops/linear/q8/q8_a16_mma.cuh#L79-L267)**:
     Unlike [`nvfp4_a16_mma.cuh`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a16_mma.cuh#L60-L249) (which double-buffers `code_shared[2]` and `scale_shared[2]` and issues `stage_x(next)` + `stage_codes(next)` *before* `widen_codes`), `q8_a16_mma_kernel` allocates single-buffered `Cr[BM * BK]` and `Sr[BM * kScaleCacheBytes]` (`lines 79–80`) and executes `cp_async_wait_group<0>()` -> `__syncthreads()` -> `dequant_w(kt)` -> `__syncthreads()` (`lines 254–258`) *before* issuing `stage_x(next)` and `stage_w(next)` (`lines 263–266`). This leaves the LPDDR5X memory pipe completely idle during `dequant_w` + two CTA barriers every K-tile (`BK=64`).
  2. **Scalar FP32 Conversion in `dequant_w` ([`q8_a16_mma.cuh`](file:///opt/etc/Infernix/src/ops/linear/q8/q8_a16_mma.cuh#L218-L242))**:
     `dequant_w` unpacks each `int8` byte to `float`, multiplies by `float s`, and calls `__floats2bfloat162_rn(w0, w1)` instead of using `q8_bf16_pair_from_s8` / packed `__nv_bfloat162` multiply (`__hmul2`), burning 24 SIMT instructions per 16-byte code load.
  3. **Unpromoted Prefill Tiles on 9 Shapes**:
     `[2560, 12800]`, `[4608, 4608]`, `[5120, 4608]`, `[12288, 2048]`, `[12288, 2560]`, and `[2048, 16384]` still dispatch `r64_t128` or `r64x16_t256` at `T >= 256` instead of `MmaR128x64T128K64A1` (`128x128`, `Grouped, 8`).

#### Required Implementation Steps
1. In [`q8_a16_mma.cuh`](file:///opt/etc/Infernix/src/ops/linear/q8/q8_a16_mma.cuh), double-buffer `Cr[2][BM * BK]` (`2 * 128 * 64 = 16 KiB` for `BM=128`, or overlap `stage_x(next)` + `stage_w(next)` into `Cr[(kt + 1) & 1]` / `Sr[(kt + 1) & 1]` immediately after `cp_async_wait_group<0>()` + `__syncthreads()`, before `dequant_w(kt)`), matching the overlap pattern in [`nvfp4_a16_mma.cuh`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a16_mma.cuh#L244-L253) so global memory `cp.async` for tile `kt + 1` runs concurrently with `dequant_w(kt)` and the first `kSteps/2` HMMA steps.
2. Vectorize `dequant_w` in [`q8_a16_mma.cuh`](file:///opt/etc/Infernix/src/ops/linear/q8/q8_a16_mma.cuh#L218-L242) using `q8_bf16_pair_from_s8` (`ops/linear/q8/q8_shared.cuh` or exact `__nv_bfloat162` scale multiplication where bit-compatible with the FP32 oracle, or pipelined register staging of `uint4` code loads before the barrier).
3. Update the prefill (`T >= 256`) dispatch in [`n2560_k12800.cpp`](file:///opt/etc/Infernix/src/ops/linear/q8/shapes/n2560_k12800.cpp), [`n4608_k4608.cpp`](file:///opt/etc/Infernix/src/ops/linear/q8/shapes/n4608_k4608.cpp), [`n5120_k4608.cpp`](file:///opt/etc/Infernix/src/ops/linear/q8/shapes/n5120_k4608.cpp), [`n12288_k2048.cpp`](file:///opt/etc/Infernix/src/ops/linear/q8/shapes/n12288_k2048.cpp), [`n12288_k2560.cpp`](file:///opt/etc/Infernix/src/ops/linear/q8/shapes/n12288_k2560.cpp), [`n2048_k16384.cpp`](file:///opt/etc/Infernix/src/ops/linear/q8/shapes/n2048_k16384.cpp), [`n6144_k2560.cpp`](file:///opt/etc/Infernix/src/ops/linear/q8/shapes/n6144_k2560.cpp), [`n6144_k5120.cpp`](file:///opt/etc/Infernix/src/ops/linear/q8/shapes/n6144_k5120.cpp), and [`n10240_k5120.cpp`](file:///opt/etc/Infernix/src/ops/linear/q8/shapes/n10240_k5120.cpp) to use `MmaR128x64T128K64A1` (`128x128` `Grouped, 8`).
4. Touch all consuming `.cu` files under `src/ops/linear/q8/` (`find src/ops/linear/q8 -name '*.cu' -exec touch {} +`) before building.

#### Verification & Acceptance Criteria
```bash
find src/ops/linear/q8 -name '*.cu' -exec touch {} +
ninja -C build-native -j "$(nproc)" infernix-serve infernix_linear_bench infernix_linear_q8_a16_test
./build-native/tests/infernix_linear_q8_a16_test
ctest --test-dir build-native -R infernix_cuda_stack_check --output-on-failure
for nk in "16384 2560" "17408 5120" "34816 5120" "2560 12800" "4608 4608" "5120 4608" "12288 2048" "12288 2560" "2048 16384"; do
  read n k <<< "$nk"
  ./build-native/bench/infernix_linear_bench --qtype Q8_G32_FP16 --routes fast --n $n --k $k --t 512 --warmup 2 --repeat 10 | grep "^explicit"
  ./build-native/bench/infernix_linear_bench --qtype Q8_G32_FP16 --routes fast --n $n --k $k --t 4096 --warmup 2 --repeat 10 | grep "^explicit"
done
```
- **Acceptance**:
  - Large Q8 prefill shapes (`N >= 6144, K >= 2560`) achieve **`>= 87.3 TFLOP/s` (`>= 90.0%` of the `97.0 TFLOP/s` `BF16_F32ACC` reference ceiling)** at `T = 4096`.
  - `[2560, 12800]`, `[4608, 4608]`, `[5120, 4608]`, `[12288, 2048]`, `[12288, 2560]`, and `[2048, 16384]` improve from `59.9–64.4 TFLOP/s` to **`>= 80–87+ TFLOP/s`** at `T = 4096`.
  - `infernix_linear_q8_a16_test` and `infernix_cuda_stack_check` pass 100%.

---

### Task T2 (Priority P0): Eliminate Q8 Decode/Verify (`T = 1..64`) Sub-90% Bandwidth Cliffs Across Large-K and Medium-N Shapes (`[6144,5120]`, `[2048,4608]`, `[2048,4096]`, `[2048,16384]`, `[5120,10240]`, `[5120,17408]`, `[2560,2560]`)

#### Affected Files & Symbols
- [n6144_k5120.cpp](file:///opt/etc/Infernix/src/ops/linear/q8/shapes/n6144_k5120.cpp) (`q8_n6144_k5120_select`, `q8_n6144_k5120_select_sliced`)
- [n2048_k4608.cpp](file:///opt/etc/Infernix/src/ops/linear/q8/shapes/n2048_k4608.cpp) (`q8_n2048_k4608_select`, `q8_n2048_k4608_select_sliced`)
- [n2048_k4096.cpp](file:///opt/etc/Infernix/src/ops/linear/q8/shapes/n2048_k4096.cpp) (`q8_n2048_k4096_select`, `q8_n2048_k4096_select_sliced`)
- [n2048_k16384.cpp](file:///opt/etc/Infernix/src/ops/linear/q8/shapes/n2048_k16384.cpp) (`q8_n2048_k16384_select`, `q8_n2048_k16384_select_sliced`)
- [n5120_k10240.cpp](file:///opt/etc/Infernix/src/ops/linear/q8/shapes/n5120_k10240.cpp) (`q8_n5120_k10240_select`, `q8_n5120_k10240_select_sliced`)
- [n5120_k17408.cpp](file:///opt/etc/Infernix/src/ops/linear/q8/shapes/n5120_k17408.cpp) (`q8_n5120_k17408_select`, `q8_n5120_k17408_select_sliced`)
- [n2560_k2560.cpp](file:///opt/etc/Infernix/src/ops/linear/q8/shapes/n2560_k2560.cpp) (`q8_n2560_k2560_select`, `q8_n2560_k2560_select_sliced`)
- [q8_instances.cuh](file:///opt/etc/Infernix/src/ops/linear/q8/q8_instances.cuh) / [q8_gemv.cu](file:///opt/etc/Infernix/src/ops/linear/q8/q8_gemv.cu) / [q8_sliced_k_mma.cu](file:///opt/etc/Infernix/src/ops/linear/q8/q8_sliced_k_mma.cu)

#### Root Cause & Hardware Rationale
- **Evidence**: Live cold-L2 benchmarks (`infernix_linear_bench --qtype Q8_G32_FP16 --routes fast`) reveal severe sub-90% bandwidth cliffs (`< 221.4 GB/s` / `< 90%` `READ_%`) across 7 critical Q8 decode/verification shapes:
  1. **`[6144, 5120]` at `T = 1`**: `173.95 us`, `192.3 GB/s` (**`78.16%` `READ_%`**), whereas `T = 2` runs in `148.32 us` (`225.7 GB/s`, **`91.73%` `READ_%`**) — `T = 1` is `25.6 us` (`17.3%`) slower than `T = 2`! Inspection of [`n6144_k5120.cpp`](file:///opt/etc/Infernix/src/ops/linear/q8/shapes/n6144_k5120.cpp) shows `T = 1` dispatches a suboptimal GEMV/SIMT schedule while `T = 2..16` uses `sliced_r16_t16_k8_s1`.
  2. **`[2048, 4608]` at `T = 1..64`**: `T = 1` is `56.13 us`, `178.9 GB/s` (**`72.72%` `READ_%`**), while `T = 2..16` is `49.98–51.01 us` (`197.1–204.7 GB/s`, `80.1–83.2%`), and `T = 64` drops to `121.3 GB/s` (`49.30%`).
  3. **`[2048, 4096]` at `T = 4` & `T = 32..64`**: `T = 4` drops to `66.98 us` (`133.8 GB/s`, **`54.39%` `READ_%`**) vs `T = 2, 8` at `42.82 us` (`210.5 GB/s`, `85.55%` `READ_%`); `T = 32..64` drops to `141.7–169.3 GB/s` (`57.6–68.8%`).
  4. **`[2048, 16384]` at `T = 2` & `T = 32..64`**: `T = 2` drops to `191.42 us` (`186.6 GB/s`, **`75.87%` `READ_%`**) vs `T = 1, 4` at `158.59 us` (`225.7 GB/s`, `91.76%`); `T = 64` drops to `321.34 us` (`118.3 GB/s`, `48.08%`).
  5. **`[5120, 10240]` at `T = 16`**: jumps to `294.11 us` (`191.1 GB/s`, **`77.67%` `READ_%`**) vs `T = 32` at `268.10 us` (`211.4 GB/s`, `85.95%`).
  6. **`[5120, 17408]` at `T = 64`**: drops to `612.35 us` (`159.4 GB/s`, **`64.78%` `READ_%`**) whereas `[5120, 25600]` sustains `199.8 GB/s` (`81.23%`) at `T = 64`.
  7. **`[2560, 2560]` (`12×`/pass in Qwen4Exp) at `T = 4..64`**: drops from `219.6 GB/s` (`89.25%` at `T = 2`) down to **`150.0–171.7 GB/s` (`60.96–69.78%` `READ_%`)** at `T = 4..32` and `134.9 GB/s` (`54.82%`) at `T = 64`.

#### Required Implementation Steps
1. Sweep all registered GEMV (`gemv_r8_w1_b2`, `gemv_r8_w2_b2`, `gemv_r8_w2_b4`, `gemv_r8_w4_b2`) and sliced-K (`sliced_r16_t8_k8_s2`, `sliced_r16_t8_k4_s2`, `sliced_r32_t8_k4_s2`, `sliced_r16_t16_k8_s1`, `sliced_r32_t16_k4_s2`, `sliced_r16_t32_k4_s2`, `sliced_r16_t64_k4_s2`, `sliced_r32_t64_k4_s2`) instances on GB10 across `T = 1, 2, 4, 8, 16, 24, 32, 48, 64` for `[6144, 5120]`, `[2048, 4608]`, `[2048, 4096]`, `[2048, 16384]`, `[5120, 10240]`, `[5120, 17408]`, and `[2560, 2560]`.
2. For `N = 2048` and `N = 2560` shapes (`128` or `160` 16-row blocks, or `64`/`80` 32-row blocks on 48 SMs), ensure `T = 4..32` uses `sliced_r16_t8_k4_s2` / `sliced_r16_t16_k4_s2` / `sliced_r16_t32_k4_s2` (in both `_select` and `_select_sliced`) instead of falling back to SIMT or 1-stage `s1` schedules that stall on LPDDR5X latency.
3. For `[6144, 5120]` and `[2048, 4608]` at `T = 1`, replace the `72–78%` `READ_%` `T = 1` schedule with the winning `gemv_r8_w2_b2` / `sliced_r16_t8_k8_s2` schedule so `T = 1` matches or beats `T = 2` (`>= 90%` `READ_%` on `[6144, 5120]`, `>= 85–90%` on `[2048, 4608]`).

#### Verification & Acceptance Criteria
```bash
ninja -C build-native -j "$(nproc)" infernix-serve infernix_linear_bench infernix_linear_q8_a16_test
for nk in "6144 5120" "2048 4608" "2048 4096" "2048 16384" "5120 10240" "5120 17408" "2560 2560"; do
  read n k <<< "$nk"
  ./build-native/tests/infernix_linear_q8_a16_test --shape $n $k
  for t in 1 2 4 8 16 32 64; do
    ./build-native/bench/infernix_linear_bench --qtype Q8_G32_FP16 --routes fast --n $n --k $k --t $t --warmup 2 --repeat 10 | grep "^explicit"
  done
done
ctest --test-dir build-native -R infernix_cuda_stack_check --output-on-failure
```
- **Acceptance**:
  - `[6144, 5120]` at `T = 1` improves from `173.95 us` (`78.16%` `READ_%`) to **`<= 149 us` (`>= 91%` `READ_%`)**.
  - `[2048, 4096]` at `T = 4` (`66.98 us`, `54.39%`), `[2048, 16384]` at `T = 2` (`191.42 us`, `75.87%`), and `[5120, 10240]` at `T = 16` (`294.11 us`, `77.67%`) eliminate their non-monotonic latency spikes and sustain **`>= 86–92%` `READ_%`**.
  - `[2560, 2560]` at `T = 4..32` improves from `150.0–171.7 GB/s` (`61.0–69.8%`) to **`>= 205–221+ GB/s` (`>= 83–90%` `READ_%`)**.

---

### Task T3 (Priority P0): Retune Q8 Skinny/LoRA/Drafter Shapes (`[10240,320]`, `[320,10240]`, `[324,10240]`, `[1280,2560]`, `[2560,640]`, `[1024,2048]`, `[1024,5120]`) Across `T = 1..4096` to Reach `90%+` Bandwidth and Compute Ceilings

#### Affected Files & Symbols
- [q8_a16_stream.cuh](file:///opt/etc/Infernix/src/ops/linear/q8/q8_a16_stream.cuh) (`q8_a16_skinny_n_kernel`, `q8_a16_tall_skinny_k320_kernel`)
- [q8_a16_grouped_sliced_k_mma.cuh](file:///opt/etc/Infernix/src/ops/linear/q8/q8_a16_grouped_sliced_k_mma.cuh) (`q8_a16_grouped_sliced_k_mma_kernel`)
- [n10240_k320.cpp](file:///opt/etc/Infernix/src/ops/linear/q8/shapes/n10240_k320.cpp) (`q8_n10240_k320_select`)
- [n320_k10240.cpp](file:///opt/etc/Infernix/src/ops/linear/q8/shapes/n320_k10240.cpp) (`q8_n320_k10240_select`)
- [n324_k10240.cpp](file:///opt/etc/Infernix/src/ops/linear/q8/shapes/n324_k10240.cpp) (`q8_n324_k10240_select`)
- [n1280_k2560.cpp](file:///opt/etc/Infernix/src/ops/linear/q8/shapes/n1280_k2560.cpp) (`q8_n1280_k2560_select`, `q8_n1280_k2560_select_sliced`)
- [n2560_k640.cpp](file:///opt/etc/Infernix/src/ops/linear/q8/shapes/n2560_k640.cpp) (`q8_n2560_k640_select`, `q8_n2560_k640_select_sliced`)
- [n1024_k2048.cpp](file:///opt/etc/Infernix/src/ops/linear/q8/shapes/n1024_k2048.cpp) (`q8_n1024_k2048_select`)
- [n1024_k5120.cpp](file:///opt/etc/Infernix/src/ops/linear/q8/shapes/n1024_k5120.cpp) (`q8_n1024_k5120_select`)

#### Root Cause & Hardware Rationale
- **Evidence**: Live `infernix_linear_bench --qtype Q8_G32_FP16 --routes fast` measurements on the high-invocation-count Qwen4Exp projections (`49×` to `98×` per forward pass!) show major gaps against GB10 ceilings:
  1. **`[10240, 320]` (`98×`/pass)**:
     - At `T = 1..2`: hits `216.2–218.7 GB/s` (`87.9–88.9%` `READ_%`).
     - At `T = 4..8`: drops to **`164.5–190.8 GB/s` (`66.85–77.57%` `READ_%`, `21.25 us` at `T=4` vs `16.00 us` at `T=2` and `16.45 us` at `T=32`!)** because `q8_a16_tall_skinny_k320_kernel` loops serially over `t = 0..T-1` (`#pragma unroll 1 for (int t = 0; t < operands.tokens; ++t)` in [`q8_a16_stream.cuh`](file:///opt/etc/Infernix/src/ops/linear/q8/q8_a16_stream.cuh#L129-L176)) instead of switching to MMA (`r64_t8` / `sliced_r32_t8_k2_s2`) at `T = 4..16`!
     - At `T = 4096`: reaches only **`60.16 TFLOP/s` (`62.02%` of `97.0 TF`)**.
  2. **`[320, 10240]` & `[324, 10240]` (`98×`/pass)**:
     - At `T = 1`: `17.98–18.43 us` (`194.2–203.3 GB/s`, `78.9–82.6%` `READ_%`).
     - At `T = 4..8`: drops to `26.88–35.23 us` (`105.5–136.4 GB/s`, `42.9–55.5%` `READ_%`).
     - At `T = 16..64`: drops to `61.4–223.2 us` (`21.6–62.6 GB/s`, `8.8–25.4%` `READ_%`) because `q8_a16_skinny_n_kernel` (`SplitK = 16`, `MaxT = 8`, looping `t0 += 8` with atomic `float` adds in [`q8_a16_stream.cuh`](file:///opt/etc/Infernix/src/ops/linear/q8/q8_a16_stream.cuh#L49-L115)) stays active up to `T = 48` instead of handing off to `grouped_sliced_r16_t16_k8` / `grouped_sliced_r16_t32_k8` / `grouped_sliced_r16_t64_k8`!
     - At `T = 512..4096`: reaches only **`37.80–58.47 TFLOP/s` (`38.97–60.28%` of `97.0 TF`)** because `N = 320`/`324` (`5` M-blocks of `64`) without split-K at `T = 512` launches only `5 * 4 = 20` CTAs on 48 SMs (`41.7%` SM utilization)!
  3. **`[1280, 2560]` (`49×`/pass) & `[2560, 640]` (`49×`/pass) & `[1024, 2048]` & `[1024, 5120]`**:
     - `[1280, 2560]`: `T = 1..64` sits at `154.0–193.7 GB/s` (`62.6–78.8%` `READ_%`), and `T = 512..4096` sits at `58.95–73.17 TFLOP/s` (`60.8–75.4%`).
     - `[2560, 640]`: `T = 1..64` sits at `114.2–158.0 GB/s` (`46.4–64.2%` `READ_%`), and `T = 4096` sits at `66.37 TFLOP/s` (`68.4%`).
     - `[1024, 2048]` & `[1024, 5120]`: `T = 1..64` sits at `137.4–198.9 GB/s` (`55.8–80.9%` `READ_%`), and `T = 4096` sits at `63.67–70.20 TFLOP/s` (`65.6–72.4%`).

#### Required Implementation Steps
1. For `[10240, 320]` ([`n10240_k320.cpp`](file:///opt/etc/Infernix/src/ops/linear/q8/shapes/n10240_k320.cpp)):
   - Replace `q8_a16_tall_skinny_k320_kernel` at `T = 4..16` with `sliced_r16_t8_k2_s2` / `sliced_r32_t8_k2_s2` or `r64_t16` / `r64_t32` MMA so `T = 4..8` drops from `21.25 us` (`66.8%` `READ_%`) to `<= 16.5 us` (`>= 88–90%` `READ_%`).
   - At `T >= 256`, use `MmaR128x64T128K64A1` (`128x128`) so `T = 4096` rises from `60.16 TFLOP/s` to `>= 78–85 TFLOP/s`.
2. For `[320, 10240]` and `[324, 10240]` ([`n320_k10240.cpp`](file:///opt/etc/Infernix/src/ops/linear/q8/shapes/n320_k10240.cpp), [`n324_k10240.cpp`](file:///opt/etc/Infernix/src/ops/linear/q8/shapes/n324_k10240.cpp), [`q8_a16_grouped_sliced_k_mma.cuh`](file:///opt/etc/Infernix/src/ops/linear/q8/q8_a16_grouped_sliced_k_mma.cuh)):
   - Transition from `q8_a16_skinny_n_kernel` to `q8_a16_grouped_sliced_k_mma_kernel` (`grouped_sliced_r16_t8_k8`, `grouped_sliced_r16_t16_k8`, `grouped_sliced_r16_t32_k8`, `grouped_sliced_r16_t64_k8`) starting at `T = 4` (or `T = 9`), eliminating the `61–223 us` cliff at `T = 8..64`.
   - At `T = 128..512`, use split-K / grouped-sliced-K or `r32_t64` (`10..11` M-blocks × `8` N-blocks = `80..88` CTAs on 48 SMs instead of `20` CTAs) so `T = 512` improves from `37.8 TFLOP/s` (`39.0%`) to `>= 65 TFLOP/s`.
3. For `[1280, 2560]`, `[2560, 640]`, `[1024, 2048]`, and `[1024, 5120]`, tune `T = 1` (`gemv_r4_w2_b4` / `gemv_r8_w2_b4` so CTAs >= 96 on 48 SMs), `T = 2..64` (`sliced_r16_t8_k4_s2` / `sliced_r16_t16_k4_s2`), and `T >= 256` (`MmaR128x64T128K64A1` where `N >= 1280`).

#### Verification & Acceptance Criteria
```bash
find src/ops/linear/q8 -name '*.cu' -exec touch {} +
ninja -C build-native -j "$(nproc)" infernix-serve infernix_linear_bench infernix_linear_q8_a16_test
for nk in "10240 320" "320 10240" "324 10240" "1280 2560" "2560 640" "1024 2048" "1024 5120"; do
  read n k <<< "$nk"
  ./build-native/tests/infernix_linear_q8_a16_test --shape $n $k
  for t in 1 2 4 8 16 32 64 512 4096; do
    ./build-native/bench/infernix_linear_bench --qtype Q8_G32_FP16 --routes fast --n $n --k $k --t $t --warmup 2 --repeat 10 | grep "^explicit"
  done
done
ctest --test-dir build-native -R infernix_cuda_stack_check --output-on-failure
```
- **Acceptance**:
  - `[10240, 320]` at `T = 4..8` improves from `21.25 us` (`66.85%` `READ_%`) to **`<= 16.5 us` (`>= 88–90%` `READ_%`)** and at `T = 4096` improves from `60.16 TFLOP/s` to **`>= 78 TFLOP/s`**.
  - `[320, 10240]` and `[324, 10240]` at `T = 16..64` improve by **`2.5×–3.5×`** (from `61–223 us` down to `<= 25–42 us`) and at `T = 512` improve from `37.8 TFLOP/s` to **`>= 60 TFLOP/s`**.

---

### Task T4 (Priority P0): Enable and Retune NVFP4 `A4` (`1-term`) and `Nvfp4A4Pair` (`2-term`) TMA Prefill Across All 15 NVFP4 Shapes to Reach `90%+` of Effective `OMMA.16864` Pipe Ceiling

#### Affected Files & Symbols
- [nvfp4_a4.cu](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a4.cu) (`nvfp4_linear_a4_supported`, `nvfp4_linear_a4_uses_tma`, `nvfp4_linear_a4_launch`)
- [nvfp4_a4_tma.cuh](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a4_tma.cuh) (`nvfp4_a4_tma_mma_kernel`, `launch_nvfp4_a4_tma_mma`)
- [nvfp4_a4_tma.cu](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a4_tma.cu) (`launch_nvfp4_a4_tma`, `launch_nvfp4_a4_pair`)
- [n5120_k6144.cu](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n5120_k6144.cu) (`kNvfp4N5120K6144`, `uses_a4`, `launch_a4`)
- [n5120_k17408.cu](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n5120_k17408.cu) (`kNvfp4N5120K17408`, `uses_a4`, `launch_a4`)
- [n14336_k5120.cu](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n14336_k5120.cu) (`kNvfp4N14336K5120`)
- [n16384_k5120.cu](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n16384_k5120.cu) (`kNvfp4N16384K5120`)
- [n34816_k5120.cu](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n34816_k5120.cu) (`kNvfp4N34816K5120`)
- [n2560_k6144.cu](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n2560_k6144.cu), [n6144_k2560.cu](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n6144_k2560.cu), [n10240_k2560.cu](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n10240_k2560.cu), [n12800_k2560.cu](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n12800_k2560.cu), [n13952_k2560.cu](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n13952_k2560.cu), [n16384_k2560.cu](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n16384_k2560.cu)

#### Root Cause & Hardware Rationale
- **Evidence**: Live `infernix_linear_bench --qtype NVFP4` sweeps under both `--routes fast` (`A16` + `Nvfp4A4Pair`) and `--policy a4` expose 4 major prefill (`T = 128..4096`) bottlenecks on GB10:
  1. **`--policy a4` is Completely Unimplemented (`nvfp4_linear_a4_supported == false`) for All 10 `K = 2560` / `[2560, 6144]` Shapes**:
     Running `infernix_linear_bench --qtype NVFP4 --policy a4` on `[6144, 2560]`, `[10240, 2560]`, `[12800, 2560]`, `[13952, 2560]`, `[16384, 2560]`, and `[2560, 6144]` falls back to `BF16_F32ACC` at **`74.64–79.59 TFLOP/s` (`17%` of NVFP4 ceiling!)** because [`nvfp4_a4.cu`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a4.cu#L78-L90) only admits the 5 `K = 5120` / `N = 5120` shapes!
  2. **`Nvfp4A4Pair` (`--routes fast`) is Disabled (`pair_min_tokens = 0`) on `[5120, 17408]`, `[14336, 5120]`, `[16384, 5120]`, and `[34816, 5120]`**:
     Under `--routes fast`, `[5120, 17408]` (`73.36 TFLOP/s`), `[14336, 5120]` (`63.90 TFLOP/s`), `[16384, 5120]` (`64.61 TFLOP/s`), and `[34816, 5120]` (`71.56 TFLOP/s`) stay on `BF16_F32ACC` (`65.9–75.6%` of BF16 ceiling, `14.5–16.7%` of NVFP4 ceiling) because `pair_min_tokens = 0` in [`n5120_k17408.cu`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n5120_k17408.cu#L60), [`n14336_k5120.cu`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n14336_k5120.cu#L58), [`n16384_k5120.cu`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n16384_k5120.cu#L58), and [`n34816_k5120.cu`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n34816_k5120.cu#L59). Moreover, [`n14336_k5120.cu`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n14336_k5120.cu#L39) and [`n16384_k5120.cu`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n16384_k5120.cu#L39) still use the ungrouped RTX 5090 `Nvfp4A16MmaSchedule<64, 256, 64, 32, 32, 2, 1>` tail instead of `Grouped, 8` (`64x128` / `128x128`).
  3. **`[5120, 17408]` and `[5120, 6144]` `A4` (`1-term`) Prefill Collapse at `T = 4096` (`196.79 TFLOP/s` / `44.72%` and `258.97 TFLOP/s` / `58.86%`)**:
     In [`n5120_k17408.cu`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n5120_k17408.cu#L45-L56) and [`n5120_k6144.cu`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n5120_k6144.cu#L45-L56), `launch_a4` calls `launch_nvfp4_a4_tma(p, y, stream)` with `Tiled256` (`BM = 256`), which gives `N = 5120` only `20` M-blocks (`20` is not divisible by `48` SMs or `GroupM = 8`) and uses `GroupM = 1` (zero L2 reuse of `b_codes` / `b_scales` across M-blocks!). At `K = 17408, T = 4096`, `GroupM = 1` re-reads `b_codes` (`17408 * 4096 / 2 = 35.65 MiB` > `24 MiB` L2!) `20` times from LPDDR5X DRAM (`63.3 GB/s` effective weight bandwidth, `3,710 us`, **`196.79 TFLOP/s` = `44.72%`**)! Switching `N = 5120` TMA to `Tiled128` (`BM = 128`, `40` M-blocks) with `GroupM = 4` or `8` and `L2_PROMOTION_L2_128B` on all 4 tensor maps (`a_codes`, `a_scales`, `b_codes`, `b_scales` in [`nvfp4_a4_tma.cuh`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a4_tma.cuh#L540-L553)) keeps the active `b_codes` panel inside GB10's `24 MiB` L2.
  4. **Missing `L2_PROMOTION_L2_128B` on `a_codes`, `a_scales`, `b_scales` in [`nvfp4_a4_tma.cuh`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a4_tma.cuh#L540-L553)**:
     `launch_nvfp4_a4_tma_mma` passes `CU_TENSOR_MAP_L2_PROMOTION_NONE` for `a_codes`, `a_scales`, and `b_scales` (`lines 541, 545, 553`), only promoting `b_codes` (`line 549`). Because `GroupM = 8` shares `a_codes` (`128 * 128 / 2 = 8 KiB` per tile) across N-blocks in the group and `b_scales` across M-blocks, enabling `Schedule::kL2Promotion` on `a_codes` and `b_scales` reduces L2 misses during both `1-term` (`A4`) and `2-term` (`Nvfp4A4Pair`, where 1 useful FLOP = 2 hardware `OMMA.16864` FLOPs, so `198+ useful TFLOP/s` = `396+ hardware OMMA TFLOP/s` = `90%+` of the `440.0 TFLOP/s` dense reference).

#### Required Implementation Steps
1. In [`nvfp4_a4_tma.cuh`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a4_tma.cuh#L540-L553), apply `Schedule::kL2Promotion` to `a_codes`, `b_codes`, and `b_scales` tensor maps, and add `GroupM` specialization in [`nvfp4_a4_tma.cu`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a4_tma.cu) (`GroupM = 8` for `N = 5120, K = 6144/17408` with `BM = 128`, `Tiled128`).
2. Update [`n5120_k6144.cu`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n5120_k6144.cu) and [`n5120_k17408.cu`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n5120_k17408.cu) so `launch_a4` at `uses_tma(T)` (`T >= 256`) uses `Tiled128` (`BM = 128, Stages = 3, GroupM = 8`) instead of `Tiled256` (`BM = 256, GroupM = 1`).
3. Enable `A4` (`uses_a4`, `launch_a4`) in [`nvfp4_a4.cu`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a4.cu) and the `K = 2560` / `[2560, 6144]` shape files (`[6144, 2560]`, `[10240, 2560]`, `[12800, 2560]`, `[13952, 2560]`, `[16384, 2560]`, `[2560, 6144]`), and wire `launch_nvfp4_a4_pair` (`pair_min_tokens = 65`) + `Grouped, 8` A16 fallback on `[5120, 6144]`, `[5120, 17408]`, `[14336, 5120]`, `[16384, 5120]`, and `[34816, 5120]`.

#### Verification & Acceptance Criteria
```bash
find src/ops/linear/nvfp4 -name '*.cu' -exec touch {} +
ninja -C build-native -j "$(nproc)" infernix-serve infernix_linear_bench infernix_linear_nvfp4_a16_test infernix_ops_linear_test
./build-native/tests/infernix_linear_nvfp4_a16_test
ctest --test-dir build-native -R "infernix_ops_linear_test|infernix_cuda_stack_check" --output-on-failure
for nk in "5120 6144" "5120 17408" "14336 5120" "16384 5120" "34816 5120" "6144 2560" "10240 2560" "12800 2560" "13952 2560" "16384 2560" "2560 6144"; do
  read n k <<< "$nk"
  ./build-native/bench/infernix_linear_bench --qtype NVFP4 --policy a4 --n $n --k $k --t 4096 --warmup 2 --repeat 10 | grep "^explicit"
  ./build-native/bench/infernix_linear_bench --qtype NVFP4 --routes fast --n $n --k $k --t 4096 --warmup 2 --repeat 10 | grep "^explicit"
done
```
- **Acceptance**:
  - `[5120, 17408]` `--policy a4` at `T = 4096` improves from `196.79 TFLOP/s` (`44.72%`) to **`>= 330–360+ TFLOP/s` (`>= 75–82%+` useful, `>= 90%` hardware pipe)**.
  - `[5120, 6144]` `--policy a4` at `T = 4096` improves from `258.97 TFLOP/s` (`58.86%`) to **`>= 320–355+ TFLOP/s`**.
  - All `K = 2560` shapes under `--policy a4` switch from `BF16_F32ACC` (`74–79 TFLOP/s`) to `NVFP4_F32ACC` (**`>= 300–360+ TFLOP/s`**), and all `K = 5120` shapes under `--routes fast` switch from `BF16_F32ACC` (`63.9–73.4 TFLOP/s`) to `Nvfp4A4Pair` (**`>= 165–195+ useful TFLOP/s` = `330–390+ hardware OMMA TFLOP/s`**).

---

### Task T5 (Priority P0): Eliminate NVFP4 Decode & MTP Verify (`T = 1..64`) Bandwidth Cliffs Across All 15 NVFP4 Shapes (`90%+` `READ_%`)

#### Affected Files & Symbols
- [n34816_k5120.cu](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n34816_k5120.cu) (`select_a16`, `launch_a4`)
- [n16384_k5120.cu](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n16384_k5120.cu) (`select_a16`)
- [n14336_k5120.cu](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n14336_k5120.cu) (`select_a16`)
- [n5120_k17408.cu](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n5120_k17408.cu) (`select_a16`, `launch_a4`)
- [n5120_k6144.cu](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n5120_k6144.cu) (`select_a16`, `launch_a4`)
- [n2560_k6144.cu](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n2560_k6144.cu) (`select_a16`)
- [n2560_k2560.cu](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n2560_k2560.cu) (`select_a16`)
- [n2048_k2560.cu](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n2048_k2560.cu) (`select_a16`)
- [n6144_k2560.cu](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n6144_k2560.cu), [n10240_k2560.cu](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n10240_k2560.cu), [n12800_k2560.cu](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n12800_k2560.cu), [n13952_k2560.cu](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n13952_k2560.cu), [n16384_k2560.cu](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n16384_k2560.cu)

#### Root Cause & Hardware Rationale
- **Evidence**: Live 15-shape sweeps of `infernix_linear_bench --qtype NVFP4 --routes fast` and `--policy a4` at `T = 1..64` reveal concrete sub-90% bandwidth cliffs:
  1. **`[34816, 5120]` at `T = 2` (`--routes fast` / `A16`)**: drops from `426.85 us` (`235.1 GB/s`, `95.57%` `READ_%` at `T = 1`) down to **`498.59 us` (`201.4 GB/s`, `81.88%` `READ_%` at `T = 2`)** before recovering to `434.18 us` (`231.5 GB/s`, `94.09%`) at `T = 4`! Why? In [`n34816_k5120.cu:24`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n34816_k5120.cu#L24), `T == 2` dispatches `Nvfp4SlicedInstance<8, 8, 2>` (`RowTiles = 1`, `2176` CTAs) while `T = 3..16` dispatches `Nvfp4SlicedInstance<8, 8, 2, 2>` (`RowTiles = 2`, `1088` CTAs, `434 us`)!
  2. **`[16384, 5120]` at `T = 2` (`A16`)**: drops to `216.90 us` (`217.9 GB/s`, `88.60%` `READ_%`) vs `T = 1` (`202.59 us`, `94.77%`) and `T = 4` (`208.70 us`, `92.24%`) because [`n16384_k5120.cu:32`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n16384_k5120.cu#L32) still dispatches `nvfp4_linear_a16_simt<Geometry, 2, C2, true>` at `T == 2` and `RowTiles = 1` at `T = 3..16`.
  3. **`[5120, 17408]` at `T = 2..16`**:
     - Under `--routes fast` (`A16`): drops from `230.2 GB/s` (`93.59%` `READ_%` at `T = 1`) down to **`178.1–192.8 GB/s` (`72.40–78.36%` `READ_%`, `262–285 us`)** at `T = 2..16`, whereas `T = 32` runs in **`247.8 us` (`208.1 GB/s`, `84.61%`)**! Why? Because [`n5120_k17408.cu:25-27`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n5120_k17408.cu#L25-L27) uses `RowTiles = 1` (`Nvfp4SlicedInstance<8, 8, 2>`, `320` CTAs) instead of `RowTiles = 2` (`160` CTAs, `32` rows/CTA) or `Nvfp4SlicedInstance<32, 4, 2, 2>`!
     - Under `--policy a4`: `T = 2..4` drops to **`266.0–268.1 us` (`187.3–189.1 GB/s`, `76.15–76.88%` `READ_%`)** because `uses_a4` in [`n5120_k17408.cu:58`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n5120_k17408.cu#L58) only switches to `A4` at `tokens >= 8` (`234.3 us`, `215.5 GB/s`, `87.59%`), leaving `T = 2..7` on the slow `A16` path!
  4. **`[2560, 6144]` & `[5120, 6144]` at `T = 32..64` (`--routes fast`)**:
     - `[2560, 6144]` drops from `216.1 GB/s` (`87.84%` at `T = 16`) down to **`148.1–150.7 GB/s` (`60.20–61.26%` `READ_%`)** at `T = 32..64`.
     - `[5120, 6144]` drops from `216.0 GB/s` (`87.82%` at `T = 16`) down to **`147.7–155.3 GB/s` (`60.06–63.12%` `READ_%`)** at `T = 32..64`.
  5. **`[2560, 2560]` (`12×`/pass) & `[2048, 2560]` (`2×`/pass) at `T = 1..64`**:
     - `[2560, 2560]` sits at `185.0 GB/s` (`75.20%`) at `T = 1`, `199.3–202.7 GB/s` (`81.0–82.4%`) at `T = 2..16`, and `121.3–149.4 GB/s` (`49.3–60.7%`) at `T = 32..64`.
     - `[2048, 2560]` sits at `162.4 GB/s` (`66.03%`) at `T = 1`, `190.0–194.2 GB/s` (`77.2–79.0%`) at `T = 2..16`, and `115.6–139.6 GB/s` (`47.0–56.8%`) at `T = 32..64`.
  6. **All `K = 2560` shapes (`[6144, 2560]`, `[10240, 2560]`, `[12800, 2560]`, `[13952, 2560]`, `[16384, 2560]`) at `T = 32..64`**: drop from `90.2–93.1%` `READ_%` at `T = 1..16` down to `66.1–82.1%` `READ_%` at `T = 32..64` because `Nvfp4A4Pair` admission is currently `T >= 65` (or `T = 32..64` uses `Nvfp4SlicedInstance<32, 4, 1, 2>` with `Stages = 1` instead of `Stages = 2` / `Nvfp4A4Pair` from `T >= 33`).

#### Required Implementation Steps
1. In [`n34816_k5120.cu`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n34816_k5120.cu) and [`n16384_k5120.cu`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n16384_k5120.cu), unify `T = 2..8` onto `Nvfp4SlicedInstance<8, 8, 2, 2>` (`RowTiles = 2`) and `T = 9..16` onto `Nvfp4SlicedInstance<16, 8, 2, 2>`.
2. In [`n5120_k17408.cu`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n5120_k17408.cu), replace the `RowTiles = 1` `T = 2..16` schedules (`262–285 us`) with `RowTiles = 2` (`Nvfp4SlicedInstance<8, 4, 2, 2>` / `Nvfp4SlicedInstance<16, 4, 2, 2>`) and lower `uses_a4` to `tokens >= 2` so both `A16` and `A4` exceed `215–225 GB/s` (`87.5–91.5%` `READ_%`).
3. For `[2560, 6144]`, `[5120, 6144]`, `[6144, 2560]`, `[10240, 2560]`, `[12800, 2560]`, `[13952, 2560]`, and `[16384, 2560]` at `T = 25..64`, replace `Stages = 1` (`Nvfp4SlicedInstance<32, 4, 1, 2>`) with `Stages = 2` (`Nvfp4SlicedInstance<32, 4, 2, 2>`, `Nvfp4SlicedInstance<64, 4, 2, 2>`) and benchmark `Nvfp4A4Pair` at `T = 33..64`.
4. For `[2560, 2560]` and `[2048, 2560]` at `T = 1..32`, tune `Nvfp4A16GemvSchedule` (`BlockRows = 4`, `WarpsPerRow = 2`, `MinBlocks = 4` so `N = 2048/2560` launches `512`/`640` CTAs across 48 SMs instead of `256`/`320`) and `Nvfp4SlicedInstance` schedules.

#### Verification & Acceptance Criteria
```bash
find src/ops/linear/nvfp4 -name '*.cu' -exec touch {} +
ninja -C build-native -j "$(nproc)" infernix-serve infernix_linear_bench infernix_linear_nvfp4_a16_test
./build-native/tests/infernix_linear_nvfp4_a16_test
for nk in "34816 5120" "16384 5120" "5120 17408" "5120 6144" "2560 6144" "2560 2560" "2048 2560" "16384 2560"; do
  read n k <<< "$nk"
  for t in 1 2 4 8 16 32 64; do
    ./build-native/bench/infernix_linear_bench --qtype NVFP4 --routes fast --n $n --k $k --t $t --warmup 2 --repeat 10 | grep "^explicit"
  done
done
ctest --test-dir build-native -R infernix_cuda_stack_check --output-on-failure
```
- **Acceptance**:
  - `[34816, 5120]` at `T = 2` improves from `498.59 us` (`81.88%` `READ_%`) to **`<= 436 us` (`>= 93.5%` `READ_%`)**.
  - `[16384, 5120]` at `T = 2` improves from `216.90 us` (`88.60%` `READ_%`) to **`<= 209 us` (`>= 92.0%` `READ_%`)**.
  - `[5120, 17408]` at `T = 2..16` improves from `262–285 us` (`72.4–78.4%` `READ_%`) to **`<= 235–245 us` (`>= 87.5–91.5%` `READ_%`)**.
  - `[2560, 6144]`, `[5120, 6144]`, and `K = 2560` shapes at `T = 32..64` improve from `60–75%` `READ_%` to **`>= 85–91%` `READ_%`**.

---

### Task T6 (Priority P1): Parallelize and Coalesce NVFP4 Fused `GdnConvSnapshot` Epilogue (`[16384, 2560]` & `[16384, 5120]`) to Eliminate the `-21.2 us/layer` Decode Regression

#### Affected Files & Symbols
- [nvfp4_a16_gemv.cuh](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a16_gemv.cuh) (`nvfp4_a16_gemv_kernel`)
- [nvfp4_a16_sliced_k_mma.cuh](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a16_sliced_k_mma.cuh) (`nvfp4_a16_sliced_k_mma_kernel`)
- [q8_gdn_conv_columns.cuh](file:///opt/etc/Infernix/src/ops/gdn_input_proj/q8/q8_gdn_conv_columns.cuh) (`GdnConvSnapshotColumnsOutput`)
- [n16384_k2560.cu](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n16384_k2560.cu) (`launch_nvfp4_n16384_k2560_conv_snapshot`)
- [nvfp4_gdn_snapshot_decode.cu](file:///opt/etc/Infernix/src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_decode.cu) (`nvfp4_gdn_snapshot_decode_launch`)
- [execution_plan.cpp](file:///opt/etc/Infernix/src/models/qwen4_exp/execution_plan.cpp) (`select_linear_routes`)

#### Root Cause & Hardware Rationale
- **Evidence**: As documented in `docs/gb10/20261030-changelog.md` (`T1` §2 & §5), enabling `LinearRoutes::GdnConvFused` on Qwen4Exp NVFP4 regressed `launch_nvfp4_n16384_k2560_conv_snapshot` to **`120.2 us/call` vs `96.2 + 2.8 = 99.0 us` composed (`-21.2 us/layer` = `-763 us/token` across 36 GDN layers, `-3.5%` E2E decode tok/s)**, forcing `GdnConvFused` to be left off for NVFP4 while enabled for Q8 (`+1.5%` E2E).
- **Exact Root Cause**:
  1. In [`nvfp4_a16_gemv.cuh`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a16_gemv.cuh#L249-L291), each CTA (`kBlockRows = 8`, `kWarpsPerRow = 2`, `16` warps = `512` threads) maps `local_row = 0..7` through `row = nvfp4_perm_row(m128_base + local_row)`, where `nvfp4_perm_row(r) = (r & 3) * 32 + ((r >> 2) & 7) * 4 + (r >> 5)`. Consequently, the `8` rows in a CTA are spaced **32 channels (`64 bytes`) apart** in `conv_weight`, `conv_states`, `query/key/value`, and `snapshot`!
  2. Worse still, in [`nvfp4_a16_gemv.cuh:284-290`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a16_gemv.cuh#L284-L290), after `__syncthreads()`, `linear_finish_row` is executed inside `if (warp_in_row == 0 && lane == 0)` — meaning **only `1` lane (`lane 0`) in each of `8` separate warps** executes [`GdnConvSnapshotColumnsOutput::store_row`](file:///opt/etc/Infernix/src/ops/gdn_input_proj/q8/q8_gdn_conv_columns.cuh#L38-L110), issuing **7 uncoalesced single-thread global loads (`conv_weight[0..3]`, `initial[0..2]` at `20 KiB` stride) and 4 single-thread global stores** after the GEMV barrier!
  3. By contrast, if `warp 0` (`lane = 0..7`) reduces and finishes all `kBlockRows = 8` rows of the CTA (or prefetches `conv_weight` and `initial` history taps in parallel across lanes *before* the GEMV loop / during the final reduction), or if `nvfp4_a16_gemv_kernel` prefetches the 7 `bf16` values (`w0..w3`, `h0..h2`) per `row_in_cta` in `warp_in_row == 0, lane == 0..6` at the start of the kernel so they are already in registers when `partials` are summed at the end of the kernel, the entire `21.2 us` post-barrier DRAM latency disappears!

#### Required Implementation Steps
1. Add an optional epilogue prefetch hook (or specialize `GdnConvSnapshotColumnsOutput` / `GdnConvOutput` in [`nvfp4_a16_gemv.cuh`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a16_gemv.cuh) and [`nvfp4_a16_sliced_k_mma.cuh`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a16_sliced_k_mma.cuh)) so that for `row < kGdnChannels`, threads in the CTA issue the `conv_weight` (`4` taps) and `conv_states` (`3` taps per active sequence) global loads **at kernel entry** (overlapping their ~400-cycle LPDDR5X latency with the `2560`-element or `5120`-element GEMV/sliced-K mainloop!) instead of issuing them serially after `__syncthreads()` in the epilogue.
2. Verify via targeted microbenchmark (`infernix_gdn_input_proj_bench` / `launch_nvfp4_n16384_k2560_conv_snapshot` and `nvfp4_gdn_snapshot_decode_launch`) that the fused NVFP4 kernel runs in **`<= 96.5 us`** on `[16384, 2560]` (beating the `96.2 + 2.8 = 99.0 us` composed route and saving `36` kernel launches per token).
3. Enable `LinearRoutes::GdnConvFused` for NVFP4 in [`src/models/qwen4_exp/execution_plan.cpp`](file:///opt/etc/Infernix/src/models/qwen4_exp/execution_plan.cpp#L47-L56).

#### Verification & Acceptance Criteria
```bash
find src/ops/linear/nvfp4 src/ops/gdn_input_proj -name '*.cu' -exec touch {} +
ninja -C build-native -j "$(nproc)" infernix-serve infernix_gdn_input_proj_bench infernix_ops_gdn_input_proj_test infernix_qwen4_exp_model_test
ctest --test-dir build-native -R "infernix_ops_gdn_input_proj_test|infernix_qwen4_exp_model_test|infernix_cuda_stack_check" --output-on-failure
./build-native/bench/infernix_gdn_input_proj_bench --format nvfp4 --tokens 1,2,4,8 --cache cold --warmup 2 --repeat 10
```
- **Acceptance**:
  - Fused `launch_nvfp4_n16384_k2560_conv_snapshot` latency at `T = 1` drops from `120.2 us` to **`<= 96.5 us` (`>= 221 GB/s`, `>= 90.0%` `READ_%`)**, beating composed `gdn_input_proj + causal_conv1d_silu` (`99.0 us`).
  - `infernix_ops_gdn_input_proj_test`, `infernix_qwen4_exp_model_test`, and `infernix_cuda_stack_check` pass 100%.

---

### Task T7 (Priority P1): Retune Fused `nvfp4_linear_add`, `nvfp4_gdn_input_proj`, `nvfp4_attn_input_proj`, and `nvfp4_linear_swiglu` Decode/Prefill Schedules on GB10

#### Affected Files & Symbols
- [nvfp4_linear_add_decode.cu](file:///opt/etc/Infernix/src/ops/linear_add/nvfp4/nvfp4_linear_add_decode.cu) (`nvfp4_linear_add_decode_launch`)
- [nvfp4_linear_add_a16.cu](file:///opt/etc/Infernix/src/ops/linear_add/nvfp4/nvfp4_linear_add_a16.cu) (`nvfp4_linear_add_a16_launch`, `launch_matrix`)
- [nvfp4_linear_add_a4_tma.cu](file:///opt/etc/Infernix/src/ops/linear_add/nvfp4/nvfp4_linear_add_a4_tma.cu) (`launch_nvfp4_a4_tma_linear_add`)
- [nvfp4_linear_add_a4.cu](file:///opt/etc/Infernix/src/ops/linear_add/nvfp4/nvfp4_linear_add_a4.cu) (`nvfp4_linear_add_a4_launch`)
- [nvfp4_gdn_input_plan.cpp](file:///opt/etc/Infernix/src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.cpp) (`resolve_route`)
- [nvfp4_gdn_input_a16.cu](file:///opt/etc/Infernix/src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_a16.cu) (`nvfp4_gdn_input_a16_launch`)
- [nvfp4_attn_input_plan.cpp](file:///opt/etc/Infernix/src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.cpp) (`resolve_route`)
- [nvfp4_attn_input_a16.cu](file:///opt/etc/Infernix/src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_a16.cu) (`nvfp4_attn_input_a16_launch`)

#### Root Cause & Hardware Rationale
- **Evidence**: Live benchmarks of the fused NVFP4 ops (`infernix_nvfp4_linear_add_bench`, `infernix_gdn_input_proj_bench`, `infernix_attn_input_proj_bench`, `infernix_nvfp4_linear_swiglu_bench`) reveal major performance losses compared to the standalone NVFP4 linear kernels:
  1. **`nvfp4_linear_add` at `N = 5120, K = 6144, T = 1` runs at `98.30 us` (`180.3 GB/s`, `73.29%` `READ_%`) vs standalone `[5120, 6144]` at `79.62 us` (`222.6 GB/s`, `90.47%` `READ_%`)**:
     In [`nvfp4_linear_add_decode.cu:14-15`](file:///opt/etc/Infernix/src/ops/linear_add/nvfp4/nvfp4_linear_add_decode.cu#L14-L15), `N5120K6144` uses `Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>` (`WarpsPerRow = 2`, `StagedRaw`), whereas [`n5120_k6144.cu:13`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n5120_k6144.cu#L13) uses `Nvfp4A16GemvSchedule<8, 1, 16, 4, Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 2>` (`WarpsPerRow = 1`, `Direct`), costing **`+18.7 us` (`+23.5%` slower)** per call!
  2. **`nvfp4_linear_add` at `T = 2..5` (`K = 6144` and `K = 17408`)**:
     [`nvfp4_linear_add_a16.cu:232`](file:///opt/etc/Infernix/src/ops/linear_add/nvfp4/nvfp4_linear_add_a16.cu#L232) routes `T = 2` (`K = 6144`) and `T = 2..5` (`K = 17408`) to `nvfp4_linear_add_small_t_launch` (SIMT `240.6 us` / `208.8 GB/s` at `K = 17408, T = 2`) instead of `Nvfp4SlicedInstance<8, 8, 2, 2>`, and all `Nvfp4SlicedInstance` schedules in `nvfp4_linear_add_a16.cu:18-65` use `RowTiles = 1` instead of `RowTiles = 2`!
  3. **`nvfp4_linear_add` TMA Prefill at `T = 512..4096` (`217.17 TFLOP/s` at `K = 6144`, `227.59 TFLOP/s` at `K = 17408`)**:
     [`nvfp4_linear_add_a4_tma.cu:8-9`](file:///opt/etc/Infernix/src/ops/linear_add/nvfp4/nvfp4_linear_add_a4_tma.cu#L8-L9) hardcodes `Nvfp4A4TmaMmaSchedule<256, 3, 1, ...>` (`BM = 256, GroupM = 1`, `20` M-blocks on 48 SMs) instead of `BM = 128, GroupM = 8` (`Tiled128`, `40` M-blocks), losing `~100+ TFLOP/s` at prefill!
  4. **`nvfp4_gdn_input_proj` (`[16384, 5120]`) under `--nvfp4-policy a4` at `T = 1..16`**:
     [`nvfp4_gdn_input_plan.cpp:27`](file:///opt/etc/Infernix/src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.cpp#L27) unconditionally returns `Nvfp4GdnInputRoute::A4` even at `T = 1` (`217.95 us`, `216.7 GB/s`, `88.1%` `READ_%`), whereas `Nvfp4GdnInputRoute::A16` at `T = 1` runs in **`202.56 us` (`233.2 GB/s`, `94.8%` `READ_%`)** and at `T = 4..16` runs in **`209.7–214.7 us` (`223.0–225.8 GB/s`)**! And in `nvfp4_gdn_input_a16.cu` and `nvfp4_attn_input_a16.cu`, `T = 2` still dispatches the slow SIMT `small_t` kernel (`224.1 us` on GDN, `191.3 us` on Attn) instead of `Nvfp4SlicedInstance<8, 8, 2, 2>`.

#### Required Implementation Steps
1. In [`nvfp4_linear_add_decode.cu`](file:///opt/etc/Infernix/src/ops/linear_add/nvfp4/nvfp4_linear_add_decode.cu), specialize `N5120K6144` to `Nvfp4A16GemvSchedule<8, 1, 16, 4, Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 2>`.
2. In [`nvfp4_linear_add_a16.cu`](file:///opt/etc/Infernix/src/ops/linear_add/nvfp4/nvfp4_linear_add_a16.cu), route `T = 2..8` directly to `Nvfp4SlicedInstance<8, 8, 2, 2>` (and `RowTiles = 2` for `T = 9..32`), and in [`nvfp4_linear_add_a4_tma.cu`](file:///opt/etc/Infernix/src/ops/linear_add/nvfp4/nvfp4_linear_add_a4_tma.cu) + [`nvfp4_linear_add_a4.cu`](file:///opt/etc/Infernix/src/ops/linear_add/nvfp4/nvfp4_linear_add_a4.cu), use `Tiled128` with `Nvfp4A4TmaMmaSchedule<128, 3, 8, CU_TENSOR_MAP_L2_PROMOTION_L2_128B>`.
3. In [`nvfp4_gdn_input_plan.cpp`](file:///opt/etc/Infernix/src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.cpp), change `resolve_route` to keep `T = 1..16` (or `T = 1..3`) on `Nvfp4GdnInputRoute::A16`, and in [`nvfp4_gdn_input_a16.cu`](file:///opt/etc/Infernix/src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_a16.cu) and [`nvfp4_attn_input_a16.cu`](file:///opt/etc/Infernix/src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_a16.cu), replace the `T = 2` SIMT `small_t` call with `Nvfp4SlicedInstance<8, 8, 2, 2>`.

#### Verification & Acceptance Criteria
```bash
find src/ops/linear_add/nvfp4 src/ops/gdn_input_proj/nvfp4 src/ops/attn_input_proj/nvfp4 -name '*.cu' -exec touch {} +
ninja -C build-native -j "$(nproc)" infernix-serve infernix_nvfp4_linear_add_bench infernix_gdn_input_proj_bench infernix_attn_input_proj_bench infernix_ops_linear_add_test infernix_ops_gdn_input_proj_test infernix_ops_attention_test
ctest --test-dir build-native -R "infernix_ops_linear_add_test|infernix_ops_gdn_input_proj_test|infernix_ops_attention_test|infernix_cuda_stack_check" --output-on-failure
./build-native/bench/infernix_nvfp4_linear_add_bench --n 5120 --k 6144 --policy a4 --t-sweep 1,2,4,8,16,32,64,512,4096 --warmup 2 --repeat 10
./build-native/bench/infernix_nvfp4_linear_add_bench --n 5120 --k 17408 --policy a4 --t-sweep 1,2,4,8,16,32,64,512,4096 --warmup 2 --repeat 10
./build-native/bench/infernix_gdn_input_proj_bench --format nvfp4 --nvfp4-policy a4 --tokens 1,2,4,8,16,32,64,512,4096 --cache cold --warmup 2 --repeat 10
./build-native/bench/infernix_attn_input_proj_bench --format nvfp4 --nvfp4-policy a4 --tokens 1,2,4,8,16,32,64,512,4096 --cache cold --warmup 2 --repeat 10
```
- **Acceptance**:
  - `nvfp4_linear_add` `N=5120, K=6144` at `T=1` improves from `98.30 us` (`180.3 GB/s`) to **`<= 82.0 us` (`>= 216 GB/s`)**, and at `T=4096` improves from `217.17 TFLOP/s` to **`>= 310+ TFLOP/s`**.
  - `nvfp4_linear_add` `N=5120, K=17408` at `T=2..4` improves from `236.5–240.6 us` (`208.8–212.9 GB/s`) to **`<= 226 us` (`>= 223 GB/s`, `>= 90.6%` `READ_%`)**, and at `T=4096` improves from `227.59 TFLOP/s` to **`>= 320+ TFLOP/s`**.
  - `nvfp4_gdn_input_proj` (`policy=a4`) at `T=1..2` improves from `217.95–219.01 us` (`215.8–216.7 GB/s`) to **`<= 203–209 us` (`>= 226–233 GB/s`, `>= 92–94.8%` `READ_%`)**, and `nvfp4_attn_input_proj` at `T=2` improves from `191.33 us` (`216.2 GB/s`) to **`<= 184 us` (`>= 225 GB/s`, `>= 91.5%` `READ_%`)**.

---

### Task T8 (Priority P1): Upgrade Fused Q8 Projections (`q8_linear_add`, `q8_attn_input_proj`, `q8_gdn_input_proj`, `q8_linear_pair`) to `90%+` GB10 Bandwidth and Compute Ceilings

#### Affected Files & Symbols
- [q8_linear_add_plan.cpp](file:///opt/etc/Infernix/src/ops/linear_add/q8/q8_linear_add_plan.cpp) (`kK4096Routes`, `kK6144Routes`, `kN5120Routes`)
- [q8_linear_add_gemm_mma.cu](file:///opt/etc/Infernix/src/ops/linear_add/q8/q8_linear_add_gemm_mma.cu)
- [q8_attn_input_plan.cpp](file:///opt/etc/Infernix/src/ops/attn_input_proj/q8/q8_attn_input_plan.cpp) (`kTargetRoutes`, `kCompanionRoutes`, `kDFlash2Routes`)
- [q8_attn_input_gemm_mma.cu](file:///opt/etc/Infernix/src/ops/attn_input_proj/q8/q8_attn_input_gemm_mma.cu)
- [q8_gdn_input_plan.cpp](file:///opt/etc/Infernix/src/ops/gdn_input_proj/q8/q8_gdn_input_plan.cpp) (`kRoutes`)
- [q8_gdn_input_gemm_mma.cu](file:///opt/etc/Infernix/src/ops/gdn_input_proj/q8/q8_gdn_input_gemm_mma.cu)
- [q8_pair_plan.cpp](file:///opt/etc/Infernix/src/ops/linear_pair/q8/q8_pair_plan.cpp) (`kK5120Routes`, `kK2048Routes`)

#### Root Cause & Hardware Rationale
- **Evidence**: Live benchmarks of the fused Q8 operators (`infernix_q8_linear_add_bench`, `infernix_attn_input_proj_bench`, `infernix_gdn_input_proj_bench`) reveal substantial gaps below GB10's `90%` bandwidth (`221.4 GB/s`) and `90%` compute (`87.3 TFLOP/s`) ceilings:
  1. **`q8_linear_add` (`N = 2048, K = 4096` and `K = 6144`)**:
     - At `K = 4096, T = 1`, [`q8_linear_add_plan.cpp:30`](file:///opt/etc/Infernix/src/ops/linear_add/q8/q8_linear_add_plan.cpp#L30) dispatches `SimtR8C4` (`45.06 us`, **`198.2 GB/s` = `80.57%` `READ_%`**), whereas `DecodeR16` / `q8_a16_gemv_kernel` achieves `210–215+ GB/s`.
     - At `T = 512..4096`, `q8_linear_add` reaches only **`53.07–55.02 TFLOP/s` (`54.7–56.7%` of `97.0 TF`)** at `K = 4096` and **`53.77–65.11 TFLOP/s` (`55.4–67.1%`)** at `K = 6144` because [`q8_linear_add_gemm_mma.cu`](file:///opt/etc/Infernix/src/ops/linear_add/q8/q8_linear_add_gemm_mma.cu) uses ungrouped `MmaR64C128` (`BM = 64, BN = 128, GroupM = 1`) instead of `128x128` `Grouped, 8` (`MmaR128x64T128K64A1`)!
  2. **`q8_attn_input_proj` (`q8-qgkv` `[9216, 2048]`, `q8-dflash2-qkv` `[6144, 5120]`) and `q8_gdn_input_proj` (`[12288, 2048]`)**:
     - At `T = 64`: `q8-qgkv` drops to **`194.7 GB/s` (`79.1%`)**, `q8-dflash2-qkv` drops to **`189.3 GB/s` (`76.9%`)**, and `q8_gdn_input_proj` drops to **`199.5 GB/s` (`81.1%`)** (`SplitKMmaDirect` stays active up to `T = 64` or `T = 96`).
     - At `T = 512..4096`: `q8-qgkv` reaches **`73.20–76.54 TFLOP/s` (`75.5–78.9%` of `97.0 TF`)**, `q8-dflash2-qkv` reaches **`76.02 TFLOP/s` (`78.4%`)**, and `q8_gdn_input_proj` reaches **`74.48–77.53 TFLOP/s` (`76.8–79.9%`)** because all three dispatch `MmaR64C128` (`BM = 64`) instead of `128x128` `Grouped, 8` (`MmaR128C128` / `MmaR128x64T128K64A1`)!
  3. **`q8_linear_pair` (`kK5120Routes` in [`q8_pair_plan.cpp:25-29`](file:///opt/etc/Infernix/src/ops/linear_pair/q8/q8_pair_plan.cpp#L25-L29))**:
     - `kK5120Routes` still uses the RTX 5090 3-entry table (`TwoSimtR8C4` for `1..20`, `DualMmaR32C64` for `21..256`, `DualMmaR32C128` for `257+`), never using `DualDecodeR4`, `DualSplitKMmaExactT`, or `ConcatMmaR64C128` / `ConcatMmaR128C128`!

#### Required Implementation Steps
1. Add `MmaR128C128` (`Q8A16MmaSchedule<128, 128, 64, 64, 32, 2, 2, ..., Q8MmaBlockOrder::Grouped, 8>`) to `q8_linear_add_gemm_mma.cu`, `q8_attn_input_gemm_mma.cu`, `q8_gdn_input_gemm_mma.cu`, and `q8_pair_gemm_mma.cu` (which automatically inherits the double-buffered `q8_a16_mma_kernel` from Task T1).
2. In [`q8_linear_add_plan.cpp`](file:///opt/etc/Infernix/src/ops/linear_add/q8/q8_linear_add_plan.cpp), wire `DecodeR16` for `K = 4096, T = 1` and `MmaR128C128` for `T >= 256`.
3. In [`q8_gdn_input_plan.cpp`](file:///opt/etc/Infernix/src/ops/gdn_input_proj/q8/q8_gdn_input_plan.cpp) and [`q8_attn_input_plan.cpp`](file:///opt/etc/Infernix/src/ops/attn_input_proj/q8/q8_attn_input_plan.cpp), wire `MmaR128C128` for prefill (`T >= 128`) and retune the `T = 48..96` transition.
4. In [`q8_pair_plan.cpp`](file:///opt/etc/Infernix/src/ops/linear_pair/q8/q8_pair_plan.cpp), replace `kK5120Routes` (`TwoSimtR8C4` at `1..20`, `DualMmaR32C128` at `257+`) with GB10-tuned decode, split-K (`2..32`), and wide MMA (`64x128` / `128x128`) schedules.

#### Verification & Acceptance Criteria
```bash
find src/ops/linear_add/q8 src/ops/attn_input_proj/q8 src/ops/gdn_input_proj/q8 src/ops/linear_pair/q8 -name '*.cu' -exec touch {} +
ninja -C build-native -j "$(nproc)" infernix-serve infernix_q8_linear_add_bench infernix_attn_input_proj_bench infernix_gdn_input_proj_bench infernix_ops_linear_add_test infernix_ops_attention_test infernix_ops_gdn_input_proj_test
ctest --test-dir build-native -R "infernix_ops_linear_add_test|infernix_ops_attention_test|infernix_ops_gdn_input_proj_test|infernix_cuda_stack_check" --output-on-failure
./build-native/bench/infernix_q8_linear_add_bench --k 4096 --production-only --t-sweep 1,2,4,8,16,32,64,512,4096 --warmup 2 --repeat 10
./build-native/bench/infernix_q8_linear_add_bench --k 6144 --production-only --t-sweep 1,2,4,8,16,32,64,512,4096 --warmup 2 --repeat 10
./build-native/bench/infernix_attn_input_proj_bench --format q8-qgkv --tokens 1,2,4,8,16,32,64,512,4096 --cache cold --warmup 2 --repeat 10
./build-native/bench/infernix_gdn_input_proj_bench --format q8 --tokens 1,2,4,8,16,32,64,512,4096 --cache cold --warmup 2 --repeat 10
```
- **Acceptance**:
  - `q8_linear_add` at `T = 4096` improves from `55.02 TFLOP/s` (`K = 4096`) and `65.11 TFLOP/s` (`K = 6144`) to **`>= 80–86+ TFLOP/s`**, and `K = 4096, T = 1` improves from `198.2 GB/s` to **`>= 210 GB/s`**.
  - `q8_attn_input_proj` and `q8_gdn_input_proj` at `T = 4096` improve from `76.5–77.5 TFLOP/s` to **`>= 86–90+ TFLOP/s` (`>= 90%` of `97.0 TF` `BF16_F32ACC` ref)**.

---

### Task T9 (Priority P1): Implement `Q8` `W8A8` (`IMMA.16832.S8`) Fast-Path Prefill Route to Unlock GB10's `247.4 TOP/s` INT8 Tensor Core Pipe

#### Affected Files & Symbols
- [q8_dispatch.cpp](file:///opt/etc/Infernix/src/ops/linear/q8/q8_dispatch.cpp) (`q8_linear_dispatch`)
- [q8_shapes.h](file:///opt/etc/Infernix/src/ops/linear/q8/q8_shapes.h) (`Q8LinearShape`)
- [ggml_a8.h](file:///opt/etc/Infernix/src/ops/ggml/ggml_a8.h) / [ggml_a8_mma.cuh](file:///opt/etc/Infernix/src/ops/ggml/ggml_a8_mma.cuh)
- `src/ops/linear/q8/q8_a8_mma.cuh` (new header or extension of `q8_a16_mma.cuh`)
- [types.h](file:///opt/etc/Infernix/include/infernix/types.h) (`LinearRoutes`)

#### Root Cause & Hardware Rationale
- **Evidence**: `ckt env` proves that on NVIDIA GB10 (`sm_121a`), the `bf16` / `fp16_f32acc` Tensor Core pipe (`HMMA.16816`) has a hardware ceiling of **`123.60–123.74 TFLOP/s`** (`97.0 TFLOP/s` dense reference), whereas the `int8` Tensor Core pipe (`IMMA.16832.S8`) has a measured hardware ceiling of **`247.43 TOP/s` (`2.0×` HMMA throughput!)**.
- Currently, `Q8_G32_FP16` weights are stored as packed 32-element blocks of signed `int8_t` codes (`codes`) plus one `fp16` scale per 32 elements (`scales`). However, `q8_a16_mma_kernel` dequantizes every `int8_t` weight in shared memory to `bfloat16` (`2×` SMEM capacity, `2×` `ldmatrix` bandwidth, plus SIMT dequantization overhead) to feed `HMMA.16816` (`123.7 TFLOP/s` ceiling).
- By quantizing the `bfloat16` input activations `x` dynamically per 32-element K-group to `int8_t` + `fp16`/`fp32` scale (reusing the exact `a8_quantize_slice` / `GgmlA8` 32-element group quantizer already in [`src/ops/ggml/ggml_a8.h`](file:///opt/etc/Infernix/src/ops/ggml/ggml_a8.h)), each 32-element K-group (`G32`) of `W` (`s8`) and `X` (`s8`) maps **1:1 onto a single `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` (`IMMA.16832.S8`)** instruction! After each `k = 32` step, the `int32` tile fragment is scaled by `scale_w * scale_x` into the `float` accumulator (`acc += float(imma_sum) * (scale_w * scale_x)`).
- This eliminates `int8 -> bf16` shared-memory expansion (`2×` less weight SMEM), cuts `ldmatrix` traffic in half, and runs on the **`247.43 TOP/s` `IMMA.16832.S8` pipe**, allowing Q8 prefill (`T >= 128`) under `--routes fast` / `AllowA8` to reach **`135–180+ TOP/s` (`140–185%` of the BF16 ceiling)**!

#### Required Implementation Steps
1. Implement `q8_a8_mma_kernel` (`128x128x64`, `2` `IMMA.16832.S8` steps per `BK = 64` stage, scaling the `s32` fragment per `k = 32` group by `scale_w[m][g] * scale_x[n][g]` into FP32 accumulators) and the lightweight per-32-element activation quantizer (`q8_a8_quantize_x_kernel`).
2. Wire the route behind `LinearPolicy::AllowA8` / `LinearRoutes` fast path for `T >= 128` in [`q8_dispatch.cpp`](file:///opt/etc/Infernix/src/ops/linear/q8/q8_dispatch.cpp) while preserving the exact `A16` route for `--reproducible` / `A16Only`.
3. Qualify numerical accuracy against the FP32 oracle and `Q8_G32_FP16` acceptance gates in `infernix_ops_linear_test`.

#### Verification & Acceptance Criteria
```bash
find src/ops/linear/q8 -name '*.cu' -exec touch {} +
ninja -C build-native -j "$(nproc)" infernix-serve infernix_linear_bench infernix_linear_q8_a16_test infernix_ops_linear_test
./build-native/tests/infernix_linear_q8_a16_test
ctest --test-dir build-native -R "infernix_ops_linear_test|infernix_cuda_stack_check" --output-on-failure
for nk in "16384 2560" "17408 5120" "34816 5120" "14336 5120" "25600 5120"; do
  read n k <<< "$nk"
  ./build-native/bench/infernix_linear_bench --qtype Q8_G32_FP16 --policy a8 --n $n --k $k --t 512 --warmup 2 --repeat 10 | grep "^explicit"
  ./build-native/bench/infernix_linear_bench --qtype Q8_G32_FP16 --policy a8 --n $n --k $k --t 4096 --warmup 2 --repeat 10 | grep "^explicit"
done
```
- **Acceptance**:
  - `Q8_G32_FP16` `--policy a8` / fast prefill at `T = 4096` achieves **`>= 135–175+ TFLOP/s` (`1.4×–1.8×` the `97.0 TFLOP/s` `BF16_F32ACC` ceiling, and `>= 90%` of effective `IMMA.16832.S8` + group-scale pipeline throughput)**.
  - All `infernix_linear_q8_a16_test`, `infernix_ops_linear_test`, and `infernix_cuda_stack_check` tests pass 100%.

---

### Task T10 (Priority P1): Optimize `offloaded_sparse_moe` NVFP4 (`W4A16` / `W4A8`) and Q8/GGUF Narrow & Wide Expert Kernels for GB10 `90%+` UMA Bandwidth & Tensor Core Utilization

#### Affected Files & Symbols
- [narrow_expert.cu](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/narrow_expert.cu) (`gate_up_kernel`, `down_kernel`)
- [nvfp4_expert_a8.cuh](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/nvfp4_expert_a8.cuh) (`gate_up_kernel_nvfp4_a8`, `down_kernel_nvfp4_a8`)
- [ggml_expert.cuh](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/ggml_expert.cuh) / [ggml_expert_a8.cuh](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/ggml_expert_a8.cuh)
- [wide_expert_a16.cuh](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert_a16.cuh) (`wide_gate_up_kernel_a16`, `wide_down_kernel_a16`)
- [wide_expert_a8.cuh](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert_a8.cuh) (`wide_gate_up_kernel_a8`, `wide_down_kernel_a8`)

#### Root Cause & Hardware Rationale
- **Evidence**: In `Qwen4ExpForCausalLM` (Qwen3.8-Flash-Next, `49` MoE layers × `10` active routed experts + `1` shared expert per token, `H = 2560, I = 640`), the routed MoE expert kernels (`offloaded_sparse_moe`, resident in GB10's 128 GiB unified LPDDR5X memory) account for **`~62%` of total decode latency and `~55%` of total prefill latency**.
- Inspection of [`nvfp4_expert_a8.cuh`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/nvfp4_expert_a8.cuh#L130-L240), [`narrow_expert.cu`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/narrow_expert.cu), [`wide_expert_a16.cuh`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert_a16.cuh), and [`wide_expert_a8.cuh`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert_a8.cuh) reveals three GB10-specific bottlenecks:
  1. **Single-Staged Weight Copy Before `pass` Loop in `narrow_expert.cu` & `nvfp4_expert_a8.cuh`**:
     For `H = 2560, I = 640`, `gate_up` launches `80` row-group CTAs × `10` experts = `800` CTAs (`16.7` waves on 48 SMs), and `down` launches `40` CTAs × `10` experts = `400` CTAs (`8.3` waves on 48 SMs). Each CTA stages its `18 KiB` (`gate_up`) or `18 KiB` (`down`) weight slice via `stage_async` -> `stage_wait()` -> `__syncthreads()`, and only *after* `stage_wait()` does it load `x_rows` (`nvfp4_expert_a8.cuh:154-168`). Issuing the first pass's `x_rows` / `scales` loads *before* `stage_wait()` overlaps activation reads with weight DMA across all `16.7` waves.
  2. **Wide Expert Prefill (`wide_expert_a16.cuh` & `wide_expert_a8.cuh`) Wave Quantization & SMEM Staging on 48 SMs**:
     At prefill (`T = 512..4096`), `512` experts receive an average of `10..80` tokens per expert. `wide_expert_a16.cuh` and `wide_expert_a8.cuh` use tile schedules tuned for RTX 5090 (170 SMs), leaving GB10's 48 SMs under-occupied on medium-token expert buckets (`16..64` tokens/expert) and serializing dequantization in `wide_expert_a16.cuh`.
  3. **End-to-End Verification Across Plain Decode, Speculative (`--dflash2`), and Prefill (`T = 512..4096`)**:
     Measuring `infernix_offloaded_moe_decode_bench` and `infernix_offloaded_moe_wide_bench` alongside end-to-end `infernix-bench` (`Qwen3.8-Flash-Next` NVFP4 and Q8) verifies that both standalone linears and MoE experts reach `90%+` of GB10's LPDDR5X bandwidth (`>= 221.4 GB/s`) and Tensor Core ceilings.

#### Required Implementation Steps
1. In [`nvfp4_expert_a8.cuh`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/nvfp4_expert_a8.cuh) and [`narrow_expert.cu`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/narrow_expert.cu), overlap the initial activation/scale loads (`xs[0..n-1]`, `ds[0..n-1]`) with `stage_async(...)` before `stage_wait()`, and verify register count stays within `__launch_bounds__(128, 3)` (`<= 64` regs/thread, zero spills).
2. In [`wide_expert_a16.cuh`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert_a16.cuh) and [`wide_expert_a8.cuh`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert_a8.cuh), overlap next-K-tile `cp.async` staging with in-SMEM dequantization / `IMMA.16832.S8` / `HMMA.16816` inner loops so wide expert prefill reaches `>= 90%` of the active pipe ceiling.
3. Run `infernix_offloaded_moe_decode_bench` and `infernix_offloaded_moe_wide_bench` across NVFP4 (`W4A16`, `W4A8`) and Q8/GGUF (`W8A16`, `W8A8`) to confirm `>= 90%` bandwidth in decode (`>= 221.4 GB/s`) and `>= 90%` compute efficiency in wide prefill.

#### Verification & Acceptance Criteria
```bash
find src/ops/offloaded_sparse_moe -name '*.cu' -exec touch {} +
ninja -C build-native -j "$(nproc)" infernix-serve infernix_offloaded_moe_decode_bench infernix_offloaded_moe_wide_bench infernix_ops_moe_test
ctest --test-dir build-native -R "infernix_ops_moe_test|infernix_cuda_stack_check" --output-on-failure
./build-native/bench/infernix_offloaded_moe_decode_bench
./build-native/bench/infernix_offloaded_moe_wide_bench
```
- **Acceptance**:
  - `infernix_offloaded_moe_decode_bench` achieves **`>= 221.4 GB/s` (`>= 90.0%` of `246.0 GB/s` sustained read reference)** on resident NVFP4 and Q8/GGUF narrow decode passes.
  - `infernix_offloaded_moe_wide_bench` improves wide expert prefill throughput by **`>= 10–20%`**, reaching **`>= 90%`** of the active Tensor Core pipe ceiling, with zero stack spills (`infernix_cuda_stack_check` passing) and 100% `infernix_ops_moe_test` pass rate.
