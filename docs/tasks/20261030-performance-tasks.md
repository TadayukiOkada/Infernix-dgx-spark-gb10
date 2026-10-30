# GB10 (`sm_121a`) NVFP4 Inference Performance Task Roadmap (2026-10-30)

**Target Hardware**: NVIDIA GB10 (`sm_121a`, 48 SMs, 99 KiB SMEM/SM, 65,536 32-bit registers/SM, 24 MiB L2, 128 GiB Unified LPDDR5X UMA at 273 GB/s, 20-core Grace ARMv9.2 CPU)
**Baseline Commit**: `6f63ebfc4` (`origin/gb10` post-`20261029-changelog.md` T1–T12 completion)
**Primary Target Artifact**: `/opt/etc/models/Qwen3.8-Flash-Next-F4-Local-NVPF4-Dense/Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix` (`Qwen4ExpForCausalLM`, `DenseKernels::Nvfp4` + `Q4_G32_FP16` routed experts + `Q8_G32_FP16` shared experts / hyper-connection / MTP / LM head)
**Companion Speculative Head**: `/opt/etc/models/Qwen3.8-Flash-Next-DFlash2-Head/Qwen3.8-Flash-Next-DFlash2-Infernix.infernix` (`qwen4_exp_dflash2`, 5 layers, 16 draft columns)

---

## Executive Summary & Architectural Motivation

In the `20261029-changelog.md` campaign (`T1`–`T12`), GB10 (`sm_121a`) optimizations established key hardware invariants and surfaced high-leverage gaps specific to NVFP4 model inference (`Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix`):

1. **Unwired NVFP4 Fast Paths in `Qwen4Exp` (`T1`, P0)**:
   - Inspection of [`src/models/qwen4_exp/execution/forward.cpp`](file:///opt/etc/Infernix/src/models/qwen4_exp/execution/forward.cpp#L631-L649) and [`src/models/qwen4_exp/program/fast_paths.cpp`](file:///opt/etc/Infernix/src/models/qwen4_exp/program/fast_paths.cpp#L397-L660) reveals that three production fast paths are already implemented at the kernel/op layer or qualified for `DenseKernels::Q8`, yet are disabled for `DenseKernels::Nvfp4`:
     1. **`FastPath::GdnConvFused`**: [`launch_nvfp4_n16384_k2560_conv_snapshot`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n16384_k2560.cu#L140-L154) and [`ops::gdn_input_proj_conv_snapshot`](file:///opt/etc/Infernix/src/ops/wrapper/gdn_input_proj.cpp#L1280-L1331) already support `QType::NVFP4` `[16384, 2560]`, fusing the causal `conv1d + SiLU` snapshot directly into the GDN input projection epilogue. However, [`Forward::gdn`](file:///opt/etc/Infernix/src/models/qwen4_exp/execution/forward.cpp#L632) gates `fused_conv` on `p.projection.weight.qtype == QType::Q8_G32_FP16`, and [`fast_path_applies(FastPath::GdnConvFused)`](file:///opt/etc/Infernix/src/models/qwen4_exp/program/fast_paths.cpp#L659) only accepts `DenseKernels::Q8`. Every `NVFP4-W4Dense` decode step therefore launches `36` separate `causal_conv1d_silu_snapshot_split` kernels and round-trips `[10240, B]` BF16 (`20.48 KB * B` per GDN layer) through unified LPDDR5X memory.
     2. **`FastPath::DFlash2Pairs`**: In `20261029-changelog.md` (`T7` §2 & §5), `NVFP4-W4Dense` with `--spec dflash2` achieved **`114.8 tok/s`** at `c = 1` (`+25.2%` faster than MTP's `91.7 tok/s`), but collapsed to **`35.6 tok/s`** at `c = 2` (`-33.8%` behind MTP's `53.8 tok/s`) because `kQualified` in [`fast_paths.cpp`](file:///opt/etc/Infernix/src/models/qwen4_exp/program/fast_paths.cpp#L397-L445) omits `{DenseKernels::Nvfp4, true/false, kv}` for `FastPath::DFlash2Pairs` (`spec_rounds = 6 / 723` vs `Dense8`'s `68.5 tok/s`, `+27.3%` over MTP once pair speculation was enabled).
     3. **`FastPath::Q8SlicedK` on hybrid NVFP4 artifacts**: `Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix` contains **`303` `Q8_G32_FP16` weight tensors** executed on every forward pass (`49x` shared-expert `gate` `[1280, 2560]`, `49x` shared-expert `down` `[2560, 640]`, `200x` hyper-connection weights, `4x` MTP weights, `1x` LM head), yet [`fast_path_applies(FastPath::Q8SlicedK)`](file:///opt/etc/Infernix/src/models/qwen4_exp/program/fast_paths.cpp#L658) only enables `LinearRoutes::Q8SlicedK` when `configuration.dense == DenseKernels::Q8`.

2. **NVFP4 Linear `[2560, 6144]` `T = 17..32` `2.56x` Latency Cliff & `Nvfp4A4Pair` `T = 65..127` Crossover (`T2`, P0)**:
   - Live `infernix_linear_bench` sweeps on GB10 show that `[2560, 6144]` (`48x` per forward pass: `36x` `gdn/output` + `12x` `attention/output`) jumps from **`44.93 us` (`203.1 GB/s`) at `T = 16`** to **`90.34 us` at `T = 17` and `115.07 us` (`79.9 GB/s`, `2.56x` slower!) at `T = 19..20`** because [`n2560_k6144.cu`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n2560_k6144.cu#L32-L33) drops from `WarpsK = 8` (`Nvfp4SlicedInstance<16, 8, 2, 2>`) to `WarpsK = 4` (`Nvfp4SlicedInstance<32, 4, 2, 2>`) at `T = 17..24` and non-sliced MMA at `T = 25..32`. That `T = 17..32` window is the exact token batch size hit by `c = 2` DFlash2 pair verification (`2 * 9 = 18` tokens) and `c = 2..4` speculative verification.
   - Across `T = 65..127`, [`launch_nvfp4_a4_pair`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a4.cu#L305-L343) (`Nvfp4A4Pair` 2-term row-scaled FP4 TMA MMA) already pads workspace tokens to `128` (`Tiled128`) and guards out-of-bounds TMA/epilogue rows, and is **18%–31% faster** than A16 across `[16384, 2560]` (`127.8 us` vs `186.2 us`), `[13952, 2560]` (`119.9 us` vs `165.9 us`), `[2560, 6144]` (`63.3 us` vs `77.7 us`), and `[12800, 2560]` (`114.6 us` vs `161.6 us`), yet `pair_min_tokens` is gated at `128` instead of `65`.

3. **`hyper_connection_mix` Bottleneck at Speculative Verification Widths `T = 4..16` (`T3`, P0)**:
   - `hyper_connection_mix` runs **`98` times per forward pass** (`49` layers $\times$ `2` residual sites, `7.70 MB` per call = `754.6 MB` per pass).
   - Live `infernix_hyper_connection_bench` measurements on GB10 show that while `T = 1` achieves `30.41 us` (`253 GB/s`, `92.7%` peak DRAM BW), `T = 4` (MTP 3-draft verification width) degrades to **`61.04 us` (`126 GB/s`)**, and `T = 16` (DFlash2 16-column verification width) degrades to **`114.25 us` (`67 GB/s`)** — slower than `T = 17` (**`103.23 us`, min `92.67 us`**) where execution exits `hc_mix_fused` and uses Tensor Core MMA (`ops::linear` + `hc_up_collapse`)!
   - Testing `INFERNIX_TMP_HC_DOWN_COLUMNS=1` (`hc_down_partials_v4_kernel<..., true>`) on GB10 immediately reduces `T = 4` from **`61.04 us` to `48.31 us` (`-20.9%`, saving `1.25 ms` per MTP verification pass)** with identical FNV-1a output hash (`4e4a349854d217cd`). Furthermore, [`hc_up_collapse_v4_kernel`](file:///opt/etc/Infernix/src/ops/hyper_connection/hyper_connection_mix_fused.cu#L903-L912) statically allocates `36,928 B` SMEM for worst-case `16` columns (`__launch_bounds__(256, 1)`), throttling occupancy on GB10.

4. **Un-tuned Flash-Next Q8 Shared-Expert / DFlash2 Shapes & Remaining NVFP4 `/shapes/` (`T4`, P1)**:
   - `ckt shapes list --qtype q8` identified that [`src/ops/linear/q8/shapes/n1280_k2560.cu`](file:///opt/etc/Infernix/src/ops/linear/q8/shapes/n1280_k2560.cu) (`49x` shared-expert `gate` per forward pass in `NVFP4-W4Dense`!) has `origin = unknown`: its `SlicedK` schedules (`SlicedK_T8W8S1M1RT1`, `SlicedK_T8W8S2M1RT2`, `SlicedK_T16W8S1M1RT1`) were defined at lines 14–30 on 2026-10-24 **but never wired into `select_q8_n1280_k2560` or `q8_dispatch.cpp` (`fast = nullptr`)**!
   - Simultaneously, `DFlash2::materialize` (`dflash2.cpp:108`) invokes `ops::linear` on `p_.feature_projection` (`Q8_G32_FP16` `[2560, 12800]`), which is missing from `kShapes` in [`src/ops/linear/q8/q8_dispatch.cpp`](file:///opt/etc/Infernix/src/ops/linear/q8/q8_dispatch.cpp#L43-L102) and falls back to `select_q8_generic`.
   - Finally, the 3 remaining `origin = unknown` NVFP4 shape files ([`n5120_k6144.cu`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n5120_k6144.cu), [`n5120_k17408.cu`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n5120_k17408.cu), [`n34816_k5120.cu`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n34816_k5120.cu)) still carry RTX 5090 dispatch tables.

5. **`Qwen4Exp` Hybrid Prefix-Cache Cost Wiring (`set_hybrid_cost`) & GB10 Page-Setup Cost Term (`T5`, P1)**:
   - As uncovered in `20261029-changelog.md` (`T1` §7 & `T5` §2/§5), [`src/runtime/engine/qwen4_exp_instance.cpp`](file:///opt/etc/Infernix/src/runtime/engine/qwen4_exp_instance.cpp) never populates `LoadSummary::prefill_signature` and never calls `set_hybrid_cost` (unlike `qwen3_5` in [`src/runtime/engine/model_instance.cpp`](file:///opt/etc/Infernix/src/runtime/engine/model_instance.cpp#L139-L145)), and `ContextPrefillCost` lacks a page-setup cost term to capture GB10's `54%` cold-vs-warm prefix-cache setup delta at `P = 4096, U = 64`.

---

## Mandatory GB10 Workflow Rules for All Tasks

1. **Dead CUDA Depfiles in `build-native` (`CRITICAL`)**:
   - Editing `.cuh` or `.h` headers does **not** automatically trigger recompilation of `.cu.o` objects in `build-native`.
   - Whenever you edit any header under `src/ops/`, **always `touch` the consuming `.cu` files before running Ninja** (e.g., `find src/ops/<subdir> -name '*.cu' -exec touch {} +`), and verify `.cu.o` timestamps with `stat`.
2. **Single-Invocation Ninja Build (`nvlink` Safety)**:
   - Never run concurrent `ninja` invocations against `build-native`. Build `infernix-serve`, benchmarks, and tests in a single command to share the ~55.8s `cmake_device_link.o` link step.
3. **One-Way CUDA Stack-Spill Ratchet Gate (`infernix_cuda_stack_check`)**:
   - Every task must pass `ctest --test-dir build-native -R infernix_cuda_stack_check --output-on-failure` with **zero** new or increased `STACK > 0` or `LOCAL > 0` symbols. Never run `check_cuda_stack.py update` to mask a new spill.
4. **GB10 Changelog Documentation (`docs/gb10/20261030-changelog.md`)**:
   - Every completed task must record its accounting, task-premise verification, subsystem breakdown, verification metrics, and accepted gaps in `docs/gb10/20261030-changelog.md`.

---

### Task T1 (Priority P0): Enable & Qualify `FastPath::GdnConvFused`, `FastPath::DFlash2Pairs`, and `FastPath::Q8SlicedK` for `DenseKernels::Nvfp4` (`Qwen4Exp`)

#### Affected Files & Symbols
- [`src/models/qwen4_exp/execution/forward.cpp`](file:///opt/etc/Infernix/src/models/qwen4_exp/execution/forward.cpp#L631-L649)
  - [`Forward::gdn`](file:///opt/etc/Infernix/src/models/qwen4_exp/execution/forward.cpp#L619-L660) (`fused_conv` qtype check at line 632)
- [`src/models/qwen4_exp/program/fast_paths.cpp`](file:///opt/etc/Infernix/src/models/qwen4_exp/program/fast_paths.cpp#L397-L665)
  - [`kQualified`](file:///opt/etc/Infernix/src/models/qwen4_exp/program/fast_paths.cpp#L397-L565) (`FastPath::DFlash2Pairs`, `FastPath::GdnConvFused`, `FastPath::Q8SlicedK` entries for `DenseKernels::Nvfp4`)
  - [`fast_path_applies`](file:///opt/etc/Infernix/src/models/qwen4_exp/program/fast_paths.cpp#L649-L664)
  - [`FastPathSelection::dense_routes`](file:///opt/etc/Infernix/src/models/qwen4_exp/program/fast_paths.cpp#L815-L837)
- [`src/ops/wrapper/gdn_input_proj.cpp`](file:///opt/etc/Infernix/src/ops/wrapper/gdn_input_proj.cpp#L1280-L1331)
  - [`ops::gdn_input_proj_conv_snapshot`](file:///opt/etc/Infernix/src/ops/wrapper/gdn_input_proj.cpp#L1280-L1331)
- [`tests/models/qwen4_exp/test_fast_paths.cpp`](file:///opt/etc/Infernix/tests/models/qwen4_exp/test_fast_paths.cpp#L230-L430)

#### Root Cause & Hardware Rationale
Direct code inspection and `20261029-changelog.md` measurements established three artificial fast-path gates that penalize `Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix` (`DenseKernels::Nvfp4`, `routed_q4 = true`):

1. **`FastPath::GdnConvFused` is implemented in `infernix_ops` for `NVFP4 [16384, 2560]` but blocked in `qwen4_exp`**:
   - In [`src/ops/linear/nvfp4/shapes/n16384_k2560.cu`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n16384_k2560.cu#L140-L154), `launch_nvfp4_n16384_k2560_conv_snapshot` is already compiled and wired into [`ops::gdn_input_proj_conv_snapshot`](file:///opt/etc/Infernix/src/ops/wrapper/gdn_input_proj.cpp#L1318-L1324) for `w.qtype == QType::NVFP4` and `B = 1..8`.
   - However, in [`src/models/qwen4_exp/execution/forward.cpp`](file:///opt/etc/Infernix/src/models/qwen4_exp/execution/forward.cpp#L632):
     ```cpp
     const bool fused_conv =
         fast_.gdn_conv_fused && decode && p.projection.weight.qtype == QType::Q8_G32_FP16;
     ```
     and in [`src/models/qwen4_exp/program/fast_paths.cpp`](file:///opt/etc/Infernix/src/models/qwen4_exp/program/fast_paths.cpp#L659):
     ```cpp
     case FastPath::GdnConvFused: return configuration.dense == DenseKernels::Q8;
     ```
   - Because `NVFP4` is excluded in both places and missing from `kQualified`, every decode step on `NVFP4-W4Dense` runs `ops::gdn_input_proj` followed by a standalone `ops::causal_conv1d_silu_snapshot_split` launch across all `36` GDN layers (`36` extra kernel launches + `36 * 20.48 KB * B` extra LPDDR5X round-trip reads/writes per token).

2. **`FastPath::DFlash2Pairs` is missing from `kQualified` for `DenseKernels::Nvfp4`**:
   - In [`src/models/qwen4_exp/program/fast_paths.cpp`](file:///opt/etc/Infernix/src/models/qwen4_exp/program/fast_paths.cpp#L397-L445), `FastPath::DFlash2Pairs` is qualified for `DenseKernels::Q8`, `Ggml`, `Bf16`, `Fp8Row`, and `Mxfp8`, but has **zero** entries for `DenseKernels::Nvfp4`.
   - As measured in `20261029-changelog.md` (`T7` §2 & §5) on `Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix` (`FP8` KV):
     - At `c = 1`, `--spec dflash2` reaches **`114.8 tok/s`** (`+25.2%` faster than `--spec mtp` at `91.7 tok/s`, and `+176.0%` faster than `plain` at `41.6 tok/s`).
     - At `c = 2`, because `FastPath::DFlash2Pairs` is not enabled for `DenseKernels::Nvfp4`, DFlash2 refuses 2-request pair verification once both lanes are active (`spec_rounds = 6 / 723`), collapsing aggregate throughput to **`35.6 tok/s`** (`~= plain 34.1 tok/s`, `-33.8%` behind MTP's `53.8 tok/s`). On `Dense8`, enabling `DFlash2Pairs` (`kDFlashPairMaxDrafts = 8`, `W = 9`, `2 * 9 = 18` verify tokens) lifted `c = 2` throughput to **`68.5 tok/s`** (`+27.3%` ahead of MTP).

3. **`FastPath::Q8SlicedK` is disabled when `configuration.dense == DenseKernels::Nvfp4`**:
   - Inspection of `/opt/etc/models/Qwen3.8-Flash-Next-F4-Local-NVPF4-Dense/Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix.conversion.json` confirms that `NVFP4-W4Dense` stores **`303` `Q8_G32_FP16` weights**, including `49x` `moe/shared/gate` (`[1280, 2560]`), `49x` `moe/shared/down` (`[2560, 640]`), `200x` hyper-connection weights, `4x` MTP weights (`[16384, 2560]`, `[2560, 6144]`, `[5120, 2560]`), and `1x` LM head (`[248320, 2560]`).
   - Yet [`fast_path_applies(FastPath::Q8SlicedK)`](file:///opt/etc/Infernix/src/models/qwen4_exp/program/fast_paths.cpp#L658) checks `configuration.dense == DenseKernels::Q8`, preventing `LinearRoutes::Q8SlicedK` from being added in `FastPathSelection::dense_routes()` when `configuration.dense == DenseKernels::Nvfp4`.

#### Required Implementation Steps
1. **Allow `QType::NVFP4` in `Forward::gdn` (`src/models/qwen4_exp/execution/forward.cpp`)**:
   - In [`Forward::gdn`](file:///opt/etc/Infernix/src/models/qwen4_exp/execution/forward.cpp#L631-L633), update `fused_conv` to accept both `QType::Q8_G32_FP16` and `QType::NVFP4`:
     ```cpp
     const bool fused_conv =
         fast_.gdn_conv_fused && decode &&
         (p.projection.weight.qtype == QType::Q8_G32_FP16 ||
          p.projection.weight.qtype == QType::NVFP4);
     ```
2. **Extend `fast_path_applies` and `kQualified` in `src/models/qwen4_exp/program/fast_paths.cpp`**:
   - In [`fast_path_applies`](file:///opt/etc/Infernix/src/models/qwen4_exp/program/fast_paths.cpp#L658-L659), allow `DenseKernels::Nvfp4` for `FastPath::Q8SlicedK` and `FastPath::GdnConvFused`:
     ```cpp
     case FastPath::Q8SlicedK:
     case FastPath::GdnConvFused:
         return configuration.dense == DenseKernels::Q8 ||
                configuration.dense == DenseKernels::Nvfp4;
     ```
   - In [`kQualified`](file:///opt/etc/Infernix/src/models/qwen4_exp/program/fast_paths.cpp#L397-L545), add the qualified `{DenseKernels::Nvfp4, true, kv}` and `{DenseKernels::Nvfp4, false, kv}` entries (across `KvDType::FP16`, `BF16`, `FP8_E4M3`, `INT8`, `INT4`) for:
     - `FastPath::DFlash2Pairs`
     - `FastPath::GdnConvFused`
     - `FastPath::Q8SlicedK`
3. **Update `tests/models/qwen4_exp/test_fast_paths.cpp`**:
   - Add unit test coverage verifying that a `Configuration{DenseKernels::Nvfp4, true, KvDType::FP8_E4M3}` (and `routed_q4 = false`) selects `GdnConvFused`, `DFlash2Pairs`, and `Q8SlicedK` (with `LinearRoutes::Q8SlicedK` present in `dense_routes()`) under `ReproducibilityMode::Fast`, and disables non-request-invariant fast paths under `ReproducibilityMode::Reproducible`.
4. **Verify Oracle & End-to-End Serving (`c = 1` and `c = 2` with `--spec dflash2` and `plain`/`mtp`)**:
   - Run `infernix_gdn_input_proj_test` to confirm `ops::gdn_input_proj_conv_snapshot` on `NVFP4 [16384, 2560]` passes against the FP32 oracle for `B = 1..8`.
   - Benchmark `Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix` with `infernix-serve` at `c = 1` and `c = 2` under `--spec dflash2`, `--spec mtp`, and `plain`.

#### Verification & Acceptance Criteria
```bash
# 1. Single-invocation build of test targets and infernix-serve
ninja -C build-native -j "$(nproc)" \
  infernix_qwen4_exp_fast_paths_test \
  infernix_gdn_input_proj_test \
  infernix-serve

# 2. Fast-path unit test & GDN fused conv snapshot oracle test
./build-native/tests/infernix_qwen4_exp_fast_paths_test
./build-native/tests/infernix_gdn_input_proj_test

# 3. One-way CUDA stack-spill ratchet gate
ctest --test-dir build-native -R infernix_cuda_stack_check --output-on-failure

# 4. End-to-end NVFP4-W4Dense serving verification (c=1 and c=2 DFlash2 vs MTP vs Plain)
# Verify spec_rounds > 0 at c=2 with --spec dflash2 and record tok/s in docs/gb10/20261030-changelog.md
```
- **Acceptance Criteria**:
  1. `infernix_qwen4_exp_fast_paths_test` and `infernix_gdn_input_proj_test` pass 100%.
  2. On `Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix` (`--kv-dtype fp8_e4m3`), `c = 2` `--spec dflash2` actively executes 2-lane pair speculation (`spec_rounds` firing across multi-lane rounds rather than falling back to `6 / 723` single-lane rounds) and improves `c = 2` DFlash2 aggregate throughput significantly over the `35.6 tok/s` non-paired baseline.
  3. Plain decode on `NVFP4-W4Dense` fuses the 36 GDN `causal_conv1d_silu_snapshot` steps into `launch_nvfp4_n16384_k2560_conv_snapshot` with zero numerical drift beyond the qualified `GdnConvFused` contract.
  4. `infernix_cuda_stack_check` passes with zero new spills.

---

### Task T2 (Priority P0): Eliminate NVFP4 Linear `[2560, 6144]` `T = 17..32` `2.56x` Latency Cliff and Lower `Nvfp4A4Pair` Admission Crossover from `T >= 128` to `T >= 65` on Flash-Next Shapes

#### Affected Files & Symbols
- [`src/ops/linear/nvfp4/shapes/n2560_k6144.cu`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n2560_k6144.cu#L29-L53)
  - [`select_a16`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n2560_k6144.cu#L29-L41)
  - [`kNvfp4N2560K6144`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n2560_k6144.cu#L43-L53) (`pair_min_tokens`)
- [`src/ops/linear/nvfp4/shapes/n16384_k2560.cu`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n16384_k2560.cu#L114-L138)
  - [`kNvfp4N16384K2560`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n16384_k2560.cu#L114-L116) (`pair_min_tokens`)
  - [`launch_nvfp4_n16384_k2560_split`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n16384_k2560.cu#L120-L138) (`x.ne[1] >= 65` check at line 123)
- [`src/ops/linear/nvfp4/shapes/n13952_k2560.cu`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n13952_k2560.cu#L115-L142)
  - [`kNvfp4N13952K2560`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n13952_k2560.cu#L115-L120) (`pair_min_tokens`)
  - [`launch_nvfp4_n13952_k2560_split`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n13952_k2560.cu#L122-L142) (`x.ne[1] >= 65` check at line 127)
- [`src/ops/linear/nvfp4/shapes/n12800_k2560.cu`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n12800_k2560.cu#L44-L53)
  - [`kNvfp4N12800K2560`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n12800_k2560.cu#L44-L53) (`pair_min_tokens`)
- [`src/ops/wrapper/gdn_input_proj.cpp`](file:///opt/etc/Infernix/src/ops/wrapper/gdn_input_proj.cpp#L896-L900)
  - [`gdn_input_proj_workspace_capacity_bytes`](file:///opt/etc/Infernix/src/ops/wrapper/gdn_input_proj.cpp#L893-L903) (`max_tokens >= 65` check at line 897)
- [`src/ops/wrapper/attn_input_proj.cpp`](file:///opt/etc/Infernix/src/ops/wrapper/attn_input_proj.cpp#L214-L219)
  - [`attn_input_proj_workspace_capacity_bytes`](file:///opt/etc/Infernix/src/ops/wrapper/attn_input_proj.cpp#L211-L222) (`max_tokens >= 65` check at line 215)

#### Root Cause & Hardware Rationale
Live session benchmarks using `./build-native/bench/infernix_linear_bench` on GB10 exposed two major performance losses in the four `Qwen3.8-Flash-Next-Local-NVFP4-W4Dense` NVFP4 linear shapes (`[2560, 6144]` $\times 48$, `[16384, 2560]` $\times 36$, `[13952, 2560]` $\times 12$, `[12800, 2560]` $\times 1$):

1. **`[2560, 6144]` (`48x` per forward pass) suffers a `+101%` to `+156%` latency cliff across `T = 17..32`**:
   - Command run: `./build-native/bench/infernix_linear_bench --qtype NVFP4 --n 2560 --k 6144 --sweep 1:64:1 --warmup 2 --iters 10 --routes exact`
   - Measured GB10 latencies:
     - `T = 16`: **`44.928 us`** (`203.1 GB/s`, `74.40%` DRAM peak) via `Nvfp4SlicedInstance<16, 8, 2, 2>` (`WarpsK = 8`, `Slices = 2`, `160` CTAs).
     - `T = 17`: **`90.336 us`** (`101.2 GB/s`, **`+101.07%` cliff!**) and `T = 19..20`: **`115.072 us`** (`79.9 GB/s`, **`2.56x` slower!**) because [`select_a16` in `n2560_k6144.cu:32`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n2560_k6144.cu#L32) dispatches `Nvfp4SlicedInstance<32, 4, 2, 2>` (`WarpsK = 4`), cutting K-reduction warp parallelism in half on `K = 6144` (`192` scale blocks).
     - `T = 25..32`: **`63.296..110.528 us`** (`84.6..149.2 GB/s`) because line 33 switches at `T = 25` to non-sliced `Nvfp4A16MmaSchedule<32, 32, 128, 16, 16, 2, 2>` (`80` CTAs iterating serially over `K = 6144`), whereas `T = 33..64` uses `Nvfp4A16MmaSchedule<32, 64, 128, 16, 32, 2, 2>` at `77.7 us`.
   - **Why this matters**: `T = 18` (`2 * 9`) is the exact verification batch size of `c = 2` DFlash2 pair speculation (`kDFlashPairMaxDrafts = 8`), and `T = 16..32` is hit by `c = 1..2` DFlash2 (`W = 16`) and `c = 4..8` MTP (`W = 4`) verification rounds! Across `48` layers per pass, a `45–70 us` regression per `[2560, 6144]` call costs **`2.16–3.36 ms` per forward pass**!

2. **`Nvfp4A4Pair` (`launch_nvfp4_a4_pair`) is 18%–31% faster across `T = 65..127` on all 4 Flash-Next shapes, but `pair_min_tokens` is hardcoded to `128`**:
   - Commands run:
     `./build-native/bench/infernix_linear_bench --qtype NVFP4 --n <N> --k <K> --sweep 64:160:8 --warmup 2 --iters 10 --routes fast`
   - Measured GB10 latencies (`T = 64..120` currently runs A16; `T = 128` activates `Nvfp4A4Pair`):
     - `[16384, 2560]` (`36x`/pass): `T = 72..120` takes **`147.3..186.2 us`** (`152.3..192.6 GB/s`), then drops **`-31.35%`** at `T = 128` to **`127.8 us`** (`222.5 GB/s`, `81.5%` DRAM) when `Nvfp4A4Pair` activates. Saving `58.4 us` across `36` GDN input projections saves **`2.10 ms`** per pass at `T = 96..127`.
     - `[13952, 2560]` (`12x`/pass): `T = 72..120` takes **`133.0..165.9 us`**, then drops **`-27.74%`** at `T = 128` to **`119.9 us`** (`202.9 GB/s`).
     - `[2560, 6144]` (`48x`/pass): `T = 72..120` takes **`77.7..78.0 us`**, then drops **`-18.46%`** at `T = 128` to **`63.3 us`** (`174.9 GB/s`).
     - `[12800, 2560]` (`1x`/pass): `T = 72..120` takes **`120.0..161.6 us`**, then drops **`-29.1%`** at `T = 128` to **`114.6 us`** (`192.6 GB/s`).
   - **Hardware & Safety Verification**:
     - [`allocate_nvfp4_a4_workspace`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a4_plan.h#L56-L86) already rounds `written_tokens` up to a multiple of `128` (`Tiled128`), and [`nvfp4_a4_row_scaled_residual_quantize_kernel`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a4.cu#L73-L86) zeroes padding rows `token >= tokens` up to `written_tokens`.
     - [`make_nvfp4_a4_tma_descriptors`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a4_tma.cuh#L93-L111) sets `CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE`, and all three epilogue paths in [`nvfp4_a4_tma.cuh`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a4_tma.cuh#L464-L521) guard stores with `if (token_begin + t < token_count)`.
     - Even for `T = 65..127` (where TMA executes one 128-token tile of `QMMA.16864.F32.E2M1`), `Nvfp4A4Pair` takes `~127.8 us` on `[16384, 2560]` vs `147.3..186.2 us` for A16 `TileTokens = 64` (which requires 2 token tiles across `N = 16384` and reads weights twice!).

#### Required Implementation Steps
1. **Fix the `T = 17..32` latency cliff in [`src/ops/linear/nvfp4/shapes/n2560_k6144.cu`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n2560_k6144.cu#L29-L41)**:
   - Benchmark candidate schedules across `T = 17..64` on `[2560, 6144]` using `infernix_linear_bench`, specifically comparing:
     - `Nvfp4SlicedInstance<24, 8, 2, 2>` / `Nvfp4SlicedInstance<32, 8, 2, 2>` (`WarpsK = 8`, `Slices = 2`, `MinBlocks = 2`) for `T = 17..24` and `T = 25..32`
     - `Nvfp4A16MmaSchedule<32, 64, 128, 16, 32, 2, 2>` vs `Nvfp4A16MmaSchedule<32, 32, 128, 16, 16, 2, 2>` for `T = 25..64`
   - Replace `Nvfp4SlicedInstance<32, 4, 2, 2>` (and the slow `T = 25..32` dispatch) with the measured winner so that `T = 17..32` stays monotonic (`<= 58 us` at `T = 17..24`, eliminating the `90–115 us` cliff).
2. **Lower `pair_min_tokens` from `128` to `65` on the 4 Flash-Next shapes**:
   - Update `pair_min_tokens` to `65` in:
     - [`src/ops/linear/nvfp4/shapes/n16384_k2560.cu`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n16384_k2560.cu#L115) (`kNvfp4N16384K2560`) and change `x.ne[1] >= 128` to `x.ne[1] >= 65` in [`launch_nvfp4_n16384_k2560_split`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n16384_k2560.cu#L123).
     - [`src/ops/linear/nvfp4/shapes/n13952_k2560.cu`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n13952_k2560.cu#L118) (`kNvfp4N13952K2560`) and change `x.ne[1] >= 128` to `x.ne[1] >= 65` in [`launch_nvfp4_n13952_k2560_split`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n13952_k2560.cu#L127).
     - [`src/ops/linear/nvfp4/shapes/n2560_k6144.cu`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n2560_k6144.cu#L50) (`kNvfp4N2560K6144`).
     - [`src/ops/linear/nvfp4/shapes/n12800_k2560.cu`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n12800_k2560.cu#L51) (`kNvfp4N12800K2560`).
   - Update workspace sizing thresholds in [`gdn_input_proj_workspace_capacity_bytes`](file:///opt/etc/Infernix/src/ops/wrapper/gdn_input_proj.cpp#L897) and [`attn_input_proj_workspace_capacity_bytes`](file:///opt/etc/Infernix/src/ops/wrapper/attn_input_proj.cpp#L215) from `max_tokens >= 128` to `max_tokens >= 65` so pre-allocated workspace is available whenever `T >= 65`.

#### Verification & Acceptance Criteria
```bash
# 1. Build bench, tests, and infernix-serve in one invocation
ninja -C build-native -j "$(nproc)" \
  infernix_linear_bench \
  infernix_linear_nvfp4_a16_test \
  infernix_linear_nvfp4_a4_test \
  infernix_gdn_input_proj_test \
  infernix_attn_input_proj_test \
  infernix-serve

# 2. Oracle verification across all modified NVFP4 shapes and wrapper tests (including T=65..127 non-128-multiple token counts)
./build-native/tests/infernix_linear_nvfp4_a16_test --shape 2560 6144
./build-native/tests/infernix_linear_nvfp4_a4_test
./build-native/tests/infernix_gdn_input_proj_test
./build-native/tests/infernix_attn_input_proj_test

# 3. Benchmark [2560, 6144] T=1..64 and all 4 Flash-Next shapes T=64..160 (--routes exact and --routes fast)
./build-native/bench/infernix_linear_bench --qtype NVFP4 --n 2560 --k 6144 --sweep 1:64:1 --warmup 2 --iters 10 --routes exact
./build-native/bench/infernix_linear_bench --qtype NVFP4 --n 16384 --k 2560 --sweep 64:160:8 --warmup 2 --iters 10 --routes fast
./build-native/bench/infernix_linear_bench --qtype NVFP4 --n 13952 --k 2560 --sweep 64:160:8 --warmup 2 --iters 10 --routes fast
./build-native/bench/infernix_linear_bench --qtype NVFP4 --n 2560 --k 6144 --sweep 64:160:8 --warmup 2 --iters 10 --routes fast
./build-native/bench/infernix_linear_bench --qtype NVFP4 --n 12800 --k 2560 --sweep 64:160:8 --warmup 2 --iters 10 --routes fast

# 4. One-way CUDA stack-spill ratchet gate
ctest --test-dir build-native -R infernix_cuda_stack_check --output-on-failure
```
- **Acceptance Criteria**:
  1. On `[2560, 6144]`, `T = 17..24` latency drops from `90.3–115.1 us` (`80–101 GB/s`) to `<= 58 us` (`>= 155 GB/s`), eliminating the `2.56x` cliff.
  2. On `[16384, 2560]`, `[13952, 2560]`, `[2560, 6144]`, and `[12800, 2560]`, `--routes fast` dispatches `Nvfp4A4Pair` for `T >= 65`, reducing `T = 72..120` latency by `18%–31%`.
  3. All NVFP4 A16/A4 and wrapper oracle tests (`infernix_linear_nvfp4_a16_test`, `infernix_linear_nvfp4_a4_test`, `infernix_gdn_input_proj_test`, `infernix_attn_input_proj_test`) and `infernix_cuda_stack_check` pass 100%.

---

### Task T3 (Priority P0): Optimize `hyper_connection_mix` at Speculative Verification & Medium Widths (`T = 2..16`, `98x` per Forward Pass) on GB10

#### Affected Files & Symbols
- [`src/ops/hyper_connection/hyper_connection_mix_fused.cu`](file:///opt/etc/Infernix/src/ops/hyper_connection/hyper_connection_mix_fused.cu#L719-L1168)
  - [`hc_down_partials_v4_kernel`](file:///opt/etc/Infernix/src/ops/hyper_connection/hyper_connection_mix_fused.cu#L719-L808)
  - [`hc_up_collapse_v4_kernel`](file:///opt/etc/Infernix/src/ops/hyper_connection/hyper_connection_mix_fused.cu#L884-L1016)
  - [`launch_down_v4`](file:///opt/etc/Infernix/src/ops/hyper_connection/hyper_connection_mix_fused.cu#L1035-L1052)
  - [`launch`](file:///opt/etc/Infernix/src/ops/hyper_connection/hyper_connection_mix_fused.cu#L1125-L1168)
- [`src/ops/hyper_connection/hyper_connection_mix_fused.h`](file:///opt/etc/Infernix/src/ops/hyper_connection/hyper_connection_mix_fused.h#L16-L32)
  - [`kHcMixFusedMaxColumns`](file:///opt/etc/Infernix/src/ops/hyper_connection/hyper_connection_mix_fused.h#L18)
- [`src/ops/hyper_connection/hyper_connection.cu`](file:///opt/etc/Infernix/src/ops/hyper_connection/hyper_connection.cu#L255-L321)
  - [`hyper_connection_mix`](file:///opt/etc/Infernix/src/ops/hyper_connection/hyper_connection.cu#L255-L287)
  - [`hyper_connection_inject_mix`](file:///opt/etc/Infernix/src/ops/hyper_connection/hyper_connection.cu#L289-L321)

#### Root Cause & Hardware Rationale
In `Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix` (and `Dense8`), `hyper_connection_mix` executes **`98` times per forward pass** (`49` layers $\times$ `2` residual mixing sites, reading `7.70 MB` of `Q8_G32_FP16` `down_proj` `[324, 10240]` + `up_proj` `[10240, 320]` weights per call = **`754.6 MB` per pass**). Because the dense projections in `NVFP4-W4Dense` are compressed to 4-bit `NVFP4`, `hyper_connection_mix` represents an even larger fraction of total forward pass time on `NVFP4-W4Dense` than on `Dense8`.

Live session measurements on GB10 via `./build-native/bench/infernix_hyper_connection_bench --columns 1,2,3,4,5,6,8,12,16,17,24,32,48,64 --warmup 3 --repeat 20` revealed severe degradation across `T = 4..16`:
- `T = 1`: **`30.41 us`** (`253 GB/s`, `92.7%` of GB10 DRAM peak — uses `hc_down_partials_kernel<..., 1, 2>` + `hc_up_collapse_kernel<..., 1, 2>` with `MinBlocks = 2` and `5.1 KB` dynamic SMEM).
- `T = 2`: **`35.44 us`** (`217 GB/s`).
- `T = 3`: **`41.49 us`** (`186 GB/s`).
- `T = 4` (MTP 3-draft verification width!): **`61.04 us` (`126 GB/s`)** — a `+47.1%` jump over `T = 3` (`41.49 us`) and slower than `T = 5` (**`53.13 us`, `145 GB/s`**)!
- `T = 8` (DFlash2 pair per-lane draft width): **`67.98 us` (`113 GB/s`)**.
- `T = 12`: **`94.73 us` (`81 GB/s`)**.
- `T = 16` (DFlash2 single-lane 16-column verification width!): **`114.25 us` (`67 GB/s`)** — which is **slower than `T = 17` (`103.23 us`, min `92.67 us`, `75–83 GB/s`)** where `columns > kHcMixFusedMaxColumns` exits `hc_mix_fused` and runs `ops::linear` (Q8 Tensor Core MMA) + `hc_up_collapse`!

Three concrete root causes in [`src/ops/hyper_connection/hyper_connection_mix_fused.cu`](file:///opt/etc/Infernix/src/ops/hyper_connection/hyper_connection_mix_fused.cu) explain these measurements:
1. **`T = 4` `hc_down_partials_v4_kernel` wave quantization & `MinBlocks = 1` serialization**:
   - At `T = 4`, `launch` (`line 1143`) selects `PassColumns = 4` (`ColumnGrid = false`) with `grid = (41, 4, 1) = 164` CTAs and `__launch_bounds__(256, 1)` (`MinBlocks = 1`).
   - Testing `INFERNIX_TMP_HC_DOWN_COLUMNS=1 ./build-native/bench/infernix_hyper_connection_bench --columns 4 --warmup 3 --repeat 20` (which dispatches `hc_down_partials_v4_kernel<..., true>` with `PassColumns = 1`, `ColumnGrid = true`) immediately drops `T = 4` from **`61.04 us` (`126 GB/s`) to `48.31 us` (`159 GB/s`, `-20.86%` faster!)** with the exact same FNV-1a output hash (`4e4a349854d217cd`)! Across `98` calls per pass, `-12.73 us/call` saves **`1.25 ms` per MTP verification step**!
2. **Oversized `36,928 B` static shared memory and `MinBlocks = 1` in `hc_up_collapse_v4_kernel` (`lines 885–912`)**:
   - Unlike `hc_up_collapse_kernel` (`line 424`), which uses dynamic shared memory (`extern __shared__ float smem[];` sized to `columns * rank * sizeof(float) = 5,120 B` at `T = 4` or `10,240 B` at `T = 8`) and `MinBlocks = 2`, `hc_up_collapse_v4_kernel` statically allocates `__shared__ float v[kHcMixFusedMaxColumns * kMaxRank]` (`16 * 512 * 4 = 32,768 B`) + `__shared__ uint32_t u[1024]` (`4,096 B`) = **`36,928 B` static SMEM** and forces `__launch_bounds__(256, 1)` even when `columns = 2..8` only needs `6.7–14.5 KB` of SMEM!
3. **Suboptimal SIMT-vs-MMA crossover (`kHcMixFusedMaxColumns = 16`) on GB10**:
   - At `T = 12..16`, `hc_mix_fused`'s SIMT dot-product loop (`94.7–114.3 us`) is outperformed by or tied with the Tensor Core MMA route (`ops::linear` + `hc_up_collapse`), and shrinking `hc_up_collapse_v4_kernel` to dynamic SMEM with `MinBlocks = 2` (or tuning `PassColumns` / `ColumnGrid` and `kHcMixFusedMaxColumns` on GB10) directly accelerates both `T = 4..8` and `T = 12..16` (DFlash2 `W = 16` verification).

#### Required Implementation Steps
1. **Convert `hc_up_collapse_v4_kernel` (`hyper_connection_mix_fused.cu:884-915`) from worst-case `36.9 KB` static SMEM to dynamic SMEM and template `MinBlocks`**:
   - Allocate `v` and `u` in dynamic shared memory sized to `columns * kMaxRank * sizeof(float) + (kHiddenSize / 8) * sizeof(uint32_t)` (or template on `MaxCols` `4` / `8` / `16` so `T <= 4` uses `10.2 KB` SMEM and `MinBlocks = 2`, `T <= 8` uses `18.4 KB` SMEM and `MinBlocks = 2`).
   - Verify with `cuobjdump --dump-resource-usage` that `REG <= 64`, `STACK = 0`, `LOCAL = 0` for all `hc_up_collapse_v4_kernel` and `hc_down_partials_v4_kernel` instantiations.
2. **Optimize `T = 2..16` dispatch in `launch_down_v4` / `launch` (`hyper_connection_mix_fused.cu:1035-1168`) and `kHcMixFusedMaxColumns`**:
   - Remove the temporary `std::getenv("INFERNIX_TMP_HC_DOWN_COLUMNS")` hack (`lines 1044–1050`) and replace it with a clean compile-time/column-count dispatch calibrated on GB10 across `T = 1..16` (using `ColumnGrid = true` or `PassColumns = 1..2` where measured faster, such as `T = 4` at `48.3 us` vs `61.0 us`).
   - Measure `T = 8, 9, 12, 16, 17` with dynamic-SMEM `v4` vs the `ops::linear` + `hc_up_collapse` Tensor Core MMA path, and set the GB10 fused-mix column threshold so `T = 4` (MTP), `T = 8..9` (DFlash2 pair), and `T = 16` (DFlash2 single) all run the fastest route.
3. **Touch `.cu` files if any `.h` header is modified**:
   - If `src/ops/hyper_connection/hyper_connection_mix_fused.h` is edited, run `find src/ops/hyper_connection -name '*.cu' -exec touch {} +` before building.

#### Verification & Acceptance Criteria
```bash
# 1. If any header in src/ops/hyper_connection was edited, touch consuming .cu files first:
find src/ops/hyper_connection -name '*.cu' -exec touch {} +

# 2. Build bench, unit tests, and infernix-serve in a single invocation
ninja -C build-native -j "$(nproc)" \
  infernix_hyper_connection_bench \
  infernix_ops_state_recurrent_test \
  infernix-serve

# 3. Run hyper_connection oracle tests
ctest --test-dir build-native -R "hyper_connection" --output-on-failure

# 4. Run GB10 hyper_connection_bench across T=1..64
./build-native/bench/infernix_hyper_connection_bench --columns 1,2,3,4,5,6,8,9,12,16,17,24,32,48,64 --warmup 3 --repeat 20

# 5. One-way CUDA stack-spill ratchet gate
ctest --test-dir build-native -R infernix_cuda_stack_check --output-on-failure
```
- **Acceptance Criteria**:
  1. `T = 4` latency in `infernix_hyper_connection_bench` improves from `61.04 us` (`126 GB/s`) to **`<= 48.5 us` (`>= 158 GB/s`, `>= 20%` speedup)**.
  2. `T = 8`, `T = 12`, and `T = 16` latencies improve monotonically without the `T = 16` (`114.25 us`) > `T = 17` (`103.23 us`) inversion (`T = 16 <= 98 us`).
  3. `T = 1` (`~30.4 us`, `253 GB/s`) does not regress.
  4. All `hyper_connection` CTests and `infernix_cuda_stack_check` pass with zero new stack spills.

---

### Task T4 (Priority P1): Wire & Retune Flash-Next Shared-Expert / DFlash2 Q8 Projections (`[1280, 2560]`, `[2560, 12800]`) and Remaining Unmarked NVFP4 `/shapes/` (`[5120, 6144]`, `[5120, 17408]`, `[34816, 5120]`) on GB10

#### Affected Files & Symbols
- [`src/ops/linear/q8/shapes/n1280_k2560.cu`](file:///opt/etc/Infernix/src/ops/linear/q8/shapes/n1280_k2560.cu#L1-L42)
  - [`select_q8_n1280_k2560`](file:///opt/etc/Infernix/src/ops/linear/q8/shapes/n1280_k2560.cu#L33-L40)
  - `select_q8_n1280_k2560_fast` (new fast-route selector wiring `SlicedK` schedules)
- [`src/ops/linear/q8/q8_dispatch.cpp`](file:///opt/etc/Infernix/src/ops/linear/q8/q8_dispatch.cpp#L43-L102)
  - [`kShapes`](file:///opt/etc/Infernix/src/ops/linear/q8/q8_dispatch.cpp#L43-L102) (wire `select_q8_n1280_k2560_fast` for `{1280, 2560}` and add `{2560, 12800}` for DFlash2 `feature_projection`)
- [`src/ops/linear/q8/shapes/shapes.h`](file:///opt/etc/Infernix/src/ops/linear/q8/shapes/shapes.h)
- [`src/ops/linear/nvfp4/shapes/n5120_k6144.cu`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n5120_k6144.cu)
- [`src/ops/linear/nvfp4/shapes/n5120_k17408.cu`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n5120_k17408.cu)
- [`src/ops/linear/nvfp4/shapes/n34816_k5120.cu`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n34816_k5120.cu)

#### Root Cause & Hardware Rationale
Running `PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt shapes list --qtype q8` and `--qtype nvfp4` alongside inspection of the `NVFP4-W4Dense` and `DFlash2` conversion manifests revealed three concrete shape-dispatch gaps:

1. **`src/ops/linear/q8/shapes/n1280_k2560.cu` (`49x` `moe/shared/gate` per forward pass in `NVFP4-W4Dense` and `Dense8`) has dead `SlicedK` definitions that were never wired into `select_q8_n1280_k2560` or `q8_dispatch.cpp`!**
   - In [`src/ops/linear/q8/shapes/n1280_k2560.cu`](file:///opt/etc/Infernix/src/ops/linear/q8/shapes/n1280_k2560.cu#L14-L30), `SlicedK_T8W8S1M1RT1`, `SlicedK_T8W8S2M1RT2`, `SlicedK_T16W8S1M1RT1`, and `sliced<Schedule>` are defined at lines 14–30, **but lines 33–40 (`select_q8_n1280_k2560`) never call them** (`T = 3..4` calls `launch_q8_a16_simt_r8_t4`, `T = 5..48` calls `launch_q8_a16_simt_r8_t8`, and [`q8_dispatch.cpp:50`](file:///opt/etc/Infernix/src/ops/linear/q8/q8_dispatch.cpp#L50) has `fast = nullptr`)!
   - On GB10 (48 SMs), `N = 1280` with `8` rows/CTA (`launch_q8_a16_simt_r8_t4` / `r8_t8`) launches `160` CTAs that iterate serially over `K = 2560`, whereas `Q8SlicedK` splits `K = 2560` across warps and uses Tensor Core MMA for `T >= 3`. Because `moe/shared/gate` (`[1280, 2560]`) runs **`49` times per forward pass** in `NVFP4-W4Dense` (both during decode `T = 1..2`, MTP `T = 4`, and DFlash2 verification `T = 9..18`), leaving `[1280, 2560]` on non-sliced SIMT up to `T = 48` wastes substantial latency on every speculative verification pass.

2. **DFlash2 `feature_projection` (`Q8_G32_FP16` `[2560, 12800]`, `33.20 MB`) is missing from `kShapes` in `q8_dispatch.cpp`**:
   - In [`src/models/qwen4_exp/execution/dflash2.cpp`](file:///opt/etc/Infernix/src/models/qwen4_exp/execution/dflash2.cpp#L108), `DFlash2::materialize` calls `ops::linear(features, p_.feature_projection, ...)` on `N = 2560, K = 12800` (`33.20 MB` in `Qwen3.8-Flash-Next-DFlash2-Infernix.infernix`).
   - Because `{2560, 12800}` is absent from `kShapes` in [`src/ops/linear/q8/q8_dispatch.cpp`](file:///opt/etc/Infernix/src/ops/linear/q8/q8_dispatch.cpp#L43-L102), every DFlash2 draft materialization falls back to `select_q8_generic` (`launch_q8_a16_simt_r8_t4` for `T = 3..4`, `r8_t8` for `T = 5..32` across a huge `K = 12800` reduction!). Adding a GB10-tuned shape selector (with `Q8SlicedK` for `K = 12800`) directly accelerates `DFlash2::materialize`.

3. **Three NVFP4 `/shapes/` files (`n5120_k6144.cu`, `n5120_k17408.cu`, `n34816_k5120.cu`) still have `origin = unknown` (RTX 5090 tables)**:
   - `ckt shapes list --qtype nvfp4` shows 12/15 NVFP4 shapes have `origin = gb10`, while `n5120_k6144.cu`, `n5120_k17408.cu`, and `n34816_k5120.cu` (Qwen3.5-27B NVFP4 shapes) remain `origin = unknown` with RTX 5090 `pair_min_tokens = 128` / `split_k_min_tokens = 192` and uncalibrated `T = 1..64` tables. Retuning them with `infernix_linear_bench` on GB10 brings `src/ops/linear/nvfp4/shapes/` to 15/15 GB10 coverage.

#### Required Implementation Steps
1. **Sweep and wire `n1280_k2560.cu` (`[1280, 2560]`, `49x` shared-expert gate) on GB10**:
   - Benchmark `[1280, 2560]` across `T = 1..64` using `infernix_linear_bench --qtype Q8_G32_FP16 --n 1280 --k 2560` (comparing exact SIMT/MMA routes and `Q8SlicedK` schedules `SlicedK_T8W8S1M1RT1`, `SlicedK_T8W8S2M1RT2`, `SlicedK_T16W8S1M1RT1`, etc.).
   - Wire both `select_q8_n1280_k2560` (exact) and `select_q8_n1280_k2560_fast` (fast `Q8SlicedK`) in [`src/ops/linear/q8/shapes/n1280_k2560.cu`](file:///opt/etc/Infernix/src/ops/linear/q8/shapes/n1280_k2560.cu) and [`src/ops/linear/q8/q8_dispatch.cpp`](file:///opt/etc/Infernix/src/ops/linear/q8/q8_dispatch.cpp#L50), and add the `// GB10 (sm_121a) tuned dispatch` header comment so `ckt shapes list --qtype q8` reports `origin = gb10`.
2. **Add GB10-tuned dispatch for DFlash2 `feature_projection` (`Q8_G32_FP16` `[2560, 12800]`)**:
   - Create `src/ops/linear/q8/shapes/n2560_k12800.cu` (and register in `CMakeLists.txt`, `shapes.h`, and `q8_dispatch.cpp`) or wire `[2560, 12800]` in `q8_dispatch.cpp` with GB10-calibrated exact and `Q8SlicedK` fast routes across `T = 1..64` (especially `T = 1..2` for single/pair DFlash2 materialization and `T = 16..32`).
3. **Retune the 3 remaining `origin = unknown` NVFP4 shape files on GB10**:
   - Sweep [`src/ops/linear/nvfp4/shapes/n5120_k6144.cu`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n5120_k6144.cu), [`n5120_k17408.cu`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n5120_k17408.cu), and [`n34816_k5120.cu`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n34816_k5120.cu) with `infernix_linear_bench --qtype NVFP4`, update their `select_a16` tables and `pair_min_tokens`, and stamp their `// GB10 (sm_121a)` provenance headers.

#### Verification & Acceptance Criteria
```bash
# 1. Build bench, Q8 & NVFP4 oracle tests, and infernix-serve in a single invocation
ninja -C build-native -j "$(nproc)" \
  infernix_linear_bench \
  infernix_linear_q8_a16_test \
  infernix_linear_nvfp4_a16_test \
  infernix_linear_nvfp4_a4_test \
  infernix-serve

# 2. Verify 100% GB10 shape coverage in ckt shapes list
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt shapes list --qtype nvfp4
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt shapes list --qtype q8

# 3. Run oracle tests on all modified Q8 and NVFP4 shapes
./build-native/tests/infernix_linear_q8_a16_test --shape 1280 2560
./build-native/tests/infernix_linear_q8_a16_test --shape 2560 12800
./build-native/tests/infernix_linear_nvfp4_a16_test --shape 5120 6144
./build-native/tests/infernix_linear_nvfp4_a16_test --shape 5120 17408
./build-native/tests/infernix_linear_nvfp4_a16_test --shape 34816 5120

# 4. Benchmark before/after on [1280, 2560] and [2560, 12800] (Q8) and [5120, 6144], [5120, 17408], [34816, 5120] (NVFP4)
./build-native/bench/infernix_linear_bench --qtype Q8_G32_FP16 --n 1280 --k 2560 --sweep 1:64:1 --warmup 2 --iters 10 --routes fast
./build-native/bench/infernix_linear_bench --qtype Q8_G32_FP16 --n 2560 --k 12800 --sweep 1:32:1 --warmup 2 --iters 10 --routes fast

# 5. One-way CUDA stack-spill ratchet gate
ctest --test-dir build-native -R infernix_cuda_stack_check --output-on-failure
```
- **Acceptance Criteria**:
  1. `ckt shapes list --qtype nvfp4` and `ckt shapes list --qtype q8` report `origin = gb10` for **100%** of shape files (`15/15` NVFP4 and all Q8 shape files).
  2. `Q8_G32_FP16` `[1280, 2560]` (`49x` shared-expert gate) and `[2560, 12800]` (DFlash2 `feature_projection`) achieve measurable speedups across `T = 1..32` (`--routes fast` and `--routes exact`) and pass `infernix_linear_q8_a16_test`.
  3. All 3 retuned NVFP4 shapes (`[5120, 6144]`, `[5120, 17408]`, `[34816, 5120]`) pass `infernix_linear_nvfp4_a16_test` and `infernix_cuda_stack_check` with zero new spills.

---

### Task T5 (Priority P1): Wire `Qwen4Exp` Hybrid Prefix-Cache Cost Model (`set_hybrid_cost`) & GB10 Page-Setup Term for Multi-Turn NVFP4 Serving

#### Affected Files & Symbols
- [`src/runtime/engine/qwen4_exp_instance.cpp`](file:///opt/etc/Infernix/src/runtime/engine/qwen4_exp_instance.cpp#L65-L210)
  - [`Qwen4ExpInstance::Qwen4ExpInstance`](file:///opt/etc/Infernix/src/runtime/engine/qwen4_exp_instance.cpp#L65-L165) (populate `summary_.prefill_signature` and wire `set_hybrid_cost`)
- [`src/models/qwen4_exp/program.h`](file:///opt/etc/Infernix/src/models/qwen4_exp/program.h) / [`src/models/qwen4_exp/program/program.cpp`](file:///opt/etc/Infernix/src/models/qwen4_exp/program/program.cpp)
  - `Program::set_hybrid_cost` (expose or forward to the underlying hybrid prefix cache / context store)
- [`src/runtime/context_prefill_cost.h`](file:///opt/etc/Infernix/src/runtime/context_prefill_cost.h) / [`src/runtime/context_prefill_cost.cpp`](file:///opt/etc/Infernix/src/runtime/context_prefill_cost.cpp)
  - [`PrefillCostSignature`](file:///opt/etc/Infernix/src/runtime/context_prefill_cost.h) / [`ContextPrefillCost`](file:///opt/etc/Infernix/src/runtime/context_prefill_cost.h)
- [`tests/runtime/test_context_prefill_cost.cpp`](file:///opt/etc/Infernix/tests/runtime/test_context_prefill_cost.cpp)

#### Root Cause & Hardware Rationale
In `20261029-changelog.md` (`T1` §7 item 1 and `T5` §2 & §5), two architectural gaps in the hybrid prefix-cache cost model were identified for `Qwen4Exp` (`Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix`):

1. **`Qwen4ExpInstance` never populates `LoadSummary::prefill_signature` and never calls `set_hybrid_cost`**:
   - In [`src/runtime/engine/model_instance.cpp`](file:///opt/etc/Infernix/src/runtime/engine/model_instance.cpp#L139-L145) (`Qwen3_5ForCausalLM` / `Qwen3_5MoeForCausalLM`), the constructor populates `summary_.prefill_signature` and calls `program->set_hybrid_cost(cost)` whenever `--context-cost-table` or a built-in `ContextPrefillCost` preset matches.
   - In contrast, [`src/runtime/engine/qwen4_exp_instance.cpp`](file:///opt/etc/Infernix/src/runtime/engine/qwen4_exp_instance.cpp) leaves `summary_.prefill_signature` default-empty and never configures `ContextPrefillCost` on its `qwen4_exp::Program` / `HybridPrefixCache` instances! Consequently, `--context-cost-table` and `infernix-bench-prefill-cost` cannot calibrate or drive prefix-cache checkpoint retention/eviction cost decisions on `Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix`.

2. **Missing GB10 calibration and page-setup overhead representation**:
   - In `20261029-changelog.md` (`T5`), stale RTX 5090 presets were removed from `context_prefill_cost.cpp` because GB10 (`sm_121a`) prefill latencies are `3.6x–4.9x` higher in absolute nanoseconds and exhibit a `54%` cold-vs-warm prefix-setup gap at `P = 4096, U = 64` (`103.57 ms` cold vs `67.15 ms` warm).
   - Wiring `prefill_signature` and `set_hybrid_cost` for `Qwen4ExpInstance` and calibrating a GB10 `sm_121` cost preset for `Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix` (or extending `ContextPrefillCost` if needed so `max_abs_rel_error <= 0.15`) enables cost-aware hybrid GDN/KV snapshot selection during multi-turn and agentic NVFP4 serving.

#### Required Implementation Steps
1. **Populate `LoadSummary::prefill_signature` and wire `set_hybrid_cost` in `Qwen4ExpInstance`**:
   - Inspect [`src/runtime/engine/model_instance.cpp`](file:///opt/etc/Infernix/src/runtime/engine/model_instance.cpp#L115-L150) and [`src/runtime/engine/qwen4_exp_instance.cpp`](file:///opt/etc/Infernix/src/runtime/engine/qwen4_exp_instance.cpp#L65-L170).
   - Construct `summary_.prefill_signature` for `Qwen4ExpInstance` (encoding `gpu_arch`, `model_family = "qwen4_exp"`, layer counts, GDN/attention dimensions, `dense_qtype`, `kv_dtype`, etc.) and wire `set_hybrid_cost` through `qwen4_exp::Program` to its hybrid prefix cache / cost model when a built-in preset or `--context-cost-table` matches.
2. **Calibrate GB10 (`sm_121`) `ContextPrefillCost` for `Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix`**:
   - Run `infernix-bench-prefill-cost` on `/opt/etc/models/Qwen3.8-Flash-Next-F4-Local-NVPF4-Dense/Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix` (`--kv-dtype fp8_e4m3` and `bf16`).
   - If the 5-parameter linear fit exceeds `max_abs_rel_error <= 0.15` due to warm vs cold page-table setup overhead on GB10, refine the fit / feature weighting in `ContextPrefillCost` and register the verified GB10 (`arch = "sm_121"`) preset in [`src/runtime/context_prefill_cost.cpp`](file:///opt/etc/Infernix/src/runtime/context_prefill_cost.cpp).
3. **Update `tests/runtime/test_context_prefill_cost.cpp`**:
   - Add unit tests verifying `Qwen4Exp` `PrefillCostSignature` matching, JSON serialization/deserialization, and GB10 `sm_121` preset lookup.

#### Verification & Acceptance Criteria
```bash
# 1. Build runtime cost tests, prefill-cost benchmark tool, and infernix-serve in one invocation
ninja -C build-native -j "$(nproc)" \
  infernix_context_prefill_cost_test \
  infernix-bench-prefill-cost \
  infernix-serve

# 2. Run context prefill cost unit tests
./build-native/tests/infernix_context_prefill_cost_test

# 3. Calibrate / verify on Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix
./build-native/bench/infernix-bench-prefill-cost \
  --model /opt/etc/models/Qwen3.8-Flash-Next-F4-Local-NVPF4-Dense/Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix \
  --kv-dtype fp8_e4m3

# 4. One-way CUDA stack-spill ratchet gate
ctest --test-dir build-native -R infernix_cuda_stack_check --output-on-failure
```
- **Acceptance Criteria**:
  1. `Qwen4ExpInstance` populates `LoadSummary::prefill_signature` and applies matching `ContextPrefillCost` tables (`--context-cost-table` and built-in `sm_121` presets).
  2. `infernix-bench-prefill-cost` runs cleanly on `Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix` (`fp8_e4m3` KV) and the registered GB10 preset satisfies `r2 >= 0.99` across its calibrated domain.
  3. `infernix_context_prefill_cost_test` and `infernix_cuda_stack_check` pass 100%.
