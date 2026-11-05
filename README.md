# Infernix — Retuned for NVIDIA DGX Spark (`NVIDIA GB10`, `sm_121a`)

**Infernix** is a high-performance, from-scratch C++20/CUDA inference engine for the Qwen3.5, Qwen3.6, Qwen3.8, and **Qwen3.8-Flash-Next** (`Qwen4ExpForCausalLM`) model families.

> **Project Lineage & DGX Spark (`GB10`) Retune:** This project is built on the **Infernix** inference engine (which originally began as a fork of [NInfer](https://github.com/Neroued/ninfer) by [Neroued](https://github.com/Neroued) targeting the desktop RTX 5090 `sm_120a`) and has been **architecturally retuned end-to-end for the NVIDIA DGX Spark (`NVIDIA GB10`, Blackwell `sm_121a`)** on 64-bit Linux `aarch64`. Every CUDA kernel family, `/shapes/` dispatch table (`103/103` shapes), TMA descriptor policy, L2 cache swizzle, unified-memory (UMA) expert placement, and multi-row Multi-Token Prediction (MTP) speculative scheduler has been calibrated specifically for the GB10's **48 SMs**, **24 MiB L2 cache**, **128 GB unified LPDDR5X memory**, and **20-core Grace ARMv9.2 CPU**.

---

## Key Architectural Highlights on NVIDIA DGX Spark (`GB10`)

### 1. Full-Model Unified Memory Residency for `Qwen3.8-Flash-Next`
`Qwen3.8-Flash-Next` has 24,576 routed experts (63–68 GiB in NVFP4 / Dense8, 46.9 GiB in IQ3_S) plus a 29–52 GiB PLE n-gram embedding volume. On a 32 GB desktop GPU, experts must stream across PCIe on every token. On **DGX Spark's 128 GB unified LPDDR5X**, Infernix pins all 48 layers of routed experts directly in device-accessible unified memory (`INFERNIX_EXPERTS_ON_GPU=1`, zero PCIe miss transfers) and streams the PLE n-gram volume asynchronously from NVMe with a host cache and 20-thread ARMv9.2 prefill workers:
- **3,238–3,371 tok/s prompt prefill (`pp4096`)** on `Qwen3.8-Flash-Next-NVFP4` across `bf16`, `fp8`, `int8`, `k8v4`, and `nvfp4` KV caches (up **2.6×** over the initial un-tuned baseline, peaking at **3,371.4 tok/s** with `nvfp4` KV).
- **98.0–105.7 tok/s single-stream decode (`pp4096+tg256`)** with MTP speculative decoding + `--lm-head-draft` (`0.905–1.000` draft acceptance rate), and **45.4–46.2 tok/s** plain non-speculative decode (`88–96%` of GB10's physical LPDDR5X read bandwidth ceiling, peaking at **46.2 tok/s** with `nvfp4` KV and **46.1 tok/s** with `k8v4` KV).
- **179.4 tok/s aggregate multi-request decode (`c = 4`, `44.9 tok/s` per stream)** and **125.0–127.1 tok/s at `c = 2` (`62.5–63.5 tok/s` per stream)** via multi-row batched MTP speculation.

### 2. Microarchitectural Retuning for Blackwell `sm_121a` (48 SMs, 24 MiB L2)
- **Native 2-Term Row-Scaled `NVFP4 A4` TMA (`OMMA.16864.F32.E2M1`, `501.6 TFLOP/s` pipe)**: Trunk and MoE projections dynamically quantize BF16 activations into two per-token row-scaled NVFP4 terms (`FastPath::Nvfp4A4Pair`) using 1-pass register-cached reciprocals and 2-CTA/SM `128×64` TMA tiles with `CU_TENSOR_MAP_L2_PROMOTION_L2_128B`, reaching **315–350 hardware `OMMA` TFLOP/s** (`174 useful TFLOP/s`, **2.20×** faster than `W4A16`).
- **24 MiB L2 Cache Thrashing Elimination**: Added 2D `Grouped, 8` CTA rasterization across all `BF16`, `FP8`, `Q8`, `NVFP4`, `Q6`, `Q5`, and `Q4` MMA kernels (`95.2–100.1 TFLOP/s` on `BF16`, **100%** of GB10's dense BF16 Tensor Core ceiling; `77.4–83.5 TFLOP/s` on `Q8`, **86%** of ceiling).
- **Direct FP32 Tensor Core Accumulation (`1:1` HMMA Parity on `sm_121a`)**: Unlike `sm_120a` (where FP32-accumulating HMMA is half-rate relative to FP16-accumulating HMMA), GB10 `sm_121a` has **1:1 throughput parity (`125.31` vs `125.44 TFLOP/s`)**. Replacing `mma_f16_f16acc` with direct FP32-accumulating `mma_f16` across dense causal attention (`fp8`, `k8v4`, `nvfp4`, `int8`, `vq2`) and Gated Delta Net (`chunked/recurrence.cu`) eliminated **100% of local-memory stack spills (`STACK = 0`)** while improving numerical precision.
- **Hardware `F2FP` + `HMUL2` Bit-Exact NVFP4 Codec & 2×/3× Occupancy Narrow MoE**: Replaced software LUT dequantization with hardware `__nv_cvt_fp4x2_to_halfraw2` + `__hmul2` (`0 ULP` exact in IEEE FP16), shrank `Nvfp4SlicedInstance<16, 8, 2, 2>` to `48.0 KiB` static SMEM (`2 CTAs/SM`, `208–230 GB/s` at `T = 2..32`), and refactored narrow MoE (`moe_layer.cu`) to quantize `A16`/`A4` blocks directly in registers (`2 CTAs/SM` on `gate_up_kernel_a16`, `3 CTAs/SM` on `down_kernel` and `down_kernel_a16`, `STACK = 0`).

### 3. Production Serving & Agentic Capabilities
- **OpenAI & Anthropic APIs**: `/v1/chat/completions`, `/v1/responses`, `/v1/messages`, and `/v1/decide` with streaming, reasoning budgets (`--reasoning-loop`), tolerant XML/JSON tool-call recovery (`--tolerant-tool-calls`), and constrained decoding (JSON Schema, regex, GBNF, choice grammars).
- **Hybrid Prefix Cache**: Content-addressed 64-token KV blocks plus recurrent GDN/conv state snapshots across GPU, host RAM, and persistent disk tiers (`--prefix-cache-save-mins`).
- **Seven KV Cache Formats**: `bf16`, `fp8`, `int8`, `k8v4`, `nvfp4`, `k4v2`, and `vq2` (holding 3–4× the context of INT8).

---

## Benchmarks on NVIDIA DGX Spark (`NVIDIA GB10`)

### Benchmark Environment

All benchmarks below were measured directly on the **NVIDIA DGX Spark** system with the following hardware and software stack:

| Component | Specification |
|---|---|
| **System / Device** | **NVIDIA DGX Spark** |
| **GPU** | **NVIDIA GB10** (Blackwell `sm_121a`, Compute Capability `12.1`, **48 SMs** @ up to `3,003 MHz`, **24 MiB L2 cache**) |
| **Memory Subsystem** | **128 GB Unified LPDDR5X** (`cudaDevAttrIntegrated = 1`, `~121 GiB` usable; `273.0 GB/s` theoretical peak, `247.6 GB/s` measured sustained read) |
| **CPU** | **20-core NVIDIA Grace ARMv9.2 (`aarch64`)** — `10× Cortex-X925` (`r0p1`) + `10× Cortex-A725` (`r0p1`), `48 MiB` L3 cache (`2 × 24 MiB`) |
| **Operating System** | **Ubuntu 24.04.5 LTS** (`Noble Numbat`, 64-bit `aarch64`) |
| **Linux Kernel** | **`7.0.0-1019-nvidia`** (`#19-Ubuntu SMP PREEMPT_DYNAMIC`) |
| **NVIDIA GPU Driver** | **`580.178.04`** |
| **CUDA Toolkit** | **CUDA `13.0`** (`release 13.0, V13.0.88`, `nvcc` build `cuda_13.0.r13.0/compiler.36424714_0`) |
| **Host C++ Compiler** | **GCC `13.3.0`** (`Ubuntu 13.3.0-6ubuntu2~24.04.1`) |

---

### 1. Single-Request Throughput Across All 7 KV Cache Formats (`Qwen3.8-Flash-Next-NVFP4`)

Measured via `./bench/run_kv_matrix_bench.sh` (`infernix_bench`) on `Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix` (`INFERNIX_EXPERTS_ON_GPU=1`, `--ngram-residency volume`, `--prefill-chunk 16384`, `c = 1`).
- **`plain` mode**: Exact non-speculative greedy decoding (`1` token/round).
- **`mtp` mode**: Multi-Token Prediction speculative decoding (`--spec mtp --draft-tokens auto --lm-head-draft`), where every draft token is mathematically verified by the target model.

| Configuration (`Model_KV_Mode`) | Benchmark | Prefill (`tok/s`) | Decode (`tok/s`) | MTP Acceptance Rate |
|---|---|---:|---:|---:|
| **`nvfp4_bf16_mtp`** | `pp4096` | **3,238.2** | — | — |
| **`nvfp4_bf16_mtp`** | `tg128` | — | **59.8** | `0.583` |
| **`nvfp4_bf16_mtp`** | `tg256` | — | **72.3** | `0.710` |
| **`nvfp4_bf16_mtp`** | `pp4096+tg128` | **3,224.1** | **96.8** | `0.962` |
| **`nvfp4_bf16_mtp`** | `pp4096+tg256` | **3,226.0** | **98.0** | `0.905` |
| **`nvfp4_bf16_plain`** | `pp4096` | **3,241.9** | — | — |
| **`nvfp4_bf16_plain`** | `tg128` | — | **45.7** | — |
| **`nvfp4_bf16_plain`** | `tg256` | — | **45.7** | — |
| **`nvfp4_bf16_plain`** | `pp4096+tg128` | **3,243.6** | **44.6** | — |
| **`nvfp4_bf16_plain`** | `pp4096+tg256` | **3,234.2** | **44.6** | — |
| **`nvfp4_fp8_mtp`** | `pp4096` | **3,267.5** | — | — |
| **`nvfp4_fp8_mtp`** | `tg128` | — | **57.8** | `0.560` |
| **`nvfp4_fp8_mtp`** | `tg256` | — | **63.2** | `0.561` |
| **`nvfp4_fp8_mtp`** | `pp4096+tg128` | **3,242.9** | **102.4** | `0.946` |
| **`nvfp4_fp8_mtp`** | `pp4096+tg256` | **3,262.8** | **105.7** | `0.947` |
| **`nvfp4_fp8_plain`** | `pp4096` | **3,293.7** | — | — |
| **`nvfp4_fp8_plain`** | `tg128` | — | **45.7** | — |
| **`nvfp4_fp8_plain`** | `tg256` | — | **45.6** | — |
| **`nvfp4_fp8_plain`** | `pp4096+tg128` | **3,285.2** | **44.6** | — |
| **`nvfp4_fp8_plain`** | `pp4096+tg256` | **3,275.6** | **44.6** | — |
| **`nvfp4_int8_mtp`** | `pp4096` | **3,311.3** | — | — |
| **`nvfp4_int8_mtp`** | `tg128` | — | **60.0** | `0.505` |
| **`nvfp4_int8_mtp`** | `tg256` | — | **57.6** | `0.467` |
| **`nvfp4_int8_mtp`** | `pp4096+tg128` | **3,306.7** | **92.7** | `0.899` |
| **`nvfp4_int8_mtp`** | `pp4096+tg256` | **3,276.2** | **101.5** | `0.928` |
| **`nvfp4_int8_plain`** | `pp4096` | **3,327.2** | — | — |
| **`nvfp4_int8_plain`** | `tg128` | — | **45.4** | — |
| **`nvfp4_int8_plain`** | `tg256` | — | **45.7** | — |
| **`nvfp4_int8_plain`** | `pp4096+tg128` | **3,303.7** | **44.6** | — |
| **`nvfp4_int8_plain`** | `pp4096+tg256` | **3,322.1** | **44.6** | — |
| **`nvfp4_k8v4_mtp`** | `pp4096` | **3,300.9** | — | — |
| **`nvfp4_k8v4_mtp`** | `tg128` | — | **64.3** | `0.746` |
| **`nvfp4_k8v4_mtp`** | `tg256` | — | **67.5** | `0.697` |
| **`nvfp4_k8v4_mtp`** | `pp4096+tg128` | **3,275.6** | **99.1** | `1.000` |
| **`nvfp4_k8v4_mtp`** | `pp4096+tg256` | **3,283.7** | **101.0** | `0.923` |
| **`nvfp4_k8v4_plain`** | `pp4096` | **3,307.9** | — | — |
| **`nvfp4_k8v4_plain`** | `tg128` | — | **46.1** | — |
| **`nvfp4_k8v4_plain`** | `tg256` | — | **46.1** | — |
| **`nvfp4_k8v4_plain`** | `pp4096+tg128` | **3,312.3** | **45.0** | — |
| **`nvfp4_k8v4_plain`** | `pp4096+tg256` | **3,299.7** | **45.1** | — |
| **`nvfp4_nvfp4_mtp`** | `pp4096` | **3,337.3** | — | — |
| **`nvfp4_nvfp4_mtp`** | `tg128` | — | **57.3** | `0.650` |
| **`nvfp4_nvfp4_mtp`** | `tg256` | — | **68.8** | `0.757` |
| **`nvfp4_nvfp4_mtp`** | `pp4096+tg128` | **3,324.0** | **82.9** | `0.890` |
| **`nvfp4_nvfp4_mtp`** | `pp4096+tg256` | **3,330.3** | **92.0** | `0.866` |
| **`nvfp4_nvfp4_plain`** | `pp4096` | **3,371.4** | — | — |
| **`nvfp4_nvfp4_plain`** | `tg128` | — | **46.2** | — |
| **`nvfp4_nvfp4_plain`** | `tg256` | — | **46.0** | — |
| **`nvfp4_nvfp4_plain`** | `pp4096+tg128` | **3,347.2** | **45.1** | — |
| **`nvfp4_nvfp4_plain`** | `pp4096+tg256` | **3,321.9** | **45.2** | — |
| **`nvfp4_k4v2_mtp`** | `pp4096` | **3,053.7** | — | — |
| **`nvfp4_k4v2_mtp`** | `tg128` | — | **58.8** | `0.551` |
| **`nvfp4_k4v2_mtp`** | `tg256` | — | **67.9** | `0.636` |
| **`nvfp4_k4v2_mtp`** | `pp4096+tg128` | **3,020.8** | **89.0** | `0.925` |
| **`nvfp4_k4v2_mtp`** | `pp4096+tg256` | **3,023.0** | **93.1** | `0.911` |
| **`nvfp4_k4v2_plain`** | `pp4096` | **3,065.0** | — | — |
| **`nvfp4_k4v2_plain`** | `tg128` | — | **45.2** | — |
| **`nvfp4_k4v2_plain`** | `tg256` | — | **45.0** | — |
| **`nvfp4_k4v2_plain`** | `pp4096+tg128` | **3,059.5** | **44.2** | — |
| **`nvfp4_k4v2_plain`** | `pp4096+tg256` | **3,025.6** | **44.1** | — |
| **`nvfp4_vq2_mtp`** | `pp4096` | **3,143.5** | — | — |
| **`nvfp4_vq2_mtp`** | `tg128` | — | **59.5** | `0.562` |
| **`nvfp4_vq2_mtp`** | `tg256` | — | **66.5** | `0.596` |
| **`nvfp4_vq2_mtp`** | `pp4096+tg128` | **3,130.0** | **56.1** | `0.657` |
| **`nvfp4_vq2_mtp`** | `pp4096+tg256` | **3,118.5** | **52.8** | `0.539` |
| **`nvfp4_vq2_plain`** | `pp4096` | **3,165.6** | — | — |
| **`nvfp4_vq2_plain`** | `tg128` | — | **45.2** | — |
| **`nvfp4_vq2_plain`** | `tg256` | — | **45.0** | — |
| **`nvfp4_vq2_plain`** | `pp4096+tg128` | **3,150.3** | **44.3** | — |
| **`nvfp4_vq2_plain`** | `pp4096+tg256` | **3,154.1** | **44.3** | — |

---

### 2. Concurrent Multi-Request Scaling (`c = 1, 2, 3, 4`) on DGX Spark (`GB10`)

Measured on `Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix` (`--kv-dtype fp8 --spec mtp --draft-tokens auto --lm-head-draft`, `pp2048+tg128`) after enabling multi-row batched MTP speculation (`batch = 2..4`), register-cached `A16`/`A4` narrow MoE (`2–3 CTAs/SM`), and `round_cost_fit` truncation protection:

| Concurrency (`c`) | Infernix GB10 Per-Stream Decode (`tok/s`) | Infernix GB10 Aggregate Decode (`tok/s`) | Infernix GB10 Peak Aggregate (`tok/s`) | Reference Framework on GB10 (Median / Peak `tok/s`) | Infernix Gain vs. Reference Framework |
|---:|---:|---:|---:|---:|---:|
| **`c = 1`** | **93.65 ± 0.46** (`105.7` on `pp4096+tg256`) | **93.65** (`105.7` on `pp4096+tg256`) | **94.11** | `69.53` / `69.91` (`69.58` per-stream) | **+34.6%** (up to **+51.2%** on `pp4096`) |
| **`c = 2`** | **62.49 ± 2.32** | **124.98** | **127.06** (`63.53`/stream) | `114.02` / `115.28` (`56.93` per-stream) | **+9.6% median / +10.2% peak** |
| **`c = 3`** | **50.41 ± 0.83** | **151.23** | **154.06** (`51.35`/stream) | `138.51` / `149.00` (`46.12` per-stream) | **+9.2% median / +3.4% peak** |
| **`c = 4`** | **44.86 ± 1.15** | **179.44** | **184.04** (`46.01`/stream) | `175.41` / `177.86` (`43.95` per-stream) | **+2.3% median / +3.5% peak** (**+94.6%** over un-tuned `c=4`) |

#### OpenAI HTTP Server (`infernix-serve`) Context-Depth Scaling (`llama-benchy`)

`Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix` — launch via `./tools/run_server.sh` (no extra args):

```
INFERNIX_EXPERTS_ON_GPU=1 infernix-serve \
  Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix \
  --ngram-volume Qwen3.8-Flash-Next-Local-NVFP4.ngram \
  --ngram-residency volume --ngram-cache-mib 4096 \
  --kv-dtype nvfp4 \
  --spec mtp --draft-tokens auto --lm-head-draft \
  --ngram-draft-tokens 31 --ngram-min-match 4 \
  --temperature 1.0 --top-p 0.9 --min-p 0.1 --top-k 20 \
  --max-context 262144 --max-concurrency 4
```

`llama-benchy` `pp2048 tg128` × prefix depths `d0 / d4096 / d8192` × `c = 1, 2, 4`, `temperature = 1.0`, 2026-11-05:

| Test | `c` | `pp t/s` | `tg t/s` | TTFT (ms) | Total (ms) |
|---|---:|---:|---:|---:|---:|
| `pp2048 tg128 @ d0` | `c1` | **3,038\*** | **71.7** | 674 | 2,446 |
| `pp2048 tg128 @ d0` | `c2` | 2,863 | **95.8** | 1,352 | 3,894 |
| `pp2048 tg128 @ d0` | `c4` | 2,869 | **132.6** | 2,634 | 6,229 |
| `pp2048 tg128 @ d4096` | `c1` | **3,826\*** | **70.2** | 1,606 | 3,414 |
| `pp2048 tg128 @ d4096` | `c2` | 3,613 | **88.7** | 3,119 | 5,720 |
| `pp2048 tg128 @ d4096` | `c4` | 3,624 | **128.4** | 6,213 | 9,710 |
| `pp2048 tg128 @ d8192` | `c1` | **4,355\*** | **69.8** | 2,351 | 4,170 |
| `pp2048 tg128 @ d8192` | `c2` | 4,010 | **102.8** | 4,627 | 7,051 |
| `pp2048 tg128 @ d8192` | `c4` | 4,028 | **118.4** | 9,310 | 13,221 |

\* `c = 1` prefill column is an estimator output, not wall-clock throughput — only `e2e_ttft`-derived rates are meaningful at `c = 1`.

`Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix` — same launcher and flags with `--kv-dtype int8`:

```
INFERNIX_EXPERTS_ON_GPU=1 infernix-serve \
  Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix \
  --ngram-volume Qwen3.8-Flash-Next-Local-NVFP4.ngram \
  --ngram-residency volume --ngram-cache-mib 4096 \
  --kv-dtype int8 \
  --spec mtp --draft-tokens auto --lm-head-draft \
  --ngram-draft-tokens 31 --ngram-min-match 4 \
  --temperature 1.0 --top-p 0.9 --min-p 0.1 --top-k 20 \
  --max-context 262144 --max-concurrency 4
```

`llama-benchy` `pp2048 tg128` × prefix depths `d0 / d4096 / d8192` × `c = 1, 2, 4`, `temperature = 1.0`, 2026-11-05:

| Test | `c` | `pp t/s` | `tg t/s` | TTFT (ms) | Total (ms) |
|---|---:|---:|---:|---:|---:|
| `pp2048 tg128 @ d0` | `c1` | **2,978\*** | **83.0** | 688 | 2,217 |
| `pp2048 tg128 @ d0` | `c2` | 2,862 | **109.4** | 1,341 | 3,583 |
| `pp2048 tg128 @ d0` | `c4` | 2,929 | **140.2** | 2,631 | 6,038 |
| `pp2048 tg128 @ d4096` | `c1` | **3,921\*** | **74.7** | 1,567 | 3,266 |
| `pp2048 tg128 @ d4096` | `c2` | 3,617 | **108.7** | 3,067 | 5,353 |
| `pp2048 tg128 @ d4096` | `c4` | 3,645 | **141.4** | 6,187 | 9,446 |
| `pp2048 tg128 @ d8192` | `c1` | **4,335\*** | **73.1** | 2,362 | 4,100 |
| `pp2048 tg128 @ d8192` | `c2` | 3,968 | **101.1** | 4,712 | 7,118 |
| `pp2048 tg128 @ d8192` | `c4` | 3,988 | **139.7** | 9,376 | 12,781 |

`Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix-00001-of-00003.infernix` — same launcher and flags with `--kv-dtype int8` and the volume n-gram:

```
INFERNIX_EXPERTS_ON_GPU=1 infernix-serve \
  Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix-00001-of-00003.infernix \
  --ngram-volume Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram \
  --ngram-residency volume --ngram-cache-mib 4096 \
  --kv-dtype int8 \
  --spec mtp --draft-tokens auto --lm-head-draft \
  --ngram-draft-tokens 31 --ngram-min-match 4 \
  --temperature 1.0 --top-p 0.9 --min-p 0.1 --top-k 20 \
  --max-context 262144 --max-concurrency 4
```

`llama-benchy` `pp2048 tg128` × prefix depths `d0 / d4096 / d8192` × `c = 1, 2, 4`, `temperature = 1.0`, 2026-11-05:

| Test | `c` | `pp t/s` | `tg t/s` | TTFT (ms) | Total (ms) |
|---|---:|---:|---:|---:|---:|
| `pp2048 tg128 @ d0` | `c1` | **2,612\*** | **64.9** | 784 | 2,740 |
| `pp2048 tg128 @ d0` | `c2` | 2,473 | **84.5** | 1,575 | 4,287 |
| `pp2048 tg128 @ d0` | `c4` | 2,493 | **126.8** | 3,094 | 6,904 |
| `pp2048 tg128 @ d4096` | `c1` | **3,206\*** | **62.9** | 1,916 | 3,937 |
| `pp2048 tg128 @ d4096` | `c2` | 3,065 | **93.4** | 3,650 | 6,299 |
| `pp2048 tg128 @ d4096` | `c4` | 3,109 | **124.7** | 7,246 | 10,984 |
| `pp2048 tg128 @ d8192` | `c1` | **3,569\*** | **54.0** | 2,869 | 5,222 |
| `pp2048 tg128 @ d8192` | `c2` | 3,258 | **90.0** | 5,750 | 8,483 |
| `pp2048 tg128 @ d8192` | `c4` | 3,221 | **126.3** | 11,621 | 15,399 |

`Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix-00001-of-00002.infernix` — same launcher and flags with `--kv-dtype int8`, the RAM-pinned n-gram, and `--max-context 131072` (at `262144` the weights + 27.8 GiB pinned n-gram + host cache claim exceed GB10's unified-memory ledger at startup):

```
INFERNIX_EXPERTS_ON_GPU=1 infernix-serve \
  Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix-00001-of-00002.infernix \
  --ngram-volume Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix.ngram \
  --ngram-residency ram \
  --kv-dtype int8 \
  --spec mtp --draft-tokens auto --lm-head-draft \
  --ngram-draft-tokens 31 --ngram-min-match 4 \
  --temperature 1.0 --top-p 0.9 --min-p 0.1 --top-k 20 \
  --max-context 262144 --max-concurrency 4
```

`llama-benchy` `pp2048 tg128` × prefix depths `d0 / d4096 / d8192` × `c = 1, 2, 4`, `temperature = 1.0`, 2026-11-05:

| Test | `c` | `pp t/s` | `tg t/s` | TTFT (ms) | Total (ms) |
|---|---:|---:|---:|---:|---:|
| `pp2048 tg128 @ d0` | `c1` | **1,406\*** | **63.4** | 1,457 | 3,460 |
| `pp2048 tg128 @ d0` | `c2` | 1,318 | **84.5** | 2,972 | 5,850 |
| `pp2048 tg128 @ d0` | `c4` | 1,305 | **112.6** | 5,846 | 10,196 |
| `pp2048 tg128 @ d4096` | `c1` | **1,670\*** | **68.4** | 3,679 | 5,537 |
| `pp2048 tg128 @ d4096` | `c2` | 1,538 | **83.9** | 7,334 | 10,274 |
| `pp2048 tg128 @ d4096` | `c4` | 1,539 | **104.7** | 14,644 | 19,022 |
| `pp2048 tg128 @ d8192` | `c1` | **1,758\*** | **64.1** | 5,825 | 7,808 |
| `pp2048 tg128 @ d8192` | `c2` | 1,605 | **90.1** | 11,653 | 14,390 |
| `pp2048 tg128 @ d8192` | `c4` | 1,605 | **106.1** | 23,421 | 27,954 |

---

### 3. Cross-Artifact Comparison on DGX Spark (`NVFP4_W4Dense` vs `Dense8` vs `IQ3_S`)

| Model Artifact (`bf16` KV, `MTP + --lm-head-draft`) | `pp4096` Prefill (`tok/s`) | `pp4096+tg128` Decode (`tok/s`) | `pp4096+tg256` Decode (`tok/s`) | `tg128` Decode (`tok/s`) | `tg256` Decode (`tok/s`) |
|---|---:|---:|---:|---:|---:|
| **`Qwen3.8-Flash-Next-Local-NVFP4-W4Dense`** | **3,238.9** | **97.7** (`102.4` w/ `fp8` KV) | **101.6** (`105.7` w/ `fp8` KV) | **62.5** | **71.3** |
| **`Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8`** | **2,866.2** | **88.2** | **69.5** | **62.4** | **70.2** |
| **`Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S`** | **1,045.0** | **98.9** | **81.5** | **74.8** | **82.9** |

---

## Numerical Quality & Fast-Path Gate

Infernix provides two numerical execution routes:

- **The exact route (`--disable-fast-paths`)**: Produces bit-identical output from run to run and release to release. Where an artifact's formats allow it, it is also request-invariant (concurrent requests, prefix-cache state, and memory pressure do not alter a request's output).
- **Fast paths (enabled by default)**: Hardware-accelerated schedules and dynamic activation quantizations (`nvfp4-a4-pair`, `qsa-prepare-fused`, `gdn-f16-accumulate`, `sampled-mtp-drafts`, `adaptive-speculation`) that run only for the kernels they were qualified on, only after passing the numerical quality gate below, and only where measured faster on GB10.

### The Quality Gate
Every fast path is qualified against a high-precision reference (16-bit activations in every layer and a BF16 KV cache) over a 1.04M-token teacher-forced prefill corpus and 1,024 forced decoding segments (65,536 scored positions):
- **Shared budgets (at the 95% upper confidence bound)**: KL divergence $\le 1.5 \times 10^{-3}$, top-1 token agreement within $0.5$ percentage points, decode total variation (TV) $\le 4 \times 10^{-3}$, and 99.9th-percentile KL growth $\le 25\%$.
- Full specification: [Numerical Acceptance Policy (`docs/maintainer/model-quality.md`)](docs/maintainer/model-quality.md#6-numerical-acceptance-policy-and-fast-paths) and [GB10 Optimization Roadmap (`docs/gb10_roadmap.md`)](docs/gb10_roadmap.md).

---

## Quick Start (DGX Spark / Linux `aarch64`)

### 1. Prerequisites
Builds and runs natively on **64-bit Linux `aarch64`** (Ubuntu 24.04 LTS, CUDA 13.0, GCC 13.3):

```bash
sudo apt-get update && sudo apt-get install -y \
  build-essential cmake ninja-build pkg-config \
  libavformat-dev libavcodec-dev libavutil-dev libswscale-dev \
  libcurl4-openssl-dev
```

### 2. Build
```bash
cmake -B build-native -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=121a
ninja -C build-native -j "$(nproc)" infernix-serve infernix_bench
```

### 3. Run the Server (`infernix-serve`)
Use [`tools/run_server.sh`](tools/run_server.sh) or launch `build-native/apps/infernix-serve` directly:

```bash
INFERNIX_EXPERTS_ON_GPU=1 ./build-native/apps/infernix-serve \
  --weights /opt/etc/models/Qwen3.8-Flash-Next-F4-Local-NVPF4-Dense/Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix \
  --ngram-volume /opt/etc/models/Qwen3.8-Flash-Next-F4-Local-NVPF4-Dense/Qwen3.8-Flash-Next-Local-NVFP4.ngram \
  --ngram-residency volume \
  --kv-dtype fp8 \
  --spec mtp --draft-tokens auto --lm-head-draft \
  --prefill-chunk 4096 \
  --max-batch 4 --max-context 65536 \
  --host 0.0.0.0 --port 8080
```

---

## Supported Models

Infernix loads `.infernix` v3 containers as well as NInfer `.ninfer` artifacts. Use `python3 -m tools.convert` ([weight conversion guide](docs/weight-conversion.md)) to convert checkpoints from Hugging Face, NVIDIA ModelOpt, Quasar, or GGUF.

### Qwen3.8-Flash-Next (`Qwen4ExpForCausalLM`)

| Artifact / Repository | Source Checkpoint | Description |
|---|---|---|
| **`Qwen3.8-Flash-Next-Local-NVFP4-W4Dense`** / [Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix) (**recommended on GB10**) | [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) | NVIDIA's NVFP4 trunk and routed experts with FP8 PLE n-gram table and proposal head (`--lm-head-draft`). Executes trunk projections on GB10's 501.6 TFLOP/s FP4 Tensor Cores via `FastPath::Nvfp4A4Pair` (**3,238–3,327 tok/s** `pp4096`, **105.7 tok/s** `c=1` MTP decode, **179.4 tok/s** `c=4` decode). |
| [Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix) | [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) | NVIDIA's NVFP4 experts (`W4A4`) with 8-bit (`Q8_G32_FP16`) dense trunk projections (**2,866 tok/s** `pp4096`, **88.2 tok/s** `pp4096+tg128` MTP decode on GB10). |
| [Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix) | [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) (`IQ3_S`) | 3-bit codebook quantization with `IQ4_NL` n-gram volume (46.9 GiB expert footprint; fits alongside its 29.1 GiB n-gram table entirely in GB10 RAM). |
| [Qwen3.8-Flash-Next-FP8-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-FP8-Infernix) | [Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) | Bit-exact conversion of Qwen's FP8 checkpoint with BF16 n-gram volume. |
| [Qwen3.8-Flash-Next-Uncensored-NVFP4-Infernix](https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-Uncensored-NVFP4-Infernix) | [orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4) | Abliterated NVFP4 build with bit-exact weight preservation. |
| [Swift-1.5-Qwen3.8-Flash-Next-NVFP4-Dense8-Infernix](https://huggingface.co/Wallawalla47/Swift-1.5-Qwen3.8-Flash-Next-NVFP4-Dense8-Infernix) | [ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4) | Token-efficient reasoning fine-tune in Dense8 layout. |

### Qwen3.8-27B & Qwen3.6 (`Qwen3_5ForCausalLM` / `Qwen3_5MoeForCausalLM`)

| Repository | Source Checkpoint | Description |
|---|---|---|
| **[Qwen3.8-27B-NVIDIA-NVFP4-NInferV3](https://huggingface.co/Wallawalla47/Qwen3.8-27B-NVIDIA-NVFP4-NInferV3)** | [nvidia/Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4) | NVIDIA's NVFP4 checkpoint with DFlash2 drafter and `--lm-head-draft` proposal head. |
| [Qwen3.8-27B-Quasar-NVFP4-NInferV3](https://huggingface.co/Wallawalla47/Qwen3.8-27B-Quasar-NVFP4-NInferV3) | [QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4) | QAT-trained NVFP4 checkpoint with DFlash2 drafter and proposal head. |
| [Qwen3.8-27B-Uncensored-NVFP4-NInferV3](https://huggingface.co/Wallawalla47/Qwen3.8-27B-Uncensored-NVFP4-NInferV3) | [orcarouter/Qwen3.8-27B-Uncensored-NVFP4](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-NVFP4) | Abliterated NVFP4 checkpoint with DFlash2 drafter and proposal head. |
| [neroued/Qwen3.8-27B-nvfp4-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | Official NInfer NVFP4 artifact. |
| [neroued/Qwen3.6-27B-nvfp4-NInfer](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) | Official NInfer Qwen3.6-27B NVFP4 artifact. |
| [neroued/Qwen3.6-35B-A3B-NInfer](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | Official NInfer `groupwise-int` MoE artifact with DFlash drafter. |

---

## Documentation

- **DGX Spark (`GB10`) Engineering & Performance**:
  - [DGX Spark (`NVIDIA GB10`) Optimization Roadmap & Hardware Ceiling Audit](docs/gb10_roadmap.md)
  - [Performance Overview](docs/performance.md)
- **User & Product Guides**:
  - [CLI Guide](docs/cli.md)
  - [HTTP Serving & API Reference](docs/serving.md)
  - [Qwen3.8-Flash-Next Guide](docs/qwen3_8-flash-next.md)
  - [Perplexity & Quality Evaluation](docs/perplexity.md)
  - [Weight Conversion Guide](docs/weight-conversion.md)
- **Maintainer Architecture**:
  - [Documentation Map](docs/README.md)
  - [Engine Architecture](docs/maintainer/engine-architecture.md)
  - [Op Development & Qualification](docs/maintainer/op-development.md)

---

## Credits & Acknowledgments

- **Infernix** ([Wallawalla47](https://github.com/Wallawalla47)) provided the base engine, `.infernix` v3 container format, hybrid prefix cache, and `Qwen3.8-Flash-Next` implementation that this repository retuned for the **NVIDIA DGX Spark (`NVIDIA GB10`, `sm_121a`)**.
- **[NInfer](https://github.com/Neroued/ninfer)** by **[Neroued](https://github.com/Neroued)** ([Ko-fi](https://ko-fi.com/neroued)) created the original C++/CUDA engine and artifact architecture that Infernix grew from.
- **[Strata](https://github.com/Niko1221/Strata)** by **[Niko1221](https://github.com/Niko1221)** pioneered single-GPU `Qwen3.8-Flash-Next` execution and mathematical transcription.
- Thanks also to [IST-DASLab](https://github.com/IST-DASLab) (`llmq`), the authors of HyperQuant and Four Over Six, the Qwen team for the models, NVIDIA for the NVFP4 checkpoints, and all open-source contributors whose research and ideas shaped this engine.
