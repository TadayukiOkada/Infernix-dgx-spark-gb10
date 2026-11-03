# Linear kernels — NVFP4 (GB10)

Part of the [Linear kernel tables](README.md).

Measured 2026-11-02 at `f2a495e3c` (dev-rebased-20261028, 3 uncommitted file(s) (source edits: `bench/README.md`, `bench/ops/nvfp4_route_sweep.cu`, `docs/README.md`)) with `build-native/bench/infernix_linear_bench --qtype <qtype> --policy <policy> --n N --k K --sweep … --warmup 3 --repeat 12 --flush-mib 48` (cold-L2: the eviction buffer is read between samples; the default `exact` route set).

One row per token count `T`. `kernel` is the schedule the dispatch table in `src/ops/linear/nvfp4/shapes/` selects at that `T`, with its dispatch band in parentheses: the shape file's own launch call, with its `using` aliases and `constexpr` constants substituted and literal ternaries folded where the result fits the cell (a parameterized alias such as `Simt<2>`, or one whose expansion is too long, stays as the source writes it). `TFLOP/s` = 2·N·K·T / median; `GB/s` = (weight + activation + output bytes) / median.

`%pipe` and `%mem` put each row against the envelope that governs it, so a shortfall is attributable rather than a gap to a single device number. `%pipe` divides by the measured MMA-pipe peak of the pipe that policy actually runs on (`a16` → BF16 HMMA 123.71 TFLOP/s; `a4` → NVFP4 OMMA 494.30 TFLOP/s); the packed weight format is not the ceiling — an A16 route dequantizes its stored weights and lands on the BF16 HMMA pipe whatever the codec. `%mem` divides by 240.6 GB/s, the `ckt env` measured read-only ceiling. A row is at its envelope when the percentage of the envelope that binds it — `mem` at decode widths, `pipe` at prefill widths — is near 100.

Two regimes keep both percentages far below 100 and no schedule closes them. First, a payload smaller than the bandwidth × memory-latency product (~295 KB here) cannot reach the read ceiling at all: the whole tensor must be in flight before the stream ends. Second, the timed window is the whole public Op, so a route that composes several launches pays every one of them.

`nsys` on `[128, 2560]` `T=1` separates the two: its single `nvfp4_a16_gemv_kernel` runs 2.5–2.8 µs against a 0.77 µs memory floor (the ramp), and the same call measures 5.98 µs inside the CUDA-event window (the launch gap on top). The sweep times the Op eagerly, while the decode route replays a captured CUDA Graph (`src/core/decode_graph.cpp`), so that host-side gap is not paid per launch there. A route cell naming a `*_tma_route` launcher owns its schedule outside the shape file: `src/ops/linear/nvfp4/nvfp4_a4_tma.cu` instantiates `Nvfp4A4TmaMmaSchedule<256, 3, 1, kNvfp4A4TmaWeightCodeL2_128B, 1, 128>` for the `Tiled256` band and `Nvfp4A4TmaMmaSchedule<128, 3, 2, …, 1, 64>` for `Tiled128`. Every A4 row is a quantize kernel plus that GEMM launched back to back, so it is the sum of both, and the split is measurable: the sweep's `a4_split_2q1g` − `a4_split_1q1g` isolates the quantizer and `a4_split_1q2g` − `a4_split_1q1g` the GEMM. At `[5120, 17408]` `T=1024` that is a 736 µs pair = 206 µs quantize (at the memory ceiling) + 560 µs GEMM (326 TFLOP/s, 66% of the OMMA pipe); at `[16384, 2560]` `T=2048` it is a 648 µs pair = 15 µs + 631 µs, so there the quantize is 2% and the whole shortfall is the GEMM (272 TFLOP/s, 55%). Two mechanisms aimed at that shortfall were measured and regress: pipelining the quantizer and the GEMM across token chunks on a second stream costs +6.4% (`T=2048`) to +24.0% (`T=1024`) with two chunks and up to +93% with four, because both kernels are bandwidth-hungry and each chunk's GEMM fits the 48-SM wave worse; and a second widened BF16 weight plane in the A16 tile (dequant overlapping the HMMA block instead of preceding it) regresses at every width measured — at `[16384, 2560]` the incumbent `mma_r64_t128_k64_s2_a2` beats the three widened-plane variants by 8.3-33.3% at `T=2048` and by 22.0-38.1% at `T=256`. The A16 prefill plateau is therefore not the serialized dequant.

Repeatability: each row is the better median of three independent cold-L2 passes. Across these 1090 rows the passes differ by 1.2% (median), 6.74% (p90) and 36.43% (worst), so a single pass is not a repeatable number for the microsecond rows. Every pass ran with a quiet host: host-side memory work (a build, a `cuobjdump` of a large binary) shares GB10's LPDDR5X with the GPU and inflates measured kernel times. Rows above 246.0 GB/s for [13952, 2560] (T 256-256), [16384, 2560] (T 128-128) exceed it by at most 1.0%: the reference is a measured pure-read ceiling, and the metric counts modelled traffic (weights + activations + outputs), so output bytes still held in cache when the sample ends never reached DRAM inside the interval.

Not covered: the `--routes fast` tiers (Q8 sliced-K, FP8 A8 pair, Q4 W4A8, NVFP4 A4 pair), the `invariant` policy, and route entries the benchmark never admits — the f16-accumulate route (`LinearRoutes::F16Accumulate`, `launch_f16`) and `launch_pair_block32`. A row with empty metrics is a policy band above the measured cap (`T > 2048`): production dispatch selects that schedule there, but the sweep did not time it.

| N | K | policy | kernel | T | TFLOP/s | %pipe | GB/s | %mem |
|---:|---:|---|---|---:|---:|---:|---:|---:|
| 128 | 2560 | a16 | `nvfp4_linear_a16_gemv<Nvfp4Geometry<128, 2560>, Nvfp4A16GemvSchedule<4, 1, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>>` (T=1) | 1 | 0.11 | 0.1% | 31.7 | 13% |
| 128 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<128, 2560>, Nvfp4SlicedInstance<16, 8, 2>>` (T=2) | 2 | 0.16 | 0.1% | 24.3 | 10% |
| 128 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<128, 2560>, Nvfp4A16SlicedKMmaSchedule<8, 8, 2, Cache::ca, Cache::cg, Nvfp4ActivationStage::ActiveOnly, 2>>` (T 3-4) | 3 | 0.24 | 0.2% | 25.0 | 10% |
| 128 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<128, 2560>, Nvfp4A16SlicedKMmaSchedule<8, 8, 2, Cache::ca, Cache::cg, Nvfp4ActivationStage::ActiveOnly, 2>>` (T 3-4) | 4 | 0.33 | 0.3% | 25.7 | 11% |
| 128 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<128, 2560>, Nvfp4SlicedInstance<8, 8, 2>>` (T 5-8) | 5 | 0.41 | 0.3% | 26.3 | 11% |
| 128 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<128, 2560>, Nvfp4SlicedInstance<8, 8, 2>>` (T 5-8) | 6 | 0.49 | 0.4% | 27.0 | 11% |
| 128 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<128, 2560>, Nvfp4SlicedInstance<8, 8, 2>>` (T 5-8) | 7 | 0.57 | 0.5% | 27.7 | 12% |
| 128 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<128, 2560>, Nvfp4SlicedInstance<8, 8, 2>>` (T 5-8) | 8 | 0.66 | 0.5% | 28.4 | 12% |
| 128 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<128, 2560>, Nvfp4SlicedInstance<32, 8, 2>>` (T 9-32) | 9 | 0.73 | 0.6% | 29.0 | 12% |
| 128 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<128, 2560>, Nvfp4SlicedInstance<32, 8, 2>>` (T 9-32) | 12 | 0.98 | 0.8% | 31.1 | 13% |
| 128 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<128, 2560>, Nvfp4SlicedInstance<32, 8, 2>>` (T 9-32) | 16 | 1.30 | 1.1% | 33.5 | 14% |
| 128 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<128, 2560>, Nvfp4SlicedInstance<32, 8, 2>>` (T 9-32) | 24 | 1.57 | 1.3% | 31.2 | 13% |
| 128 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<128, 2560>, Nvfp4SlicedInstance<32, 8, 2>>` (T 9-32) | 32 | 2.09 | 1.7% | 35.5 | 15% |
| 128 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<128, 2560>, Nvfp4SlicedInstance<48, 4, 2>>` (T 33-48) | 33 | 1.78 | 1.4% | 29.7 | 12% |
| 128 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<128, 2560>, Nvfp4SlicedInstance<48, 4, 2>>` (T 33-48) | 48 | 2.22 | 1.8% | 31.3 | 13% |
| 128 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<128, 2560>, Nvfp4A16MmaSchedule<32, 16, 128, 16, 8, 2, 3>>` (T 49-64) | 49 | 1.98 | 1.6% | 27.7 | 12% |
| 128 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<128, 2560>, Nvfp4A16MmaSchedule<32, 16, 128, 16, 8, 2, 3>>` (T 49-64) | 64 | 2.59 | 2.1% | 32.6 | 14% |
| 128 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<128, 2560>, Nvfp4A16MmaSchedule<32, 32, 128, 16, 16, 2, 2>>` (T 65-1024) | 65 | 2.63 | 2.1% | 33.0 | 14% |
| 128 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<128, 2560>, Nvfp4A16MmaSchedule<32, 32, 128, 16, 16, 2, 2>>` (T 65-1024) | 128 | 5.18 | 4.2% | 53.9 | 22% |
| 128 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<128, 2560>, Nvfp4A16MmaSchedule<32, 32, 128, 16, 16, 2, 2>>` (T 65-1024) | 256 | 9.13 | 7.4% | 85.0 | 35% |
| 128 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<128, 2560>, Nvfp4A16MmaSchedule<32, 32, 128, 16, 16, 2, 2>>` (T 65-1024) | 384 | 12.38 | 10% | 110.7 | 46% |
| 128 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<128, 2560>, Nvfp4A16MmaSchedule<32, 32, 128, 16, 16, 2, 2>>` (T 65-1024) | 512 | 15.00 | 12% | 131.3 | 55% |
| 128 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<128, 2560>, Nvfp4A16MmaSchedule<32, 32, 128, 16, 16, 2, 2>>` (T 65-1024) | 640 | 17.20 | 14% | 148.7 | 62% |
| 128 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<128, 2560>, Nvfp4A16MmaSchedule<32, 32, 128, 16, 16, 2, 2>>` (T 65-1024) | 768 | 19.04 | 15% | 163.2 | 68% |
| 128 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<128, 2560>, Nvfp4A16MmaSchedule<32, 32, 128, 16, 16, 2, 2>>` (T 65-1024) | 896 | 20.62 | 17% | 175.6 | 73% |
| 128 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<128, 2560>, Nvfp4A16MmaSchedule<32, 32, 128, 16, 16, 2, 2>>` (T 65-1024) | 1024 | 21.96 | 18% | 186.2 | 77% |
| 128 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<128, 2560>, Nvfp4A16MmaSchedule<64, 96, 128, 32, 16, 2, 1>>` (T>=1025) | 1025 | 17.33 | 14% | 147.0 | 61% |
| 128 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<128, 2560>, Nvfp4A16MmaSchedule<64, 96, 128, 32, 16, 2, 1>>` (T>=1025) | 1152 | 19.47 | 16% | 164.4 | 68% |
| 128 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<128, 2560>, Nvfp4A16MmaSchedule<64, 96, 128, 32, 16, 2, 1>>` (T>=1025) | 1280 | 19.59 | 16% | 165.0 | 69% |
| 128 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<128, 2560>, Nvfp4A16MmaSchedule<64, 96, 128, 32, 16, 2, 1>>` (T>=1025) | 1408 | 21.54 | 17% | 181.0 | 75% |
| 128 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<128, 2560>, Nvfp4A16MmaSchedule<64, 96, 128, 32, 16, 2, 1>>` (T>=1025) | 1536 | 22.44 | 18% | 188.2 | 78% |
| 128 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<128, 2560>, Nvfp4A16MmaSchedule<64, 96, 128, 32, 16, 2, 1>>` (T>=1025) | 1664 | 22.27 | 18% | 186.5 | 78% |
| 128 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<128, 2560>, Nvfp4A16MmaSchedule<64, 96, 128, 32, 16, 2, 1>>` (T>=1025) | 1792 | 23.99 | 19% | 200.5 | 83% |
| 128 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<128, 2560>, Nvfp4A16MmaSchedule<64, 96, 128, 32, 16, 2, 1>>` (T>=1025) | 1920 | 22.83 | 18% | 190.7 | 79% |
| 128 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<128, 2560>, Nvfp4A16MmaSchedule<64, 96, 128, 32, 16, 2, 1>>` (T>=1025) | 2048 | 25.21 | 20% | 210.2 | 87% |
| 128 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<128, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 1 | 0.11 | 0.0% | 31.9 | 13% |
| 128 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<128, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 2 | 0.16 | 0.0% | 24.4 | 10% |
| 128 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<128, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 3 | 0.24 | 0.0% | 25.0 | 10% |
| 128 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<128, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 4 | 0.33 | 0.1% | 25.7 | 11% |
| 128 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<128, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 5 | 0.41 | 0.1% | 26.4 | 11% |
| 128 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<128, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 6 | 0.49 | 0.1% | 27.1 | 11% |
| 128 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<128, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 7 | 0.57 | 0.1% | 27.7 | 12% |
| 128 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<128, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 8 | 0.66 | 0.1% | 28.4 | 12% |
| 128 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<128, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 12 | 0.98 | 0.2% | 31.1 | 13% |
| 128 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<128, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 16 | 1.31 | 0.3% | 33.7 | 14% |
| 128 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<128, 2560>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 17-32) | 17 | 1.35 | 0.3% | 33.4 | 14% |
| 128 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<128, 2560>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 17-32) | 24 | 1.57 | 0.3% | 31.2 | 13% |
| 128 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<128, 2560>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 17-32) | 32 | 2.06 | 0.4% | 35.0 | 15% |
| 128 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<128, 2560>, Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>>` (T 33-64) | 33 | 1.79 | 0.4% | 29.9 | 12% |
| 128 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<128, 2560>, Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>>` (T 33-64) | 48 | 2.60 | 0.5% | 36.6 | 15% |
| 128 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<128, 2560>, Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>>` (T 33-64) | 64 | 2.97 | 0.6% | 37.4 | 16% |
| 128 | 2560 | a4 | `nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>` (T>=65) | 65 | 2.63 | 0.5% | 33.0 | 14% |
| 128 | 2560 | a4 | `nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>` (T>=65) | 128 | 4.63 | 0.9% | 48.2 | 20% |
| 128 | 2560 | a4 | `nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>` (T>=65) | 256 | 9.18 | 1.9% | 85.4 | 35% |
| 128 | 2560 | a4 | `nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>` (T>=65) | 384 | 12.29 | 2.5% | 109.8 | 46% |
| 128 | 2560 | a4 | `nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>` (T>=65) | 512 | 15.00 | 3.0% | 131.3 | 55% |
| 128 | 2560 | a4 | `nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>` (T>=65) | 640 | 17.18 | 3.5% | 148.5 | 62% |
| 128 | 2560 | a4 | `nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>` (T>=65) | 768 | 19.04 | 3.9% | 163.2 | 68% |
| 128 | 2560 | a4 | `nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>` (T>=65) | 896 | 20.64 | 4.2% | 175.8 | 73% |
| 128 | 2560 | a4 | `nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>` (T>=65) | 1024 | 20.60 | 4.2% | 174.6 | 73% |
| 128 | 2560 | a4 | `nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>` (T>=65) | 1152 | 19.48 | 3.9% | 164.6 | 68% |
| 128 | 2560 | a4 | `nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>` (T>=65) | 1280 | 20.54 | 4.2% | 173.0 | 72% |
| 128 | 2560 | a4 | `nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>` (T>=65) | 1408 | 20.61 | 4.2% | 173.2 | 72% |
| 128 | 2560 | a4 | `nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>` (T>=65) | 1536 | 22.44 | 4.5% | 188.2 | 78% |
| 128 | 2560 | a4 | `nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>` (T>=65) | 1664 | 22.17 | 4.5% | 185.6 | 77% |
| 128 | 2560 | a4 | `nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>` (T>=65) | 1792 | 22.94 | 4.6% | 191.8 | 80% |
| 128 | 2560 | a4 | `nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>` (T>=65) | 1920 | 22.82 | 4.6% | 190.6 | 79% |
| 128 | 2560 | a4 | `nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>` (T>=65) | 2048 | 22.67 | 4.6% | 189.1 | 79% |
| 512 | 2560 | a16 | `nvfp4_linear_a16_gemv<Nvfp4Geometry<512, 2560>, Nvfp4A16GemvSchedule<4, 1, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>>` (T=1) | 1 | 0.33 | 0.3% | 92.6 | 38% |
| 512 | 2560 | a16 | `nvfp4_linear_a16_simt<Nvfp4Geometry<512, 2560>, 2, C2, true>` (T=2) | 2 | 0.65 | 0.5% | 93.3 | 39% |
| 512 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<512, 2560>, Nvfp4SlicedInstance<8, 8, 2>>` (T 3-4) | 3 | 0.78 | 0.6% | 75.0 | 31% |
| 512 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<512, 2560>, Nvfp4SlicedInstance<8, 8, 2>>` (T 3-4) | 4 | 1.04 | 0.8% | 75.6 | 31% |
| 512 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<512, 2560>, Nvfp4SlicedInstance<24, 8, 2>>` (T 5-8) | 5 | 1.30 | 1.1% | 76.2 | 32% |
| 512 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<512, 2560>, Nvfp4SlicedInstance<24, 8, 2>>` (T 5-8) | 6 | 1.57 | 1.3% | 77.0 | 32% |
| 512 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<512, 2560>, Nvfp4SlicedInstance<24, 8, 2>>` (T 5-8) | 7 | 1.82 | 1.5% | 77.4 | 32% |
| 512 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<512, 2560>, Nvfp4SlicedInstance<24, 8, 2>>` (T 5-8) | 8 | 2.08 | 1.7% | 78.0 | 32% |
| 512 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<512, 2560>, Nvfp4SlicedInstance<16, 8, 2>>` (T 9-16) | 9 | 2.34 | 1.9% | 78.6 | 33% |
| 512 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<512, 2560>, Nvfp4SlicedInstance<16, 8, 2>>` (T 9-16) | 12 | 3.13 | 2.5% | 80.7 | 34% |
| 512 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<512, 2560>, Nvfp4SlicedInstance<16, 8, 2>>` (T 9-16) | 16 | 4.17 | 3.4% | 83.2 | 35% |
| 512 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<512, 2560>, Nvfp4SlicedInstance<32, 8, 2>>` (T 17-32) | 17 | 4.42 | 3.6% | 83.5 | 35% |
| 512 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<512, 2560>, Nvfp4SlicedInstance<32, 8, 2>>` (T 17-32) | 24 | 5.20 | 4.2% | 73.1 | 30% |
| 512 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<512, 2560>, Nvfp4SlicedInstance<32, 8, 2>>` (T 17-32) | 32 | 6.94 | 5.6% | 77.2 | 32% |
| 512 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<512, 2560>, Nvfp4A16MmaSchedule<32, 16, 128, 16, 8, 2, 3>>` (T 33-64) | 33 | 5.34 | 4.3% | 58.1 | 24% |
| 512 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<512, 2560>, Nvfp4A16MmaSchedule<32, 16, 128, 16, 8, 2, 3>>` (T 33-64) | 48 | 7.76 | 6.3% | 63.6 | 26% |
| 512 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<512, 2560>, Nvfp4A16MmaSchedule<32, 16, 128, 16, 8, 2, 3>>` (T 33-64) | 64 | 10.34 | 8.4% | 69.7 | 29% |
| 512 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<512, 2560>, Nvfp4A16MmaSchedule<32, 32, 128, 16, 16, 2, 2>>` (T 65-384) | 65 | 10.42 | 8.4% | 69.5 | 29% |
| 512 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<512, 2560>, Nvfp4A16MmaSchedule<32, 32, 128, 16, 16, 2, 2>>` (T 65-384) | 128 | 18.40 | 15% | 83.5 | 35% |
| 512 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<512, 2560>, Nvfp4A16MmaSchedule<32, 32, 128, 16, 16, 2, 2>>` (T 65-384) | 256 | 30.05 | 24% | 103.4 | 43% |
| 512 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<512, 2560>, Nvfp4A16MmaSchedule<32, 32, 128, 16, 16, 2, 2>>` (T 65-384) | 384 | 28.91 | 23% | 88.9 | 37% |
| 512 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<512, 2560>, Nvfp4A16MmaSchedule<64, 128, 128, 32, 16, 2, 1>>` (T 385-512) | 385 | 29.15 | 24% | 89.6 | 37% |
| 512 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<512, 2560>, Nvfp4A16MmaSchedule<64, 128, 128, 32, 16, 2, 1>>` (T 385-512) | 512 | 38.76 | 31% | 112.1 | 47% |
| 512 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<512, 2560>, Nvfp4A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>>` (T 513-1024) | 513 | 36.67 | 30% | 106.1 | 44% |
| 512 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<512, 2560>, Nvfp4A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>>` (T 513-1024) | 640 | 37.40 | 30% | 104.1 | 43% |
| 512 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<512, 2560>, Nvfp4A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>>` (T 513-1024) | 768 | 46.81 | 38% | 126.9 | 53% |
| 512 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<512, 2560>, Nvfp4A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>>` (T 513-1024) | 896 | 50.03 | 40% | 133.0 | 55% |
| 512 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<512, 2560>, Nvfp4A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>>` (T 513-1024) | 1024 | 54.83 | 44% | 143.6 | 60% |
| 512 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<512, 2560>, Nvfp4A16MmaSchedule<128, 64, 64, 32, 16, 2, 2>>` (T>=1025) | 1025 | 43.87 | 35% | 114.9 | 48% |
| 512 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<512, 2560>, Nvfp4A16MmaSchedule<128, 64, 64, 32, 16, 2, 2>>` (T>=1025) | 1152 | 51.01 | 41% | 132.0 | 55% |
| 512 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<512, 2560>, Nvfp4A16MmaSchedule<128, 64, 64, 32, 16, 2, 2>>` (T>=1025) | 1280 | 56.65 | 46% | 145.2 | 60% |
| 512 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<512, 2560>, Nvfp4A16MmaSchedule<128, 64, 64, 32, 16, 2, 2>>` (T>=1025) | 1408 | 60.23 | 49% | 153.2 | 64% |
| 512 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<512, 2560>, Nvfp4A16MmaSchedule<128, 64, 64, 32, 16, 2, 2>>` (T>=1025) | 1536 | 65.60 | 53% | 165.8 | 69% |
| 512 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<512, 2560>, Nvfp4A16MmaSchedule<128, 64, 64, 32, 16, 2, 2>>` (T>=1025) | 1664 | 48.49 | 39% | 121.9 | 51% |
| 512 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<512, 2560>, Nvfp4A16MmaSchedule<128, 64, 64, 32, 16, 2, 2>>` (T>=1025) | 1792 | 51.08 | 41% | 127.7 | 53% |
| 512 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<512, 2560>, Nvfp4A16MmaSchedule<128, 64, 64, 32, 16, 2, 2>>` (T>=1025) | 1920 | 54.71 | 44% | 136.2 | 57% |
| 512 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<512, 2560>, Nvfp4A16MmaSchedule<128, 64, 64, 32, 16, 2, 2>>` (T>=1025) | 2048 | 57.10 | 46% | 141.7 | 59% |
| 512 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<512, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 1 | 0.33 | 0.1% | 92.9 | 39% |
| 512 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<512, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 2 | 0.65 | 0.1% | 93.0 | 39% |
| 512 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<512, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 3 | 0.78 | 0.2% | 75.2 | 31% |
| 512 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<512, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 4 | 1.04 | 0.2% | 75.8 | 32% |
| 512 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<512, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 5 | 1.30 | 0.3% | 76.4 | 32% |
| 512 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<512, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 6 | 1.57 | 0.3% | 77.0 | 32% |
| 512 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<512, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 7 | 1.83 | 0.4% | 77.7 | 32% |
| 512 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<512, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 8 | 2.09 | 0.4% | 78.3 | 33% |
| 512 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<512, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 12 | 3.12 | 0.6% | 80.5 | 33% |
| 512 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<512, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 16 | 4.16 | 0.8% | 82.9 | 34% |
| 512 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<512, 2560>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 17-32) | 17 | 3.67 | 0.7% | 69.4 | 29% |
| 512 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<512, 2560>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 17-32) | 24 | 5.20 | 1.1% | 73.1 | 30% |
| 512 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<512, 2560>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 17-32) | 32 | 6.94 | 1.4% | 77.2 | 32% |
| 512 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<512, 2560>, Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>>` (T 33-64) | 33 | 5.34 | 1.1% | 58.1 | 24% |
| 512 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<512, 2560>, Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>>` (T 33-64) | 48 | 7.77 | 1.6% | 63.7 | 26% |
| 512 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<512, 2560>, Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>>` (T 33-64) | 64 | 10.36 | 2.1% | 69.8 | 29% |
| 512 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<512, 2560>, Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>>` (T 65-128) | 65 | 10.46 | 2.1% | 69.8 | 29% |
| 512 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<512, 2560>, Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>>` (T 65-128) | 128 | 18.40 | 3.7% | 83.5 | 35% |
| 512 | 2560 | a4 | `nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>` (T>=129) | 129 | 18.54 | 3.8% | 83.9 | 35% |
| 512 | 2560 | a4 | `nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>` (T>=129) | 256 | 34.78 | 7.0% | 119.7 | 50% |
| 512 | 2560 | a4 | `nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>` (T>=129) | 384 | 41.28 | 8.4% | 127.0 | 53% |
| 512 | 2560 | a4 | `nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>` (T>=129) | 512 | 45.49 | 9.2% | 131.6 | 55% |
| 512 | 2560 | a4 | `nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>` (T>=129) | 640 | 53.12 | 11% | 147.8 | 61% |
| 512 | 2560 | a4 | `nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>` (T>=129) | 768 | 54.85 | 11% | 148.6 | 62% |
| 512 | 2560 | a4 | `nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>` (T>=129) | 896 | 54.82 | 11% | 145.7 | 61% |
| 512 | 2560 | a4 | `nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>` (T>=129) | 1024 | 57.22 | 12% | 149.8 | 62% |
| 512 | 2560 | a4 | `nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>` (T>=129) | 1152 | 61.64 | 12% | 159.5 | 66% |
| 512 | 2560 | a4 | `nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>` (T>=129) | 1280 | 63.43 | 13% | 162.6 | 68% |
| 512 | 2560 | a4 | `nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>` (T>=129) | 1408 | 64.69 | 13% | 164.5 | 68% |
| 512 | 2560 | a4 | `nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>` (T>=129) | 1536 | 63.58 | 13% | 160.7 | 67% |
| 512 | 2560 | a4 | `nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>` (T>=129) | 1664 | 59.58 | 12% | 149.7 | 62% |
| 512 | 2560 | a4 | `nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>` (T>=129) | 1792 | 62.15 | 13% | 155.4 | 65% |
| 512 | 2560 | a4 | `nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>` (T>=129) | 1920 | 63.14 | 13% | 157.2 | 65% |
| 512 | 2560 | a4 | `nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>` (T>=129) | 2048 | 61.79 | 13% | 153.3 | 64% |
| 2048 | 2560 | a16 | `nvfp4_linear_a16_gemv<Nvfp4Geometry<2048, 2560>, Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>>` (T=1) | 1 | 0.64 | 0.5% | 180.2 | 75% |
| 2048 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<2048, 2560>, Nvfp4SlicedInstance<8, 8, 2>>` (T 2-8) | 2 | 1.15 | 0.9% | 162.4 | 67% |
| 2048 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<2048, 2560>, Nvfp4SlicedInstance<8, 8, 2>>` (T 2-8) | 3 | 1.72 | 1.4% | 162.9 | 68% |
| 2048 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<2048, 2560>, Nvfp4SlicedInstance<8, 8, 2>>` (T 2-8) | 4 | 2.30 | 1.9% | 163.4 | 68% |
| 2048 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<2048, 2560>, Nvfp4SlicedInstance<8, 8, 2>>` (T 2-8) | 5 | 2.87 | 2.3% | 163.9 | 68% |
| 2048 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<2048, 2560>, Nvfp4SlicedInstance<8, 8, 2>>` (T 2-8) | 6 | 3.44 | 2.8% | 164.4 | 68% |
| 2048 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<2048, 2560>, Nvfp4SlicedInstance<8, 8, 2>>` (T 2-8) | 7 | 4.02 | 3.2% | 164.9 | 69% |
| 2048 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<2048, 2560>, Nvfp4SlicedInstance<8, 8, 2>>` (T 2-8) | 8 | 4.59 | 3.7% | 165.4 | 69% |
| 2048 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<2048, 2560>, Nvfp4SlicedInstance<16, 8, 2, 2>>` (T 9-16) | 9 | 5.16 | 4.2% | 165.9 | 69% |
| 2048 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<2048, 2560>, Nvfp4SlicedInstance<16, 8, 2, 2>>` (T 9-16) | 12 | 6.89 | 5.6% | 167.5 | 70% |
| 2048 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<2048, 2560>, Nvfp4SlicedInstance<16, 8, 2, 2>>` (T 9-16) | 16 | 9.18 | 7.4% | 169.5 | 70% |
| 2048 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<2048, 2560>, Nvfp4SlicedInstance<24, 4, 2>>` (T 17-24) | 17 | 8.79 | 7.1% | 153.1 | 64% |
| 2048 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<2048, 2560>, Nvfp4SlicedInstance<24, 4, 2>>` (T 17-24) | 24 | 12.40 | 10% | 156.3 | 65% |
| 2048 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<2048, 2560>, Nvfp4SlicedInstance<32, 4, 1>>` (T 25-32) | 25 | 11.72 | 9.5% | 142.1 | 59% |
| 2048 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<2048, 2560>, Nvfp4SlicedInstance<32, 4, 1>>` (T 25-32) | 32 | 14.89 | 12% | 144.0 | 60% |
| 2048 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2048, 2560>, Nvfp4A16MmaSchedule<32, 32, 128, 16, 16, 2, 2>>` (T 33-64) | 33 | 14.17 | 11% | 133.2 | 55% |
| 2048 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2048, 2560>, Nvfp4A16MmaSchedule<32, 32, 128, 16, 16, 2, 2>>` (T 33-64) | 48 | 20.61 | 17% | 138.9 | 58% |
| 2048 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2048, 2560>, Nvfp4A16MmaSchedule<32, 32, 128, 16, 16, 2, 2>>` (T 33-64) | 64 | 27.49 | 22% | 144.9 | 60% |
| 2048 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2048, 2560>, Nvfp4A16MmaSchedule<64, 64, 128, 32, 16, 1, 2>>` (T 65-96) | 65 | 19.68 | 16% | 102.5 | 43% |
| 2048 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2048, 2560>, Nvfp4A16MmaSchedule<64, 64, 128, 32, 16, 1, 2>>` (T 65-96) | 96 | 29.07 | 23% | 110.7 | 46% |
| 2048 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2048, 2560>, Nvfp4A16MmaSchedule<64, 128, 128, 32, 16, 2, 1>>` (T 97-128) | 97 | 29.38 | 24% | 111.0 | 46% |
| 2048 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2048, 2560>, Nvfp4A16MmaSchedule<64, 128, 128, 32, 16, 2, 1>>` (T 97-128) | 128 | 38.73 | 31% | 119.1 | 50% |
| 2048 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2048, 2560>, Nvfp4A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>>` (T 129-256) | 129 | 30.13 | 24% | 92.2 | 38% |
| 2048 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2048, 2560>, Nvfp4A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>>` (T 129-256) | 256 | 57.22 | 46% | 113.2 | 47% |
| 2048 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2048, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=257) | 257 | 43.98 | 36% | 86.8 | 36% |
| 2048 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2048, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=257) | 384 | 65.74 | 53% | 105.9 | 44% |
| 2048 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2048, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=257) | 512 | 59.71 | 48% | 85.3 | 35% |
| 2048 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2048, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=257) | 640 | 59.68 | 48% | 78.7 | 33% |
| 2048 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2048, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=257) | 768 | 70.32 | 57% | 87.6 | 36% |
| 2048 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2048, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=257) | 896 | 67.54 | 55% | 80.6 | 33% |
| 2048 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2048, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=257) | 1024 | 65.61 | 53% | 75.7 | 31% |
| 2048 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2048, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=257) | 1152 | 72.00 | 58% | 80.9 | 34% |
| 2048 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2048, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=257) | 1280 | 69.79 | 56% | 76.7 | 32% |
| 2048 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2048, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=257) | 1408 | 68.01 | 55% | 73.4 | 31% |
| 2048 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2048, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=257) | 1536 | 72.21 | 58% | 76.7 | 32% |
| 2048 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2048, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=257) | 1664 | 71.64 | 58% | 75.1 | 31% |
| 2048 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2048, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=257) | 1792 | 70.08 | 57% | 72.6 | 30% |
| 2048 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2048, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=257) | 1920 | 73.96 | 60% | 75.8 | 32% |
| 2048 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2048, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=257) | 2048 | 72.87 | 59% | 74.0 | 31% |
| 2048 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2048, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 1 | 0.61 | 0.1% | 171.5 | 71% |
| 2048 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2048, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 2 | 1.16 | 0.2% | 163.8 | 68% |
| 2048 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2048, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 3 | 1.72 | 0.3% | 163.2 | 68% |
| 2048 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2048, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 4 | 2.30 | 0.5% | 163.4 | 68% |
| 2048 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2048, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 5 | 2.87 | 0.6% | 164.2 | 68% |
| 2048 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2048, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 6 | 3.45 | 0.7% | 164.7 | 68% |
| 2048 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2048, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 7 | 4.02 | 0.8% | 165.2 | 69% |
| 2048 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2048, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 8 | 4.59 | 0.9% | 165.4 | 69% |
| 2048 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2048, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 12 | 6.89 | 1.4% | 167.5 | 70% |
| 2048 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2048, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 16 | 9.20 | 1.9% | 169.8 | 71% |
| 2048 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2048, 2560>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 17-32) | 17 | 9.70 | 2.0% | 169.1 | 70% |
| 2048 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2048, 2560>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 17-32) | 24 | 12.40 | 2.5% | 156.3 | 65% |
| 2048 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2048, 2560>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 17-32) | 32 | 15.00 | 3.0% | 145.0 | 60% |
| 2048 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2048, 2560>, Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>>` (T 33-64) | 33 | 17.06 | 3.5% | 160.4 | 67% |
| 2048 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2048, 2560>, Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>>` (T 33-64) | 48 | 24.77 | 5.0% | 166.9 | 69% |
| 2048 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2048, 2560>, Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>>` (T 33-64) | 64 | 30.00 | 6.1% | 158.2 | 66% |
| 2048 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2048, 2560>, Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>>` (T 65-128) | 65 | 27.73 | 5.6% | 144.4 | 60% |
| 2048 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2048, 2560>, Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>>` (T 65-128) | 128 | 54.90 | 11% | 168.9 | 70% |
| 2048 | 2560 | a4 | `nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>` (T 129-256) | 129 | 47.50 | 9.6% | 145.3 | 60% |
| 2048 | 2560 | a4 | `nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>` (T 129-256) | 256 | 90.98 | 18% | 179.9 | 75% |
| 2048 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 257 | 75.60 | 15% | 149.2 | 62% |
| 2048 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 384 | 103.99 | 21% | 167.6 | 70% |
| 2048 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 512 | 128.56 | 26% | 183.6 | 76% |
| 2048 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 640 | 152.97 | 31% | 201.7 | 84% |
| 2048 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 768 | 171.55 | 35% | 213.6 | 89% |
| 2048 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 896 | 129.51 | 26% | 154.5 | 64% |
| 2048 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1024 | 143.76 | 29% | 165.8 | 69% |
| 2048 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1152 | 155.22 | 31% | 174.3 | 72% |
| 2048 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1280 | 162.00 | 33% | 178.0 | 74% |
| 2048 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1408 | 174.10 | 35% | 187.8 | 78% |
| 2048 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1536 | 177.04 | 36% | 188.0 | 78% |
| 2048 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1664 | 152.35 | 31% | 159.7 | 66% |
| 2048 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1792 | 158.40 | 32% | 164.1 | 68% |
| 2048 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1920 | 164.14 | 33% | 168.3 | 70% |
| 2048 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 2048 | 167.81 | 34% | 170.5 | 71% |
| 2560 | 2560 | a16 | `nvfp4_linear_a16_gemv<Nvfp4Geometry<2560, 2560>, Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>>` (T=1) | 1 | 0.68 | 0.5% | 191.9 | 80% |
| 2560 | 2560 | a16 | `nvfp4_linear_a16_simt<Nvfp4Geometry<2560, 2560>, 2, C2, true>` (T=2) | 2 | 1.29 | 1.0% | 182.4 | 76% |
| 2560 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<2560, 2560>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 3-8) | 3 | 1.93 | 1.6% | 182.6 | 76% |
| 2560 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<2560, 2560>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 3-8) | 4 | 2.57 | 2.1% | 182.6 | 76% |
| 2560 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<2560, 2560>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 3-8) | 5 | 3.22 | 2.6% | 183.4 | 76% |
| 2560 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<2560, 2560>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 3-8) | 6 | 3.86 | 3.1% | 183.9 | 76% |
| 2560 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<2560, 2560>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 3-8) | 7 | 4.31 | 3.5% | 176.3 | 73% |
| 2560 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<2560, 2560>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 3-8) | 8 | 4.92 | 4.0% | 176.8 | 73% |
| 2560 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<2560, 2560>, Nvfp4SlicedInstance<16, 8, 2, 2>>` (T 9-16) | 9 | 5.54 | 4.5% | 177.3 | 74% |
| 2560 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<2560, 2560>, Nvfp4SlicedInstance<16, 8, 2, 2>>` (T 9-16) | 12 | 7.37 | 6.0% | 178.5 | 74% |
| 2560 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<2560, 2560>, Nvfp4SlicedInstance<16, 8, 2, 2>>` (T 9-16) | 16 | 9.84 | 8.0% | 180.7 | 75% |
| 2560 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<2560, 2560>, Nvfp4SlicedInstance<24, 4, 2, 2>>` (T 17-24) | 17 | 9.98 | 8.1% | 172.8 | 72% |
| 2560 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<2560, 2560>, Nvfp4SlicedInstance<24, 4, 2, 2>>` (T 17-24) | 24 | 14.08 | 11% | 176.0 | 73% |
| 2560 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<2560, 2560>, Nvfp4SlicedInstance<32, 4, 2, 2>>` (T 25-32) | 25 | 13.44 | 11% | 161.7 | 67% |
| 2560 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<2560, 2560>, Nvfp4SlicedInstance<32, 4, 2, 2>>` (T 25-32) | 32 | 17.18 | 14% | 164.4 | 68% |
| 2560 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2560, 2560>, Nvfp4A16MmaSchedule<64, 64, 128, 32, 16, 2, 1>>` (T 33-64) | 33 | 15.14 | 12% | 140.8 | 59% |
| 2560 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2560, 2560>, Nvfp4A16MmaSchedule<64, 64, 128, 32, 16, 2, 1>>` (T 33-64) | 48 | 22.07 | 18% | 146.5 | 61% |
| 2560 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2560, 2560>, Nvfp4A16MmaSchedule<64, 64, 128, 32, 16, 2, 1>>` (T 33-64) | 64 | 29.42 | 24% | 152.3 | 63% |
| 2560 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2560, 2560>, Nvfp4A16MmaSchedule<64, 128, 128, 32, 16, 2, 1>>` (T 65-128) | 65 | 24.58 | 20% | 125.6 | 52% |
| 2560 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2560, 2560>, Nvfp4A16MmaSchedule<64, 128, 128, 32, 16, 2, 1>>` (T 65-128) | 128 | 45.75 | 37% | 136.3 | 57% |
| 2560 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2560, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 129 | 27.61 | 22% | 81.8 | 34% |
| 2560 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2560, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 256 | 54.78 | 44% | 103.0 | 43% |
| 2560 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2560, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 384 | 55.89 | 45% | 84.6 | 35% |
| 2560 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2560, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 512 | 58.61 | 47% | 78.0 | 32% |
| 2560 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2560, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 640 | 59.50 | 48% | 72.6 | 30% |
| 2560 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2560, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 768 | 70.30 | 57% | 80.7 | 34% |
| 2560 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2560, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 896 | 69.17 | 56% | 75.7 | 31% |
| 2560 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2560, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 1024 | 69.05 | 56% | 72.9 | 30% |
| 2560 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2560, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 1152 | 68.97 | 56% | 70.7 | 29% |
| 2560 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2560, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 1280 | 68.31 | 55% | 68.4 | 28% |
| 2560 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2560, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 1408 | 68.81 | 56% | 67.5 | 28% |
| 2560 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2560, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 1536 | 72.87 | 59% | 70.3 | 29% |
| 2560 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2560, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 1664 | 72.49 | 59% | 68.9 | 29% |
| 2560 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2560, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 1792 | 72.17 | 58% | 67.7 | 28% |
| 2560 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2560, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 1920 | 72.32 | 58% | 67.1 | 28% |
| 2560 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2560, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 2048 | 72.05 | 58% | 66.2 | 28% |
| 2560 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2560, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 1 | 0.68 | 0.1% | 191.9 | 80% |
| 2560 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2560, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 2 | 1.29 | 0.3% | 182.7 | 76% |
| 2560 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2560, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 3 | 1.94 | 0.4% | 183.2 | 76% |
| 2560 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2560, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 4 | 2.58 | 0.5% | 183.7 | 76% |
| 2560 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2560, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 5 | 3.23 | 0.7% | 183.9 | 76% |
| 2560 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2560, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 6 | 3.87 | 0.8% | 184.4 | 77% |
| 2560 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2560, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 7 | 4.52 | 0.9% | 185.2 | 77% |
| 2560 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2560, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 8 | 5.15 | 1.0% | 185.2 | 77% |
| 2560 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2560, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 12 | 7.37 | 1.5% | 178.5 | 74% |
| 2560 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2560, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 16 | 9.83 | 2.0% | 180.4 | 75% |
| 2560 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2560, 2560>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 17-32) | 17 | 9.98 | 2.0% | 172.8 | 72% |
| 2560 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2560, 2560>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 17-32) | 24 | 14.06 | 2.8% | 175.8 | 73% |
| 2560 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2560, 2560>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 17-32) | 32 | 17.20 | 3.5% | 164.6 | 68% |
| 2560 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2560, 2560>, Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>>` (T 33-64) | 33 | 19.28 | 3.9% | 179.4 | 75% |
| 2560 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2560, 2560>, Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>>` (T 33-64) | 48 | 25.73 | 5.2% | 170.9 | 71% |
| 2560 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2560, 2560>, Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>>` (T 33-64) | 64 | 34.22 | 6.9% | 177.1 | 74% |
| 2560 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2560, 2560>, Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>>` (T 65-128) | 65 | 30.99 | 6.3% | 158.3 | 66% |
| 2560 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2560, 2560>, Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>>` (T 65-128) | 128 | 61.03 | 12% | 181.8 | 76% |
| 2560 | 2560 | a4 | `nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>` (T 129-256) | 129 | 55.39 | 11% | 164.0 | 68% |
| 2560 | 2560 | a4 | `nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>` (T 129-256) | 256 | 106.35 | 22% | 199.9 | 83% |
| 2560 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 257 | 94.41 | 19% | 177.1 | 74% |
| 2560 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 384 | 129.88 | 26% | 196.6 | 82% |
| 2560 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 512 | 160.46 | 32% | 213.5 | 89% |
| 2560 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 640 | 126.39 | 26% | 154.3 | 64% |
| 2560 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 768 | 144.90 | 29% | 166.3 | 69% |
| 2560 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 896 | 164.21 | 33% | 179.8 | 75% |
| 2560 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1024 | 175.13 | 35% | 184.9 | 77% |
| 2560 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1152 | 152.26 | 31% | 156.1 | 65% |
| 2560 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1280 | 164.30 | 33% | 164.5 | 68% |
| 2560 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1408 | 170.17 | 34% | 166.9 | 69% |
| 2560 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1536 | 177.37 | 36% | 171.1 | 71% |
| 2560 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1664 | 187.09 | 38% | 177.8 | 74% |
| 2560 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1792 | 191.55 | 39% | 179.7 | 75% |
| 2560 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1920 | 174.22 | 35% | 161.6 | 67% |
| 2560 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 2048 | 178.56 | 36% | 164.0 | 68% |
| 2560 | 6144 | a16 | `nvfp4_linear_a16_gemv<Nvfp4Geometry<2560, 6144>, Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>>` (T=1) | 1 | 0.75 | 0.6% | 212.0 | 88% |
| 2560 | 6144 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<2560, 6144>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 2 | 1.43 | 1.2% | 202.5 | 84% |
| 2560 | 6144 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<2560, 6144>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 3 | 2.15 | 1.7% | 203.0 | 84% |
| 2560 | 6144 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<2560, 6144>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 4 | 2.87 | 2.3% | 203.2 | 84% |
| 2560 | 6144 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<2560, 6144>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 5 | 3.59 | 2.9% | 204.1 | 85% |
| 2560 | 6144 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<2560, 6144>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 6 | 4.31 | 3.5% | 204.2 | 85% |
| 2560 | 6144 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<2560, 6144>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 7 | 5.02 | 4.1% | 204.4 | 85% |
| 2560 | 6144 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<2560, 6144>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 8 | 5.74 | 4.6% | 204.8 | 85% |
| 2560 | 6144 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<2560, 6144>, Nvfp4SlicedInstance<16, 8, 2, 2>>` (T 9-16) | 9 | 6.31 | 5.1% | 200.6 | 83% |
| 2560 | 6144 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<2560, 6144>, Nvfp4SlicedInstance<16, 8, 2, 2>>` (T 9-16) | 12 | 8.57 | 6.9% | 205.7 | 85% |
| 2560 | 6144 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<2560, 6144>, Nvfp4SlicedInstance<16, 8, 2, 2>>` (T 9-16) | 16 | 11.22 | 9.1% | 203.4 | 85% |
| 2560 | 6144 | a16 | `nvfp4_linear_a16_sliced_k_dyn<Nvfp4Geometry<2560, 6144>, Nvfp4SlicedInstance<24, 4, 2, 2>>` (T 17-24) | 17 | 10.92 | 8.8% | 186.6 | 78% |
| 2560 | 6144 | a16 | `nvfp4_linear_a16_sliced_k_dyn<Nvfp4Geometry<2560, 6144>, Nvfp4SlicedInstance<24, 4, 2, 2>>` (T 17-24) | 24 | 15.38 | 12% | 188.7 | 78% |
| 2560 | 6144 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<2560, 6144>, Nvfp4SlicedInstance<32, 4, 1>>` (T 25-32) | 25 | 14.26 | 12% | 168.4 | 70% |
| 2560 | 6144 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<2560, 6144>, Nvfp4SlicedInstance<32, 4, 1>>` (T 25-32) | 32 | 17.55 | 14% | 164.0 | 68% |
| 2560 | 6144 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2560, 6144>, Nvfp4A16MmaSchedule<32, 64, 128, 32, 16, 2, 2>>` (T 33-64) | 33 | 15.89 | 13% | 144.3 | 60% |
| 2560 | 6144 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2560, 6144>, Nvfp4A16MmaSchedule<32, 64, 128, 32, 16, 2, 2>>` (T 33-64) | 48 | 23.10 | 19% | 148.1 | 62% |
| 2560 | 6144 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2560, 6144>, Nvfp4A16MmaSchedule<32, 64, 128, 32, 16, 2, 2>>` (T 33-64) | 64 | 31.79 | 26% | 157.3 | 65% |
| 2560 | 6144 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2560, 6144>, Nvfp4A16MmaSchedule<64, 128, 128, 32, 16, 2, 1>>` (T 65-128) | 65 | 26.39 | 21% | 128.8 | 54% |
| 2560 | 6144 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2560, 6144>, Nvfp4A16MmaSchedule<64, 128, 128, 32, 16, 2, 1>>` (T 65-128) | 128 | 53.20 | 43% | 146.3 | 61% |
| 2560 | 6144 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2560, 6144>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 129 | 30.06 | 24% | 82.2 | 34% |
| 2560 | 6144 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2560, 6144>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 256 | 60.57 | 49% | 100.1 | 42% |
| 2560 | 6144 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2560, 6144>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 384 | 60.23 | 49% | 77.4 | 32% |
| 2560 | 6144 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2560, 6144>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 512 | 62.95 | 51% | 69.4 | 29% |
| 2560 | 6144 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2560, 6144>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 640 | 63.45 | 51% | 63.0 | 26% |
| 2560 | 6144 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2560, 6144>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 768 | 75.18 | 61% | 69.1 | 29% |
| 2560 | 6144 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2560, 6144>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 896 | 74.42 | 60% | 64.5 | 27% |
| 2560 | 6144 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2560, 6144>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 1024 | 73.18 | 59% | 60.6 | 25% |
| 2560 | 6144 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2560, 6144>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 1152 | 72.26 | 58% | 57.6 | 24% |
| 2560 | 6144 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2560, 6144>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 1280 | 71.25 | 58% | 55.1 | 23% |
| 2560 | 6144 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2560, 6144>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 1408 | 71.16 | 58% | 53.6 | 22% |
| 2560 | 6144 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2560, 6144>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 1536 | 76.37 | 62% | 56.2 | 23% |
| 2560 | 6144 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2560, 6144>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 1664 | 75.64 | 61% | 54.6 | 23% |
| 2560 | 6144 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2560, 6144>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 1792 | 75.00 | 61% | 53.3 | 22% |
| 2560 | 6144 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2560, 6144>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 1920 | 74.68 | 60% | 52.3 | 22% |
| 2560 | 6144 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<2560, 6144>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 2048 | 74.21 | 60% | 51.3 | 21% |
| 2560 | 6144 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2560, 6144>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 1 | 0.75 | 0.2% | 212.0 | 88% |
| 2560 | 6144 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2560, 6144>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 2 | 1.46 | 0.3% | 206.7 | 86% |
| 2560 | 6144 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2560, 6144>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 3 | 2.15 | 0.4% | 203.2 | 84% |
| 2560 | 6144 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2560, 6144>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 4 | 2.87 | 0.6% | 203.4 | 85% |
| 2560 | 6144 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2560, 6144>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 5 | 3.67 | 0.7% | 208.4 | 87% |
| 2560 | 6144 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2560, 6144>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 6 | 4.31 | 0.9% | 204.2 | 85% |
| 2560 | 6144 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2560, 6144>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 7 | 5.02 | 1.0% | 204.6 | 85% |
| 2560 | 6144 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2560, 6144>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 8 | 5.63 | 1.1% | 200.9 | 83% |
| 2560 | 6144 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2560, 6144>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 12 | 8.57 | 1.7% | 205.5 | 85% |
| 2560 | 6144 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2560, 6144>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 16 | 11.47 | 2.3% | 208.0 | 86% |
| 2560 | 6144 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2560, 6144>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 17-32) | 17 | 11.40 | 2.3% | 194.9 | 81% |
| 2560 | 6144 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2560, 6144>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 17-32) | 24 | 15.74 | 3.2% | 193.2 | 80% |
| 2560 | 6144 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2560, 6144>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 17-32) | 32 | 20.99 | 4.2% | 196.1 | 82% |
| 2560 | 6144 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2560, 6144>, Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>>` (T 33-64) | 33 | 21.26 | 4.3% | 192.9 | 80% |
| 2560 | 6144 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2560, 6144>, Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>>` (T 33-64) | 48 | 30.90 | 6.3% | 198.2 | 82% |
| 2560 | 6144 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2560, 6144>, Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>>` (T 33-64) | 64 | 40.20 | 8.1% | 198.9 | 83% |
| 2560 | 6144 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2560, 6144>, Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>>` (T 65-128) | 65 | 38.52 | 7.8% | 188.0 | 78% |
| 2560 | 6144 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2560, 6144>, Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>>` (T 65-128) | 128 | 73.03 | 15% | 200.9 | 83% |
| 2560 | 6144 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2560, 6144>, Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 1, 2>>` (T 129-256) | 129 | 53.10 | 11% | 145.2 | 60% |
| 2560 | 6144 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<2560, 6144>, Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 1, 2>>` (T 129-256) | 256 | 96.05 | 19% | 158.7 | 66% |
| 2560 | 6144 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 257 | 112.79 | 23% | 185.8 | 77% |
| 2560 | 6144 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 384 | 151.54 | 31% | 194.9 | 81% |
| 2560 | 6144 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 512 | 189.86 | 38% | 209.4 | 87% |
| 2560 | 6144 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 640 | 142.66 | 29% | 141.6 | 59% |
| 2560 | 6144 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 768 | 162.92 | 33% | 149.8 | 62% |
| 2560 | 6144 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 896 | 181.31 | 37% | 157.2 | 65% |
| 2560 | 6144 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1024 | 196.80 | 40% | 163.0 | 68% |
| 2560 | 6144 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1152 | 167.82 | 34% | 133.8 | 56% |
| 2560 | 6144 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1280 | 177.88 | 36% | 137.5 | 57% |
| 2560 | 6144 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1408 | 186.54 | 38% | 140.5 | 58% |
| 2560 | 6144 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1536 | 194.31 | 39% | 143.1 | 59% |
| 2560 | 6144 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1664 | 202.90 | 41% | 146.6 | 61% |
| 2560 | 6144 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1792 | 210.24 | 43% | 149.3 | 62% |
| 2560 | 6144 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1920 | 185.94 | 38% | 130.1 | 54% |
| 2560 | 6144 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 2048 | 193.06 | 39% | 133.4 | 55% |
| 5120 | 6144 | a16 | `nvfp4_linear_a16_gemv<Nvfp4Geometry<5120, 6144>, Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>>` (T=1) | 1 | 0.80 | 0.6% | 224.2 | 93% |
| 5120 | 6144 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<5120, 6144>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 2 | 1.54 | 1.2% | 217.0 | 90% |
| 5120 | 6144 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<5120, 6144>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 3 | 2.31 | 1.9% | 217.2 | 90% |
| 5120 | 6144 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<5120, 6144>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 4 | 3.08 | 2.5% | 217.5 | 90% |
| 5120 | 6144 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<5120, 6144>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 5 | 3.85 | 3.1% | 217.8 | 91% |
| 5120 | 6144 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<5120, 6144>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 6 | 4.62 | 3.7% | 218.1 | 91% |
| 5120 | 6144 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<5120, 6144>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 7 | 5.39 | 4.4% | 218.4 | 91% |
| 5120 | 6144 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<5120, 6144>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 8 | 6.14 | 5.0% | 218.2 | 91% |
| 5120 | 6144 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<5120, 6144>, Nvfp4SlicedInstance<16, 8, 2, 2>>` (T 9-16) | 9 | 6.75 | 5.5% | 213.2 | 89% |
| 5120 | 6144 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<5120, 6144>, Nvfp4SlicedInstance<16, 8, 2, 2>>` (T 9-16) | 12 | 9.12 | 7.4% | 217.0 | 90% |
| 5120 | 6144 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<5120, 6144>, Nvfp4SlicedInstance<16, 8, 2, 2>>` (T 9-16) | 16 | 12.15 | 9.8% | 217.9 | 91% |
| 5120 | 6144 | a16 | `nvfp4_linear_a16_sliced_k_dyn<Nvfp4Geometry<5120, 6144>, Nvfp4SlicedInstance<24, 4, 1, 2>>` (T 17-20) | 17 | 12.43 | 10% | 210.1 | 87% |
| 5120 | 6144 | a16 | `nvfp4_linear_a16_sliced_k_dyn<Nvfp4Geometry<5120, 6144>, Nvfp4SlicedInstance<24, 4, 1, 2>>` (T 17-20) | 20 | 14.64 | 12% | 211.2 | 88% |
| 5120 | 6144 | a16 | `nvfp4_linear_a16_sliced_k_dyn<Nvfp4Geometry<5120, 6144>, Nvfp4SlicedInstance<32, 4, 1, 2>>` (T 21-32) | 21 | 15.04 | 12% | 206.8 | 86% |
| 5120 | 6144 | a16 | `nvfp4_linear_a16_sliced_k_dyn<Nvfp4Geometry<5120, 6144>, Nvfp4SlicedInstance<32, 4, 1, 2>>` (T 21-32) | 24 | 16.79 | 14% | 202.8 | 84% |
| 5120 | 6144 | a16 | `nvfp4_linear_a16_sliced_k_dyn<Nvfp4Geometry<5120, 6144>, Nvfp4SlicedInstance<32, 4, 1, 2>>` (T 21-32) | 32 | 22.38 | 18% | 204.7 | 85% |
| 5120 | 6144 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<5120, 6144>, Nvfp4A16MmaSchedule<64, 64, 128, 32, 16, 2, 1>>` (T 33-64) | 33 | 18.46 | 15% | 163.9 | 68% |
| 5120 | 6144 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<5120, 6144>, Nvfp4A16MmaSchedule<64, 64, 128, 32, 16, 2, 1>>` (T 33-64) | 48 | 26.86 | 22% | 167.0 | 69% |
| 5120 | 6144 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<5120, 6144>, Nvfp4A16MmaSchedule<64, 64, 128, 32, 16, 2, 1>>` (T 33-64) | 64 | 36.43 | 29% | 173.1 | 72% |
| 5120 | 6144 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<5120, 6144>, Nvfp4A16MmaSchedule<64, 96, 128, 32, 16, 1, 2>>` (T 65-96) | 65 | 32.78 | 26% | 153.6 | 64% |
| 5120 | 6144 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<5120, 6144>, Nvfp4A16MmaSchedule<64, 96, 128, 32, 16, 1, 2>>` (T 65-96) | 96 | 48.43 | 39% | 159.2 | 66% |
| 5120 | 6144 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<5120, 6144>, Nvfp4A16MmaSchedule<64, 128, 128, 32, 16, 2, 1>>` (T 97-128) | 97 | 40.32 | 33% | 131.3 | 55% |
| 5120 | 6144 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<5120, 6144>, Nvfp4A16MmaSchedule<64, 128, 128, 32, 16, 2, 1>>` (T 97-128) | 128 | 53.93 | 44% | 137.8 | 57% |
| 5120 | 6144 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<5120, 6144>, Nvfp4A16MmaSchedule<64, 64, 64, 32, 16, 2, 2>>` (T 129-256) | 129 | 42.65 | 34% | 108.3 | 45% |
| 5120 | 6144 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<5120, 6144>, Nvfp4A16MmaSchedule<64, 64, 64, 32, 16, 2, 2>>` (T 129-256) | 256 | 62.96 | 51% | 91.7 | 38% |
| 5120 | 6144 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<5120, 6144>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=257) | 257 | 49.99 | 40% | 72.6 | 30% |
| 5120 | 6144 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<5120, 6144>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=257) | 384 | 74.71 | 60% | 81.5 | 34% |
| 5120 | 6144 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<5120, 6144>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=257) | 512 | 73.53 | 59% | 66.7 | 28% |
| 5120 | 6144 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<5120, 6144>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=257) | 640 | 72.84 | 59% | 58.1 | 24% |
| 5120 | 6144 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<5120, 6144>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=257) | 768 | 77.89 | 63% | 56.4 | 23% |
| 5120 | 6144 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<5120, 6144>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=257) | 896 | 76.27 | 62% | 51.3 | 21% |
| 5120 | 6144 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<5120, 6144>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=257) | 1024 | 75.45 | 61% | 47.7 | 20% |
| 5120 | 6144 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<5120, 6144>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=257) | 1152 | 78.83 | 64% | 47.5 | 20% |
| 5120 | 6144 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<5120, 6144>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=257) | 1280 | 77.12 | 62% | 44.6 | 19% |
| 5120 | 6144 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<5120, 6144>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=257) | 1408 | 75.63 | 61% | 42.2 | 18% |
| 5120 | 6144 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<5120, 6144>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=257) | 1536 | 77.87 | 63% | 42.1 | 17% |
| 5120 | 6144 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<5120, 6144>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=257) | 1664 | 77.47 | 63% | 40.8 | 17% |
| 5120 | 6144 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<5120, 6144>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=257) | 1792 | 76.79 | 62% | 39.5 | 16% |
| 5120 | 6144 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<5120, 6144>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=257) | 1920 | 78.55 | 63% | 39.6 | 16% |
| 5120 | 6144 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<5120, 6144>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=257) | 2048 | 77.78 | 63% | 38.5 | 16% |
| 5120 | 6144 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<5120, 6144>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 1 | 0.80 | 0.2% | 225.2 | 94% |
| 5120 | 6144 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<5120, 6144>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 2 | 1.54 | 0.3% | 217.0 | 90% |
| 5120 | 6144 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<5120, 6144>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 3 | 2.31 | 0.5% | 217.2 | 90% |
| 5120 | 6144 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<5120, 6144>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 4 | 3.08 | 0.6% | 217.6 | 90% |
| 5120 | 6144 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<5120, 6144>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 5 | 3.81 | 0.8% | 215.4 | 90% |
| 5120 | 6144 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<5120, 6144>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 6 | 4.62 | 0.9% | 218.2 | 91% |
| 5120 | 6144 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<5120, 6144>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 7 | 5.39 | 1.1% | 218.4 | 91% |
| 5120 | 6144 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<5120, 6144>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 8 | 6.16 | 1.2% | 218.6 | 91% |
| 5120 | 6144 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<5120, 6144>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 12 | 9.12 | 1.8% | 217.1 | 90% |
| 5120 | 6144 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<5120, 6144>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 16 | 12.14 | 2.5% | 217.8 | 91% |
| 5120 | 6144 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<5120, 6144>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 17-32) | 17 | 12.61 | 2.6% | 213.2 | 89% |
| 5120 | 6144 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<5120, 6144>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 17-32) | 24 | 17.39 | 3.5% | 210.0 | 87% |
| 5120 | 6144 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<5120, 6144>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 17-32) | 32 | 23.17 | 4.7% | 212.0 | 88% |
| 5120 | 6144 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<5120, 6144>, Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>>` (T 33-64) | 33 | 23.90 | 4.8% | 212.2 | 88% |
| 5120 | 6144 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<5120, 6144>, Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>>` (T 33-64) | 48 | 34.37 | 7.0% | 213.7 | 89% |
| 5120 | 6144 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<5120, 6144>, Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>>` (T 33-64) | 64 | 44.78 | 9.1% | 212.8 | 88% |
| 5120 | 6144 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<5120, 6144>, Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>>` (T 65-128) | 65 | 44.98 | 9.1% | 210.7 | 88% |
| 5120 | 6144 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<5120, 6144>, Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>>` (T 65-128) | 128 | 86.57 | 18% | 221.2 | 92% |
| 5120 | 6144 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<5120, 6144>, Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 1, 2>>` (T 129-256) | 129 | 82.72 | 17% | 210.0 | 87% |
| 5120 | 6144 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<5120, 6144>, Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 1, 2>>` (T 129-256) | 256 | 154.44 | 31% | 225.0 | 94% |
| 5120 | 6144 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 257 | 130.83 | 26% | 190.0 | 79% |
| 5120 | 6144 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 384 | 183.11 | 37% | 199.7 | 83% |
| 5120 | 6144 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 512 | 229.93 | 47% | 208.6 | 87% |
| 5120 | 6144 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 640 | 209.75 | 42% | 167.3 | 70% |
| 5120 | 6144 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 768 | 239.79 | 49% | 173.7 | 72% |
| 5120 | 6144 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 896 | 219.46 | 44% | 147.5 | 61% |
| 5120 | 6144 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1024 | 238.45 | 48% | 150.9 | 63% |
| 5120 | 6144 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1152 | 224.09 | 45% | 135.0 | 56% |
| 5120 | 6144 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1280 | 239.15 | 48% | 138.2 | 57% |
| 5120 | 6144 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1408 | 249.32 | 50% | 139.1 | 58% |
| 5120 | 6144 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1536 | 263.01 | 53% | 142.3 | 59% |
| 5120 | 6144 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1664 | 247.04 | 50% | 130.2 | 54% |
| 5120 | 6144 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1792 | 258.62 | 52% | 133.2 | 55% |
| 5120 | 6144 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1920 | 246.47 | 50% | 124.4 | 52% |
| 5120 | 6144 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 2048 | 256.40 | 52% | 127.0 | 53% |
| 5120 | 17408 | a16 | `nvfp4_linear_a16_gemv<Nvfp4Geometry<5120, 17408>, Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>>` (T=1) | 1 | 0.81 | 0.7% | 229.2 | 95% |
| 5120 | 17408 | a16 | `nvfp4_linear_a16_sliced_k_dyn<Nvfp4Geometry<5120, 17408>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 2 | 1.62 | 1.3% | 228.3 | 95% |
| 5120 | 17408 | a16 | `nvfp4_linear_a16_sliced_k_dyn<Nvfp4Geometry<5120, 17408>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 3 | 2.43 | 2.0% | 228.5 | 95% |
| 5120 | 17408 | a16 | `nvfp4_linear_a16_sliced_k_dyn<Nvfp4Geometry<5120, 17408>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 4 | 3.24 | 2.6% | 228.7 | 95% |
| 5120 | 17408 | a16 | `nvfp4_linear_a16_sliced_k_dyn<Nvfp4Geometry<5120, 17408>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 5 | 4.01 | 3.2% | 226.8 | 94% |
| 5120 | 17408 | a16 | `nvfp4_linear_a16_sliced_k_dyn<Nvfp4Geometry<5120, 17408>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 6 | 4.76 | 3.8% | 224.4 | 93% |
| 5120 | 17408 | a16 | `nvfp4_linear_a16_sliced_k_dyn<Nvfp4Geometry<5120, 17408>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 7 | 5.58 | 4.5% | 225.8 | 94% |
| 5120 | 17408 | a16 | `nvfp4_linear_a16_sliced_k_dyn<Nvfp4Geometry<5120, 17408>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 8 | 6.42 | 5.2% | 227.4 | 95% |
| 5120 | 17408 | a16 | `nvfp4_linear_a16_sliced_k_dyn<Nvfp4Geometry<5120, 17408>, Nvfp4SlicedInstance<16, 8, 2, 2>>` (T 9-16) | 9 | 7.12 | 5.8% | 224.4 | 93% |
| 5120 | 17408 | a16 | `nvfp4_linear_a16_sliced_k_dyn<Nvfp4Geometry<5120, 17408>, Nvfp4SlicedInstance<16, 8, 2, 2>>` (T 9-16) | 12 | 9.50 | 7.7% | 225.1 | 94% |
| 5120 | 17408 | a16 | `nvfp4_linear_a16_sliced_k_dyn<Nvfp4Geometry<5120, 17408>, Nvfp4SlicedInstance<16, 8, 2, 2>>` (T 9-16) | 16 | 12.51 | 10% | 223.1 | 93% |
| 5120 | 17408 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<5120, 17408>, Nvfp4SlicedInstance<24, 4, 1, 2>>` (T 17-24) | 17 | 12.88 | 10% | 216.3 | 90% |
| 5120 | 17408 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<5120, 17408>, Nvfp4SlicedInstance<24, 4, 1, 2>>` (T 17-24) | 24 | 17.87 | 14% | 213.9 | 89% |
| 5120 | 17408 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<5120, 17408>, Nvfp4SlicedInstance<32, 4, 1, 2>>` (T 25-32) | 25 | 18.60 | 15% | 213.9 | 89% |
| 5120 | 17408 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<5120, 17408>, Nvfp4SlicedInstance<32, 4, 1, 2>>` (T 25-32) | 32 | 23.23 | 19% | 210.0 | 87% |
| 5120 | 17408 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<5120, 17408>, Nvfp4A16MmaSchedule<64, 64, 128, 32, 16, 1, 2>>` (T 33-64) | 33 | 18.91 | 15% | 165.9 | 69% |
| 5120 | 17408 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<5120, 17408>, Nvfp4A16MmaSchedule<64, 64, 128, 32, 16, 1, 2>>` (T 33-64) | 48 | 27.14 | 22% | 165.9 | 69% |
| 5120 | 17408 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<5120, 17408>, Nvfp4A16MmaSchedule<64, 64, 128, 32, 16, 1, 2>>` (T 33-64) | 64 | 36.19 | 29% | 168.2 | 70% |
| 5120 | 17408 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<5120, 17408>, Nvfp4A16MmaSchedule<64, 96, 128, 32, 16, 1, 2>>` (T 65-96) | 65 | 33.49 | 27% | 153.4 | 64% |
| 5120 | 17408 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<5120, 17408>, Nvfp4A16MmaSchedule<64, 96, 128, 32, 16, 1, 2>>` (T 65-96) | 96 | 48.02 | 39% | 152.8 | 64% |
| 5120 | 17408 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<5120, 17408>, Nvfp4A16MmaSchedule<64, 128, 128, 32, 16, 2, 1>>` (T 97-128) | 97 | 42.23 | 34% | 133.1 | 55% |
| 5120 | 17408 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<5120, 17408>, Nvfp4A16MmaSchedule<64, 128, 128, 32, 16, 2, 1>>` (T 97-128) | 128 | 56.55 | 46% | 138.6 | 58% |
| 5120 | 17408 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<5120, 17408>, Nvfp4A16MmaSchedule<64, 96, 128, 64, 16, 1, 2>>` (T 129-192) | 129 | 40.11 | 32% | 97.6 | 41% |
| 5120 | 17408 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<5120, 17408>, Nvfp4A16MmaSchedule<64, 96, 128, 64, 16, 1, 2>>` (T 129-192) | 192 | 57.25 | 46% | 98.3 | 41% |
| 5120 | 17408 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<5120, 17408>, Nvfp4A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>` (T 193-256) | 193 | 48.14 | 39% | 82.3 | 34% |
| 5120 | 17408 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<5120, 17408>, Nvfp4A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>` (T 193-256) | 256 | 64.42 | 52% | 87.1 | 36% |
| 5120 | 17408 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<5120, 17408>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=257) | 257 | 51.31 | 41% | 69.1 | 29% |
| 5120 | 17408 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<5120, 17408>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=257) | 384 | 77.74 | 63% | 76.6 | 32% |
| 5120 | 17408 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<5120, 17408>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=257) | 512 | 75.16 | 61% | 60.3 | 25% |
| 5120 | 17408 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<5120, 17408>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=257) | 640 | 72.16 | 58% | 50.0 | 21% |
| 5120 | 17408 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<5120, 17408>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=257) | 768 | 80.55 | 65% | 49.9 | 21% |
| 5120 | 17408 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<5120, 17408>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=257) | 896 | 74.74 | 60% | 42.4 | 18% |
| 5120 | 17408 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<5120, 17408>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=257) | 1024 | 72.94 | 59% | 38.5 | 16% |
| 5120 | 17408 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<5120, 17408>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=257) | 1152 | 71.02 | 57% | 35.3 | 15% |
| 5120 | 17408 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<5120, 17408>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=257) | 1280 | 68.52 | 55% | 32.4 | 13% |
| 5120 | 17408 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<5120, 17408>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=257) | 1408 | 66.57 | 54% | 30.1 | 13% |
| 5120 | 17408 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<5120, 17408>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=257) | 1536 | 68.65 | 55% | 29.9 | 12% |
| 5120 | 17408 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<5120, 17408>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=257) | 1664 | 66.05 | 53% | 27.9 | 12% |
| 5120 | 17408 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<5120, 17408>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=257) | 1792 | 66.16 | 53% | 27.1 | 11% |
| 5120 | 17408 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<5120, 17408>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=257) | 1920 | 66.17 | 53% | 26.4 | 11% |
| 5120 | 17408 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<5120, 17408>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=257) | 2048 | 66.73 | 54% | 26.0 | 11% |
| 5120 | 17408 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<5120, 17408>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 1 | 0.81 | 0.2% | 229.2 | 95% |
| 5120 | 17408 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<5120, 17408>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 2 | 1.62 | 0.3% | 228.3 | 95% |
| 5120 | 17408 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<5120, 17408>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 3 | 2.42 | 0.5% | 227.4 | 95% |
| 5120 | 17408 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<5120, 17408>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 4 | 3.23 | 0.7% | 227.6 | 95% |
| 5120 | 17408 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<5120, 17408>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 5 | 4.05 | 0.8% | 228.7 | 95% |
| 5120 | 17408 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<5120, 17408>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 6 | 4.77 | 1.0% | 224.9 | 93% |
| 5120 | 17408 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<5120, 17408>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 7 | 5.59 | 1.1% | 226.2 | 94% |
| 5120 | 17408 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<5120, 17408>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 8 | 6.17 | 1.2% | 218.3 | 91% |
| 5120 | 17408 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<5120, 17408>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 12 | 9.25 | 1.9% | 219.1 | 91% |
| 5120 | 17408 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<5120, 17408>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 16 | 12.28 | 2.5% | 219.0 | 91% |
| 5120 | 17408 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<5120, 17408>, Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>>` (T 17-32) | 17 | 12.99 | 2.6% | 218.2 | 91% |
| 5120 | 17408 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<5120, 17408>, Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>>` (T 17-32) | 24 | 18.18 | 3.7% | 217.6 | 90% |
| 5120 | 17408 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<5120, 17408>, Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>>` (T 17-32) | 32 | 24.03 | 4.9% | 217.3 | 90% |
| 5120 | 17408 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<5120, 17408>, Nvfp4A4MmaSchedule<64, 64, 256, 2, 4, 2, 1>>` (T 33-48) | 33 | 24.78 | 5.0% | 217.5 | 90% |
| 5120 | 17408 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<5120, 17408>, Nvfp4A4MmaSchedule<64, 64, 256, 2, 4, 2, 1>>` (T 33-48) | 48 | 35.42 | 7.2% | 216.5 | 90% |
| 5120 | 17408 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<5120, 17408>, Nvfp4A4MmaSchedule<64, 128, 256, 4, 2, 2, 1>>` (T 49-64) | 49 | 36.16 | 7.3% | 216.7 | 90% |
| 5120 | 17408 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<5120, 17408>, Nvfp4A4MmaSchedule<64, 128, 256, 4, 2, 2, 1>>` (T 49-64) | 64 | 46.46 | 9.4% | 215.9 | 90% |
| 5120 | 17408 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<5120, 17408>, Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>>` (T 65-128) | 65 | 46.99 | 9.5% | 215.2 | 89% |
| 5120 | 17408 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<5120, 17408>, Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>>` (T 65-128) | 128 | 87.40 | 18% | 214.1 | 89% |
| 5120 | 17408 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<5120, 17408>, Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 1, 2>>` (T 129-256) | 129 | 84.47 | 17% | 205.5 | 85% |
| 5120 | 17408 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<5120, 17408>, Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 1, 2>>` (T 129-256) | 256 | 153.75 | 31% | 207.8 | 86% |
| 5120 | 17408 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 257 | 131.65 | 27% | 177.3 | 74% |
| 5120 | 17408 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 384 | 179.77 | 36% | 177.1 | 74% |
| 5120 | 17408 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 512 | 218.55 | 44% | 175.3 | 73% |
| 5120 | 17408 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 640 | 215.98 | 44% | 149.5 | 62% |
| 5120 | 17408 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 768 | 244.48 | 49% | 151.3 | 63% |
| 5120 | 17408 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 896 | 226.76 | 46% | 128.5 | 53% |
| 5120 | 17408 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1024 | 245.93 | 50% | 129.7 | 54% |
| 5120 | 17408 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1152 | 191.75 | 39% | 95.3 | 40% |
| 5120 | 17408 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1280 | 206.54 | 42% | 97.6 | 41% |
| 5120 | 17408 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1408 | 225.34 | 46% | 102.0 | 42% |
| 5120 | 17408 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1536 | 235.41 | 48% | 102.6 | 43% |
| 5120 | 17408 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1664 | 235.92 | 48% | 99.5 | 41% |
| 5120 | 17408 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1792 | 243.37 | 49% | 99.7 | 41% |
| 5120 | 17408 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1920 | 240.66 | 49% | 96.1 | 40% |
| 5120 | 17408 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 2048 | 249.16 | 50% | 97.2 | 40% |
| 6144 | 2560 | a16 | `nvfp4_linear_a16_gemv<Nvfp4Geometry<6144, 2560>, Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>>` (T=1) | 1 | 0.77 | 0.6% | 217.4 | 90% |
| 6144 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<6144, 2560>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 2 | 1.47 | 1.2% | 207.4 | 86% |
| 6144 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<6144, 2560>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 3 | 2.15 | 1.7% | 203.0 | 84% |
| 6144 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<6144, 2560>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 4 | 2.87 | 2.3% | 203.4 | 85% |
| 6144 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<6144, 2560>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 5 | 3.59 | 2.9% | 203.8 | 85% |
| 6144 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<6144, 2560>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 6 | 4.31 | 3.5% | 204.2 | 85% |
| 6144 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<6144, 2560>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 7 | 5.02 | 4.1% | 204.6 | 85% |
| 6144 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<6144, 2560>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 8 | 5.75 | 4.6% | 205.3 | 85% |
| 6144 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<6144, 2560>, Nvfp4SlicedInstance<16, 8, 2, 2>>` (T 9-16) | 9 | 6.31 | 5.1% | 200.7 | 83% |
| 6144 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<6144, 2560>, Nvfp4SlicedInstance<16, 8, 2, 2>>` (T 9-16) | 12 | 8.60 | 7.0% | 206.4 | 86% |
| 6144 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<6144, 2560>, Nvfp4SlicedInstance<16, 8, 2, 2>>` (T 9-16) | 16 | 11.47 | 9.3% | 208.0 | 86% |
| 6144 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<6144, 2560>, Nvfp4SlicedInstance<24, 4, 1, 2>>` (T 17-24) | 17 | 11.92 | 9.6% | 203.8 | 85% |
| 6144 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<6144, 2560>, Nvfp4SlicedInstance<24, 4, 1, 2>>` (T 17-24) | 24 | 16.82 | 14% | 206.4 | 86% |
| 6144 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<6144, 2560>, Nvfp4SlicedInstance<32, 4, 1, 2>>` (T 25-32) | 25 | 17.52 | 14% | 206.8 | 86% |
| 6144 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<6144, 2560>, Nvfp4SlicedInstance<32, 4, 1, 2>>` (T 25-32) | 32 | 22.42 | 18% | 209.5 | 87% |
| 6144 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<6144, 2560>, Nvfp4A16MmaSchedule<64, 64, 128, 32, 16, 1, 2>>` (T 33-64) | 33 | 19.55 | 16% | 177.5 | 74% |
| 6144 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<6144, 2560>, Nvfp4A16MmaSchedule<64, 64, 128, 32, 16, 1, 2>>` (T 33-64) | 48 | 28.44 | 23% | 182.4 | 76% |
| 6144 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<6144, 2560>, Nvfp4A16MmaSchedule<64, 64, 128, 32, 16, 1, 2>>` (T 33-64) | 64 | 36.51 | 30% | 180.7 | 75% |
| 6144 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<6144, 2560>, Nvfp4A16MmaSchedule<64, 96, 128, 32, 16, 1, 2>>` (T 65-96) | 65 | 33.38 | 27% | 162.9 | 68% |
| 6144 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<6144, 2560>, Nvfp4A16MmaSchedule<64, 96, 128, 32, 16, 1, 2>>` (T 65-96) | 96 | 47.71 | 39% | 166.2 | 69% |
| 6144 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<6144, 2560>, Nvfp4A16MmaSchedule<64, 128, 128, 32, 16, 2, 1>>` (T 97-128) | 97 | 43.94 | 36% | 151.7 | 63% |
| 6144 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<6144, 2560>, Nvfp4A16MmaSchedule<64, 128, 128, 32, 16, 2, 1>>` (T 97-128) | 128 | 57.93 | 47% | 159.4 | 66% |
| 6144 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<6144, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 129 | 35.44 | 29% | 96.9 | 40% |
| 6144 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<6144, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 256 | 69.00 | 56% | 114.0 | 47% |
| 6144 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<6144, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 384 | 71.14 | 58% | 91.5 | 38% |
| 6144 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<6144, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 512 | 72.88 | 59% | 80.4 | 33% |
| 6144 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<6144, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 640 | 73.97 | 60% | 73.4 | 31% |
| 6144 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<6144, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 768 | 75.17 | 61% | 69.1 | 29% |
| 6144 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<6144, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 896 | 75.66 | 61% | 65.6 | 27% |
| 6144 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<6144, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 1024 | 76.00 | 61% | 62.9 | 26% |
| 6144 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<6144, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 1152 | 75.97 | 61% | 60.6 | 25% |
| 6144 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<6144, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 1280 | 76.23 | 62% | 58.9 | 24% |
| 6144 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<6144, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 1408 | 76.17 | 62% | 57.4 | 24% |
| 6144 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<6144, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 1536 | 76.63 | 62% | 56.4 | 23% |
| 6144 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<6144, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 1664 | 76.99 | 62% | 55.6 | 23% |
| 6144 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<6144, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 1792 | 76.69 | 62% | 54.5 | 23% |
| 6144 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<6144, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 1920 | 76.82 | 62% | 53.8 | 22% |
| 6144 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<6144, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 2048 | 76.92 | 62% | 53.1 | 22% |
| 6144 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<6144, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 1 | 0.77 | 0.2% | 217.3 | 90% |
| 6144 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<6144, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 2 | 1.47 | 0.3% | 207.3 | 86% |
| 6144 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<6144, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 3 | 2.16 | 0.4% | 203.6 | 85% |
| 6144 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<6144, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 4 | 2.88 | 0.6% | 204.0 | 85% |
| 6144 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<6144, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 5 | 3.67 | 0.7% | 208.5 | 87% |
| 6144 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<6144, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 6 | 4.30 | 0.9% | 204.0 | 85% |
| 6144 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<6144, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 7 | 5.02 | 1.0% | 204.4 | 85% |
| 6144 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<6144, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 8 | 5.87 | 1.2% | 209.7 | 87% |
| 6144 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<6144, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 12 | 8.57 | 1.7% | 205.7 | 85% |
| 6144 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<6144, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 16 | 11.47 | 2.3% | 208.0 | 86% |
| 6144 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<6144, 2560>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 17-32) | 17 | 11.92 | 2.4% | 203.8 | 85% |
| 6144 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<6144, 2560>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 17-32) | 24 | 16.83 | 3.4% | 206.5 | 86% |
| 6144 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<6144, 2560>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 17-32) | 32 | 21.52 | 4.4% | 201.0 | 84% |
| 6144 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<6144, 2560>, Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>>` (T 33-64) | 33 | 22.61 | 4.6% | 205.2 | 85% |
| 6144 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<6144, 2560>, Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>>` (T 33-64) | 48 | 32.19 | 6.5% | 206.4 | 86% |
| 6144 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<6144, 2560>, Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>>` (T 33-64) | 64 | 41.94 | 8.5% | 207.5 | 86% |
| 6144 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<6144, 2560>, Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>>` (T 65-128) | 65 | 41.74 | 8.4% | 203.7 | 85% |
| 6144 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<6144, 2560>, Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>>` (T 65-128) | 128 | 82.24 | 17% | 226.2 | 94% |
| 6144 | 2560 | a4 | `nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>` (T 129-256) | 129 | 73.60 | 15% | 201.2 | 84% |
| 6144 | 2560 | a4 | `nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>` (T 129-256) | 256 | 143.40 | 29% | 236.9 | 98% |
| 6144 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 257 | 129.76 | 26% | 213.8 | 89% |
| 6144 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 384 | 181.92 | 37% | 233.9 | 97% |
| 6144 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 512 | 222.12 | 45% | 244.9 | 102% |
| 6144 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 640 | 214.07 | 43% | 212.5 | 88% |
| 6144 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 768 | 241.21 | 49% | 221.8 | 92% |
| 6144 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 896 | 222.31 | 45% | 192.8 | 80% |
| 6144 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1024 | 238.59 | 48% | 197.6 | 82% |
| 6144 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1152 | 224.69 | 45% | 179.2 | 74% |
| 6144 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1280 | 238.54 | 48% | 184.4 | 77% |
| 6144 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1408 | 226.68 | 46% | 170.7 | 71% |
| 6144 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1536 | 236.11 | 48% | 173.9 | 72% |
| 6144 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1664 | 229.42 | 46% | 165.7 | 69% |
| 6144 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1792 | 235.45 | 48% | 167.2 | 69% |
| 6144 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1920 | 227.76 | 46% | 159.4 | 66% |
| 6144 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 2048 | 233.88 | 47% | 161.5 | 67% |
| 10240 | 2560 | a16 | `nvfp4_linear_a16_gemv<Nvfp4Geometry<10240, 2560>, Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>>` (T=1) | 1 | 0.80 | 0.6% | 225.9 | 94% |
| 10240 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<10240, 2560>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 2 | 1.53 | 1.2% | 216.3 | 90% |
| 10240 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<10240, 2560>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 3 | 2.30 | 1.9% | 216.7 | 90% |
| 10240 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<10240, 2560>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 4 | 3.10 | 2.5% | 219.7 | 91% |
| 10240 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<10240, 2560>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 5 | 3.82 | 3.1% | 216.7 | 90% |
| 10240 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<10240, 2560>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 6 | 4.54 | 3.7% | 215.1 | 89% |
| 10240 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<10240, 2560>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 7 | 5.28 | 4.3% | 214.8 | 89% |
| 10240 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<10240, 2560>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 8 | 6.04 | 4.9% | 215.3 | 89% |
| 10240 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<10240, 2560>, Nvfp4SlicedInstance<16, 8, 2, 2>>` (T 9-16) | 9 | 6.79 | 5.5% | 215.4 | 90% |
| 10240 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<10240, 2560>, Nvfp4SlicedInstance<16, 8, 2, 2>>` (T 9-16) | 12 | 9.06 | 7.3% | 216.7 | 90% |
| 10240 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<10240, 2560>, Nvfp4SlicedInstance<16, 8, 2, 2>>` (T 9-16) | 16 | 12.07 | 9.8% | 218.1 | 91% |
| 10240 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<10240, 2560>, Nvfp4SlicedInstance<24, 4, 1, 2>>` (T 17-24) | 17 | 12.83 | 10% | 218.5 | 91% |
| 10240 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<10240, 2560>, Nvfp4SlicedInstance<24, 4, 1, 2>>` (T 17-24) | 24 | 17.86 | 14% | 218.0 | 91% |
| 10240 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<10240, 2560>, Nvfp4SlicedInstance<32, 4, 1, 2>>` (T 25-32) | 25 | 17.82 | 14% | 209.2 | 87% |
| 10240 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<10240, 2560>, Nvfp4SlicedInstance<32, 4, 1, 2>>` (T 25-32) | 32 | 22.80 | 18% | 211.6 | 88% |
| 10240 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<10240, 2560>, Nvfp4A16MmaSchedule<64, 64, 128, 32, 16, 1, 2>>` (T 33-64) | 33 | 19.24 | 16% | 173.4 | 72% |
| 10240 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<10240, 2560>, Nvfp4A16MmaSchedule<64, 64, 128, 32, 16, 1, 2>>` (T 33-64) | 48 | 28.64 | 23% | 181.8 | 76% |
| 10240 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<10240, 2560>, Nvfp4A16MmaSchedule<64, 64, 128, 32, 16, 1, 2>>` (T 33-64) | 64 | 38.14 | 31% | 186.2 | 77% |
| 10240 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<10240, 2560>, Nvfp4A16MmaSchedule<64, 96, 128, 64, 16, 1, 2>>` (T 65-96) | 65 | 32.68 | 26% | 157.3 | 65% |
| 10240 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<10240, 2560>, Nvfp4A16MmaSchedule<64, 96, 128, 64, 16, 1, 2>>` (T 65-96) | 96 | 45.58 | 37% | 155.8 | 65% |
| 10240 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<10240, 2560>, Nvfp4A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>` (T 97-128) | 97 | 40.79 | 33% | 138.2 | 57% |
| 10240 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<10240, 2560>, Nvfp4A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>` (T 97-128) | 128 | 53.79 | 43% | 144.4 | 60% |
| 10240 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<10240, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 129 | 35.55 | 29% | 94.9 | 39% |
| 10240 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<10240, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 256 | 69.05 | 56% | 109.6 | 46% |
| 10240 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<10240, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 384 | 73.90 | 60% | 90.2 | 37% |
| 10240 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<10240, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 512 | 72.86 | 59% | 75.6 | 31% |
| 10240 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<10240, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 640 | 75.19 | 61% | 69.8 | 29% |
| 10240 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<10240, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 768 | 76.23 | 62% | 65.1 | 27% |
| 10240 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<10240, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 896 | 75.47 | 61% | 60.5 | 25% |
| 10240 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<10240, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 1024 | 76.23 | 62% | 58.2 | 24% |
| 10240 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<10240, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 1152 | 77.02 | 62% | 56.4 | 23% |
| 10240 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<10240, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 1280 | 76.22 | 62% | 54.0 | 22% |
| 10240 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<10240, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 1408 | 76.53 | 62% | 52.7 | 22% |
| 10240 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<10240, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 1536 | 76.81 | 62% | 51.6 | 21% |
| 10240 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<10240, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 1664 | 76.49 | 62% | 50.3 | 21% |
| 10240 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<10240, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 1792 | 76.98 | 62% | 49.7 | 21% |
| 10240 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<10240, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 1920 | 77.18 | 62% | 49.0 | 20% |
| 10240 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<10240, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 2048 | 76.89 | 62% | 48.1 | 20% |
| 10240 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<10240, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 1 | 0.80 | 0.2% | 225.4 | 94% |
| 10240 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<10240, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 2 | 1.53 | 0.3% | 216.3 | 90% |
| 10240 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<10240, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 3 | 2.30 | 0.5% | 216.6 | 90% |
| 10240 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<10240, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 4 | 3.06 | 0.6% | 216.9 | 90% |
| 10240 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<10240, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 5 | 3.88 | 0.8% | 220.4 | 92% |
| 10240 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<10240, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 6 | 4.60 | 0.9% | 217.7 | 90% |
| 10240 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<10240, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 7 | 5.36 | 1.1% | 218.0 | 91% |
| 10240 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<10240, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 8 | 6.13 | 1.2% | 218.4 | 91% |
| 10240 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<10240, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 12 | 9.17 | 1.9% | 219.4 | 91% |
| 10240 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<10240, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 16 | 12.10 | 2.4% | 218.6 | 91% |
| 10240 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<10240, 2560>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 17-32) | 17 | 12.83 | 2.6% | 218.5 | 91% |
| 10240 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<10240, 2560>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 17-32) | 24 | 18.11 | 3.7% | 221.1 | 92% |
| 10240 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<10240, 2560>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 17-32) | 32 | 22.82 | 4.6% | 211.7 | 88% |
| 10240 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<10240, 2560>, Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>>` (T 33-64) | 33 | 21.44 | 4.3% | 193.2 | 80% |
| 10240 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<10240, 2560>, Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>>` (T 33-64) | 48 | 31.17 | 6.3% | 197.9 | 82% |
| 10240 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<10240, 2560>, Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>>` (T 33-64) | 64 | 41.04 | 8.3% | 200.4 | 83% |
| 10240 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<10240, 2560>, Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>>` (T 65-128) | 65 | 46.22 | 9.4% | 222.6 | 93% |
| 10240 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<10240, 2560>, Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>>` (T 65-128) | 128 | 90.01 | 18% | 241.7 | 100% |
| 10240 | 2560 | a4 | `nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>` (T 129-256) | 129 | 80.73 | 16% | 215.4 | 90% |
| 10240 | 2560 | a4 | `nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>` (T 129-256) | 256 | 154.49 | 31% | 245.2 | 102% |
| 10240 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 257 | 130.48 | 26% | 206.5 | 86% |
| 10240 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 384 | 185.75 | 38% | 226.7 | 94% |
| 10240 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 512 | 226.35 | 46% | 234.9 | 98% |
| 10240 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 640 | 227.85 | 46% | 211.4 | 88% |
| 10240 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 768 | 249.22 | 50% | 213.0 | 89% |
| 10240 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 896 | 230.53 | 47% | 184.9 | 77% |
| 10240 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1024 | 245.32 | 50% | 187.2 | 78% |
| 10240 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1152 | 230.54 | 47% | 168.9 | 70% |
| 10240 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1280 | 248.27 | 50% | 175.8 | 73% |
| 10240 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1408 | 238.83 | 48% | 164.3 | 68% |
| 10240 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1536 | 249.09 | 50% | 167.2 | 69% |
| 10240 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1664 | 242.16 | 49% | 159.2 | 66% |
| 10240 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1792 | 252.15 | 51% | 162.7 | 68% |
| 10240 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1920 | 240.44 | 49% | 152.6 | 63% |
| 10240 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 2048 | 250.50 | 51% | 156.7 | 65% |
| 12800 | 2560 | a16 | `nvfp4_linear_a16_gemv<Nvfp4Geometry<12800, 2560>, Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>>` (T=1) | 1 | 0.81 | 0.7% | 228.8 | 95% |
| 12800 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<12800, 2560>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 2 | 1.56 | 1.3% | 220.7 | 92% |
| 12800 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<12800, 2560>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 3 | 2.32 | 1.9% | 218.4 | 91% |
| 12800 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<12800, 2560>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 4 | 3.13 | 2.5% | 221.4 | 92% |
| 12800 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<12800, 2560>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 5 | 3.86 | 3.1% | 219.1 | 91% |
| 12800 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<12800, 2560>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 6 | 4.69 | 3.8% | 222.1 | 92% |
| 12800 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<12800, 2560>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 7 | 5.41 | 4.4% | 219.9 | 91% |
| 12800 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<12800, 2560>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 8 | 6.18 | 5.0% | 220.3 | 92% |
| 12800 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<12800, 2560>, Nvfp4SlicedInstance<16, 8, 2, 2>>` (T 9-16) | 9 | 6.83 | 5.5% | 216.8 | 90% |
| 12800 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<12800, 2560>, Nvfp4SlicedInstance<16, 8, 2, 2>>` (T 9-16) | 12 | 9.16 | 7.4% | 219.0 | 91% |
| 12800 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<12800, 2560>, Nvfp4SlicedInstance<16, 8, 2, 2>>` (T 9-16) | 16 | 12.22 | 9.9% | 220.5 | 92% |
| 12800 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<12800, 2560>, Nvfp4SlicedInstance<24, 4, 2, 2>>` (T 17-24) | 17 | 12.98 | 10% | 220.8 | 92% |
| 12800 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<12800, 2560>, Nvfp4SlicedInstance<24, 4, 2, 2>>` (T 17-24) | 24 | 18.10 | 15% | 220.6 | 92% |
| 12800 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<12800, 2560>, Nvfp4SlicedInstance<32, 4, 1, 2>>` (T 25-32) | 25 | 17.06 | 14% | 199.9 | 83% |
| 12800 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<12800, 2560>, Nvfp4SlicedInstance<32, 4, 1, 2>>` (T 25-32) | 32 | 21.82 | 18% | 202.0 | 84% |
| 12800 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<12800, 2560>, Nvfp4A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>>` (T 33-64) | 33 | 21.61 | 17% | 194.3 | 81% |
| 12800 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<12800, 2560>, Nvfp4A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>>` (T 33-64) | 48 | 31.40 | 25% | 198.7 | 83% |
| 12800 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<12800, 2560>, Nvfp4A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>>` (T 33-64) | 64 | 40.22 | 33% | 195.6 | 81% |
| 12800 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<12800, 2560>, Nvfp4A16MmaSchedule<64, 96, 128, 32, 16, 1, 2>>` (T 65-96) | 65 | 33.59 | 27% | 161.1 | 67% |
| 12800 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<12800, 2560>, Nvfp4A16MmaSchedule<64, 96, 128, 32, 16, 1, 2>>` (T 65-96) | 96 | 46.61 | 38% | 158.4 | 66% |
| 12800 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<12800, 2560>, Nvfp4A16MmaSchedule<64, 64, 64, 32, 16, 2, 2>>` (T 97-128) | 97 | 42.57 | 34% | 143.4 | 60% |
| 12800 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<12800, 2560>, Nvfp4A16MmaSchedule<64, 64, 64, 32, 16, 2, 2>>` (T 97-128) | 128 | 55.41 | 45% | 147.7 | 61% |
| 12800 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<12800, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 129 | 35.61 | 29% | 94.3 | 39% |
| 12800 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<12800, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 256 | 69.47 | 56% | 108.9 | 45% |
| 12800 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<12800, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 384 | 73.18 | 59% | 87.9 | 37% |
| 12800 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<12800, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 512 | 75.19 | 61% | 76.5 | 32% |
| 12800 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<12800, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 640 | 75.87 | 61% | 68.9 | 29% |
| 12800 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<12800, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 768 | 76.82 | 62% | 64.1 | 27% |
| 12800 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<12800, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 896 | 76.07 | 61% | 59.5 | 25% |
| 12800 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<12800, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 1024 | 76.74 | 62% | 57.0 | 24% |
| 12800 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<12800, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 1152 | 76.66 | 62% | 54.6 | 23% |
| 12800 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<12800, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 1280 | 77.00 | 62% | 53.0 | 22% |
| 12800 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<12800, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 1408 | 77.17 | 62% | 51.6 | 21% |
| 12800 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<12800, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 1536 | 77.41 | 63% | 50.5 | 21% |
| 12800 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<12800, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 1664 | 77.07 | 62% | 49.2 | 20% |
| 12800 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<12800, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 1792 | 77.29 | 62% | 48.4 | 20% |
| 12800 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<12800, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 1920 | 77.39 | 63% | 47.6 | 20% |
| 12800 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<12800, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 2048 | 75.24 | 61% | 45.6 | 19% |
| 12800 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<12800, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 1 | 0.81 | 0.2% | 228.8 | 95% |
| 12800 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<12800, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 2 | 1.55 | 0.3% | 218.5 | 91% |
| 12800 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<12800, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 3 | 2.34 | 0.5% | 220.7 | 92% |
| 12800 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<12800, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 4 | 3.13 | 0.6% | 221.4 | 92% |
| 12800 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<12800, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 5 | 3.86 | 0.8% | 219.1 | 91% |
| 12800 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<12800, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 6 | 4.64 | 0.9% | 219.5 | 91% |
| 12800 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<12800, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 7 | 5.41 | 1.1% | 219.8 | 91% |
| 12800 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<12800, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 8 | 6.18 | 1.3% | 220.1 | 91% |
| 12800 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<12800, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 12 | 9.16 | 1.9% | 219.1 | 91% |
| 12800 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<12800, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 16 | 12.22 | 2.5% | 220.5 | 92% |
| 12800 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<12800, 2560>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 17-32) | 17 | 12.83 | 2.6% | 218.2 | 91% |
| 12800 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<12800, 2560>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 17-32) | 24 | 18.29 | 3.7% | 222.9 | 93% |
| 12800 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<12800, 2560>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 17-32) | 32 | 22.30 | 4.5% | 206.4 | 86% |
| 12800 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<12800, 2560>, Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>>` (T 33-64) | 33 | 21.61 | 4.4% | 194.3 | 81% |
| 12800 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<12800, 2560>, Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>>` (T 33-64) | 48 | 31.08 | 6.3% | 196.7 | 82% |
| 12800 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<12800, 2560>, Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>>` (T 33-64) | 64 | 41.04 | 8.3% | 199.6 | 83% |
| 12800 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<12800, 2560>, Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>>` (T 65-128) | 65 | 43.40 | 8.8% | 208.2 | 87% |
| 12800 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<12800, 2560>, Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>>` (T 65-128) | 128 | 85.47 | 17% | 227.9 | 95% |
| 12800 | 2560 | a4 | `nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>` (T 129-256) | 129 | 81.87 | 17% | 216.9 | 90% |
| 12800 | 2560 | a4 | `nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>` (T 129-256) | 256 | 154.84 | 31% | 242.7 | 101% |
| 12800 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 257 | 132.81 | 27% | 207.6 | 86% |
| 12800 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 384 | 185.00 | 37% | 222.2 | 92% |
| 12800 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 512 | 223.01 | 45% | 227.0 | 94% |
| 12800 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 640 | 215.79 | 44% | 196.0 | 81% |
| 12800 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 768 | 239.99 | 49% | 200.4 | 83% |
| 12800 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 896 | 229.55 | 46% | 179.7 | 75% |
| 12800 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1024 | 247.45 | 50% | 184.0 | 76% |
| 12800 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1152 | 238.60 | 48% | 170.1 | 71% |
| 12800 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1280 | 249.92 | 51% | 172.1 | 72% |
| 12800 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1408 | 244.97 | 50% | 163.8 | 68% |
| 12800 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1536 | 252.18 | 51% | 164.4 | 68% |
| 12800 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1664 | 250.71 | 51% | 159.9 | 66% |
| 12800 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1792 | 261.94 | 53% | 163.9 | 68% |
| 12800 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1920 | 248.85 | 50% | 153.1 | 64% |
| 12800 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 2048 | 259.13 | 52% | 157.1 | 65% |
| 13952 | 2560 | a16 | `nvfp4_linear_a16_gemv<Nvfp4Geometry<13952, 2560>, Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>>` (T=1) | 1 | 0.82 | 0.7% | 231.4 | 96% |
| 13952 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<13952, 2560>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 2 | 1.57 | 1.3% | 221.7 | 92% |
| 13952 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<13952, 2560>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 3 | 2.36 | 1.9% | 222.2 | 92% |
| 13952 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<13952, 2560>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 4 | 3.18 | 2.6% | 224.7 | 93% |
| 13952 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<13952, 2560>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 5 | 3.97 | 3.2% | 225.3 | 94% |
| 13952 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<13952, 2560>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 6 | 4.71 | 3.8% | 223.1 | 93% |
| 13952 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<13952, 2560>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 7 | 5.50 | 4.4% | 223.6 | 93% |
| 13952 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<13952, 2560>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 8 | 6.27 | 5.1% | 223.4 | 93% |
| 13952 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<13952, 2560>, Nvfp4SlicedInstance<16, 8, 2, 2>>` (T 9-16) | 9 | 6.99 | 5.7% | 221.7 | 92% |
| 13952 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<13952, 2560>, Nvfp4SlicedInstance<16, 8, 2, 2>>` (T 9-16) | 12 | 9.42 | 7.6% | 225.2 | 94% |
| 13952 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<13952, 2560>, Nvfp4SlicedInstance<16, 8, 2, 2>>` (T 9-16) | 16 | 12.56 | 10% | 226.6 | 94% |
| 13952 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<13952, 2560>, Nvfp4SlicedInstance<32, 4, 1, 2>>` (T 17-32) | 17 | 12.37 | 10.0% | 210.4 | 87% |
| 13952 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<13952, 2560>, Nvfp4SlicedInstance<32, 4, 1, 2>>` (T 17-32) | 24 | 17.47 | 14% | 212.9 | 88% |
| 13952 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<13952, 2560>, Nvfp4SlicedInstance<32, 4, 1, 2>>` (T 17-32) | 32 | 22.86 | 18% | 211.5 | 88% |
| 13952 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<13952, 2560>, Nvfp4A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>>` (T 33-48) | 33 | 21.35 | 17% | 191.9 | 80% |
| 13952 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<13952, 2560>, Nvfp4A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>>` (T 33-48) | 48 | 30.49 | 25% | 192.8 | 80% |
| 13952 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<13952, 2560>, Nvfp4A16MmaSchedule<64, 64, 128, 32, 16, 2, 1>>` (T 49-64) | 49 | 29.01 | 23% | 179.9 | 75% |
| 13952 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<13952, 2560>, Nvfp4A16MmaSchedule<64, 64, 128, 32, 16, 2, 1>>` (T 49-64) | 64 | 38.54 | 31% | 187.2 | 78% |
| 13952 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<13952, 2560>, Nvfp4A16MmaSchedule<64, 96, 128, 32, 16, 1, 2>>` (T 65-96) | 65 | 36.04 | 29% | 172.6 | 72% |
| 13952 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<13952, 2560>, Nvfp4A16MmaSchedule<64, 96, 128, 32, 16, 1, 2>>` (T 65-96) | 96 | 50.81 | 41% | 172.3 | 72% |
| 13952 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<13952, 2560>, Nvfp4A16MmaSchedule<64, 64, 128, 32, 16, 1, 2>>` (T 97-128) | 97 | 43.43 | 35% | 146.0 | 61% |
| 13952 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<13952, 2560>, Nvfp4A16MmaSchedule<64, 64, 128, 32, 16, 1, 2>>` (T 97-128) | 128 | 55.91 | 45% | 148.7 | 62% |
| 13952 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<13952, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T 129-256) | 129 | 35.18 | 28% | 93.0 | 39% |
| 13952 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<13952, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T 129-256) | 256 | 69.26 | 56% | 108.1 | 45% |
| 13952 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<13952, 2560>, Nvfp4A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>` (T 257-512) | 257 | 58.97 | 48% | 91.8 | 38% |
| 13952 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<13952, 2560>, Nvfp4A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>` (T 257-512) | 384 | 74.87 | 61% | 89.4 | 37% |
| 13952 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<13952, 2560>, Nvfp4A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>` (T 257-512) | 512 | 75.06 | 61% | 75.9 | 32% |
| 13952 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<13952, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=513) | 513 | 60.66 | 49% | 61.3 | 25% |
| 13952 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<13952, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=513) | 640 | 76.74 | 62% | 69.2 | 29% |
| 13952 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<13952, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=513) | 768 | 76.34 | 62% | 63.2 | 26% |
| 13952 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<13952, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=513) | 896 | 77.76 | 63% | 60.4 | 25% |
| 13952 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<13952, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=513) | 1024 | 77.16 | 62% | 56.9 | 24% |
| 13952 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<13952, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=513) | 1152 | 77.88 | 63% | 55.0 | 23% |
| 13952 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<13952, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=513) | 1280 | 77.52 | 63% | 52.9 | 22% |
| 13952 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<13952, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=513) | 1408 | 77.71 | 63% | 51.4 | 21% |
| 13952 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<13952, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=513) | 1536 | 77.44 | 63% | 50.0 | 21% |
| 13952 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<13952, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=513) | 1664 | 77.50 | 63% | 48.9 | 20% |
| 13952 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<13952, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=513) | 1792 | 77.66 | 63% | 48.1 | 20% |
| 13952 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<13952, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=513) | 1920 | 77.61 | 63% | 47.2 | 20% |
| 13952 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<13952, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=513) | 2048 | 77.27 | 62% | 46.3 | 19% |
| 13952 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<13952, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 1 | 0.82 | 0.2% | 231.6 | 96% |
| 13952 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<13952, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 2 | 1.60 | 0.3% | 226.4 | 94% |
| 13952 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<13952, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 3 | 2.38 | 0.5% | 224.5 | 93% |
| 13952 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<13952, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 4 | 3.18 | 0.6% | 224.8 | 93% |
| 13952 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<13952, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 5 | 3.97 | 0.8% | 225.2 | 94% |
| 13952 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<13952, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 6 | 4.72 | 1.0% | 223.4 | 93% |
| 13952 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<13952, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 7 | 5.56 | 1.1% | 225.9 | 94% |
| 13952 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<13952, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 8 | 6.35 | 1.3% | 226.3 | 94% |
| 13952 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<13952, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 12 | 9.32 | 1.9% | 222.7 | 93% |
| 13952 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<13952, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 16 | 12.43 | 2.5% | 224.2 | 93% |
| 13952 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<13952, 2560>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 17-32) | 17 | 12.37 | 2.5% | 210.4 | 87% |
| 13952 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<13952, 2560>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 17-32) | 24 | 17.47 | 3.5% | 212.8 | 88% |
| 13952 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<13952, 2560>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 17-32) | 32 | 22.58 | 4.6% | 208.9 | 87% |
| 13952 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<13952, 2560>, Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>>` (T 33-64) | 33 | 23.03 | 4.7% | 206.9 | 86% |
| 13952 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<13952, 2560>, Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>>` (T 33-64) | 48 | 33.82 | 6.8% | 213.8 | 89% |
| 13952 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<13952, 2560>, Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>>` (T 33-64) | 64 | 43.85 | 8.9% | 213.0 | 89% |
| 13952 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<13952, 2560>, Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>>` (T 65-128) | 65 | 45.90 | 9.3% | 219.8 | 91% |
| 13952 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<13952, 2560>, Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>>` (T 65-128) | 128 | 91.29 | 18% | 242.8 | 101% |
| 13952 | 2560 | a4 | `nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>` (T 129-256) | 129 | 86.66 | 18% | 229.0 | 95% |
| 13952 | 2560 | a4 | `nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>` (T 129-256) | 256 | 158.26 | 32% | 247.0 | 103% |
| 13952 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 257 | 141.38 | 29% | 220.1 | 91% |
| 13952 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 384 | 191.68 | 39% | 229.0 | 95% |
| 13952 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 512 | 230.67 | 47% | 233.3 | 97% |
| 13952 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 640 | 226.81 | 46% | 204.5 | 85% |
| 13952 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 768 | 250.54 | 51% | 207.6 | 86% |
| 13952 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 896 | 227.39 | 46% | 176.5 | 73% |
| 13952 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1024 | 247.31 | 50% | 182.3 | 76% |
| 13952 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1152 | 242.17 | 49% | 171.1 | 71% |
| 13952 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1280 | 261.98 | 53% | 178.7 | 74% |
| 13952 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1408 | 240.26 | 49% | 159.1 | 66% |
| 13952 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1536 | 261.50 | 53% | 168.8 | 70% |
| 13952 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1664 | 253.54 | 51% | 160.1 | 67% |
| 13952 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1792 | 267.78 | 54% | 165.8 | 69% |
| 13952 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1920 | 250.44 | 51% | 152.5 | 63% |
| 13952 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 2048 | 254.68 | 52% | 152.7 | 63% |
| 14336 | 5120 | a16 | `nvfp4_linear_a16_gemv<Nvfp4Geometry<14336, 5120>, Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>>` (T=1) | 1 | 0.83 | 0.7% | 233.6 | 97% |
| 14336 | 5120 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<14336, 5120>, Nvfp4SlicedInstance<16, 8, 2, 2>>` (T 2-16) | 2 | 1.61 | 1.3% | 227.2 | 94% |
| 14336 | 5120 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<14336, 5120>, Nvfp4SlicedInstance<16, 8, 2, 2>>` (T 2-16) | 3 | 2.42 | 2.0% | 227.4 | 95% |
| 14336 | 5120 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<14336, 5120>, Nvfp4SlicedInstance<16, 8, 2, 2>>` (T 2-16) | 4 | 3.22 | 2.6% | 227.6 | 95% |
| 14336 | 5120 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<14336, 5120>, Nvfp4SlicedInstance<16, 8, 2, 2>>` (T 2-16) | 5 | 4.03 | 3.3% | 227.8 | 95% |
| 14336 | 5120 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<14336, 5120>, Nvfp4SlicedInstance<16, 8, 2, 2>>` (T 2-16) | 6 | 4.83 | 3.9% | 227.8 | 95% |
| 14336 | 5120 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<14336, 5120>, Nvfp4SlicedInstance<16, 8, 2, 2>>` (T 2-16) | 7 | 5.57 | 4.5% | 225.4 | 94% |
| 14336 | 5120 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<14336, 5120>, Nvfp4SlicedInstance<16, 8, 2, 2>>` (T 2-16) | 8 | 6.34 | 5.1% | 224.7 | 93% |
| 14336 | 5120 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<14336, 5120>, Nvfp4SlicedInstance<16, 8, 2, 2>>` (T 2-16) | 12 | 9.57 | 7.7% | 226.7 | 94% |
| 14336 | 5120 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<14336, 5120>, Nvfp4SlicedInstance<16, 8, 2, 2>>` (T 2-16) | 16 | 12.83 | 10% | 228.8 | 95% |
| 14336 | 5120 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<14336, 5120>, Nvfp4SlicedInstance<32, 4, 1>>` (T 17-32) | 17 | 12.97 | 10% | 218.1 | 91% |
| 14336 | 5120 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<14336, 5120>, Nvfp4SlicedInstance<32, 4, 1>>` (T 17-32) | 24 | 17.94 | 15% | 215.0 | 89% |
| 14336 | 5120 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<14336, 5120>, Nvfp4SlicedInstance<32, 4, 1>>` (T 17-32) | 32 | 23.42 | 19% | 212.1 | 88% |
| 14336 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<14336, 5120>, Nvfp4A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>>` (T 33-64) | 33 | 22.12 | 18% | 194.4 | 81% |
| 14336 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<14336, 5120>, Nvfp4A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>>` (T 33-64) | 48 | 31.31 | 25% | 191.7 | 80% |
| 14336 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<14336, 5120>, Nvfp4A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>>` (T 33-64) | 64 | 40.63 | 33% | 189.3 | 79% |
| 14336 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<14336, 5120>, Nvfp4A16MmaSchedule<64, 96, 128, 32, 16, 1, 2>>` (T 65-96) | 65 | 34.53 | 28% | 158.6 | 66% |
| 14336 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<14336, 5120>, Nvfp4A16MmaSchedule<64, 96, 128, 32, 16, 1, 2>>` (T 65-96) | 96 | 49.18 | 40% | 157.1 | 65% |
| 14336 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<14336, 5120>, Nvfp4A16MmaSchedule<64, 128, 128, 32, 16, 2, 1>>` (T 97-128) | 97 | 46.38 | 37% | 146.8 | 61% |
| 14336 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<14336, 5120>, Nvfp4A16MmaSchedule<64, 128, 128, 32, 16, 2, 1>>` (T 97-128) | 128 | 61.62 | 50% | 151.7 | 63% |
| 14336 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<14336, 5120>, Nvfp4A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>` (T>=129) | 129 | 50.28 | 41% | 122.9 | 51% |
| 14336 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<14336, 5120>, Nvfp4A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>` (T>=129) | 256 | 74.00 | 60% | 100.9 | 42% |
| 14336 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<14336, 5120>, Nvfp4A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>` (T>=129) | 384 | 79.34 | 64% | 79.1 | 33% |
| 14336 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<14336, 5120>, Nvfp4A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>` (T>=129) | 512 | 79.11 | 64% | 64.4 | 27% |
| 14336 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<14336, 5120>, Nvfp4A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>` (T>=129) | 640 | 78.56 | 64% | 55.3 | 23% |
| 14336 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<14336, 5120>, Nvfp4A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>` (T>=129) | 768 | 80.26 | 65% | 50.7 | 21% |
| 14336 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<14336, 5120>, Nvfp4A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>` (T>=129) | 896 | 79.68 | 64% | 46.1 | 19% |
| 14336 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<14336, 5120>, Nvfp4A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>` (T>=129) | 1024 | 78.47 | 63% | 42.4 | 18% |
| 14336 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<14336, 5120>, Nvfp4A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>` (T>=129) | 1152 | 79.78 | 64% | 40.6 | 17% |
| 14336 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<14336, 5120>, Nvfp4A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>` (T>=129) | 1280 | 79.31 | 64% | 38.4 | 16% |
| 14336 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<14336, 5120>, Nvfp4A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>` (T>=129) | 1408 | 75.58 | 61% | 35.1 | 15% |
| 14336 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<14336, 5120>, Nvfp4A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>` (T>=129) | 1536 | 76.65 | 62% | 34.4 | 14% |
| 14336 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<14336, 5120>, Nvfp4A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>` (T>=129) | 1664 | 76.49 | 62% | 33.2 | 14% |
| 14336 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<14336, 5120>, Nvfp4A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>` (T>=129) | 1792 | 75.34 | 61% | 31.8 | 13% |
| 14336 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<14336, 5120>, Nvfp4A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>` (T>=129) | 1920 | 75.87 | 61% | 31.2 | 13% |
| 14336 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<14336, 5120>, Nvfp4A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>` (T>=129) | 2048 | 75.46 | 61% | 30.4 | 13% |
| 14336 | 5120 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<14336, 5120>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 1-32) | 1 | 0.83 | 0.2% | 234.9 | 98% |
| 14336 | 5120 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<14336, 5120>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 1-32) | 2 | 1.62 | 0.3% | 228.4 | 95% |
| 14336 | 5120 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<14336, 5120>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 1-32) | 3 | 2.42 | 0.5% | 227.4 | 95% |
| 14336 | 5120 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<14336, 5120>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 1-32) | 4 | 3.17 | 0.6% | 223.8 | 93% |
| 14336 | 5120 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<14336, 5120>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 1-32) | 5 | 3.94 | 0.8% | 222.8 | 93% |
| 14336 | 5120 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<14336, 5120>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 1-32) | 6 | 4.67 | 0.9% | 220.4 | 92% |
| 14336 | 5120 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<14336, 5120>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 1-32) | 7 | 5.49 | 1.1% | 222.0 | 92% |
| 14336 | 5120 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<14336, 5120>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 1-32) | 8 | 6.31 | 1.3% | 223.5 | 93% |
| 14336 | 5120 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<14336, 5120>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 1-32) | 12 | 9.41 | 1.9% | 223.0 | 93% |
| 14336 | 5120 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<14336, 5120>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 1-32) | 16 | 12.61 | 2.6% | 225.1 | 94% |
| 14336 | 5120 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<14336, 5120>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 1-32) | 24 | 18.92 | 3.8% | 226.7 | 94% |
| 14336 | 5120 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<14336, 5120>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 1-32) | 32 | 25.07 | 5.1% | 227.0 | 94% |
| 14336 | 5120 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<14336, 5120>, Nvfp4A4MmaSchedule<64, 64, 256, 2, 4, 2, 1>>` (T 33-64) | 33 | 25.46 | 5.2% | 223.7 | 93% |
| 14336 | 5120 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<14336, 5120>, Nvfp4A4MmaSchedule<64, 64, 256, 2, 4, 2, 1>>` (T 33-64) | 48 | 36.84 | 7.5% | 225.6 | 94% |
| 14336 | 5120 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<14336, 5120>, Nvfp4A4MmaSchedule<64, 64, 256, 2, 4, 2, 1>>` (T 33-64) | 64 | 49.11 | 9.9% | 228.8 | 95% |
| 14336 | 5120 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<14336, 5120>, Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>>` (T 65-128) | 65 | 49.08 | 9.9% | 225.4 | 94% |
| 14336 | 5120 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<14336, 5120>, Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>>` (T 65-128) | 128 | 93.67 | 19% | 230.6 | 96% |
| 14336 | 5120 | a4 | `nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>` (T 129-256) | 129 | 86.07 | 17% | 210.5 | 87% |
| 14336 | 5120 | a4 | `nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>` (T 129-256) | 256 | 157.74 | 32% | 215.1 | 89% |
| 14336 | 5120 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 257 | 151.04 | 31% | 205.3 | 85% |
| 14336 | 5120 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 384 | 210.27 | 43% | 209.7 | 87% |
| 14336 | 5120 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 512 | 261.36 | 53% | 212.8 | 88% |
| 14336 | 5120 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 640 | 267.62 | 54% | 188.5 | 78% |
| 14336 | 5120 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 768 | 303.41 | 61% | 191.5 | 80% |
| 14336 | 5120 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 896 | 272.81 | 55% | 157.9 | 66% |
| 14336 | 5120 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1024 | 298.47 | 60% | 161.1 | 67% |
| 14336 | 5120 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1152 | 286.35 | 58% | 145.8 | 61% |
| 14336 | 5120 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1280 | 310.05 | 63% | 150.3 | 62% |
| 14336 | 5120 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1408 | 296.49 | 60% | 137.8 | 57% |
| 14336 | 5120 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1536 | 313.30 | 63% | 140.4 | 58% |
| 14336 | 5120 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1664 | 299.70 | 61% | 130.1 | 54% |
| 14336 | 5120 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1792 | 316.44 | 64% | 133.5 | 55% |
| 14336 | 5120 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1920 | 303.81 | 61% | 125.0 | 52% |
| 14336 | 5120 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 2048 | 318.84 | 65% | 128.3 | 53% |
| 16384 | 2560 | a16 | `nvfp4_linear_a16_gemv<Nvfp4Geometry<16384, 2560>, Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>>` (T=1) | 1 | 0.82 | 0.7% | 231.1 | 96% |
| 16384 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<16384, 2560>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 2 | 1.59 | 1.3% | 224.8 | 93% |
| 16384 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<16384, 2560>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 3 | 2.39 | 1.9% | 225.1 | 94% |
| 16384 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<16384, 2560>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 4 | 3.19 | 2.6% | 225.5 | 94% |
| 16384 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<16384, 2560>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 5 | 3.95 | 3.2% | 223.8 | 93% |
| 16384 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<16384, 2560>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 6 | 4.73 | 3.8% | 224.0 | 93% |
| 16384 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<16384, 2560>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 7 | 5.52 | 4.5% | 224.4 | 93% |
| 16384 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<16384, 2560>, Nvfp4SlicedInstance<8, 8, 2, 2>>` (T 2-8) | 8 | 6.31 | 5.1% | 224.8 | 93% |
| 16384 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<16384, 2560>, Nvfp4SlicedInstance<16, 8, 2, 2>>` (T 9-16) | 9 | 7.03 | 5.7% | 222.7 | 93% |
| 16384 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<16384, 2560>, Nvfp4SlicedInstance<16, 8, 2, 2>>` (T 9-16) | 12 | 9.38 | 7.6% | 224.0 | 93% |
| 16384 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<16384, 2560>, Nvfp4SlicedInstance<16, 8, 2, 2>>` (T 9-16) | 16 | 12.62 | 10% | 227.6 | 95% |
| 16384 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<16384, 2560>, Nvfp4SlicedInstance<24, 4, 2, 2>>` (T 17-24) | 17 | 12.92 | 10% | 219.5 | 91% |
| 16384 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<16384, 2560>, Nvfp4SlicedInstance<24, 4, 2, 2>>` (T 17-24) | 24 | 18.40 | 15% | 224.0 | 93% |
| 16384 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<16384, 2560>, Nvfp4SlicedInstance<32, 4, 1, 2>>` (T 25-32) | 25 | 18.98 | 15% | 222.2 | 92% |
| 16384 | 2560 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<16384, 2560>, Nvfp4SlicedInstance<32, 4, 1, 2>>` (T 25-32) | 32 | 23.89 | 19% | 220.7 | 92% |
| 16384 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<16384, 2560>, Nvfp4A16MmaSchedule<64, 64, 128, 32, 16, 1, 2>>` (T 33-64) | 33 | 21.49 | 17% | 192.8 | 80% |
| 16384 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<16384, 2560>, Nvfp4A16MmaSchedule<64, 64, 128, 32, 16, 1, 2>>` (T 33-64) | 48 | 31.25 | 25% | 197.2 | 82% |
| 16384 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<16384, 2560>, Nvfp4A16MmaSchedule<64, 64, 128, 32, 16, 1, 2>>` (T 33-64) | 64 | 41.01 | 33% | 198.7 | 83% |
| 16384 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<16384, 2560>, Nvfp4A16MmaSchedule<64, 96, 128, 64, 16, 1, 2>>` (T 65-96) | 65 | 36.53 | 30% | 174.6 | 73% |
| 16384 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<16384, 2560>, Nvfp4A16MmaSchedule<64, 96, 128, 64, 16, 1, 2>>` (T 65-96) | 96 | 53.93 | 44% | 182.4 | 76% |
| 16384 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<16384, 2560>, Nvfp4A16MmaSchedule<64, 64, 128, 32, 16, 1, 2>>` (T 97-128) | 97 | 46.25 | 37% | 155.0 | 64% |
| 16384 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<16384, 2560>, Nvfp4A16MmaSchedule<64, 64, 128, 32, 16, 1, 2>>` (T 97-128) | 128 | 59.64 | 48% | 158.0 | 66% |
| 16384 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<16384, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 129 | 37.50 | 30% | 98.7 | 41% |
| 16384 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<16384, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 256 | 73.41 | 59% | 113.8 | 47% |
| 16384 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<16384, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 384 | 76.76 | 62% | 90.9 | 38% |
| 16384 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<16384, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 512 | 76.01 | 61% | 76.1 | 32% |
| 16384 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<16384, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 640 | 77.58 | 63% | 69.1 | 29% |
| 16384 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<16384, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 768 | 77.69 | 63% | 63.5 | 26% |
| 16384 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<16384, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 896 | 77.61 | 63% | 59.4 | 25% |
| 16384 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<16384, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 1024 | 78.27 | 63% | 56.8 | 24% |
| 16384 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<16384, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 1152 | 78.39 | 63% | 54.5 | 23% |
| 16384 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<16384, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 1280 | 77.92 | 63% | 52.3 | 22% |
| 16384 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<16384, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 1408 | 78.16 | 63% | 50.9 | 21% |
| 16384 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<16384, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 1536 | 78.46 | 63% | 49.8 | 21% |
| 16384 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<16384, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 1664 | 78.17 | 63% | 48.5 | 20% |
| 16384 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<16384, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 1792 | 75.95 | 61% | 46.2 | 19% |
| 16384 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<16384, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 1920 | 75.92 | 61% | 45.4 | 19% |
| 16384 | 2560 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<16384, 2560>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=129) | 2048 | 75.65 | 61% | 44.6 | 19% |
| 16384 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<16384, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 1 | 0.82 | 0.2% | 230.8 | 96% |
| 16384 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<16384, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 2 | 1.58 | 0.3% | 222.6 | 93% |
| 16384 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<16384, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 3 | 2.39 | 0.5% | 225.0 | 94% |
| 16384 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<16384, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 4 | 3.16 | 0.6% | 223.4 | 93% |
| 16384 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<16384, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 5 | 3.95 | 0.8% | 223.7 | 93% |
| 16384 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<16384, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 6 | 4.73 | 1.0% | 223.7 | 93% |
| 16384 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<16384, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 7 | 5.48 | 1.1% | 222.6 | 93% |
| 16384 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<16384, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 8 | 6.37 | 1.3% | 226.9 | 94% |
| 16384 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<16384, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 12 | 9.47 | 1.9% | 226.1 | 94% |
| 16384 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<16384, 2560>, Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>` (T 1-16) | 16 | 12.62 | 2.6% | 227.6 | 95% |
| 16384 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<16384, 2560>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 17-32) | 17 | 13.03 | 2.6% | 221.5 | 92% |
| 16384 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<16384, 2560>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 17-32) | 24 | 18.07 | 3.7% | 219.9 | 91% |
| 16384 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<16384, 2560>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 17-32) | 32 | 23.87 | 4.8% | 220.5 | 92% |
| 16384 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<16384, 2560>, Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>>` (T 33-64) | 33 | 20.82 | 4.2% | 186.8 | 78% |
| 16384 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<16384, 2560>, Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>>` (T 33-64) | 48 | 31.25 | 6.3% | 197.2 | 82% |
| 16384 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<16384, 2560>, Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>>` (T 33-64) | 64 | 41.01 | 8.3% | 198.7 | 83% |
| 16384 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<16384, 2560>, Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>>` (T 65-128) | 65 | 48.46 | 9.8% | 231.6 | 96% |
| 16384 | 2560 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<16384, 2560>, Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>>` (T 65-128) | 128 | 93.78 | 19% | 248.4 | 103% |
| 16384 | 2560 | a4 | `nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>` (T 129-256) | 129 | 85.33 | 17% | 224.6 | 93% |
| 16384 | 2560 | a4 | `nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>` (T 129-256) | 256 | 153.25 | 31% | 237.6 | 99% |
| 16384 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 257 | 137.75 | 28% | 213.0 | 89% |
| 16384 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 384 | 188.54 | 38% | 223.3 | 93% |
| 16384 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 512 | 225.69 | 46% | 225.9 | 94% |
| 16384 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 640 | 230.11 | 47% | 205.1 | 85% |
| 16384 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 768 | 254.97 | 52% | 208.5 | 87% |
| 16384 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 896 | 240.78 | 49% | 184.3 | 77% |
| 16384 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1024 | 260.64 | 53% | 189.3 | 79% |
| 16384 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1152 | 248.39 | 50% | 172.8 | 72% |
| 16384 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1280 | 265.57 | 54% | 178.3 | 74% |
| 16384 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1408 | 250.29 | 51% | 163.0 | 68% |
| 16384 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1536 | 266.59 | 54% | 169.2 | 70% |
| 16384 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1664 | 258.25 | 52% | 160.3 | 67% |
| 16384 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1792 | 268.01 | 54% | 163.1 | 68% |
| 16384 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1920 | 253.75 | 51% | 151.8 | 63% |
| 16384 | 2560 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 2048 | 262.62 | 53% | 154.7 | 64% |
| 16384 | 5120 | a16 | `nvfp4_linear_a16_gemv<Nvfp4Geometry<16384, 5120>, Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>>` (T=1) | 1 | 0.83 | 0.7% | 233.2 | 97% |
| 16384 | 5120 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<16384, 5120>, Nvfp4SlicedInstance<16, 8, 2, 2>>` (T 2-16) | 2 | 1.60 | 1.3% | 225.4 | 94% |
| 16384 | 5120 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<16384, 5120>, Nvfp4SlicedInstance<16, 8, 2, 2>>` (T 2-16) | 3 | 2.40 | 1.9% | 225.6 | 94% |
| 16384 | 5120 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<16384, 5120>, Nvfp4SlicedInstance<16, 8, 2, 2>>` (T 2-16) | 4 | 3.20 | 2.6% | 225.8 | 94% |
| 16384 | 5120 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<16384, 5120>, Nvfp4SlicedInstance<16, 8, 2, 2>>` (T 2-16) | 5 | 3.98 | 3.2% | 224.9 | 93% |
| 16384 | 5120 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<16384, 5120>, Nvfp4SlicedInstance<16, 8, 2, 2>>` (T 2-16) | 6 | 4.77 | 3.9% | 224.7 | 93% |
| 16384 | 5120 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<16384, 5120>, Nvfp4SlicedInstance<16, 8, 2, 2>>` (T 2-16) | 7 | 5.57 | 4.5% | 225.3 | 94% |
| 16384 | 5120 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<16384, 5120>, Nvfp4SlicedInstance<16, 8, 2, 2>>` (T 2-16) | 8 | 6.36 | 5.1% | 225.4 | 94% |
| 16384 | 5120 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<16384, 5120>, Nvfp4SlicedInstance<16, 8, 2, 2>>` (T 2-16) | 12 | 9.51 | 7.7% | 225.2 | 94% |
| 16384 | 5120 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<16384, 5120>, Nvfp4SlicedInstance<16, 8, 2, 2>>` (T 2-16) | 16 | 12.67 | 10% | 226.0 | 94% |
| 16384 | 5120 | a16 | `nvfp4_linear_a16_sliced_k_dyn<Nvfp4Geometry<16384, 5120>, Nvfp4SlicedInstance<24, 4, 1, 2>>` (T 17-24) | 17 | 13.15 | 11% | 220.9 | 92% |
| 16384 | 5120 | a16 | `nvfp4_linear_a16_sliced_k_dyn<Nvfp4Geometry<16384, 5120>, Nvfp4SlicedInstance<24, 4, 1, 2>>` (T 17-24) | 24 | 18.48 | 15% | 221.3 | 92% |
| 16384 | 5120 | a16 | `nvfp4_linear_a16_sliced_k_dyn<Nvfp4Geometry<16384, 5120>, Nvfp4SlicedInstance<32, 4, 1, 2>>` (T 25-32) | 25 | 19.33 | 16% | 222.4 | 92% |
| 16384 | 5120 | a16 | `nvfp4_linear_a16_sliced_k_dyn<Nvfp4Geometry<16384, 5120>, Nvfp4SlicedInstance<32, 4, 1, 2>>` (T 25-32) | 32 | 24.29 | 20% | 219.7 | 91% |
| 16384 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<16384, 5120>, Nvfp4A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>>` (T 33-64) | 33 | 22.26 | 18% | 195.5 | 81% |
| 16384 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<16384, 5120>, Nvfp4A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>>` (T 33-64) | 48 | 31.73 | 26% | 194.1 | 81% |
| 16384 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<16384, 5120>, Nvfp4A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>>` (T 33-64) | 64 | 42.65 | 34% | 198.4 | 82% |
| 16384 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<16384, 5120>, Nvfp4A16MmaSchedule<64, 96, 128, 32, 16, 1, 2>>` (T 65-96) | 65 | 36.01 | 29% | 165.0 | 69% |
| 16384 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<16384, 5120>, Nvfp4A16MmaSchedule<64, 96, 128, 32, 16, 1, 2>>` (T 65-96) | 96 | 50.77 | 41% | 161.8 | 67% |
| 16384 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<16384, 5120>, Nvfp4A16MmaSchedule<64, 64, 128, 32, 16, 1, 2>>` (T 97-128) | 97 | 47.89 | 39% | 151.1 | 63% |
| 16384 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<16384, 5120>, Nvfp4A16MmaSchedule<64, 64, 128, 32, 16, 1, 2>>` (T 97-128) | 128 | 61.72 | 50% | 151.4 | 63% |
| 16384 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<16384, 5120>, Nvfp4A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>` (T>=129) | 129 | 51.09 | 41% | 124.5 | 52% |
| 16384 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<16384, 5120>, Nvfp4A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>` (T>=129) | 256 | 77.12 | 62% | 104.5 | 43% |
| 16384 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<16384, 5120>, Nvfp4A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>` (T>=129) | 384 | 80.06 | 65% | 79.2 | 33% |
| 16384 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<16384, 5120>, Nvfp4A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>` (T>=129) | 512 | 78.41 | 63% | 63.2 | 26% |
| 16384 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<16384, 5120>, Nvfp4A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>` (T>=129) | 640 | 79.93 | 65% | 55.6 | 23% |
| 16384 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<16384, 5120>, Nvfp4A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>` (T>=129) | 768 | 80.46 | 65% | 50.1 | 21% |
| 16384 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<16384, 5120>, Nvfp4A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>` (T>=129) | 896 | 78.80 | 64% | 44.9 | 19% |
| 16384 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<16384, 5120>, Nvfp4A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>` (T>=129) | 1024 | 79.44 | 64% | 42.2 | 18% |
| 16384 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<16384, 5120>, Nvfp4A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>` (T>=129) | 1152 | 79.91 | 65% | 40.0 | 17% |
| 16384 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<16384, 5120>, Nvfp4A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>` (T>=129) | 1280 | 76.04 | 61% | 36.2 | 15% |
| 16384 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<16384, 5120>, Nvfp4A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>` (T>=129) | 1408 | 75.94 | 61% | 34.6 | 14% |
| 16384 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<16384, 5120>, Nvfp4A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>` (T>=129) | 1536 | 75.87 | 61% | 33.3 | 14% |
| 16384 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<16384, 5120>, Nvfp4A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>` (T>=129) | 1664 | 75.67 | 61% | 32.2 | 13% |
| 16384 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<16384, 5120>, Nvfp4A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>` (T>=129) | 1792 | 75.69 | 61% | 31.3 | 13% |
| 16384 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<16384, 5120>, Nvfp4A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>` (T>=129) | 1920 | 75.80 | 61% | 30.5 | 13% |
| 16384 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<16384, 5120>, Nvfp4A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>` (T>=129) | 2048 | 75.78 | 61% | 29.8 | 12% |
| 16384 | 5120 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<16384, 5120>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 1-32) | 1 | 0.79 | 0.2% | 221.9 | 92% |
| 16384 | 5120 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<16384, 5120>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 1-32) | 2 | 1.58 | 0.3% | 222.1 | 92% |
| 16384 | 5120 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<16384, 5120>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 1-32) | 3 | 2.37 | 0.5% | 222.3 | 92% |
| 16384 | 5120 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<16384, 5120>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 1-32) | 4 | 3.15 | 0.6% | 222.5 | 92% |
| 16384 | 5120 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<16384, 5120>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 1-32) | 5 | 3.96 | 0.8% | 223.8 | 93% |
| 16384 | 5120 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<16384, 5120>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 1-32) | 6 | 4.69 | 0.9% | 220.8 | 92% |
| 16384 | 5120 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<16384, 5120>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 1-32) | 7 | 5.49 | 1.1% | 221.9 | 92% |
| 16384 | 5120 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<16384, 5120>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 1-32) | 8 | 6.28 | 1.3% | 222.3 | 92% |
| 16384 | 5120 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<16384, 5120>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 1-32) | 12 | 9.46 | 1.9% | 224.2 | 93% |
| 16384 | 5120 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<16384, 5120>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 1-32) | 16 | 12.61 | 2.6% | 224.9 | 93% |
| 16384 | 5120 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<16384, 5120>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 1-32) | 24 | 18.83 | 3.8% | 225.5 | 94% |
| 16384 | 5120 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<16384, 5120>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 1-32) | 32 | 24.87 | 5.0% | 225.0 | 94% |
| 16384 | 5120 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<16384, 5120>, Nvfp4A4MmaSchedule<64, 64, 256, 2, 4, 2, 1>>` (T 33-64) | 33 | 25.42 | 5.1% | 223.1 | 93% |
| 16384 | 5120 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<16384, 5120>, Nvfp4A4MmaSchedule<64, 64, 256, 2, 4, 2, 1>>` (T 33-64) | 48 | 36.95 | 7.5% | 226.0 | 94% |
| 16384 | 5120 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<16384, 5120>, Nvfp4A4MmaSchedule<64, 64, 256, 2, 4, 2, 1>>` (T 33-64) | 64 | 48.81 | 9.9% | 227.0 | 94% |
| 16384 | 5120 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<16384, 5120>, Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>>` (T 65-128) | 65 | 49.34 | 10.0% | 226.1 | 94% |
| 16384 | 5120 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<16384, 5120>, Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>>` (T 65-128) | 128 | 94.11 | 19% | 230.9 | 96% |
| 16384 | 5120 | a4 | `nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>` (T 129-256) | 129 | 85.63 | 17% | 208.7 | 87% |
| 16384 | 5120 | a4 | `nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>` (T 129-256) | 256 | 156.50 | 32% | 212.1 | 88% |
| 16384 | 5120 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 257 | 145.78 | 29% | 196.9 | 82% |
| 16384 | 5120 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 384 | 204.37 | 41% | 202.1 | 84% |
| 16384 | 5120 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 512 | 254.32 | 51% | 204.9 | 85% |
| 16384 | 5120 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 640 | 274.61 | 56% | 191.1 | 79% |
| 16384 | 5120 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 768 | 308.59 | 62% | 192.1 | 80% |
| 16384 | 5120 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 896 | 282.43 | 57% | 161.1 | 67% |
| 16384 | 5120 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1024 | 310.28 | 63% | 164.8 | 68% |
| 16384 | 5120 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1152 | 290.89 | 59% | 145.6 | 61% |
| 16384 | 5120 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1280 | 313.56 | 63% | 149.3 | 62% |
| 16384 | 5120 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1408 | 300.04 | 61% | 136.8 | 57% |
| 16384 | 5120 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1536 | 317.83 | 64% | 139.7 | 58% |
| 16384 | 5120 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1664 | 307.08 | 62% | 130.6 | 54% |
| 16384 | 5120 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1792 | 323.05 | 65% | 133.5 | 55% |
| 16384 | 5120 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1920 | 306.95 | 62% | 123.6 | 51% |
| 16384 | 5120 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 2048 | 319.95 | 65% | 126.0 | 52% |
| 34816 | 5120 | a16 | `nvfp4_linear_a16_gemv<Nvfp4Geometry<34816, 5120>, Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>>` (T=1) | 1 | 0.84 | 0.7% | 236.8 | 98% |
| 34816 | 5120 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<34816, 5120>, Nvfp4SlicedInstance<8, 8, 2>>` (T 2-8) | 2 | 1.65 | 1.3% | 232.5 | 97% |
| 34816 | 5120 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<34816, 5120>, Nvfp4SlicedInstance<8, 8, 2>>` (T 2-8) | 3 | 2.46 | 2.0% | 230.8 | 96% |
| 34816 | 5120 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<34816, 5120>, Nvfp4SlicedInstance<8, 8, 2>>` (T 2-8) | 4 | 3.29 | 2.7% | 232.2 | 97% |
| 34816 | 5120 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<34816, 5120>, Nvfp4SlicedInstance<8, 8, 2>>` (T 2-8) | 5 | 4.10 | 3.3% | 231.4 | 96% |
| 34816 | 5120 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<34816, 5120>, Nvfp4SlicedInstance<8, 8, 2>>` (T 2-8) | 6 | 4.92 | 4.0% | 231.6 | 96% |
| 34816 | 5120 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<34816, 5120>, Nvfp4SlicedInstance<8, 8, 2>>` (T 2-8) | 7 | 5.74 | 4.6% | 231.8 | 96% |
| 34816 | 5120 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<34816, 5120>, Nvfp4SlicedInstance<8, 8, 2>>` (T 2-8) | 8 | 6.54 | 5.3% | 231.4 | 96% |
| 34816 | 5120 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<34816, 5120>, Nvfp4SlicedInstance<16, 8, 2, 2>>` (T 9-16) | 9 | 7.29 | 5.9% | 229.5 | 95% |
| 34816 | 5120 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<34816, 5120>, Nvfp4SlicedInstance<16, 8, 2, 2>>` (T 9-16) | 12 | 9.77 | 7.9% | 231.1 | 96% |
| 34816 | 5120 | a16 | `nvfp4_linear_a16_sliced_k<Nvfp4Geometry<34816, 5120>, Nvfp4SlicedInstance<16, 8, 2, 2>>` (T 9-16) | 16 | 12.99 | 11% | 231.2 | 96% |
| 34816 | 5120 | a16 | `nvfp4_linear_a16_sliced_k_dyn<Nvfp4Geometry<34816, 5120>, Nvfp4SlicedInstance<24, 4, 2, 2>>` (T 17-24) | 17 | 13.58 | 11% | 227.7 | 95% |
| 34816 | 5120 | a16 | `nvfp4_linear_a16_sliced_k_dyn<Nvfp4Geometry<34816, 5120>, Nvfp4SlicedInstance<24, 4, 2, 2>>` (T 17-24) | 24 | 18.95 | 15% | 226.3 | 94% |
| 34816 | 5120 | a16 | `nvfp4_linear_a16_sliced_k_dyn<Nvfp4Geometry<34816, 5120>, Nvfp4SlicedInstance<32, 4, 1, 2>>` (T 25-32) | 25 | 19.61 | 16% | 225.0 | 94% |
| 34816 | 5120 | a16 | `nvfp4_linear_a16_sliced_k_dyn<Nvfp4Geometry<34816, 5120>, Nvfp4SlicedInstance<32, 4, 1, 2>>` (T 25-32) | 32 | 24.88 | 20% | 224.2 | 93% |
| 34816 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<34816, 5120>, Nvfp4A16MmaSchedule<64, 64, 128, 32, 16, 1, 2>>` (T 33-48) | 33 | 22.98 | 19% | 201.0 | 84% |
| 34816 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<34816, 5120>, Nvfp4A16MmaSchedule<64, 64, 128, 32, 16, 1, 2>>` (T 33-48) | 48 | 33.17 | 27% | 201.8 | 84% |
| 34816 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<34816, 5120>, Nvfp4A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>>` (T 49-64) | 49 | 34.41 | 28% | 205.2 | 85% |
| 34816 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<34816, 5120>, Nvfp4A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>>` (T 49-64) | 64 | 43.88 | 35% | 202.6 | 84% |
| 34816 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<34816, 5120>, Nvfp4A16MmaSchedule<64, 96, 128, 32, 16, 1, 2>>` (T 65-96) | 65 | 38.90 | 31% | 177.0 | 74% |
| 34816 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<34816, 5120>, Nvfp4A16MmaSchedule<64, 96, 128, 32, 16, 1, 2>>` (T 65-96) | 96 | 52.40 | 42% | 165.3 | 69% |
| 34816 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<34816, 5120>, Nvfp4A16MmaSchedule<64, 64, 128, 32, 16, 1, 2>>` (T 97-128) | 97 | 50.72 | 41% | 158.4 | 66% |
| 34816 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<34816, 5120>, Nvfp4A16MmaSchedule<64, 64, 128, 32, 16, 1, 2>>` (T 97-128) | 128 | 65.74 | 53% | 159.2 | 66% |
| 34816 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<34816, 5120>, Nvfp4A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>` (T 129-384) | 129 | 52.24 | 42% | 125.6 | 52% |
| 34816 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<34816, 5120>, Nvfp4A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>` (T 129-384) | 256 | 79.32 | 64% | 104.9 | 44% |
| 34816 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<34816, 5120>, Nvfp4A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>` (T 129-384) | 384 | 80.84 | 65% | 77.3 | 32% |
| 34816 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<34816, 5120>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=385) | 385 | 58.26 | 47% | 55.6 | 23% |
| 34816 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<34816, 5120>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=385) | 512 | 79.37 | 64% | 61.4 | 26% |
| 34816 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<34816, 5120>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=385) | 640 | 79.47 | 64% | 52.7 | 22% |
| 34816 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<34816, 5120>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=385) | 768 | 79.58 | 64% | 47.0 | 20% |
| 34816 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<34816, 5120>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=385) | 896 | 77.21 | 62% | 41.5 | 17% |
| 34816 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<34816, 5120>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=385) | 1024 | 75.28 | 61% | 37.5 | 16% |
| 34816 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<34816, 5120>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=385) | 1152 | 76.07 | 61% | 35.6 | 15% |
| 34816 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<34816, 5120>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=385) | 1280 | 75.75 | 61% | 33.6 | 14% |
| 34816 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<34816, 5120>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=385) | 1408 | 75.65 | 61% | 32.1 | 13% |
| 34816 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<34816, 5120>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=385) | 1536 | 75.69 | 61% | 30.8 | 13% |
| 34816 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<34816, 5120>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=385) | 1664 | 75.55 | 61% | 29.7 | 12% |
| 34816 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<34816, 5120>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=385) | 1792 | 75.62 | 61% | 28.8 | 12% |
| 34816 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<34816, 5120>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=385) | 1920 | 75.59 | 61% | 28.0 | 12% |
| 34816 | 5120 | a16 | `nvfp4_linear_a16_mma<Nvfp4Geometry<34816, 5120>, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>` (T>=385) | 2048 | 75.23 | 61% | 27.2 | 11% |
| 34816 | 5120 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<34816, 5120>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 1-24) | 1 | 0.81 | 0.2% | 226.9 | 94% |
| 34816 | 5120 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<34816, 5120>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 1-24) | 2 | 1.62 | 0.3% | 227.6 | 95% |
| 34816 | 5120 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<34816, 5120>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 1-24) | 3 | 2.41 | 0.5% | 226.8 | 94% |
| 34816 | 5120 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<34816, 5120>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 1-24) | 4 | 3.19 | 0.6% | 224.7 | 93% |
| 34816 | 5120 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<34816, 5120>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 1-24) | 5 | 3.99 | 0.8% | 225.1 | 94% |
| 34816 | 5120 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<34816, 5120>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 1-24) | 6 | 4.77 | 1.0% | 224.6 | 93% |
| 34816 | 5120 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<34816, 5120>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 1-24) | 7 | 5.58 | 1.1% | 225.3 | 94% |
| 34816 | 5120 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<34816, 5120>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 1-24) | 8 | 6.36 | 1.3% | 225.0 | 94% |
| 34816 | 5120 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<34816, 5120>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 1-24) | 12 | 9.59 | 1.9% | 226.8 | 94% |
| 34816 | 5120 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<34816, 5120>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 1-24) | 16 | 12.78 | 2.6% | 227.5 | 95% |
| 34816 | 5120 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<34816, 5120>, Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>` (T 1-24) | 24 | 18.95 | 3.8% | 226.4 | 94% |
| 34816 | 5120 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<34816, 5120>, Nvfp4A4MmaSchedule<64, 128, 256, 4, 2, 2, 1>>` (T 25-32) | 25 | 19.66 | 4.0% | 225.5 | 94% |
| 34816 | 5120 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<34816, 5120>, Nvfp4A4MmaSchedule<64, 128, 256, 4, 2, 2, 1>>` (T 25-32) | 32 | 24.99 | 5.1% | 225.2 | 94% |
| 34816 | 5120 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<34816, 5120>, Nvfp4A4MmaSchedule<64, 64, 256, 2, 4, 2, 1>>` (T 33-64) | 33 | 25.83 | 5.2% | 225.9 | 94% |
| 34816 | 5120 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<34816, 5120>, Nvfp4A4MmaSchedule<64, 64, 256, 2, 4, 2, 1>>` (T 33-64) | 48 | 37.15 | 7.5% | 226.0 | 94% |
| 34816 | 5120 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<34816, 5120>, Nvfp4A4MmaSchedule<64, 64, 256, 2, 4, 2, 1>>` (T 33-64) | 64 | 49.21 | 10.0% | 227.3 | 94% |
| 34816 | 5120 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<34816, 5120>, Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>>` (T 65-128) | 65 | 49.75 | 10% | 226.4 | 94% |
| 34816 | 5120 | a4 | `nvfp4_a4_mma_route<Nvfp4Geometry<34816, 5120>, Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>>` (T 65-128) | 128 | 94.06 | 19% | 227.8 | 95% |
| 34816 | 5120 | a4 | `nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>` (T 129-256) | 129 | 86.73 | 18% | 208.5 | 87% |
| 34816 | 5120 | a4 | `nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>` (T 129-256) | 256 | 157.24 | 32% | 208.0 | 86% |
| 34816 | 5120 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 257 | 152.47 | 31% | 201.0 | 84% |
| 34816 | 5120 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 384 | 210.93 | 43% | 201.7 | 84% |
| 34816 | 5120 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 512 | 264.17 | 53% | 204.3 | 85% |
| 34816 | 5120 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 640 | 285.73 | 58% | 189.6 | 79% |
| 34816 | 5120 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 768 | 325.78 | 66% | 192.3 | 80% |
| 34816 | 5120 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 896 | 303.76 | 61% | 163.4 | 68% |
| 34816 | 5120 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1024 | 335.13 | 68% | 167.1 | 69% |
| 34816 | 5120 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1152 | 319.63 | 65% | 149.6 | 62% |
| 34816 | 5120 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1280 | 347.12 | 70% | 154.0 | 64% |
| 34816 | 5120 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1408 | 322.55 | 65% | 136.7 | 57% |
| 34816 | 5120 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1536 | 345.74 | 70% | 140.8 | 59% |
| 34816 | 5120 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1664 | 331.09 | 67% | 130.1 | 54% |
| 34816 | 5120 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1792 | 352.10 | 71% | 134.1 | 56% |
| 34816 | 5120 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 1920 | 324.21 | 66% | 120.1 | 50% |
| 34816 | 5120 | a4 | `nvfp4_a4_tma_route<>` (T>=257) | 2048 | 341.84 | 69% | 123.5 | 51% |
