# Linear kernels — FP8 (GB10)

Part of the [Linear kernel tables](README.md).

Measured 2026-11-02 at `f2a495e3c` (dev-rebased-20261028, 3 uncommitted file(s) (source edits: `bench/README.md`, `bench/ops/nvfp4_route_sweep.cu`, `docs/README.md`)) with `build-native/bench/infernix_linear_bench --qtype <qtype> --policy <policy> --n N --k K --sweep … --warmup 3 --repeat 12 --flush-mib 48` (cold-L2: the eviction buffer is read between samples; the default `exact` route set).

One row per token count `T`. `kernel` is the schedule the dispatch table in `src/ops/linear/fp8/shapes/` selects at that `T`, with its dispatch band in parentheses: the shape file's own launch call, with its `using` aliases and `constexpr` constants substituted and literal ternaries folded where the result fits the cell (a parameterized alias such as `Simt<2>`, or one whose expansion is too long, stays as the source writes it). `TFLOP/s` = 2·N·K·T / median; `GB/s` = (weight + activation + output bytes) / median.

`%pipe` and `%mem` put each row against the envelope that governs it, so a shortfall is attributable rather than a gap to a single device number. `%pipe` divides by the measured MMA-pipe peak of the pipe that policy actually runs on (`a16` → BF16 HMMA 123.71 TFLOP/s; `a8` → FP8 QMMA 244.03 TFLOP/s); the packed weight format is not the ceiling — an A16 route dequantizes its stored weights and lands on the BF16 HMMA pipe whatever the codec. `%mem` divides by 240.6 GB/s, the `ckt env` measured read-only ceiling. A row is at its envelope when the percentage of the envelope that binds it — `mem` at decode widths, `pipe` at prefill widths — is near 100.

Two regimes keep both percentages far below 100 and no schedule closes them. First, a payload smaller than the bandwidth × memory-latency product (~295 KB here) cannot reach the read ceiling at all: the whole tensor must be in flight before the stream ends. Second, the timed window is the whole public Op, so a route that composes several launches pays every one of them.

Repeatability: each row is the better median of two independent cold-L2 passes. Across these 666 rows the passes differ by 0.43% (median), 2.78% (p90) and 32.18% (worst), so a single pass is not a repeatable number for the microsecond rows. Every pass ran with a quiet host: host-side memory work (a build, a `cuobjdump` of a large binary) shares GB10's LPDDR5X with the GPU and inflates measured kernel times.

Not covered: the `--routes fast` tiers (Q8 sliced-K, FP8 A8 pair, Q4 W4A8, NVFP4 A4 pair), the `invariant` policy, and route entries the benchmark never admits — the f16-accumulate route (`LinearRoutes::F16Accumulate`, `launch_f16`) and `launch_pair_block32`. A row with empty metrics is a policy band above the measured cap (`T > 2048`): production dispatch selects that schedule there, but the sweep did not time it.

Those [3072, 1024], [248320, 3072] problems are measured with `--qtype fp8f32`; every other shape here uses `--qtype fp8`.

| N | K | policy | kernel | T | TFLOP/s | %pipe | GB/s | %mem |
|---:|---:|---|---|---:|---:|---:|---:|---:|
| 1280 | 2560 | a16 | `fp8_linear_a16_gemv<Fp8Geometry<1280, 2560>, Fp8A16GemvSchedule<8, 2, 16, 1, Fp8CodeCache::Default, 2, 2>>` (T=1) | 1 | 0.38 | 0.3% | 190.9 | 79% |
| 1280 | 2560 | a16 | `fp8_linear_a16_simt<Fp8Geometry<1280, 2560>, 8, Fp8A16SimtSchedule<4, 2, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked, Fp8CodeCache::Streaming, 1, Fp8SimtBlockOrder::RowsContiguous, 1>>` (T 2-8) | 2 | 0.72 | 0.6% | 180.3 | 75% |
| 1280 | 2560 | a16 | `fp8_linear_a16_simt<Fp8Geometry<1280, 2560>, 8, Fp8A16SimtSchedule<4, 2, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked, Fp8CodeCache::Streaming, 1, Fp8SimtBlockOrder::RowsContiguous, 1>>` (T 2-8) | 3 | 1.08 | 0.9% | 180.7 | 75% |
| 1280 | 2560 | a16 | `fp8_linear_a16_simt<Fp8Geometry<1280, 2560>, 8, Fp8A16SimtSchedule<4, 2, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked, Fp8CodeCache::Streaming, 1, Fp8SimtBlockOrder::RowsContiguous, 1>>` (T 2-8) | 4 | 1.43 | 1.2% | 181.2 | 75% |
| 1280 | 2560 | a16 | `fp8_linear_a16_simt<Fp8Geometry<1280, 2560>, 8, Fp8A16SimtSchedule<4, 2, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked, Fp8CodeCache::Streaming, 1, Fp8SimtBlockOrder::RowsContiguous, 1>>` (T 2-8) | 5 | 1.79 | 1.4% | 181.6 | 75% |
| 1280 | 2560 | a16 | `fp8_linear_a16_simt<Fp8Geometry<1280, 2560>, 8, Fp8A16SimtSchedule<4, 2, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked, Fp8CodeCache::Streaming, 1, Fp8SimtBlockOrder::RowsContiguous, 1>>` (T 2-8) | 6 | 2.15 | 1.7% | 181.7 | 76% |
| 1280 | 2560 | a16 | `fp8_linear_a16_simt<Fp8Geometry<1280, 2560>, 8, Fp8A16SimtSchedule<4, 2, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked, Fp8CodeCache::Streaming, 1, Fp8SimtBlockOrder::RowsContiguous, 1>>` (T 2-8) | 7 | 2.27 | 1.8% | 165.1 | 69% |
| 1280 | 2560 | a16 | `fp8_linear_a16_simt<Fp8Geometry<1280, 2560>, 8, Fp8A16SimtSchedule<4, 2, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked, Fp8CodeCache::Streaming, 1, Fp8SimtBlockOrder::RowsContiguous, 1>>` (T 2-8) | 8 | 2.58 | 2.1% | 164.4 | 68% |
| 1280 | 2560 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<1280, 2560>, Fp8SlicedInstance<16, 8, 2>>` (T 9-16) | 9 | 3.23 | 2.6% | 183.3 | 76% |
| 1280 | 2560 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<1280, 2560>, Fp8SlicedInstance<16, 8, 2>>` (T 9-16) | 12 | 4.30 | 3.5% | 184.5 | 77% |
| 1280 | 2560 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<1280, 2560>, Fp8SlicedInstance<16, 8, 2>>` (T 9-16) | 16 | 5.44 | 4.4% | 176.6 | 73% |
| 1280 | 2560 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<1280, 2560>, Fp8SlicedInstance<32, 4, 2>>` (T 17-32) | 17 | 6.05 | 4.9% | 185.3 | 77% |
| 1280 | 2560 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<1280, 2560>, Fp8SlicedInstance<32, 4, 2>>` (T 17-32) | 24 | 7.79 | 6.3% | 171.5 | 71% |
| 1280 | 2560 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<1280, 2560>, Fp8SlicedInstance<32, 4, 2>>` (T 17-32) | 32 | 10.34 | 8.4% | 173.8 | 72% |
| 1280 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<1280, 2560>, Fp8A16TmaT32R64>` (T 33-128) | 33 | 10.64 | 8.6% | 173.9 | 72% |
| 1280 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<1280, 2560>, Fp8A16TmaT32R64>` (T 33-128) | 48 | 15.51 | 13% | 179.8 | 75% |
| 1280 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<1280, 2560>, Fp8A16TmaT32R64>` (T 33-128) | 64 | 20.67 | 17% | 185.9 | 77% |
| 1280 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<1280, 2560>, Fp8A16TmaT32R64>` (T 33-128) | 128 | 31.74 | 26% | 161.3 | 67% |
| 1280 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<1280, 2560>, Fp8A16TmaT64R128>` (T 129-512) | 129 | 31.98 | 26% | 161.6 | 67% |
| 1280 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<1280, 2560>, Fp8A16TmaT64R128>` (T 129-512) | 256 | 58.91 | 48% | 184.2 | 77% |
| 1280 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<1280, 2560>, Fp8A16TmaT64R128>` (T 129-512) | 384 | 49.31 | 40% | 122.0 | 51% |
| 1280 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<1280, 2560>, Fp8A16TmaT64R128>` (T 129-512) | 512 | 63.21 | 51% | 135.8 | 56% |
| 1280 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<1280, 2560>, Fp8A16TmaT64R128S3B2K64>` (T 513-1536) | 513 | 68.67 | 56% | 147.5 | 61% |
| 1280 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<1280, 2560>, Fp8A16TmaT64R128S3B2K64>` (T 513-1536) | 640 | 58.67 | 47% | 114.6 | 48% |
| 1280 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<1280, 2560>, Fp8A16TmaT64R128S3B2K64>` (T 513-1536) | 768 | 70.37 | 57% | 128.3 | 53% |
| 1280 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<1280, 2560>, Fp8A16TmaT64R128S3B2K64>` (T 513-1536) | 896 | 82.10 | 66% | 142.1 | 59% |
| 1280 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<1280, 2560>, Fp8A16TmaT64R128S3B2K64>` (T 513-1536) | 1024 | 72.94 | 59% | 121.1 | 50% |
| 1280 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<1280, 2560>, Fp8A16TmaT64R128S3B2K64>` (T 513-1536) | 1152 | 80.44 | 65% | 129.2 | 54% |
| 1280 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<1280, 2560>, Fp8A16TmaT64R128S3B2K64>` (T 513-1536) | 1280 | 74.58 | 60% | 116.6 | 48% |
| 1280 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<1280, 2560>, Fp8A16TmaT64R128S3B2K64>` (T 513-1536) | 1408 | 80.43 | 65% | 122.8 | 51% |
| 1280 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<1280, 2560>, Fp8A16TmaT64R128S3B2K64>` (T 513-1536) | 1536 | 84.81 | 69% | 127.0 | 53% |
| 1280 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<1280, 2560>, Fp8A16TmaT128R128S2B2K64>` (T 1537-3584) | 1537 | 75.76 | 61% | 113.4 | 47% |
| 1280 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<1280, 2560>, Fp8A16TmaT128R128S2B2K64>` (T 1537-3584) | 1664 | 82.02 | 66% | 120.8 | 50% |
| 1280 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<1280, 2560>, Fp8A16TmaT128R128S2B2K64>` (T 1537-3584) | 1792 | 86.99 | 70% | 126.2 | 52% |
| 1280 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<1280, 2560>, Fp8A16TmaT128R128S2B2K64>` (T 1537-3584) | 1920 | 75.91 | 61% | 108.7 | 45% |
| 1280 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<1280, 2560>, Fp8A16TmaT128R128S2B2K64>` (T 1537-3584) | 2048 | 80.00 | 65% | 113.3 | 47% |
| 1280 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<1280, 2560>, Fp8A16TmaT64R256RF>` (T>=3585) | — | — | — | — | — |
| 2560 | 640 | a16 | `fp8_linear_a16_ring<Fp8Geometry<2560, 640>, Fp8A16RingT32R64S3>` (T 1-64) | 1 | 0.25 | 0.2% | 125.8 | 52% |
| 2560 | 640 | a16 | `fp8_linear_a16_ring<Fp8Geometry<2560, 640>, Fp8A16RingT32R64S3>` (T 1-64) | 2 | 0.50 | 0.4% | 126.2 | 52% |
| 2560 | 640 | a16 | `fp8_linear_a16_ring<Fp8Geometry<2560, 640>, Fp8A16RingT32R64S3>` (T 1-64) | 3 | 0.81 | 0.7% | 137.1 | 57% |
| 2560 | 640 | a16 | `fp8_linear_a16_ring<Fp8Geometry<2560, 640>, Fp8A16RingT32R64S3>` (T 1-64) | 4 | 1.01 | 0.8% | 128.2 | 53% |
| 2560 | 640 | a16 | `fp8_linear_a16_ring<Fp8Geometry<2560, 640>, Fp8A16RingT32R64S3>` (T 1-64) | 5 | 1.25 | 1.0% | 127.7 | 53% |
| 2560 | 640 | a16 | `fp8_linear_a16_ring<Fp8Geometry<2560, 640>, Fp8A16RingT32R64S3>` (T 1-64) | 6 | 1.50 | 1.2% | 128.2 | 53% |
| 2560 | 640 | a16 | `fp8_linear_a16_ring<Fp8Geometry<2560, 640>, Fp8A16RingT32R64S3>` (T 1-64) | 7 | 1.75 | 1.4% | 129.0 | 54% |
| 2560 | 640 | a16 | `fp8_linear_a16_ring<Fp8Geometry<2560, 640>, Fp8A16RingT32R64S3>` (T 1-64) | 8 | 2.01 | 1.6% | 129.8 | 54% |
| 2560 | 640 | a16 | `fp8_linear_a16_ring<Fp8Geometry<2560, 640>, Fp8A16RingT32R64S3>` (T 1-64) | 12 | 3.02 | 2.4% | 132.1 | 55% |
| 2560 | 640 | a16 | `fp8_linear_a16_ring<Fp8Geometry<2560, 640>, Fp8A16RingT32R64S3>` (T 1-64) | 16 | 4.00 | 3.2% | 133.1 | 55% |
| 2560 | 640 | a16 | `fp8_linear_a16_ring<Fp8Geometry<2560, 640>, Fp8A16RingT32R64S3>` (T 1-64) | 24 | 6.40 | 5.2% | 146.2 | 61% |
| 2560 | 640 | a16 | `fp8_linear_a16_ring<Fp8Geometry<2560, 640>, Fp8A16RingT32R64S3>` (T 1-64) | 32 | 8.65 | 7.0% | 152.4 | 63% |
| 2560 | 640 | a16 | `fp8_linear_a16_ring<Fp8Geometry<2560, 640>, Fp8A16RingT32R64S3>` (T 1-64) | 48 | 12.97 | 10% | 160.8 | 67% |
| 2560 | 640 | a16 | `fp8_linear_a16_ring<Fp8Geometry<2560, 640>, Fp8A16RingT32R64S3>` (T 1-64) | 64 | 17.29 | 14% | 169.3 | 70% |
| 2560 | 640 | a16 | `fp8_linear_a16_tma<Fp8Geometry<2560, 640>, Fp8A16TmaT64R128>` (T 65-128) | 65 | 15.06 | 12% | 145.6 | 61% |
| 2560 | 640 | a16 | `fp8_linear_a16_tma<Fp8Geometry<2560, 640>, Fp8A16TmaT64R128>` (T 65-128) | 128 | 29.65 | 24% | 174.1 | 72% |
| 2560 | 640 | a16 | `fp8_linear_a16_tma<Fp8Geometry<2560, 640>, Fp8A16TmaMmaSchedule<96, 128, 2, 4, 3, 1, 64>>` (T 129-192) | 129 | 29.89 | 24% | 174.6 | 73% |
| 2560 | 640 | a16 | `fp8_linear_a16_tma<Fp8Geometry<2560, 640>, Fp8A16TmaMmaSchedule<96, 128, 2, 4, 3, 1, 64>>` (T 129-192) | 192 | 38.86 | 31% | 177.4 | 74% |
| 2560 | 640 | a16 | `fp8_linear_a16_tma<Fp8Geometry<2560, 640>, Fp8A16TmaT64R128S3B2K64>` (T 193-640) | 193 | 38.98 | 32% | 177.4 | 74% |
| 2560 | 640 | a16 | `fp8_linear_a16_tma<Fp8Geometry<2560, 640>, Fp8A16TmaT64R128S3B2K64>` (T 193-640) | 256 | 51.70 | 42% | 202.3 | 84% |
| 2560 | 640 | a16 | `fp8_linear_a16_tma<Fp8Geometry<2560, 640>, Fp8A16TmaT64R128S3B2K64>` (T 193-640) | 384 | 56.33 | 46% | 183.6 | 76% |
| 2560 | 640 | a16 | `fp8_linear_a16_tma<Fp8Geometry<2560, 640>, Fp8A16TmaT64R128S3B2K64>` (T 193-640) | 512 | 63.47 | 51% | 186.2 | 77% |
| 2560 | 640 | a16 | `fp8_linear_a16_tma<Fp8Geometry<2560, 640>, Fp8A16TmaT64R128S3B2K64>` (T 193-640) | 640 | 64.38 | 52% | 176.2 | 73% |
| 2560 | 640 | a16 | `fp8_linear_a16_tma<Fp8Geometry<2560, 640>, Fp8A16TmaT128R128S2B2K64>` (T>=641) | 641 | 54.20 | 44% | 148.3 | 62% |
| 2560 | 640 | a16 | `fp8_linear_a16_tma<Fp8Geometry<2560, 640>, Fp8A16TmaT128R128S2B2K64>` (T>=641) | 768 | 61.73 | 50% | 160.9 | 67% |
| 2560 | 640 | a16 | `fp8_linear_a16_tma<Fp8Geometry<2560, 640>, Fp8A16TmaT128R128S2B2K64>` (T>=641) | 896 | 72.02 | 58% | 181.0 | 75% |
| 2560 | 640 | a16 | `fp8_linear_a16_tma<Fp8Geometry<2560, 640>, Fp8A16TmaT128R128S2B2K64>` (T>=641) | 1024 | 65.78 | 53% | 160.7 | 67% |
| 2560 | 640 | a16 | `fp8_linear_a16_tma<Fp8Geometry<2560, 640>, Fp8A16TmaT128R128S2B2K64>` (T>=641) | 1152 | 73.87 | 60% | 176.4 | 73% |
| 2560 | 640 | a16 | `fp8_linear_a16_tma<Fp8Geometry<2560, 640>, Fp8A16TmaT128R128S2B2K64>` (T>=641) | 1280 | 68.48 | 55% | 160.6 | 67% |
| 2560 | 640 | a16 | `fp8_linear_a16_tma<Fp8Geometry<2560, 640>, Fp8A16TmaT128R128S2B2K64>` (T>=641) | 1408 | 72.89 | 59% | 168.3 | 70% |
| 2560 | 640 | a16 | `fp8_linear_a16_tma<Fp8Geometry<2560, 640>, Fp8A16TmaT128R128S2B2K64>` (T>=641) | 1536 | 77.03 | 62% | 175.6 | 73% |
| 2560 | 640 | a16 | `fp8_linear_a16_tma<Fp8Geometry<2560, 640>, Fp8A16TmaT128R128S2B2K64>` (T>=641) | 1664 | 74.12 | 60% | 167.1 | 69% |
| 2560 | 640 | a16 | `fp8_linear_a16_tma<Fp8Geometry<2560, 640>, Fp8A16TmaT128R128S2B2K64>` (T>=641) | 1792 | 77.46 | 63% | 173.0 | 72% |
| 2560 | 640 | a16 | `fp8_linear_a16_tma<Fp8Geometry<2560, 640>, Fp8A16TmaT128R128S2B2K64>` (T>=641) | 1920 | 75.10 | 61% | 166.3 | 69% |
| 2560 | 640 | a16 | `fp8_linear_a16_tma<Fp8Geometry<2560, 640>, Fp8A16TmaT128R128S2B2K64>` (T>=641) | 2048 | 78.19 | 63% | 171.9 | 71% |
| 2560 | 6144 | a16 | `fp8_linear_a16_gemv<Fp8Geometry<2560, 6144>, Fp8A16GemvSchedule<8, 2, 16, 1, Fp8CodeCache::Default, 2, 2>>` (T=1) | 1 | 0.45 | 0.4% | 223.5 | 93% |
| 2560 | 6144 | a16 | `fp8_linear_a16_simt<Fp8Geometry<2560, 6144>, 8, Fp8A16SimtSchedule<4, 2, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked, Fp8CodeCache::Streaming, 1, Fp8SimtBlockOrder::RowsContiguous, 1>>` (T 2-8) | 2 | 0.88 | 0.7% | 220.6 | 92% |
| 2560 | 6144 | a16 | `fp8_linear_a16_simt<Fp8Geometry<2560, 6144>, 8, Fp8A16SimtSchedule<4, 2, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked, Fp8CodeCache::Streaming, 1, Fp8SimtBlockOrder::RowsContiguous, 1>>` (T 2-8) | 3 | 1.32 | 1.1% | 220.7 | 92% |
| 2560 | 6144 | a16 | `fp8_linear_a16_simt<Fp8Geometry<2560, 6144>, 8, Fp8A16SimtSchedule<4, 2, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked, Fp8CodeCache::Streaming, 1, Fp8SimtBlockOrder::RowsContiguous, 1>>` (T 2-8) | 4 | 1.76 | 1.4% | 221.1 | 92% |
| 2560 | 6144 | a16 | `fp8_linear_a16_simt<Fp8Geometry<2560, 6144>, 8, Fp8A16SimtSchedule<4, 2, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked, Fp8CodeCache::Streaming, 1, Fp8SimtBlockOrder::RowsContiguous, 1>>` (T 2-8) | 5 | 2.17 | 1.8% | 218.2 | 91% |
| 2560 | 6144 | a16 | `fp8_linear_a16_simt<Fp8Geometry<2560, 6144>, 8, Fp8A16SimtSchedule<4, 2, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked, Fp8CodeCache::Streaming, 1, Fp8SimtBlockOrder::RowsContiguous, 1>>` (T 2-8) | 6 | 2.60 | 2.1% | 218.4 | 91% |
| 2560 | 6144 | a16 | `fp8_linear_a16_simt<Fp8Geometry<2560, 6144>, 8, Fp8A16SimtSchedule<4, 2, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked, Fp8CodeCache::Streaming, 1, Fp8SimtBlockOrder::RowsContiguous, 1>>` (T 2-8) | 7 | 3.04 | 2.5% | 218.7 | 91% |
| 2560 | 6144 | a16 | `fp8_linear_a16_simt<Fp8Geometry<2560, 6144>, 8, Fp8A16SimtSchedule<4, 2, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked, Fp8CodeCache::Streaming, 1, Fp8SimtBlockOrder::RowsContiguous, 1>>` (T 2-8) | 8 | 3.42 | 2.8% | 215.9 | 90% |
| 2560 | 6144 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<2560, 6144>, Fp8SlicedInstance<16, 8, 2>>` (T 9-16) | 9 | 3.96 | 3.2% | 222.3 | 92% |
| 2560 | 6144 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<2560, 6144>, Fp8SlicedInstance<16, 8, 2>>` (T 9-16) | 12 | 5.27 | 4.3% | 222.4 | 92% |
| 2560 | 6144 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<2560, 6144>, Fp8SlicedInstance<16, 8, 2>>` (T 9-16) | 16 | 6.94 | 5.6% | 220.7 | 92% |
| 2560 | 6144 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<2560, 6144>, Fp8SlicedInstance<32, 8, 2>>` (T 17-32) | 17 | 6.98 | 5.6% | 209.2 | 87% |
| 2560 | 6144 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<2560, 6144>, Fp8SlicedInstance<32, 8, 2>>` (T 17-32) | 24 | 9.62 | 7.8% | 205.8 | 86% |
| 2560 | 6144 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<2560, 6144>, Fp8SlicedInstance<32, 8, 2>>` (T 17-32) | 32 | 12.31 | 10.0% | 199.3 | 83% |
| 2560 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<2560, 6144>, Fp8A16TmaT32R64S3>` (T 33-64) | 33 | 13.73 | 11% | 215.7 | 90% |
| 2560 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<2560, 6144>, Fp8A16TmaT32R64S3>` (T 33-64) | 48 | 19.44 | 16% | 213.3 | 89% |
| 2560 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<2560, 6144>, Fp8A16TmaT32R64S3>` (T 33-64) | 64 | 25.60 | 21% | 214.2 | 89% |
| 2560 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<2560, 6144>, Fp8A16TmaT64R128>` (T 65-128) | 65 | 25.32 | 20% | 208.8 | 87% |
| 2560 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<2560, 6144>, Fp8A16TmaT64R128>` (T 65-128) | 128 | 48.04 | 39% | 214.3 | 89% |
| 2560 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<2560, 6144>, Fp8A16TmaT64R256G8>` (T 129-192) | 129 | 41.36 | 33% | 183.3 | 76% |
| 2560 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<2560, 6144>, Fp8A16TmaT64R256G8>` (T 129-192) | 192 | 60.30 | 49% | 190.5 | 79% |
| 2560 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<2560, 6144>, Fp8A16TmaMmaSchedule<128, 128, 4, 4, 2, 1>>` (T 193-256) | 193 | 59.40 | 48% | 186.8 | 78% |
| 2560 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<2560, 6144>, Fp8A16TmaMmaSchedule<128, 128, 4, 4, 2, 1>>` (T 193-256) | 256 | 77.22 | 62% | 193.6 | 80% |
| 2560 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<2560, 6144>, Fp8A16TmaT32R128S4B2K64>` (T 257-288) | 257 | 63.77 | 52% | 159.4 | 66% |
| 2560 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<2560, 6144>, Fp8A16TmaT32R128S4B2K64>` (T 257-288) | 288 | 71.46 | 58% | 163.6 | 68% |
| 2560 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<2560, 6144>, Fp8A16TmaT64R128S3B2K64>` (T 289-1024) | 289 | 56.97 | 46% | 130.1 | 54% |
| 2560 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<2560, 6144>, Fp8A16TmaT64R128S3B2K64>` (T 289-1024) | 384 | 75.69 | 61% | 140.5 | 58% |
| 2560 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<2560, 6144>, Fp8A16TmaT64R128S3B2K64>` (T 289-1024) | 512 | 77.94 | 63% | 119.3 | 50% |
| 2560 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<2560, 6144>, Fp8A16TmaT64R128S3B2K64>` (T 289-1024) | 640 | 79.33 | 64% | 105.9 | 44% |
| 2560 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<2560, 6144>, Fp8A16TmaT64R128S3B2K64>` (T 289-1024) | 768 | 92.95 | 75% | 112.0 | 47% |
| 2560 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<2560, 6144>, Fp8A16TmaT64R128S3B2K64>` (T 289-1024) | 896 | 91.20 | 74% | 101.4 | 42% |
| 2560 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<2560, 6144>, Fp8A16TmaT64R128S3B2K64>` (T 289-1024) | 1024 | 90.44 | 73% | 94.2 | 39% |
| 2560 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<2560, 6144>, Fp8A16TmaT128R128S2B2K64>` (T 1025-2304) | 1025 | 84.23 | 68% | 87.7 | 36% |
| 2560 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<2560, 6144>, Fp8A16TmaT128R128S2B2K64>` (T 1025-2304) | 1152 | 93.67 | 76% | 92.5 | 38% |
| 2560 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<2560, 6144>, Fp8A16TmaT128R128S2B2K64>` (T 1025-2304) | 1280 | 84.05 | 68% | 79.4 | 33% |
| 2560 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<2560, 6144>, Fp8A16TmaT128R128S2B2K64>` (T 1025-2304) | 1408 | 91.67 | 74% | 83.3 | 35% |
| 2560 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<2560, 6144>, Fp8A16TmaT128R128S2B2K64>` (T 1025-2304) | 1536 | 99.58 | 80% | 87.5 | 36% |
| 2560 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<2560, 6144>, Fp8A16TmaT128R128S2B2K64>` (T 1025-2304) | 1664 | 90.99 | 74% | 77.7 | 32% |
| 2560 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<2560, 6144>, Fp8A16TmaT128R128S2B2K64>` (T 1025-2304) | 1792 | 95.93 | 78% | 79.9 | 33% |
| 2560 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<2560, 6144>, Fp8A16TmaT128R128S2B2K64>` (T 1025-2304) | 1920 | 88.32 | 71% | 71.9 | 30% |
| 2560 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<2560, 6144>, Fp8A16TmaT128R128S2B2K64>` (T 1025-2304) | 2048 | 92.55 | 75% | 73.8 | 31% |
| 2560 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<2560, 6144>, Fp8A16TmaT64R256RF>` (T>=2305) | — | — | — | — | — |
| 3072 | 1024 | a16 | `launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Fp8SlicedInstance<8, 8, 2>, 1024>>` (T 1-8) | 1 | 0.36 | 0.3% | 183.6 | 76% |
| 3072 | 1024 | a16 | `launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Fp8SlicedInstance<8, 8, 2>, 1024>>` (T 1-8) | 2 | 0.69 | 0.6% | 174.0 | 72% |
| 3072 | 1024 | a16 | `launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Fp8SlicedInstance<8, 8, 2>, 1024>>` (T 1-8) | 3 | 1.03 | 0.8% | 174.2 | 72% |
| 3072 | 1024 | a16 | `launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Fp8SlicedInstance<8, 8, 2>, 1024>>` (T 1-8) | 4 | 1.38 | 1.1% | 174.9 | 73% |
| 3072 | 1024 | a16 | `launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Fp8SlicedInstance<8, 8, 2>, 1024>>` (T 1-8) | 5 | 1.81 | 1.5% | 183.8 | 76% |
| 3072 | 1024 | a16 | `launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Fp8SlicedInstance<8, 8, 2>, 1024>>` (T 1-8) | 6 | 2.09 | 1.7% | 177.7 | 74% |
| 3072 | 1024 | a16 | `launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Fp8SlicedInstance<8, 8, 2>, 1024>>` (T 1-8) | 7 | 2.41 | 1.9% | 176.3 | 73% |
| 3072 | 1024 | a16 | `launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Fp8SlicedInstance<8, 8, 2>, 1024>>` (T 1-8) | 8 | 2.76 | 2.2% | 176.7 | 73% |
| 3072 | 1024 | a16 | `launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Fp8SlicedInstance<16, 8, 2>, 1024>>` (T 9-16) | 9 | 2.59 | 2.1% | 148.1 | 62% |
| 3072 | 1024 | a16 | `launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Fp8SlicedInstance<16, 8, 2>, 1024>>` (T 9-16) | 12 | 4.11 | 3.3% | 177.3 | 74% |
| 3072 | 1024 | a16 | `launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Fp8SlicedInstance<16, 8, 2>, 1024>>` (T 9-16) | 16 | 5.23 | 4.2% | 170.7 | 71% |
| 3072 | 1024 | a16 | `launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Fp8SlicedInstance<32, 4, 2>, 1024>>` (T 17-32) | 17 | 5.80 | 4.7% | 178.9 | 74% |
| 3072 | 1024 | a16 | `launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Fp8SlicedInstance<32, 4, 2>, 1024>>` (T 17-32) | 24 | 7.44 | 6.0% | 165.3 | 69% |
| 3072 | 1024 | a16 | `launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Fp8SlicedInstance<32, 4, 2>, 1024>>` (T 17-32) | 32 | 9.91 | 8.0% | 168.3 | 70% |
| 3072 | 1024 | a16 | `launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Fp8SlicedInstance<64, 4, 1>, 1024>>` (T 33-64) | 33 | 9.28 | 7.5% | 153.3 | 64% |
| 3072 | 1024 | a16 | `launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Fp8SlicedInstance<64, 4, 1>, 1024>>` (T 33-64) | 48 | 13.50 | 11% | 158.8 | 66% |
| 3072 | 1024 | a16 | `launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Fp8SlicedInstance<64, 4, 1>, 1024>>` (T 33-64) | 64 | 16.49 | 13% | 150.8 | 63% |
| 3072 | 1024 | a16 | `launch_fp8_a16_ring_mma<Fp8ScheduleInstance<Fp8A16RingT32R64S3, 1024>>` (T 65-320) | 65 | 18.28 | 15% | 165.0 | 69% |
| 3072 | 1024 | a16 | `launch_fp8_a16_ring_mma<Fp8ScheduleInstance<Fp8A16RingT32R64S3, 1024>>` (T 65-320) | 128 | 36.00 | 29% | 188.1 | 78% |
| 3072 | 1024 | a16 | `launch_fp8_a16_ring_mma<Fp8ScheduleInstance<Fp8A16RingT32R64S3, 1024>>` (T 65-320) | 256 | 49.39 | 40% | 161.2 | 67% |
| 3072 | 1024 | a16 | `launch_fp8_a16_ring_mma<Fp8ScheduleInstance<Fp8A16RingT32R64S3, 1024>>` (T 65-320) | 320 | 49.34 | 40% | 141.7 | 59% |
| 3072 | 1024 | a16 | `launch_fp8_a16_ring_mma<Fp8ScheduleInstance<Fp8A16RingT64R128, 1024>>` (T>=321) | 321 | 49.50 | 40% | 141.9 | 59% |
| 3072 | 1024 | a16 | `launch_fp8_a16_ring_mma<Fp8ScheduleInstance<Fp8A16RingT64R128, 1024>>` (T>=321) | 384 | 59.21 | 48% | 154.5 | 64% |
| 3072 | 1024 | a16 | `launch_fp8_a16_ring_mma<Fp8ScheduleInstance<Fp8A16RingT64R128, 1024>>` (T>=321) | 512 | 60.68 | 49% | 138.5 | 58% |
| 3072 | 1024 | a16 | `launch_fp8_a16_ring_mma<Fp8ScheduleInstance<Fp8A16RingT64R128, 1024>>` (T>=321) | 640 | 61.53 | 50% | 128.4 | 53% |
| 3072 | 1024 | a16 | `launch_fp8_a16_ring_mma<Fp8ScheduleInstance<Fp8A16RingT64R128, 1024>>` (T>=321) | 768 | 62.11 | 50% | 121.5 | 50% |
| 3072 | 1024 | a16 | `launch_fp8_a16_ring_mma<Fp8ScheduleInstance<Fp8A16RingT64R128, 1024>>` (T>=321) | 896 | 61.29 | 50% | 114.1 | 47% |
| 3072 | 1024 | a16 | `launch_fp8_a16_ring_mma<Fp8ScheduleInstance<Fp8A16RingT64R128, 1024>>` (T>=321) | 1024 | 61.81 | 50% | 110.8 | 46% |
| 3072 | 1024 | a16 | `launch_fp8_a16_ring_mma<Fp8ScheduleInstance<Fp8A16RingT64R128, 1024>>` (T>=321) | 1152 | 62.19 | 50% | 108.1 | 45% |
| 3072 | 1024 | a16 | `launch_fp8_a16_ring_mma<Fp8ScheduleInstance<Fp8A16RingT64R128, 1024>>` (T>=321) | 1280 | 61.56 | 50% | 104.3 | 43% |
| 3072 | 1024 | a16 | `launch_fp8_a16_ring_mma<Fp8ScheduleInstance<Fp8A16RingT64R128, 1024>>` (T>=321) | 1408 | 61.87 | 50% | 102.6 | 43% |
| 3072 | 1024 | a16 | `launch_fp8_a16_ring_mma<Fp8ScheduleInstance<Fp8A16RingT64R128, 1024>>` (T>=321) | 1536 | 61.74 | 50% | 100.6 | 42% |
| 3072 | 1024 | a16 | `launch_fp8_a16_ring_mma<Fp8ScheduleInstance<Fp8A16RingT64R128, 1024>>` (T>=321) | 1664 | 61.56 | 50% | 98.7 | 41% |
| 3072 | 1024 | a16 | `launch_fp8_a16_ring_mma<Fp8ScheduleInstance<Fp8A16RingT64R128, 1024>>` (T>=321) | 1792 | 61.56 | 50% | 97.4 | 40% |
| 3072 | 1024 | a16 | `launch_fp8_a16_ring_mma<Fp8ScheduleInstance<Fp8A16RingT64R128, 1024>>` (T>=321) | 1920 | 61.81 | 50% | 96.6 | 40% |
| 3072 | 1024 | a16 | `launch_fp8_a16_ring_mma<Fp8ScheduleInstance<Fp8A16RingT64R128, 1024>>` (T>=321) | 2048 | 61.73 | 50% | 95.5 | 40% |
| 5120 | 6144 | a16 | `fp8_linear_a16_gemv<Fp8Geometry<5120, 6144>, Fp8A16GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>>` (T=1) | 1 | 0.46 | 0.4% | 228.1 | 95% |
| 5120 | 6144 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<5120, 6144>, Fp8SlicedInstance<16, 8, 2>>` (T 2-16) | 2 | 0.91 | 0.7% | 228.3 | 95% |
| 5120 | 6144 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<5120, 6144>, Fp8SlicedInstance<16, 8, 2>>` (T 2-16) | 3 | 1.38 | 1.1% | 230.1 | 96% |
| 5120 | 6144 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<5120, 6144>, Fp8SlicedInstance<16, 8, 2>>` (T 2-16) | 4 | 1.84 | 1.5% | 230.3 | 96% |
| 5120 | 6144 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<5120, 6144>, Fp8SlicedInstance<16, 8, 2>>` (T 2-16) | 5 | 2.29 | 1.9% | 230.3 | 96% |
| 5120 | 6144 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<5120, 6144>, Fp8SlicedInstance<16, 8, 2>>` (T 2-16) | 6 | 2.73 | 2.2% | 228.9 | 95% |
| 5120 | 6144 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<5120, 6144>, Fp8SlicedInstance<16, 8, 2>>` (T 2-16) | 7 | 3.17 | 2.6% | 227.4 | 95% |
| 5120 | 6144 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<5120, 6144>, Fp8SlicedInstance<16, 8, 2>>` (T 2-16) | 8 | 3.62 | 2.9% | 227.6 | 95% |
| 5120 | 6144 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<5120, 6144>, Fp8SlicedInstance<16, 8, 2>>` (T 2-16) | 12 | 5.47 | 4.4% | 229.9 | 96% |
| 5120 | 6144 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<5120, 6144>, Fp8SlicedInstance<16, 8, 2>>` (T 2-16) | 16 | 7.29 | 5.9% | 230.5 | 96% |
| 5120 | 6144 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<5120, 6144>, Fp8SlicedInstance<32, 8, 2>>` (T 17-32) | 17 | 7.52 | 6.1% | 224.1 | 93% |
| 5120 | 6144 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<5120, 6144>, Fp8SlicedInstance<32, 8, 2>>` (T 17-32) | 24 | 10.55 | 8.5% | 223.6 | 93% |
| 5120 | 6144 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<5120, 6144>, Fp8SlicedInstance<32, 8, 2>>` (T 17-32) | 32 | 13.86 | 11% | 221.7 | 92% |
| 5120 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 6144>, Fp8A16TmaT64R128>` (T 33-64) | 33 | 14.10 | 11% | 218.7 | 91% |
| 5120 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 6144>, Fp8A16TmaT64R128>` (T 33-64) | 48 | 20.35 | 16% | 219.4 | 91% |
| 5120 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 6144>, Fp8A16TmaT64R128>` (T 33-64) | 64 | 27.13 | 22% | 221.7 | 92% |
| 5120 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 6144>, Fp8A16TmaT64R256G8>` (T 65-128) | 65 | 27.01 | 22% | 217.5 | 90% |
| 5120 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 6144>, Fp8A16TmaT64R256G8>` (T 65-128) | 128 | 51.73 | 42% | 220.7 | 92% |
| 5120 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 6144>, Fp8A16TmaT32R128>` (T 129-192) | 129 | 44.57 | 36% | 188.8 | 78% |
| 5120 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 6144>, Fp8A16TmaT32R128>` (T 129-192) | 192 | 67.08 | 54% | 198.8 | 83% |
| 5120 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 6144>, Fp8A16TmaMmaSchedule<128, 128, 4, 4, 2, 1>>` (T 193-256) | 193 | 61.23 | 49% | 180.6 | 75% |
| 5120 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 6144>, Fp8A16TmaMmaSchedule<128, 128, 4, 4, 2, 1>>` (T 193-256) | 256 | 80.31 | 65% | 185.7 | 77% |
| 5120 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 6144>, Fp8A16TmaT96R128S3K64>` (T 257-288) | 257 | 68.11 | 55% | 156.9 | 65% |
| 5120 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 6144>, Fp8A16TmaT96R128S3K64>` (T 257-288) | 288 | 75.67 | 61% | 158.5 | 66% |
| 5120 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 6144>, Fp8A16TmaT64R128S3B2K64>` (T 289-512) | 289 | 71.60 | 58% | 149.5 | 62% |
| 5120 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 6144>, Fp8A16TmaT64R128S3B2K64>` (T 289-512) | 384 | 93.62 | 76% | 155.5 | 65% |
| 5120 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 6144>, Fp8A16TmaT64R128S3B2K64>` (T 289-512) | 512 | 91.49 | 74% | 122.1 | 51% |
| 5120 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 6144>, Fp8A16TmaT128R128S2B2K64>` (T 513-2304) | 513 | 68.54 | 55% | 91.4 | 38% |
| 5120 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 6144>, Fp8A16TmaT128R128S2B2K64>` (T 513-2304) | 640 | 84.44 | 68% | 96.2 | 40% |
| 5120 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 6144>, Fp8A16TmaT128R128S2B2K64>` (T 513-2304) | 768 | 100.43 | 81% | 101.4 | 42% |
| 5120 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 6144>, Fp8A16TmaT128R128S2B2K64>` (T 513-2304) | 896 | 99.40 | 80% | 91.1 | 38% |
| 5120 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 6144>, Fp8A16TmaT128R128S2B2K64>` (T 513-2304) | 1024 | 97.72 | 79% | 82.7 | 34% |
| 5120 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 6144>, Fp8A16TmaT128R128S2B2K64>` (T 513-2304) | 1152 | 96.98 | 78% | 76.8 | 32% |
| 5120 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 6144>, Fp8A16TmaT128R128S2B2K64>` (T 513-2304) | 1280 | 95.93 | 78% | 71.8 | 30% |
| 5120 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 6144>, Fp8A16TmaT128R128S2B2K64>` (T 513-2304) | 1408 | 94.67 | 77% | 67.5 | 28% |
| 5120 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 6144>, Fp8A16TmaT128R128S2B2K64>` (T 513-2304) | 1536 | 102.36 | 83% | 70.0 | 29% |
| 5120 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 6144>, Fp8A16TmaT128R128S2B2K64>` (T 513-2304) | 1664 | 99.28 | 80% | 65.4 | 27% |
| 5120 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 6144>, Fp8A16TmaT128R128S2B2K64>` (T 513-2304) | 1792 | 96.73 | 78% | 61.6 | 26% |
| 5120 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 6144>, Fp8A16TmaT128R128S2B2K64>` (T 513-2304) | 1920 | 94.38 | 76% | 58.4 | 24% |
| 5120 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 6144>, Fp8A16TmaT128R128S2B2K64>` (T 513-2304) | 2048 | 92.94 | 75% | 56.0 | 23% |
| 5120 | 6144 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 6144>, Fp8A16TmaT128R128G8>` (T>=2305) | — | — | — | — | — |
| 5120 | 6144 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 6144>, Fp8A8TmaMmaSchedule<32, 64, 128, 1, 2, 3, 2>>` (T 1-64) | 1 | 0.46 | 0.2% | 227.8 | 95% |
| 5120 | 6144 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 6144>, Fp8A8TmaMmaSchedule<32, 64, 128, 1, 2, 3, 2>>` (T 1-64) | 2 | 0.92 | 0.4% | 229.9 | 96% |
| 5120 | 6144 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 6144>, Fp8A8TmaMmaSchedule<32, 64, 128, 1, 2, 3, 2>>` (T 1-64) | 3 | 1.38 | 0.6% | 230.1 | 96% |
| 5120 | 6144 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 6144>, Fp8A8TmaMmaSchedule<32, 64, 128, 1, 2, 3, 2>>` (T 1-64) | 4 | 1.84 | 0.8% | 230.3 | 96% |
| 5120 | 6144 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 6144>, Fp8A8TmaMmaSchedule<32, 64, 128, 1, 2, 3, 2>>` (T 1-64) | 5 | 2.29 | 0.9% | 230.4 | 96% |
| 5120 | 6144 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 6144>, Fp8A8TmaMmaSchedule<32, 64, 128, 1, 2, 3, 2>>` (T 1-64) | 6 | 2.73 | 1.1% | 228.9 | 95% |
| 5120 | 6144 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 6144>, Fp8A8TmaMmaSchedule<32, 64, 128, 1, 2, 3, 2>>` (T 1-64) | 7 | 3.19 | 1.3% | 229.1 | 95% |
| 5120 | 6144 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 6144>, Fp8A8TmaMmaSchedule<32, 64, 128, 1, 2, 3, 2>>` (T 1-64) | 8 | 3.62 | 1.5% | 227.6 | 95% |
| 5120 | 6144 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 6144>, Fp8A8TmaMmaSchedule<32, 64, 128, 1, 2, 3, 2>>` (T 1-64) | 12 | 5.47 | 2.2% | 229.9 | 96% |
| 5120 | 6144 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 6144>, Fp8A8TmaMmaSchedule<32, 64, 128, 1, 2, 3, 2>>` (T 1-64) | 16 | 7.29 | 3.0% | 230.6 | 96% |
| 5120 | 6144 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 6144>, Fp8A8TmaMmaSchedule<32, 64, 128, 1, 2, 3, 2>>` (T 1-64) | 24 | 10.11 | 4.1% | 214.3 | 89% |
| 5120 | 6144 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 6144>, Fp8A8TmaMmaSchedule<32, 64, 128, 1, 2, 3, 2>>` (T 1-64) | 32 | 13.39 | 5.5% | 214.1 | 89% |
| 5120 | 6144 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 6144>, Fp8A8TmaMmaSchedule<32, 64, 128, 1, 2, 3, 2>>` (T 1-64) | 48 | 20.23 | 8.3% | 218.0 | 91% |
| 5120 | 6144 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 6144>, Fp8A8TmaMmaSchedule<32, 64, 128, 1, 2, 3, 2>>` (T 1-64) | 64 | 26.78 | 11% | 218.9 | 91% |
| 5120 | 6144 | a8 | `launch_fp8_a8<Fp8Geometry<5120, 6144>, Fp8A8T64R64K128>` (T 65-128) | 65 | 27.01 | 11% | 217.5 | 90% |
| 5120 | 6144 | a8 | `launch_fp8_a8<Fp8Geometry<5120, 6144>, Fp8A8T64R64K128>` (T 65-128) | 128 | 51.79 | 21% | 220.9 | 92% |
| 5120 | 6144 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 6144>, Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 3, 1>>` (T 129-192) | 129 | 50.86 | 21% | 215.4 | 90% |
| 5120 | 6144 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 6144>, Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 3, 1>>` (T 129-192) | 192 | 73.80 | 30% | 218.7 | 91% |
| 5120 | 6144 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 6144>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 3, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 193-1024) | 193 | 69.80 | 29% | 205.9 | 86% |
| 5120 | 6144 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 6144>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 3, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 193-1024) | 256 | 89.46 | 37% | 206.8 | 86% |
| 5120 | 6144 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 6144>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 3, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 193-1024) | 384 | 126.97 | 52% | 210.8 | 88% |
| 5120 | 6144 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 6144>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 3, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 193-1024) | 512 | 147.79 | 61% | 197.3 | 82% |
| 5120 | 6144 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 6144>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 3, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 193-1024) | 640 | 156.15 | 64% | 177.9 | 74% |
| 5120 | 6144 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 6144>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 3, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 193-1024) | 768 | 169.85 | 70% | 171.4 | 71% |
| 5120 | 6144 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 6144>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 3, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 193-1024) | 896 | 156.46 | 64% | 143.4 | 60% |
| 5120 | 6144 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 6144>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 3, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 193-1024) | 1024 | 155.00 | 64% | 131.2 | 55% |
| 5120 | 6144 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 6144>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 1025-2048) | 1025 | 140.62 | 58% | 119.0 | 49% |
| 5120 | 6144 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 6144>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 1025-2048) | 1152 | 152.92 | 63% | 121.2 | 50% |
| 5120 | 6144 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 6144>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 1025-2048) | 1280 | 160.88 | 66% | 120.5 | 50% |
| 5120 | 6144 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 6144>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 1025-2048) | 1408 | 152.88 | 63% | 109.1 | 45% |
| 5120 | 6144 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 6144>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 1025-2048) | 1536 | 169.43 | 69% | 115.8 | 48% |
| 5120 | 6144 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 6144>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 1025-2048) | 1664 | 160.54 | 66% | 105.7 | 44% |
| 5120 | 6144 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 6144>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 1025-2048) | 1792 | 157.53 | 65% | 100.4 | 42% |
| 5120 | 6144 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 6144>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 1025-2048) | 1920 | 162.30 | 67% | 100.4 | 42% |
| 5120 | 6144 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 6144>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 1025-2048) | 2048 | 156.19 | 64% | 94.1 | 39% |
| 5120 | 6144 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 6144>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 3, 1, Fp8MmaRaster::Grouped, 8>, kGb10SplitWaveCtas, 4, 8>>` (T>=2049) | — | — | — | — | — |
| 5120 | 17408 | a16 | `fp8_linear_a16_gemv<Fp8Geometry<5120, 17408>, Fp8A16GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>>` (T=1) | 1 | 0.46 | 0.4% | 232.4 | 97% |
| 5120 | 17408 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<5120, 17408>, Fp8SlicedInstance<8, 8, 2>>` (T 2-8) | 2 | 0.94 | 0.8% | 234.3 | 97% |
| 5120 | 17408 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<5120, 17408>, Fp8SlicedInstance<8, 8, 2>>` (T 2-8) | 3 | 1.40 | 1.1% | 233.9 | 97% |
| 5120 | 17408 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<5120, 17408>, Fp8SlicedInstance<8, 8, 2>>` (T 2-8) | 4 | 1.85 | 1.5% | 231.5 | 96% |
| 5120 | 17408 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<5120, 17408>, Fp8SlicedInstance<8, 8, 2>>` (T 2-8) | 5 | 2.31 | 1.9% | 231.7 | 96% |
| 5120 | 17408 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<5120, 17408>, Fp8SlicedInstance<8, 8, 2>>` (T 2-8) | 6 | 2.75 | 2.2% | 229.8 | 96% |
| 5120 | 17408 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<5120, 17408>, Fp8SlicedInstance<8, 8, 2>>` (T 2-8) | 7 | 3.22 | 2.6% | 231.1 | 96% |
| 5120 | 17408 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<5120, 17408>, Fp8SlicedInstance<8, 8, 2>>` (T 2-8) | 8 | 3.67 | 3.0% | 230.6 | 96% |
| 5120 | 17408 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<5120, 17408>, Fp8SlicedInstance<16, 8, 2>>` (T 9-16) | 9 | 4.14 | 3.3% | 231.3 | 96% |
| 5120 | 17408 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<5120, 17408>, Fp8SlicedInstance<16, 8, 2>>` (T 9-16) | 12 | 5.59 | 4.5% | 234.3 | 97% |
| 5120 | 17408 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<5120, 17408>, Fp8SlicedInstance<16, 8, 2>>` (T 9-16) | 16 | 7.45 | 6.0% | 234.7 | 98% |
| 5120 | 17408 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<5120, 17408>, Fp8SlicedInstance<32, 8, 2>>` (T 17-32) | 17 | 7.75 | 6.3% | 230.0 | 96% |
| 5120 | 17408 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<5120, 17408>, Fp8SlicedInstance<32, 8, 2>>` (T 17-32) | 24 | 10.72 | 8.7% | 226.1 | 94% |
| 5120 | 17408 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<5120, 17408>, Fp8SlicedInstance<32, 8, 2>>` (T 17-32) | 32 | 14.22 | 11% | 225.7 | 94% |
| 5120 | 17408 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 17408>, Fp8A16TmaT64R128>` (T 33-64) | 33 | 14.48 | 12% | 223.0 | 93% |
| 5120 | 17408 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 17408>, Fp8A16TmaT64R128>` (T 33-64) | 48 | 20.95 | 17% | 223.6 | 93% |
| 5120 | 17408 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 17408>, Fp8A16TmaT64R128>` (T 33-64) | 64 | 27.73 | 22% | 223.7 | 93% |
| 5120 | 17408 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 17408>, Fp8A16TmaMmaSchedule<64, 256, 2, 8, 2, 1>>` (T 65-128) | 65 | 28.02 | 23% | 222.7 | 93% |
| 5120 | 17408 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 17408>, Fp8A16TmaMmaSchedule<64, 256, 2, 8, 2, 1>>` (T 65-128) | 128 | 53.97 | 44% | 224.5 | 93% |
| 5120 | 17408 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 17408>, Fp8A16TmaT32R128>` (T 129-192) | 129 | 42.94 | 35% | 177.3 | 74% |
| 5120 | 17408 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 17408>, Fp8A16TmaT32R128>` (T 129-192) | 192 | 66.73 | 54% | 190.7 | 79% |
| 5120 | 17408 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 17408>, Fp8A16TmaMmaSchedule<128, 128, 4, 4, 2, 1>>` (T 193-1024) | 193 | 64.14 | 52% | 182.4 | 76% |
| 5120 | 17408 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 17408>, Fp8A16TmaMmaSchedule<128, 128, 4, 4, 2, 1>>` (T 193-1024) | 256 | 83.17 | 67% | 183.5 | 76% |
| 5120 | 17408 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 17408>, Fp8A16TmaMmaSchedule<128, 128, 4, 4, 2, 1>>` (T 193-1024) | 384 | 87.05 | 70% | 135.4 | 56% |
| 5120 | 17408 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 17408>, Fp8A16TmaMmaSchedule<128, 128, 4, 4, 2, 1>>` (T 193-1024) | 512 | 86.38 | 70% | 106.2 | 44% |
| 5120 | 17408 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 17408>, Fp8A16TmaMmaSchedule<128, 128, 4, 4, 2, 1>>` (T 193-1024) | 640 | 86.11 | 70% | 89.0 | 37% |
| 5120 | 17408 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 17408>, Fp8A16TmaMmaSchedule<128, 128, 4, 4, 2, 1>>` (T 193-1024) | 768 | 101.14 | 82% | 91.4 | 38% |
| 5120 | 17408 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 17408>, Fp8A16TmaMmaSchedule<128, 128, 4, 4, 2, 1>>` (T 193-1024) | 896 | 97.01 | 78% | 78.7 | 33% |
| 5120 | 17408 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 17408>, Fp8A16TmaMmaSchedule<128, 128, 4, 4, 2, 1>>` (T 193-1024) | 1024 | 94.17 | 76% | 69.8 | 29% |
| 5120 | 17408 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 17408>, Fp8A16TmaT64R256G8>` (T 1025-1537) | 1025 | 79.88 | 65% | 59.2 | 25% |
| 5120 | 17408 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 17408>, Fp8A16TmaT64R256G8>` (T 1025-1537) | 1152 | 95.13 | 77% | 65.3 | 27% |
| 5120 | 17408 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 17408>, Fp8A16TmaT64R256G8>` (T 1025-1537) | 1280 | 88.01 | 71% | 56.6 | 24% |
| 5120 | 17408 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 17408>, Fp8A16TmaT64R256G8>` (T 1025-1537) | 1408 | 86.43 | 70% | 52.5 | 22% |
| 5120 | 17408 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 17408>, Fp8A16TmaT64R256G8>` (T 1025-1537) | 1536 | 99.93 | 81% | 57.8 | 24% |
| 5120 | 17408 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 17408>, Fp8A16TmaT64R256G8>` (T 1025-1537) | 1537 | 87.61 | 71% | 50.6 | 21% |
| 5120 | 17408 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 17408>, Fp8A16TmaT128R128G8>` (T 1538-2048) | 1538 | 88.66 | 72% | 51.2 | 21% |
| 5120 | 17408 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 17408>, Fp8A16TmaT128R128G8>` (T 1538-2048) | 1664 | 95.32 | 77% | 52.7 | 22% |
| 5120 | 17408 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 17408>, Fp8A16TmaT128R128G8>` (T 1538-2048) | 1792 | 94.11 | 76% | 50.0 | 21% |
| 5120 | 17408 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 17408>, Fp8A16TmaT128R128G8>` (T 1538-2048) | 1920 | 93.63 | 76% | 48.0 | 20% |
| 5120 | 17408 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 17408>, Fp8A16TmaT128R128G8>` (T 1538-2048) | 2048 | 87.58 | 71% | 43.5 | 18% |
| 5120 | 17408 | a16 | `fp8_linear_a16_tma<Fp8Geometry<5120, 17408>, Fp8A16TmaT64R256G8>` (T>=2049) | — | — | — | — | — |
| 5120 | 17408 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 17408>, Fp8A8TmaMmaSchedule<32, 64, 128, 1, 2, 3, 2>>` (T 1-64) | 1 | 0.46 | 0.2% | 232.3 | 97% |
| 5120 | 17408 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 17408>, Fp8A8TmaMmaSchedule<32, 64, 128, 1, 2, 3, 2>>` (T 1-64) | 2 | 0.94 | 0.4% | 234.4 | 97% |
| 5120 | 17408 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 17408>, Fp8A8TmaMmaSchedule<32, 64, 128, 1, 2, 3, 2>>` (T 1-64) | 3 | 1.40 | 0.6% | 233.2 | 97% |
| 5120 | 17408 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 17408>, Fp8A8TmaMmaSchedule<32, 64, 128, 1, 2, 3, 2>>` (T 1-64) | 4 | 1.86 | 0.8% | 232.5 | 97% |
| 5120 | 17408 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 17408>, Fp8A8TmaMmaSchedule<32, 64, 128, 1, 2, 3, 2>>` (T 1-64) | 5 | 2.30 | 0.9% | 231.0 | 96% |
| 5120 | 17408 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 17408>, Fp8A8TmaMmaSchedule<32, 64, 128, 1, 2, 3, 2>>` (T 1-64) | 6 | 2.74 | 1.1% | 228.7 | 95% |
| 5120 | 17408 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 17408>, Fp8A8TmaMmaSchedule<32, 64, 128, 1, 2, 3, 2>>` (T 1-64) | 7 | 3.22 | 1.3% | 231.1 | 96% |
| 5120 | 17408 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 17408>, Fp8A8TmaMmaSchedule<32, 64, 128, 1, 2, 3, 2>>` (T 1-64) | 8 | 3.69 | 1.5% | 231.3 | 96% |
| 5120 | 17408 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 17408>, Fp8A8TmaMmaSchedule<32, 64, 128, 1, 2, 3, 2>>` (T 1-64) | 12 | 5.59 | 2.3% | 234.3 | 97% |
| 5120 | 17408 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 17408>, Fp8A8TmaMmaSchedule<32, 64, 128, 1, 2, 3, 2>>` (T 1-64) | 16 | 7.43 | 3.0% | 234.1 | 97% |
| 5120 | 17408 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 17408>, Fp8A8TmaMmaSchedule<32, 64, 128, 1, 2, 3, 2>>` (T 1-64) | 24 | 10.40 | 4.3% | 219.3 | 91% |
| 5120 | 17408 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 17408>, Fp8A8TmaMmaSchedule<32, 64, 128, 1, 2, 3, 2>>` (T 1-64) | 32 | 13.79 | 5.7% | 219.0 | 91% |
| 5120 | 17408 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 17408>, Fp8A8TmaMmaSchedule<32, 64, 128, 1, 2, 3, 2>>` (T 1-64) | 48 | 20.19 | 8.3% | 215.4 | 90% |
| 5120 | 17408 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 17408>, Fp8A8TmaMmaSchedule<32, 64, 128, 1, 2, 3, 2>>` (T 1-64) | 64 | 26.54 | 11% | 214.1 | 89% |
| 5120 | 17408 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 17408>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 3, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 65-128) | 65 | 26.33 | 11% | 209.2 | 87% |
| 5120 | 17408 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 17408>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 3, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 65-128) | 128 | 49.10 | 20% | 204.2 | 85% |
| 5120 | 17408 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 17408>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 3, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 129-512) | 129 | 47.70 | 20% | 197.0 | 82% |
| 5120 | 17408 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 17408>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 3, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 129-512) | 256 | 85.73 | 35% | 189.1 | 79% |
| 5120 | 17408 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 17408>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 3, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 129-512) | 384 | 117.73 | 48% | 183.1 | 76% |
| 5120 | 17408 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 17408>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 3, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 129-512) | 512 | 137.16 | 56% | 168.6 | 70% |
| 5120 | 17408 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 17408>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<192, 128, 128, 3, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 513-768) | 513 | 136.58 | 56% | 167.7 | 70% |
| 5120 | 17408 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 17408>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<192, 128, 128, 3, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 513-768) | 640 | 141.41 | 58% | 146.2 | 61% |
| 5120 | 17408 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 17408>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<192, 128, 128, 3, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 513-768) | 768 | 157.49 | 65% | 142.3 | 59% |
| 5120 | 17408 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 17408>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 769-2048) | 769 | 143.52 | 59% | 129.6 | 54% |
| 5120 | 17408 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 17408>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 769-2048) | 896 | 157.42 | 65% | 127.6 | 53% |
| 5120 | 17408 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 17408>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 769-2048) | 1024 | 161.63 | 66% | 119.8 | 50% |
| 5120 | 17408 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 17408>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 769-2048) | 1152 | 154.16 | 63% | 105.9 | 44% |
| 5120 | 17408 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 17408>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 769-2048) | 1280 | 158.40 | 65% | 101.9 | 42% |
| 5120 | 17408 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 17408>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 769-2048) | 1408 | 148.38 | 61% | 90.2 | 37% |
| 5120 | 17408 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 17408>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 769-2048) | 1536 | 166.52 | 68% | 96.3 | 40% |
| 5120 | 17408 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 17408>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 769-2048) | 1664 | 152.17 | 62% | 84.2 | 35% |
| 5120 | 17408 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 17408>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 769-2048) | 1792 | 153.61 | 63% | 81.7 | 34% |
| 5120 | 17408 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 17408>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 769-2048) | 1920 | 153.75 | 63% | 78.9 | 33% |
| 5120 | 17408 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 17408>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 769-2048) | 2048 | 150.19 | 62% | 74.6 | 31% |
| 5120 | 17408 | a8 | `launch_fp8_a8_tma<Fp8Geometry<5120, 17408>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 3, 1, Fp8MmaRaster::Grouped, 8>, kGb10SplitWaveCtas, 4, 8>>` (T>=2049) | — | — | — | — | — |
| 13312 | 2560 | a16 | `fp8_linear_a16_gemv<Fp8Geometry<13312, 2560>, Fp8A16GemvSchedule<8, 2, 16, 1, Fp8CodeCache::Default, 2, 2>>` (T=1) | 1 | 0.46 | 0.4% | 230.2 | 96% |
| 13312 | 2560 | a16 | `fp8_linear_a16_simt<Fp8Geometry<13312, 2560>, 8, Fp8A16SimtSchedule<4, 2, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked, Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>>` (T 2-8) | 2 | 0.93 | 0.8% | 232.0 | 96% |
| 13312 | 2560 | a16 | `fp8_linear_a16_simt<Fp8Geometry<13312, 2560>, 8, Fp8A16SimtSchedule<4, 2, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked, Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>>` (T 2-8) | 3 | 1.38 | 1.1% | 230.7 | 96% |
| 13312 | 2560 | a16 | `fp8_linear_a16_simt<Fp8Geometry<13312, 2560>, 8, Fp8A16SimtSchedule<4, 2, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked, Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>>` (T 2-8) | 4 | 1.84 | 1.5% | 231.1 | 96% |
| 13312 | 2560 | a16 | `fp8_linear_a16_simt<Fp8Geometry<13312, 2560>, 8, Fp8A16SimtSchedule<4, 2, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked, Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>>` (T 2-8) | 5 | 2.31 | 1.9% | 232.7 | 97% |
| 13312 | 2560 | a16 | `fp8_linear_a16_simt<Fp8Geometry<13312, 2560>, 8, Fp8A16SimtSchedule<4, 2, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked, Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>>` (T 2-8) | 6 | 2.76 | 2.2% | 231.2 | 96% |
| 13312 | 2560 | a16 | `fp8_linear_a16_simt<Fp8Geometry<13312, 2560>, 8, Fp8A16SimtSchedule<4, 2, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked, Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>>` (T 2-8) | 7 | 3.19 | 2.6% | 229.9 | 96% |
| 13312 | 2560 | a16 | `fp8_linear_a16_simt<Fp8Geometry<13312, 2560>, 8, Fp8A16SimtSchedule<4, 2, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked, Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>>` (T 2-8) | 8 | 3.60 | 2.9% | 226.7 | 94% |
| 13312 | 2560 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<13312, 2560>, Fp8SlicedInstance<16, 8, 2>>` (T 9-16) | 9 | 4.13 | 3.3% | 231.6 | 96% |
| 13312 | 2560 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<13312, 2560>, Fp8SlicedInstance<16, 8, 2>>` (T 9-16) | 12 | 5.51 | 4.5% | 232.5 | 97% |
| 13312 | 2560 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<13312, 2560>, Fp8SlicedInstance<16, 8, 2>>` (T 9-16) | 16 | 7.46 | 6.0% | 236.6 | 98% |
| 13312 | 2560 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<13312, 2560>, Fp8SlicedR2Instance<32, 8, 2>>` (T 17-32) | 17 | 7.60 | 6.1% | 227.4 | 95% |
| 13312 | 2560 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<13312, 2560>, Fp8SlicedR2Instance<32, 8, 2>>` (T 17-32) | 24 | 10.73 | 8.7% | 228.8 | 95% |
| 13312 | 2560 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<13312, 2560>, Fp8SlicedR2Instance<32, 8, 2>>` (T 17-32) | 32 | 14.12 | 11% | 227.4 | 95% |
| 13312 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<13312, 2560>, Fp8A16TmaT32R64S3>` (T 33-160) | 33 | 14.19 | 11% | 221.7 | 92% |
| 13312 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<13312, 2560>, Fp8A16TmaT32R64S3>` (T 33-160) | 48 | 20.77 | 17% | 226.2 | 94% |
| 13312 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<13312, 2560>, Fp8A16TmaT32R64S3>` (T 33-160) | 64 | 27.52 | 22% | 228.0 | 95% |
| 13312 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<13312, 2560>, Fp8A16TmaT32R64S3>` (T 33-160) | 128 | 52.01 | 42% | 227.5 | 95% |
| 13312 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<13312, 2560>, Fp8A16TmaT32R64S3>` (T 33-160) | 160 | 59.88 | 48% | 215.2 | 89% |
| 13312 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<13312, 2560>, Fp8A16TmaT64R128S3B2K64>` (T 161-320) | 161 | 57.67 | 47% | 206.1 | 86% |
| 13312 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<13312, 2560>, Fp8A16TmaT64R128S3B2K64>` (T 161-320) | 256 | 85.25 | 69% | 206.3 | 86% |
| 13312 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<13312, 2560>, Fp8A16TmaT64R128S3B2K64>` (T 161-320) | 320 | 90.31 | 73% | 183.3 | 76% |
| 13312 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<13312, 2560>, Fp8A16TmaT128R128S2B2K64>` (T 321-4096) | 321 | 77.46 | 63% | 156.8 | 65% |
| 13312 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<13312, 2560>, Fp8A16TmaT128R128S2B2K64>` (T 321-4096) | 384 | 91.96 | 74% | 162.7 | 68% |
| 13312 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<13312, 2560>, Fp8A16TmaT128R128S2B2K64>` (T 321-4096) | 512 | 96.83 | 78% | 139.7 | 58% |
| 13312 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<13312, 2560>, Fp8A16TmaT128R128S2B2K64>` (T 321-4096) | 640 | 99.57 | 80% | 124.2 | 52% |
| 13312 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<13312, 2560>, Fp8A16TmaT128R128S2B2K64>` (T 321-4096) | 768 | 101.83 | 82% | 113.8 | 47% |
| 13312 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<13312, 2560>, Fp8A16TmaT128R128S2B2K64>` (T 321-4096) | 896 | 97.80 | 79% | 100.2 | 42% |
| 13312 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<13312, 2560>, Fp8A16TmaT128R128S2B2K64>` (T 321-4096) | 1024 | 99.36 | 80% | 94.8 | 39% |
| 13312 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<13312, 2560>, Fp8A16TmaT128R128S2B2K64>` (T 321-4096) | 1152 | 100.65 | 81% | 90.6 | 38% |
| 13312 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<13312, 2560>, Fp8A16TmaT128R128S2B2K64>` (T 321-4096) | 1280 | 101.93 | 82% | 87.3 | 36% |
| 13312 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<13312, 2560>, Fp8A16TmaT128R128S2B2K64>` (T 321-4096) | 1408 | 102.33 | 83% | 84.0 | 35% |
| 13312 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<13312, 2560>, Fp8A16TmaT128R128S2B2K64>` (T 321-4096) | 1536 | 102.87 | 83% | 81.4 | 34% |
| 13312 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<13312, 2560>, Fp8A16TmaT128R128S2B2K64>` (T 321-4096) | 1664 | 100.89 | 82% | 77.3 | 32% |
| 13312 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<13312, 2560>, Fp8A16TmaT128R128S2B2K64>` (T 321-4096) | 1792 | 100.75 | 81% | 75.1 | 31% |
| 13312 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<13312, 2560>, Fp8A16TmaT128R128S2B2K64>` (T 321-4096) | 1920 | 102.09 | 83% | 74.2 | 31% |
| 13312 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<13312, 2560>, Fp8A16TmaT128R128S2B2K64>` (T 321-4096) | 2048 | 101.74 | 82% | 72.2 | 30% |
| 13312 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<13312, 2560>, Fp8A16TmaT128R128G8>` (T>=4097) | — | — | — | — | — |
| 14336 | 5120 | a16 | `fp8_linear_a16_gemv<Fp8Geometry<14336, 5120>, Fp8A16GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>>` (T=1) | 1 | 0.46 | 0.4% | 232.3 | 97% |
| 14336 | 5120 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<14336, 5120>, Fp8SlicedInstance<16, 8, 2>>` (T 2-16) | 2 | 0.93 | 0.8% | 233.9 | 97% |
| 14336 | 5120 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<14336, 5120>, Fp8SlicedInstance<16, 8, 2>>` (T 2-16) | 3 | 1.40 | 1.1% | 233.3 | 97% |
| 14336 | 5120 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<14336, 5120>, Fp8SlicedInstance<16, 8, 2>>` (T 2-16) | 4 | 1.86 | 1.5% | 233.4 | 97% |
| 14336 | 5120 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<14336, 5120>, Fp8SlicedInstance<16, 8, 2>>` (T 2-16) | 5 | 2.33 | 1.9% | 233.6 | 97% |
| 14336 | 5120 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<14336, 5120>, Fp8SlicedInstance<16, 8, 2>>` (T 2-16) | 6 | 2.78 | 2.2% | 232.2 | 97% |
| 14336 | 5120 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<14336, 5120>, Fp8SlicedInstance<16, 8, 2>>` (T 2-16) | 7 | 3.24 | 2.6% | 232.3 | 97% |
| 14336 | 5120 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<14336, 5120>, Fp8SlicedInstance<16, 8, 2>>` (T 2-16) | 8 | 3.71 | 3.0% | 233.2 | 97% |
| 14336 | 5120 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<14336, 5120>, Fp8SlicedInstance<16, 8, 2>>` (T 2-16) | 12 | 5.57 | 4.5% | 233.7 | 97% |
| 14336 | 5120 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<14336, 5120>, Fp8SlicedInstance<16, 8, 2>>` (T 2-16) | 16 | 7.40 | 6.0% | 233.4 | 97% |
| 14336 | 5120 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<14336, 5120>, Fp8SlicedInstance<32, 8, 2>>` (T 17-32) | 17 | 7.84 | 6.3% | 232.8 | 97% |
| 14336 | 5120 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<14336, 5120>, Fp8SlicedInstance<32, 8, 2>>` (T 17-32) | 24 | 11.07 | 8.9% | 233.7 | 97% |
| 14336 | 5120 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<14336, 5120>, Fp8SlicedInstance<32, 8, 2>>` (T 17-32) | 32 | 14.66 | 12% | 233.1 | 97% |
| 14336 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<14336, 5120>, Fp8A16TmaT32R64S3>` (T 33-64) | 33 | 14.39 | 12% | 221.9 | 92% |
| 14336 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<14336, 5120>, Fp8A16TmaT32R64S3>` (T 33-64) | 48 | 20.79 | 17% | 222.2 | 92% |
| 14336 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<14336, 5120>, Fp8A16TmaT32R64S3>` (T 33-64) | 64 | 27.48 | 22% | 222.1 | 92% |
| 14336 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<14336, 5120>, Fp8A16TmaT64R128>` (T 65-192) | 65 | 27.59 | 22% | 219.6 | 91% |
| 14336 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<14336, 5120>, Fp8A16TmaT64R128>` (T 65-192) | 128 | 52.70 | 43% | 219.9 | 91% |
| 14336 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<14336, 5120>, Fp8A16TmaT64R128>` (T 65-192) | 192 | 74.83 | 60% | 214.8 | 89% |
| 14336 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<14336, 5120>, Fp8A16TmaMmaSchedule<128, 128, 4, 4, 2, 1>>` (T 193-256) | 193 | 69.20 | 56% | 197.7 | 82% |
| 14336 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<14336, 5120>, Fp8A16TmaMmaSchedule<128, 128, 4, 4, 2, 1>>` (T 193-256) | 256 | 89.55 | 72% | 198.7 | 83% |
| 14336 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<14336, 5120>, Fp8A16TmaT96R128S3K64>` (T 257-288) | 257 | 76.79 | 62% | 169.8 | 71% |
| 14336 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<14336, 5120>, Fp8A16TmaT96R128S3K64>` (T 257-288) | 288 | 85.33 | 69% | 170.8 | 71% |
| 14336 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<14336, 5120>, Fp8A16TmaT128R128S2B2K64>` (T 289-2304) | 289 | 76.75 | 62% | 153.2 | 64% |
| 14336 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<14336, 5120>, Fp8A16TmaT128R128S2B2K64>` (T 289-2304) | 384 | 99.76 | 81% | 156.4 | 65% |
| 14336 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<14336, 5120>, Fp8A16TmaT128R128S2B2K64>` (T 289-2304) | 512 | 97.11 | 78% | 120.6 | 50% |
| 14336 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<14336, 5120>, Fp8A16TmaT128R128S2B2K64>` (T 289-2304) | 640 | 101.29 | 82% | 106.0 | 44% |
| 14336 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<14336, 5120>, Fp8A16TmaT128R128S2B2K64>` (T 289-2304) | 768 | 104.08 | 84% | 95.4 | 40% |
| 14336 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<14336, 5120>, Fp8A16TmaT128R128S2B2K64>` (T 289-2304) | 896 | 100.37 | 81% | 82.6 | 34% |
| 14336 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<14336, 5120>, Fp8A16TmaT128R128S2B2K64>` (T 289-2304) | 1024 | 103.10 | 83% | 77.7 | 32% |
| 14336 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<14336, 5120>, Fp8A16TmaT128R128S2B2K64>` (T 289-2304) | 1152 | 103.49 | 84% | 72.4 | 30% |
| 14336 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<14336, 5120>, Fp8A16TmaT128R128S2B2K64>` (T 289-2304) | 1280 | 101.39 | 82% | 66.5 | 28% |
| 14336 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<14336, 5120>, Fp8A16TmaT128R128S2B2K64>` (T 289-2304) | 1408 | 102.89 | 83% | 63.8 | 27% |
| 14336 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<14336, 5120>, Fp8A16TmaT128R128S2B2K64>` (T 289-2304) | 1536 | 101.11 | 82% | 59.7 | 25% |
| 14336 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<14336, 5120>, Fp8A16TmaT128R128S2B2K64>` (T 289-2304) | 1664 | 97.57 | 79% | 55.2 | 23% |
| 14336 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<14336, 5120>, Fp8A16TmaT128R128S2B2K64>` (T 289-2304) | 1792 | 97.58 | 79% | 53.1 | 22% |
| 14336 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<14336, 5120>, Fp8A16TmaT128R128S2B2K64>` (T 289-2304) | 1920 | 96.75 | 78% | 50.9 | 21% |
| 14336 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<14336, 5120>, Fp8A16TmaT128R128S2B2K64>` (T 289-2304) | 2048 | 93.54 | 76% | 47.6 | 20% |
| 14336 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<14336, 5120>, Fp8A16TmaT128R128G8>` (T 2305-3072) | — | — | — | — | — |
| 14336 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<14336, 5120>, Fp8A16TmaT64R256G8>` (T>=3073) | — | — | — | — | — |
| 14336 | 5120 | a8 | `launch_fp8_a8<Fp8Geometry<14336, 5120>, Fp8A8T32R32K128>` (T 1-32) | 1 | 0.46 | 0.2% | 232.3 | 97% |
| 14336 | 5120 | a8 | `launch_fp8_a8<Fp8Geometry<14336, 5120>, Fp8A8T32R32K128>` (T 1-32) | 2 | 0.94 | 0.4% | 234.7 | 98% |
| 14336 | 5120 | a8 | `launch_fp8_a8<Fp8Geometry<14336, 5120>, Fp8A8T32R32K128>` (T 1-32) | 3 | 1.41 | 0.6% | 234.8 | 98% |
| 14336 | 5120 | a8 | `launch_fp8_a8<Fp8Geometry<14336, 5120>, Fp8A8T32R32K128>` (T 1-32) | 4 | 1.86 | 0.8% | 233.5 | 97% |
| 14336 | 5120 | a8 | `launch_fp8_a8<Fp8Geometry<14336, 5120>, Fp8A8T32R32K128>` (T 1-32) | 5 | 2.31 | 0.9% | 232.0 | 96% |
| 14336 | 5120 | a8 | `launch_fp8_a8<Fp8Geometry<14336, 5120>, Fp8A8T32R32K128>` (T 1-32) | 6 | 2.77 | 1.1% | 231.3 | 96% |
| 14336 | 5120 | a8 | `launch_fp8_a8<Fp8Geometry<14336, 5120>, Fp8A8T32R32K128>` (T 1-32) | 7 | 3.25 | 1.3% | 233.0 | 97% |
| 14336 | 5120 | a8 | `launch_fp8_a8<Fp8Geometry<14336, 5120>, Fp8A8T32R32K128>` (T 1-32) | 8 | 3.69 | 1.5% | 231.6 | 96% |
| 14336 | 5120 | a8 | `launch_fp8_a8<Fp8Geometry<14336, 5120>, Fp8A8T32R32K128>` (T 1-32) | 12 | 5.54 | 2.3% | 232.3 | 97% |
| 14336 | 5120 | a8 | `launch_fp8_a8<Fp8Geometry<14336, 5120>, Fp8A8T32R32K128>` (T 1-32) | 16 | 7.40 | 3.0% | 233.3 | 97% |
| 14336 | 5120 | a8 | `launch_fp8_a8<Fp8Geometry<14336, 5120>, Fp8A8T32R32K128>` (T 1-32) | 24 | 10.43 | 4.3% | 220.2 | 92% |
| 14336 | 5120 | a8 | `launch_fp8_a8<Fp8Geometry<14336, 5120>, Fp8A8T32R32K128>` (T 1-32) | 32 | 13.83 | 5.7% | 219.8 | 91% |
| 14336 | 5120 | a8 | `launch_fp8_a8<Fp8Geometry<14336, 5120>, Fp8A8T32R128K128>` (T 33-96) | 33 | 14.13 | 5.8% | 217.9 | 91% |
| 14336 | 5120 | a8 | `launch_fp8_a8<Fp8Geometry<14336, 5120>, Fp8A8T32R128K128>` (T 33-96) | 48 | 20.43 | 8.4% | 218.3 | 91% |
| 14336 | 5120 | a8 | `launch_fp8_a8<Fp8Geometry<14336, 5120>, Fp8A8T32R128K128>` (T 33-96) | 64 | 27.08 | 11% | 218.8 | 91% |
| 14336 | 5120 | a8 | `launch_fp8_a8<Fp8Geometry<14336, 5120>, Fp8A8T32R128K128>` (T 33-96) | 96 | 39.91 | 16% | 218.5 | 91% |
| 14336 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<14336, 5120>, Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 2, 1>>` (T 97-128) | 97 | 40.21 | 16% | 218.0 | 91% |
| 14336 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<14336, 5120>, Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 2, 1>>` (T 97-128) | 128 | 52.16 | 21% | 217.7 | 90% |
| 14336 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<14336, 5120>, Fp8A8TmaMmaSchedule<64, 256, 128, 2, 4, 2, 1>>` (T 129-192) | 129 | 52.86 | 22% | 219.0 | 91% |
| 14336 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<14336, 5120>, Fp8A8TmaMmaSchedule<64, 256, 128, 2, 4, 2, 1>>` (T 129-192) | 192 | 76.07 | 31% | 218.3 | 91% |
| 14336 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<14336, 5120>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 193-3072) | 193 | 74.81 | 31% | 213.7 | 89% |
| 14336 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<14336, 5120>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 193-3072) | 256 | 96.37 | 39% | 213.8 | 89% |
| 14336 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<14336, 5120>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 193-3072) | 384 | 131.43 | 54% | 206.0 | 86% |
| 14336 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<14336, 5120>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 193-3072) | 512 | 156.91 | 64% | 194.9 | 81% |
| 14336 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<14336, 5120>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 193-3072) | 640 | 169.07 | 69% | 176.9 | 74% |
| 14336 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<14336, 5120>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 193-3072) | 768 | 186.39 | 76% | 170.8 | 71% |
| 14336 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<14336, 5120>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 193-3072) | 896 | 181.21 | 74% | 149.2 | 62% |
| 14336 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<14336, 5120>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 193-3072) | 1024 | 181.06 | 74% | 136.4 | 57% |
| 14336 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<14336, 5120>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 193-3072) | 1152 | 182.94 | 75% | 127.9 | 53% |
| 14336 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<14336, 5120>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 193-3072) | 1280 | 179.23 | 73% | 117.5 | 49% |
| 14336 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<14336, 5120>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 193-3072) | 1408 | 182.71 | 75% | 113.3 | 47% |
| 14336 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<14336, 5120>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 193-3072) | 1536 | 190.18 | 78% | 112.3 | 47% |
| 14336 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<14336, 5120>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 193-3072) | 1664 | 185.23 | 76% | 104.8 | 44% |
| 14336 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<14336, 5120>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 193-3072) | 1792 | 184.05 | 75% | 100.2 | 42% |
| 14336 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<14336, 5120>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 193-3072) | 1920 | 185.00 | 76% | 97.2 | 40% |
| 14336 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<14336, 5120>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 193-3072) | 2048 | 180.81 | 74% | 92.1 | 38% |
| 14336 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<14336, 5120>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 3, 1, Fp8MmaRaster::Grouped, 8>, kGb10SplitWaveCtas, 4, 8>>` (T>=3073) | — | — | — | — | — |
| 16384 | 2560 | a16 | `fp8_linear_a16_gemv<Fp8Geometry<16384, 2560>, Fp8A16GemvSchedule<8, 2, 16, 1, Fp8CodeCache::Default, 2, 2>>` (T=1) | 1 | 0.46 | 0.4% | 230.7 | 96% |
| 16384 | 2560 | a16 | `fp8_linear_a16_simt<Fp8Geometry<16384, 2560>, 8, Fp8A16SimtSchedule<4, 2, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked, Fp8CodeCache::Streaming, 1, Fp8SimtBlockOrder::RowsContiguous, 1>>` (T 2-8) | 2 | 0.93 | 0.8% | 232.3 | 97% |
| 16384 | 2560 | a16 | `fp8_linear_a16_simt<Fp8Geometry<16384, 2560>, 8, Fp8A16SimtSchedule<4, 2, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked, Fp8CodeCache::Streaming, 1, Fp8SimtBlockOrder::RowsContiguous, 1>>` (T 2-8) | 3 | 1.39 | 1.1% | 232.5 | 97% |
| 16384 | 2560 | a16 | `fp8_linear_a16_simt<Fp8Geometry<16384, 2560>, 8, Fp8A16SimtSchedule<4, 2, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked, Fp8CodeCache::Streaming, 1, Fp8SimtBlockOrder::RowsContiguous, 1>>` (T 2-8) | 4 | 1.85 | 1.5% | 232.6 | 97% |
| 16384 | 2560 | a16 | `fp8_linear_a16_simt<Fp8Geometry<16384, 2560>, 8, Fp8A16SimtSchedule<4, 2, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked, Fp8CodeCache::Streaming, 1, Fp8SimtBlockOrder::RowsContiguous, 1>>` (T 2-8) | 5 | 2.32 | 1.9% | 232.8 | 97% |
| 16384 | 2560 | a16 | `fp8_linear_a16_simt<Fp8Geometry<16384, 2560>, 8, Fp8A16SimtSchedule<4, 2, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked, Fp8CodeCache::Streaming, 1, Fp8SimtBlockOrder::RowsContiguous, 1>>` (T 2-8) | 6 | 2.73 | 2.2% | 229.2 | 95% |
| 16384 | 2560 | a16 | `fp8_linear_a16_simt<Fp8Geometry<16384, 2560>, 8, Fp8A16SimtSchedule<4, 2, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked, Fp8CodeCache::Streaming, 1, Fp8SimtBlockOrder::RowsContiguous, 1>>` (T 2-8) | 7 | 3.16 | 2.6% | 227.7 | 95% |
| 16384 | 2560 | a16 | `fp8_linear_a16_simt<Fp8Geometry<16384, 2560>, 8, Fp8A16SimtSchedule<4, 2, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked, Fp8CodeCache::Streaming, 1, Fp8SimtBlockOrder::RowsContiguous, 1>>` (T 2-8) | 8 | 3.65 | 3.0% | 229.8 | 96% |
| 16384 | 2560 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<16384, 2560>, Fp8SlicedInstance<32, 8, 2>>` (T 9-16) | 9 | 4.20 | 3.4% | 235.2 | 98% |
| 16384 | 2560 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<16384, 2560>, Fp8SlicedInstance<32, 8, 2>>` (T 9-16) | 12 | 5.59 | 4.5% | 235.6 | 98% |
| 16384 | 2560 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<16384, 2560>, Fp8SlicedInstance<32, 8, 2>>` (T 9-16) | 16 | 7.45 | 6.0% | 236.5 | 98% |
| 16384 | 2560 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<16384, 2560>, Fp8SlicedR2Instance<32, 8, 2>>` (T 17-32) | 17 | 7.74 | 6.3% | 231.4 | 96% |
| 16384 | 2560 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<16384, 2560>, Fp8SlicedR2Instance<32, 8, 2>>` (T 17-32) | 24 | 10.81 | 8.7% | 230.3 | 96% |
| 16384 | 2560 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<16384, 2560>, Fp8SlicedR2Instance<32, 8, 2>>` (T 17-32) | 32 | 14.42 | 12% | 232.0 | 96% |
| 16384 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<16384, 2560>, Fp8A16TmaT32R64S3>` (T 33-128) | 33 | 14.39 | 12% | 224.7 | 93% |
| 16384 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<16384, 2560>, Fp8A16TmaT32R64S3>` (T 33-128) | 48 | 20.72 | 17% | 225.3 | 94% |
| 16384 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<16384, 2560>, Fp8A16TmaT32R64S3>` (T 33-128) | 64 | 27.48 | 22% | 227.2 | 94% |
| 16384 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<16384, 2560>, Fp8A16TmaT32R64S3>` (T 33-128) | 128 | 52.21 | 42% | 227.7 | 95% |
| 16384 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<16384, 2560>, Fp8A16TmaMmaSchedule<64, 256, 2, 8, 2, 1>>` (T 129-192) | 129 | 51.60 | 42% | 223.4 | 93% |
| 16384 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<16384, 2560>, Fp8A16TmaMmaSchedule<64, 256, 2, 8, 2, 1>>` (T 129-192) | 192 | 74.60 | 60% | 228.1 | 95% |
| 16384 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<16384, 2560>, Fp8A16TmaT64R128S3B2K64>` (T 193-256) | 193 | 66.47 | 54% | 202.4 | 84% |
| 16384 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<16384, 2560>, Fp8A16TmaT64R128S3B2K64>` (T 193-256) | 256 | 85.27 | 69% | 205.2 | 85% |
| 16384 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<16384, 2560>, Fp8A16TmaT96R128S3K64>` (T 257-288) | 257 | 78.61 | 64% | 188.6 | 78% |
| 16384 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<16384, 2560>, Fp8A16TmaT96R128S3K64>` (T 257-288) | 288 | 86.79 | 70% | 190.0 | 79% |
| 16384 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<16384, 2560>, Fp8A16TmaT128R128S2B2K64>` (T 289-4096) | 289 | 75.44 | 61% | 164.7 | 68% |
| 16384 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<16384, 2560>, Fp8A16TmaT128R128S2B2K64>` (T 289-4096) | 384 | 97.75 | 79% | 171.5 | 71% |
| 16384 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<16384, 2560>, Fp8A16TmaT128R128S2B2K64>` (T 289-4096) | 512 | 98.00 | 79% | 140.0 | 58% |
| 16384 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<16384, 2560>, Fp8A16TmaT128R128S2B2K64>` (T 289-4096) | 640 | 98.22 | 79% | 121.2 | 50% |
| 16384 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<16384, 2560>, Fp8A16TmaT128R128S2B2K64>` (T 289-4096) | 768 | 101.82 | 82% | 112.3 | 47% |
| 16384 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<16384, 2560>, Fp8A16TmaT128R128S2B2K64>` (T 289-4096) | 896 | 101.13 | 82% | 102.2 | 42% |
| 16384 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<16384, 2560>, Fp8A16TmaT128R128S2B2K64>` (T 289-4096) | 1024 | 100.84 | 82% | 94.8 | 39% |
| 16384 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<16384, 2560>, Fp8A16TmaT128R128S2B2K64>` (T 289-4096) | 1152 | 103.27 | 83% | 91.5 | 38% |
| 16384 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<16384, 2560>, Fp8A16TmaT128R128S2B2K64>` (T 289-4096) | 1280 | 102.41 | 83% | 86.3 | 36% |
| 16384 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<16384, 2560>, Fp8A16TmaT128R128S2B2K64>` (T 289-4096) | 1408 | 101.19 | 82% | 81.7 | 34% |
| 16384 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<16384, 2560>, Fp8A16TmaT128R128S2B2K64>` (T 289-4096) | 1536 | 102.48 | 83% | 79.7 | 33% |
| 16384 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<16384, 2560>, Fp8A16TmaT128R128S2B2K64>` (T 289-4096) | 1664 | 101.96 | 82% | 76.7 | 32% |
| 16384 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<16384, 2560>, Fp8A16TmaT128R128S2B2K64>` (T 289-4096) | 1792 | 101.39 | 82% | 74.1 | 31% |
| 16384 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<16384, 2560>, Fp8A16TmaT128R128S2B2K64>` (T 289-4096) | 1920 | 102.68 | 83% | 73.1 | 30% |
| 16384 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<16384, 2560>, Fp8A16TmaT128R128S2B2K64>` (T 289-4096) | 2048 | 101.94 | 82% | 71.0 | 30% |
| 16384 | 2560 | a16 | `fp8_linear_a16_tma<Fp8Geometry<16384, 2560>, Fp8A16TmaT128R128G8>` (T>=4097) | — | — | — | — | — |
| 16384 | 5120 | a16 | `fp8_linear_a16_gemv<Fp8Geometry<16384, 5120>, Fp8A16GemvSchedule<4, 4, 16, 4, Fp8CodeCache::Default, 1, 1>>` (T=1) | 1 | 0.47 | 0.4% | 233.7 | 97% |
| 16384 | 5120 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<16384, 5120>, Fp8SlicedInstance<16, 8, 2>>` (T 2-16) | 2 | 0.94 | 0.8% | 234.5 | 97% |
| 16384 | 5120 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<16384, 5120>, Fp8SlicedInstance<16, 8, 2>>` (T 2-16) | 3 | 1.40 | 1.1% | 234.6 | 98% |
| 16384 | 5120 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<16384, 5120>, Fp8SlicedInstance<16, 8, 2>>` (T 2-16) | 4 | 1.87 | 1.5% | 234.1 | 97% |
| 16384 | 5120 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<16384, 5120>, Fp8SlicedInstance<16, 8, 2>>` (T 2-16) | 5 | 2.32 | 1.9% | 232.9 | 97% |
| 16384 | 5120 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<16384, 5120>, Fp8SlicedInstance<16, 8, 2>>` (T 2-16) | 6 | 2.77 | 2.2% | 231.6 | 96% |
| 16384 | 5120 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<16384, 5120>, Fp8SlicedInstance<16, 8, 2>>` (T 2-16) | 7 | 3.26 | 2.6% | 233.7 | 97% |
| 16384 | 5120 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<16384, 5120>, Fp8SlicedInstance<16, 8, 2>>` (T 2-16) | 8 | 3.69 | 3.0% | 231.9 | 96% |
| 16384 | 5120 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<16384, 5120>, Fp8SlicedInstance<16, 8, 2>>` (T 2-16) | 12 | 5.56 | 4.5% | 233.1 | 97% |
| 16384 | 5120 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<16384, 5120>, Fp8SlicedInstance<16, 8, 2>>` (T 2-16) | 16 | 7.41 | 6.0% | 233.5 | 97% |
| 16384 | 5120 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<16384, 5120>, Fp8SlicedInstance<32, 8, 2>>` (T 17-25) | 17 | 7.85 | 6.3% | 233.0 | 97% |
| 16384 | 5120 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<16384, 5120>, Fp8SlicedInstance<32, 8, 2>>` (T 17-25) | 24 | 11.05 | 8.9% | 233.1 | 97% |
| 16384 | 5120 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<16384, 5120>, Fp8SlicedInstance<32, 8, 2>>` (T 17-25) | 25 | 11.41 | 9.2% | 231.3 | 96% |
| 16384 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<16384, 5120>, Fp8A16TmaT32R64>` (T 26-64) | 26 | 11.30 | 9.1% | 220.2 | 92% |
| 16384 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<16384, 5120>, Fp8A16TmaT32R64>` (T 26-64) | 32 | 13.90 | 11% | 220.9 | 92% |
| 16384 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<16384, 5120>, Fp8A16TmaT32R64>` (T 26-64) | 48 | 20.76 | 17% | 221.7 | 92% |
| 16384 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<16384, 5120>, Fp8A16TmaT32R64>` (T 26-64) | 64 | 27.33 | 22% | 220.6 | 92% |
| 16384 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<16384, 5120>, Fp8A16TmaT32R64S3>` (T 65-128) | 65 | 27.67 | 22% | 220.0 | 91% |
| 16384 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<16384, 5120>, Fp8A16TmaT32R64S3>` (T 65-128) | 128 | 52.45 | 42% | 218.4 | 91% |
| 16384 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<16384, 5120>, Fp8A16TmaT64R128>` (T 129-256) | 129 | 52.08 | 42% | 215.3 | 89% |
| 16384 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<16384, 5120>, Fp8A16TmaT64R128>` (T 129-256) | 256 | 85.98 | 70% | 190.0 | 79% |
| 16384 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<16384, 5120>, Fp8A16TmaT96R128S3K64>` (T 257-288) | 257 | 76.58 | 62% | 168.7 | 70% |
| 16384 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<16384, 5120>, Fp8A16TmaT96R128S3K64>` (T 257-288) | 288 | 84.90 | 69% | 169.2 | 70% |
| 16384 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<16384, 5120>, Fp8A16TmaT128R128S2B2K64>` (T 289-2304) | 289 | 77.39 | 63% | 153.8 | 64% |
| 16384 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<16384, 5120>, Fp8A16TmaT128R128S2B2K64>` (T 289-2304) | 384 | 99.89 | 81% | 155.7 | 65% |
| 16384 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<16384, 5120>, Fp8A16TmaT128R128S2B2K64>` (T 289-2304) | 512 | 101.33 | 82% | 125.0 | 52% |
| 16384 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<16384, 5120>, Fp8A16TmaT128R128S2B2K64>` (T 289-2304) | 640 | 100.26 | 81% | 104.1 | 43% |
| 16384 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<16384, 5120>, Fp8A16TmaT128R128S2B2K64>` (T 289-2304) | 768 | 104.51 | 84% | 94.9 | 39% |
| 16384 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<16384, 5120>, Fp8A16TmaT128R128S2B2K64>` (T 289-2304) | 896 | 103.10 | 83% | 84.0 | 35% |
| 16384 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<16384, 5120>, Fp8A16TmaT128R128S2B2K64>` (T 289-2304) | 1024 | 101.32 | 82% | 75.5 | 31% |
| 16384 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<16384, 5120>, Fp8A16TmaT128R128S2B2K64>` (T 289-2304) | 1152 | 104.63 | 85% | 72.3 | 30% |
| 16384 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<16384, 5120>, Fp8A16TmaT128R128S2B2K64>` (T 289-2304) | 1280 | 103.00 | 83% | 66.7 | 28% |
| 16384 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<16384, 5120>, Fp8A16TmaT128R128S2B2K64>` (T 289-2304) | 1408 | 102.17 | 83% | 62.5 | 26% |
| 16384 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<16384, 5120>, Fp8A16TmaT128R128S2B2K64>` (T 289-2304) | 1536 | 99.39 | 80% | 57.8 | 24% |
| 16384 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<16384, 5120>, Fp8A16TmaT128R128S2B2K64>` (T 289-2304) | 1664 | 98.32 | 79% | 54.8 | 23% |
| 16384 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<16384, 5120>, Fp8A16TmaT128R128S2B2K64>` (T 289-2304) | 1792 | 96.74 | 78% | 51.8 | 22% |
| 16384 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<16384, 5120>, Fp8A16TmaT128R128S2B2K64>` (T 289-2304) | 1920 | 96.47 | 78% | 49.9 | 21% |
| 16384 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<16384, 5120>, Fp8A16TmaT128R128S2B2K64>` (T 289-2304) | 2048 | 94.47 | 76% | 47.3 | 20% |
| 16384 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<16384, 5120>, Fp8A16TmaT128R128G8>` (T 2305-3072) | — | — | — | — | — |
| 16384 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<16384, 5120>, Fp8A16TmaT64R256G8>` (T>=3073) | — | — | — | — | — |
| 16384 | 5120 | a8 | `launch_fp8_a8<Fp8Geometry<16384, 5120>, Fp8A8T32R32K128>` (T 1-32) | 1 | 0.47 | 0.2% | 233.6 | 97% |
| 16384 | 5120 | a8 | `launch_fp8_a8<Fp8Geometry<16384, 5120>, Fp8A8T32R32K128>` (T 1-32) | 2 | 0.94 | 0.4% | 234.5 | 97% |
| 16384 | 5120 | a8 | `launch_fp8_a8<Fp8Geometry<16384, 5120>, Fp8A8T32R32K128>` (T 1-32) | 3 | 1.40 | 0.6% | 233.9 | 97% |
| 16384 | 5120 | a8 | `launch_fp8_a8<Fp8Geometry<16384, 5120>, Fp8A8T32R32K128>` (T 1-32) | 4 | 1.86 | 0.8% | 232.8 | 97% |
| 16384 | 5120 | a8 | `launch_fp8_a8<Fp8Geometry<16384, 5120>, Fp8A8T32R32K128>` (T 1-32) | 5 | 2.33 | 1.0% | 233.5 | 97% |
| 16384 | 5120 | a8 | `launch_fp8_a8<Fp8Geometry<16384, 5120>, Fp8A8T32R32K128>` (T 1-32) | 6 | 2.78 | 1.1% | 232.2 | 97% |
| 16384 | 5120 | a8 | `launch_fp8_a8<Fp8Geometry<16384, 5120>, Fp8A8T32R32K128>` (T 1-32) | 7 | 3.25 | 1.3% | 233.0 | 97% |
| 16384 | 5120 | a8 | `launch_fp8_a8<Fp8Geometry<16384, 5120>, Fp8A8T32R32K128>` (T 1-32) | 8 | 3.71 | 1.5% | 233.2 | 97% |
| 16384 | 5120 | a8 | `launch_fp8_a8<Fp8Geometry<16384, 5120>, Fp8A8T32R32K128>` (T 1-32) | 12 | 5.57 | 2.3% | 233.7 | 97% |
| 16384 | 5120 | a8 | `launch_fp8_a8<Fp8Geometry<16384, 5120>, Fp8A8T32R32K128>` (T 1-32) | 16 | 7.43 | 3.0% | 234.2 | 97% |
| 16384 | 5120 | a8 | `launch_fp8_a8<Fp8Geometry<16384, 5120>, Fp8A8T32R32K128>` (T 1-32) | 24 | 10.43 | 4.3% | 220.1 | 91% |
| 16384 | 5120 | a8 | `launch_fp8_a8<Fp8Geometry<16384, 5120>, Fp8A8T32R32K128>` (T 1-32) | 32 | 13.84 | 5.7% | 219.9 | 91% |
| 16384 | 5120 | a8 | `launch_fp8_a8<Fp8Geometry<16384, 5120>, Fp8A8T64R128K256>` (T 33-64) | 33 | 14.70 | 6.0% | 226.6 | 94% |
| 16384 | 5120 | a8 | `launch_fp8_a8<Fp8Geometry<16384, 5120>, Fp8A8T64R128K256>` (T 33-64) | 48 | 21.39 | 8.8% | 228.3 | 95% |
| 16384 | 5120 | a8 | `launch_fp8_a8<Fp8Geometry<16384, 5120>, Fp8A8T64R128K256>` (T 33-64) | 64 | 28.05 | 11% | 226.4 | 94% |
| 16384 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<16384, 5120>, Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 2, 1>>` (T 65-128) | 65 | 27.59 | 11% | 219.4 | 91% |
| 16384 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<16384, 5120>, Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 2, 1>>` (T 65-128) | 128 | 52.45 | 21% | 218.4 | 91% |
| 16384 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<16384, 5120>, Fp8A8TmaMmaSchedule<192, 128, 128, 3, 4, 2, 1>>` (T 129-192) | 129 | 52.73 | 22% | 218.0 | 91% |
| 16384 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<16384, 5120>, Fp8A8TmaMmaSchedule<192, 128, 128, 3, 4, 2, 1>>` (T 129-192) | 192 | 76.19 | 31% | 218.0 | 91% |
| 16384 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<16384, 5120>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 193-3072) | 193 | 75.13 | 31% | 214.0 | 89% |
| 16384 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<16384, 5120>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 193-3072) | 256 | 96.23 | 39% | 212.7 | 88% |
| 16384 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<16384, 5120>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 193-3072) | 384 | 134.48 | 55% | 209.6 | 87% |
| 16384 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<16384, 5120>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 193-3072) | 512 | 157.44 | 65% | 194.2 | 81% |
| 16384 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<16384, 5120>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 193-3072) | 640 | 167.01 | 68% | 173.3 | 72% |
| 16384 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<16384, 5120>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 193-3072) | 768 | 190.12 | 78% | 172.6 | 72% |
| 16384 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<16384, 5120>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 193-3072) | 896 | 184.00 | 75% | 149.9 | 62% |
| 16384 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<16384, 5120>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 193-3072) | 1024 | 180.43 | 74% | 134.4 | 56% |
| 16384 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<16384, 5120>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 193-3072) | 1152 | 194.41 | 80% | 134.2 | 56% |
| 16384 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<16384, 5120>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 193-3072) | 1280 | 185.62 | 76% | 120.1 | 50% |
| 16384 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<16384, 5120>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 193-3072) | 1408 | 184.28 | 76% | 112.7 | 47% |
| 16384 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<16384, 5120>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 193-3072) | 1536 | 191.69 | 79% | 111.6 | 46% |
| 16384 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<16384, 5120>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 193-3072) | 1664 | 187.91 | 77% | 104.7 | 44% |
| 16384 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<16384, 5120>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 193-3072) | 1792 | 185.03 | 76% | 99.1 | 41% |
| 16384 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<16384, 5120>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 193-3072) | 1920 | 193.13 | 79% | 99.8 | 41% |
| 16384 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<16384, 5120>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 193-3072) | 2048 | 184.79 | 76% | 92.5 | 38% |
| 16384 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<16384, 5120>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 3, 1, Fp8MmaRaster::Grouped, 8>, kGb10SplitWaveCtas, 4, 8>>` (T>=3073) | — | — | — | — | — |
| 34816 | 5120 | a16 | `fp8_linear_a16_simt<Fp8Geometry<34816, 5120>, 4, Fp8A16SimtSchedule<4, 2, 16, 4, 1, Fp8SimtActivationAccess::TokenPacked, Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>>` (T 1-4) | 1 | 0.47 | 0.4% | 236.4 | 98% |
| 34816 | 5120 | a16 | `fp8_linear_a16_simt<Fp8Geometry<34816, 5120>, 4, Fp8A16SimtSchedule<4, 2, 16, 4, 1, Fp8SimtActivationAccess::TokenPacked, Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>>` (T 1-4) | 2 | 0.95 | 0.8% | 236.6 | 98% |
| 34816 | 5120 | a16 | `fp8_linear_a16_simt<Fp8Geometry<34816, 5120>, 4, Fp8A16SimtSchedule<4, 2, 16, 4, 1, Fp8SimtActivationAccess::TokenPacked, Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>>` (T 1-4) | 3 | 1.40 | 1.1% | 234.1 | 97% |
| 34816 | 5120 | a16 | `fp8_linear_a16_simt<Fp8Geometry<34816, 5120>, 4, Fp8A16SimtSchedule<4, 2, 16, 4, 1, Fp8SimtActivationAccess::TokenPacked, Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>>` (T 1-4) | 4 | 1.87 | 1.5% | 233.9 | 97% |
| 34816 | 5120 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<34816, 5120>, Fp8SlicedInstance<16, 8, 2>>` (T 5-16) | 5 | 2.35 | 1.9% | 235.3 | 98% |
| 34816 | 5120 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<34816, 5120>, Fp8SlicedInstance<16, 8, 2>>` (T 5-16) | 6 | 2.82 | 2.3% | 235.4 | 98% |
| 34816 | 5120 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<34816, 5120>, Fp8SlicedInstance<16, 8, 2>>` (T 5-16) | 7 | 3.28 | 2.7% | 235.2 | 98% |
| 34816 | 5120 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<34816, 5120>, Fp8SlicedInstance<16, 8, 2>>` (T 5-16) | 8 | 3.75 | 3.0% | 235.3 | 98% |
| 34816 | 5120 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<34816, 5120>, Fp8SlicedInstance<16, 8, 2>>` (T 5-16) | 12 | 5.61 | 4.5% | 235.1 | 98% |
| 34816 | 5120 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<34816, 5120>, Fp8SlicedInstance<16, 8, 2>>` (T 5-16) | 16 | 7.46 | 6.0% | 234.9 | 98% |
| 34816 | 5120 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<34816, 5120>, Fp8SlicedInstance<32, 8, 2>>` (T 17-32) | 17 | 7.95 | 6.4% | 235.6 | 98% |
| 34816 | 5120 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<34816, 5120>, Fp8SlicedInstance<32, 8, 2>>` (T 17-32) | 24 | 11.17 | 9.0% | 235.4 | 98% |
| 34816 | 5120 | a16 | `fp8_linear_a16_sliced_k<Fp8Geometry<34816, 5120>, Fp8SlicedInstance<32, 8, 2>>` (T 17-32) | 32 | 14.80 | 12% | 234.7 | 98% |
| 34816 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<34816, 5120>, Fp8A16TmaT32R64>` (T 33-96) | 33 | 14.44 | 12% | 222.1 | 92% |
| 34816 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<34816, 5120>, Fp8A16TmaT32R64>` (T 33-96) | 48 | 20.97 | 17% | 223.2 | 93% |
| 34816 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<34816, 5120>, Fp8A16TmaT32R64>` (T 33-96) | 64 | 27.54 | 22% | 221.4 | 92% |
| 34816 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<34816, 5120>, Fp8A16TmaT32R64>` (T 33-96) | 96 | 40.77 | 33% | 221.6 | 92% |
| 34816 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<34816, 5120>, Fp8A16TmaT32R64S3>` (T 97-128) | 97 | 40.51 | 33% | 218.0 | 91% |
| 34816 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<34816, 5120>, Fp8A16TmaT32R64S3>` (T 97-128) | 128 | 53.19 | 43% | 219.8 | 91% |
| 34816 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<34816, 5120>, Fp8A16TmaT64R128>` (T 129-192) | 129 | 52.91 | 43% | 217.0 | 90% |
| 34816 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<34816, 5120>, Fp8A16TmaT64R128>` (T 129-192) | 192 | 76.50 | 62% | 216.4 | 90% |
| 34816 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<34816, 5120>, Fp8A16TmaMmaSchedule<64, 256, 2, 8, 2, 1>>` (T 193-256) | 193 | 69.28 | 56% | 195.1 | 81% |
| 34816 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<34816, 5120>, Fp8A16TmaMmaSchedule<64, 256, 2, 8, 2, 1>>` (T 193-256) | 256 | 90.59 | 73% | 197.3 | 82% |
| 34816 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<34816, 5120>, Fp8A16TmaT96R128S3K64>` (T 257-288) | 257 | 77.95 | 63% | 169.2 | 70% |
| 34816 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<34816, 5120>, Fp8A16TmaT96R128S3K64>` (T 257-288) | 288 | 87.20 | 70% | 171.0 | 71% |
| 34816 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<34816, 5120>, Fp8A16TmaT128R128S2B2K64>` (T 289-2560) | 289 | 79.11 | 64% | 154.6 | 64% |
| 34816 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<34816, 5120>, Fp8A16TmaT128R128S2B2K64>` (T 289-2560) | 384 | 103.17 | 83% | 157.5 | 65% |
| 34816 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<34816, 5120>, Fp8A16TmaT128R128S2B2K64>` (T 289-2560) | 512 | 103.65 | 84% | 124.5 | 52% |
| 34816 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<34816, 5120>, Fp8A16TmaT128R128S2B2K64>` (T 289-2560) | 640 | 102.07 | 83% | 102.6 | 43% |
| 34816 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<34816, 5120>, Fp8A16TmaT128R128S2B2K64>` (T 289-2560) | 768 | 104.29 | 84% | 91.3 | 38% |
| 34816 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<34816, 5120>, Fp8A16TmaT128R128S2B2K64>` (T 289-2560) | 896 | 103.85 | 84% | 81.2 | 34% |
| 34816 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<34816, 5120>, Fp8A16TmaT128R128S2B2K64>` (T 289-2560) | 1024 | 98.06 | 79% | 69.9 | 29% |
| 34816 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<34816, 5120>, Fp8A16TmaT128R128S2B2K64>` (T 289-2560) | 1152 | 99.40 | 80% | 65.4 | 27% |
| 34816 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<34816, 5120>, Fp8A16TmaT128R128S2B2K64>` (T 289-2560) | 1280 | 98.66 | 80% | 60.7 | 25% |
| 34816 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<34816, 5120>, Fp8A16TmaT128R128S2B2K64>` (T 289-2560) | 1408 | 98.28 | 79% | 56.9 | 24% |
| 34816 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<34816, 5120>, Fp8A16TmaT128R128S2B2K64>` (T 289-2560) | 1536 | 98.65 | 80% | 54.2 | 23% |
| 34816 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<34816, 5120>, Fp8A16TmaT128R128S2B2K64>` (T 289-2560) | 1664 | 98.26 | 79% | 51.6 | 21% |
| 34816 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<34816, 5120>, Fp8A16TmaT128R128S2B2K64>` (T 289-2560) | 1792 | 96.67 | 78% | 48.6 | 20% |
| 34816 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<34816, 5120>, Fp8A16TmaT128R128S2B2K64>` (T 289-2560) | 1920 | 96.53 | 78% | 46.8 | 19% |
| 34816 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<34816, 5120>, Fp8A16TmaT128R128S2B2K64>` (T 289-2560) | 2048 | 95.30 | 77% | 44.6 | 19% |
| 34816 | 5120 | a16 | `fp8_linear_a16_tma<Fp8Geometry<34816, 5120>, Fp8A16TmaT64R256G8>` (T>=2561) | — | — | — | — | — |
| 34816 | 5120 | a8 | `launch_fp8_a8<Fp8Geometry<34816, 5120>, Fp8A8T64R128K256>` (T 1-64) | 1 | 0.47 | 0.2% | 236.8 | 98% |
| 34816 | 5120 | a8 | `launch_fp8_a8<Fp8Geometry<34816, 5120>, Fp8A8T64R128K256>` (T 1-64) | 2 | 0.93 | 0.4% | 233.0 | 97% |
| 34816 | 5120 | a8 | `launch_fp8_a8<Fp8Geometry<34816, 5120>, Fp8A8T64R128K256>` (T 1-64) | 3 | 1.40 | 0.6% | 233.5 | 97% |
| 34816 | 5120 | a8 | `launch_fp8_a8<Fp8Geometry<34816, 5120>, Fp8A8T64R128K256>` (T 1-64) | 4 | 1.87 | 0.8% | 233.6 | 97% |
| 34816 | 5120 | a8 | `launch_fp8_a8<Fp8Geometry<34816, 5120>, Fp8A8T64R128K256>` (T 1-64) | 5 | 2.31 | 0.9% | 231.2 | 96% |
| 34816 | 5120 | a8 | `launch_fp8_a8<Fp8Geometry<34816, 5120>, Fp8A8T64R128K256>` (T 1-64) | 6 | 2.76 | 1.1% | 231.0 | 96% |
| 34816 | 5120 | a8 | `launch_fp8_a8<Fp8Geometry<34816, 5120>, Fp8A8T64R128K256>` (T 1-64) | 7 | 3.22 | 1.3% | 230.5 | 96% |
| 34816 | 5120 | a8 | `launch_fp8_a8<Fp8Geometry<34816, 5120>, Fp8A8T64R128K256>` (T 1-64) | 8 | 3.67 | 1.5% | 230.4 | 96% |
| 34816 | 5120 | a8 | `launch_fp8_a8<Fp8Geometry<34816, 5120>, Fp8A8T64R128K256>` (T 1-64) | 12 | 5.51 | 2.3% | 230.7 | 96% |
| 34816 | 5120 | a8 | `launch_fp8_a8<Fp8Geometry<34816, 5120>, Fp8A8T64R128K256>` (T 1-64) | 16 | 7.29 | 3.0% | 229.6 | 95% |
| 34816 | 5120 | a8 | `launch_fp8_a8<Fp8Geometry<34816, 5120>, Fp8A8T64R128K256>` (T 1-64) | 24 | 10.88 | 4.5% | 229.2 | 95% |
| 34816 | 5120 | a8 | `launch_fp8_a8<Fp8Geometry<34816, 5120>, Fp8A8T64R128K256>` (T 1-64) | 32 | 14.41 | 5.9% | 228.5 | 95% |
| 34816 | 5120 | a8 | `launch_fp8_a8<Fp8Geometry<34816, 5120>, Fp8A8T64R128K256>` (T 1-64) | 48 | 21.60 | 8.9% | 229.9 | 96% |
| 34816 | 5120 | a8 | `launch_fp8_a8<Fp8Geometry<34816, 5120>, Fp8A8T64R128K256>` (T 1-64) | 64 | 28.50 | 12% | 229.1 | 95% |
| 34816 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<34816, 5120>, Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 2, 1>>` (T 65-128) | 65 | 27.81 | 11% | 220.2 | 92% |
| 34816 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<34816, 5120>, Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 2, 1>>` (T 65-128) | 128 | 52.75 | 22% | 217.9 | 91% |
| 34816 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<34816, 5120>, Fp8A8TmaMmaSchedule<64, 256, 128, 2, 4, 2, 1>>` (T 129-256) | 129 | 53.22 | 22% | 218.3 | 91% |
| 34816 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<34816, 5120>, Fp8A8TmaMmaSchedule<64, 256, 128, 2, 4, 2, 1>>` (T 129-256) | 256 | 98.83 | 40% | 215.2 | 89% |
| 34816 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<34816, 5120>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 257-3072) | 257 | 95.71 | 39% | 207.7 | 86% |
| 34816 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<34816, 5120>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 257-3072) | 384 | 132.79 | 54% | 202.7 | 84% |
| 34816 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<34816, 5120>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 257-3072) | 512 | 164.50 | 67% | 197.6 | 82% |
| 34816 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<34816, 5120>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 257-3072) | 640 | 186.49 | 76% | 187.5 | 78% |
| 34816 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<34816, 5120>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 257-3072) | 768 | 197.22 | 81% | 172.6 | 72% |
| 34816 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<34816, 5120>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 257-3072) | 896 | 192.46 | 79% | 150.6 | 63% |
| 34816 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<34816, 5120>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 257-3072) | 1024 | 191.28 | 78% | 136.3 | 57% |
| 34816 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<34816, 5120>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 257-3072) | 1152 | 194.15 | 80% | 127.8 | 53% |
| 34816 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<34816, 5120>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 257-3072) | 1280 | 194.20 | 80% | 119.4 | 50% |
| 34816 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<34816, 5120>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 257-3072) | 1408 | 190.24 | 78% | 110.2 | 46% |
| 34816 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<34816, 5120>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 257-3072) | 1536 | 192.03 | 79% | 105.6 | 44% |
| 34816 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<34816, 5120>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 257-3072) | 1664 | 187.31 | 77% | 98.3 | 41% |
| 34816 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<34816, 5120>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 257-3072) | 1792 | 186.36 | 76% | 93.8 | 39% |
| 34816 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<34816, 5120>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 257-3072) | 1920 | 187.89 | 77% | 91.0 | 38% |
| 34816 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<34816, 5120>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>>` (T 257-3072) | 2048 | 186.62 | 76% | 87.4 | 36% |
| 34816 | 5120 | a8 | `launch_fp8_a8_tma<Fp8Geometry<34816, 5120>, Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 3, 1, Fp8MmaRaster::Grouped, 8>, kGb10SplitWaveCtas, 4, 8>>` (T>=3073) | — | — | — | — | — |
| 248320 | 3072 | a16 | `launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Fp8A16SlicedKMmaSchedule<16, 8, 1, Cache::ca, Cache::cg, Fp8ActivationStage::ActiveOnly, 1, 1>, 3072, 8>>` (T 1-8) | 1 | 0.48 | 0.4% | 238.7 | 99% |
| 248320 | 3072 | a16 | `launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Fp8A16SlicedKMmaSchedule<16, 8, 1, Cache::ca, Cache::cg, Fp8ActivationStage::ActiveOnly, 1, 1>, 3072, 8>>` (T 1-8) | 2 | 0.95 | 0.8% | 237.9 | 99% |
| 248320 | 3072 | a16 | `launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Fp8A16SlicedKMmaSchedule<16, 8, 1, Cache::ca, Cache::cg, Fp8ActivationStage::ActiveOnly, 1, 1>, 3072, 8>>` (T 1-8) | 3 | 1.42 | 1.1% | 238.2 | 99% |
| 248320 | 3072 | a16 | `launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Fp8A16SlicedKMmaSchedule<16, 8, 1, Cache::ca, Cache::cg, Fp8ActivationStage::ActiveOnly, 1, 1>, 3072, 8>>` (T 1-8) | 4 | 1.90 | 1.5% | 238.0 | 99% |
| 248320 | 3072 | a16 | `launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Fp8A16SlicedKMmaSchedule<16, 8, 1, Cache::ca, Cache::cg, Fp8ActivationStage::ActiveOnly, 1, 1>, 3072, 8>>` (T 1-8) | 5 | 2.36 | 1.9% | 237.3 | 99% |
| 248320 | 3072 | a16 | `launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Fp8A16SlicedKMmaSchedule<16, 8, 1, Cache::ca, Cache::cg, Fp8ActivationStage::ActiveOnly, 1, 1>, 3072, 8>>` (T 1-8) | 6 | 2.84 | 2.3% | 237.6 | 99% |
| 248320 | 3072 | a16 | `launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Fp8A16SlicedKMmaSchedule<16, 8, 1, Cache::ca, Cache::cg, Fp8ActivationStage::ActiveOnly, 1, 1>, 3072, 8>>` (T 1-8) | 7 | 3.30 | 2.7% | 237.3 | 99% |
| 248320 | 3072 | a16 | `launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Fp8A16SlicedKMmaSchedule<16, 8, 1, Cache::ca, Cache::cg, Fp8ActivationStage::ActiveOnly, 1, 1>, 3072, 8>>` (T 1-8) | 8 | 3.77 | 3.0% | 237.4 | 99% |
| 248320 | 3072 | a16 | `launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Fp8A16SlicedKMmaSchedule<8, 16, 2, Cache::ca, Cache::cg, Fp8ActivationStage::ActiveOnly, 1, 1>, 3072, 16>>` (T 9-16) | 9 | 4.27 | 3.5% | 238.9 | 99% |
| 248320 | 3072 | a16 | `launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Fp8A16SlicedKMmaSchedule<8, 16, 2, Cache::ca, Cache::cg, Fp8ActivationStage::ActiveOnly, 1, 1>, 3072, 16>>` (T 9-16) | 12 | 5.68 | 4.6% | 238.8 | 99% |
| 248320 | 3072 | a16 | `launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Fp8A16SlicedKMmaSchedule<8, 16, 2, Cache::ca, Cache::cg, Fp8ActivationStage::ActiveOnly, 1, 1>, 3072, 16>>` (T 9-16) | 16 | 7.56 | 6.1% | 239.2 | 99% |
| 248320 | 3072 | a16 | `launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Fp8A16SlicedKMmaSchedule<8, 24, 2, Cache::ca, Cache::cg, Fp8ActivationStage::ActiveOnly, 1, 2>, 3072, 24>>` (T 17-24) | 17 | 7.96 | 6.4% | 237.0 | 99% |
| 248320 | 3072 | a16 | `launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Fp8A16SlicedKMmaSchedule<8, 24, 2, Cache::ca, Cache::cg, Fp8ActivationStage::ActiveOnly, 1, 2>, 3072, 24>>` (T 17-24) | 24 | 11.15 | 9.0% | 236.3 | 98% |
| 248320 | 3072 | a16 | `launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Fp8A16SlicedKMmaSchedule<4, 32, 2, Cache::ca, Cache::cg, Fp8ActivationStage::ActiveOnly, 1, 2>, 3072, 32>>` (T 25-32) | 25 | 11.32 | 9.2% | 230.4 | 96% |
| 248320 | 3072 | a16 | `launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Fp8A16SlicedKMmaSchedule<4, 32, 2, Cache::ca, Cache::cg, Fp8ActivationStage::ActiveOnly, 1, 2>, 3072, 32>>` (T 25-32) | 32 | 14.39 | 12% | 229.9 | 96% |
| 248320 | 3072 | a16 | `launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Fp8A16SlicedKMmaSchedule<4, 40, 2, Cache::ca, Cache::cg, Fp8ActivationStage::ActiveOnly, 1, 2>, 3072, 40>>` (T 33-40) | 33 | 14.80 | 12% | 229.4 | 95% |
| 248320 | 3072 | a16 | `launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Fp8A16SlicedKMmaSchedule<4, 40, 2, Cache::ca, Cache::cg, Fp8ActivationStage::ActiveOnly, 1, 2>, 3072, 40>>` (T 33-40) | 40 | 17.92 | 14% | 230.2 | 96% |
| 248320 | 3072 | a16 | `launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Fp8A16SlicedKMmaSchedule<4, 48, 2, Cache::ca, Cache::cg, Fp8ActivationStage::ActiveOnly, 1, 2>, 3072, 48>>` (T 41-48) | 41 | 18.33 | 15% | 229.9 | 96% |
| 248320 | 3072 | a16 | `launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Fp8A16SlicedKMmaSchedule<4, 48, 2, Cache::ca, Cache::cg, Fp8ActivationStage::ActiveOnly, 1, 2>, 3072, 48>>` (T 41-48) | 48 | 21.43 | 17% | 230.6 | 96% |
| 248320 | 3072 | a16 | `launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Fp8A16SlicedKMmaSchedule<2, 56, 2, Cache::ca, Cache::cg, Fp8ActivationStage::ActiveOnly, 2, 2>, 3072, 56>>` (T 49-56) | 49 | 21.21 | 17% | 223.6 | 93% |
| 248320 | 3072 | a16 | `launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Fp8A16SlicedKMmaSchedule<2, 56, 2, Cache::ca, Cache::cg, Fp8ActivationStage::ActiveOnly, 2, 2>, 3072, 56>>` (T 49-56) | 56 | 23.98 | 19% | 222.3 | 92% |
| 248320 | 3072 | a16 | `launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Fp8A16SlicedKMmaSchedule<2, 64, 2, Cache::ca, Cache::cg, Fp8ActivationStage::ActiveOnly, 2, 2>, 3072, 64>>` (T 57-64) | 57 | 24.35 | 20% | 221.9 | 92% |
| 248320 | 3072 | a16 | `launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Fp8A16SlicedKMmaSchedule<2, 64, 2, Cache::ca, Cache::cg, Fp8ActivationStage::ActiveOnly, 2, 2>, 3072, 64>>` (T 57-64) | 64 | 27.18 | 22% | 221.6 | 92% |
| 248320 | 3072 | a16 | `launch_fp8_a16_mma<Fp8ScheduleInstance<Fp8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>, 3072>>` (T 65-96) | 65 | 25.35 | 20% | 203.6 | 85% |
| 248320 | 3072 | a16 | `launch_fp8_a16_mma<Fp8ScheduleInstance<Fp8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>, 3072>>` (T 65-96) | 96 | 36.90 | 30% | 204.6 | 85% |
| 248320 | 3072 | a16 | `launch_fp8_a16_mma<Fp8ScheduleInstance<Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>, 3072>>` (T 97-128) | 97 | 37.75 | 31% | 207.3 | 86% |
| 248320 | 3072 | a16 | `launch_fp8_a16_mma<Fp8ScheduleInstance<Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>, 3072>>` (T 97-128) | 128 | 48.82 | 39% | 207.0 | 86% |
| 248320 | 3072 | a16 | `launch_schedule<Fp8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>> + launch_schedule<Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>> + launch_ksplit` (T>=129) | 129 | 26.65 | 22% | 112.2 | 47% |
| 248320 | 3072 | a16 | `launch_schedule<Fp8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>> + launch_schedule<Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>> + launch_ksplit` (T>=129) | 256 | 71.31 | 58% | 163.0 | 68% |
| 248320 | 3072 | a16 | `launch_schedule<Fp8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>> + launch_schedule<Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>> + launch_ksplit` (T>=129) | 384 | 74.99 | 61% | 122.5 | 51% |
| 248320 | 3072 | a16 | `launch_schedule<Fp8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>> + launch_schedule<Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>> + launch_ksplit` (T>=129) | 512 | 70.92 | 57% | 92.7 | 39% |
| 248320 | 3072 | a16 | `launch_schedule<Fp8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>> + launch_schedule<Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>> + launch_ksplit` (T>=129) | 640 | 71.21 | 58% | 79.2 | 33% |
| 248320 | 3072 | a16 | `launch_schedule<Fp8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>> + launch_schedule<Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>> + launch_ksplit` (T>=129) | 768 | 71.38 | 58% | 70.1 | 29% |
| 248320 | 3072 | a16 | `launch_schedule<Fp8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>> + launch_schedule<Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>> + launch_ksplit` (T>=129) | 896 | 71.20 | 58% | 63.3 | 26% |
| 248320 | 3072 | a16 | `launch_schedule<Fp8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>> + launch_schedule<Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>> + launch_ksplit` (T>=129) | 1024 | 71.45 | 58% | 58.5 | 24% |
| 248320 | 3072 | a16 | `launch_schedule<Fp8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>> + launch_schedule<Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>> + launch_ksplit` (T>=129) | 1152 | 71.95 | 58% | 55.0 | 23% |
| 248320 | 3072 | a16 | `launch_schedule<Fp8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>> + launch_schedule<Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>> + launch_ksplit` (T>=129) | 1280 | 71.45 | 58% | 51.5 | 21% |
| 248320 | 3072 | a16 | `launch_schedule<Fp8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>> + launch_schedule<Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>> + launch_ksplit` (T>=129) | 1408 | 71.95 | 58% | 49.3 | 20% |
| 248320 | 3072 | a16 | `launch_schedule<Fp8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>> + launch_schedule<Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>> + launch_ksplit` (T>=129) | 1536 | 71.62 | 58% | 46.9 | 19% |
| 248320 | 3072 | a16 | `launch_schedule<Fp8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>> + launch_schedule<Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>> + launch_ksplit` (T>=129) | 1664 | 71.47 | 58% | 45.1 | 19% |
| 248320 | 3072 | a16 | `launch_schedule<Fp8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>> + launch_schedule<Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>> + launch_ksplit` (T>=129) | 1792 | 71.13 | 57% | 43.3 | 18% |
| 248320 | 3072 | a16 | `launch_schedule<Fp8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>> + launch_schedule<Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>> + launch_ksplit` (T>=129) | 1920 | 71.16 | 58% | 42.0 | 17% |
| 248320 | 3072 | a16 | `launch_schedule<Fp8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>> + launch_schedule<Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>> + launch_ksplit` (T>=129) | 2048 | 70.86 | 57% | 40.7 | 17% |
| 248320 | 5120 | a16 | `launch_schedule<Fp8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>>` (T>=1) | 1 | 0.47 | 0.4% | 234.8 | 98% |
| 248320 | 5120 | a16 | `launch_schedule<Fp8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>>` (T>=1) | 2 | 0.95 | 0.8% | 236.9 | 98% |
| 248320 | 5120 | a16 | `launch_schedule<Fp8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>>` (T>=1) | 3 | 1.43 | 1.2% | 238.1 | 99% |
| 248320 | 5120 | a16 | `launch_schedule<Fp8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>>` (T>=1) | 4 | 1.90 | 1.5% | 238.0 | 99% |
| 248320 | 5120 | a16 | `launch_schedule<Fp8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>>` (T>=1) | 5 | 2.37 | 1.9% | 237.8 | 99% |
| 248320 | 5120 | a16 | `launch_schedule<Fp8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>>` (T>=1) | 6 | 2.84 | 2.3% | 237.6 | 99% |
| 248320 | 5120 | a16 | `launch_schedule<Fp8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>>` (T>=1) | 7 | 3.31 | 2.7% | 237.5 | 99% |
| 248320 | 5120 | a16 | `launch_schedule<Fp8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>>` (T>=1) | 8 | 3.78 | 3.1% | 236.9 | 98% |
| 248320 | 5120 | a16 | `launch_schedule<Fp8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>>` (T>=1) | 12 | 5.62 | 4.5% | 235.3 | 98% |
| 248320 | 5120 | a16 | `launch_schedule<Fp8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>>` (T>=1) | 16 | 7.54 | 6.1% | 237.3 | 99% |
| 248320 | 5120 | a16 | `launch_schedule<Fp8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>>` (T>=1) | 24 | 11.06 | 8.9% | 232.6 | 97% |
| 248320 | 5120 | a16 | `launch_schedule<Fp8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>>` (T>=1) | 32 | 14.42 | 12% | 228.4 | 95% |
| 248320 | 5120 | a16 | `launch_schedule<Fp8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>>` (T>=1) | 48 | 21.11 | 17% | 224.2 | 93% |
| 248320 | 5120 | a16 | `launch_schedule<Fp8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>>` (T>=1) | 64 | 28.21 | 23% | 226.1 | 94% |
| 248320 | 5120 | a16 | `launch_schedule<Fp8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>>` (T>=1) | 128 | 45.47 | 37% | 186.8 | 78% |
| 248320 | 5120 | a16 | `launch_schedule<Fp8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>>` (T>=1) | 256 | 67.29 | 54% | 144.9 | 60% |
| 248320 | 5120 | a16 | `launch_schedule<Fp8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>>` (T>=1) | 384 | 70.55 | 57% | 106.0 | 44% |
| 248320 | 5120 | a16 | `launch_schedule<Fp8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>>` (T>=1) | 512 | 70.98 | 57% | 83.5 | 35% |
| 248320 | 5120 | a16 | `launch_schedule<Fp8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>>` (T>=1) | 640 | 71.27 | 58% | 69.9 | 29% |
| 248320 | 5120 | a16 | `launch_schedule<Fp8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>>` (T>=1) | 768 | 71.38 | 58% | 60.7 | 25% |
| 248320 | 5120 | a16 | `launch_schedule<Fp8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>>` (T>=1) | 896 | 71.27 | 58% | 54.0 | 22% |
| 248320 | 5120 | a16 | `launch_schedule<Fp8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>>` (T>=1) | 1024 | 71.16 | 58% | 48.9 | 20% |
| 248320 | 5120 | a16 | `launch_schedule<Fp8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>>` (T>=1) | 1152 | 70.61 | 57% | 44.7 | 19% |
| 248320 | 5120 | a16 | `launch_schedule<Fp8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>>` (T>=1) | 1280 | 70.83 | 57% | 41.8 | 17% |
| 248320 | 5120 | a16 | `launch_schedule<Fp8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>>` (T>=1) | 1408 | 70.30 | 57% | 39.0 | 16% |
| 248320 | 5120 | a16 | `launch_schedule<Fp8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>>` (T>=1) | 1536 | 69.99 | 57% | 36.7 | 15% |
| 248320 | 5120 | a16 | `launch_schedule<Fp8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>>` (T>=1) | 1664 | 69.52 | 56% | 34.8 | 14% |
| 248320 | 5120 | a16 | `launch_schedule<Fp8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>>` (T>=1) | 1792 | 69.51 | 56% | 33.3 | 14% |
| 248320 | 5120 | a16 | `launch_schedule<Fp8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>>` (T>=1) | 1920 | 69.10 | 56% | 31.8 | 13% |
| 248320 | 5120 | a16 | `launch_schedule<Fp8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>>` (T>=1) | 2048 | 68.71 | 56% | 30.5 | 13% |
