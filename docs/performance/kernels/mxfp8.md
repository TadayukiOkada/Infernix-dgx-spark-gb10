# Linear kernels — MXFP8 (GB10)

Part of the [Linear kernel tables](README.md).

Measured 2026-11-02 at `f2a495e3c` (dev-rebased-20261028, 3 uncommitted file(s) (source edits: `bench/README.md`, `bench/ops/nvfp4_route_sweep.cu`, `docs/README.md`)) with `build-native/bench/infernix_mxfp8_flash_next_sweep --shape <label> --t-list <widths> --warmup 3 --repeat 12 --csv <path>`: cold-L2 CUDA-graph timing with the rig's default eviction buffer. Shape labels: `shgu` [1280, 2560], `shdown` [2560, 640], `out6144` [2560, 6144], `qsa` [13952, 2560], `gdnqkvz` [16384, 2560]. `infernix_linear_bench` has no MXFP8 `--qtype`, so this family uses the schedule sweep rig; the `kernel` cell is the schedule the dispatch table selects at that `T`, matched to the candidate that runs it.

One row per token count `T`. `kernel` is the schedule the dispatch table in `src/ops/linear/mxfp8/shapes/` selects at that `T`, with its dispatch band in parentheses: the shape file's own launch call, with its `using` aliases and `constexpr` constants substituted and literal ternaries folded where the result fits the cell (a parameterized alias such as `Simt<2>`, or one whose expansion is too long, stays as the source writes it). `TFLOP/s` = 2·N·K·T / median; `GB/s` = (weight + activation + output bytes) / median.

`%pipe` and `%mem` put each row against the envelope that governs it, so a shortfall is attributable rather than a gap to a single device number. `%pipe` divides by the measured MMA-pipe peak of the pipe that policy actually runs on (`a16` → BF16 HMMA 123.71 TFLOP/s); the packed weight format is not the ceiling — an A16 route dequantizes its stored weights and lands on the BF16 HMMA pipe whatever the codec. `%mem` divides by 240.6 GB/s, the `ckt env` measured read-only ceiling. A row is at its envelope when the percentage of the envelope that binds it — `mem` at decode widths, `pipe` at prefill widths — is near 100.

Two regimes keep both percentages far below 100 and no schedule closes them. First, a payload smaller than the bandwidth × memory-latency product (~295 KB here) cannot reach the read ceiling at all: the whole tensor must be in flight before the stream ends. Second, the timed window is the whole public Op, so a route that composes several launches pays every one of them.

Repeatability: each row is the better median of two independent cold-L2 passes. Across these 115 rows the passes differ by 0.86% (median), 3.27% (p90) and 18.94% (worst), so a single pass is not a repeatable number for the microsecond rows. Every pass ran with a quiet host: host-side memory work (a build, a `cuobjdump` of a large binary) shares GB10's LPDDR5X with the GPU and inflates measured kernel times.

Not covered: any band above the measured cap, and the two alternate route entries the shape files register — `launch_f16` (`LinearRoutes::F16Accumulate`) and `launch_pair_block32` (`LinearRoutes::Fp8A8PairBlock32`, admitted only above each shape's minimum token count). The sweep rig has no entry point for either, so neither is timed; the `f16acc` rows with empty metrics are the f16-accumulate ladder. On GB10 `FastPath::F16AccumulatePrefill` is deliberately left unqualified (1:1 FP32 Tensor Core accumulation parity), so those bands are not the production prefill route.

| N | K | policy | kernel | T | TFLOP/s | %pipe | GB/s | %mem |
|---:|---:|---|---|---:|---:|---:|---:|---:|
| 1280 | 2560 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<1280, 2560>, Mxfp8P2<Mxfp8TmaT16R32>>` (T 1-48) | 1 | 0.32 | 0.3% | 166.4 | 69% |
| 1280 | 2560 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<1280, 2560>, Mxfp8P2<Mxfp8TmaT16R32>>` (T 1-48) | 2 | 0.65 | 0.5% | 168.4 | 70% |
| 1280 | 2560 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<1280, 2560>, Mxfp8P2<Mxfp8TmaT16R32>>` (T 1-48) | 3 | 0.98 | 0.8% | 168.8 | 70% |
| 1280 | 2560 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<1280, 2560>, Mxfp8P2<Mxfp8TmaT16R32>>` (T 1-48) | 4 | 1.30 | 1.1% | 169.1 | 70% |
| 1280 | 2560 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<1280, 2560>, Mxfp8P2<Mxfp8TmaT16R32>>` (T 1-48) | 5 | 1.63 | 1.3% | 169.5 | 70% |
| 1280 | 2560 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<1280, 2560>, Mxfp8P2<Mxfp8TmaT16R32>>` (T 1-48) | 6 | 1.95 | 1.6% | 169.9 | 71% |
| 1280 | 2560 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<1280, 2560>, Mxfp8P2<Mxfp8TmaT16R32>>` (T 1-48) | 7 | 2.28 | 1.8% | 170.3 | 71% |
| 1280 | 2560 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<1280, 2560>, Mxfp8P2<Mxfp8TmaT16R32>>` (T 1-48) | 8 | 2.58 | 2.1% | 169.6 | 70% |
| 1280 | 2560 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<1280, 2560>, Mxfp8P2<Mxfp8TmaT16R32>>` (T 1-48) | 16 | 5.20 | 4.2% | 173.7 | 72% |
| 1280 | 2560 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<1280, 2560>, Mxfp8P2<Mxfp8TmaT16R32>>` (T 1-48) | 24 | 7.79 | 6.3% | 176.5 | 73% |
| 1280 | 2560 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<1280, 2560>, Mxfp8P2<Mxfp8TmaT16R32>>` (T 1-48) | 32 | 10.40 | 8.4% | 179.8 | 75% |
| 1280 | 2560 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<1280, 2560>, Mxfp8P2<Mxfp8TmaT16R32>>` (T 1-48) | 48 | 14.16 | 11% | 168.8 | 70% |
| 1280 | 2560 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<1280, 2560>, Mxfp8P2<Mxfp8TmaT32R32>>` (T 49-64) | 64 | 18.86 | 15% | 174.0 | 72% |
| 1280 | 2560 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<1280, 2560>, Mxfp8Tma<Fp8A16RingT32R64S3>>` (T 65-128) | 96 | 25.94 | 21% | 169.7 | 71% |
| 1280 | 2560 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<1280, 2560>, Mxfp8Tma<Fp8A16RingT32R64S3>>` (T 65-128) | 128 | 36.06 | 29% | 187.5 | 78% |
| 1280 | 2560 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<1280, 2560>, Mxfp8Tma<Fp8A16RingT64R128>>` (T 129-256) | 192 | 44.18 | 36% | 170.4 | 71% |
| 1280 | 2560 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<1280, 2560>, Mxfp8Tma<Fp8A16RingT64R128>>` (T 129-256) | 256 | 58.84 | 48% | 187.5 | 78% |
| 1280 | 2560 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<1280, 2560>, Mxfp8Tma<Mxfp8TmaT128R128>>` (T 257-512) | 384 | 53.14 | 43% | 133.6 | 56% |
| 1280 | 2560 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<1280, 2560>, Mxfp8Tma<Mxfp8TmaT128R128>>` (T 257-512) | 512 | 68.49 | 55% | 149.2 | 62% |
| 1280 | 2560 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<1280, 2560>, Mxfp8Tma<Mxfp8TmaT64R64>>` (T 513-768) | 768 | 66.62 | 54% | 122.8 | 51% |
| 1280 | 2560 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<1280, 2560>, Mxfp8Tma<Mxfp8TmaT128R128>>` (T 769-2048) | 1024 | 68.38 | 55% | 114.6 | 48% |
| 1280 | 2560 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<1280, 2560>, Mxfp8Tma<Mxfp8TmaT128R128>>` (T 769-2048) | 1536 | 69.38 | 56% | 104.6 | 43% |
| 1280 | 2560 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<1280, 2560>, Mxfp8Tma<Mxfp8TmaT128R128>>` (T 769-2048) | 2048 | 68.73 | 56% | 97.8 | 41% |
| 1280 | 2560 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<1280, 2560>, Mxfp8Tma<Mxfp8TmaT64R256>>` (T>=2049) | — | — | — | — | — |
| 2560 | 640 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<2560, 640>, Mxfp8TmaT32R32>` (T 1-6) | 1 | 0.30 | 0.2% | 154.5 | 64% |
| 2560 | 640 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<2560, 640>, Mxfp8TmaT32R32>` (T 1-6) | 2 | 0.59 | 0.5% | 153.3 | 64% |
| 2560 | 640 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<2560, 640>, Mxfp8TmaT32R32>` (T 1-6) | 3 | 0.88 | 0.7% | 153.4 | 64% |
| 2560 | 640 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<2560, 640>, Mxfp8TmaT32R32>` (T 1-6) | 4 | 1.18 | 1.0% | 154.5 | 64% |
| 2560 | 640 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<2560, 640>, Mxfp8TmaT32R32>` (T 1-6) | 5 | 1.46 | 1.2% | 153.7 | 64% |
| 2560 | 640 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<2560, 640>, Mxfp8TmaT32R32>` (T 1-6) | 6 | 1.51 | 1.2% | 132.7 | 55% |
| 2560 | 640 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<2560, 640>, Mxfp8Tma<Mxfp8TmaT16R32>>` (T 7-16) | 7 | 1.92 | 1.6% | 144.9 | 60% |
| 2560 | 640 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<2560, 640>, Mxfp8Tma<Mxfp8TmaT16R32>>` (T 7-16) | 8 | 2.19 | 1.8% | 145.5 | 60% |
| 2560 | 640 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<2560, 640>, Mxfp8Tma<Mxfp8TmaT16R32>>` (T 7-16) | 16 | 4.37 | 3.5% | 149.3 | 62% |
| 2560 | 640 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<2560, 640>, Mxfp8Tma<Fp8A16RingMmaSchedule<32, 32, 1, 4, 3, 2>>>` (T 17-32) | 24 | 6.55 | 5.3% | 153.6 | 64% |
| 2560 | 640 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<2560, 640>, Mxfp8Tma<Fp8A16RingMmaSchedule<32, 32, 1, 4, 3, 2>>>` (T 17-32) | 32 | 8.58 | 6.9% | 155.0 | 64% |
| 2560 | 640 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<2560, 640>, Mxfp8P2<Fp8A16RingT32R64S3>>` (T 33-64) | 48 | 12.83 | 10% | 162.9 | 68% |
| 2560 | 640 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<2560, 640>, Mxfp8P2<Fp8A16RingT32R64S3>>` (T 33-64) | 64 | 16.98 | 14% | 169.9 | 71% |
| 2560 | 640 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<2560, 640>, Mxfp8Tma<Fp8A16RingT64R128>>` (T 65-96) | 96 | 21.65 | 18% | 158.6 | 66% |
| 2560 | 640 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<2560, 640>, Mxfp8P2<Mxfp8TmaT64R64>>` (T 97-256) | 128 | 29.65 | 24% | 177.4 | 74% |
| 2560 | 640 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<2560, 640>, Mxfp8P2<Mxfp8TmaT64R64>>` (T 97-256) | 192 | 34.68 | 28% | 160.8 | 67% |
| 2560 | 640 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<2560, 640>, Mxfp8P2<Mxfp8TmaT64R64>>` (T 97-256) | 256 | 41.54 | 34% | 164.8 | 68% |
| 2560 | 640 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<2560, 640>, Mxfp8Tma<Mxfp8TmaT64R64>>` (T 257-1024) | 384 | 50.74 | 41% | 167.2 | 69% |
| 2560 | 640 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<2560, 640>, Mxfp8Tma<Mxfp8TmaT64R64>>` (T 257-1024) | 512 | 51.70 | 42% | 153.1 | 64% |
| 2560 | 640 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<2560, 640>, Mxfp8Tma<Mxfp8TmaT64R64>>` (T 257-1024) | 768 | 56.05 | 45% | 147.1 | 61% |
| 2560 | 640 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<2560, 640>, Mxfp8Tma<Mxfp8TmaT64R64>>` (T 257-1024) | 1024 | 56.77 | 46% | 139.5 | 58% |
| 2560 | 640 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<2560, 640>, Mxfp8Tma<Mxfp8TmaT128R128>>` (T>=1025) | 1536 | 63.24 | 51% | 144.8 | 60% |
| 2560 | 640 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<2560, 640>, Mxfp8Tma<Mxfp8TmaT128R128>>` (T>=1025) | 2048 | 63.13 | 51% | 139.2 | 58% |
| 2560 | 6144 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<2560, 6144>, Mxfp8P2<Mxfp8TmaT16R64>>` (T 1-8) | 1 | 0.40 | 0.3% | 208.3 | 87% |
| 2560 | 6144 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<2560, 6144>, Mxfp8P2<Mxfp8TmaT16R64>>` (T 1-8) | 2 | 0.81 | 0.7% | 209.2 | 87% |
| 2560 | 6144 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<2560, 6144>, Mxfp8P2<Mxfp8TmaT16R64>>` (T 1-8) | 3 | 1.21 | 1.0% | 209.3 | 87% |
| 2560 | 6144 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<2560, 6144>, Mxfp8P2<Mxfp8TmaT16R64>>` (T 1-8) | 4 | 1.60 | 1.3% | 207.4 | 86% |
| 2560 | 6144 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<2560, 6144>, Mxfp8P2<Mxfp8TmaT16R64>>` (T 1-8) | 5 | 2.00 | 1.6% | 207.2 | 86% |
| 2560 | 6144 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<2560, 6144>, Mxfp8P2<Mxfp8TmaT16R64>>` (T 1-8) | 6 | 2.40 | 1.9% | 207.9 | 86% |
| 2560 | 6144 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<2560, 6144>, Mxfp8P2<Mxfp8TmaT16R64>>` (T 1-8) | 7 | 2.77 | 2.2% | 205.4 | 85% |
| 2560 | 6144 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<2560, 6144>, Mxfp8P2<Mxfp8TmaT16R64>>` (T 1-8) | 8 | 3.20 | 2.6% | 208.0 | 86% |
| 2560 | 6144 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<2560, 6144>, Mxfp8Tma<Mxfp8TmaT16R32>>` (T 9-32) | 16 | 6.32 | 5.1% | 207.1 | 86% |
| 2560 | 6144 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<2560, 6144>, Mxfp8Tma<Mxfp8TmaT16R32>>` (T 9-32) | 24 | 8.39 | 6.8% | 185.0 | 77% |
| 2560 | 6144 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<2560, 6144>, Mxfp8Tma<Mxfp8TmaT16R32>>` (T 9-32) | 32 | 10.96 | 8.9% | 182.6 | 76% |
| 2560 | 6144 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<2560, 6144>, Mxfp8Tma<Fp8A16RingT32R64S3>>` (T 33-64) | 48 | 18.25 | 15% | 206.2 | 86% |
| 2560 | 6144 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<2560, 6144>, Mxfp8Tma<Fp8A16RingT32R64S3>>` (T 33-64) | 64 | 24.35 | 20% | 209.6 | 87% |
| 2560 | 6144 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<2560, 6144>, Fp8A16RingT64R128>` (T 65-128) | 96 | 32.85 | 27% | 194.6 | 81% |
| 2560 | 6144 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<2560, 6144>, Fp8A16RingT64R128>` (T 65-128) | 128 | 42.81 | 35% | 196.2 | 82% |
| 2560 | 6144 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<2560, 6144>, Mxfp8Tma<Mxfp8TmaT64R64>>` (T 129-512) | 192 | 54.82 | 44% | 177.6 | 74% |
| 2560 | 6144 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<2560, 6144>, Mxfp8Tma<Mxfp8TmaT64R64>>` (T 129-512) | 256 | 62.55 | 51% | 160.6 | 67% |
| 2560 | 6144 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<2560, 6144>, Mxfp8Tma<Mxfp8TmaT64R64>>` (T 129-512) | 384 | 72.05 | 58% | 136.6 | 57% |
| 2560 | 6144 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<2560, 6144>, Mxfp8Tma<Mxfp8TmaT64R64>>` (T 129-512) | 512 | 71.54 | 58% | 111.6 | 46% |
| 2560 | 6144 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<2560, 6144>, Mxfp8Tma<Mxfp8TmaT128R128>>` (T 513-2048) | 768 | 74.70 | 60% | 91.5 | 38% |
| 2560 | 6144 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<2560, 6144>, Mxfp8Tma<Mxfp8TmaT128R128>>` (T 513-2048) | 1024 | 74.93 | 61% | 79.2 | 33% |
| 2560 | 6144 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<2560, 6144>, Mxfp8Tma<Mxfp8TmaT128R128>>` (T 513-2048) | 1536 | 85.76 | 69% | 76.2 | 32% |
| 2560 | 6144 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<2560, 6144>, Mxfp8Tma<Mxfp8TmaT128R128>>` (T 513-2048) | 2048 | 79.27 | 64% | 63.8 | 27% |
| 2560 | 6144 | a16 | `mxfp8_linear_a16_tma<Fp8Geometry<2560, 6144>, Mxfp8P4<Mxfp8TmaT64R256>>` (T>=2049) | — | — | — | — | — |
| 13952 | 2560 | a16 | `mxfp8_tma<Fp8Geometry<13952, 2560>, Mxfp8TmaT16R32>` (T 1-4) | 1 | 0.43 | 0.3% | 219.9 | 91% |
| 13952 | 2560 | a16 | `mxfp8_tma<Fp8Geometry<13952, 2560>, Mxfp8TmaT16R32>` (T 1-4) | 2 | 0.85 | 0.7% | 219.7 | 91% |
| 13952 | 2560 | a16 | `mxfp8_tma<Fp8Geometry<13952, 2560>, Mxfp8TmaT16R32>` (T 1-4) | 3 | 1.26 | 1.0% | 217.5 | 90% |
| 13952 | 2560 | a16 | `mxfp8_tma<Fp8Geometry<13952, 2560>, Mxfp8TmaT16R32>` (T 1-4) | 4 | 1.69 | 1.4% | 219.2 | 91% |
| 13952 | 2560 | a16 | `mxfp8_tma<Fp8Geometry<13952, 2560>, Mxfp8P4<Mxfp8TmaT16R128>>` (T 5-6) | 5 | 2.11 | 1.7% | 218.1 | 91% |
| 13952 | 2560 | a16 | `mxfp8_tma<Fp8Geometry<13952, 2560>, Mxfp8P4<Mxfp8TmaT16R128>>` (T 5-6) | 6 | 2.56 | 2.1% | 220.9 | 92% |
| 13952 | 2560 | a16 | `mxfp8_tma<Fp8Geometry<13952, 2560>, Mxfp8P2<Mxfp8TmaT16R128>>` (T 7-16) | 7 | 2.99 | 2.4% | 221.4 | 92% |
| 13952 | 2560 | a16 | `mxfp8_tma<Fp8Geometry<13952, 2560>, Mxfp8P2<Mxfp8TmaT16R128>>` (T 7-16) | 8 | 3.39 | 2.7% | 220.0 | 91% |
| 13952 | 2560 | a16 | `mxfp8_tma<Fp8Geometry<13952, 2560>, Mxfp8P2<Mxfp8TmaT16R128>>` (T 7-16) | 16 | 6.78 | 5.5% | 221.7 | 92% |
| 13952 | 2560 | a16 | `mxfp8_tma<Fp8Geometry<13952, 2560>, Fp8A16RingT32R64S3>` (T 17-64) | 24 | 10.10 | 8.2% | 221.7 | 92% |
| 13952 | 2560 | a16 | `mxfp8_tma<Fp8Geometry<13952, 2560>, Fp8A16RingT32R64S3>` (T 17-64) | 32 | 13.47 | 11% | 223.3 | 93% |
| 13952 | 2560 | a16 | `mxfp8_tma<Fp8Geometry<13952, 2560>, Fp8A16RingT32R64S3>` (T 17-64) | 48 | 19.56 | 16% | 219.2 | 91% |
| 13952 | 2560 | a16 | `mxfp8_tma<Fp8Geometry<13952, 2560>, Fp8A16RingT32R64S3>` (T 17-64) | 64 | 26.53 | 21% | 226.0 | 94% |
| 13952 | 2560 | a16 | `mxfp8_tma<Fp8Geometry<13952, 2560>, Mxfp8Tma<Fp8A16RingT32R64S3>>` (T 65-128) | 96 | 39.00 | 32% | 227.5 | 95% |
| 13952 | 2560 | a16 | `mxfp8_tma<Fp8Geometry<13952, 2560>, Mxfp8Tma<Fp8A16RingT32R64S3>>` (T 65-128) | 128 | 49.42 | 40% | 221.9 | 92% |
| 13952 | 2560 | a16 | `mxfp8_tma<Fp8Geometry<13952, 2560>, Mxfp8Tma<Fp8A16RingT64R128>>` (T 129-192) | 192 | 69.87 | 56% | 220.0 | 91% |
| 13952 | 2560 | a16 | `mxfp8_tma<Fp8Geometry<13952, 2560>, Mxfp8Tma<Mxfp8TmaT128R128>>` (T 193-4096) | 256 | 76.32 | 62% | 189.0 | 79% |
| 13952 | 2560 | a16 | `mxfp8_tma<Fp8Geometry<13952, 2560>, Mxfp8Tma<Mxfp8TmaT128R128>>` (T 193-4096) | 384 | 82.75 | 67% | 149.4 | 62% |
| 13952 | 2560 | a16 | `mxfp8_tma<Fp8Geometry<13952, 2560>, Mxfp8Tma<Mxfp8TmaT128R128>>` (T 193-4096) | 512 | 79.38 | 64% | 116.6 | 48% |
| 13952 | 2560 | a16 | `mxfp8_tma<Fp8Geometry<13952, 2560>, Mxfp8Tma<Mxfp8TmaT128R128>>` (T 193-4096) | 768 | 83.75 | 68% | 94.9 | 39% |
| 13952 | 2560 | a16 | `mxfp8_tma<Fp8Geometry<13952, 2560>, Mxfp8Tma<Mxfp8TmaT128R128>>` (T 193-4096) | 1024 | 81.39 | 66% | 78.6 | 33% |
| 13952 | 2560 | a16 | `mxfp8_tma<Fp8Geometry<13952, 2560>, Mxfp8Tma<Mxfp8TmaT128R128>>` (T 193-4096) | 1536 | 81.80 | 66% | 65.3 | 27% |
| 13952 | 2560 | a16 | `mxfp8_tma<Fp8Geometry<13952, 2560>, Mxfp8Tma<Mxfp8TmaT128R128>>` (T 193-4096) | 2048 | 80.95 | 65% | 57.8 | 24% |
| 13952 | 2560 | a16 | `mxfp8_tma<Fp8Geometry<13952, 2560>, Mxfp8P4<Mxfp8TmaT16R128>>` (T>=4097) | — | — | — | — | — |
| 13952 | 2560 | f16acc | `mxfp8_f16acc<Fp8Geometry<13952, 2560>, Mxfp8Tma<Mxfp8TmaT128R128>>` (T 1-768) | — | — | — | — | — |
| 13952 | 2560 | f16acc | `mxfp8_f16acc<Fp8Geometry<13952, 2560>, Mxfp8Tma<Fp8A16RingT64R128>>` (T 769-1024) | — | — | — | — | — |
| 13952 | 2560 | f16acc | `mxfp8_f16acc<Fp8Geometry<13952, 2560>, Mxfp8Tma<Mxfp8TmaT128R128>>` (T>=1025) | — | — | — | — | — |
| 16384 | 2560 | a16 | `mxfp8_tma<Fp8Geometry<16384, 2560>, Mxfp8TmaT16R64>` (T 1-16) | 1 | 0.44 | 0.4% | 225.2 | 94% |
| 16384 | 2560 | a16 | `mxfp8_tma<Fp8Geometry<16384, 2560>, Mxfp8TmaT16R64>` (T 1-16) | 2 | 0.87 | 0.7% | 224.2 | 93% |
| 16384 | 2560 | a16 | `mxfp8_tma<Fp8Geometry<16384, 2560>, Mxfp8TmaT16R64>` (T 1-16) | 3 | 1.30 | 1.1% | 224.4 | 93% |
| 16384 | 2560 | a16 | `mxfp8_tma<Fp8Geometry<16384, 2560>, Mxfp8TmaT16R64>` (T 1-16) | 4 | 1.73 | 1.4% | 224.1 | 93% |
| 16384 | 2560 | a16 | `mxfp8_tma<Fp8Geometry<16384, 2560>, Mxfp8TmaT16R64>` (T 1-16) | 5 | 2.16 | 1.7% | 223.7 | 93% |
| 16384 | 2560 | a16 | `mxfp8_tma<Fp8Geometry<16384, 2560>, Mxfp8TmaT16R64>` (T 1-16) | 6 | 2.58 | 2.1% | 222.5 | 92% |
| 16384 | 2560 | a16 | `mxfp8_tma<Fp8Geometry<16384, 2560>, Mxfp8TmaT16R64>` (T 1-16) | 7 | 3.01 | 2.4% | 222.9 | 93% |
| 16384 | 2560 | a16 | `mxfp8_tma<Fp8Geometry<16384, 2560>, Mxfp8TmaT16R64>` (T 1-16) | 8 | 3.45 | 2.8% | 224.2 | 93% |
| 16384 | 2560 | a16 | `mxfp8_tma<Fp8Geometry<16384, 2560>, Mxfp8TmaT16R64>` (T 1-16) | 16 | 6.83 | 5.5% | 223.3 | 93% |
| 16384 | 2560 | a16 | `mxfp8_tma<Fp8Geometry<16384, 2560>, Fp8A16RingMmaSchedule<32, 16, 1, 2, 4, 2>>` (T 17-32) | 24 | 10.14 | 8.2% | 222.3 | 92% |
| 16384 | 2560 | a16 | `mxfp8_tma<Fp8Geometry<16384, 2560>, Fp8A16RingMmaSchedule<32, 16, 1, 2, 4, 2>>` (T 17-32) | 32 | 13.53 | 11% | 224.2 | 93% |
| 16384 | 2560 | a16 | `mxfp8_tma<Fp8Geometry<16384, 2560>, Mxfp8Tma<Mxfp8TmaT32R32>>` (T 33-64) | 48 | 19.98 | 16% | 223.6 | 93% |
| 16384 | 2560 | a16 | `mxfp8_tma<Fp8Geometry<16384, 2560>, Mxfp8Tma<Mxfp8TmaT32R32>>` (T 33-64) | 64 | 26.50 | 21% | 225.5 | 94% |
| 16384 | 2560 | a16 | `mxfp8_tma<Fp8Geometry<16384, 2560>, Mxfp8Tma<Fp8A16RingT32R64S3>>` (T 65-128) | 96 | 38.78 | 31% | 225.8 | 94% |
| 16384 | 2560 | a16 | `mxfp8_tma<Fp8Geometry<16384, 2560>, Mxfp8Tma<Fp8A16RingT32R64S3>>` (T 65-128) | 128 | 50.01 | 40% | 224.0 | 93% |
| 16384 | 2560 | a16 | `mxfp8_tma<Fp8Geometry<16384, 2560>, Mxfp8P2<Mxfp8TmaT64R256>>` (T 129-192) | 192 | 68.48 | 55% | 214.8 | 89% |
| 16384 | 2560 | a16 | `mxfp8_tma<Fp8Geometry<16384, 2560>, Mxfp8Tma<Mxfp8TmaT128R128>>` (T 193-3072) | 256 | 74.97 | 61% | 184.9 | 77% |
| 16384 | 2560 | a16 | `mxfp8_tma<Fp8Geometry<16384, 2560>, Mxfp8Tma<Mxfp8TmaT128R128>>` (T 193-3072) | 384 | 84.56 | 68% | 151.7 | 63% |
| 16384 | 2560 | a16 | `mxfp8_tma<Fp8Geometry<16384, 2560>, Mxfp8Tma<Mxfp8TmaT128R128>>` (T 193-3072) | 512 | 82.92 | 67% | 121.0 | 50% |
| 16384 | 2560 | a16 | `mxfp8_tma<Fp8Geometry<16384, 2560>, Mxfp8Tma<Mxfp8TmaT128R128>>` (T 193-3072) | 768 | 83.02 | 67% | 93.2 | 39% |
| 16384 | 2560 | a16 | `mxfp8_tma<Fp8Geometry<16384, 2560>, Mxfp8Tma<Mxfp8TmaT128R128>>` (T 193-3072) | 1024 | 81.78 | 66% | 78.1 | 32% |
| 16384 | 2560 | a16 | `mxfp8_tma<Fp8Geometry<16384, 2560>, Mxfp8Tma<Mxfp8TmaT128R128>>` (T 193-3072) | 1536 | 81.51 | 66% | 64.2 | 27% |
| 16384 | 2560 | a16 | `mxfp8_tma<Fp8Geometry<16384, 2560>, Mxfp8Tma<Mxfp8TmaT128R128>>` (T 193-3072) | 2048 | 79.94 | 65% | 56.2 | 23% |
| 16384 | 2560 | a16 | `mxfp8_tma<Fp8Geometry<16384, 2560>, Mxfp8Tma<Mxfp8TmaT64R256>>` (T 3073-4096) | — | — | — | — | — |
| 16384 | 2560 | a16 | `mxfp8_tma<Fp8Geometry<16384, 2560>, Mxfp8P4<Mxfp8TmaT64R256>>` (T>=4097) | — | — | — | — | — |
| 16384 | 2560 | f16acc | `mxfp8_f16acc<Fp8Geometry<16384, 2560>, Mxfp8Tma<Fp8A16RingT64R128>>` (T 1-3072) | — | — | — | — | — |
| 16384 | 2560 | f16acc | `mxfp8_f16acc<Fp8Geometry<16384, 2560>, Mxfp8Tma<Mxfp8TmaT128R128>>` (T>=3073) | — | — | — | — | — |
