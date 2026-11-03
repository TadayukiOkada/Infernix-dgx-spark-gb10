# Linear kernel tables (GB10)

One file per Linear Op family. Each table has one row per token count `T` for every registered
`/shapes/` problem in that family, with the schedule the production dispatch table selects at that
`T`, its dispatch band, and the measured `TFLOP/s` and effective `GB/s`.

| Family | Shapes | Policies | Measured rows | T range |
|---|---:|---|---:|---|
| [Q8](q8.md) | 33 | a16 | 1234 | 1–2048 |
| [NVFP4](nvfp4.md) | 15 | a16, a4 | 1090 | 1–2048 |
| [BF16](bf16.md) | 25 | a16 | 900 | 1–2048 |
| [FP8](fp8.md) | 13 | a16, a8 | 666 | 1–2048 |
| [Q4](q4.md) | 11 | a16 | 429 | 1–2048 |
| [Q5](q5.md) | 7 | a16 | 291 | 1–2048 |
| [Q6](q6.md) | 4 | a16 | 164 | 1–2048 |
| [MXFP8](mxfp8.md) | 5 | a16, f16acc | 115 | 1–2048 |

Every table states its own provenance: the engine build (`git` commit and tree state), the exact
benchmark command and flags, the metric formulas, the GB10 reference denominators, and which bands
are not covered. Seven families are timed with `infernix_linear_bench` at the default `exact` route
set with a 48 MiB cold-L2 eviction read between samples; MXFP8 has no `--qtype` in that benchmark and
is timed with `infernix_mxfp8_flash_next_sweep` instead.

Each row is the better median of two independent cold-L2 passes (three for NVFP4), and each file
states how far apart those passes were for its rows. One pass is not a repeatable number at decode
widths: on this rig the pass-to-pass spread is about 1% at the median but reaches tens of percent on
the microsecond rows. Both passes must run with an idle host — GB10 shares one LPDDR5X pool between
the CPU and the GPU, so a concurrent build or a large-binary `cuobjdump` shows up as slower GPU
kernels. Measured rows are keyed to the benchmark binary that produced them, so a rebuild cannot
leave a stale row published unnoticed.

Each row also carries `%pipe` and `%mem`: its measured rate against the envelope that governs it —
the measured MMA-pipe peak of the pipe that policy actually runs on, and the `ckt env` measured
read-only bandwidth ceiling. The packed weight format is never the compute ceiling: an A16 route
dequantizes its stored weights and runs on the BF16 HMMA pipe whatever the codec, so only a native
A4/A8 route can approach the FP4/FP8 rate. A row whose binding envelope is near 100% is at the
hardware limit for that shape; the per-file notes name the two regimes (small-transfer ramp,
multi-kernel launch cost) where both percentages stay low without any schedule closing them.

Widths are decode 1–8, the dispatch-band boundaries up to 64, and prefill 128–2048 in 128-token steps
plus each shape's own band edges. A row whose metrics are empty is a dispatch band above the 2048
measured cap: production selects that schedule there, but the sweep did not time it.

These are Op-level kernel measurements, not end-to-end serving results; see
[Single-GPU serving performance](../../performance.md) for those and
[measurement and publication rules](../../benchmark_methodology.md) for the serving metric contract.
Commands and per-Op benchmark contracts are in [bench/README.md](../../../bench/README.md).
