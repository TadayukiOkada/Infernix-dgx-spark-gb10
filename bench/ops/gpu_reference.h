#pragma once
//
// gpu_reference.h — hardware reference constants for op-bench efficiency reporting.
//
// These are the DENOMINATORS used to turn an op's measured time into an
// efficiency percentage. They must describe the machine the benchmark actually
// runs on, or the percentages are meaningless.
//
// History: every bench in this tree previously reported against RTX 5090 GDDR7
// figures (1792 GB/s spec, 1674.5 GB/s sustained-read, 209.5 BF16 TFLOP/s). On
// GB10 those denominators are ~6.5x too large, so a kernel at a genuine 95% of
// achievable bandwidth displayed as ~14%. The tree is now tuned for GB10 only,
// and the values below are the GB10 ones.
//
// Provenance of each GB10 number (do not replace with datasheet peaks):
//   * kDramSpecGBs      273.0   Published DGX Spark / GB10 specification.
//                               https://docs.nvidia.com/dgx/dgx-spark/hardware.html
//   * kSustainedReadGBs 246.0   Measured pure-read ceiling (bench/gb10_peak_bw.cu,
//                               best config blocks=48 threads=512 -> 245.6-248.4).
//                               This is the honest denominator: ~90% of spec.
//   * kDenseBf16Tflops   97.0   Measured cuBLAS 8192^3 BF16 x BF16 -> FP32 accum.
//   * kDenseFp16Tflops   99.0   Measured cuBLAS 8192^3 FP16 x FP16 -> FP32 accum.
//   * kDenseFp8Tflops   220.0   Measured cuBLASLt 8192^3 E4M3 -> FP32 accum.
//                               MXFP8/NVFP4 block-scaled paths run at this same
//                               dense rate; the advertised "1 PFLOPS FP4" is a
//                               sparse marketing figure, NOT a dense denominator.
//
// All Tensor Core figures above are DENSE with FP32 accumulation. Do not
// substitute sparse or reduced-precision-accumulate numbers.

#include <cstdio>
#include <cstring>

#include <cuda_runtime.h>

namespace infernix::bench {

// Published memory specification. Use only as a theoretical reference; never as
// the denominator for "how close to the hardware are we".
constexpr double kDramSpecGBs = 273.0;

// Measured sustained read ceiling on this unit (see bench/gb10_peak_bw.cu).
constexpr double kSustainedReadGBs = 246.0;

// Measured dense Tensor Core rates with FP32 accumulation (cuBLAS / cuBLASLt).
constexpr double kDenseBf16Tflops = 97.0;
constexpr double kDenseFp16Tflops = 99.0;
constexpr double kDenseFp8Tflops  = 220.0;
// Block-scaled FP8 (MXFP8) sustains the same dense rate as FP8 on Blackwell.
constexpr double kDenseMxFloat8Tflops = kDenseFp8Tflops;
// Native block-scaled NVFP4 (OMMA.16864.F32.E2M1) sustains 2x dense FP8 rate (~440-494.5 TFLOP/s).
constexpr double kDenseNvfp4Tflops = 440.0;

// Reporting label for the reference machine these constants describe.
constexpr const char* kReferenceGpu = "GB10";

// Bandwidth denominator for efficiency percentages.
inline double device_peak_bw_gbs(int /*dev*/ = 0) { return kSustainedReadGBs; }

// Emits the standard bench banner. `actual` is the device actually in use; if it
// is not a GB10 the caller should treat the percentages as advisory only.
inline void print_reference_banner(const char* actual_gpu) {
    const bool is_gb10 = actual_gpu != nullptr && std::strstr(actual_gpu, "GB10") != nullptr;
    std::printf("# actual_gpu=%s reference_gpu=%s%s\n", actual_gpu == nullptr ? "?" : actual_gpu,
                kReferenceGpu, is_gb10 ? "" : " (MISMATCH: denominators are GB10)");
    std::printf("# dram_spec_gbs=%.1f sustained_read_gbs=%.1f\n", kDramSpecGBs, kSustainedReadGBs);
}

} // namespace infernix::bench
