#pragma once

#include "gpu_reference.h"

// GB10 (sm_121a) reference rooflines for benchmark efficiency reporting.
// Aliased to gpu_reference.h (273 GB/s spec, 246 GB/s sustained read, 97/99/220/440 TFLOP/s).
namespace infernix::bench::roofline {

inline constexpr double kDramSpecGBs = ::infernix::bench::kDramSpecGBs;
inline constexpr double kDramReadGBs = ::infernix::bench::kSustainedReadGBs;

inline constexpr double kFp8Fp16AccumulateTFLOPs = ::infernix::bench::kDenseFp8Tflops;
inline constexpr double kFp8Fp32AccumulateTFLOPs = ::infernix::bench::kDenseFp8Tflops;
inline constexpr double kMxfp8Fp32AccumulateTFLOPs = ::infernix::bench::kDenseMxFloat8Tflops;
inline constexpr double kNvfp4Fp32AccumulateTFLOPs = ::infernix::bench::kDenseNvfp4Tflops;
inline constexpr double kBf16Fp32AccumulateTFLOPs  = ::infernix::bench::kDenseBf16Tflops;
inline constexpr double kFp16Fp16AccumulateTFLOPs = ::infernix::bench::kDenseFp16Tflops;

} // namespace infernix::bench::roofline
