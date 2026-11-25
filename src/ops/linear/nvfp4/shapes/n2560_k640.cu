#include "ops/linear/nvfp4/nvfp4_shapes.h"
#include "ops/linear/nvfp4/nvfp4_launch.cuh"
#include "ops/linear/nvfp4/nvfp4_instances.cuh"

namespace infernix::ops::detail {
namespace {
// Qwen3.8-Flash-Next shared expert down projection [2560, 640] (A16-only, K=640 = 5*128).
using Geometry = Nvfp4Geometry<2560, 640>;

Nvfp4Launch select_a16(int tokens) {
    if (tokens <= 8) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<8, 2, 2>>;
    if (tokens <= 16) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<16, 2, 2, 2>>;
    if (tokens <= 24) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<24, 2, 2>>;
    if (tokens <= 32) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<32, 2, 1>>;
    if (tokens <= 64) return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<32, 32, 128, 16, 16, 2, 2>>;
    if (tokens <= 96) return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<64, 64, 128, 32, 16, 1, 2>>;
    if (tokens <= 128) return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<64, 128, 128, 32, 16, 2, 1>>;
    return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>;
}

void launch_a16(const Tensor& x, const Weight& w, Tensor& y, cudaStream_t stream) {
    select_a16(x.ne[1])(x, w, y, stream);
}

bool uses_a4(std::int32_t, std::int32_t) { return false; }
} // namespace

const Nvfp4LinearShape kNvfp4N2560K640{
    2560, 640, launch_a16, nullptr, uses_a4, nullptr, 0};
} // namespace infernix::ops::detail
