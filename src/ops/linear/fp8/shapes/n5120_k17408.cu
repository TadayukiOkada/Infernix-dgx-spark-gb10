// GB10 (sm_121a, 48 SMs) retune of the A16 band; cycle 2 widened the candidate grid (BlockK=64
// deep rings, wide 128-token MMA tiles). Cycle 3 (2026-10-25, free-device boundary + CUDA-graph
// sweep) replaced the dominated Fp8SlicedInstance<32, 8, 1> (T = 17..32) and isolated k_slices=2
// <64, 2, 2> (T = 33..64, 1.36x slower at T = 64) with the k_slices=8 sliced family through
// T = 32 and the single-order TMA family for T = 33..64; the 65..128 band keeps TmaT64R256. All
// MMA and TMA schedules accumulate each output in one order (sweep invariance hash
// 7a848b43c199ec78); the k_slices=8 sliced family is bit-identical to the T = 9..16 incumbent, so
// every changed band is bit-exact (ckt ab: 22/22 BIT_EXACT, PASS_NET_IMPROVEMENT).
#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/fp8/fp8_launch.cuh"

namespace infernix::ops::detail {
namespace {
using Geometry   = Fp8Geometry<5120, 17408>;
using Gemv       = Fp8A16GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
using TmaT64R256  = Fp8A16TmaMmaSchedule<64, 256, 2, 8, 2, 1>;
using TmaT128R128 = Fp8A16TmaMmaSchedule<128, 128, 4, 4, 2, 1>;
using Tma32x64   = Fp8A8TmaMmaSchedule<32, 64, 128, 1, 2, 3, 2>;
using Small = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 3, 1>, 170, 4, 8>;
using Mid   = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 3, 1>, 170, 4, 8>;
using Wide  = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<192, 128, 128, 3, 4, 2, 1>, 170, 4, 8>;
using Bulk  = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, 170, 4, 8>;
using BulkG8 =
    Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 3, 1, Fp8MmaRaster::Grouped, 8>, 170, 4, 8>;

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens == 1) return fp8_linear_a16_gemv<Geometry, Gemv>(x, weight, out, stream);
    if (tokens <= 8) return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<8, 8, 2>>(x, weight, out, stream);
    if (tokens <= 16) return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<16, 8, 2>>(x, weight, out, stream);
    if (tokens <= 32) return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<32, 8, 2>>(x, weight, out, stream);
    if (tokens <= 64) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT64R128>(x, weight, out, stream);
    if (tokens <= 128) return fp8_linear_a16_tma<Geometry, TmaT64R256>(x, weight, out, stream);
    if (tokens <= 192) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT32R128>(x, weight, out, stream);
    if (tokens <= 1024) return fp8_linear_a16_tma<Geometry, TmaT128R128>(x, weight, out, stream);
    if (tokens <= 1537) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT64R256G8>(x, weight, out, stream);
    if (tokens <= 2048) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT128R128G8>(x, weight, out, stream);
    fp8_linear_a16_tma<Geometry, Fp8A16TmaT64R256G8>(x, weight, out, stream);
}

void launch_a8(const Tensor& x, const Weight& weight, Tensor& out, Fp8A8Workspace scratch,
               cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens <= 64)
        return launch_fp8_a8_tma<Geometry, Tma32x64>(x, weight, out, scratch, stream);
    if (tokens <= 128)
        return launch_fp8_a8_tma<Geometry, Small>(x, weight, out, scratch, stream);
    // On GB10 (48 SMs), 128x128 Mid gives 120 CTAs at T=384 (2 waves + 24 split 2x) and
    // 160 CTAs at T=512 (3 waves + 16 split 3x), avoiding Bulk's unsplit 80-CTA (48+32) tail.
    if (tokens <= 512)
        return launch_fp8_a8_tma<Geometry, Mid>(x, weight, out, scratch, stream);
    if (tokens <= 768)
        return launch_fp8_a8_tma<Geometry, Wide>(x, weight, out, scratch, stream);
    if (tokens <= 2048)
        return launch_fp8_a8_tma<Geometry, Bulk>(x, weight, out, scratch, stream);
    launch_fp8_a8_tma<Geometry, BulkG8>(x, weight, out, scratch, stream);
}

bool uses_a8(std::int32_t, std::int32_t max_tokens) { return max_tokens >= 17; }

std::size_t partial_capacity_bytes(std::int32_t max_tokens) {
    if (max_tokens > 768) return Bulk::kPartialBytes;
    if (max_tokens > 512) return Wide::kPartialBytes;
    if (max_tokens > 128) return Mid::kPartialBytes;
    return max_tokens > 64 ? Small::kPartialBytes : 0;
}

} // namespace

const Fp8LinearShape kFp8N5120K17408{5120,      17408,   launch_a16,
                                     launch_a8, uses_a8, partial_capacity_bytes};
} // namespace infernix::ops::detail
