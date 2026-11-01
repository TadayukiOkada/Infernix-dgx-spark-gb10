// GB10 (sm_121a, 48 SMs) retune 2026-10-23 of the A16 tensor-core band; cycle 2 widened the
// candidate grid (BlockK=64 deep rings, RowFast raster, wide 128-token MMA tiles) and re-derived
// the bands (1.309 times the sweep's family-best was still avoidable after cycle 1). All MMA and
// TMA schedules of this shape accumulate each output in one order (sweep invariance hash
// 693e7f6eb27f49ae), so the band is bit-identical; GEMV and sliced-K bands are untouched.
#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/fp8/fp8_launch.cuh"

namespace infernix::ops::detail {
namespace {
using Geometry   = Fp8Geometry<16384, 5120>;
using Gemv       = Fp8A16GemvSchedule<4, 4, 16, 4, Fp8CodeCache::Default, 1, 1>;
using Tma64x128  = Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 2, 1>;
using Tma192x128 = Fp8A8TmaMmaSchedule<192, 128, 128, 3, 4, 2, 1>;
using Bulk = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>;
using BulkG8 =
    Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 3, 1, Fp8MmaRaster::Grouped, 8>,
                           kGb10SplitWaveCtas, 4, 8>;

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens == 1) return fp8_linear_a16_gemv<Geometry, Gemv>(x, weight, out, stream);
    if (tokens <= 16) return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<16, 8, 2>>(x, weight, out, stream);
    if (tokens <= 25) return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<32, 8, 2>>(x, weight, out, stream);
    if (tokens <= 64) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT32R64>(x, weight, out, stream);
    if (tokens <= 128) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT32R64S3>(x, weight, out, stream);
    if (tokens <= 256) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT64R128>(x, weight, out, stream);
    if (tokens <= 288) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT96R128S3K64>(x, weight, out, stream);
    if (tokens <= 2304) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT128R128S2B2K64>(x, weight, out, stream);
    if (tokens <= 3072) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT128R128G8>(x, weight, out, stream);
    fp8_linear_a16_tma<Geometry, Fp8A16TmaT64R256G8>(x, weight, out, stream);
}

void launch_a8(const Tensor& x, const Weight& weight, Tensor& out, Fp8A8Workspace scratch,
               cudaStream_t stream) {
    if (x.ne[1] <= 32)
        return launch_fp8_a8<Geometry, Fp8A8T32R32K128>(x, weight, out, scratch, stream);
    if (x.ne[1] <= 64)
        return launch_fp8_a8<Geometry, Fp8A8T64R128K256>(x, weight, out, scratch, stream);
    if (x.ne[1] <= 128)
        return launch_fp8_a8_tma<Geometry, Tma64x128>(x, weight, out, scratch, stream);
    if (x.ne[1] <= 192)
        return launch_fp8_a8_tma<Geometry, Tma192x128>(x, weight, out, scratch, stream);
    // On GB10 (48 SMs), Bulk at T=512 has 256 tiles (5*48+16, split 3x) whereas MidBulk had an
    // unsplit 512-tile (10*48+32) tail; past T=3072, Grouped-8 raster avoids L2 activation eviction.
    if (x.ne[1] <= 3072)
        return launch_fp8_a8_tma<Geometry, Bulk>(x, weight, out, scratch, stream);
    launch_fp8_a8_tma<Geometry, BulkG8>(x, weight, out, scratch, stream);
}

bool uses_a8(std::int32_t, std::int32_t max_tokens) { return max_tokens >= 17; }

std::size_t partial_capacity_bytes(std::int32_t max_tokens) {
    return max_tokens > 256 ? Bulk::kPartialBytes : 0;
}
} // namespace

const Fp8LinearShape kFp8N16384K5120{16384,     5120,    launch_a16,
                                     launch_a8, uses_a8, partial_capacity_bytes};
} // namespace infernix::ops::detail
