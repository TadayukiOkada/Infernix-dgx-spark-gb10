// GB10 (sm_121a, 48 SMs) retune of the A16 band; cycle 2 widened the candidate grid (BlockK=64
// deep rings, wide 128-token MMA tiles). Cycle 3 (2026-10-25, free-device boundary + CUDA-graph
// sweep) replaced the dominated Fp8SlicedInstance<16, 4, *>/<32, 4, 1> mid-band (1.63x slower at
// T = 32, 1.56x at T = 33) with the k_slices=8 sliced family through T = 32 and the single-order
// TMA family from T = 33. All MMA and TMA schedules of this shape accumulate each output in one
// order (sweep invariance hash f097b49ca56bc13a); the k_slices=8 sliced family is bit-identical
// to the prior T = 2..16 incumbent, so every changed band is bit-exact (ckt ab: 14/14 BIT_EXACT).
#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/fp8/fp8_launch.cuh"
#include "ops/linear/f16acc/f16acc_mma.cuh"

namespace infernix::ops::detail {
namespace {
using Geometry   = Fp8Geometry<5120, 6144>;
using Gemv       = Fp8A16GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
using TmaT128R128 = Fp8A16TmaMmaSchedule<128, 128, 4, 4, 2, 1>;
using Tma32x64   = Fp8A8TmaMmaSchedule<32, 64, 128, 1, 2, 3, 2>;
using Tma64x128  = Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 3, 1>;
using MidBulk    = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 3, 1>, kGb10SplitWaveCtas, 4, 8>;
using Bulk       = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, kGb10SplitWaveCtas, 4, 8>;
using BulkG8 =
    Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 3, 1, Fp8MmaRaster::Grouped, 8>,
                           kGb10SplitWaveCtas, 4, 8>;

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens == 1) return fp8_linear_a16_gemv<Geometry, Gemv>(x, weight, out, stream);
    if (tokens <= 16) return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<16, 8, 2>>(x, weight, out, stream);
    if (tokens <= 32) return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<32, 8, 2>>(x, weight, out, stream);
    if (tokens <= 64) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT64R128>(x, weight, out, stream);
    if (tokens <= 128) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT64R256G8>(x, weight, out, stream);
    if (tokens <= 192) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT32R128>(x, weight, out, stream);
    if (tokens <= 256) return fp8_linear_a16_tma<Geometry, TmaT128R128>(x, weight, out, stream);
    if (tokens <= 288) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT96R128S3K64>(x, weight, out, stream);
    if (tokens <= 512) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT64R128S3B2K64>(x, weight, out, stream);
    if (tokens <= 2304) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT128R128S2B2K64>(x, weight, out, stream);
    fp8_linear_a16_tma<Geometry, Fp8A16TmaT128R128G8>(x, weight, out, stream);
}

void launch_a8(const Tensor& x, const Weight& weight, Tensor& out, Fp8A8Workspace scratch,
               cudaStream_t stream) {
    if (x.ne[1] <= 64)
        return launch_fp8_a8_tma<Geometry, Tma32x64>(x, weight, out, scratch, stream);
    if (x.ne[1] <= 128)
        return launch_fp8_a8<Geometry, Fp8A8T64R64K128>(x, weight, out, scratch, stream);
    if (x.ne[1] <= 192)
        return launch_fp8_a8_tma<Geometry, Tma64x128>(x, weight, out, scratch, stream);
    // The narrower row tile fills the GPU before the large-tile path reaches a full wave; past
    // T=2048, Grouped-8 raster avoids L2 activation eviction on GB10's 24 MiB L2.
    if (x.ne[1] <= 1024)
        return launch_fp8_a8_tma<Geometry, MidBulk>(x, weight, out, scratch, stream);
    if (x.ne[1] <= 2048)
        return launch_fp8_a8_tma<Geometry, Bulk>(x, weight, out, scratch, stream);
    launch_fp8_a8_tma<Geometry, BulkG8>(x, weight, out, scratch, stream);
}

bool uses_a8(std::int32_t, std::int32_t max_tokens) { return max_tokens >= 17; }

std::size_t partial_capacity_bytes(std::int32_t max_tokens) {
    if (max_tokens > 1024) return Bulk::kPartialBytes;
    return max_tokens > 192 ? MidBulk::kPartialBytes : 0;
}

// f16-accumulate-prefill (A16Only permissions: under AllowA8 the single-term A8 route takes these
// widths): the ring tiles of the Flash-Next problems, whose K this shares (not yet swept for this route).
void launch_f16(const Tensor& x, const Weight& weight, Tensor& out, F16AccumulateWorkspace scratch,
                cudaStream_t stream) {
    if (x.ne[1] <= 4096) return fp8_linear_f16acc<Geometry, Fp8A16RingT64R128>(x, weight, out, scratch, stream);
    fp8_linear_f16acc<Geometry, Fp8A16RingMmaSchedule<128, 128, 4, 4, 2, 1>>(x, weight, out, scratch, stream);
}

} // namespace

const Fp8LinearShape kFp8N5120K6144{5120,      6144,    launch_a16,
                                    launch_a8, uses_a8, partial_capacity_bytes,
                                    /*invariant*/ nullptr, /*pair*/ nullptr, 0,
                                    /*pair_block32*/ nullptr, launch_f16, kFp8F16MinTokens};
} // namespace infernix::ops::detail
