// Qwen3.8-Flash-Next GDN query/key/value/z projection, row-scaled FP8 with BF16 activations (W8A16, recipe C).
// The GEMV at T = 1 and the SIMT tile through T = 8 both reduce a column as 16 values per lane in one
// accumulator chain, so a column has the same bits at every width (column invariance, design §11.3).
// GB10 (sm_121a, 48 SMs) retune 2026-10-23 of the tensor-core band: the 5090-fitted TMA ladder kept the
// 32x64 tile to 256 and the 64x128 tile to 4096, which on 48 SMs leaves most of a prefill wave unsold
// (1.24-1.96 times the best tile at 384, 512, 4096 and 8192). Every MMA and TMA schedule accumulates each
// output over K in the same order, so the retuned band is bit-identical: the sweep's column-invariance
// hash is one value across all mma/tma candidates of this shape (1e21b70a4722c80a).
#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/fp8/fp8_launch.cuh"
#include "ops/linear/fp8/fp8_a8_pair_tma_mma.cuh"
#include "ops/linear/f16acc/f16acc_mma.cuh"

namespace infernix::ops::detail {
namespace {
using Geometry   = Fp8Geometry<16384, 2560>;
using Gemv       = Fp8A16GemvSchedule<8, 2, 16, 1, Fp8CodeCache::Default, 2, 2>;
using Simt       = Fp8A16SimtSchedule<4, 2, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked, Fp8CodeCache::Streaming, 1,
                                      Fp8SimtBlockOrder::RowsContiguous, 1>;
using RingT128R128 = Fp8A16RingMmaSchedule<128, 128, 4, 4, 2, 1>;
using TmaT64R256   = Fp8A16TmaMmaSchedule<64, 256, 2, 8, 2, 1>;

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens == 1) return fp8_linear_a16_gemv<Geometry, Gemv>(x, weight, out, stream);
    if (tokens <= 8) return fp8_linear_a16_simt<Geometry, 8, Simt>(x, weight, out, stream);
    if (tokens <= 16) return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<32, 8, 2>>(x, weight, out, stream);
    if (tokens <= 32) return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedR2Instance<32, 8, 2>>(x, weight, out, stream);
    if (tokens <= 128) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT32R64S3>(x, weight, out, stream);
    if (tokens <= 192) return fp8_linear_a16_tma<Geometry, TmaT64R256>(x, weight, out, stream);
    if (tokens <= 256) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT64R128S3B2K64>(x, weight, out, stream);
    if (tokens <= 288) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT96R128S3K64>(x, weight, out, stream);
    if (tokens <= 4096) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT128R128S2B2K64>(x, weight, out, stream);
    fp8_linear_a16_tma<Geometry, Fp8A16TmaT128R128G8>(x, weight, out, stream);
}

// No single-term A8 route: the checkpoint is weight-only, served with BF16 activations. The
// fp8-a8-pair fast path carries them as two E4M3 terms from T = 33.
bool uses_a8(std::int32_t, std::int32_t) { return false; }
// The pair route's admission for the opt-in --a8-activations route: faster from T = 33.
constexpr std::int32_t kPairMinTokens = 33;

// f16-accumulate-prefill: the exact route's tiles at the same widths (not yet swept for this route).
void launch_f16(const Tensor& x, const Weight& weight, Tensor& out, F16AccumulateWorkspace scratch,
                cudaStream_t stream) {
    if (x.ne[1] <= 4096) return fp8_linear_f16acc<Geometry, Fp8A16RingT64R128>(x, weight, out, scratch, stream);
    fp8_linear_f16acc<Geometry, RingT128R128>(x, weight, out, scratch, stream);
}
} // namespace

const Fp8LinearShape kFp8N16384K2560{16384,   2560, launch_a16, nullptr, uses_a8, nullptr,
                                     /*invariant*/ nullptr, launch_fp8_a8_pair<Geometry>, kPairMinTokens,
                                     launch_fp8_a8_pair_block32<Geometry>, launch_f16, kFp8F16MinTokens};
} // namespace infernix::ops::detail
