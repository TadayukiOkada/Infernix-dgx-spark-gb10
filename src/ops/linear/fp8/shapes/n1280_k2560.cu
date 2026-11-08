// Qwen3.8-Flash-Next shared expert gate and up projection, row-scaled FP8 with BF16 activations (W8A16,
// recipe C). The GEMV and SIMT routes through T = 8 both reduce a column as 16 values per lane in one
// accumulator chain, so a column has the same bits at every width (design §11.3).
// GB10 (sm_121a, 48 SMs) retune 2026-10-23 of the tensor-core band from
// bench/ops/fp8_flash_next_sweep.cu: the 5090 ladder's tile order loses 1.05-1.15 times the best tile at
// 384-768 and beyond 1536 here. All MMA and TMA schedules of this shape share one accumulation order
// (sweep invariance hash fc58b64ad44acbaa), so the band is retuned without changing any output bit.
#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/fp8/fp8_launch.cuh"
#include "ops/linear/fp8/fp8_a8_pair_tma_mma.cuh"
#include "ops/linear/f16acc/f16acc_mma.cuh"

namespace infernix::ops::detail {
namespace {
using Geometry = Fp8Geometry<1280, 2560>;
using Gemv     = Fp8A16GemvSchedule<8, 2, 16, 1, Fp8CodeCache::Default, 2, 2>;
using Simt     = Fp8A16SimtSchedule<4, 2, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked, Fp8CodeCache::Streaming, 1,
                                    Fp8SimtBlockOrder::RowsContiguous, 1>;
using RingT64R256     = Fp8A16RingMmaSchedule<64, 256, 2, 8, 2, 1>;
using TmaT64R256      = Fp8A16TmaMmaSchedule<64, 256, 2, 8, 2, 1>;
using TmaT96R128S3K64 = Fp8A16TmaMmaSchedule<96, 128, 2, 4, 3, 1, 64>;

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens == 1) return fp8_linear_a16_gemv<Geometry, Gemv>(x, weight, out, stream);
    if (tokens <= 8) return fp8_linear_a16_simt<Geometry, 8, Simt>(x, weight, out, stream);
    if (tokens <= 16) return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<16, 8, 2>>(x, weight, out, stream);
    if (tokens <= 32) return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<32, 4, 2>>(x, weight, out, stream);
    if (tokens <= 128) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT32R64>(x, weight, out, stream);
    if (tokens <= 512) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT64R128>(x, weight, out, stream);
    if (tokens <= 1536) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT64R128S3B2K64>(x, weight, out, stream);
    if (tokens <= 3584) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT128R128S2B2K64>(x, weight, out, stream);
    fp8_linear_a16_tma<Geometry, Fp8A16TmaT64R256RF>(x, weight, out, stream);
}

// No single-term A8 route: the checkpoint is weight-only, served with BF16 activations. The
// fp8-a8-pair fast path carries them as two E4M3 terms from T = 1025.
bool uses_a8(std::int32_t, std::int32_t) { return false; }
// The pair route's admission for the opt-in --a8-activations route: faster from T = 1025.
constexpr std::int32_t kPairMinTokens = 1025;

// f16-accumulate-prefill: the exact route's tiles at the same widths (not yet swept for this route).
void launch_f16(const Tensor& x, const Weight& weight, Tensor& out, F16AccumulateWorkspace scratch,
                cudaStream_t stream) {
    if (x.ne[1] <= 3072) return fp8_linear_f16acc<Geometry, Fp8A16RingT64R128>(x, weight, out, scratch, stream);
    fp8_linear_f16acc<Geometry, RingT64R256>(x, weight, out, scratch, stream);
}
} // namespace

const Fp8LinearShape kFp8N1280K2560{1280,    2560, launch_a16, nullptr, uses_a8, nullptr,
                                    /*invariant*/ nullptr, launch_fp8_a8_pair<Geometry>, kPairMinTokens,
                                    launch_fp8_a8_pair_block32<Geometry>, launch_f16, kFp8F16MinTokens};
} // namespace infernix::ops::detail
