#include "ops/linear/bf16/bf16_mma_epilogue.cuh"
#include "ops/linear/q4/q4_a16_tma_prefetch.cuh"
#include "ops/linear/q4/q4_instance_launch.cuh"
#include "ops/linear/q4/q4_shapes.h"

namespace infernix::ops::detail {
namespace {

using Gemv =
    Q4A16GemvSchedule<4, 1, 16, 1, Q4GemvActivationAccess::Direct, Q4GemvLaneMapping::PackedWord8,
                      Q4GemvDecodeMode::Fp16Mantissa, Q4GemvCodeTransfer::AsyncVector16,
                      Q4GemvScaleAccess::SharedPair32, Cache::ca, 80, 1>;

template <int Capacity>
using Small = Q4A16SlicedKMmaSchedule<32, 8, 4, 2, Cache::cg, Cache::ca, 2, 5120, Capacity>;

template <class Schedule>
struct HeadSchedule : Schedule {
    static constexpr bool kIndependentTokenTail = true;
};

using Fallback80  = HeadSchedule<q4_instances::MmaR64T80K64Wr16Wt40S2A2B1>;
using Fallback96  = HeadSchedule<q4_instances::MmaR64T96K64Wr32Wt16S2A2B1>;
using Fallback128 = HeadSchedule<q4_instances::MmaR64T128K64Wr64Wt32S2A2B1>;

// GB10 (48 SMs, 24 MiB L2): RasterRows = 8 across 2D TMA prefetched prefill tiles.
using Tma96   = Q4A16TmaPrefetchSchedule<64, 96, 32, 24, 2, 5120, 1, 8, 4, 1, 128, 1>;
using Tma128  = Q4A16TmaPrefetchSchedule<64, 128, 64, 16, 2, 5120, 1, 8>;
using Mid96   = Q4A16TmaPrefetchSchedule<128, 96, 64, 24, 2, 5120, 1, 8, 4, 1, 64, 1>;
using Bulk256 = Q4A16TmaPrefetchSchedule<64, 256, 64, 32, 1, 5120, 1, 8, 4, 1, 64, 1>;

template <int Capacity>
void launch_small(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if (w.padded_shape[1] == 5120)
        launch_q4_a16_sliced_instance<Small<Capacity>>(x, w, out, stream);
    else
        launch_q4_a16_sliced_r32_t8_w4_s2(x, w, out, stream);
}

template <class Schedule, class PaddedFallback>
void launch_tma(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if (w.padded_shape[1] != 5120) {
        launch_q4_a16_mma_instance<PaddedFallback>(x, w, out, stream);
        return;
    }
    if (out.dtype == DType::FP32) {
        launch_q4_a16_tma_prefetch<Schedule>(
            q4_linear_operands(x, w), LinearFp32Output{static_cast<float*>(out.data), w.n},
            LinearIdentityEpilogue{}, stream);
    } else {
        launch_q4_a16_tma_prefetch<Schedule>(
            q4_linear_operands(x, w), LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
            Bf16MmaCtaWarpStoreEpilogue<>{}, stream);
    }
}

void launch_bulk(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if (w.padded_shape[1] != 5120) {
        launch_q4_a16_mma_instance<Fallback128>(x, w, out, stream);
        return;
    }
    const int tail = x.ne[1] % Bulk256::kBlockTokens;
    if (!tail || tail > Tma128::kBlockTokens) {
        launch_tma<Bulk256, Fallback128>(x, w, out, stream);
        return;
    }
    const int prefix = x.ne[1] - tail;
    auto input       = x.slice(1, 0, prefix);
    auto output      = out.slice(1, 0, prefix);
    launch_tma<Bulk256, Fallback128>(input, w, output, stream);
    input  = x.slice(1, prefix, tail);
    output = out.slice(1, prefix, tail);
    select_q4_n248320_k5120(tail)(input, w, output, stream);
}

} // namespace

Q4Launch select_q4_n248320_k5120(std::int32_t tokens) {
    if (tokens <= 1) return launch_q4_a16_gemv_instance<Gemv>;
    if (tokens <= 4) return launch_small<4>;
    if (tokens <= 8) return launch_small<8>;
    if (tokens <= 16) return launch_q4_a16_sliced_r32_t16_w4_s1;
    if (tokens <= 32) return launch_q4_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 80) return launch_tma<Tma96, Fallback80>;
    if (tokens <= 96) return launch_tma<Tma96, Fallback96>;
    if (tokens <= 640) return launch_tma<Tma128, Fallback128>;
    return launch_bulk;
}

} // namespace infernix::ops::detail
