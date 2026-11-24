#include "ops/linear/bf16/bf16_mma_epilogue.cuh"
#include "ops/linear/q8/q8_a16_tma_k32.cuh"
#include "ops/linear/q8/q8_gemv_launch.cuh"
#include "ops/linear/q8/q8_grouped_sliced_k_launch.cuh"
#include "ops/linear/q8/q8_instances.cuh"
#include "ops/linear/q8/q8_mma_launch.cuh"
#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_simt_launch.cuh"
#include "ops/linear/q8/q8_sliced_k_launch.cuh"
#include "ops/linear/q8/q8_stream_launch.cuh"
#include "ops/gdn_input_proj/q8/q8_gdn_conv_columns.cuh"

#include <stdexcept>

namespace infernix::ops::detail {
namespace {
// Qwen3.8-Flash-Next GDN q/k/v/z projection (packed): the register-streamed route up to 8 columns (5-28 %
// faster than SIMT), the MMA tiles beyond (M1 sweep). K1: 2 rows per warp, x read through L1.
// NOTE (GB10, 2026): a bit-exact retune to RowsPerWarp=1/Warps=4 (lower regs 128->104, NCU long_scoreboard
// 30->12, issue_active 8.6->20.5%, faster in the isolated op-bench at every decode width) REGRESSED the
// full-model decode (60.86 vs 61.4 t/s). Like the token-axis gate_up split, the extra CTAs (grid 1024->4096)
// multiply concurrent weight streams that contend the unified LPDDR5x fabric at model scale, outweighing
// the per-kernel latency win. Kept at 2/8 pending a model-scale re-evaluation.
template <int MaxCols, int ColsPerPass>
using Stream = Q8A16StreamSchedule<2560, 2, 1, 8, MaxCols, ColsPerPass, Q8StreamX::Global, 1>;

// The routes of both tables up to 8 columns, where one call holds the one-position-per-sequence
// columns of a decode round. The fused GDN convolution (launch_q8_n16384_k2560_conv_snapshot) runs
// these same routes with its own row consumer; the contraction and so each projected FP32 value is
// unchanged by the consumer.
template <class Output, class Epilogue>
void launch_decode_routes(const Q8LinearOperands& operands, Output output, Epilogue epilogue,
                          cudaStream_t stream) {
    if (operands.tokens <= 4) return launch_q8_a16_stream<Stream<4, 4>>(operands, output, epilogue, stream);
    launch_q8_a16_stream<Stream<8, 8>>(operands, output, epilogue, stream);
}

using TmaHot96   = Q8A16TmaMmaSchedule<64, 96, 64, 16, 2, 8, 2560>;
using TmaBulk128 = Q8A16TmaMmaSchedule<64, 128, 64, 16, 2, 8, 2560>;
using TmaK32     = Q8A16TmaK32Schedule<8, 2560>;

// One route table for both outputs: the packed [16384,T] linear result, and the q/k/v rows and z rows
// written straight to their two destinations (gdn_input_proj). The output map changes only where a
// value is stored, so both produce the same bits.
template <class Output>
void launch_routes(const Q8LinearOperands& operands, Output output, cudaStream_t stream) {
    const LinearIdentityEpilogue epilogue{};
    const Bf16MmaCtaWarpStoreEpilogue<> tma_epilogue{};
    const std::int32_t tokens = operands.tokens;
    if (tokens <= 2) return launch_decode_routes(operands, output, epilogue, stream);
    if (tokens <= 4) return launch_q8_a16_simt<q8_instances::SimtR8T4>(operands, output, epilogue, stream);
    if (tokens <= 5) return launch_q8_a16_simt<q8_instances::SimtR8T5>(operands, output, epilogue, stream);
    if (tokens <= 6) return launch_q8_a16_simt<q8_instances::SimtR8T6>(operands, output, epilogue, stream);
    if (tokens <= 8) return launch_q8_a16_simt<q8_instances::SimtR8T8>(operands, output, epilogue, stream);
    if (tokens <= 16) return launch_q8_a16_sliced_k_mma<q8_instances::SlicedR16T16W8S2>(operands, output, epilogue, stream);
    if (tokens <= 32) return launch_q8_a16_sliced_k_mma<q8_instances::SlicedR16T32W8S1M2>(operands, output, epilogue, stream);
    if (tokens <= 64) return launch_q8_a16_mma<q8_instances::MmaR128x16T64K128A2M1>(operands, output, epilogue, stream);
    if (operands.k == 2560 && operands.padded_k == 2560 && operands.rows % 128 == 0) {
        if (tokens <= 96 || (tokens > 128 && tokens <= 192))
            return launch_q8_a16_tma_mma<TmaHot96>(operands, output, tma_epilogue, stream);
        if (tokens <= 511)
            return launch_q8_a16_tma_mma<TmaBulk128>(operands, output, tma_epilogue, stream);
        return launch_q8_a16_tma_k32<TmaK32>(operands, output, tma_epilogue, stream);
    }
    if (tokens <= 96) return launch_q8_a16_mma<q8_instances::MmaR96T96>(operands, output, epilogue, stream);
    if (tokens <= 128) return launch_q8_a16_mma<q8_instances::MmaR64T128>(operands, output, epilogue, stream);
    if (tokens <= 192) return launch_q8_a16_mma<q8_instances::MmaR64x32T96K128A1>(operands, output, epilogue, stream);
    if (tokens <= 384) {
        return launch_q8_a16_mma<q8_instances::MmaR64T128>(operands, output, epilogue, stream);
    }
    launch_q8_a16_mma<q8_instances::MmaR128x64T128K64A1>(operands, output, epilogue, stream);
}

// The q8-sliced-k fast path: NInfer's routes for this problem (NInfer cd521ab7d). The T = 1 GEMV, the
// sliced-K MMA at T = 2-8 (group scales applied after each MMA) and the grouped sliced-K MMA accumulate
// in different orders, so a column's bits depend on the call's T.
using FastGemv    = Q8A16GemvSchedule<8, 1, 2, 2560, 8>;
using FastSliced8 = Q8A16SlicedKMmaSchedule<8, 4, 1, 4, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                            Q8ActivationStage::RuntimeActive, 2560>;
using FastGrouped = Q8A16GroupedSlicedKMmaSchedule<16, 2, 2, 2, 1, 2560, Cache::cg, Cache::ca, true>;
using FastSliced24 = Q8A16SlicedKMmaSchedule<24, 4, 1, 2, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                             Q8ActivationStage::RuntimeActive, 2560>;
using FastSliced32 = Q8A16SlicedKMmaSchedule<32, 4, 2, 2, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                             Q8ActivationStage::RuntimeActive, 2560>;
using FastSliced40 = Q8A16SlicedKMmaSchedule<40, 4, 2, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                             Q8ActivationStage::RuntimeActive, 2560>;

template <class Output, class Epilogue>
void launch_fast_decode_routes(const Q8LinearOperands& operands, Output output, Epilogue epilogue,
                               cudaStream_t stream) {
    if (operands.tokens <= 1) return launch_q8_a16_gemv<FastGemv>(operands, output, epilogue, stream);
    launch_q8_a16_sliced_k_mma<FastSliced8>(operands, output, epilogue, stream);
}

template <class Output>
void launch_fast_routes(const Q8LinearOperands& operands, Output output, cudaStream_t stream) {
    const LinearIdentityEpilogue epilogue{};
    const std::int32_t tokens = operands.tokens;
    if (tokens <= 8) return launch_fast_decode_routes(operands, output, epilogue, stream);
    if (tokens <= 16) return launch_q8_a16_grouped_sliced_k_mma<FastGrouped>(operands, output, epilogue, stream);
    if (tokens <= 24) return launch_q8_a16_sliced_k_mma<FastSliced24>(operands, output, epilogue, stream);
    if (tokens <= 32) return launch_q8_a16_sliced_k_mma<FastSliced32>(operands, output, epilogue, stream);
    if (tokens <= 40) return launch_q8_a16_sliced_k_mma<FastSliced40>(operands, output, epilogue, stream);
    launch_routes(operands, output, stream);
}

void launch_linear(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_routes(q8_linear_operands(x, weight),
                  LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]}, stream);
}

void launch_linear_fast(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_fast_routes(q8_linear_operands(x, weight),
                       LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]}, stream);
}
} // namespace

Q8Launch select_q8_n16384_k2560(std::int32_t) { return launch_linear; }

Q8Launch select_q8_n16384_k2560_fast(std::int32_t) { return launch_linear_fast; }

void launch_q8_n16384_k2560_split(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                  LinearRoutes routes, cudaStream_t stream) {
    const LinearBf16SegmentedOutput<10240, 6144> output{
        {static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)}};
    if (has_route(routes, LinearRoutes::Q8SlicedK)) {
        launch_fast_routes(q8_linear_operands(x, weight), output, stream);
        return;
    }
    launch_routes(q8_linear_operands(x, weight), output, stream);
}

void launch_q8_n16384_k2560_conv_snapshot(const Tensor& x, const Weight& weight,
                                          const Tensor& conv_weight, Tensor& conv_states,
                                          const Tensor& initial_state_slots,
                                          const Tensor& snapshot_base_slots, Tensor& query,
                                          Tensor& key, Tensor& value, Tensor& z, LinearRoutes routes,
                                          cudaStream_t stream) {
    const Q8LinearOperands operands = q8_linear_operands(x, weight);
    if (operands.tokens < 1 || operands.tokens > 8) {
        throw std::invalid_argument("Q8 [16384,2560] fused GDN convolution covers 1..8 sequences");
    }
    const GdnConvSnapshotColumnsOutput output{
        static_cast<const __nv_bfloat16*>(conv_weight.data),
        static_cast<__nv_bfloat16*>(conv_states.data),
        static_cast<const std::int32_t*>(initial_state_slots.data),
        static_cast<const std::int32_t*>(snapshot_base_slots.data),
        static_cast<__nv_bfloat16*>(query.data),
        static_cast<__nv_bfloat16*>(key.data),
        static_cast<__nv_bfloat16*>(value.data),
        static_cast<__nv_bfloat16*>(z.data),
    };
    const GdnConvSnapshotRowEpilogue epilogue{};
    if (has_route(routes, LinearRoutes::Q8SlicedK)) {
        launch_fast_decode_routes(operands, output, epilogue, stream);
        return;
    }
    launch_decode_routes(operands, output, epilogue, stream);
}

} // namespace infernix::ops::detail
