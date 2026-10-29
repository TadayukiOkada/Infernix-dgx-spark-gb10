// Qwen3.5-122B-A10B attention input projection (Ornith-1.5-122B-NVFP4): one fp8_e4m3fn_row_f32
// parent [17408,3072] in physical order query 8192, key 512, output gate 8192, value 512, each row
// with its FP32 weight multiplier. Prefill encodes the activations once with the layer's static
// input scale (static A8); decode and verification keep BF16 activations (A16, column-invariant on
// the exact route). The schedule families and width bands are the [14336,5120] parent's
// (fp8_attn_input_*.cu).
#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"
#include "ops/linear/common/output.cuh"
#include "ops/linear/fp8/fp8_a16_gemv.cuh"
#include "ops/linear/fp8/fp8_a16_mma.cuh"
#include "ops/linear/fp8/fp8_instances.cuh"
#include "ops/linear/fp8/fp8_template_launch.cuh"

#include <stdexcept>

namespace infernix::ops::detail {
namespace {

constexpr int kHidden = 3072;
using Output          = LinearBf16SegmentedOutput<8192, 512, 8192, 512>;
using Tma64x128       = Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 2, 1>;
using Tma64x256       = Fp8A8TmaMmaSchedule<64, 256, 128, 2, 4, 2, 1>;
using Tma96x256       = Fp8A8TmaMmaSchedule<96, 256, 128, 3, 4, 2, 1>;
using Tma96x256G8     = Fp8A8TmaMmaSchedule<96, 256, 128, 3, 4, 2, 1, Fp8MmaRaster::Grouped, 8>;
using Tma128x128G8    = Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 3, 1, Fp8MmaRaster::Grouped, 8>;
using Bulk = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, 170, 4, 8>;

Output output_of(Tensor& q, Tensor& gate, Tensor& k, Tensor& v) {
    return {static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
            static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)};
}

template <class Schedule>
void sliced(const Tensor& x, const Weight& weight, Output output, cudaStream_t stream) {
    launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Schedule, kHidden>>(
        fp8_a16_operands<float>(x, weight), output, LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void gemm(const Tensor& x, const Weight& weight, Output output, cudaStream_t stream) {
    launch_fp8_a16_mma<Fp8ScheduleInstance<Schedule, kHidden>>(fp8_a16_operands<float>(x, weight), output,
                                                               LinearIdentityEpilogue{}, stream);
}

} // namespace

std::size_t fp8_attn_input_n17408_partial_capacity_bytes(std::int32_t max_tokens) {
    // The GB10-retuned route table (2026-10-29) runs every band on (Grouped-8) plain TMA tiles:
    // split-K Bulk was measured unreachable at all widths on 48 SMs, so no production call needs a
    // partial plane.
    (void)max_tokens;
    return 0;
}

void fp8_attn_input_n17408_invariant_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                            Tensor& gate, Tensor& k, Tensor& v,
                                            cudaStream_t stream) {
    // Four K-warps over the same K slices, reduced in the same order, at every width; the MMA's
    // columns are independent, and wider calls run as 64-column slices.
    const Output output = output_of(q, gate, k, v);
    if (x.ne[1] <= 8) return sliced<Fp8SlicedInstance<8, 4, 1>>(x, weight, output, stream);
    if (x.ne[1] <= 16) return sliced<Fp8SlicedInstance<16, 4, 1>>(x, weight, output, stream);
    if (x.ne[1] <= 32) return sliced<Fp8SlicedInstance<32, 4, 1>>(x, weight, output, stream);
    sliced<Fp8SlicedInstance<64, 4, 1>>(x, weight, output, stream);
}

void fp8_attn_input_n17408_a16_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                      Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream) {
    const Output output = output_of(q, gate, k, v);
    const int tokens    = x.ne[1];
    if (tokens == 1) {
        using Schedule = Fp8A16GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 6>;
        return launch_fp8_a16_gemv<Fp8ScheduleInstance<Schedule, kHidden>>(
            fp8_a16_operands<float>(x, weight), output, LinearIdentityEpilogue{}, stream);
    }
    if (tokens <= 16) return sliced<Fp8SlicedInstance<16, 4, 1>>(x, weight, output, stream);
    if (tokens <= 24) return sliced<Fp8SlicedInstance<32, 4, 2>>(x, weight, output, stream);
    if (tokens <= 32) return sliced<Fp8SlicedInstance<32, 4, 1>>(x, weight, output, stream);
    if (tokens <= 64)
        return gemm<Fp8A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>>(x, weight, output, stream);
    if (tokens <= 80 || (tokens >= 129 && tokens <= 160))
        return gemm<Fp8A16MmaSchedule<32, 80, 128, 32, 16, 1, 3>>(x, weight, output, stream);
    if (tokens <= 96)
        return gemm<Fp8A16MmaSchedule<64, 96, 128, 64, 16, 1, 2>>(x, weight, output, stream);
    gemm<Fp8A16MmaSchedule<64, 128, 64, 32, 16, 2, 2>>(x, weight, output, stream);
}

void fp8_attn_input_n17408_a8_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                     Tensor& gate, Tensor& k, Tensor& v, Fp8A8Workspace workspace,
                                     cudaStream_t stream) {
    launch_fp8_a8_quantize(x, weight, workspace, stream);
    const Output output = output_of(q, gate, k, v);
    const auto operands = fp8_a8_operands<float>(weight, workspace, x.ne[1]);
    const auto launch   = [&]<class Schedule>() {
        using S = Fp8ScheduleInstance<Schedule, kHidden>;
        if constexpr (S::kTmaSwizzle)
            launch_fp8_a8_tma_mma<S>(operands, output, LinearIdentityEpilogue{}, stream,
                                     workspace.partials);
        else
            launch_fp8_a8_mma<S>(operands, output, LinearIdentityEpilogue{}, stream, {},
                                 pdl::Dependency::Programmatic);
    };
    const int tokens = x.ne[1];
    // GB10 retune (2026-10-29, 48 SMs, 24 MiB L2): argmin over the 12 candidate schedules, band
    // boundaries arbitrated with interleaved ckt ab (6 rounds); cold-L2 medians cited below.
    if (tokens <= 32) return launch.template operator()<Fp8A8T32R32K128>();
    if (tokens <= 96) return launch.template operator()<Fp8A8T32R128K128>();
    if (tokens <= 128) return launch.template operator()<Tma64x128>();
    if (tokens <= 192) return launch.template operator()<Tma64x256>();
    // 193..320: the narrow 64x128 tile beats 96x256 (ab +3.5% at 256) and the former split-K
    // Bulk (+7.2% at 289), and it removes the old 288 -> 289 Bulk admission cliff.
    if (tokens <= 320) return launch.template operator()<Tma64x128>();
    // 321..448: 96x256 (204 CTAs over 48 SMs) beats the Grouped-8 tiles by 5.3% at 448 (ab).
    if (tokens <= 448) return launch.template operator()<Tma96x256>();
    // 449..1792: Grouped-8 128x128 tiles: +7.2% at 512, +7.0% at 768, +6.0% at 1024 over the
    // former Bulk band; ab ties Bulk.G8 at 768 and beats 96x256 by 1.6% at 1536.
    if (tokens <= 1792) return launch.template operator()<Tma128x128G8>();
    // 1793..4096: 96x256 again — ab -4.9% vs Bulk at 4096, sweep argmin at 2048/3072/4096; the
    // Grouped-8 raster loses this tile by 8.1% here (ab, SNR 15.9) and ties it at 5120.
    if (tokens <= 4096) return launch.template operator()<Tma96x256>();
    // 4097+: the Grouped-8 raster on the same 96x256 tile. The fp8 activation slab (3,072 B per
    // token) crosses the 24 MiB L2 near here and the plain raster thrashes it: ab -44.5% at 8192
    // (SNR 28.2); cold-L2 8192-token medians 8571 us plain vs 4967 us Grouped-8. Split-K Bulk stays
    // unreachable, so production needs no partial plane (see capacity above).
    launch.template operator()<Tma96x256G8>();
}

namespace {

using Tma64x128G8  = Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 2, 1, Fp8MmaRaster::Grouped, 8>;
using Tma64x256G8  = Fp8A8TmaMmaSchedule<64, 256, 128, 2, 4, 2, 1, Fp8MmaRaster::Grouped, 8>;
using BulkG8       = Fp8A8TmaSplitKSchedule<Tma128x128G8, 170, 4, 8>;
using BulkG8Wave48 = Fp8A8TmaSplitKSchedule<Tma128x128G8, 48, 4, 8>;

void launch_n17408_a8_schedule(Fp8AttnInputN17408ScheduleId schedule, const Tensor& x,
                               const Weight& weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                               Fp8A8Workspace workspace, cudaStream_t stream) {
    launch_fp8_a8_quantize(x, weight, workspace, stream);
    const Output output   = output_of(q, gate, k, v);
    const auto operands   = fp8_a8_operands<float>(weight, workspace, x.ne[1]);
    const auto launch_one = [&]<class Schedule>() {
        using S = Fp8ScheduleInstance<Schedule, kHidden>;
        if constexpr (S::kTmaSwizzle)
            launch_fp8_a8_tma_mma<S>(operands, output, LinearIdentityEpilogue{}, stream,
                                     workspace.partials);
        else
            launch_fp8_a8_mma<S>(operands, output, LinearIdentityEpilogue{}, stream, {},
                                 pdl::Dependency::Programmatic);
    };
    switch (schedule) {
    case Fp8AttnInputN17408ScheduleId::A8T32R32K128:
        return launch_one.template operator()<Fp8A8T32R32K128>();
    case Fp8AttnInputN17408ScheduleId::A8T32R128K128:
        return launch_one.template operator()<Fp8A8T32R128K128>();
    case Fp8AttnInputN17408ScheduleId::A8Tma64x128:
        return launch_one.template operator()<Tma64x128>();
    case Fp8AttnInputN17408ScheduleId::A8Tma64x128G8:
        return launch_one.template operator()<Tma64x128G8>();
    case Fp8AttnInputN17408ScheduleId::A8Tma64x256:
        return launch_one.template operator()<Tma64x256>();
    case Fp8AttnInputN17408ScheduleId::A8Tma64x256G8:
        return launch_one.template operator()<Tma64x256G8>();
    case Fp8AttnInputN17408ScheduleId::A8Tma96x256:
        return launch_one.template operator()<Tma96x256>();
    case Fp8AttnInputN17408ScheduleId::A8Tma96x256G8:
        return launch_one.template operator()<Tma96x256G8>();
    case Fp8AttnInputN17408ScheduleId::A8Tma128x128G8:
        return launch_one.template operator()<Tma128x128G8>();
    case Fp8AttnInputN17408ScheduleId::A8Bulk:
        return launch_one.template operator()<Bulk>();
    case Fp8AttnInputN17408ScheduleId::A8BulkG8:
        return launch_one.template operator()<BulkG8>();
    case Fp8AttnInputN17408ScheduleId::A8BulkG8Wave48:
        return launch_one.template operator()<BulkG8Wave48>();
    }
    throw std::logic_error("fp8 A8 attn_input_proj n17408: unknown schedule");
}

} // namespace

std::size_t fp8_attn_input_n17408_candidate_partial_capacity_bytes() {
    // Largest partial plane any forced candidate may need: Bulk's 170-wave 128x256 split-K tiles.
    return Bulk::kPartialBytes;
}

const char* fp8_attn_input_n17408_schedule_name(Fp8AttnInputN17408ScheduleId schedule) noexcept {
    switch (schedule) {
    case Fp8AttnInputN17408ScheduleId::A8T32R32K128:
        return "attn_input_proj.fp8.n17408.a8.t32.r32.k128";
    case Fp8AttnInputN17408ScheduleId::A8T32R128K128:
        return "attn_input_proj.fp8.n17408.a8.t32.r128.k128";
    case Fp8AttnInputN17408ScheduleId::A8Tma64x128:
        return "attn_input_proj.fp8.n17408.a8.tma64x128";
    case Fp8AttnInputN17408ScheduleId::A8Tma64x128G8:
        return "attn_input_proj.fp8.n17408.a8.tma64x128.g8";
    case Fp8AttnInputN17408ScheduleId::A8Tma64x256:
        return "attn_input_proj.fp8.n17408.a8.tma64x256";
    case Fp8AttnInputN17408ScheduleId::A8Tma64x256G8:
        return "attn_input_proj.fp8.n17408.a8.tma64x256.g8";
    case Fp8AttnInputN17408ScheduleId::A8Tma96x256:
        return "attn_input_proj.fp8.n17408.a8.tma96x256";
    case Fp8AttnInputN17408ScheduleId::A8Tma96x256G8:
        return "attn_input_proj.fp8.n17408.a8.tma96x256.g8";
    case Fp8AttnInputN17408ScheduleId::A8Tma128x128G8:
        return "attn_input_proj.fp8.n17408.a8.tma128x128.g8";
    case Fp8AttnInputN17408ScheduleId::A8Bulk:
        return "attn_input_proj.fp8.n17408.a8.bulk";
    case Fp8AttnInputN17408ScheduleId::A8BulkG8:
        return "attn_input_proj.fp8.n17408.a8.bulk.g8";
    case Fp8AttnInputN17408ScheduleId::A8BulkG8Wave48:
        return "attn_input_proj.fp8.n17408.a8.bulk.g8.w48";
    }
    return "attn_input_proj.fp8.n17408.a8.unknown";
}

bool fp8_attn_input_n17408_candidate_is_legal(Fp8AttnInputN17408ScheduleId schedule,
                                              const Fp8AttnInputProblem& problem) noexcept {
    if (problem.input_rows != 3072 || problem.query_rows != 8192 || problem.kv_rows != 512 ||
        problem.parent_rows != 17408 || problem.padded_k != 3072 || problem.cols < 1) {
        return false;
    }
    // Every A8 schedule slices tokens internally and validates only row/K tiles, so it is legal at
    // any token count; the sweep measures each one everywhere to find the true per-token argmin.
    (void)schedule;
    return true;
}

void fp8_attn_input_n17408_execute_candidate(Fp8AttnInputN17408ScheduleId schedule, const Tensor& x,
                                             const Weight& weight, Tensor& q, Tensor& gate,
                                             Tensor& k, Tensor& v, WorkspaceArena& workspace,
                                             cudaStream_t stream) {
    const Fp8AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], weight.n, weight.padded_shape[1],
                                      x.ne[1]};
    if (!fp8_attn_input_n17408_candidate_is_legal(schedule, problem)) {
        throw std::invalid_argument(
            "fp8 A8 attn_input_proj n17408: candidate schedule is not legal for problem");
    }
    auto scope = workspace.scope();
    // Always reserve the largest split-K partial plane so a forced split-K candidate has aligned
    // partials regardless of token count (production sizing keys on tokens, candidates do not).
    const Fp8A8Workspace scratch = allocate_fp8_a8_workspace(
        workspace, x.ne[1], weight.k, fp8_attn_input_n17408_candidate_partial_capacity_bytes());
    launch_n17408_a8_schedule(schedule, x, weight, q, gate, k, v, scratch, stream);
}

} // namespace infernix::ops::detail
