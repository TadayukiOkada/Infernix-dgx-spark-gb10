// Qwen3.5-122B-A10B GDN input projection (Ornith-1.5-122B-NVFP4): one fp8_e4m3fn_row_f32 parent
// [20480,3072], query/key/value 12288 (16 key heads and 64 value heads of 128) then z 8192, each
// row with its FP32 weight multiplier. Prefill encodes the activations once with the layer's static
// input scale (static A8); decode and verification keep BF16 activations (A16, column-invariant on
// the exact route). The schedule families and width bands are the [16384,5120] parent's
// (fp8_gdn_input_*.cu).
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"
#include "ops/linear/common/output.cuh"
#include "ops/linear/fp8/fp8_a16_gemv.cuh"
#include "ops/linear/fp8/fp8_a16_mma.cuh"
#include "ops/linear/fp8/fp8_a16_simt.cuh"
#include "ops/linear/fp8/fp8_a16_sliced_k_mma.cuh"
#include "ops/linear/fp8/fp8_instances.cuh"
#include "ops/linear/fp8/fp8_template_launch.cuh"

#include <stdexcept>

namespace infernix::ops::detail {
namespace {

constexpr int kHidden = 3072;
using Output          = LinearBf16SegmentedOutput<12288, 8192>;
using Tma64x128       = Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 2, 1>;
using Tma192x128      = Fp8A8TmaMmaSchedule<192, 128, 128, 3, 4, 2, 1>;
using Tma192x128G8    = Fp8A8TmaMmaSchedule<192, 128, 128, 3, 4, 2, 1, Fp8MmaRaster::Grouped, 8>;
using Tma128x128G8    = Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 3, 1, Fp8MmaRaster::Grouped, 8>;
using MidBulk = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 2, 1>, 170, 4, 8>;
using Bulk    = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, 170, 4, 8>;

Output output_of(Tensor& qkv, Tensor& z) {
    return {static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)};
}

template <class Schedule, int Capacity = 0>
void sliced(const Tensor& x, const Weight& weight, Output output, cudaStream_t stream) {
    launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Schedule, kHidden, Capacity>>(
        fp8_a16_operands<float>(x, weight), output, LinearIdentityEpilogue{}, stream);
}

template <int Capacity>
void small_mma(const Tensor& x, const Weight& weight, Output output, cudaStream_t stream) {
    constexpr int warps = Capacity <= 8 ? 16 : Capacity <= 24 ? 8 : 4;
    sliced<Fp8A16SlicedKMmaSchedule<warps, Capacity, warps == 16 ? 1 : 2>, Capacity>(
        x, weight, output, stream);
}

template <class Schedule>
void gemm(const Tensor& x, const Weight& weight, Output output, cudaStream_t stream) {
    static_assert(12288 % Schedule::kBlockRows == 0 && 8192 % Schedule::kBlockRows == 0);
    static_assert(Schedule::kSharedBytes <= 48 * 1024);
    launch_fp8_a16_mma<Fp8ScheduleInstance<Schedule, kHidden>>(fp8_a16_operands<float>(x, weight), output,
                                                               LinearIdentityEpilogue{}, stream);
}

} // namespace

std::size_t fp8_gdn_input_n20480_partial_capacity_bytes(std::int32_t max_tokens) {
    // GB10 retune: the first split-K band is 321..384 (Bulk's 80 row tiles are the 384 argmin);
    // narrower maxima run entirely on plain TMA tiles.
    return max_tokens > 320 ? Bulk::kPartialBytes : 0;
}

void fp8_gdn_input_n20480_invariant_launch(const Tensor& x, const Weight& weight, Tensor& qkv,
                                           Tensor& z, cudaStream_t stream) {
    // Four K-warps over the same K slices, reduced in the same order, at every width; the MMA's
    // columns are independent, and wider calls run as 64-column slices.
    const Output output = output_of(qkv, z);
    if (x.ne[1] <= 8) return sliced<Fp8SlicedInstance<8, 4, 1>>(x, weight, output, stream);
    if (x.ne[1] <= 16) return sliced<Fp8SlicedInstance<16, 4, 1>>(x, weight, output, stream);
    if (x.ne[1] <= 32) return sliced<Fp8SlicedInstance<32, 4, 1>>(x, weight, output, stream);
    sliced<Fp8SlicedInstance<64, 4, 1>>(x, weight, output, stream);
}

void fp8_gdn_input_n20480_a16_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                     cudaStream_t stream) {
    const Output output = output_of(qkv, z);
    const int columns   = x.ne[1];
    if (columns == 1) {
        using Schedule = Fp8A16GemvSchedule<4, 4, 16, 4, Fp8CodeCache::Default, 1, 1>;
        return launch_fp8_a16_gemv<Fp8ScheduleInstance<Schedule, kHidden>>(
            fp8_a16_operands<float>(x, weight), output, LinearIdentityEpilogue{}, stream);
    }
    if (columns <= 4) {
        using Schedule =
            Fp8A16SimtSchedule<8, 2, 16, 4, 1, Fp8SimtActivationAccess::SharedPhase,
                               Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
        return launch_fp8_a16_simt<Fp8ScheduleInstance<Schedule, kHidden, 4>>(
            fp8_a16_operands<float>(x, weight), output, LinearIdentityEpilogue{}, stream);
    }
    if (columns <= 16) {
        using Schedule = Fp8A16SlicedKMmaSchedule<4, 16, 7, Cache::ca, Cache::cg,
                                                  Fp8ActivationStage::PaddedZero, 1>;
        return sliced<Schedule>(x, weight, output, stream);
    }
    if (columns <= 24) return small_mma<24>(x, weight, output, stream);
    if (columns <= 32) return small_mma<32>(x, weight, output, stream);
    if (columns <= 64)
        return gemm<Fp8A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>>(x, weight, output, stream);
    if (columns <= 96)
        return gemm<Fp8A16MmaSchedule<64, 96, 128, 64, 16, 1, 2>>(x, weight, output, stream);
    gemm<Fp8A16MmaSchedule<64, 128, 64, 32, 16, 2, 2>>(x, weight, output, stream);
}

void fp8_gdn_input_n20480_a8_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                    Fp8A8Workspace workspace, cudaStream_t stream) {
    launch_fp8_a8_quantize(x, weight, workspace, stream);
    const Output output = output_of(qkv, z);
    const auto operands = fp8_a8_operands<float>(weight, workspace, x.ne[1]);
    const auto launch   = [&]<class Schedule>() {
        using S = Fp8ScheduleInstance<Schedule, kHidden>;
        if constexpr (S::kTmaSwizzle)
            launch_fp8_a8_tma_mma<S>(operands, output, LinearIdentityEpilogue{}, stream,
                                     workspace.partials);
        else
            launch_fp8_a8_mma<S>(operands, output, LinearIdentityEpilogue{}, stream);
    };
    const int tokens = x.ne[1];
    // GB10 retune (2026-10-29, 48 SMs, 24 MiB L2): argmin over the 12 candidate schedules, bands
    // arbitrated with interleaved ckt ab (6 rounds); cold-L2 medians cited below.
    if (tokens <= 32) return launch.template operator()<Fp8A8T32R32K128>();
    if (tokens <= 64) return launch.template operator()<Fp8A8T64R128K256>();
    if (tokens <= 128) return launch.template operator()<Tma64x128>();
    if (tokens <= 192) return launch.template operator()<Tma192x128>();
    // 193..320: 64x128 beats the former Bulk band (+3.6% at 256; ties at 289).
    if (tokens <= 320) return launch.template operator()<Tma64x128>();
    // 321..384: Bulk stays the argmin at 384 (160 row tiles = 3.33 waves on 48 SMs).
    if (tokens <= 384) return launch.template operator()<Bulk>();
    // 385..512: the narrow TMA tiles beat the former MidBulk band (sweep: 64x128 +5.6% at 448,
    // 192x128.G8 +4.1% at 512).
    if (tokens <= 448) return launch.template operator()<Tma64x128>();
    if (tokens <= 512) return launch.template operator()<Tma192x128G8>();
    // 513..576: 192x128 beats Bulk by 9.6% at 576 (ab).
    if (tokens <= 576) return launch.template operator()<Tma192x128>();
    // 577..768: Bulk argmin at 768.
    if (tokens <= 768) return launch.template operator()<Bulk>();
    // 769..1024: Grouped-8 128x128 beats Bulk by 6.5% at 1024 (ab, SNR 19.8).
    if (tokens <= 1024) return launch.template operator()<Tma128x128G8>();
    // 1025..1792: 192x128 Grouped-8: sweep argmin at 1280, ab +5.4% over Bulk at 1536.
    if (tokens <= 1792) return launch.template operator()<Tma192x128G8>();
    // 1793..2048: plain 192x128 (ab +10.6% over Bulk at 2048; sweep: 1.8% ahead of Grouped-8 there).
    if (tokens <= 2048) return launch.template operator()<Tma192x128>();
    // 2049..4096: Bulk; the Grouped-8 192x128 tile already ties it at 4096 (ab -0.55%, SNR 0.99).
    if (tokens <= 4096) return launch.template operator()<Bulk>();
    // 4097+: the Grouped-8 raster on the 192x128 tile. The fp8 activation slab (3,072 B per token)
    // crosses the 24 MiB L2 near here and Bulk's plain raster thrashes it: ab -35.1% at 6144
    // (SNR 18.8) and -43.9% at 8192 (SNR 23.0); sweep medians at 6144: 6504 us Bulk vs 4601 us G8.
    launch.template operator()<Tma192x128G8>();
}

namespace {

using Tma64x128G8  = Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 2, 1, Fp8MmaRaster::Grouped, 8>;
using Tma64x256G8  = Fp8A8TmaMmaSchedule<64, 256, 128, 2, 4, 2, 1, Fp8MmaRaster::Grouped, 8>;
using BulkG8       = Fp8A8TmaSplitKSchedule<Tma128x128G8, 170, 4, 8>;
using BulkG8Wave48 = Fp8A8TmaSplitKSchedule<Tma128x128G8, 48, 4, 8>;

void launch_n20480_a8_schedule(Fp8GdnInputN20480ScheduleId schedule, const Tensor& x,
                               const Weight& weight, Tensor& qkv, Tensor& z,
                               Fp8A8Workspace workspace, cudaStream_t stream) {
    launch_fp8_a8_quantize(x, weight, workspace, stream);
    const Output output = output_of(qkv, z);
    const auto operands = fp8_a8_operands<float>(weight, workspace, x.ne[1]);
    const auto launch   = [&]<class Schedule>() {
        using S = Fp8ScheduleInstance<Schedule, kHidden>;
        if constexpr (S::kTmaSwizzle)
            launch_fp8_a8_tma_mma<S>(operands, output, LinearIdentityEpilogue{}, stream,
                                     workspace.partials);
        else
            launch_fp8_a8_mma<S>(operands, output, LinearIdentityEpilogue{}, stream);
    };
    switch (schedule) {
    case Fp8GdnInputN20480ScheduleId::A8T32R32K128:
        return launch.template operator()<Fp8A8T32R32K128>();
    case Fp8GdnInputN20480ScheduleId::A8T64R128K256:
        return launch.template operator()<Fp8A8T64R128K256>();
    case Fp8GdnInputN20480ScheduleId::A8Tma64x128:
        return launch.template operator()<Tma64x128>();
    case Fp8GdnInputN20480ScheduleId::A8Tma64x128G8:
        return launch.template operator()<Tma64x128G8>();
    case Fp8GdnInputN20480ScheduleId::A8Tma192x128:
        return launch.template operator()<Tma192x128>();
    case Fp8GdnInputN20480ScheduleId::A8Tma192x128G8:
        return launch.template operator()<Tma192x128G8>();
    case Fp8GdnInputN20480ScheduleId::A8Tma64x256G8:
        return launch.template operator()<Tma64x256G8>();
    case Fp8GdnInputN20480ScheduleId::A8Tma128x128G8:
        return launch.template operator()<Tma128x128G8>();
    case Fp8GdnInputN20480ScheduleId::A8MidBulk:
        return launch.template operator()<MidBulk>();
    case Fp8GdnInputN20480ScheduleId::A8Bulk:
        return launch.template operator()<Bulk>();
    case Fp8GdnInputN20480ScheduleId::A8BulkG8:
        return launch.template operator()<BulkG8>();
    case Fp8GdnInputN20480ScheduleId::A8BulkG8Wave48:
        return launch.template operator()<BulkG8Wave48>();
    }
    throw std::logic_error("fp8 A8 gdn_input_proj n20480: unknown schedule");
}

} // namespace

std::size_t fp8_gdn_input_n20480_candidate_partial_capacity_bytes() {
    // Largest partial plane any forced candidate may need: Bulk's 170-wave 128x256 split-K tiles.
    return Bulk::kPartialBytes;
}

const char* fp8_gdn_input_n20480_schedule_name(Fp8GdnInputN20480ScheduleId schedule) noexcept {
    switch (schedule) {
    case Fp8GdnInputN20480ScheduleId::A8T32R32K128:
        return "gdn_input_proj.fp8.n20480.a8.t32.r32.k128";
    case Fp8GdnInputN20480ScheduleId::A8T64R128K256:
        return "gdn_input_proj.fp8.n20480.a8.t64.r128.k256";
    case Fp8GdnInputN20480ScheduleId::A8Tma64x128:
        return "gdn_input_proj.fp8.n20480.a8.tma64x128";
    case Fp8GdnInputN20480ScheduleId::A8Tma64x128G8:
        return "gdn_input_proj.fp8.n20480.a8.tma64x128.g8";
    case Fp8GdnInputN20480ScheduleId::A8Tma192x128:
        return "gdn_input_proj.fp8.n20480.a8.tma192x128";
    case Fp8GdnInputN20480ScheduleId::A8Tma192x128G8:
        return "gdn_input_proj.fp8.n20480.a8.tma192x128.g8";
    case Fp8GdnInputN20480ScheduleId::A8Tma64x256G8:
        return "gdn_input_proj.fp8.n20480.a8.tma64x256.g8";
    case Fp8GdnInputN20480ScheduleId::A8Tma128x128G8:
        return "gdn_input_proj.fp8.n20480.a8.tma128x128.g8";
    case Fp8GdnInputN20480ScheduleId::A8MidBulk:
        return "gdn_input_proj.fp8.n20480.a8.midbulk";
    case Fp8GdnInputN20480ScheduleId::A8Bulk:
        return "gdn_input_proj.fp8.n20480.a8.bulk";
    case Fp8GdnInputN20480ScheduleId::A8BulkG8:
        return "gdn_input_proj.fp8.n20480.a8.bulk.g8";
    case Fp8GdnInputN20480ScheduleId::A8BulkG8Wave48:
        return "gdn_input_proj.fp8.n20480.a8.bulk.g8.w48";
    }
    return "gdn_input_proj.fp8.n20480.a8.unknown";
}

bool fp8_gdn_input_n20480_candidate_is_legal(Fp8GdnInputN20480ScheduleId schedule,
                                             const Fp8GdnInputProblem& problem) noexcept {
    if (problem.input_rows != 3072 || problem.qkv_rows != 12288 || problem.z_rows != 8192 ||
        problem.parent_rows != 20480 || problem.padded_k != 3072 || problem.cols < 1) {
        return false;
    }
    // Every A8 schedule slices tokens internally and validates only row/K tiles (every row tile
    // divides both the 12288 and 8192 segments), so it is legal at any token count.
    (void)schedule;
    return true;
}

void fp8_gdn_input_n20480_execute_candidate(Fp8GdnInputN20480ScheduleId schedule, const Tensor& x,
                                            const Weight& weight, Tensor& qkv, Tensor& z,
                                            WorkspaceArena& workspace, cudaStream_t stream) {
    const Fp8GdnInputProblem problem{x.ne[0], qkv.ne[0], z.ne[0], weight.n, weight.padded_shape[1],
                                     x.ne[1]};
    if (!fp8_gdn_input_n20480_candidate_is_legal(schedule, problem)) {
        throw std::invalid_argument(
            "fp8 A8 gdn_input_proj n20480: candidate schedule is not legal for problem");
    }
    auto scope = workspace.scope();
    const Fp8A8Workspace scratch = allocate_fp8_a8_workspace(
        workspace, x.ne[1], weight.k, fp8_gdn_input_n20480_candidate_partial_capacity_bytes());
    launch_n20480_a8_schedule(schedule, x, weight, qkv, z, scratch, stream);
}

} // namespace infernix::ops::detail
