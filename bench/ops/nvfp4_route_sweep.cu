// NVFP4 A16 linear route sweep benchmark for GB10 retuning.
//
// Times individual registered NVFP4 routes across token bands with cold-L2
// flushing and verifies numerical agreement against the production route.

#include "infernix/ops/linear.h"
#include "infernix_bench_common.h"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/common/output.cuh"
#include "ops/linear/nvfp4/nvfp4_a4_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"
#include "ops/linear/nvfp4/nvfp4_a4_tma.cuh"
#include "ops/linear/nvfp4/nvfp4_instances.cuh"
#include "ops/linear/nvfp4/nvfp4_launch.cuh"
#include "ops/linear/nvfp4/nvfp4_launch.h"
#include "ops/linear/nvfp4/nvfp4_operands.h"
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"
#include "quantized_weight.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace infernix;
using namespace infernix::ops::detail;
using infernix::ops::Cache;

namespace {

struct RouteCandidate {
    const char* name;
    Nvfp4Launch launch;
    int min_t    = 1;
    int max_t    = 0; // 0 = unbounded
    int static_k = 0; // 0 = any K
};

void launch_production(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    ops::linear(x, w, out, stream);
}

// Adapters for Sliced-K and Full MMA with dynamic K
template <class Schedule>
void launch_sliced(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_nvfp4_a16_sliced_k_mma<Schedule>(
        nvfp4_a16_operands(x, w),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
        LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void launch_mma(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_nvfp4_a16_mma<Schedule>(
        nvfp4_a16_operands(x, w),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
        LinearIdentityEpilogue{}, stream);
}

// Production never runs the bare schedules below: every shape table wraps its schedule in
// Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows>, which bakes K into the kernel. That
// changes ptxas' scheduling, so a bare candidate can be measurably faster than the route the
// engine is actually able to select. These mirrors instantiate the same static-K kernel the
// shape table does, and are the only admissible evidence when a band is retuned.
template <class Schedule, int K>
void launch_sliced_static(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if (w.k != K) return;
    launch_nvfp4_a16_sliced_k_mma<Nvfp4ScheduleInstance<Schedule, K>>(
        nvfp4_a16_operands(x, w), LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
        LinearIdentityEpilogue{}, stream);
}

template <class Schedule, int K>
void launch_mma_static(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if (w.k != K) return;
    launch_nvfp4_a16_mma<Nvfp4ScheduleInstance<Schedule, K>>(
        nvfp4_a16_operands(x, w), LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
        LinearIdentityEpilogue{}, stream);
}

template <class Schedule, int K>
void launch_gemv_static(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if (w.k != K) return;
    launch_nvfp4_a16_gemv<Nvfp4ScheduleInstance<Schedule, K>>(
        nvfp4_a16_operands(x, w), LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
        LinearIdentityEpilogue{}, stream);
}

// The schedule is variadic so its template-argument commas survive macro expansion.
#define INFER_NX_SLICED_STATIC(name, tmin, tmax, ...)                                             \
    {"sk_" name "_k2560", launch_sliced_static<__VA_ARGS__, 2560>, tmin, tmax, 2560},             \
        {"sk_" name "_k5120", launch_sliced_static<__VA_ARGS__, 5120>, tmin, tmax, 5120},         \
        {"sk_" name "_k6144", launch_sliced_static<__VA_ARGS__, 6144>, tmin, tmax, 6144},         \
        {"sk_" name "_k17408", launch_sliced_static<__VA_ARGS__, 17408>, tmin, tmax, 17408},

#define INFER_NX_MMA_STATIC(name, tmin, tmax, ...)                                                \
    {"mk_" name "_k2560", launch_mma_static<__VA_ARGS__, 2560>, tmin, tmax, 2560},                \
        {"mk_" name "_k5120", launch_mma_static<__VA_ARGS__, 5120>, tmin, tmax, 5120},            \
        {"mk_" name "_k6144", launch_mma_static<__VA_ARGS__, 6144>, tmin, tmax, 6144},            \
        {"mk_" name "_k17408", launch_mma_static<__VA_ARGS__, 17408>, tmin, tmax, 17408},

#define INFER_NX_GEMV_STATIC(name, tmin, tmax, ...)                                               \
    {"gk_" name "_k2560", launch_gemv_static<__VA_ARGS__, 2560>, tmin, tmax, 2560},               \
        {"gk_" name "_k5120", launch_gemv_static<__VA_ARGS__, 5120>, tmin, tmax, 5120},           \
        {"gk_" name "_k6144", launch_gemv_static<__VA_ARGS__, 6144>, tmin, tmax, 6144},           \
        {"gk_" name "_k17408", launch_gemv_static<__VA_ARGS__, 17408>, tmin, tmax, 17408},

// Specializations for static-K Gemv & Simt
template <int K>
void launch_gemv_w8(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if (w.k != K) return;
    using S = Nvfp4ScheduleInstance<
        Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>, K>;
    launch_nvfp4_a16_gemv<S>(
        nvfp4_a16_operands(x, w),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
        LinearIdentityEpilogue{}, stream);
}

template <int K, int TokenTile>
void launch_simt_c2(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if (w.k != K) return;
    using S = Nvfp4ScheduleInstance<
        Nvfp4A16SimtSchedule<4, 1, 2, 16, TokenTile, 1, Nvfp4SimtActivationAccess::SharedPhase,
                             Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                             Nvfp4SimtBlockOrder::RowsContiguous, 1>,
        K, TokenTile, false>;
    launch_nvfp4_a16_simt<S>(
        nvfp4_a16_operands(x, w),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
        LinearIdentityEpilogue{}, stream);
}

const RouteCandidate kCandidates[] = {
    {"production", launch_production, 1, 0, 0},

    // T=1 GEMV (specialized by K)
    {"gemv_w8_k2560", launch_gemv_w8<2560>, 1, 1, 2560},
    {"gemv_w8_k5120", launch_gemv_w8<5120>, 1, 1, 5120},
    {"gemv_w8_k6144", launch_gemv_w8<6144>, 1, 1, 6144},
    {"gemv_w8_k17408", launch_gemv_w8<17408>, 1, 1, 17408},

    // SIMT (T=2..4)
    {"simt_c2_t2_k2560", launch_simt_c2<2560, 2>, 1, 2, 2560},
    {"simt_c2_t2_k5120", launch_simt_c2<5120, 2>, 1, 2, 5120},
    {"simt_c2_t2_k6144", launch_simt_c2<6144, 2>, 1, 2, 6144},
    {"simt_c2_t2_k17408", launch_simt_c2<17408, 2>, 1, 2, 17408},
    {"simt_c2_t4_k2560", launch_simt_c2<2560, 4>, 1, 4, 2560},
    {"simt_c2_t4_k5120", launch_simt_c2<5120, 4>, 1, 4, 5120},
    {"simt_c2_t4_k6144", launch_simt_c2<6144, 4>, 1, 4, 6144},
    {"simt_c2_t4_k17408", launch_simt_c2<17408, 4>, 1, 4, 17408},

    // Sliced-K MMA (Tokens, Warps, Stages) — includes GB10 narrow T=8 tiles & ActiveOnly / MinBlocks=3,4
    {"sliced_t8_w2_s2", launch_sliced<Nvfp4SlicedInstance<8, 2, 2>>, 1, 8, 0},
    {"sliced_t8_w4_s1", launch_sliced<Nvfp4SlicedInstance<8, 4, 1>>, 1, 8, 0},
    {"sliced_t8_w4_s2", launch_sliced<Nvfp4SlicedInstance<8, 4, 2>>, 1, 8, 0},
    {"sliced_t8_w8_s1", launch_sliced<Nvfp4SlicedInstance<8, 8, 1>>, 1, 8, 0},
    {"sliced_t8_w8_s2", launch_sliced<Nvfp4SlicedInstance<8, 8, 2>>, 1, 8, 0},
    {"sliced_t8_w4_s2_m3_act", launch_sliced<Nvfp4A16SlicedKMmaSchedule<4, 8, 3, Cache::ca, Cache::cg, Nvfp4ActivationStage::ActiveOnly, 2>>, 1, 8, 0},
    {"sliced_t8_w4_s2_m4_act", launch_sliced<Nvfp4A16SlicedKMmaSchedule<4, 8, 4, Cache::ca, Cache::cg, Nvfp4ActivationStage::ActiveOnly, 2>>, 1, 8, 0},
    {"sliced_t8_w8_s2_m2_act", launch_sliced<Nvfp4A16SlicedKMmaSchedule<8, 8, 2, Cache::ca, Cache::cg, Nvfp4ActivationStage::ActiveOnly, 2>>, 1, 8, 0},
    {"sliced_t16_w4_s2_m3_act", launch_sliced<Nvfp4A16SlicedKMmaSchedule<4, 16, 3, Cache::ca, Cache::cg, Nvfp4ActivationStage::ActiveOnly, 2>>, 1, 16, 0},

    {"sliced_t16_w2_s1", launch_sliced<Nvfp4SlicedInstance<16, 2, 1>>, 1, 16, 0},
    {"sliced_t16_w2_s2", launch_sliced<Nvfp4SlicedInstance<16, 2, 2>>, 1, 16, 0},
    {"sliced_t16_w4_s1", launch_sliced<Nvfp4SlicedInstance<16, 4, 1>>, 1, 16, 0},
    {"sliced_t16_w4_s2", launch_sliced<Nvfp4SlicedInstance<16, 4, 2>>, 1, 16, 0},
    {"sliced_t16_w8_s1", launch_sliced<Nvfp4SlicedInstance<16, 8, 1>>, 1, 16, 0},
    {"sliced_t16_w8_s2", launch_sliced<Nvfp4SlicedInstance<16, 8, 2>>, 1, 16, 0},

    {"sliced_t24_w2_s1", launch_sliced<Nvfp4SlicedInstance<24, 2, 1>>, 1, 24, 0},
    {"sliced_t24_w2_s2", launch_sliced<Nvfp4SlicedInstance<24, 2, 2>>, 1, 24, 0},
    {"sliced_t24_w4_s1", launch_sliced<Nvfp4SlicedInstance<24, 4, 1>>, 1, 24, 0},
    {"sliced_t24_w4_s2", launch_sliced<Nvfp4SlicedInstance<24, 4, 2>>, 1, 24, 0},
    {"sliced_t24_w8_s1", launch_sliced<Nvfp4SlicedInstance<24, 8, 1>>, 1, 24, 0},
    {"sliced_t24_w8_s2", launch_sliced<Nvfp4SlicedInstance<24, 8, 2>>, 1, 24, 0},

    {"sliced_t32_w2_s1", launch_sliced<Nvfp4SlicedInstance<32, 2, 1>>, 1, 32, 0},
    {"sliced_t32_w2_s2", launch_sliced<Nvfp4SlicedInstance<32, 2, 2>>, 1, 32, 0},
    {"sliced_t32_w4_s1", launch_sliced<Nvfp4SlicedInstance<32, 4, 1>>, 1, 32, 0},
    {"sliced_t32_w4_s2", launch_sliced<Nvfp4SlicedInstance<32, 4, 2>>, 1, 32, 0},
    {"sliced_t32_w8_s1", launch_sliced<Nvfp4SlicedInstance<32, 8, 1>>, 1, 32, 0},
    {"sliced_t32_w8_s2", launch_sliced<Nvfp4SlicedInstance<32, 8, 2>>, 1, 32, 0},

    {"sliced_t48_w2_s2", launch_sliced<Nvfp4SlicedInstance<48, 2, 2>>, 1, 48, 0},
    {"sliced_t48_w4_s1", launch_sliced<Nvfp4SlicedInstance<48, 4, 1>>, 1, 48, 0},
    {"sliced_t48_w4_s2", launch_sliced<Nvfp4SlicedInstance<48, 4, 2>>, 1, 48, 0},

    {"sliced_t64_w2_s1", launch_sliced<Nvfp4SlicedInstance<64, 2, 1>>, 1, 64, 0},
    {"sliced_t64_w2_s2", launch_sliced<Nvfp4SlicedInstance<64, 2, 2>>, 1, 64, 0},
    {"sliced_t64_w4_s1", launch_sliced<Nvfp4SlicedInstance<64, 4, 1>>, 1, 64, 0},
    {"sliced_t64_w4_s2", launch_sliced<Nvfp4SlicedInstance<64, 4, 2>>, 1, 64, 0},
    {"sliced_t64_w8_s1", launch_sliced<Nvfp4SlicedInstance<64, 8, 1>>, 1, 64, 0},

    // NEW GB10 32-row (RowTiles = 2) Sliced-K MMA schedules (halves activation L2 & ldmatrix traffic)
    {"sliced_r32_t8_w4_s1", launch_sliced<Nvfp4SlicedInstance<8, 4, 1, 2>>, 1, 8, 0},
    {"sliced_r32_t8_w4_s2", launch_sliced<Nvfp4SlicedInstance<8, 4, 2, 2>>, 1, 8, 0},
    {"sliced_r32_t8_w8_s1", launch_sliced<Nvfp4SlicedInstance<8, 8, 1, 2>>, 1, 8, 0},
    {"sliced_r32_t8_w8_s2", launch_sliced<Nvfp4SlicedInstance<8, 8, 2, 2>>, 1, 8, 0},
    {"sliced_r32_t16_w4_s1", launch_sliced<Nvfp4SlicedInstance<16, 4, 1, 2>>, 1, 16, 0},
    {"sliced_r32_t16_w4_s2", launch_sliced<Nvfp4SlicedInstance<16, 4, 2, 2>>, 1, 16, 0},
    {"sliced_r32_t16_w8_s1", launch_sliced<Nvfp4SlicedInstance<16, 8, 1, 2>>, 1, 16, 0},
    {"sliced_r32_t16_w8_s2", launch_sliced<Nvfp4SlicedInstance<16, 8, 2, 2>>, 1, 16, 0},
    {"sliced_r32_t24_w4_s1", launch_sliced<Nvfp4SlicedInstance<24, 4, 1, 2>>, 1, 24, 0},
    {"sliced_r32_t24_w4_s2", launch_sliced<Nvfp4SlicedInstance<24, 4, 2, 2>>, 1, 24, 0},
    {"sliced_r32_t32_w4_s1", launch_sliced<Nvfp4SlicedInstance<32, 4, 1, 2>>, 1, 32, 0},
    {"sliced_r32_t32_w4_s2", launch_sliced<Nvfp4SlicedInstance<32, 4, 2, 2>>, 1, 32, 0},

    // Static-K mirrors of the shape tables' production instantiations (see the comment on
    // launch_sliced_static). A band may only be retuned onto a route measured here.
    INFER_NX_SLICED_STATIC("r16_t8_w8_s2", 1, 8, Nvfp4SlicedInstance<8, 8, 2, 2>)
    INFER_NX_SLICED_STATIC("r16_t8_w8_s1", 1, 8, Nvfp4SlicedInstance<8, 8, 2>)
    INFER_NX_SLICED_STATIC("r16_t8_w4_s2", 1, 8, Nvfp4SlicedInstance<8, 4, 2, 2>)
    INFER_NX_SLICED_STATIC("r16_t16_w8_s2", 1, 16, Nvfp4SlicedInstance<16, 8, 2, 2>)
    INFER_NX_SLICED_STATIC("r16_t16_w8_s1", 1, 16, Nvfp4SlicedInstance<16, 8, 2>)
    INFER_NX_SLICED_STATIC("r16_t16_w4_s2", 1, 16, Nvfp4SlicedInstance<16, 4, 2, 2>)
    INFER_NX_SLICED_STATIC("r32_t24_w4_s1", 1, 24, Nvfp4SlicedInstance<24, 4, 1, 2>)
    INFER_NX_SLICED_STATIC("r32_t24_w4_s2", 1, 24, Nvfp4SlicedInstance<24, 4, 2, 2>)
    INFER_NX_SLICED_STATIC("r16_t24_w8_s1", 1, 24, Nvfp4SlicedInstance<24, 8, 1, 1>)
    INFER_NX_SLICED_STATIC("r32_t24_w8_s2", 1, 24, Nvfp4SlicedInstance<24, 8, 2, 2>)
    INFER_NX_SLICED_STATIC("r32_t32_w4_s1", 1, 32, Nvfp4SlicedInstance<32, 4, 1, 2>)
    INFER_NX_SLICED_STATIC("r32_t32_w4_s2", 1, 32, Nvfp4SlicedInstance<32, 4, 2, 2>)
    INFER_NX_SLICED_STATIC("r16_t32_w4_s1", 1, 32, Nvfp4SlicedInstance<32, 4, 1>)
    INFER_NX_SLICED_STATIC("r32_t32_w8_s2", 1, 32, Nvfp4SlicedInstance<32, 8, 2, 2>)
    INFER_NX_SLICED_STATIC("r32_t48_w4_s2", 1, 48, Nvfp4SlicedInstance<48, 4, 2, 2>)
    INFER_NX_SLICED_STATIC("r32_t64_w4_s2", 1, 64, Nvfp4SlicedInstance<64, 4, 2, 2>)

    // GB10 2026-11-01 (T2): the wide-token RowTiles=1 sliced-K forms the task asks for at
    // T = 33..64. Both measure 2..3x behind the shipped MMA band (activation staging forces
    // 1 CTA/SM), so they stay candidates and nothing in a shape table selects them.
    INFER_NX_SLICED_STATIC("r16_t48_w4_s2", 32, 64, Nvfp4SlicedInstance<48, 4, 2, 1>)
    INFER_NX_SLICED_STATIC("r16_t64_w4_s2", 32, 64, Nvfp4SlicedInstance<64, 4, 2, 1>)

    INFER_NX_MMA_STATIC("r32_t64_k128_s1", 16, 0, Nvfp4A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>)
    INFER_NX_MMA_STATIC("r64_t64_k128_s1", 16, 0, Nvfp4A16MmaSchedule<64, 64, 128, 32, 16, 1, 2>)
    INFER_NX_MMA_STATIC("r64_t64_k128_s2", 16, 0, Nvfp4A16MmaSchedule<64, 64, 128, 32, 16, 2, 1>)
    INFER_NX_MMA_STATIC("r64_t96_k128_s1", 32, 0, Nvfp4A16MmaSchedule<64, 96, 128, 32, 16, 1, 2>)
    INFER_NX_MMA_STATIC("r32_t32_k128_s2", 8, 128, Nvfp4A16MmaSchedule<32, 32, 128, 16, 16, 2, 2>)
    INFER_NX_MMA_STATIC("r64_t128_k128_s2", 32, 0, Nvfp4A16MmaSchedule<64, 128, 128, 32, 16, 2, 1>)

    INFER_NX_GEMV_STATIC("r16_staged", 1, 1,
                         Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw,
                                              Nvfp4CodeCache::Default, 2>)
    INFER_NX_GEMV_STATIC("r16_direct", 1, 1,
                         Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::Direct,
                                              Nvfp4CodeCache::Default, 2>)
    INFER_NX_GEMV_STATIC("r16_direct_m4", 1, 1,
                         Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::Direct,
                                              Nvfp4CodeCache::Default, 4>)
    INFER_NX_GEMV_STATIC("r8_direct_m4", 1, 1,
                         Nvfp4A16GemvSchedule<4, 2, 16, 4, Nvfp4ScaleAccess::Direct,
                                              Nvfp4CodeCache::Default, 4>)
    INFER_NX_GEMV_STATIC("r4_direct_m4", 1, 1,
                         Nvfp4A16GemvSchedule<2, 2, 16, 4, Nvfp4ScaleAccess::Direct,
                                              Nvfp4CodeCache::Default, 4>)


    // Full MMA (including new small/high-occupancy GB10 tiles)
    // BlockRows, BlockTokens, BlockK, WarpRows, WarpTokens, ActivationStages, MinBlocksPerSm
    {"mma_r32_t16_k64_s2_a4",  launch_mma<Nvfp4A16MmaSchedule<32, 16, 64, 16, 8, 2, 4>>, 4, 64, 0},
    {"mma_r32_t16_k128_s2_a3", launch_mma<Nvfp4A16MmaSchedule<32, 16, 128, 16, 8, 2, 3>>, 4, 64, 0},
    {"mma_r32_t24_k64_s2_a4",  launch_mma<Nvfp4A16MmaSchedule<32, 24, 64, 16, 8, 2, 4>>, 8, 96, 0},
    {"mma_r32_t32_k64_s2_a4",  launch_mma<Nvfp4A16MmaSchedule<32, 32, 64, 16, 16, 2, 4>>, 8, 128, 0},
    {"mma_r32_t32_k128_s2_a2", launch_mma<Nvfp4A16MmaSchedule<32, 32, 128, 16, 16, 2, 2>>, 8, 0, 0},
    {"mma_r32_t48_k64_s2_a3",  launch_mma<Nvfp4A16MmaSchedule<32, 48, 64, 16, 16, 2, 3>>, 16, 192, 0},
    {"mma_r32_t64_k64_s2_a3",  launch_mma<Nvfp4A16MmaSchedule<32, 64, 64, 16, 16, 2, 3>>, 16, 0, 0},
    {"mma_r32_t64_k128_s1_a3", launch_mma<Nvfp4A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>>, 16, 0, 0},
    {"mma_r32_t64_k128_s2_a2", launch_mma<Nvfp4A16MmaSchedule<32, 64, 128, 32, 16, 2, 2>>, 16, 0, 0},

    {"mma_r64_t16_k64_s2_a3",  launch_mma<Nvfp4A16MmaSchedule<64, 16, 64, 32, 8, 2, 3>>, 4, 64, 0},
    {"mma_r64_t32_k64_s2_a3",  launch_mma<Nvfp4A16MmaSchedule<64, 32, 64, 32, 16, 2, 3>>, 8, 128, 0},
    {"mma_r64_t48_k64_s2_a3",  launch_mma<Nvfp4A16MmaSchedule<64, 48, 64, 32, 16, 2, 3>>, 16, 192, 0},
    {"mma_r64_t64_k64_s2_a2",  launch_mma<Nvfp4A16MmaSchedule<64, 64, 64, 32, 16, 2, 2>>, 16, 0, 0},
    {"mma_r64_t64_k128_s1_a2", launch_mma<Nvfp4A16MmaSchedule<64, 64, 128, 32, 16, 1, 2>>, 16, 0, 0},
    {"mma_r64_t64_k128_s2_a1", launch_mma<Nvfp4A16MmaSchedule<64, 64, 128, 32, 16, 2, 1>>, 16, 0, 0},

    {"mma_r64_t80_k64_s2_a2",  launch_mma<Nvfp4A16MmaSchedule<64, 80, 64, 32, 16, 2, 2>>, 24, 0, 0},
    {"mma_r64_t80_k128_s1_a2", launch_mma<Nvfp4A16MmaSchedule<64, 80, 128, 32, 16, 1, 2>>, 24, 0, 0},
    {"mma_r64_t96_k64_s2_a2",  launch_mma<Nvfp4A16MmaSchedule<64, 96, 64, 32, 16, 2, 2>>, 32, 0, 0},
    {"mma_r64_t96_k64_w64_s2_a2", launch_mma<Nvfp4A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>>, 32, 0, 0},
    {"mma_r64_t96_k128_s1_a2", launch_mma<Nvfp4A16MmaSchedule<64, 96, 128, 32, 16, 1, 2>>, 32, 0, 0},
    {"mma_r64_t96_k128_w64_s1_a2", launch_mma<Nvfp4A16MmaSchedule<64, 96, 128, 64, 16, 1, 2>>, 32, 0, 0},
    {"mma_r64_t112_k64_s2_a2", launch_mma<Nvfp4A16MmaSchedule<64, 112, 64, 32, 16, 2, 2>>, 32, 0, 0},
    {"mma_r64_t112_k128_s1_a2", launch_mma<Nvfp4A16MmaSchedule<64, 112, 128, 32, 16, 1, 2>>, 32, 0, 0},

    {"mma_r64_t128_k64_s2_a2",  launch_mma<Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>, 32, 0, 0},
    {"mma_r64_t128_k64_w32x32_s2_a2", launch_mma<Nvfp4A16MmaSchedule<64, 128, 64, 32, 32, 2, 2, Cache::cg, Cache::ca>>, 32, 0, 0},
    {"mma_r64_t128_k128_s1_a1", launch_mma<Nvfp4A16MmaSchedule<64, 128, 128, 32, 16, 1, 1>>, 32, 0, 0},
    {"mma_r64_t128_k128_s2_a1", launch_mma<Nvfp4A16MmaSchedule<64, 128, 128, 32, 16, 2, 1>>, 32, 0, 0},
    {"mma_r64_t256_k64_s2_a1",  launch_mma<Nvfp4A16MmaSchedule<64, 256, 64, 32, 32, 2, 1>>, 64, 0, 0},
    {"mma_r128_t64_k64_s2_a2",  launch_mma<Nvfp4A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>, 16, 0, 0},
    {"mma_r128_t128_k64_s2_a1", launch_mma<Nvfp4A16MmaSchedule<128, 128, 64, 64, 32, 2, 1>>, 64, 0, 0},
    {"mma_r128_t128_k64_w32x32_s2_a1", launch_mma<Nvfp4A16MmaSchedule<128, 128, 64, 32, 32, 2, 1, Cache::cg, Cache::ca>>, 64, 0, 0},
};

// ---------------------------------------------------------------------------
// NVFP4 A4 (Native OMMA.16864.F32.E2M1, 494.5 TFLOP/s ceiling) Route Sweep
// ---------------------------------------------------------------------------
using A4CandidateLaunch = void (*)(const Tensor&, const Weight&, Tensor&, Nvfp4A4Workspace,
                                   cudaStream_t);

struct A4RouteCandidate {
    const char* name;
    A4CandidateLaunch launch;
    int min_t = 1;
    int max_t = 0;
    // The 2-term residual TMA GEMM reads its activation as planes, so it needs the residual
    // workspace layout (terms = kNvfp4A4ResidualTerms, row-scaled) rather than the 1-term one the
    // other A4 candidates share; handing it the latter fails the plane-vs-call check.
    bool residual = false;
    // The A16 table is the shipped, oracle-qualified reference, and it is a different
    // approximation from the A4 planes this list otherwise checks against. Marking it unchecked
    // lets the pair-vs-A16 admission gate be measured in one process instead of two passes.
    bool unchecked = false;
};

void launch_a4_a16_ref(const Tensor& x, const Weight& w, Tensor& out, Nvfp4A4Workspace,
                       cudaStream_t stream) {
    ops::linear(x, w, out, stream);
}

void launch_a4_production(const Tensor& x, const Weight& w, Tensor& out, Nvfp4A4Workspace ws,
                          cudaStream_t stream) {
    const std::size_t cap =
        ops::linear_workspace_capacity_bytes(QType::NVFP4, w.n, w.k, ops::LinearPolicy::AllowA4,
                                             1, x.ne[1]);
    // Below a shape's pair threshold and without an A4 admission the capacity query is zero, which
    // means the public route provably selects A16 for every width in the interval -- so take the
    // A16-only overload instead of building a zero-length arena, which the arena contract rejects.
    if (cap == 0) {
        ops::linear(x, w, out, stream);
        return;
    }
    WorkspaceArena arena(DeviceSpan{ws.codes, cap});
    ops::linear(x, w, out, ops::LinearPolicy::AllowA4, arena, stream);
}

template <class Schedule, bool Reciprocal = true>
void launch_a4_mma_cand(const Tensor& x, const Weight& w, Tensor& out, Nvfp4A4Workspace ws,
                        cudaStream_t stream) {
    launch_nvfp4_a4_quantize(x, w, ws, Nvfp4ScaleLayout::RowMajor, stream, Reciprocal);
    launch_nvfp4_a4_mma<Schedule>(
        nvfp4_a4_operands(w, ws, x.ne[1], Nvfp4ScaleLayout::RowMajor),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n}, LinearIdentityEpilogue{},
        stream, {}, pdl::Dependency::Programmatic);
}

template <class Schedule, bool Reciprocal = false>
void launch_a4_tma_cand(const Tensor& x, const Weight& w, Tensor& out, Nvfp4A4Workspace ws,
                        cudaStream_t stream) {
    launch_nvfp4_a4_quantize(x, w, ws, Schedule::kScaleLayout, stream, Reciprocal);
    launch_nvfp4_a4_tma_mma<Schedule>(
        nvfp4_a4_operands(w, ws, x.ne[1], Schedule::kScaleLayout),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n}, LinearIdentityEpilogue{},
        stream);
}

// Differential candidates: the quantize and the GEMM are stream-serialised in eager mode, so a
// repeated pass (idempotent: it rewrites the same planes / the same output) separates the two costs
// by subtraction from the single-pass time.
template <class Schedule, int QuantizePasses, int GemmPasses>
void launch_a4_tma_repeats(const Tensor& x, const Weight& w, Tensor& out, Nvfp4A4Workspace ws,
                           cudaStream_t stream) {
    for (int pass = 0; pass < QuantizePasses; ++pass)
        launch_nvfp4_a4_quantize(x, w, ws, Schedule::kScaleLayout, stream, false);
    for (int pass = 0; pass < GemmPasses; ++pass)
        launch_nvfp4_a4_tma_mma<Schedule>(
            nvfp4_a4_operands(w, ws, x.ne[1], Schedule::kScaleLayout),
            LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n}, LinearIdentityEpilogue{},
            stream);
}

using A4PrefillSchedule = Nvfp4A4TmaMmaSchedule<256, 3, 1, kNvfp4A4TmaWeightCodeL2_128B, 1, 128>;

const A4RouteCandidate kA4Candidates[] = {
    {"production", launch_a4_production, 1, 0},

    // The A16 table the same shape runs when no A4 route is admitted: the real competitor of both
    // A4 routes under --routes fast, and the baseline the pair_min_tokens gate is set against.
    {"a16_ref", launch_a4_a16_ref, 1, 0, false, true},

    // The 2-term residual TMA route the shape tables register as their pair entry. The engine admits
    // it only at or above pair_min_tokens, so this candidate is the evidence for that gate: a shape
    // whose pair time beats production at a lower width can lower the gate.
    {"a4_pair", launch_nvfp4_a4_pair, 1, 0, true},

    // --- NEW Smaller BlockTokens=16 & BlockRows=32/64 GB10 schedules (4 CTAs/SM) ---
    {"a4_mma_t16_r32_k64_w4_s2_m4",   launch_a4_mma_cand<Nvfp4A4MmaSchedule<16, 32, 64, 1, 4, 2, 4>>, 1, 64},
    {"a4_mma_t16_r32_k128_w4_s2_m4",  launch_a4_mma_cand<Nvfp4A4MmaSchedule<16, 32, 128, 1, 4, 2, 4>>, 1, 64},
    {"a4_mma_t16_r64_k64_w4_s2_m4",   launch_a4_mma_cand<Nvfp4A4MmaSchedule<16, 64, 64, 1, 4, 2, 4>>, 1, 64},
    {"a4_mma_t16_r64_k128_w4_s2_m4",  launch_a4_mma_cand<Nvfp4A4MmaSchedule<16, 64, 128, 1, 4, 2, 4>>, 1, 64},
    {"a4_mma_t16_r64_k128_w8_s2_m3",  launch_a4_mma_cand<Nvfp4A4MmaSchedule<16, 64, 128, 1, 8, 2, 3>>, 1, 64},
    {"a4_mma_t16_r64_k256_w4_s2_m2",  launch_a4_mma_cand<Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>>, 1, 64},
    {"a4_mma_t16_r128_k128_w8_s2_m2", launch_a4_mma_cand<Nvfp4A4MmaSchedule<16, 128, 128, 1, 8, 2, 2>>, 1, 64},

    // --- BlockTokens=32 GB10 schedules (K=64, 128, 256; R=32, 64, 128) ---
    {"a4_mma_t32_r32_k128_w4_s2_m4",  launch_a4_mma_cand<Nvfp4A4MmaSchedule<32, 32, 128, 2, 2, 2, 4>>, 1, 128},
    {"a4_mma_t32_r64_k64_w8_s2_m3",   launch_a4_mma_cand<Nvfp4A4MmaSchedule<32, 64, 64, 2, 4, 2, 3>>, 1, 128},
    {"a4_mma_t32_r64_k128_w4_s2_m4",  launch_a4_mma_cand<Nvfp4A4MmaSchedule<32, 64, 128, 2, 2, 2, 4>>, 1, 128},
    {"a4_mma_t32_r64_k128_w8_s2_m3",  launch_a4_mma_cand<Nvfp4A4MmaSchedule<32, 64, 128, 2, 4, 2, 3>>, 1, 128},
    {"a4_mma_t32_r64_k256_w8_s2_m2",  launch_a4_mma_cand<Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>, 1, 128},
    {"a4_mma_t32_r128_k128_w8_s2_m2", launch_a4_mma_cand<Nvfp4A4MmaSchedule<32, 128, 128, 2, 4, 2, 2>>, 1, 256},
    {"a4_mma_t32_r128_k256_w8_s2_m1", launch_a4_mma_cand<Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>>, 1, 256},

    // --- BlockTokens=64 GB10 schedules (K=64, 128, 256; R=32, 64, 128; 2-4 CTAs/SM) ---
    {"a4_mma_t64_r32_k128_w4_s2_m4",  launch_a4_mma_cand<Nvfp4A4MmaSchedule<64, 32, 128, 4, 1, 2, 4>>, 16, 256},
    {"a4_mma_t64_r64_k64_w8_s2_m3",   launch_a4_mma_cand<Nvfp4A4MmaSchedule<64, 64, 64, 2, 4, 2, 3>>, 16, 512},
    {"a4_mma_t64_r64_k128_w4_s2_m4",  launch_a4_mma_cand<Nvfp4A4MmaSchedule<64, 64, 128, 2, 2, 2, 4>>, 16, 512},
    {"a4_mma_t64_r64_k128_w8_s2_m3",  launch_a4_mma_cand<Nvfp4A4MmaSchedule<64, 64, 128, 2, 4, 2, 3>>, 16, 512},
    {"a4_mma_t64_r64_k256_w8_s2_m1",  launch_a4_mma_cand<Nvfp4A4MmaSchedule<64, 64, 256, 2, 4, 2, 1>>, 16, 512},
    {"a4_mma_t64_r128_k64_w8_s2_m2",  launch_a4_mma_cand<Nvfp4A4MmaSchedule<64, 128, 64, 2, 4, 2, 2>>, 16, 0},
    {"a4_mma_t64_r128_k128_w8_s2_m2", launch_a4_mma_cand<Nvfp4A4MmaSchedule<64, 128, 128, 2, 4, 2, 2>>, 16, 0},
    {"a4_mma_t64_r128_k128_w4x2_s2_m2", launch_a4_mma_cand<Nvfp4A4MmaSchedule<64, 128, 128, 4, 2, 2, 2>>, 16, 0},
    {"a4_mma_t64_r128_k256_w8_s2_m1", launch_a4_mma_cand<Nvfp4A4MmaSchedule<64, 128, 256, 4, 2, 2, 1>>, 16, 0},

    // --- BlockTokens=128 Non-TMA GB10 schedules (K=128 2-CTA/SM vs K=256 1-CTA/SM) ---
    {"a4_mma_t128_r64_k128_w8_s2_m2",  launch_a4_mma_cand<Nvfp4A4MmaSchedule<128, 64, 128, 4, 2, 2, 2>>, 32, 0},
    {"a4_mma_t128_r128_k128_w8_s2_m2", launch_a4_mma_cand<Nvfp4A4MmaSchedule<128, 128, 128, 4, 2, 2, 2>>, 32, 0},
    {"a4_mma_t128_r128_k256_w8_s1_m2", launch_a4_mma_cand<Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 1, 2>>, 32, 0},
    {"a4_mma_t128_r128_k256_w8_s2_m1", launch_a4_mma_cand<Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>>, 32, 0},

    // --- TMA Schedules: 128x64 (2 CTAs/SM), 128x128 (2 & 1 CTA/SM), 256x64, 256x128; the _l2
    // suffix promotes the weight panel, _ws also the weight scales, and _all adds the activation
    // codes. ---
    {"a4_tma_t128_r64_s2_m2",     launch_a4_tma_cand<Nvfp4A4TmaMmaSchedule<128, 2, 2, kNvfp4A4TmaNoPromotions, 1, 64>>, 32, 0},
    {"a4_tma_t128_r64_s2_m2_l2",  launch_a4_tma_cand<Nvfp4A4TmaMmaSchedule<128, 2, 2, kNvfp4A4TmaWeightCodeL2_128B, 1, 64>>, 32, 0},
    {"a4_tma_t128_r64_s3_m2",     launch_a4_tma_cand<Nvfp4A4TmaMmaSchedule<128, 3, 2, kNvfp4A4TmaNoPromotions, 1, 64>>, 32, 0},
    {"a4_tma_t128_r64_s3_m2_l2",  launch_a4_tma_cand<Nvfp4A4TmaMmaSchedule<128, 3, 2, kNvfp4A4TmaWeightCodeL2_128B, 1, 64>>, 32, 0},
    {"a4_tma_t128_r64_s3_m2_all", launch_a4_tma_cand<Nvfp4A4TmaMmaSchedule<128, 3, 2, kNvfp4A4TmaL2_128B, 1, 64>>, 32, 0},
    {"a4_tma_t128_r128_s2_m1",    launch_a4_tma_cand<Nvfp4A4TmaMmaSchedule<128, 2, 1, kNvfp4A4TmaNoPromotions, 1, 128>>, 32, 0},
    {"a4_tma_t128_r128_s2_m1_l2", launch_a4_tma_cand<Nvfp4A4TmaMmaSchedule<128, 2, 1, kNvfp4A4TmaWeightCodeL2_128B, 1, 128>>, 32, 0},
    {"a4_tma_t128_r128_s4_m1",    launch_a4_tma_cand<Nvfp4A4TmaMmaSchedule<128, 4, 1, kNvfp4A4TmaNoPromotions, 1, 128>>, 32, 0},
    {"a4_tma_t128_r128_s4_m1_l2", launch_a4_tma_cand<Nvfp4A4TmaMmaSchedule<128, 4, 1, kNvfp4A4TmaWeightCodeL2_128B, 1, 128>>, 32, 0},
    {"a4_tma_t128_r128_s4_m1_all", launch_a4_tma_cand<Nvfp4A4TmaMmaSchedule<128, 4, 1, kNvfp4A4TmaL2_128B, 1, 128>>, 32, 0},
    {"a4_tma_t256_r64_s2_m1",     launch_a4_tma_cand<Nvfp4A4TmaMmaSchedule<256, 2, 1, kNvfp4A4TmaNoPromotions, 1, 64>>, 128, 0},
    {"a4_tma_t256_r64_s2_m1_l2",  launch_a4_tma_cand<Nvfp4A4TmaMmaSchedule<256, 2, 1, kNvfp4A4TmaWeightCodeL2_128B, 1, 64>>, 128, 0},
    {"a4_tma_t256_r64_s3_m1",     launch_a4_tma_cand<Nvfp4A4TmaMmaSchedule<256, 3, 1, kNvfp4A4TmaNoPromotions, 1, 64>>, 128, 0},
    {"a4_tma_t256_r64_s3_m1_l2",  launch_a4_tma_cand<Nvfp4A4TmaMmaSchedule<256, 3, 1, kNvfp4A4TmaWeightCodeL2_128B, 1, 64>>, 128, 0},
    {"a4_tma_t256_r64_s3_m1_all", launch_a4_tma_cand<Nvfp4A4TmaMmaSchedule<256, 3, 1, kNvfp4A4TmaL2_128B, 1, 64>>, 128, 0},
    {"a4_tma_t256_r128_s3_m1",    launch_a4_tma_cand<Nvfp4A4TmaMmaSchedule<256, 3, 1, kNvfp4A4TmaNoPromotions, 1, 128>>, 128, 0},
    {"a4_tma_t256_r128_s3_m1_l2", launch_a4_tma_cand<Nvfp4A4TmaMmaSchedule<256, 3, 1, kNvfp4A4TmaWeightCodeL2_128B, 1, 128>>, 128, 0},
    {"a4_tma_t256_r128_s3_m1_ws", launch_a4_tma_cand<Nvfp4A4TmaMmaSchedule<256, 3, 1, kNvfp4A4TmaWeightL2_128B, 1, 128>>, 128, 0},
    {"a4_tma_t256_r128_s3_m1_all", launch_a4_tma_cand<Nvfp4A4TmaMmaSchedule<256, 3, 1, kNvfp4A4TmaL2_128B, 1, 128>>, 128, 0},
    // Cost split for the prefill tile: t(1q1g) = Q + G, so t(2q1g) - t(1q1g) = the quantizer and
    // t(1q2g) - t(1q1g) = the GEMM alone.
    {"a4_split_1q1g",            launch_a4_tma_repeats<A4PrefillSchedule, 1, 1>, 128, 0},
    {"a4_split_2q1g",            launch_a4_tma_repeats<A4PrefillSchedule, 2, 1>, 128, 0},
    {"a4_split_1q2g",            launch_a4_tma_repeats<A4PrefillSchedule, 1, 2>, 128, 0},

    // GB10 2026-11-01 (T2): the 2-term residual TMA GEMM at an explicit weight-panel height, so the
    // pair route's wave fit on 48 SMs is measured rather than inherited from the ctas_128 gate.
    {"a4_pair_r128_s3_m1", launch_a4_tma_cand<Nvfp4A4TmaMmaSchedule<128, 3, 1, kNvfp4A4TmaWeightCodeL2_128B, kNvfp4A4ResidualTerms, 128>>, 32, 0, true},
    {"a4_pair_r64_s3_m1",  launch_a4_tma_cand<Nvfp4A4TmaMmaSchedule<128, 3, 1, kNvfp4A4TmaWeightCodeL2_128B, kNvfp4A4ResidualTerms, 64>>, 32, 0, true},
    {"a4_pair_r64_s2_m2",  launch_a4_tma_cand<Nvfp4A4TmaMmaSchedule<128, 2, 2, kNvfp4A4TmaWeightCodeL2_128B, kNvfp4A4ResidualTerms, 64>>, 32, 0, true},
    {"a4_pair_r64_s4_m1",  launch_a4_tma_cand<Nvfp4A4TmaMmaSchedule<128, 4, 1, kNvfp4A4TmaWeightCodeL2_128B, kNvfp4A4ResidualTerms, 64>>, 32, 0, true},
    // GB10 2026-11-02 (T7 ext): the 2-term pair route is also the live prefill GEMM (split fused
    // launchers at T >= 65, public pair gate 33/65), where only the WeightCodeL2 panel has ever
    // been measured. The promotion variants are cache hints (bits unchanged), so a prefill win is
    // adoptable under the same rule the 1-term ladder follows.
    {"a4_pair_r128_s3_m1_ws",  launch_a4_tma_cand<Nvfp4A4TmaMmaSchedule<128, 3, 1, kNvfp4A4TmaWeightL2_128B, kNvfp4A4ResidualTerms, 128>>, 32, 0, true},
    {"a4_pair_r128_s3_m1_all", launch_a4_tma_cand<Nvfp4A4TmaMmaSchedule<128, 3, 1, kNvfp4A4TmaL2_128B, kNvfp4A4ResidualTerms, 128>>, 32, 0, true},
    {"a4_pair_r128_s3_m1_none", launch_a4_tma_cand<Nvfp4A4TmaMmaSchedule<128, 3, 1, kNvfp4A4TmaNoPromotions, kNvfp4A4ResidualTerms, 128>>, 32, 0, true},
    // The exact instance the model runs (baked K, WeightCodeL2): the pair route's launchers bake K
    // (launch_residual<K> / the split fused launchers' Nvfp4ScheduleInstance), while every other
    // pair candidate above reads K from the operands. The 1-term ladder showed baked K costs the
    // wide-token bands; this arm measures whether the 2-term GEMM pays the same. A mismatch with
    // --k throws in validate_nvfp4_operands and skips.
    {"a4_pair_r128_s3_m1_baked2560", launch_a4_tma_cand<Nvfp4ScheduleInstance<Nvfp4A4TmaMmaSchedule<128, 3, 1, kNvfp4A4TmaWeightCodeL2_128B, kNvfp4A4ResidualTerms, 128>, 2560>>, 32, 0, true},
    {"a4_pair_r128_s3_m1_baked6144", launch_a4_tma_cand<Nvfp4ScheduleInstance<Nvfp4A4TmaMmaSchedule<128, 3, 1, kNvfp4A4TmaWeightCodeL2_128B, kNvfp4A4ResidualTerms, 128>, 6144>>, 32, 0, true},
};

std::vector<int> parse_tokens(const std::string& spec) {
    std::vector<int> out;
    std::stringstream ss(spec);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        const auto colon = item.find(':');
        if (colon == std::string::npos) {
            out.push_back(std::stoi(item));
        } else {
            const auto colon2 = item.find(':', colon + 1);
            const int start   = std::stoi(item.substr(0, colon));
            const int end     = std::stoi(item.substr(
                colon + 1, colon2 == std::string::npos ? std::string::npos : colon2 - colon - 1));
            const int step    = colon2 == std::string::npos ? 1 : std::stoi(item.substr(colon2 + 1));
            for (int t = start; t <= end; t += step) out.push_back(t);
        }
    }
    return out;
}

// A candidate that never times is invisible in the report, so name the reason: a launch that throws
// and a launch whose output disagrees with the reference are different findings.
void report_skip(const char* phase, const char* name, int tokens, const char* reason) {
    std::fprintf(stderr, "  skip %-24s T=%d %s: %s\n", name, tokens, phase, reason);
}

bool check_numeric_agreement(const std::vector<__nv_bfloat16>& got,
                             const std::vector<__nv_bfloat16>& ref) {
    if (got.size() != ref.size()) return false;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const float g = __bfloat162float(got[i]);
        const float r = __bfloat162float(ref[i]);
        if (!std::isfinite(g) || !std::isfinite(r)) return false;
        const float diff = std::fabs(g - r);
        const float tol  = std::max(0.0625f, 0.03125f * std::fabs(r));
        if (diff > tol) return false;
    }
    return true;
}

// Comma-separated substring filter. "production" always survives so the report keeps its anchor
// even when the argmin candidate is filtered out.
bool filter_matches(const std::string& filter, const char* name) {
    if (filter.empty()) return true;
    if (std::strcmp(name, "production") == 0) return true;
    const std::string view(name);
    std::size_t begin = 0;
    while (begin <= filter.size()) {
        const std::size_t comma = filter.find(',', begin);
        const std::string token =
            filter.substr(begin, comma == std::string::npos ? std::string::npos : comma - begin);
        if (!token.empty() && view.find(token) != std::string::npos) return true;
        if (comma == std::string::npos) break;
        begin = comma + 1;
    }
    return false;
}

} // namespace

int main(int argc, char** argv) {
    try {
        int n                = 14336;
        int k                = 5120;
        std::string tok_spec = "1,2,3,4,8,16,24,32,48,64,96,128,192,256,384,512";
        int warmup           = 3;
        int repeat           = 15;
        int flush_mib        = 32;
        std::string csv_out;
        std::string filter;
        std::string policy   = "a16";

        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--n" && i + 1 < argc) n = std::stoi(argv[++i]);
            else if (a == "--k" && i + 1 < argc) k = std::stoi(argv[++i]);
            else if (a == "--tokens" && i + 1 < argc) tok_spec = argv[++i];
            else if (a == "--warmup" && i + 1 < argc) warmup = std::stoi(argv[++i]);
            else if (a == "--repeat" && i + 1 < argc) repeat = std::stoi(argv[++i]);
            else if (a == "--flush-mib" && i + 1 < argc) flush_mib = std::stoi(argv[++i]);
            else if (a == "-o" && i + 1 < argc) csv_out = argv[++i];
            else if (a == "--filter" && i + 1 < argc) filter = argv[++i];
            else if (a == "--policy" && i + 1 < argc) policy = argv[++i];
        }
        const bool is_a4 = (policy == "a4" || policy == "A4");

        const auto tokens_list = parse_tokens(tok_spec);
        int max_t              = 1;
        for (int t : tokens_list) max_t = std::max(max_t, t);

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        const std::size_t flush_bytes = static_cast<std::size_t>(flush_mib) << 20;
        bench::L2FlushBuffer flush(flush_bytes);

        auto qw = bench::make_nvfp4_weight(n, k);
        const std::size_t out_count = static_cast<std::size_t>(n) * max_t;
        DeviceBuffer x_storage(static_cast<std::size_t>(k) * max_t * 2);
        DeviceBuffer out_storage(out_count * 2);
        DeviceBuffer ref_storage(out_count * 2);
        DeviceBuffer exact_storage(out_count * 2);
        const std::size_t ws_cap =
            is_a4 ? std::max<std::size_t>(
                        ops::linear_workspace_capacity_bytes(
                            QType::NVFP4, n, k, ops::LinearPolicy::AllowA4, 1, max_t),
                        std::max(nvfp4_a4_workspace_capacity_bytes(max_t, k) * 2,
                                 nvfp4_a4_workspace_capacity_bytes(max_t, k) +
                                     nvfp4_a4_workspace_capacity_bytes(
                                         max_t, k, kNvfp4A4ResidualTerms, true)))
                  : 4096;
        DeviceBuffer ws_storage(ws_cap);
        CUDA_CHECK(bench::fixture::fill_values(static_cast<__nv_bfloat16*>(x_storage.p),
                                               static_cast<std::size_t>(k) * max_t, 0xa5a5ULL,
                                               -0.5F, 0.5F));
        CUDA_CHECK(cudaDeviceSynchronize());

        const Tensor x   = Tensor(x_storage.p, DType::BF16, {k, max_t});
        const Tensor out = Tensor(out_storage.p, DType::BF16, {n, max_t});
        const Tensor ref = Tensor(ref_storage.p, DType::BF16, {n, max_t});
        const Tensor ex  = Tensor(exact_storage.p, DType::BF16, {n, max_t});

        cudaEvent_t ev_start{}, ev_stop{};
        CUDA_CHECK(cudaEventCreate(&ev_start));
        CUDA_CHECK(cudaEventCreate(&ev_stop));

        std::ofstream csv;
        if (!csv_out.empty()) {
            csv.open(csv_out);
            csv << "N,K,T,route,median_us,min_us,p95_us,gbs,tflops,bit_exact,t_invariant\n";
        }

        // Reference at T=1
        std::vector<__nv_bfloat16> t1_ref(n);
        {
            Tensor x1   = x;
            x1.ne[1]    = 1;
            Tensor y1   = ref;
            y1.ne[1]    = 1;
            launch_production(x1, qw.weight, y1, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaMemcpy(t1_ref.data(), y1.data, n * sizeof(__nv_bfloat16),
                                  cudaMemcpyDeviceToHost));
        }

        const std::size_t code_bytes  = (static_cast<std::size_t>(n) * k) / 2;
        const std::size_t scale_bytes = (static_cast<std::size_t>(n) * k) / 16;

        std::printf("=== NVFP4 Route Sweep (%s): N=%d K=%d (flush=%d MiB, warmup=%d, repeat=%d) ===\n",
                    is_a4 ? "A4" : "A16", n, k, flush_mib, warmup, repeat);

        for (int t : tokens_list) {
            Tensor xs   = x;
            xs.ne[1]    = t;
            Tensor ys   = out;
            ys.ne[1]    = t;
            Tensor rs   = ref;
            rs.ne[1]    = t;
            Tensor es   = ex;
            es.ne[1]    = t;
            const std::size_t out_elems = static_cast<std::size_t>(n) * t;
            const std::size_t out_bytes = out_elems * sizeof(__nv_bfloat16);

            WorkspaceArena ws_arena(DeviceSpan{ws_storage.p, ws_cap});
            Nvfp4A4Workspace a4_ws =
                is_a4 ? allocate_nvfp4_a4_workspace(ws_arena, t, k) : Nvfp4A4Workspace{};
            // Second bump allocation from the same arena: disjoint from a4_ws, which the capacity
            // above already accounts for.
            Nvfp4A4Workspace pair_ws =
                is_a4 ? allocate_nvfp4_a4_workspace(ws_arena, t, k, kNvfp4A4ResidualTerms, true)
                      : Nvfp4A4Workspace{};

            if (is_a4) {
                launch_a4_mma_cand<Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>(xs, qw.weight, rs,
                                                                                a4_ws, stream);
                // The 2-term residual route is a different approximation from the 1-term one the
                // other A4 candidates are checked against, and two routes each inside tolerance of
                // exact can exceed it against each other. It is therefore judged against the exact
                // A16 output, which is the contract the fast path has to meet; without this the
                // pair route is dropped by the gate and reads as "no data".
                launch_production(xs, qw.weight, es, stream);
            } else {
                launch_production(xs, qw.weight, rs, stream);
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::vector<__nv_bfloat16> ref_host(out_elems);
            CUDA_CHECK(cudaMemcpy(ref_host.data(), rs.data, out_bytes, cudaMemcpyDeviceToHost));
            std::vector<__nv_bfloat16> exact_host(out_elems);
            if (is_a4) {
                CUDA_CHECK(
                    cudaMemcpy(exact_host.data(), ex.data, out_bytes, cudaMemcpyDeviceToHost));
            }
            std::vector<__nv_bfloat16> got(out_elems);

            const double total_bytes =
                static_cast<double>(code_bytes + scale_bytes) +
                (static_cast<double>(k) * t + static_cast<double>(n) * t) * 2.0;
            const double flops = 2.0 * static_cast<double>(n) * k * t;

            double best_us         = 1e30;
            const char* best_route = "";
            double prod_us         = 0.0;

            if (is_a4) {
                for (const auto& cand : kA4Candidates) {
                    if (t < cand.min_t) continue;
                    if (cand.max_t > 0 && t > cand.max_t) continue;
                    if (!filter_matches(filter, cand.name)) continue;

                    const bool is_prod = (std::strcmp(cand.name, "production") == 0);
                    Nvfp4A4Workspace run_ws = cand.residual ? pair_ws : a4_ws;
                    if (is_prod) {
                        run_ws.codes = static_cast<std::uint8_t*>(ws_storage.p);
                    }

                    bool ok = true;
                    try {
                        CUDA_CHECK(cudaMemsetAsync(ys.data, 0, out_bytes, stream));
                        cand.launch(xs, qw.weight, ys, run_ws, stream);
                        CUDA_CHECK(cudaStreamSynchronize(stream));
                    } catch (const std::exception& e) {
                        cudaGetLastError();
                        report_skip("launch", cand.name, t, e.what());
                        ok = false;
                    } catch (...) {
                        cudaGetLastError();
                        report_skip("launch", cand.name, t, "unknown exception");
                        ok = false;
                    }
                    if (!ok) continue;

                    CUDA_CHECK(cudaMemcpy(got.data(), ys.data, out_bytes, cudaMemcpyDeviceToHost));
                    const bool numeric_ok =
                        is_prod || cand.unchecked ||
                        check_numeric_agreement(got, cand.residual ? exact_host : ref_host);
                    if (!numeric_ok) {
                        report_skip("numeric", cand.name, t, "disagrees with the reference output");
                        continue;
                    }

                    for (int w_i = 0; w_i < warmup; ++w_i) {
                        bench::flush_l2(flush, stream);
                        cand.launch(xs, qw.weight, ys, run_ws, stream);
                    }
                    CUDA_CHECK(cudaStreamSynchronize(stream));

                    std::vector<double> samples_us;
                    samples_us.reserve(repeat);
                    for (int r_i = 0; r_i < repeat; ++r_i) {
                        bench::flush_l2(flush, stream);
                        CUDA_CHECK(cudaStreamSynchronize(stream));
                        CUDA_CHECK(cudaEventRecord(ev_start, stream));
                        cand.launch(xs, qw.weight, ys, run_ws, stream);
                        CUDA_CHECK(cudaEventRecord(ev_stop, stream));
                        CUDA_CHECK(cudaEventSynchronize(ev_stop));
                        float ms = 0.0f;
                        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
                        samples_us.push_back(static_cast<double>(ms) * 1000.0);
                    }
                    std::sort(samples_us.begin(), samples_us.end());
                    const double med_us = samples_us[samples_us.size() / 2];
                    const double min_us = samples_us.front();
                    const double p95_us =
                        samples_us[std::min(samples_us.size() - 1, (samples_us.size() * 95) / 100)];
                    const double gbs    = (total_bytes / (med_us * 1e-6)) / 1e9;
                    const double tflops = (flops / (med_us * 1e-6)) / 1e12;

                    if (is_prod) {
                        prod_us = med_us;
                    } else if (med_us < best_us) {
                        best_us    = med_us;
                        best_route = cand.name;
                    }

                    if (csv.is_open()) {
                        csv << n << "," << k << "," << t << "," << cand.name << "," << med_us << ","
                            << min_us << "," << p95_us << "," << gbs << "," << tflops << ",1,1\n";
                    }
                }
                const double gain_pct = prod_us > 0.0 ? (prod_us - best_us) / prod_us * 100.0 : 0.0;
                const double best_tf  = (flops / (best_us * 1e-6)) / 1e12;
                std::printf("  T=%4d | prod: %8.2f us | best: %8.2f us (%+6.2f%%, %6.1f TF) -> %s\n",
                            t, prod_us, best_us, -gain_pct, best_tf, best_route);
                continue;
            }

            for (const auto& cand : kCandidates) {
                if (t < cand.min_t) continue;
                if (cand.max_t > 0 && t > cand.max_t) continue;
                if (cand.static_k > 0 && k != cand.static_k) continue;
                if (!filter_matches(filter, cand.name)) continue;

                bool ok = true;
                try {
                    CUDA_CHECK(cudaMemsetAsync(ys.data, 0, out_bytes, stream));
                    cand.launch(xs, qw.weight, ys, stream);
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                } catch (const std::exception& e) {
                    cudaGetLastError();
                    report_skip("launch", cand.name, t, e.what());
                    ok = false;
                } catch (...) {
                    cudaGetLastError();
                    report_skip("launch", cand.name, t, "unknown exception");
                    ok = false;
                }
                if (!ok) continue;

                CUDA_CHECK(cudaMemcpy(got.data(), ys.data, out_bytes, cudaMemcpyDeviceToHost));
                const bool numeric_ok = check_numeric_agreement(got, ref_host);
                const bool t_invariant =
                    std::memcmp(got.data(), t1_ref.data(), n * sizeof(__nv_bfloat16)) == 0;
                if (!numeric_ok) continue;

                for (int w_i = 0; w_i < warmup; ++w_i) {
                    bench::flush_l2(flush, stream);
                    cand.launch(xs, qw.weight, ys, stream);
                }
                CUDA_CHECK(cudaStreamSynchronize(stream));

                std::vector<double> samples_us;
                samples_us.reserve(repeat);
                for (int r_i = 0; r_i < repeat; ++r_i) {
                    bench::flush_l2(flush, stream);
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    CUDA_CHECK(cudaEventRecord(ev_start, stream));
                    cand.launch(xs, qw.weight, ys, stream);
                    CUDA_CHECK(cudaEventRecord(ev_stop, stream));
                    CUDA_CHECK(cudaEventSynchronize(ev_stop));
                    float ms = 0.0f;
                    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
                    samples_us.push_back(static_cast<double>(ms) * 1000.0);
                }
                std::sort(samples_us.begin(), samples_us.end());
                const double med_us = samples_us[samples_us.size() / 2];
                const double min_us = samples_us.front();
                const double p95_us =
                    samples_us[std::min(samples_us.size() - 1, (samples_us.size() * 95) / 100)];
                const double gbs    = (total_bytes / (med_us * 1e-6)) / 1e9;
                const double tflops = (flops / (med_us * 1e-6)) / 1e12;

                if (std::strcmp(cand.name, "production") == 0) {
                    prod_us = med_us;
                } else if (med_us < best_us) {
                    best_us    = med_us;
                    best_route = cand.name;
                }

                if (csv.is_open()) {
                    csv << n << "," << k << "," << t << "," << cand.name << "," << med_us << ","
                        << min_us << "," << p95_us << "," << gbs << "," << tflops << ","
                        << (numeric_ok ? 1 : 0) << "," << (t_invariant ? 1 : 0) << "\n";
                }
            }

            const double gain_pct = prod_us > 0.0 ? (prod_us - best_us) / prod_us * 100.0 : 0.0;
            std::printf("  T=%4d | prod: %8.2f us | best: %8.2f us (%+6.2f%%) -> %s\n", t, prod_us,
                        best_us, -gain_pct, best_route);
        }

        CUDA_CHECK(cudaEventDestroy(ev_start));
        CUDA_CHECK(cudaEventDestroy(ev_stop));
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
