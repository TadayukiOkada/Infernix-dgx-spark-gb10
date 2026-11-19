#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"
#include "ops/linear/nvfp4/nvfp4_a16_slice_ring_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"
#include "ops/linear/nvfp4/nvfp4_instances.cuh"

#include <stdexcept>

namespace infernix::ops::detail {
namespace {
// GB10 (sm_121a) retune, 2026-10-31 (T7), from the same-process candidate sweep in
// bench/ops/nvfp4_linear_add_bench.cu --variants (cold L2, 25 repeats, second pass).
//
// The K-baked sliced-K tiles this table carried from the RTX 5090 fit are the dominant loss: at
// N=5120 the 8-token 4 K-warp tile costs 322.4 us at K=17408 T=1 against 226.3 for the runtime-K
// RowTiles-2 twin of the same tile (+42%), and the 32-token tile 403.5 against 252.0 at T=24 and
// 432.1 against 262.1 at T=32 (K=6144: 118.6 against 96.3 at T=24, 135.4 against 100.4 at T=32).
// Baking K lets ptxas fully unroll the K-group loop, which costs more than it saves at these
// widths -- see nvfp4_linear_a16_sliced_k_dyn. Every sliced-K band below is therefore taken in the
// runtime-K form, and with RowTiles=2, which the same sweep puts 1..4% ahead of RowTiles=1 where
// both are legal. RowTiles only sets how many row tiles share a CTA and its launch bounds, so a
// column's K slices and their reduction order are unchanged: the RowTiles-2 tiles are bit-identical
// to their RowTiles-1 twins.
//
// The 2 K-warp tiles of the 33..64 bands are dropped outright. At K=6144 they pay 200.5/215.0 us
// (T=48/64) and the 64-token twin 251.9/274.4, where the 64-row K128 MMA tile the 65..128 band
// already runs pays 116.7; at K=17408 they pay 768.0/833.5 and 618.3/651.3 against the 32x64 MMA
// tile's 350.2/346.1.
template <int K>
void launch_matrix(const Tensor& x, const Weight& weight, Tensor& residual, bool wide_k, cudaStream_t stream) {
    const int tokens  = x.ne[1];
    const auto p      = nvfp4_a16_operands(x, weight);
    const auto output = LinearBf16Output{static_cast<__nv_bfloat16*>(residual.data), weight.n};
    const auto epilogue =
        LinearResidualAddEpilogue{{static_cast<__nv_bfloat16*>(residual.data), weight.n}};
    if constexpr (K == 6144) {
        if (tokens <= 8) {
            launch_nvfp4_a16_sliced_k_mma<Nvfp4SlicedInstance<8, 8, 2, 2>>(p, output, epilogue,
                                                                           stream);
            return;
        }
        if (tokens <= 16) {
            launch_nvfp4_a16_sliced_k_mma<Nvfp4SlicedInstance<16, 8, 2, 2>>(p, output, epilogue,
                                                                            stream);
            return;
        }
        if (tokens <= 32) {
            launch_nvfp4_a16_sliced_k_mma<Nvfp4SlicedInstance<32, 4, 1, 2>>(p, output, epilogue,
                                                                            stream);
            return;
        }
        if (tokens <= 64) {
            launch_nvfp4_a16_mma<
                Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<64, 64, 128, 32, 16, 2, 1>, 6144>>(
                p, output, epilogue, stream);
            return;
        }
        {
            launch_nvfp4_a16_mma<
                Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>, 6144>>(
                p, output, epilogue, stream);
            return;
        }
    } else {
        // N=5120 gives these tiles 320 CTAs of 16 rows, so their time is the weight stream each
        // keeps in flight: 8 K-warps stream 512 K per stage where 4 streamed 256. The 8 K-warp tiles
        // are the argmin at every width from 1 to 16 (223.3 against 228.2 for the 4 K-warp twin at
        // T=2, 227.1 against 230.2 at T=8), but the K-warp count sets each column's reduction order,
        // so they run only under LinearRoutes::WideDownTile (`wide_k`: batched-decode's decode and
        // verification rounds, where they were gated); the exact route and prefill keep the 4
        // K-warp tiles of the release, which the sweep still puts ahead of the SIMT kernel and of
        // every K-baked tile.
        if (tokens <= 8) {
            if (wide_k) {
                launch_nvfp4_a16_sliced_k_mma<Nvfp4SlicedInstance<8, 8, 2, 2>>(
                    p, output, epilogue, stream);
            } else {
                launch_nvfp4_a16_sliced_k_mma<Nvfp4SlicedInstance<8, 4, 2, 2>>(
                    p, output, epilogue, stream);
            }
            return;
        }
        if (tokens <= 16) {
            if (wide_k) {
                launch_nvfp4_a16_sliced_k_mma<Nvfp4SlicedInstance<16, 8, 2, 2>>(
                    p, output, epilogue, stream);
            } else {
                launch_nvfp4_a16_sliced_k_mma<Nvfp4SlicedInstance<16, 4, 2, 2>>(
                    p, output, epilogue, stream);
            }
            return;
        }
        if (tokens <= 32) {
            launch_nvfp4_a16_sliced_k_mma<Nvfp4SlicedInstance<32, 4, 1, 2>>(p, output, epilogue,
                                                                            stream);
            return;
        }
        if (tokens <= 128) {
            launch_nvfp4_a16_mma<
                Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<32, 64, 128, 32, 16, 2, 2>, 17408>>(
                p, output, epilogue, stream);
            return;
        }
        {
            launch_nvfp4_a16_mma<
                Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>, 17408>>(
                p, output, epilogue, stream);
            return;
        }
    }
}
} // namespace

void nvfp4_linear_add_invariant_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                       cudaStream_t stream) {
    // The slice-major ring tile, bit-identical to the sliced-K tiles, where it is faster (2026-10-25): [5120,6144]
    // from 17 columns (0.45-0.68x), [5120,17408] from 32 (1.23x at 17, 1.06x at 24, 0.93x at 32, 0.52-0.85x
    // beyond); the reproducible route's DFlash2 decode +19 %.
    if (x.ne[1] >= (weight.k == 17408 ? 32 : 17)) {
        nvfp4_linear_add_invariant_ring_launch(x, weight, residual, stream);
        return;
    }
    nvfp4_linear_add_invariant_sliced_launch(x, weight, residual, stream);
}

namespace {
template <int KSlices, int K>
void launch_invariant_ring(const Tensor& x, const Weight& weight, Tensor& residual,
                           cudaStream_t stream) {
    const auto p      = nvfp4_a16_operands(x, weight);
    const auto output = LinearBf16Output{static_cast<__nv_bfloat16*>(residual.data), weight.n};
    const auto epilogue =
        LinearResidualAddEpilogue{{static_cast<__nv_bfloat16*>(residual.data), weight.n}};
    // N=5120: 32-row tiles give 160 CTAs a token tile, two an SM; above 64 columns 64-row tiles
    // keep 160 CTAs over two token tiles with half the activation reads.
    if (x.ne[1] <= 32) {
        launch_nvfp4_a16_slice_ring_mma<
            Nvfp4ScheduleInstance<Nvfp4A16SliceRingSchedule<KSlices, 32, 32, 2, 2, 4, 2>, K>>(
            p, output, epilogue, stream);
    } else if (x.ne[1] <= 64) {
        launch_nvfp4_a16_slice_ring_mma<
            Nvfp4ScheduleInstance<Nvfp4A16SliceRingSchedule<KSlices, 32, 64, 2, 4, 4, 2>, K>>(
            p, output, epilogue, stream);
    } else {
        launch_nvfp4_a16_slice_ring_mma<
            Nvfp4ScheduleInstance<Nvfp4A16SliceRingSchedule<KSlices, 64, 64, 4, 2, 4, 2>, K>>(
            p, output, epilogue, stream);
    }
}
} // namespace

void nvfp4_linear_add_invariant_ring_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                            cudaStream_t stream) {
    // The K-warp counts of the sliced-K tiles below: 8 at K=6144, 4 at K=17408.
    if (x.ne[1] < 17) throw std::invalid_argument("nvfp4 linear_add ring tile: T must be 17 or more");
    if (weight.k == 6144) return launch_invariant_ring<8, 6144>(x, weight, residual, stream);
    if (weight.k == 17408) return launch_invariant_ring<4, 17408>(x, weight, residual, stream);
    throw std::invalid_argument("nvfp4 linear_add: unsupported column-invariant problem");
}

void nvfp4_linear_add_invariant_sliced_launch(const Tensor& x, const Weight& weight,
                                              Tensor& residual, cudaStream_t stream) {
    // Tiles of one K-warp count walk the same K slices and reduce them in the same order, and the
    // MMA's columns are independent: the tile width only sets how many columns share a launch,
    // and wider calls run as 64-column slices.
    const int tokens  = x.ne[1];
    const auto p      = nvfp4_a16_operands(x, weight);
    const auto output = LinearBf16Output{static_cast<__nv_bfloat16*>(residual.data), weight.n};
    const auto epilogue =
        LinearResidualAddEpilogue{{static_cast<__nv_bfloat16*>(residual.data), weight.n}};
    if (weight.k == 6144) {
        if (tokens <= 8)
            return launch_nvfp4_a16_sliced_k_mma<
                Nvfp4ScheduleInstance<Nvfp4SlicedInstance<8, 8, 2>, 6144>>(p, output, epilogue,
                                                                           stream);
        if (tokens <= 16)
            return launch_nvfp4_a16_sliced_k_mma<
                Nvfp4ScheduleInstance<Nvfp4SlicedInstance<16, 8, 2>, 6144>>(p, output, epilogue,
                                                                            stream);
        return launch_nvfp4_a16_sliced_k_mma<
            Nvfp4ScheduleInstance<Nvfp4SlicedInstance<64, 8, 1>, 6144>>(p, output, epilogue,
                                                                        stream);
    }
    if (weight.k == 17408) {
        if (tokens <= 8)
            return launch_nvfp4_a16_sliced_k_mma<
                Nvfp4ScheduleInstance<Nvfp4SlicedInstance<8, 4, 2>, 17408>>(p, output, epilogue,
                                                                            stream);
        if (tokens <= 16)
            return launch_nvfp4_a16_sliced_k_mma<
                Nvfp4ScheduleInstance<Nvfp4SlicedInstance<16, 4, 2>, 17408>>(p, output, epilogue,
                                                                             stream);
        if (tokens <= 32)
            return launch_nvfp4_a16_sliced_k_mma<
                Nvfp4ScheduleInstance<Nvfp4SlicedInstance<32, 4, 2>, 17408>>(p, output, epilogue,
                                                                             stream);
        return launch_nvfp4_a16_sliced_k_mma<
            Nvfp4ScheduleInstance<Nvfp4SlicedInstance<64, 4, 2>, 17408>>(p, output, epilogue,
                                                                         stream);
    }
    throw std::invalid_argument("nvfp4 linear_add: unsupported column-invariant problem");
}

void nvfp4_linear_add_a16_launch(const Tensor& x, const Weight& weight, Tensor& residual, bool wide_k,
                                 cudaStream_t stream) {
    if (x.ne[1] == 1) {
        nvfp4_linear_add_decode_launch(x, weight, residual, stream);
        return;
    }
    // The SIMT kernel of the 2..5-token bands loses to the sliced-K tile of the same K-warp count
    // at every one of them on GB10 (K=17408: 237.5/238.4 us at T=2/4 against 228.2/230.2, 246.6 at
    // T=5 against 231.3; K=6144: 85.0 at T=2 against 83.6), so every T >= 2 goes to the tile table.
    if (weight.k == 6144)
        launch_matrix<6144>(x, weight, residual, wide_k, stream);
    else
        launch_matrix<17408>(x, weight, residual, wide_k, stream);
}
} // namespace infernix::ops::detail
