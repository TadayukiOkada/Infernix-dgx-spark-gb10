#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"
#include "ops/linear/nvfp4/nvfp4_instances.cuh"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh"

namespace infernix::ops::detail {
void nvfp4_gdn_input_invariant_launch(const Tensor& x, const Weight& weight, Tensor& qkv,
                                      Tensor& z, cudaStream_t stream) {
    // The tiles run 8 K-warps over the same K slices and reduce them in the same order, and the
    // MMA's columns are independent; wider calls run as 64-column slices.
    const auto p        = nvfp4_a16_operands(x, weight);
    const auto output   = Nvfp4GdnInputOutput{static_cast<__nv_bfloat16*>(qkv.data),
                                            static_cast<__nv_bfloat16*>(z.data)};
    const auto epilogue = LinearIdentityEpilogue{};
    if (x.ne[1] <= 8)
        return launch_nvfp4_a16_sliced_k_mma<
            Nvfp4ScheduleInstance<Nvfp4SlicedInstance<8, 8, 2>, 5120>>(p, output, epilogue, stream);
    if (x.ne[1] <= 16)
        return launch_nvfp4_a16_sliced_k_mma<
            Nvfp4ScheduleInstance<Nvfp4SlicedInstance<16, 8, 2>, 5120>>(p, output, epilogue,
                                                                        stream);
    launch_nvfp4_a16_sliced_k_mma<Nvfp4ScheduleInstance<Nvfp4SlicedInstance<64, 8, 1>, 5120>>(
        p, output, epilogue, stream);
}

void nvfp4_gdn_input_a16_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens == 1) {
        nvfp4_gdn_input_decode_launch(x, weight, qkv, z, stream);
        return;
    }
    const auto p        = nvfp4_a16_operands(x, weight);
    const auto output   = Nvfp4GdnInputOutput{static_cast<__nv_bfloat16*>(qkv.data),
                                            static_cast<__nv_bfloat16*>(z.data)};
    const auto epilogue = LinearIdentityEpilogue{};
    // GB10 (sm_121a) 2026-10-31 (T7): the 16-token RowTiles-2 sliced-K tile the sibling standalone
    // [16384, 5120] table ships takes the whole 2..16 band. It replaces the exact SIMT kernel at
    // T=2, which measured 215.7 us against the A4 lane's 212.8 and against this tile, and the
    // RowTiles-1 tiles the rest of the band ran; RowTiles only sets how many row tiles share a CTA
    // and its launch bounds, so a column's 8 K slices and their order are unchanged.
    if (tokens <= 16) {
        launch_nvfp4_a16_sliced_k_mma<
            Nvfp4ScheduleInstance<Nvfp4SlicedInstance<16, 8, 2, 2>, 5120>>(p, output, epilogue,
                                                                           stream);
        return;
    }
    if (tokens <= 24) {
        launch_nvfp4_a16_sliced_k_mma<Nvfp4ScheduleInstance<Nvfp4SlicedInstance<32, 8, 1>, 5120>>(
            p, output, epilogue, stream);
        return;
    }
    if (tokens <= 32) {
        launch_nvfp4_a16_sliced_k_mma<Nvfp4ScheduleInstance<Nvfp4SlicedInstance<32, 8, 2>, 5120>>(
            p, output, epilogue, stream);
        return;
    }
    if (tokens <= 64) {
        launch_nvfp4_a16_mma<
            Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<64, 64, 64, 32, 16, 2, 2>, 5120>>(
            p, output, epilogue, stream);
        return;
    }
    if (tokens <= 96) {
        launch_nvfp4_a16_mma<
            Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<64, 96, 128, 64, 16, 1, 2>, 5120>>(
            p, output, epilogue, stream);
        return;
    }
    {
        launch_nvfp4_a16_mma<
            Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>, 5120>>(
            p, output, epilogue, stream);
        return;
    }
}
} // namespace infernix::ops::detail
