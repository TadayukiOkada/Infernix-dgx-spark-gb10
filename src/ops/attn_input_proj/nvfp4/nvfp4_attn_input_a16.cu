#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"
#include "ops/linear/nvfp4/nvfp4_instances.cuh"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_output.cuh"

namespace infernix::ops::detail {
void nvfp4_attn_input_invariant_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                       Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream) {
    // The tiles run 8 K-warps over the same K slices and reduce them in the same order, and the
    // MMA's columns are independent; wider calls run as 64-column slices.
    const auto p      = nvfp4_a16_operands(x, weight);
    const auto output = Nvfp4AttentionInputOutput{
        static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)};
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

void nvfp4_attn_input_a16_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                 Tensor& k, Tensor& v, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens == 1) {
        nvfp4_attn_input_decode_launch(x, weight, q, gate, k, v, stream);
        return;
    }
    const auto p      = nvfp4_a16_operands(x, weight);
    const auto output = Nvfp4AttentionInputOutput{
        static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)};
    const auto epilogue = LinearIdentityEpilogue{};
    // GB10 (sm_121a) 2026-10-31 (T7): the 16-token RowTiles-2 sliced-K tile the sibling standalone
    // [14336, 5120] table ships takes the whole 2..16 band. It replaces the exact SIMT kernel at
    // T=2 (185.2 us, and 186.2 under the A16 policy, against this tile) and the RowTiles-1 tiles
    // the rest of the band ran; RowTiles only sets how many row tiles share a CTA and its launch
    // bounds, so a column's 8 K slices and their order are unchanged.
    if (tokens <= 16) {
        launch_nvfp4_a16_sliced_k_mma<
            Nvfp4ScheduleInstance<Nvfp4SlicedInstance<16, 8, 2, 2>, 5120>>(p, output, epilogue,
                                                                           stream);
        return;
    }
    if (tokens <= 24) {
        launch_nvfp4_a16_sliced_k_mma<Nvfp4ScheduleInstance<Nvfp4SlicedInstance<32, 4, 1>, 5120>>(
            p, output, epilogue, stream);
        return;
    }
    if (tokens <= 32) {
        launch_nvfp4_a16_mma<
            Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<32, 32, 128, 16, 16, 2, 2>, 5120>>(
            p, output, epilogue, stream);
        return;
    }
    if (tokens <= 64) {
        launch_nvfp4_a16_mma<
            Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>, 5120>>(
            p, output, epilogue, stream);
        return;
    }
    if (tokens <= 128) {
        launch_nvfp4_a16_mma<
            Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<64, 64, 64, 32, 16, 2, 2>, 5120>>(
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
