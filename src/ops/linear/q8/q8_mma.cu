#include "ops/linear/q8/q8_mma_launch.cuh"
#include "ops/linear/q8/q8_instances.cuh"
#include "ops/linear/q8/q8_launch.h"

namespace infernix::ops::detail {
// The runtime-shape schedules: FP32 or BF16 output (a vocabulary head's FP32 logits take them).
#define INFERNIX_Q8_MMA_LAUNCHER(Name, Schedule)                                                     \
    void Name(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {                \
        with_linear_output(out, out.ne[0], [&](auto output) {                                      \
            launch_q8_a16_mma<q8_instances::Schedule>(q8_linear_operands(x, w), output,           \
                                                      LinearIdentityEpilogue{}, stream);           \
        });                                                                                        \
    }

INFERNIX_Q8_MMA_LAUNCHER(launch_q8_a16_mma_r32_t16, MmaR32T16)
INFERNIX_Q8_MMA_LAUNCHER(launch_q8_a16_mma_r32_t24, MmaR32T24)
INFERNIX_Q8_MMA_LAUNCHER(launch_q8_a16_mma_r32_t32, MmaR32T32)
INFERNIX_Q8_MMA_LAUNCHER(launch_q8_a16_mma_r32_t48, MmaR32T48)
INFERNIX_Q8_MMA_LAUNCHER(launch_q8_a16_mma_r32_t64, MmaR32T64)
INFERNIX_Q8_MMA_LAUNCHER(launch_q8_a16_mma_r32_t96, MmaR32T96)
INFERNIX_Q8_MMA_LAUNCHER(launch_q8_a16_mma_r32_t128, MmaR32T128)
INFERNIX_Q8_MMA_LAUNCHER(launch_q8_a16_mma_r48_t64, MmaR48T64)
INFERNIX_Q8_MMA_LAUNCHER(launch_q8_a16_mma_r64_t32, MmaR64T32)
INFERNIX_Q8_MMA_LAUNCHER(launch_q8_a16_mma_r64_t48, MmaR64T48)
INFERNIX_Q8_MMA_LAUNCHER(launch_q8_a16_mma_r64_t96, MmaR64T96)
INFERNIX_Q8_MMA_LAUNCHER(launch_q8_a16_mma_r64_t128, MmaR64T128)
INFERNIX_Q8_MMA_LAUNCHER(launch_q8_a16_mma_r96_t96, MmaR96T96)
INFERNIX_Q8_MMA_LAUNCHER(launch_q8_a16_mma_r128_t64, MmaR128T64)
INFERNIX_Q8_MMA_LAUNCHER(launch_q8_a16_mma_r128_t80, MmaR128T80)
INFERNIX_Q8_MMA_LAUNCHER(launch_q8_a16_mma_r64x16_t48_k128_a1, MmaR64x16T48K128A1)
INFERNIX_Q8_MMA_LAUNCHER(launch_q8_a16_mma_r64x32_t64_k128_a1, MmaR64x32T64K128A1)
INFERNIX_Q8_MMA_LAUNCHER(launch_q8_a16_mma_r64x32_t96_k128_a1, MmaR64x32T96K128A1)
INFERNIX_Q8_MMA_LAUNCHER(launch_q8_a16_mma_r64x16_t128_k128_a2, MmaR64x16T128K128A2)
INFERNIX_Q8_MMA_LAUNCHER(launch_q8_a16_mma_r64x16_t256_k64_a2, MmaR64x16T256K64A2)
INFERNIX_Q8_MMA_LAUNCHER(launch_q8_a16_mma_r64x32_t64_k64_a2, MmaR64x32T64K64A2)
INFERNIX_Q8_MMA_LAUNCHER(launch_q8_a16_mma_r64x32_t128_k64_a2, MmaR64x32T128K64A2)
INFERNIX_Q8_MMA_LAUNCHER(launch_q8_a16_mma_r128x32_t64_k64_a2, MmaR128x32T64K64A2)
INFERNIX_Q8_MMA_LAUNCHER(launch_q8_a16_mma_r128x64_t128_k64_a1, MmaR128x64T128K64A1)
INFERNIX_Q8_MMA_LAUNCHER(launch_q8_a16_mma_r128x32_t128_k64_a2, MmaR128x32T128K64A2)
// T2 (2026-10-31) GB10 decode/verify tiles.
INFERNIX_Q8_MMA_LAUNCHER(launch_q8_a16_mma_r16_t32_k128_a1, MmaR16T32K128A1)
INFERNIX_Q8_MMA_LAUNCHER(launch_q8_a16_mma_r32_t32_k128_a1, MmaR32T32K128A1)
INFERNIX_Q8_MMA_LAUNCHER(launch_q8_a16_mma_r128x16_t64_k128_a2, MmaR128x16T64K128A2M1)
// T3 (2026-10-31) GB10 skinny/LoRA/drafter tiles.
INFERNIX_Q8_MMA_LAUNCHER(launch_q8_a16_mma_r32_t64_k128_a1, MmaR32T64K128A1)
INFERNIX_Q8_MMA_LAUNCHER(launch_q8_a16_mma_r32_t64_w16x32_k128_a1, MmaR32T64W16x32K128A1)

#undef INFERNIX_Q8_MMA_LAUNCHER

} // namespace infernix::ops::detail
