#pragma once
// MIRROR-OF: src/ops/linear/nvfp4/nvfp4_operands.h (the operand struct only; its builders are host code)
#include <cuda_bf16.h>
#include <cstdint>

namespace infernix::ops::detail {
struct Nvfp4A16Operands {
    const __nv_bfloat16* x;
    const std::uint8_t* codes;
    const std::uint8_t* scales;
    int rows, k, tokens;
    float alpha;
};
} // namespace infernix::ops::detail
