#include "ops/linear/q4/q4_dispatch.h"
#include "ops/linear/q4/q4_shapes.h"
#include <array>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {
struct ShapeEntry {
    std::int32_t n, k;
    Q4Launch (*select)(std::int32_t);
};

constexpr std::array kShapes{
    ShapeEntry{1024, 5120, select_q4_n1024_k5120},
    ShapeEntry{4096, 5120, select_q4_n4096_k5120},
    ShapeEntry{5120, 6144, select_q4_n5120_k6144},
    ShapeEntry{6144, 5120, select_q4_n6144_k5120},
    ShapeEntry{7168, 5120, select_q4_n7168_k5120},
    ShapeEntry{34816, 5120, select_q4_n34816_k5120},
    ShapeEntry{131072, 5120, select_q4_n131072_k5120},
    ShapeEntry{131072, 2048, select_q4_n131072_k2048},
    ShapeEntry{131072, 2560, select_q4_n131072_k2560},
    ShapeEntry{131072, 3072, select_q4_n131072_k3072},
    ShapeEntry{248320, 2560, select_q4_n248320_k2560},
    ShapeEntry{248320, 5120, select_q4_n248320_k5120},
    ShapeEntry{3456, 1152, select_q4_n3456_k1152},
    ShapeEntry{4304, 1152, select_q4_n4304_k1152},
    ShapeEntry{1280, 2560, select_q4_n1280_k2560},
    ShapeEntry{2560, 640, select_q4_n2560_k640},
    ShapeEntry{324, 10240, select_q4_n324_k10240},
    ShapeEntry{320, 10240, select_q4_n320_k10240},
    ShapeEntry{10240, 320, select_q4_n10240_k320},
};

// The proposal heads' column-invariant route: the four-K-warp sliced family at every width (the
// 32-column launch takes wider calls 32 at a time). Its instances differ only in token capacity
// and staging, so a column reduces identically whatever the call's width.
Q4Launch select_q4_invariant(std::int32_t t) {
    if (t <= 8) return launch_q4_a16_sliced_r32_t8_w4_s2;
    if (t <= 16) return launch_q4_a16_sliced_r32_t16_w4_s1;
    return launch_q4_a16_sliced_r32_t32_w4_s1;
}
Q4Launch select_q4_generic(std::int32_t n, std::int32_t k, std::int32_t t) {
    if (k % 128 != 0) {
        if (t <= 32) return launch_q4_a16_mma_r32_t32_k64_wr16_wt16_s3_a3_b2;
        return launch_q4_a16_mma_r32_t64_k64_wr16_wt32_s2_a2_b2;
    }
    if (t == 1) return launch_q4_a16_simt_r4_t1_w2_g8_s2;
    if (t <= 4) return launch_q4_a16_simt_r4_t4_w2_g8_s2;
    if (t <= 8) return launch_q4_a16_sliced_r32_t8_w4_s2;
    if (t <= 16) return launch_q4_a16_sliced_r32_t16_w4_s1;
    if (t <= 32) return launch_q4_a16_sliced_r32_t32_w4_s1;
    if (t <= 64 && (n % 32) == 0) return launch_q4_a16_sliced_r32_t32_w2_s2;
    if ((n % 64) == 0) return launch_q4_a16_mma_r64_t64_k128_s2_a1;
    return launch_q4_a16_mma_r32_t64_k64_wr16_wt32_s2_a2_b2;
}
} // namespace

bool q4_column_invariant_route(std::int32_t n, std::int32_t k) noexcept {
    return (n == 131072 && (k == 5120 || k == 3072 || k == 2560 || k == 2048)) ||
           (n == 248320 && (k == 5120 || k == 2560));
}

Q4Launch select_q4_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    if (t <= 0) throw std::invalid_argument("q4 linear: T must be positive");
    for (const auto& entry : kShapes) {
        if (entry.n == n && entry.k == k) return entry.select(t);
    }
    return select_q4_generic(n, k, t);
}

Q4Launch select_q4_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy) {
    if (!valid_linear_policy(policy)) throw std::invalid_argument("q4 linear: unsupported policy");
    if (column_invariant(policy)) {
        if (t <= 0) throw std::invalid_argument("q4 linear: T must be positive");
        if (!q4_column_invariant_route(n, k))
            throw std::invalid_argument("q4 linear: no column-invariant route for this problem");
        return select_q4_invariant(t);
    }
    return select_q4_a16_launch(n, k, t);
}

void q4_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                 cudaStream_t stream) {
    select_q4_launch(weight.n, weight.k, x.ne[1], policy)(x, weight, out, stream);
}
} // namespace infernix::ops::detail
