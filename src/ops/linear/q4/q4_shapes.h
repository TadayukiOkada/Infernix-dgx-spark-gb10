#pragma once

#include "ops/linear/q4/q4_launch.h"

namespace infernix::ops::detail {

[[nodiscard]] Q4Launch select_q4_n1024_k5120(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n4096_k5120(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n5120_k6144(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n6144_k5120(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n7168_k5120(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n34816_k5120(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n131072_k5120(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n131072_k2048(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n131072_k2560(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n131072_k3072(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n248320_k2560(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n248320_k5120(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n3456_k1152(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n4304_k1152(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n1280_k2560(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n2560_k640(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n324_k10240(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n320_k10240(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n10240_k320(std::int32_t tokens);

} // namespace infernix::ops::detail
