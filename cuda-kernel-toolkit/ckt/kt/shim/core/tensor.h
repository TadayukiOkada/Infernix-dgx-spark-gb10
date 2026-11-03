#pragma once
// Host Tensor is not needed by device code; output.cuh only names it inside host templates.
namespace infernix {
enum class DType { FP32, BF16 };
struct Tensor {
    void* data;
    DType dtype;
};
} // namespace infernix
namespace infernix::ops::detail { using ::infernix::Tensor; using ::infernix::DType; }
