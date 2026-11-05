#include "ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_kernels.h"
#include "core/device.h"
#include "core/pdl.cuh"
#include <cuda_bf16.h>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {
// Channels each reduce thread folds; 16 / this value threads cover one 16-channel group. Four
// channels per thread (a 64-thread CTA) measured 1.1-3.7% slower on the whole Op, so the fold
// stays at two-channel `__nv_bfloat162` accesses.
constexpr int kChannelsPerThread = 2;

// Hidden width H: G = H / 16 groups and 4 G coefficient rows (2 sides x 2 taps).
template <int Hidden, int Capacity>
__global__ __launch_bounds__(Capacity * (16 / kChannelsPerThread), 4)
    void dynamic_grouped_conv_prepare_reduce_kernel(const __nv_bfloat16* base, const float* partial,
                                                    __nv_bfloat16* prepared, __nv_bfloat16* finish,
                                                    int width, int batch_size, int splits) {
    constexpr int kGroups       = Hidden / 16, kCoefficientRows = 4 * kGroups;
    constexpr int kGroupThreads = 16 / kChannelsPerThread;
    pdl::enter();
    __shared__ float projected[4][Capacity];
    __shared__ float normalized[Capacity][16];
    const int tid = threadIdx.x, group = blockIdx.x, batch = blockIdx.y;
    const int tokens = width * batch_size;
    if (tid < 4 * Capacity && tid % Capacity < width) {
        const int coefficient = tid / Capacity, position = tid % Capacity,
                  row = coefficient * kGroups + group;
        float sum     = 0;
        for (int split = 0; split < splits; ++split)
            sum += partial[(static_cast<std::int64_t>(split) * tokens + batch * width + position) *
                               kCoefficientRows +
                           row];
        projected[coefficient][position] = sum;
    }
    const int position = tid / kGroupThreads, channel = (tid % kGroupThreads) * kChannelsPerThread;
    const int hidden = group * 16 + channel;
    const std::int64_t offset = static_cast<std::int64_t>(batch * width + position) * Hidden + hidden;
    if (position < width) {
#pragma unroll
        for (int lane = 0; lane < kChannelsPerThread; lane += 2) {
            const float2 values = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(prepared + offset + lane));
            normalized[position][channel + lane]     = values.x;
            normalized[position][channel + lane + 1] = values.y;
        }
    }
    __syncthreads();
    if (tid < 2 * Capacity && tid % Capacity < width) {
        const int tap = tid / Capacity, pos = tid % Capacity;
        finish[((batch * width + pos) * 2 + tap) * kGroups + group] =
            __float2bfloat16_rn(projected[2 + tap][pos]);
    }
    if (position < width) {
#pragma unroll
        for (int lane = 0; lane < kChannelsPerThread; lane += 2) {
            const float2 scale = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(base + hidden + lane));
            const float2 shift = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(base + Hidden + hidden + lane));
            float2 value;
            value.x = (scale.x + projected[0][position]) * normalized[position][channel + lane];
            value.y = (scale.y + projected[0][position]) * normalized[position][channel + lane + 1];
            if (position > 0) {
                value.x = fmaf(shift.x + projected[1][position],
                               normalized[position - 1][channel + lane], value.x);
                value.y = fmaf(shift.y + projected[1][position],
                               normalized[position - 1][channel + lane + 1], value.y);
            }
            *reinterpret_cast<__nv_bfloat162*>(prepared + offset + lane) =
                __floats2bfloat162_rn(value.x, value.y);
        }
    }
}

template <int Hidden, int Capacity>
void launch(DynamicConvPrepareRoute route, const Tensor& base, const float* partial,
            Tensor& prepared, Tensor& finish, cudaStream_t stream) {
    const dim3 grid(Hidden / 16, prepared.ne[2]);
    const dim3 block(Capacity * (16 / kChannelsPerThread));
    CUDA_CHECK(pdl::launch_consumer({dim3(grid), block, 0, stream},
                                    dynamic_grouped_conv_prepare_reduce_kernel<Hidden, Capacity>,
                                    static_cast<const __nv_bfloat16*>(base.data), partial,
                                    static_cast<__nv_bfloat16*>(prepared.data),
                                    static_cast<__nv_bfloat16*>(finish.data), prepared.ne[1],
                                    prepared.ne[2], route.split_k));
    CUDA_CHECK(cudaGetLastError());
}

template <int Hidden>
void launch_width(DynamicConvPrepareRoute route, const Tensor& base, const float* partial,
                  Tensor& prepared, Tensor& finish, cudaStream_t stream) {
    // Capacity determines the CTA layout; width determines every public tensor address.
    if (prepared.ne[1] <= 8)
        launch<Hidden, 8>(route, base, partial, prepared, finish, stream);
    else
        launch<Hidden, 16>(route, base, partial, prepared, finish, stream);
}
} // namespace

void bf16_dynamic_grouped_conv_prepare_reduce_launch(DynamicConvPrepareRoute route,
                                                     const Tensor& base, const float* partial,
                                                     Tensor& prepared, Tensor& finish,
                                                     cudaStream_t stream) {
    switch (prepared.ne[0]) {
    case 5120:
        return launch_width<5120>(route, base, partial, prepared, finish, stream);
    case 2560:
        return launch_width<2560>(route, base, partial, prepared, finish, stream);
    case 3072:
        return launch_width<3072>(route, base, partial, prepared, finish, stream);
    }
    throw std::logic_error("dynamic grouped conv prepare: unregistered hidden width");
}
} // namespace infernix::ops::detail
