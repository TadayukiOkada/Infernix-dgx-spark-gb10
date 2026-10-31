// Q8 W8A8 fast-path prefill (q8_a8.h): the BF16 activation columns of a batch are encoded once by
// q8_a8_quantize_x_kernel into ggml_a8.h's codes and slice scales, then the 128x128x64 int8 Tensor
// Core tiles of q8_a8_mma.cuh sweep the rows. One encode per batch, one tile launch per batch over
// every row tile; an output's arithmetic is the same per K-group term whatever the batching.
#include "ops/linear/q8/q8_a8.h"

#include "core/device.h"
#include "ops/linear/q8/q8_a8_mma.cuh"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {

constexpr std::int32_t kQuantizeThreads = kQ8A8Threads;

struct Problem {
    std::int32_t n;
    std::int32_t k;
};

// Every registered Q8 problem whose K slices into whole 32-element groups and complete 64-element
// MMA stages, except the vocabulary head (its FP32 logits stay on the exact route) and the 324-row
// hyper-connection downs (their rows are not 16-row multiples). The runtime-shape problems are not
// registered: the route answers only for the tuned table.
constexpr Problem kProblems[] = {
    {1024, 2048},   {1024, 5120},   {2048, 4096},   {2048, 4608},   {2048, 16384},
    {4608, 4608},   {5120, 4608},   {5120, 6144},   {5120, 10240},  {5120, 17408},
    {5120, 25600},  {6144, 5120},   {9216, 2048},   {12288, 2048},  {14336, 5120},
    {17408, 5120},  {34816, 5120},  {16384, 2560},  {12800, 2560},  {13952, 2560},
    {1024, 2560},   {128, 2560},    {2560, 2560},   {1280, 2560},   {2560, 6144},
    {2560, 12800},  {2560, 640},    {6144, 2560},   {10240, 2560},  {10240, 5120},
    {12288, 2560},  {320, 10240},   {10240, 320},  {25600, 5120},
};

std::size_t align256(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

template <bool Full>
void launch_tiles(const std::int32_t* xcodes, const float* xscales, const std::uint8_t* codes,
                  const std::uint16_t* wscales, LinearBf16Output output, int rows, int k, int slices,
                  int kg, int count, int live, int column_begin, cudaStream_t stream) {
    constexpr auto kernel = q8_a8_mma_kernel<Full, LinearBf16Output>;
    constexpr int shared  = kQ8A8SharedBytes;
    // The stage ring is static shared below 48 KiB (q8_shared_storage's branch); q8_prepare_shared
    // returns the dynamic size to launch with (0 while the ring is static).
    const int dynamic_shared = q8_prepare_shared<shared, kernel>();
    const dim3 grid((rows + kQ8A8BlockRows - 1) / kQ8A8BlockRows,
                    (count + kQ8A8BlockTokens - 1) / kQ8A8BlockTokens);
    kernel<<<grid, kQ8A8Threads, dynamic_shared, stream>>>(xcodes, xscales, codes, wscales, output,
                                                           rows, k, slices, kg, count, live,
                                                           column_begin);
    CUDA_CHECK(cudaGetLastError());
}

void launch(const std::int32_t* xcodes, const float* xscales, const std::uint8_t* codes,
            const std::uint16_t* wscales, LinearBf16Output output, int rows, int k, int slices, int kg,
            int count, int live, int column_begin, cudaStream_t stream) {
    if (rows % kQ8A8BlockRows == 0 && count % kQ8A8BlockTokens == 0) {
        launch_tiles<true>(xcodes, xscales, codes, wscales, output, rows, k, slices, kg, count, live,
                           column_begin, stream);
    } else {
        launch_tiles<false>(xcodes, xscales, codes, wscales, output, rows, k, slices, kg, count, live,
                            column_begin, stream);
    }
}

} // namespace

bool q8_a8_problem(std::int32_t n, std::int32_t k) noexcept {
    if (k % 64 != 0) { return false; }
    for (const auto& p : kProblems) {
        if (p.n == n && p.k == k) { return true; }
    }
    return false;
}

bool q8_a8_admits(std::int32_t n, std::int32_t k, std::int32_t tokens) noexcept {
    return tokens >= kQ8A8MinTokens && q8_a8_problem(n, k);
}

std::size_t q8_a8_workspace_capacity_bytes(std::int32_t n, std::int32_t k, std::int32_t max_tokens) {
    if (!q8_a8_admits(n, k, max_tokens)) { return 0; }
    const std::size_t batch = std::min<std::int32_t>(max_tokens, kQ8A8BatchColumns);
    // The codes live in 16-column fragment blocks; a partial block still costs a whole block.
    const std::size_t coded = (batch + 15) / 16 * 16;
    const std::size_t slices = static_cast<std::size_t>(k) / 32;
    return align256(coded * slices * 8 * sizeof(std::int32_t)) +
           align256(batch * slices * sizeof(float));
}

void q8_linear_a8(const Tensor& x, const Weight& weight, Tensor& out, WorkspaceArena& workspace,
                  cudaStream_t stream) {
    if (weight.qtype != QType::Q8_G32_FP16 || out.dtype != DType::BF16 ||
        !q8_a8_admits(weight.n, weight.k, x.ne[1]) || weight.padded_shape[1] % 128 != 0) {
        throw std::invalid_argument("q8 a8: the call is outside the route's problems or widths");
    }
    const auto aligned = [](const void* pointer, std::uintptr_t alignment) {
        return pointer && (reinterpret_cast<std::uintptr_t>(pointer) % alignment) == 0;
    };
    if (!aligned(x.data, 16) || !aligned(out.data, 16) || !aligned(weight.qdata, 16) ||
        !aligned(weight.scales, 16)) {
        throw std::invalid_argument("q8 a8: the call needs aligned activation, weight and output planes");
    }

    auto scope = workspace.scope();
    const int k = weight.k, slices = k / 32;
    const int kg = weight.padded_shape[1] / 32;
    const int tokens = x.ne[1];
    const int batch  = std::min(tokens, kQ8A8BatchColumns);
    auto* codes = static_cast<std::int32_t*>(
        workspace
            .alloc_bytes(static_cast<std::size_t>((batch + 15) / 16 * 16) * slices * 8 *
                         sizeof(std::int32_t))
            .data);
    auto* xscales = static_cast<float*>(
        workspace.alloc_bytes(static_cast<std::size_t>(batch) * slices * sizeof(float)).data);
    const LinearBf16Output output{static_cast<__nv_bfloat16*>(out.data), weight.n};

    for (int first = 0; first < tokens; first += batch) {
        const int count = std::min(batch, tokens - first);
        const int items = count * slices; // (column, slice) items, four threads each
        q8_a8_quantize_x_kernel<<<(items * 4 + kQuantizeThreads - 1) / kQuantizeThreads,
                                  kQuantizeThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data) + static_cast<std::size_t>(first) * k, k,
            slices, codes, xscales, items);
        CUDA_CHECK(cudaGetLastError());
        launch(codes, xscales, static_cast<const std::uint8_t*>(weight.qdata),
               static_cast<const std::uint16_t*>(weight.scales), output, weight.n, k, slices, kg,
               count, tokens, first, stream);
    }
}

} // namespace infernix::ops::detail
