#pragma once

// Narrow private BF16 vector storage shared by elementwise kernels. Arithmetic
// remains Op-specific so each contract keeps its exact FP32 operation order and
// BF16 rounding boundary.

#include <cuda_bf16.h>

#include <cstdint>

namespace infernix::ops {

template <int Pairs>
struct alignas(Pairs* static_cast<int>(sizeof(__nv_bfloat162))) Bf16PairPack {
    static_assert(Pairs == 1 || Pairs == 2 || Pairs == 4);
    __nv_bfloat162 pair[Pairs];
};

using Bf16x4Pack = Bf16PairPack<2>;
using Bf16x8Pack = Bf16PairPack<4>;

// On GB10 (24 MiB L2, 1/4 of RTX 5090's 96 MiB L2), explicit 16-byte packs win while a BF16
// activation remains in the cache-sized regime (8M elements = 16 MiB). AddBias and GELU switch to
// their higher-occupancy BF16x2 streams above this finite boundary.
inline constexpr std::int64_t kBf16x8CacheSizedMaxElements = 8LL * 1024LL * 1024LL;

static_assert(sizeof(Bf16x4Pack) == 8);
static_assert(sizeof(Bf16x8Pack) == 16);

} // namespace infernix::ops
