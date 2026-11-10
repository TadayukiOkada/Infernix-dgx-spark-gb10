#include "ops/ggml/ggml_linear_internal.cuh"

#include <algorithm>
#include <vector>

namespace infernix::ops {
namespace {

constexpr int kTiledMaxThreads = 256;

__device__ __forceinline__ void cp_async16_evict_first(void* shared, const void* global, std::uint64_t policy) {
    const auto s = static_cast<unsigned>(__cvta_generic_to_shared(shared));
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;\n" ::"r"(s), "l"(global), "l"(policy));
}

__device__ __forceinline__ void cp_async_wait_upto(int n) {
    switch (n) {
    case 0: ggml::tile::cp_async_wait<0>(); break;
    case 1: ggml::tile::cp_async_wait<1>(); break;
    case 2: ggml::tile::cp_async_wait<2>(); break;
    default: ggml::tile::cp_async_wait<3>(); break;
    }
}

std::size_t tiled_shared_bytes(int warps, int split, int stages, int slot_bytes) {
    const std::size_t rings    = static_cast<std::size_t>(warps) * stages * slot_bytes;
    const std::size_t partials = split > 1 ? static_cast<std::size_t>(warps / split) * (split - 1) * 32 * 4 * sizeof(ggml::I128) : 0;
    return rings + partials;
}

template <QType F, bool OutF32>
__device__ __forceinline__ void tiled_mma_body(const GroupSlot& seg, int slices, const std::int32_t* __restrict__ encoded,
                                               const std::int32_t* __restrict__ emax, int columns, int split, int stages,
                                               int slot_bytes, unsigned char* shared) {
    static_assert(kGgmlTiledFormat<F>, "tiled rows: formats of 256-element blocks");
    constexpr int kBlockBytes = static_cast<int>(ggml_block_bytes(F));
    constexpr int kRun        = kGgmlTiledRun<F>;
    static_assert(kRun % 16 == 0, "a tile's block run is whole 16-byte copies");
    void* const out = seg.out;
    const int warps = static_cast<int>(blockDim.x) / 32, warp = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const int per_cta = warps / split, slot = warp / split, part = warp % split;
    const int tiles = (seg.rows + 15) / 16, tile = seg.block * per_cta + slot;
    const bool active = tile < tiles;
    const int blocks  = slices / 8;
    const int mine    = active && part < blocks ? (blocks - part + split - 1) / split : 0;
    const std::uint8_t* tile_base = seg.weights + static_cast<std::size_t>(active ? tile : 0) * 16 * seg.row_bytes;
    unsigned char* ring = shared + static_cast<std::size_t>(warp) * stages * slot_bytes;
    const std::size_t column_words = static_cast<std::size_t>(slices) * (fast::kSliceWords + 1);
    std::uint64_t policy;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
    const auto issue = [&](int i) {
        if (i < mine) {
            const std::uint8_t* src = tile_base + static_cast<std::size_t>(part + i * split) * kRun;
            unsigned char* dst      = ring + (i % stages) * slot_bytes;
            for (int c = lane; c < kRun / 16; c += 32) { cp_async16_evict_first(dst + 16 * c, src + 16 * c, policy); }
        }
        ggml::tile::cp_async_commit();
    };
    const int g = lane / 4, t = lane % 4;
    const std::int32_t* bcol = encoded + static_cast<std::size_t>(min(g, columns - 1)) * column_words + 2 * t;
    const std::int32_t* xs0  = encoded + static_cast<std::size_t>(min(2 * t, columns - 1)) * column_words +
                              static_cast<std::size_t>(slices) * fast::kSliceWords;
    const std::int32_t* xs1 = encoded + static_cast<std::size_t>(min(2 * t + 1, columns - 1)) * column_words +
                              static_cast<std::size_t>(slices) * fast::kSliceWords;
    ggml::I128 acc[4];
    for (int i = 0; i < stages - 1; ++i) { issue(i); }
    pdl::wait_for_dependencies();
    for (int i = 0; i < mine; ++i) {
        __syncwarp();
        issue(i + stages - 1);
        cp_async_wait_upto(stages - 1);
        __syncwarp();
        const unsigned char* run = ring + (i % stages) * slot_bytes;
        ggml::mma_rows::unit_sums<F>(acc, run + g * kBlockBytes, run + (g + 8) * kBlockBytes, part + i * split, 8, bcol, xs0,
                                     xs1, t);
    }
    ggml::tile::cp_async_wait<0>();
    if (split > 1) {
        auto* partial = reinterpret_cast<unsigned long long*>(shared + static_cast<std::size_t>(warps) * stages * slot_bytes);
        const auto at = [&](int p, int q) {
            return partial + ((static_cast<std::size_t>(slot) * (split - 1) + (p - 1)) * 32 + lane) * 8 + 2 * q;
        };
        if (part > 0) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                at(part, q)[0] = acc[q].lo;
                at(part, q)[1] = static_cast<unsigned long long>(acc[q].hi);
            }
        }
        __syncthreads();
        if (part == 0) {
            for (int p = 1; p < split; ++p) {
#pragma unroll
                for (int q = 0; q < 4; ++q) { ggml::add(acc[q], ggml::I128{at(p, q)[0], static_cast<std::int64_t>(at(p, q)[1])}); }
            }
        }
    }
    if (part != 0 || !active) { return; }
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int row = tile * 16 + g + 8 * (q >> 1), c = 2 * t + (q & 1);
        if (row < seg.rows && c < columns) {
            const std::size_t o = static_cast<std::size_t>(c) * seg.n + seg.out_offset + row;
            if constexpr (OutF32) {
                static_cast<float*>(out)[o] = ggml::output_f32(acc[q], emax[c], F);
            } else {
                static_cast<std::uint16_t*>(out)[o] = ggml::output_bf16(acc[q], emax[c], F);
            }
        }
    }
}

template <QType F, bool OutF32>
__global__ void __launch_bounds__(kTiledMaxThreads, 2)
    tiled_mma_kernel(RowsGroup grp, int slices, const std::int32_t* __restrict__ encoded,
                     const std::int32_t* __restrict__ emax, int columns, int split, int stages, int slot_bytes) {
    extern __shared__ __align__(16) unsigned char tiled_shared[];
    tiled_mma_body<F, OutF32>(group_slot(grp), slices, encoded, emax, columns, split, stages, slot_bytes, tiled_shared);
}

template <QType F0, QType F1, bool OutF32>
__global__ void __launch_bounds__(kTiledMaxThreads, 2)
    tiled_mma_mixed_kernel(RowsGroup grp, int slices, const std::int32_t* __restrict__ encoded,
                           const std::int32_t* __restrict__ emax, int columns, int split, int stages, int slot_bytes) {
    extern __shared__ __align__(16) unsigned char tiled_shared[];
    const GroupSlot seg = group_slot(grp);
    if (seg.format == 0) {
        tiled_mma_body<F0, OutF32>(seg, slices, encoded, emax, columns, split, stages, slot_bytes, tiled_shared);
    } else {
        tiled_mma_body<F1, OutF32>(seg, slices, encoded, emax, columns, split, stages, slot_bytes, tiled_shared);
    }
}

struct TiledShape {
    int warps  = 4;
    int split  = 1;
    int stages = 2;
};
constexpr TiledShape kTiledShapes[] = {{8, 8, 2}, {4, 4, 2}, {2, 2, 2}, {4, 2, 2}, {4, 1, 2}, {2, 1, 2}};
constexpr TiledShape kTiledManyWaves{4, 1, 2};
constexpr int kTiledWarpsPerSm = 6;
constexpr int kTiledMinBlocks  = 4;

struct TiledDevice {
    int sms   = 0;
    int limit = 0;
    int l2    = 0;
};

TiledDevice tiled_device() {
    static const TiledDevice d = [] {
        TiledDevice out{0, 0, 0};
        int current = 0;
        CUDA_CHECK(cudaGetDevice(&current));
        CUDA_CHECK(cudaDeviceGetAttribute(&out.sms, cudaDevAttrMultiProcessorCount, current));
        CUDA_CHECK(cudaDeviceGetAttribute(&out.limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, current));
        CUDA_CHECK(cudaDeviceGetAttribute(&out.l2, cudaDevAttrL2CacheSize, current));
        return out;
    }();
    return d;
}

template <class Kernel>
void tiled_allow_shared(Kernel kernel, int limit) {
    thread_local std::vector<const void*> done;
    const void* key = reinterpret_cast<const void*>(kernel);
    if (std::find(done.begin(), done.end(), key) != done.end()) { return; }
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, limit));
    done.push_back(key);
}

template <class Kernel>
int tiled_resident(Kernel kernel, int threads, std::size_t shared) {
    struct Entry {
        const void* kernel;
        int threads;
        std::size_t shared;
        int resident;
    };
    thread_local std::vector<Entry> cache;
    const void* key = reinterpret_cast<const void*>(kernel);
    for (const Entry& e : cache) {
        if (e.kernel == key && e.threads == threads && e.shared == shared) { return e.resident; }
    }
    int resident = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, kernel, threads, shared));
    cache.push_back({key, threads, shared, resident});
    return resident;
}

template <class Kernel>
TiledShape tiled_shape(Kernel kernel, int tiles, int blocks, int slot_bytes, const TiledDevice& dev) {
    const int warp_budget = dev.sms * kTiledWarpsPerSm;
    for (const TiledShape& s : kTiledShapes) {
        if (s.split > 1 && tiles * s.split > warp_budget && blocks < kTiledMinBlocks * s.split) { continue; }
        const std::size_t shared = tiled_shared_bytes(s.warps, s.split, s.stages, slot_bytes);
        if (shared > static_cast<std::size_t>(dev.limit)) { continue; }
        const int per_cta = s.warps / s.split;
        const int ctas    = (tiles + per_cta - 1) / per_cta;
        if (ctas <= tiled_resident(kernel, 32 * s.warps, shared) * dev.sms) { return s; }
    }
    return kTiledManyWaves;
}

bool tiled_rows_route(QType format, int columns, int total_rows) {
    return columns == 1 && (format == QType::GGML_Q6_K || format == QType::GGML_IQ4_XS) && total_rows < 8192;
}

template <QType F, bool OutF32>
void launch_rows_tiled(RowsGroup g, int k, const Columns& in, int columns, cudaStream_t stream) {
    const int row_bytes = static_cast<int>(k / ggml_block_elems(F) * ggml_block_bytes(F));
    const int slices    = k / 32;
    const int lanes     = ggml_lanes_for(slices);
    const int per_cta   = kGgmlWarps * (32 / lanes);
    g.first[0]          = 0;
    for (int i = 0; i < g.count; ++i) { g.first[i + 1] = g.first[i] + (g.rows[i] + per_cta - 1) / per_cta; }
    const dim3 grid(g.first[g.count]);
#define INFERNIX_GGML_TILED_LANES(LN)                                                                                  \
    case LN:                                                                                                           \
        if (columns == 1) {                                                                                            \
            CUDA_CHECK(pdl::launch_consumer({grid, dim3(kGgmlThreads), 0, stream},                                     \
                                            rows_kernel<F, OutF32, LN, 1, true>, g, row_bytes, slices,                 \
                                            in.encoded, in.emax, columns));                                             \
        } else if (columns > kGgmlColumns && columns <= 2 * kGgmlColumns) {                                            \
            CUDA_CHECK(pdl::launch_consumer({grid, dim3(kGgmlThreads), 0, stream},                                     \
                                            rows_kernel<F, OutF32, LN, 2 * kGgmlColumns, true>, g, row_bytes, slices,   \
                                            in.encoded, in.emax, columns));                                             \
        } else {                                                                                                       \
            CUDA_CHECK(pdl::launch_consumer({grid, dim3(kGgmlThreads), 0, stream},                                     \
                                            rows_kernel<F, OutF32, LN, kGgmlColumns, true>, g, row_bytes, slices,       \
                                            in.encoded, in.emax, columns));                                             \
        }                                                                                                              \
        break;
    switch (lanes) {
        INFERNIX_GGML_TILED_LANES(4)
        INFERNIX_GGML_TILED_LANES(8)
        INFERNIX_GGML_TILED_LANES(16)
        INFERNIX_GGML_TILED_LANES(32)
    default:
        throw std::logic_error("ggml_linear: lanes");
    }
#undef INFERNIX_GGML_TILED_LANES
    ggml_check_launch("rows tiled");
}

template <class Kernel>
void launch_tiled_mma(Kernel kernel, RowsGroup g, int k, const Columns& in, int columns, int slot_bytes,
                      cudaStream_t stream, std::size_t out_bytes = 2) {
    int tiles = 0;
    for (int i = 0; i < g.count; ++i) { tiles += (g.rows[i] + 15) / 16; }
    const TiledDevice dev = tiled_device();
    tiled_allow_shared(kernel, dev.limit);
    const TiledShape shape = tiled_shape(kernel, tiles, k / 256, slot_bytes, dev);
    const int per_cta      = shape.warps / shape.split;
    g.first[0]             = 0;
    for (int i = 0; i < g.count; ++i) { g.first[i + 1] = g.first[i] + ((g.rows[i] + 15) / 16 + per_cta - 1) / per_cta; }
    const std::size_t shared       = tiled_shared_bytes(shape.warps, shape.split, shape.stages, slot_bytes);
    const int batches              = (columns + kGgmlMmaMaxColumns - 1) / kGgmlMmaMaxColumns;
    const int span                 = (columns + batches - 1) / batches;
    const int slices               = k / 32;
    const std::size_t column_words = static_cast<std::size_t>(slices) * (fast::kSliceWords + 1);
    for (int first = 0; first < columns; first += span) {
        const int count = std::min(span, columns - first);
        if (count == columns) {
            CUDA_CHECK(pdl::launch_consumer({dim3(g.first[g.count]), dim3(32 * shape.warps), shared, stream}, kernel, g,
                                            slices, in.encoded, in.emax, columns, shape.split, shape.stages, slot_bytes));
            continue;
        }
        RowsGroup part = g;
        for (int i = 0; i < part.count; ++i) {
            part.out[i] = static_cast<std::byte*>(g.out[i]) + static_cast<std::size_t>(first) * g.n[i] * out_bytes;
        }
        CUDA_CHECK(pdl::launch_consumer({dim3(g.first[g.count]), dim3(32 * shape.warps), shared, stream}, kernel, part,
                                        slices, in.encoded + static_cast<std::size_t>(first) * column_words,
                                        in.emax + first, count, shape.split, shape.stages, slot_bytes));
    }
    ggml_check_launch("tiled mma");
}

template <QType F, bool OutF32>
void launch_tiled_format(RowsGroup g, int k, const Columns& in, int columns, cudaStream_t stream) {
    if constexpr (!kGgmlTiledFormat<F>) {
        throw std::logic_error("ggml_linear: a tiled segment's format has no tiled kernels");
    } else {
        int total_rows = 0;
        for (int i = 0; i < g.count; ++i) { total_rows += g.rows[i]; }
        if (tiled_rows_route(F, columns, total_rows)) {
            launch_rows_tiled<F, OutF32>(g, k, in, columns, stream);
            return;
        }
        for (int i = 0; i < g.count; ++i) { g.row_bytes[i] = static_cast<int>(k / ggml_block_elems(F) * ggml_block_bytes(F)); }
        launch_tiled_mma(tiled_mma_kernel<F, OutF32>, g, k, in, columns, kGgmlTiledRun<F>, stream, OutF32 ? 4 : 2);
    }
}

template <QType F0, QType F1>
void launch_tiled_mixed_pair(const RowsGroup& g, int k, const Columns& in, int columns, cudaStream_t stream) {
    launch_tiled_mma(tiled_mma_mixed_kernel<F0, F1, false>, g, k, in, columns,
                     std::max(kGgmlTiledRun<F0>, kGgmlTiledRun<F1>), stream);
}

} // namespace

void ggml_launch_tiled(QType format, bool out_f32, const RowsGroup& g, int k, const Columns& in, int columns,
                       cudaStream_t stream) {
#define INFERNIX_GGML_TILED_DISPATCH(Q)                                                                                \
    case Q:                                                                                                            \
        if (out_f32) {                                                                                                 \
            launch_tiled_format<Q, true>(g, k, in, columns, stream);                                                   \
        } else {                                                                                                       \
            launch_tiled_format<Q, false>(g, k, in, columns, stream);                                                  \
        }                                                                                                              \
        break;
    switch (format) {
        INFERNIX_GGML_FORMATS(INFERNIX_GGML_TILED_DISPATCH)
    default:
        throw std::invalid_argument("ggml_linear: segment format is not a ggml block format");
    }
#undef INFERNIX_GGML_TILED_DISPATCH
}

bool ggml_launch_tiled_mixed(QType f0, QType f1, const RowsGroup& g, int k, const Columns& in, int columns,
                             cudaStream_t stream) {
#define INFERNIX_GGML_TILED_PAIR(A, B)                                                                                 \
    if (f0 == QType::A && f1 == QType::B) {                                                                            \
        launch_tiled_mixed_pair<QType::A, QType::B>(g, k, in, columns, stream);                                        \
        return true;                                                                                                   \
    }
    INFERNIX_GGML_TILED_PAIR(GGML_Q4_K, GGML_Q5_K)
    INFERNIX_GGML_TILED_PAIR(GGML_Q4_K, GGML_Q6_K)
    INFERNIX_GGML_TILED_PAIR(GGML_Q4_K, GGML_IQ4_XS)
    INFERNIX_GGML_TILED_PAIR(GGML_Q5_K, GGML_Q6_K)
    INFERNIX_GGML_TILED_PAIR(GGML_Q5_K, GGML_IQ4_XS)
    INFERNIX_GGML_TILED_PAIR(GGML_Q6_K, GGML_IQ4_XS)
#undef INFERNIX_GGML_TILED_PAIR
    return false;
}

} // namespace infernix::ops
