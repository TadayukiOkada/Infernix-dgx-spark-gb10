#pragma once

#include <cuda_pipeline.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ops {

enum class Cache { ca, cg, cg_evict_first };

template <class V, class T>
__device__ __forceinline__ V load_vec(const T* ptr) {
    static_assert(sizeof(V) == 1 || sizeof(V) == 2 || sizeof(V) == 4 || sizeof(V) == 8 ||
                  sizeof(V) == 16);
    return *reinterpret_cast<const V*>(ptr);
}

template <class V, class T>
__device__ __forceinline__ V load_ldg(const T* ptr) {
    static_assert(sizeof(V) == 1 || sizeof(V) == 2 || sizeof(V) == 4 || sizeof(V) == 8 ||
                  sizeof(V) == 16);
    return __ldg(reinterpret_cast<const V*>(ptr));
}

// Read-only global load that does not allocate in L1: weights streamed once per call.
template <class V, class T>
__device__ __forceinline__ V ld_nc_na(const T* ptr) {
    static_assert(sizeof(V) == 2 || sizeof(V) == 4 || sizeof(V) == 8 || sizeof(V) == 16);
    if constexpr (sizeof(V) == 2) {
        unsigned short value;
        asm("ld.global.nc.L1::no_allocate.u16 %0, [%1];" : "=h"(value) : "l"(ptr));
        return *reinterpret_cast<const V*>(&value);
    } else if constexpr (sizeof(V) == 4) {
        unsigned value;
        asm("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(value) : "l"(ptr));
        return *reinterpret_cast<const V*>(&value);
    } else if constexpr (sizeof(V) == 8) {
        uint2 value;
        asm("ld.global.nc.L1::no_allocate.v2.u32 {%0, %1}, [%2];" : "=r"(value.x), "=r"(value.y) : "l"(ptr));
        return *reinterpret_cast<const V*>(&value);
    } else {
        uint4 value;
        asm("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
            : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
            : "l"(ptr));
        return *reinterpret_cast<const V*>(&value);
    }
}

__device__ __forceinline__ std::uint64_t l2_evict_first_policy() {
    std::uint64_t policy;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
    return policy;
}

// ld_nc_na for bytes read once per call that should leave L2 first: an L2 evict-first policy, so a
// large stream (a full-vocabulary head) does not push out data other kernels reuse.
template <class V, class T>
__device__ __forceinline__ V ld_nc_na_evict_first_vec(const T* ptr, std::uint64_t policy) {
    static_assert(sizeof(V) == 2 || sizeof(V) == 4 || sizeof(V) == 8 || sizeof(V) == 16);
    if constexpr (sizeof(V) == 2) {
        unsigned short value;
        asm("ld.global.nc.L1::no_allocate.L2::cache_hint.u16 %0, [%1], %2;"
            : "=h"(value)
            : "l"(ptr), "l"(policy));
        return *reinterpret_cast<const V*>(&value);
    } else if constexpr (sizeof(V) == 4) {
        unsigned value;
        asm("ld.global.nc.L1::no_allocate.L2::cache_hint.u32 %0, [%1], %2;"
            : "=r"(value)
            : "l"(ptr), "l"(policy));
        return *reinterpret_cast<const V*>(&value);
    } else if constexpr (sizeof(V) == 8) {
        uint2 value;
        asm("ld.global.nc.L1::no_allocate.L2::cache_hint.v2.u32 {%0, %1}, [%2], %3;"
            : "=r"(value.x), "=r"(value.y)
            : "l"(ptr), "l"(policy));
        return *reinterpret_cast<const V*>(&value);
    } else {
        uint4 value;
        asm("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;"
            : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
            : "l"(ptr), "l"(policy));
        return *reinterpret_cast<const V*>(&value);
    }
}

template <class V, class T>
__device__ __forceinline__ V ld_nc_na_evict_first_vec(const T* ptr) {
    return ld_nc_na_evict_first_vec<V>(ptr, l2_evict_first_policy());
}

__device__ __forceinline__ uint2 ld_nc_na_evict_first(const void* ptr) {
    std::uint64_t policy;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
    uint2 value;
    asm("ld.global.nc.L1::no_allocate.L2::cache_hint.v2.u32 {%0, %1}, [%2], %3;"
        : "=r"(value.x), "=r"(value.y)
        : "l"(ptr), "l"(policy));
    return value;
}

// The 4-byte form, for a packed row codec whose one-lane unit is a single 32-bit word (q4 nibbles).
__device__ __forceinline__ std::uint32_t ld_nc_na_evict_first_u32(const void* ptr) {
    std::uint64_t policy;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
    std::uint32_t value;
    asm("ld.global.nc.L1::no_allocate.L2::cache_hint.u32 %0, [%1], %2;" : "=r"(value) : "l"(ptr), "l"(policy));
    return value;
}

template <class T, class V>
__device__ __forceinline__ void store_vec(T* ptr, V value) {
    static_assert(sizeof(V) == 1 || sizeof(V) == 2 || sizeof(V) == 4 || sizeof(V) == 8 ||
                  sizeof(V) == 16);
    *reinterpret_cast<V*>(ptr) = value;
}

__device__ __forceinline__ unsigned smem_addr(const void* ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void cp_async(void* smem_dst, const void* gmem_src) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "cp_async supports 4, 8, or 16 bytes");
    if constexpr (Policy == Cache::cg_evict_first) {
        static_assert(Bytes == 16, "cp.async.cg.L2::cache_hint requires a 16-byte copy");
        const std::uint64_t policy = l2_evict_first_policy();
        unsigned dst               = smem_addr(smem_dst);
        // Prevent sm_121a ptxas from folding extern __shared__ base into LDGSTS [R+UR], desc[UR1].
        asm("" : "+r"(dst));
        asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;\n"
                     :
                     : "r"(dst), "l"(gmem_src), "l"(policy));
    } else if constexpr (Policy == Cache::cg) {
        static_assert(Bytes == 16, "cp.async.cg requires a 16-byte copy");
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src));
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src), "n"(Bytes));
    }
}

template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void cp_async_hinted(void* smem_dst, const void* gmem_src,
                                                std::uint64_t policy) {
    if constexpr (Policy == Cache::cg_evict_first) {
        static_assert(Bytes == 16, "cp.async.cg.L2::cache_hint requires a 16-byte copy");
        unsigned dst = smem_addr(smem_dst);
        // Prevent sm_121a ptxas from folding extern __shared__ base into LDGSTS [R+UR], desc[UR1].
        asm("" : "+r"(dst));
        asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;\n"
                     :
                     : "r"(dst), "l"(gmem_src), "l"(policy));
    } else {
        cp_async<Bytes, Policy>(smem_dst, gmem_src);
    }
}

template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void cp_async_zfill(void* smem_dst, const void* gmem_src,
                                               int src_bytes) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16,
                  "cp_async_zfill supports 4, 8, or 16 bytes");
    if constexpr (Policy == Cache::cg_evict_first) {
        static_assert(Bytes == 16, "cp.async.cg.L2::cache_hint requires a 16-byte copy");
        const std::uint64_t policy = l2_evict_first_policy();
        unsigned dst               = smem_addr(smem_dst);
        asm("" : "+r"(dst));
        asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2, %3;\n"
                     :
                     : "r"(dst), "l"(gmem_src), "r"(src_bytes), "l"(policy));
    } else if constexpr (Policy == Cache::cg) {
        static_assert(Bytes == 16, "cp.async.cg requires a 16-byte copy");
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src), "r"(src_bytes));
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src), "n"(Bytes), "r"(src_bytes));
    }
}

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }

template <int Groups>
__device__ __forceinline__ void cp_wait() {
    static_assert(Groups >= 0 && Groups <= 7, "cp_wait group count must fit the PTX immediate");
    asm volatile("cp.async.wait_group %0;\n" : : "n"(Groups));
}

template <int Bytes>
__device__ __forceinline__ void pipe_copy(void* smem_dst, const void* gmem_src) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "pipe_copy supports 4, 8, or 16 bytes");
    __pipeline_memcpy_async(smem_dst, gmem_src, Bytes);
}

__device__ __forceinline__ void pipe_commit() { __pipeline_commit(); }

template <int Groups>
__device__ __forceinline__ void pipe_wait() {
    static_assert(Groups >= 0 && Groups <= 7, "pipe_wait group count must fit the PTX immediate");
    __pipeline_wait_prior(Groups);
}

} // namespace infernix::ops
