// device_probe.cu — authoritative hardware snapshot for the CUDA Kernel Toolkit.
//
// The toolkit refuses to hardcode GPU constants.  When the CUDA runtime is
// available this probe is compiled (with the *detected* arch, queried via
// `nvcc --list-gpu-arch`) and executed to emit a flat, line-oriented record
// set on stdout:
//
//     key<TAB>value
//
// where `key` is a dotted path such as `device.0.multi_processor_count` and
// `value` is a number or a string.  Python parses this into nested
// dictionaries with numeric coercion.  A line-oriented format is used instead
// of hand-rolled JSON so the C++ side has no escaping/formatting logic that can
// silently corrupt the snapshot.
//
// Every field is a device attribute reported by the CUDA runtime.  Derived
// limits (occupancy, register allocation, bandwidth ceilings) are computed
// downstream from these measured facts, never from a model table.
//
// The probe allocates nothing on the device and launches no kernels; it is
// safe to run alongside other work.
//
// Build (no NVML dependency, always works):
//   nvcc -arch=<detected> -O2 -std=c++17 -o device_probe device_probe.cu
// Build (with power/clock telemetry, needs libnvidia-ml):
//   nvcc -arch=<detected> -O2 -std=c++17 -DHAVE_NVML -o device_probe \
//        device_probe.cu -lnvidia-ml
// Run:
//   ./device_probe            -> records on stdout

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cuda_runtime.h>

#ifdef HAVE_NVML
#include <nvml.h>
#endif

namespace {

void emit_int(const char* key, long long value) {
    std::printf("%s\t%lld\n", key, value);
}

// Emits nothing when the attribute is unavailable, so the parser reports it as
// missing rather than fabricating a zero.
void emit_str(const char* key, const char* value) {
    if (value) std::printf("%s\t%s\n", key, value);
}

void emit_flag(const char* key, bool value) { emit_int(key, value ? 1 : 0); }

}  // namespace

int main() {
    int device_count = 0;
    cudaError_t rc = cudaGetDeviceCount(&device_count);

    emit_int("probe.schema", 1);
    emit_int("runtime.cudart_version", static_cast<int>(CUDART_VERSION));
    {
        int drv = 0;
        if (cudaDriverGetVersion(&drv) == cudaSuccess) emit_int("runtime.driver_version", drv);
        int rt = 0;
        if (cudaRuntimeGetVersion(&rt) == cudaSuccess) emit_int("runtime.runtime_version", rt);
    }

    if (rc != cudaSuccess) {
        emit_str("probe.error", cudaGetErrorString(rc));
        emit_int("env.device_count", 0);
        return 1;
    }
    emit_int("env.device_count", device_count);

    for (int d = 0; d < device_count; ++d) {
        cudaDeviceProp p{};
        if (cudaGetDeviceProperties(&p, d) != cudaSuccess) continue;
        char key[160];
        auto K = [&](const char* field) -> const char* {
            std::snprintf(key, sizeof(key), "device.%d.%s", d, field);
            return key;
        };

        emit_str(K("name"), p.name);
        emit_int(K("compute_capability_major"), p.major);
        emit_int(K("compute_capability_minor"), p.minor);
        {
            char cc[16];
            std::snprintf(cc, sizeof(cc), "%d.%d", p.major, p.minor);
            emit_str(K("compute_capability"), cc);
            char sm[16];
            std::snprintf(sm, sizeof(sm), "sm_%d%d", p.major, p.minor);
            emit_str(K("arch"), sm);
        }
        emit_str(K("gcn_arch_name"), nullptr);  // not exposed by the CUDA 13 struct

        // Thread / block / warp limits.
        emit_int(K("multi_processor_count"), p.multiProcessorCount);
        emit_int(K("warp_size"), p.warpSize);
        emit_int(K("max_threads_per_block"), p.maxThreadsPerBlock);
        emit_int(K("max_threads_per_multiprocessor"), p.maxThreadsPerMultiProcessor);
        emit_int(K("max_blocks_per_multiprocessor"), p.maxBlocksPerMultiProcessor);

        // Register file.
        emit_int(K("regs_per_block"), p.regsPerBlock);
        emit_int(K("regs_per_multiprocessor"), p.regsPerMultiprocessor);

        // Shared memory.
        emit_int(K("shared_mem_per_block"), p.sharedMemPerBlock);
        emit_int(K("shared_mem_per_multiprocessor"), p.sharedMemPerMultiprocessor);
        emit_int(K("shared_mem_per_block_optin"), p.sharedMemPerBlockOptin);
        emit_int(K("reserved_shared_mem_per_block"), p.reservedSharedMemPerBlock);
        emit_flag(K("shared_mem_optin_supported"), p.sharedMemPerBlockOptin > p.sharedMemPerBlock);

        // Caches.
        emit_int(K("l2_cache_size"), p.l2CacheSize);
        emit_int(K("persisting_l2_cache_max_size"), p.persistingL2CacheMaxSize);
        emit_int(K("access_policy_max_window_size"), p.accessPolicyMaxWindowSize);
        emit_flag(K("l1_cache_supported"), p.globalL1CacheSupported);

        // Memory system.
        emit_int(K("total_global_mem"), static_cast<long long>(p.totalGlobalMem));
        emit_int(K("total_const_mem"), static_cast<long long>(p.totalConstMem));
        // Clock rates moved out of cudaDeviceProp in CUDA 13; query attributes
        // so an unsupported query is simply absent rather than a fabricated 0.
        {
            int clk = 0;
            if (cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, d) == cudaSuccess && clk > 0)
                emit_int(K("clock_rate_khz"), clk);
            int mclk = 0;
            if (cudaDeviceGetAttribute(&mclk, cudaDevAttrMemoryClockRate, d) == cudaSuccess && mclk > 0)
                emit_int(K("memory_clock_rate_khz"), mclk);
        }
        emit_int(K("memory_bus_width"), p.memoryBusWidth);
        emit_flag(K("memory_pools_supported"), p.memoryPoolsSupported);
        emit_flag(K("unified_addressing"), p.unifiedAddressing);
        emit_flag(K("managed_memory"), p.managedMemory);
        emit_flag(K("integrated_uma"), p.integrated);
        emit_flag(K("can_map_host_memory"), p.canMapHostMemory);
        emit_flag(K("pageable_memory_access"), p.pageableMemoryAccess);
        emit_flag(K("concurrent_managed_access"), p.concurrentManagedAccess);
        emit_flag(K("direct_managed_mem_access_from_host"), p.directManagedMemAccessFromHost);

        // Execution capabilities.
        emit_flag(K("concurrent_kernels"), p.concurrentKernels);
        emit_flag(K("cooperative_launch"), p.cooperativeLaunch);
        emit_flag(K("compute_preemption_supported"), p.computePreemptionSupported);
        emit_int(K("async_engine_count"), p.asyncEngineCount);
        emit_flag(K("stream_priorities_supported"), p.streamPrioritiesSupported);
        emit_flag(K("host_native_atomic_supported"), p.hostNativeAtomicSupported);

        // Feature gates derived from compute capability, not guessed.
        emit_flag(K("supports_tensor_cores"), p.major >= 7);
        emit_flag(K("supports_fp16"), p.major >= 6);
        emit_flag(K("supports_bf16"), p.major >= 8);
        emit_flag(K("supports_tf32"), p.major >= 8);
        emit_flag(K("supports_int8_mma"), p.major >= 8);
        emit_flag(K("supports_cp_async"), p.major >= 8);
        emit_flag(K("supports_tma"), p.major >= 9);
        emit_flag(K("supports_cluster"), p.major >= 9);
        emit_flag(K("supports_fp8"), p.major >= 9 || (p.major == 8 && p.minor >= 9));
        emit_flag(K("supports_fp4"), p.major >= 10);
        emit_flag(K("supports_mxfp8"), p.major >= 10);
        emit_flag(K("supports_mxfp6"), p.major >= 10);
        emit_flag(K("supports_mxfp4"), p.major >= 10);
        emit_flag(K("supports_nvfp4"), p.major >= 10);
        emit_flag(K("supports_block_scale_mma"), p.major >= 10);
        emit_flag(K("supports_distributed_shared_mem"), p.major >= 9);
        emit_flag(K("supports_async_allocator"), p.major >= 11);
        emit_flag(K("supports_fp64"), p.major >= 6);
        emit_flag(K("cluster_launch_supported"), p.clusterLaunch);
    }

#ifdef HAVE_NVML
    // Host-side observability.  Absent values are omitted, never zeroed.
    if (nvmlInit_v2() == NVML_SUCCESS && device_count > 0) {
        nvmlDevice_t dev = nullptr;
        if (nvmlDeviceGetHandleByIndex_v2(0, &dev) == NVML_SUCCESS) {
            char name[NVML_DEVICE_NAME_BUFFER_SIZE] = {0};
            if (nvmlDeviceGetName(dev, name, sizeof(name)) == NVML_SUCCESS)
                emit_str("nvml.name", name);
            unsigned int pw = 0;
            if (nvmlDeviceGetPowerUsage(dev, &pw) == NVML_SUCCESS) emit_int("nvml.power_draw_mw", pw);
            unsigned int pl = 0;
            if (nvmlDeviceGetEnforcedPowerLimit(dev, &pl) == NVML_SUCCESS) emit_int("nvml.power_limit_mw", pl);
            unsigned int smc = 0;
            if (nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &smc) == NVML_SUCCESS) emit_int("nvml.sm_clock_mhz", smc);
            unsigned int mmc = 0;
            if (nvmlDeviceGetClockInfo(dev, NVML_CLOCK_MEM, &mmc) == NVML_SUCCESS) emit_int("nvml.mem_clock_mhz", mmc);
            unsigned int t = 0;
            if (nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &t) == NVML_SUCCESS) emit_int("nvml.temperature_c", t);
            nvmlMemory_t m{};
            if (nvmlDeviceGetMemoryInfo(dev, &m) == NVML_SUCCESS) {
                emit_int("nvml.memory_used_bytes", static_cast<long long>(m.used));
                emit_int("nvml.memory_total_bytes", static_cast<long long>(m.total));
            }
            nvmlUtilization_t u{};
            if (nvmlDeviceGetUtilizationRates(dev, &u) == NVML_SUCCESS) {
                emit_int("nvml.gpu_utilization_pct", u.gpu);
                emit_int("nvml.memory_utilization_pct", u.memory);
            }
        }
        nvmlShutdown();
    }
#endif
    return 0;
}
