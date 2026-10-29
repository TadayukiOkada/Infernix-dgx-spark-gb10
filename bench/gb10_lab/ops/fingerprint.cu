// fingerprint.cu — Layer 1: system fingerprint.
//
// Records everything needed to reproduce later measurements: device identity,
// compute capability, SM count, cache sizes, shared-memory limits, register
// limits, memory capacity and free space, plus runtime/driver versions. These
// values come from cudaGetDeviceProperties / cudaDeviceGetAttribute so they
// reflect the installed GB10 rather than any assumed specification.
#include "gb10_lab/common.h"

namespace {

struct Attr { cudaDeviceAttr id; const char* name; };

constexpr Attr kAttrs[] = {
    {cudaDevAttrMaxThreadsPerBlock,           "max_threads_per_block"},
    {cudaDevAttrMaxBlocksPerMultiprocessor,   "max_blocks_per_multiprocessor"},
    {cudaDevAttrMaxSharedMemoryPerBlock,      "max_shared_mem_per_block"},
    {cudaDevAttrMaxSharedMemoryPerBlockOptin, "max_shared_mem_per_block_optin"},
    // Per-SM shared memory, registers/block and registers/SM are carried on
    // cudaDeviceProp (p.sharedMemPerMultiprocessor etc.) rather than as
    // cudaDevAttr enums in this toolkit, so they are read directly below.
    {cudaDevAttrL2CacheSize,                  "l2_cache_size"},
    {cudaDevAttrWarpSize,                     "warp_size"},
    {cudaDevAttrMaxThreadsPerMultiProcessor,  "max_threads_per_multiprocessor"},
    {cudaDevAttrConcurrentKernels,            "concurrent_kernels"},
    {cudaDevAttrManagedMemory,                "managed_memory"},
    {cudaDevAttrUnifiedAddressing,            "unified_addressing"},
    {cudaDevAttrCooperativeLaunch,            "cooperative_launch"},
    {cudaDevAttrMemoryPoolsSupported,         "memory_pools_supported"},
    {cudaDevAttrGlobalL1CacheSupported,       "global_l1_cache_supported"},
};

} // namespace

void run_fingerprint(const LabArgs& args) {
    cudaDeviceProp p{};
    LAB_CHECK(cudaGetDeviceProperties(&p, 0));

    const DeviceInfo& d = DeviceInfo::get();

    // Self-describing attribute object: name -> value, with unsupported
    // attributes recorded as null rather than silently dropped.
    std::string attr_json = ",\"attrs\":{";
    bool first_attr = true;
    for (const auto& a : kAttrs) {
        int v = -1;
        const bool ok = (cudaDeviceGetAttribute(&v, a.id, 0) == cudaSuccess);
        attr_json += (first_attr ? "" : ",") + std::string("\"") + a.name + "\":";
        attr_json += ok ? fmt_num((double)v) : "null";
        first_attr = false;
    }
    // Fields that live on cudaDeviceProp rather than as attribute enums.
    auto add_prop = [&](const char* name, long long val){
        attr_json += std::string(",\"") + name + "\":" + fmt_num((double)val);
    };
    add_prop("shared_mem_per_multiprocessor", (long long)p.sharedMemPerMultiprocessor);
    add_prop("registers_per_block",           (long long)p.regsPerBlock);
    add_prop("registers_per_multiprocessor",  (long long)p.regsPerMultiprocessor);
    add_prop("max_warps_per_multiprocessor",  (long long)(p.maxThreadsPerMultiProcessor / p.warpSize));
    attr_json += "}";

    int rv = 0, dv = 0;
    cudaRuntimeGetVersion(&rv);
    cudaDriverGetVersion(&dv);

    JsonLine l;
    l.kv("op", "fingerprint")
     .kv("gpu_name", p.name)
     .append_fragment(",\"cc_major\":" + fmt_num((double)p.major))
     .append_fragment(",\"cc_minor\":" + fmt_num((double)p.minor))
     .append_fragment(",\"sm_count\":" + fmt_num((double)p.multiProcessorCount))
     .append_fragment(",\"total_global_bytes\":" + fmt_num((double)p.totalGlobalMem))
     .append_fragment(",\"free_bytes\":" + fmt_num((double)d.free_bytes))
     .append_fragment(",\"cuda_runtime_api\":" + fmt_num((double)rv))
     .append_fragment(",\"cuda_driver_api\":" + fmt_num((double)dv));

    l.append_fragment(attr_json);

    emit_record(args, l);

    // Human-readable summary to stdout.
    std::printf("\n=== GB10 Fingerprint ===\n");
    std::printf("GPU                    : %s\n", p.name);
    std::printf("Compute capability     : sm_%d%d\n", p.major, p.minor);
    std::printf("SMs                    : %d\n", p.multiProcessorCount);
    std::printf("Global memory          : %.1f GiB total, %.1f GiB free\n",
                p.totalGlobalMem / 1073741824.0, d.free_bytes / 1073741824.0);
    std::printf("L2 cache               : %.1f MiB\n", p.l2CacheSize / 1048576.0);
    std::printf("Shared mem per SM      : %.1f KiB (opt-in max/block %.1f KiB)\n",
                p.sharedMemPerMultiprocessor / 1024.0,
                p.sharedMemPerBlockOptin / 1024.0);
    std::printf("Registers per SM       : %d\n", p.regsPerMultiprocessor);
    std::printf("Max threads/SM         : %d (%d warps of %d)\n",
                p.maxThreadsPerMultiProcessor,
                p.maxThreadsPerMultiProcessor / p.warpSize, p.warpSize);
    std::printf("CUDA runtime API       : %d   driver API: %d\n", rv, dv);
    std::printf("=========================\n");
}
