// One-shot device query: does GB10 (sm_121) expose a usable persistent-L2 carveout?
// No model weights, no kernels -- pure cudaGetDeviceProperties + limit get/set.
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int dev = 0;
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) { std::printf("query failed\n"); return 1; }

    std::printf("GPU                         : %s\n", p.name);
    std::printf("compute capability          : %d.%d\n", p.major, p.minor);
    std::printf("SMs                         : %d\n", p.multiProcessorCount);
    std::printf("L2 cache size               : %.1f MiB\n", p.l2CacheSize / 1048576.0);
    std::printf("persistingL2CacheMaxSize    : %.1f MiB\n", p.persistingL2CacheMaxSize / 1048576.0);
    std::printf("accessPolicyMaxWindowSize   : %.1f MiB\n", p.accessPolicyMaxWindowSize / 1048576.0);

    // Try to actually carve out a persistent region and read the limit back.
    std::size_t current = 0;
    if (cudaDeviceGetLimit(&current, cudaLimitPersistingL2CacheSize) == cudaSuccess) {
        std::printf("current persistingL2CacheSize : %.1f MiB\n", current / 1048576.0);
    } else {
        std::printf("current persistingL2CacheSize : (get failed)\n");
    }

    const std::size_t want = static_cast<std::size_t>(p.persistingL2CacheMaxSize);
    if (want == 0) {
        std::printf("VERDICT: persistent L2 NOT available on this device -> approach is dead.\n");
        return 0;
    }
    if (cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, want) == cudaSuccess) {
        std::size_t after = 0;
        cudaDeviceGetLimit(&after, cudaLimitPersistingL2CacheSize);
        std::printf("set persisting carveout       : %.1f MiB (applied %.1f MiB)\n",
                    want / 1048576.0, after / 1048576.0);
        std::printf("VERDICT: persistent L2 available -> worth prototyping an access-policy window.\n");
    } else {
        std::printf("VERDICT: cudaDeviceSetLimit(persisting) rejected -> not usable here.\n");
    }
    cudaDeviceReset();
    return 0;
}
