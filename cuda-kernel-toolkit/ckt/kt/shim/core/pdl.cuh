#pragma once
// MIRROR-OF: src/core/pdl.cuh (device half only; the launch helpers are host code)
#include <cuda_runtime.h>

namespace infernix::pdl {
__device__ __forceinline__ void trigger_dependents() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); } // KT-SUBST: NVRTC lacks the runtime wrapper
__device__ __forceinline__ void wait_for_dependencies() { asm volatile("griddepcontrol.wait;" ::: "memory"); } // KT-SUBST: NVRTC lacks the runtime wrapper
__device__ __forceinline__ unsigned sm_count() {
    unsigned count;
    asm volatile("mov.u32 %0, %%nsmid;" : "=r"(count));
    return count;
}
__device__ __forceinline__ void enter() {
    if (2U * gridDim.x * gridDim.y * gridDim.z <= sm_count()) { trigger_dependents(); }
    wait_for_dependencies();
}
__device__ __forceinline__ void enter_streaming() { wait_for_dependencies(); }
} // namespace infernix::pdl
