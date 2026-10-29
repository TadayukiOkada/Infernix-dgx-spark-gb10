#include "ops/common/device_info.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ops::detail {
namespace {

// Reference part every launch constant is tuned for: GB10 (sm_121a, 48 SMs). Used only when the
// live query cannot run (no CUDA context / driver unavailable), keeping behaviour defined instead
// of zero.
constexpr int kReferenceSmCount = 48;

} // namespace

int device_sm_count() {
    // Magic-static initialisation runs once, thread-safely, on first call and is then cached for
    // the process lifetime, so the attribute lookup happens once per process rather than per launch.
    static const int count = [] {
        int device = 0;
        int value  = 0;
        if (cudaGetDevice(&device) == cudaSuccess &&
            cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount, device) == cudaSuccess &&
            value > 0) {
            return value;
        }
        return kReferenceSmCount;
    }();
    return count;
}

} // namespace infernix::ops::detail
