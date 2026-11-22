#include "core/device.h"

#include <algorithm>
#include <atomic>
#include <cstdint>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace infernix {
namespace {

struct SyncSchedule {
    std::string_view name;
    unsigned int flags;
};

constexpr SyncSchedule kSyncSchedules[] = {
    {"spin", cudaDeviceScheduleSpin},
    {"blocking", cudaDeviceScheduleBlockingSync},
    {"yield", cudaDeviceScheduleYield},
    {"auto", cudaDeviceScheduleAuto},
};

unsigned int sync_schedule_from_environment() {
    const char* value = std::getenv("INFERNIX_CUDA_SYNC");
    if (value == nullptr) { return cudaDeviceScheduleSpin; }
    for (const auto& schedule : kSyncSchedules) {
        if (schedule.name == value) { return schedule.flags; }
    }
    throw std::invalid_argument("INFERNIX_CUDA_SYNC must be spin, blocking, yield, or auto");
}

std::string cuda_error_message(const char* prefix, cudaError_t err) {
    return std::string(prefix) + ": " + cudaGetErrorName(err) + ": " + cudaGetErrorString(err);
}

void log_cuda_error(const char* op, cudaError_t err) noexcept {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA cleanup failed during %s: %s: %s\n", op, cudaGetErrorName(err),
                     cudaGetErrorString(err));
    }
}

void destroy_stream(cudaStream_t& stream) noexcept {
    if (stream != nullptr) {
        log_cuda_error("cudaStreamDestroy", cudaStreamDestroy(stream));
        stream = nullptr;
    }
}

void destroy_event(cudaEvent_t& event) noexcept {
    if (event != nullptr) {
        log_cuda_error("cudaEventDestroy", cudaEventDestroy(event));
        event = nullptr;
    }
}

} // namespace

void cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
    if (err == cudaSuccess) { return; }
    std::fprintf(stderr, "%s:%d: CUDA_CHECK(%s) failed: %s: %s\n", file, line, expr,
                 cudaGetErrorName(err), cudaGetErrorString(err));
    std::abort();
}

DeviceContext::DeviceContext(int device_id) : device(device_id) {
    const unsigned int sync_flags = sync_schedule_from_environment();
    int count       = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaGetDeviceCount failed", err));
    }
    if (count <= 0) { throw std::runtime_error("no CUDA devices available"); }
    if (device_id < 0 || device_id >= count) { throw std::runtime_error("invalid CUDA device id"); }

    bind_to_current_thread();

    err = cudaSetDeviceFlags(sync_flags);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaSetDeviceFlags failed", err));
    }

    err = cudaGetDeviceProperties(&props, device_id);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaGetDeviceProperties failed", err));
    }

    // Reserve 3/8 of L2 (9.0 MiB on GB10's 24 MiB L2) for persisting decode workspace:
    // hardware sweeps on sm_121a measured a ~2x throughput cliff at 18 MiB (190.6 GB/s DRAM,
    // 53.38 TF BF16 GEMM vs 242.0 GB/s, 105.00 TF at 9.0 MiB and 107.38 TF at default 4.5 MiB).
    if (props.persistingL2CacheMaxSize > 0) {
        const std::size_t max_persist = static_cast<std::size_t>(props.persistingL2CacheMaxSize);
        const std::size_t l2_bytes =
            props.l2CacheSize > 0 ? static_cast<std::size_t>(props.l2CacheSize) : max_persist;
        const std::size_t target = std::min(max_persist, l2_bytes * 3 / 8);
        if (target > 0) {
            const cudaError_t l2_err = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, target);
            if (l2_err == cudaErrorNotSupported || l2_err == cudaErrorInvalidValue) {
                (void)cudaGetLastError();
            } else if (l2_err != cudaSuccess) {
                throw std::runtime_error(
                    cuda_error_message("cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize) failed", l2_err));
            }
        }
        std::size_t applied          = 0;
        const cudaError_t get_l2_err = cudaDeviceGetLimit(&applied, cudaLimitPersistingL2CacheSize);
        if (get_l2_err == cudaErrorNotSupported || get_l2_err == cudaErrorInvalidValue) {
            (void)cudaGetLastError();
        } else if (get_l2_err != cudaSuccess) {
            throw std::runtime_error(
                cuda_error_message("cudaDeviceGetLimit(cudaLimitPersistingL2CacheSize) failed", get_l2_err));
        } else {
            persisting_l2_bytes        = applied;
            active_persisting_l2_bytes = applied;
        }
    }

    cudaStream_t compute = nullptr;
    cudaStream_t load    = nullptr;
    err                  = cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            cuda_error_message("cudaStreamCreateWithFlags(stream) failed", err));
    }

    err = cudaStreamCreateWithFlags(&load, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        destroy_stream(compute);
        throw std::runtime_error(
            cuda_error_message("cudaStreamCreateWithFlags(transfer_stream) failed", err));
    }

    stream          = compute;
    transfer_stream = load;
}

DeviceContext::~DeviceContext() {
    if (stream != nullptr || transfer_stream != nullptr) { bind_to_current_thread_noexcept(); }
    destroy_stream(transfer_stream);
    destroy_stream(stream);
}

DeviceContext::DeviceContext(DeviceContext&& other) noexcept
    : device(other.device), stream(other.stream), transfer_stream(other.transfer_stream),
      props(other.props), persisting_l2_bytes(other.persisting_l2_bytes),
      active_persisting_l2_bytes(other.active_persisting_l2_bytes) {
    other.stream                     = nullptr;
    other.transfer_stream            = nullptr;
    other.persisting_l2_bytes        = 0;
    other.active_persisting_l2_bytes = 0;
}

DeviceContext& DeviceContext::operator=(DeviceContext&& other) noexcept {
    if (this == &other) { return *this; }

    if (stream != nullptr || transfer_stream != nullptr) { bind_to_current_thread_noexcept(); }
    destroy_stream(transfer_stream);
    destroy_stream(stream);

    device                     = other.device;
    props                      = other.props;
    persisting_l2_bytes        = other.persisting_l2_bytes;
    active_persisting_l2_bytes = other.active_persisting_l2_bytes;
    stream                     = other.stream;
    transfer_stream            = other.transfer_stream;

    other.stream                     = nullptr;
    other.transfer_stream            = nullptr;
    other.persisting_l2_bytes        = 0;
    other.active_persisting_l2_bytes = 0;
    return *this;
}

void DeviceContext::bind_to_current_thread() const {
    const cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaSetDevice failed", err));
    }
}

void DeviceContext::bind_to_current_thread_noexcept() const noexcept {
    log_cuda_error("cudaSetDevice", cudaSetDevice(device));
}

int DeviceContext::compute_capability() const noexcept { return props.major * 10 + props.minor; }

int DeviceContext::multiprocessor_count() const noexcept { return props.multiProcessorCount; }

bool DeviceContext::integrated_uma() const noexcept {
    return props.integrated != 0 && props.canMapHostMemory != 0 && props.unifiedAddressing != 0;
}

DeviceExecutionView DeviceContext::execution_view() const noexcept {
    return {.stream = stream, .multiprocessor_count = multiprocessor_count()};
}

std::size_t DeviceContext::persisting_l2_cache_bytes() const noexcept {
    return persisting_l2_bytes;
}

std::size_t DeviceContext::persisting_l2_cache_max_bytes() const noexcept {
    return props.persistingL2CacheMaxSize > 0 ? static_cast<std::size_t>(props.persistingL2CacheMaxSize) : 0;
}

std::size_t DeviceContext::access_policy_max_window_bytes() const noexcept {
    return props.accessPolicyMaxWindowSize > 0 ? static_cast<std::size_t>(props.accessPolicyMaxWindowSize) : 0;
}

std::size_t DeviceContext::total_vram() const noexcept { return props.totalGlobalMem; }

const char* DeviceContext::sync_mode() const {
    bind_to_current_thread();
    unsigned int flags    = 0;
    const cudaError_t err = cudaGetDeviceFlags(&flags);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaGetDeviceFlags failed", err));
    }
    for (const auto& schedule : kSyncSchedules) {
        if ((flags & cudaDeviceScheduleMask) == schedule.flags) { return schedule.name.data(); }
    }
    throw std::runtime_error("unknown CUDA synchronization schedule");
}

void DeviceContext::synchronize() const { CUDA_CHECK(cudaStreamSynchronize(stream)); }

std::size_t DeviceContext::free_bytes() const {
    std::size_t free  = 0;
    std::size_t total = 0;
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
    return free;
}

void DeviceContext::flush() const {
    const cudaError_t status = cudaStreamQuery(stream);
    if (status == cudaErrorNotReady) {
        // Pending work is the expected answer; do not leave it as the thread's last error.
        (void)cudaGetLastError();
        return;
    }
    CUDA_CHECK(status);
}

ScopedL2AccessPolicyWindow::ScopedL2AccessPolicyWindow(cudaStream_t stream, const void* base_ptr,
                                                       std::size_t num_bytes, std::size_t max_window_bytes,
                                                       float hit_ratio, cudaAccessProperty hit_prop,
                                                       cudaAccessProperty miss_prop) {
    if (stream == nullptr || base_ptr == nullptr || num_bytes == 0 || max_window_bytes == 0 || hit_ratio <= 0.0F) {
        return;
    }
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    const cudaError_t cap_err              = cudaStreamIsCapturing(stream, &capture_status);
    if (cap_err != cudaSuccess) {
        (void)cudaGetLastError();
        return;
    }
    if (capture_status != cudaStreamCaptureStatusNone) { return; }

    const cudaError_t get_err = cudaStreamGetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &previous_);
    if (get_err != cudaSuccess) {
        if (get_err == cudaErrorNotSupported || get_err == cudaErrorInvalidValue) {
            (void)cudaGetLastError();
            return;
        }
        throw std::runtime_error(cuda_error_message("cudaStreamGetAttribute(AccessPolicyWindow) failed", get_err));
    }

    cudaStreamAttrValue next{};
    next.accessPolicyWindow.base_ptr  = const_cast<void*>(base_ptr);
    next.accessPolicyWindow.num_bytes = std::min(num_bytes, max_window_bytes);
    next.accessPolicyWindow.hitRatio  = std::clamp(hit_ratio, 0.0F, 1.0F);
    next.accessPolicyWindow.hitProp   = hit_prop;
    next.accessPolicyWindow.missProp  = miss_prop;
    const cudaError_t set_err         = cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &next);
    if (set_err != cudaSuccess) {
        if (set_err == cudaErrorNotSupported || set_err == cudaErrorInvalidValue) {
            (void)cudaGetLastError();
            return;
        }
        throw std::runtime_error(cuda_error_message("cudaStreamSetAttribute(AccessPolicyWindow) failed", set_err));
    }
    stream_ = stream;
    active_ = true;
}

ScopedL2AccessPolicyWindow::~ScopedL2AccessPolicyWindow() noexcept { restore(); }

ScopedL2AccessPolicyWindow::ScopedL2AccessPolicyWindow(ScopedL2AccessPolicyWindow&& other) noexcept
    : stream_(std::exchange(other.stream_, nullptr)), previous_(other.previous_),
      active_(std::exchange(other.active_, false)) {}

ScopedL2AccessPolicyWindow& ScopedL2AccessPolicyWindow::operator=(ScopedL2AccessPolicyWindow&& other) noexcept {
    if (this == &other) { return *this; }
    restore();
    stream_   = std::exchange(other.stream_, nullptr);
    previous_ = other.previous_;
    active_   = std::exchange(other.active_, false);
    return *this;
}

void ScopedL2AccessPolicyWindow::restore() noexcept {
    if (!active_ || stream_ == nullptr) {
        active_ = false;
        stream_ = nullptr;
        return;
    }
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream_, &capture_status) == cudaSuccess &&
        capture_status == cudaStreamCaptureStatusNone) {
        log_cuda_error("cudaStreamSetAttribute(AccessPolicyWindow restore)",
                       cudaStreamSetAttribute(stream_, cudaStreamAttributeAccessPolicyWindow, &previous_));
    } else {
        (void)cudaGetLastError();
    }
    active_ = false;
    stream_ = nullptr;
}

void reset_persisting_l2_cache(const DeviceContext& ctx) noexcept {
    if (ctx.persisting_l2_cache_bytes() == 0) { return; }
    log_cuda_error("cudaCtxResetPersistingL2Cache", cudaCtxResetPersistingL2Cache());
}

void enable_persisting_l2_cache(const DeviceContext& ctx) noexcept {
    const std::size_t target = ctx.persisting_l2_cache_bytes();
    if (target == 0 || ctx.active_persisting_l2_bytes == target) { return; }
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    if (ctx.stream != nullptr &&
        cudaStreamIsCapturing(ctx.stream, &capture_status) == cudaSuccess &&
        capture_status != cudaStreamCaptureStatusNone) {
        return;
    }
    ctx.bind_to_current_thread_noexcept();
    const cudaError_t err = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, target);
    if (err == cudaSuccess) {
        ctx.active_persisting_l2_bytes = target;
    } else {
        log_cuda_error("cudaDeviceSetLimit(enable persisting L2)", err);
    }
}

void disable_persisting_l2_cache(const DeviceContext& ctx) noexcept {
    if (ctx.persisting_l2_cache_bytes() == 0 || ctx.active_persisting_l2_bytes == 0) { return; }
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    if (ctx.stream != nullptr &&
        cudaStreamIsCapturing(ctx.stream, &capture_status) == cudaSuccess &&
        capture_status != cudaStreamCaptureStatusNone) {
        return;
    }
    ctx.bind_to_current_thread_noexcept();
    log_cuda_error("cudaCtxResetPersistingL2Cache", cudaCtxResetPersistingL2Cache());
    const cudaError_t err = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0);
    if (err == cudaSuccess) {
        ctx.active_persisting_l2_bytes = 0;
    } else {
        log_cuda_error("cudaDeviceSetLimit(disable persisting L2)", err);
    }
}

CudaEventTimer::CudaEventTimer(const DeviceContext& ctx) : CudaEventTimer(ctx, ctx.stream) {}

CudaEventTimer::CudaEventTimer(const DeviceContext& ctx, cudaStream_t stream) : stream_(stream) {
    if (stream == nullptr) { throw std::invalid_argument("CUDA timer stream is null"); }
    ctx.bind_to_current_thread();

    cudaEvent_t start = nullptr;
    cudaEvent_t stop  = nullptr;
    cudaError_t err   = cudaEventCreate(&start);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaEventCreate(start) failed", err));
    }

    err = cudaEventCreate(&stop);
    if (err != cudaSuccess) {
        destroy_event(start);
        throw std::runtime_error(cuda_error_message("cudaEventCreate(stop) failed", err));
    }

    start_ = start;
    stop_  = stop;
}

CudaEventTimer::~CudaEventTimer() {
    destroy_event(stop_);
    destroy_event(start_);
}

CudaEventTimer::CudaEventTimer(CudaEventTimer&& other) noexcept
    : stream_(other.stream_), start_(other.start_), stop_(other.stop_) {
    other.stream_ = nullptr;
    other.start_  = nullptr;
    other.stop_   = nullptr;
}

CudaEventTimer& CudaEventTimer::operator=(CudaEventTimer&& other) noexcept {
    if (this == &other) { return *this; }

    destroy_event(stop_);
    destroy_event(start_);

    stream_ = other.stream_;
    start_  = other.start_;
    stop_   = other.stop_;

    other.stream_ = nullptr;
    other.start_  = nullptr;
    other.stop_   = nullptr;
    return *this;
}

void CudaEventTimer::start() { CUDA_CHECK(cudaEventRecord(start_, stream_)); }

void CudaEventTimer::record_stop() { CUDA_CHECK(cudaEventRecord(stop_, stream_)); }

float CudaEventTimer::elapsed_ms() const {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
}

std::optional<float> CudaEventTimer::try_elapsed_ms() const {
    const cudaError_t status = cudaEventQuery(stop_);
    if (status == cudaErrorNotReady) { return std::nullopt; }
    CUDA_CHECK(status);
    return elapsed_ms();
}

float CudaEventTimer::stop_ms() {
    record_stop();
    CUDA_CHECK(cudaEventSynchronize(stop_));
    return elapsed_ms();
}

CudaCompletionEvent::CudaCompletionEvent(const DeviceContext& ctx) : device_(ctx.device) {
    ctx.bind_to_current_thread();
    const cudaError_t err = cudaEventCreateWithFlags(&event_, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaEventCreateWithFlags failed", err));
    }
}

CudaCompletionEvent::~CudaCompletionEvent() { destroy_event(event_); }

CudaCompletionEvent::CudaCompletionEvent(CudaCompletionEvent&& other) noexcept
    : device_(other.device_), event_(std::exchange(other.event_, nullptr)) {}

CudaCompletionEvent& CudaCompletionEvent::operator=(CudaCompletionEvent&& other) noexcept {
    if (this == &other) { return *this; }
    destroy_event(event_);
    device_ = other.device_;
    event_  = std::exchange(other.event_, nullptr);
    return *this;
}

void CudaCompletionEvent::record(cudaStream_t stream) {
    if (event_ == nullptr || stream == nullptr) {
        throw std::logic_error("CUDA completion event is not recordable");
    }
    CUDA_CHECK(cudaEventRecord(event_, stream));
}

void CudaCompletionEvent::wait(cudaStream_t stream) const {
    if (event_ == nullptr || stream == nullptr) {
        throw std::logic_error("CUDA completion event is not waitable");
    }
    CUDA_CHECK(cudaStreamWaitEvent(stream, event_, 0));
}

void CudaCompletionEvent::record_external(cudaStream_t stream) {
    if (event_ == nullptr || stream == nullptr) {
        throw std::logic_error("CUDA completion event is not recordable");
    }
    CUDA_CHECK(cudaEventRecordWithFlags(event_, stream, cudaEventRecordExternal));
}

bool CudaCompletionEvent::ready() const {
    if (event_ == nullptr) { throw std::logic_error("CUDA completion event is empty"); }
    const cudaError_t status = cudaEventQuery(event_);
    if (status == cudaSuccess) { return true; }
    if (status == cudaErrorNotReady) { return false; }
    CUDA_CHECK(status);
    return false;
}

void CudaCompletionEvent::synchronize() const {
    if (event_ == nullptr) { throw std::logic_error("CUDA completion event is empty"); }
    CUDA_CHECK(cudaEventSynchronize(event_));
}


namespace {

__global__ void upload_pinned_kernel(void* __restrict__ dst, const void* __restrict__ src, std::size_t bytes) {
    const std::size_t thread = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    const bool vectors = (reinterpret_cast<std::uintptr_t>(dst) | reinterpret_cast<std::uintptr_t>(src)) % 16 == 0;
    const std::size_t head = vectors ? bytes / 16 : 0;
    for (std::size_t i = thread; i < head; i += stride) {
        static_cast<uint4*>(dst)[i] = static_cast<const uint4*>(src)[i];
    }
    for (std::size_t i = head * 16 + thread; i < bytes; i += stride) {
        static_cast<unsigned char*>(dst)[i] = static_cast<const unsigned char*>(src)[i];
    }
}

constexpr std::size_t kGateCtaBytes = 32 * 1024;
constexpr int kGateMaxCtas         = 16;
constexpr int kGateThreads         = 256;

// Each CTA's thread 0 polls the host word, so no CTA depends on another being resident. .cv loads:
// the source is rewritten for every round, so no cache line of an earlier round may be used.
__global__ void __launch_bounds__(kGateThreads)
    upload_pinned_when_kernel(uint4* __restrict__ dst, const uint4* __restrict__ src, std::size_t vectors,
                              const std::uint32_t* ready, const std::uint32_t* expected, std::uint64_t* stats,
                              const std::int32_t* row) {
    if (threadIdx.x == 0) {
        const std::uint32_t want = *expected;
        const auto* word         = reinterpret_cast<const volatile std::uint32_t*>(ready);
        if (*word != want) {
            std::uint64_t start, now;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
            do {
                __nanosleep(200);
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
                if (now - start > kPinnedGateTimeoutNs) { asm volatile("trap;"); }
            } while (*word != want);
            if (stats != nullptr && blockIdx.x == 0) {
                std::uint64_t* at = stats + 2 * (row != nullptr ? *row : 0);
                atomicAdd(reinterpret_cast<unsigned long long*>(at), static_cast<unsigned long long>(now - start));
                atomicAdd(reinterpret_cast<unsigned long long*>(at + 1), 1ULL);
            }
        }
        __threadfence_system();
    }
    __syncthreads();
    constexpr int kBatch     = 8;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t base = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; base < vectors;
         base += kBatch * stride) {
        uint4 loaded[kBatch];
#pragma unroll
        for (int b = 0; b < kBatch; ++b) {
            const std::size_t i = base + b * stride;
            if (i < vectors) { loaded[b] = __ldcv(src + i); }
        }
#pragma unroll
        for (int b = 0; b < kBatch; ++b) {
            const std::size_t i = base + b * stride;
            if (i < vectors) { dst[i] = loaded[b]; }
        }
    }
}

__global__ void __launch_bounds__(1)
    wait_pinned_word_kernel(const std::uint32_t* ready, const std::uint32_t* expected, std::uint64_t* stats,
                            const std::int32_t* row) {
    const std::uint32_t want = *expected;
    const auto* word         = reinterpret_cast<const volatile std::uint32_t*>(ready);
    if (*word != want) {
        std::uint64_t start, now;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
        do {
            __nanosleep(200);
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
            if (now - start > kPinnedGateTimeoutNs) { asm volatile("trap;"); }
        } while (*word != want);
        if (stats != nullptr) {
            std::uint64_t* at = stats + 2 * (row != nullptr ? *row : 0);
            atomicAdd(reinterpret_cast<unsigned long long*>(at), static_cast<unsigned long long>(now - start));
            atomicAdd(reinterpret_cast<unsigned long long*>(at + 1), 1ULL);
        }
    }
    __threadfence_system();
}

} // namespace

void upload_pinned_when(void* device_dst, const void* pinned_src, std::size_t bytes, const std::uint32_t* pinned_ready,
                        const std::uint32_t* device_expected, std::uint64_t* wait_stats, const std::int32_t* wait_row,
                        cudaStream_t stream) {
    if (bytes == 0) { return; }
    if (device_dst == pinned_src) {
        wait_pinned_word(pinned_ready, device_expected, wait_stats, wait_row, stream);
        return;
    }
    if (bytes % 16 != 0 || (reinterpret_cast<std::uintptr_t>(device_dst) | reinterpret_cast<std::uintptr_t>(pinned_src)) % 16 != 0) {
        throw std::invalid_argument("upload_pinned_when: bytes and buffers must be 16-byte multiples");
    }
    const auto ctas = static_cast<unsigned int>(
        std::min<std::size_t>((bytes + kGateCtaBytes - 1) / kGateCtaBytes, kGateMaxCtas));
    upload_pinned_when_kernel<<<ctas, kGateThreads, 0, stream>>>(static_cast<uint4*>(device_dst),
                                                                 static_cast<const uint4*>(pinned_src), bytes / 16,
                                                                 pinned_ready, device_expected, wait_stats, wait_row);
    CUDA_CHECK(cudaGetLastError());
}

void wait_pinned_word(const std::uint32_t* pinned_ready, const std::uint32_t* device_expected,
                      std::uint64_t* wait_stats, const std::int32_t* wait_row, cudaStream_t stream) {
    if (pinned_ready == nullptr || device_expected == nullptr) {
        if (pinned_ready != device_expected) {
            throw std::invalid_argument("wait_pinned_word: ready and expected pointers must both be set or null");
        }
        return;
    }
    wait_pinned_word_kernel<<<1, 1, 0, stream>>>(pinned_ready, device_expected, wait_stats, wait_row);
    CUDA_CHECK(cudaGetLastError());
}

void publish_pinned_word(std::uint32_t* pinned_ready, std::uint32_t value) noexcept {
    // The release fence orders the prior writes before the ready flag on every target, and on
    // AArch64 it lowers to the DMB ISHST that makes them visible to the device.
    std::atomic_thread_fence(std::memory_order_release);
    *static_cast<volatile std::uint32_t*>(pinned_ready) = value;
}

void upload_pinned(void* device_dst, const void* pinned_src, std::size_t bytes, cudaStream_t stream) {
    if (bytes == 0) { return; }
    static const bool unified = [] {
        int device = 0, value = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&value, cudaDevAttrUnifiedAddressing, device));
        return value != 0;
    }();
    if (!unified) { throw std::runtime_error("upload_pinned needs unified addressing"); }
    constexpr int kThreads = 256;
    // One 16-byte load per thread: every load of a call is in flight at once.
    const std::size_t blocks = std::min<std::size_t>((bytes / 16 + kThreads) / kThreads + 1, 64);
    upload_pinned_kernel<<<static_cast<unsigned int>(blocks), kThreads, 0, stream>>>(device_dst, pinned_src, bytes);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix
