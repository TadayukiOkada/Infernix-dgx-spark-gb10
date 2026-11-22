#include "core/host_memory.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>

#include <sys/mman.h>

#include <fstream>
#include <sstream>
#include <string>

namespace infernix {
namespace {

// A "Key:   N kB" field of a /proc file, in bytes.
std::optional<std::uint64_t> proc_kib_field(const char* path, const std::string& key) {
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line)) {
        if (line.rfind(key, 0) == 0) {
            std::istringstream fields(line.substr(key.size()));
            std::uint64_t kib = 0;
            if (fields >> kib) { return kib * 1024; }
        }
    }
    return std::nullopt;
}

// A cgroup v2 memory file holding one byte count ("max" means unlimited).
std::optional<std::uint64_t> cgroup_bytes(const char* path) {
    std::ifstream file(path);
    std::string value;
    if (!(file >> value) || value == "max") { return std::nullopt; }
    try {
        return std::stoull(value);
    } catch (const std::exception&) { return std::nullopt; }
}

} // namespace

std::uint64_t available_host_memory_bytes() {
    if (const auto available = proc_kib_field("/proc/meminfo", "MemAvailable:")) { return *available; }
    throw std::runtime_error("/proc/meminfo has no MemAvailable");
}

std::uint64_t reclaimable_pool_bytes() {
    const std::uint64_t free_physical = proc_kib_field("/proc/meminfo", "MemFree:").value_or(0);
    std::uint64_t available = available_host_memory_bytes();
    if (const auto limit = cgroup_bytes("/sys/fs/cgroup/memory.max")) {
        const std::uint64_t used = cgroup_bytes("/sys/fs/cgroup/memory.current").value_or(0);
        available = std::min(available, *limit > used ? *limit - used : 0);
    }
    return available > free_physical ? available - free_physical : 0;
}

HostMemorySnapshot host_memory_snapshot() {
    HostMemorySnapshot out;
    out.total_physical     = proc_kib_field("/proc/meminfo", "MemTotal:").value_or(0);
    out.available_physical = available_host_memory_bytes();
    if (const auto limit = cgroup_bytes("/sys/fs/cgroup/memory.max")) {
        const std::uint64_t used = cgroup_bytes("/sys/fs/cgroup/memory.current").value_or(0);
        out.available_physical   = std::min(out.available_physical, *limit > used ? *limit - used : 0);
    }
    out.available_commit = out.available_physical + proc_kib_field("/proc/meminfo", "SwapFree:").value_or(0);
    out.process_private  = proc_kib_field("/proc/self/status", "VmRSS:").value_or(0);
    out.process_resident = out.process_private;
    return out;
}

std::string reserve_process_working_set(std::uint64_t bytes) {
    // Linux has no per-process resident minimum short of locking pages.
    (void)bytes;
    return {};
}

RegisteredHostBuffer::RegisteredHostBuffer(std::size_t bytes) : size_(bytes) {
    if (bytes == 0) { throw std::invalid_argument("RegisteredHostBuffer size must be nonzero"); }
    void* p = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) { throw std::runtime_error("mmap of " + std::to_string(bytes) + " bytes failed"); }
    data_ = static_cast<std::byte*>(p);
}

RegisteredHostBuffer::~RegisteredHostBuffer() {
    if (device_ != nullptr) { (void)cudaHostUnregister(data_); }
    (void)::munmap(data_, size_);
}

void RegisteredHostBuffer::prefault(std::size_t begin, std::size_t end) const noexcept {
    constexpr std::size_t kPage = 4096;
    for (std::size_t at = begin / kPage * kPage; at < std::min(end, size_); at += kPage) {
        *reinterpret_cast<volatile std::byte*>(data_ + std::max(at, begin)) = std::byte{0};
    }
}

void RegisteredHostBuffer::pin() {
    if (device_ != nullptr) { throw std::logic_error("RegisteredHostBuffer is already pinned"); }
    if (const auto status = cudaHostRegister(data_, size_, cudaHostRegisterPortable | cudaHostRegisterMapped);
        status != cudaSuccess) {
        throw std::runtime_error("cudaHostRegister of " + std::to_string(size_) +
                                 " bytes failed: " + cudaGetErrorString(status));
    }
    void* device = nullptr;
    if (const auto status = cudaHostGetDevicePointer(&device, data_, 0); status != cudaSuccess) {
        (void)cudaHostUnregister(data_);
        throw std::runtime_error(std::string("cudaHostGetDevicePointer failed: ") + cudaGetErrorString(status));
    }
    device_ = static_cast<std::byte*>(device);
}

ResidentHostBuffer::ResidentHostBuffer(std::size_t bytes) : size_(bytes) {
    if (bytes == 0) { throw std::invalid_argument("ResidentHostBuffer size must be nonzero"); }
    void* p = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) { throw std::runtime_error("mmap of " + std::to_string(bytes) + " bytes failed"); }
    data_ = static_cast<std::byte*>(p);
    if (::mlock(data_, bytes) != 0) { lock_failure_ = "mlock failed (RLIMIT_MEMLOCK)"; }
}

ResidentHostBuffer::~ResidentHostBuffer() {
    if (device_ != nullptr) { (void)cudaHostUnregister(data_); }
    (void)::munmap(data_, size_);
}

void* ResidentHostBuffer::map_for_device() {
    if (device_ != nullptr) { return device_; }
    int device = 0, unified = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&unified, cudaDevAttrUnifiedAddressing, device) != cudaSuccess || unified == 0) {
        return nullptr;
    }
    if (cudaHostRegister(data_, size_, cudaHostRegisterPortable | cudaHostRegisterMapped) != cudaSuccess) {
        return nullptr;
    }
    void* dev_ptr = nullptr;
    if (cudaHostGetDevicePointer(&dev_ptr, data_, 0) != cudaSuccess) {
        (void)cudaHostUnregister(data_);
        return nullptr;
    }
    device_ = dev_ptr;
    return device_;
}

} // namespace infernix
