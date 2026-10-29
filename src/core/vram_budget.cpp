#include "core/vram_budget.h"

#include "core/host_memory.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>

#include <dlfcn.h>

namespace infernix {
namespace {

std::mutex& fake_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::function<VramSnapshot()>& fake_source() {
    static std::function<VramSnapshot()> fake;
    return fake;
}

// NVML through the driver's library, loaded at runtime so its absence never stops startup.
class Nvml {
public:
    explicit Nvml(const cudaDeviceProp& props) {
        library_ = ::dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_LOCAL);
        if (library_ == nullptr) { return; }
        const auto symbol = [&](const char* name) { return ::dlsym(library_, name); };
        init_         = reinterpret_cast<int (*)()>(symbol("nvmlInit_v2"));
        shutdown_     = reinterpret_cast<int (*)()>(symbol("nvmlShutdown"));
        by_pci_       = reinterpret_cast<int (*)(const char*, void**)>(symbol("nvmlDeviceGetHandleByPciBusId_v2"));
        display_      = reinterpret_cast<int (*)(void*, int*)>(symbol("nvmlDeviceGetDisplayActive"));
        memory_       = reinterpret_cast<int (*)(void*, MemoryV2*)>(symbol("nvmlDeviceGetMemoryInfo_v2"));
        if (init_ == nullptr || shutdown_ == nullptr || by_pci_ == nullptr || display_ == nullptr ||
            init_() != 0) {
            release();
            return;
        }
        initialized_ = true;
        char bus[32];
        std::snprintf(bus, sizeof(bus), "%08x:%02x:%02x.0", props.pciDomainID, props.pciBusID, props.pciDeviceID);
        if (by_pci_(bus, &device_) != 0) { device_ = nullptr; }
    }

    ~Nvml() { release(); }

    Nvml(const Nvml&)            = delete;
    Nvml& operator=(const Nvml&) = delete;

    // Physical device memory less the driver's reserve (nvmlDeviceGetMemoryInfo_v2); 0 unknown.
    [[nodiscard]] std::uint64_t usable_bytes() const {
        if (device_ == nullptr || memory_ == nullptr) { return 0; }
        MemoryV2 memory{};
        memory.version = static_cast<unsigned>(sizeof(MemoryV2)) | (2U << 24U);
        if (memory_(device_, &memory) != 0 || memory.reserved > memory.total) { return 0; }
        return memory.total - memory.reserved;
    }

    // 1 active, 0 inactive, -1 unknown.
    [[nodiscard]] int display_active() const {
        if (device_ == nullptr) { return -1; }
        int active = 0;
        return display_(device_, &active) == 0 ? (active != 0 ? 1 : 0) : -1;
    }

private:
    struct MemoryV2 { // nvmlMemory_v2_t
        unsigned version;
        unsigned long long total, reserved, free, used;
    };

    void release() noexcept {
        if (initialized_) { (void)shutdown_(); }
        initialized_ = false;
        device_      = nullptr;
        if (library_ == nullptr) { return; }
        ::dlclose(library_);
        library_ = nullptr;
    }

    void* library_ = nullptr;
    int (*init_)()                         = nullptr;
    int (*shutdown_)()                     = nullptr;
    int (*by_pci_)(const char*, void**)    = nullptr;
    int (*display_)(void*, int*)           = nullptr;
    int (*memory_)(void*, MemoryV2*)       = nullptr;
    void* device_                          = nullptr;
    bool initialized_                      = false;
};

class DeviceSource final : public VramBudgetSource {
public:
    explicit DeviceSource(int device) : device_(device) {
        if (cudaGetDeviceProperties(&props_, device) != cudaSuccess) {
            throw std::runtime_error("VRAM budget: cannot read the device's properties");
        }
        nvml_ = std::make_unique<Nvml>(props_);
    }

    ~DeviceSource() override = default;

    VramSnapshot query() override {
        const std::lock_guard<std::mutex> lock(mutex_);
        VramSnapshot out;
        std::size_t free_bytes = 0, total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
            throw std::runtime_error("VRAM budget: cudaMemGetInfo failed");
        }
        out.device_free  = free_bytes;
        out.device_total = total_bytes;
        const int active = nvml_->display_active();
        if (out.outputs > 0 || active == 1) {
            out.display = DisplayState::Attached;
        } else if (out.outputs == 0 && active != 1) {
            out.display = DisplayState::Headless;
        } else if (out.outputs < 0 && active == 0) {
            out.display = DisplayState::Headless;
        } else {
            out.display = DisplayState::Unknown;
        }
        out.integrated = props_.integrated != 0;
        if (out.integrated) {
            // Unified memory: the driver sees only the pool's free share, while an allocation may
            // also take what the OS holds as reclaimable file cache (GB10 under a model load).
            out.reclaimable = reclaimable_pool_bytes();
        }
        return out;
    }

private:
    int device_ = 0;
    std::uint64_t allowance_ = 0; // past the budget (measured at open); 0: the OS budget as reported
    cudaDeviceProp props_{};
    std::unique_ptr<Nvml> nvml_;
    std::mutex mutex_;
};

class FakeSource final : public VramBudgetSource {
public:
    explicit FakeSource(std::function<VramSnapshot()> fake) : fake_(std::move(fake)) {}
    VramSnapshot query() override { return fake_(); }

private:
    std::function<VramSnapshot()> fake_;
};

} // namespace

std::uint64_t spill_shortfall(std::uint64_t free_before, std::uint64_t free_after, std::uint64_t bytes,
                              std::uint64_t tolerance) noexcept {
    const std::uint64_t dropped = free_before > free_after ? free_before - free_after : 0;
    const std::uint64_t missing = bytes > dropped ? bytes - dropped : 0;
    return missing > tolerance ? missing : 0;
}

std::uint64_t SpillGuard::end(std::uint64_t bytes) {
    // On a unified-memory device there is no separate system memory for the driver to demote into:
    // every allocation draws on one pool, so a shortfall of free memory cannot mean a spill of this
    // process's data (the pool never silently moves it elsewhere; an allocation that does not fit
    // fails outright). The guard stays enabled for discrete GPUs, where WDDM can place bytes in
    // shared system memory.
    if (source_.query().integrated) { return 0; }
    after_ = source_.query();
    if (spill_shortfall(before_.device_free, after_.device_free, bytes) == 0) { return 0; }
    after_ = source_.query();
    return spill_shortfall(before_.device_free, after_.device_free, bytes);
}

std::string SpillGuard::readings(std::uint64_t bytes) const {
    const auto mib = [](std::uint64_t value) { return std::to_string(value >> 20) + " MiB"; };
    std::string out = "free device memory " + mib(before_.device_free) + " before and " + mib(after_.device_free) +
                      " after " + mib(bytes) + " of allocations";
    if (before_.shared_system != 0 && after_.shared_system != 0) {
        out += "; this process's shared system memory use " + mib(before_.nonlocal_usage) + " before and " +
               mib(after_.nonlocal_usage) + " after";
    }
    return out;
}

std::unique_ptr<VramBudgetSource> open_vram_budget_source(int device, bool /*past_budget*/) {
    {
        const std::lock_guard<std::mutex> lock(fake_mutex());
        if (fake_source()) { return std::make_unique<FakeSource>(fake_source()); }
    }
    return std::make_unique<DeviceSource>(device);
}

namespace testing {
void set_vram_budget_source(std::function<VramSnapshot()> fake) {
    const std::lock_guard<std::mutex> lock(fake_mutex());
    fake_source() = std::move(fake);
}
} // namespace testing

} // namespace infernix
