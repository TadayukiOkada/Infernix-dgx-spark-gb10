#include "runtime/expert_offload/vram_monitor.h"

#include <cuda_runtime.h>

#include <algorithm>


namespace infernix::runtime::expert_offload {

namespace testing {
namespace {
std::atomic<double> g_grow_delay{30.0};
}
void set_vram_grow_delay(double seconds) { g_grow_delay = seconds; }
double vram_grow_delay() { return g_grow_delay; }
} // namespace testing

VramMonitor::VramMonitor(int device, VramBudgetSource& source, std::function<void(const VramSnapshot&)> on_snapshot,
                         std::chrono::milliseconds period)
    : device_(device), source_(source), on_snapshot_(std::move(on_snapshot)), period_(period) {
    latest_ = source_.query();
    thread_ = std::thread([this] { run(); });
}

VramMonitor::~VramMonitor() {
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    stop_cv_.notify_all();
    if (thread_.joinable()) { thread_.join(); }
}

VramSnapshot VramMonitor::latest() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return latest_;
}

VramSnapshot VramMonitor::refresh() {
    const VramSnapshot snapshot = source_.query();
    const std::lock_guard<std::mutex> lock(mutex_);
    latest_ = snapshot;
    return snapshot;
}

void VramMonitor::run() {
    // The source reads device free memory through the runtime, which needs the device current.
    if (cudaSetDevice(device_) != cudaSuccess) { return; }
    for (;;) {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (stop_cv_.wait_for(lock, period_, [this] { return stopping_; })) { return; }
        }
        VramSnapshot snapshot;
        try {
            snapshot = source_.query();
        } catch (...) {
            continue;
        }
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) { return; }
            latest_ = snapshot;
        }
        try {
            on_snapshot_(snapshot);
        } catch (...) {}
    }
}

} // namespace infernix::runtime::expert_offload
