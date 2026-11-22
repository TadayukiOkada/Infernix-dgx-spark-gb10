#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace infernix {

// Physical Host memory the OS can hand out now without paging: free plus reclaimable file cache
// (Windows ullAvailPhys, Linux MemAvailable). Callers that lock large Host blocks check it first,
// since pinned pages cannot be paged out and an oversized lock pushes other processes to swap.
[[nodiscard]] std::uint64_t available_host_memory_bytes();

// One reading of the Host memory a startup ledger plans against.
struct HostMemorySnapshot {
    std::uint64_t total_physical = 0;
    // As available_host_memory_bytes, lowered on Linux to what the process's cgroup still allows.
    std::uint64_t available_physical = 0;
    // Memory the OS can still commit (Windows ullAvailPageFile: the commit limit less what is
    // committed; Linux: MemAvailable plus free swap, as Linux commits lazily).
    std::uint64_t available_commit = 0;
    // This process's private committed bytes (Windows PrivateUsage, Linux VmRSS).
    std::uint64_t process_private = 0;
    // This process's resident bytes (Windows WorkingSetSize, Linux VmRSS).
    std::uint64_t process_resident = 0;
};

[[nodiscard]] HostMemorySnapshot host_memory_snapshot();

// The part of the physical pool an allocation can obtain even though it is not free now: file cache
// and reclaimable slabs the OS can evict. Linux: MemAvailable less MemFree (the kernel's own
// estimate, already net of its low watermarks), lowered to what the process's cgroup still allows;
// HugeTLB pages sit outside both readings and stay excluded because they are not reclaimable. Swap
// is never included -- paged-out bytes are not device-speed memory. 0 on Windows, where
// ullAvailPhys already counts what Windows reclaims.
//
// This is why a unified-memory GPU's driver reports less than the machine holds: on GB10
// cudaMemGetInfo's free memory equals MemFree while tens of GiB of file cache sit in the same
// LPDDR5x, and allocations past it succeed by evicting that cache.
[[nodiscard]] std::uint64_t reclaimable_pool_bytes();

// Keeps at least `bytes` of this process resident (Windows: a hard minimum working set), so a small
// RAM reserve trims other programs before Infernix's own pageable memory. Empty on success and where
// the OS has no such control (Linux); otherwise the OS's reason (non-fatal: the caller logs it).
[[nodiscard]] std::string reserve_process_working_set(std::uint64_t bytes);

// Host memory pinned after it is filled. The block is committed pageable memory: prefault() touches
// pages (the OS zeroes a page at its first touch) and pin() then page-locks the whole block and maps
// it for the device in one call (cudaHostRegister, portable and mapped). cudaMallocHost commits,
// zeroes and locks in one serial call (0.11 s/GiB on an i9-13900K, 7 s for Qwen3.8-Flash-Next's
// experts); here threads zero the pages while a reader fills them, and the registration costs about
// 0.003 s/GiB. The device reaches the block at device_data(), which differs from data() where the
// driver cannot map the host address (WDDM).
class RegisteredHostBuffer {
public:
    explicit RegisteredHostBuffer(std::size_t bytes);
    ~RegisteredHostBuffer();
    RegisteredHostBuffer(const RegisteredHostBuffer&)            = delete;
    RegisteredHostBuffer& operator=(const RegisteredHostBuffer&) = delete;

    [[nodiscard]] std::byte* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    // Writes one byte of every page of [begin, end) that nothing has written yet, so its first
    // touch (the zeroing) happens here; it must not overlap bytes already filled.
    void prefault(std::size_t begin, std::size_t end) const noexcept;
    // Page-locks and maps the block for the device; once.
    void pin();
    // Attempts to register and map the block for integrated-UMA zero-copy without throwing.
    // Returns nullptr when integrated UMA is unavailable or registration fails.
    [[nodiscard]] void* try_pin_if_integrated_uma() noexcept;
    [[nodiscard]] std::byte* device_data() const noexcept { return device_; } // null before pin()

private:
    std::byte* data_   = nullptr;
    std::size_t size_  = 0;
    std::byte* device_ = nullptr;
};

// Host memory kept in RAM for the CPU without page-locking it for the device: on Windows the
// process's hard minimum working set grows by its size and its pages are VirtualLock'ed; on Linux
// they are mlock'ed. The device cannot read it, so it does not count toward the limit Windows puts
// on page-locked memory (the adapter's shared system memory, VramSnapshot::shared_system). Committed
// at construction and page-aligned; locking touches (zeroes) every page. When the OS refuses the
// lock, the memory still works but may be paged out under memory pressure: lock_failure() says why.
class ResidentHostBuffer {
public:
    explicit ResidentHostBuffer(std::size_t bytes);
    ~ResidentHostBuffer();
    ResidentHostBuffer(const ResidentHostBuffer&)            = delete;
    ResidentHostBuffer& operator=(const ResidentHostBuffer&) = delete;

    [[nodiscard]] std::byte* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    // Empty when every page is locked in RAM; otherwise the OS's reason.
    [[nodiscard]] const std::string& lock_failure() const noexcept { return lock_failure_; }
    // Attempts to map the buffer into the current CUDA device's UMA address space (zero-copy).
    // Returns nullptr if unified addressing is unavailable or registration fails.
    void* map_for_device();
    [[nodiscard]] void* device_pointer() const noexcept { return device_; }

private:
    std::byte* data_    = nullptr;
    void* device_       = nullptr;
    std::size_t size_   = 0;
    std::size_t raised_ = 0; // Windows: what the minimum working set grew by
    std::string lock_failure_;
};

} // namespace infernix
