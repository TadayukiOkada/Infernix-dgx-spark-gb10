#include "core/arena.h"
#include "core/device.h"

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <thread>
#include <vector>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace {

int fail(const char* message) {
    std::cerr << message << '\n';
    return 1;
}

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

int expect_throws_device(int device_id) {
    try {
        infernix::DeviceContext invalid(device_id);
    } catch (const std::runtime_error&) { return 0; }
    std::cerr << "DeviceContext(" << device_id << ") did not throw\n";
    return 1;
}

int check_context(const infernix::DeviceContext& ctx, const char* label) {
    int failures = 0;
    if (ctx.stream == nullptr) {
        std::cerr << label << " compute stream is null\n";
        ++failures;
    }
    if (ctx.transfer_stream == nullptr) {
        std::cerr << label << " load stream is null\n";
        ++failures;
    }
    if (ctx.compute_capability() <= 0) {
        std::cerr << label << " compute capability is not positive\n";
        ++failures;
    }
    if (ctx.total_vram() == 0) {
        std::cerr << label << " total_vram is zero\n";
        ++failures;
    }
    return failures;
}

// upload_pinned_when: the copy waits for the published word, reads the data written after the
// launch (not the poison before it), counts its waits, and a graph replay with a new expected
// word rejects the word an earlier round left in place.
int check_pinned_gate(const infernix::DeviceContext& ctx) {
    int failures = 0;
    constexpr std::size_t kMaxBytes = 327680;
    std::byte* src             = nullptr;
    std::uint32_t* ready       = nullptr;
    std::uint32_t* expected_h  = nullptr;
    void* dst                  = nullptr;
    std::uint32_t* expected    = nullptr;
    std::uint64_t* stats       = nullptr;
    CUDA_CHECK(cudaMallocHost(&src, kMaxBytes));
    CUDA_CHECK(cudaMallocHost(&ready, 64));
    CUDA_CHECK(cudaMallocHost(&expected_h, 64));
    CUDA_CHECK(cudaMalloc(&dst, kMaxBytes));
    CUDA_CHECK(cudaMalloc(&expected, sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMalloc(&stats, 2 * sizeof(std::uint64_t)));
    std::uint32_t word = 0;
    infernix::publish_pinned_word(ready, 0);
    std::vector<std::byte> host(kMaxBytes);
    const auto fill = [&](std::size_t bytes, std::uint32_t round) {
        for (std::size_t i = 0; i < bytes; ++i) { host[i] = static_cast<std::byte>((i * 131 + round * 7 + 3) & 0xff); }
    };
    // Poison, launch, wait until the gate waits, write the data, publish; returns whether the copy
    // holds the data. The host waits for the stream to pass a marker just before the gate, then
    // 200 ms: a fixed sleep after the launch alone let a busy GPU (other processes' kernels) start
    // the gate after the publish, so it recorded no wait or a short one; under a parallel CTest run
    // the gate was seen starting about 19 ms after the marker.
    std::atomic<bool> reached{false};
    const auto produce = [&](std::size_t bytes, std::uint32_t round, const std::function<void()>& launch) {
        std::memset(src, 0xa5, bytes);
        *expected_h = ++word;
        CUDA_CHECK(cudaMemcpyAsync(expected, expected_h, sizeof(std::uint32_t), cudaMemcpyHostToDevice, ctx.stream));
        reached = false;
        CUDA_CHECK(cudaLaunchHostFunc(
            ctx.stream, [](void* flag) { static_cast<std::atomic<bool>*>(flag)->store(true); }, &reached));
        launch();
        ctx.flush();
        const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (!reached.load() && std::chrono::steady_clock::now() < until) { std::this_thread::yield(); }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        fill(bytes, round);
        std::memcpy(src, host.data(), bytes);
        infernix::publish_pinned_word(ready, word);
        ctx.synchronize();
        std::vector<std::byte> out(bytes);
        CUDA_CHECK(cudaMemcpy(out.data(), dst, bytes, cudaMemcpyDeviceToHost));
        return std::memcmp(out.data(), host.data(), bytes) == 0;
    };
    const auto read_stats = [&](std::uint64_t (&out)[2]) {
        CUDA_CHECK(cudaMemcpy(out, stats, sizeof(out), cudaMemcpyDeviceToHost));
    };
    std::uint32_t round = 0;
    for (const std::size_t bytes : {std::size_t{2560}, std::size_t{12800}, kMaxBytes}) {
        CUDA_CHECK(cudaMemset(stats, 0, 2 * sizeof(std::uint64_t)));
        if (!produce(bytes, ++round, [&] {
                infernix::upload_pinned_when(dst, src, bytes, ready, expected, stats, nullptr, ctx.stream);
            })) {
            ++failures;
            std::cerr << "gated copy of " << bytes << " bytes does not hold the published data\n";
        }
        std::uint64_t waited[2] = {};
        read_stats(waited);
        if (waited[1] != 1 || waited[0] < 1000000) {
            ++failures;
            std::cerr << "gated copy of " << bytes << " bytes recorded " << waited[1] << " waits, " << waited[0]
                      << " ns (expected 1 wait of at least 1 ms)\n";
        }
    }
    // A published word passes at once and records no wait.
    {
        CUDA_CHECK(cudaMemset(stats, 0, 2 * sizeof(std::uint64_t)));
        *expected_h = ++word;
        CUDA_CHECK(cudaMemcpyAsync(expected, expected_h, sizeof(std::uint32_t), cudaMemcpyHostToDevice, ctx.stream));
        fill(2560, ++round);
        std::memcpy(src, host.data(), 2560);
        infernix::publish_pinned_word(ready, word);
        infernix::upload_pinned_when(dst, src, 2560, ready, expected, stats, nullptr, ctx.stream);
        ctx.synchronize();
        std::uint64_t waited[2] = {};
        read_stats(waited);
        if (waited[1] != 0) {
            ++failures;
            std::cerr << "a gate whose word was published before the launch recorded a wait\n";
        }
    }
    // Graph replays: each round stages a new expected word; the old word stays published until
    // the host writes the round's data, so a gate that accepted a stale word would copy poison.
    {
        constexpr std::size_t bytes = 12800;
        cudaGraph_t graph       = nullptr;
        cudaGraphExec_t exec    = nullptr;
        CUDA_CHECK(cudaStreamBeginCapture(ctx.stream, cudaStreamCaptureModeThreadLocal));
        infernix::upload_pinned_when(dst, src, bytes, ready, expected, stats, nullptr, ctx.stream);
        CUDA_CHECK(cudaStreamEndCapture(ctx.stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
        CUDA_CHECK(cudaMemset(stats, 0, 2 * sizeof(std::uint64_t)));
        for (int replay = 0; replay < 3; ++replay) {
            if (!produce(bytes, ++round, [&] { CUDA_CHECK(cudaGraphLaunch(exec, ctx.stream)); })) {
                ++failures;
                std::cerr << "graph replay " << replay << " copied a stale or partial round\n";
            }
        }
        std::uint64_t waited[2] = {};
        read_stats(waited);
        if (waited[1] != 3) {
            ++failures;
            std::cerr << "graph replays recorded " << waited[1] << " waits (expected 3)\n";
        }
        CUDA_CHECK(cudaGraphExecDestroy(exec));
        CUDA_CHECK(cudaGraphDestroy(graph));
    }
    try {
        infernix::upload_pinned_when(dst, src, 2561, ready, expected, nullptr, nullptr, ctx.stream);
        ++failures;
        std::cerr << "a gated copy of a size that is not a 16-byte multiple did not throw\n";
    } catch (const std::invalid_argument&) {}
    CUDA_CHECK(cudaFree(stats));
    CUDA_CHECK(cudaFree(expected));
    CUDA_CHECK(cudaFree(dst));
    CUDA_CHECK(cudaFreeHost(expected_h));
    CUDA_CHECK(cudaFreeHost(ready));
    CUDA_CHECK(cudaFreeHost(src));
    return failures;
}

int check_l2_persisting_policy(const infernix::DeviceContext& ctx) {
    int failures = 0;
    const std::size_t expected_persist =
        ctx.props.persistingL2CacheMaxSize > 0 ? static_cast<std::size_t>(ctx.props.persistingL2CacheMaxSize) : 0;
    const std::size_t expected_window =
        ctx.props.accessPolicyMaxWindowSize > 0 ? static_cast<std::size_t>(ctx.props.accessPolicyMaxWindowSize) : 0;
    if (ctx.persisting_l2_cache_max_bytes() != expected_persist ||
        ctx.access_policy_max_window_bytes() != expected_window) {
        ++failures;
        std::cerr << "DeviceContext L2 capacity accessors mismatch props\n";
    }
    if (expected_persist == 0 || expected_window == 0) { return failures; }

    if (ctx.persisting_l2_cache_bytes() == 0 || ctx.persisting_l2_cache_bytes() > expected_persist) {
        ++failures;
        std::cerr << "persisting_l2_cache_bytes (" << ctx.persisting_l2_cache_bytes()
                  << ") out of range (0, " << expected_persist << "]\n";
    }

    std::size_t actual_limit = 0;
    CUDA_CHECK(cudaDeviceGetLimit(&actual_limit, cudaLimitPersistingL2CacheSize));
    if (actual_limit != ctx.persisting_l2_cache_bytes()) {
        ++failures;
        std::cerr << "cudaLimitPersistingL2CacheSize expected " << ctx.persisting_l2_cache_bytes() << ", got "
                  << actual_limit << '\n';
    }

    const std::size_t buf_bytes = 65536;
    void* buf                   = nullptr;
    CUDA_CHECK(cudaMalloc(&buf, buf_bytes));
    {
        infernix::ScopedL2AccessPolicyWindow outer(ctx.stream, buf, buf_bytes, expected_window, 1.0F,
                                                   cudaAccessPropertyPersisting, cudaAccessPropertyNormal);
        if (!outer.active()) {
            ++failures;
            std::cerr << "outer ScopedL2AccessPolicyWindow failed to activate\n";
        }
        cudaStreamAttrValue attr{};
        CUDA_CHECK(cudaStreamGetAttribute(ctx.stream, cudaStreamAttributeAccessPolicyWindow, &attr));
        if (attr.accessPolicyWindow.base_ptr != buf || attr.accessPolicyWindow.num_bytes != buf_bytes ||
            attr.accessPolicyWindow.hitProp != cudaAccessPropertyPersisting ||
            attr.accessPolicyWindow.missProp != cudaAccessPropertyNormal) {
            ++failures;
            std::cerr << "outer stream L2 access policy window attributes mismatch\n";
        }
        {
            infernix::ScopedL2AccessPolicyWindow inner(ctx.stream, buf, 4096, expected_window, 1.0F,
                                                       cudaAccessPropertyNormal, cudaAccessPropertyStreaming);
            if (!inner.active()) {
                ++failures;
                std::cerr << "inner ScopedL2AccessPolicyWindow failed to activate\n";
            }
            CUDA_CHECK(cudaStreamGetAttribute(ctx.stream, cudaStreamAttributeAccessPolicyWindow, &attr));
            if (attr.accessPolicyWindow.num_bytes != 4096 ||
                attr.accessPolicyWindow.hitProp != cudaAccessPropertyNormal ||
                attr.accessPolicyWindow.missProp != cudaAccessPropertyStreaming) {
                ++failures;
                std::cerr << "inner stream L2 access policy window attributes mismatch\n";
            }
        }
        CUDA_CHECK(cudaStreamGetAttribute(ctx.stream, cudaStreamAttributeAccessPolicyWindow, &attr));
        if (attr.accessPolicyWindow.base_ptr != buf || attr.accessPolicyWindow.num_bytes != buf_bytes ||
            attr.accessPolicyWindow.hitProp != cudaAccessPropertyPersisting ||
            attr.accessPolicyWindow.missProp != cudaAccessPropertyNormal) {
            ++failures;
            std::cerr << "outer stream L2 access policy window was not restored after inner scope\n";
        }
    }
    cudaStreamAttrValue cleared{};
    CUDA_CHECK(cudaStreamGetAttribute(ctx.stream, cudaStreamAttributeAccessPolicyWindow, &cleared));
    if (cleared.accessPolicyWindow.num_bytes != 0) {
        ++failures;
        std::cerr << "stream L2 access policy window was not restored to empty\n";
    }
    infernix::reset_persisting_l2_cache(ctx);

    // Phase-switching helpers toggle hardware persisting carveout idempotently (0 during prefill,
    // persisting_l2_cache_bytes() during decode).
    infernix::disable_persisting_l2_cache(ctx);
    infernix::disable_persisting_l2_cache(ctx);
    CUDA_CHECK(cudaDeviceGetLimit(&actual_limit, cudaLimitPersistingL2CacheSize));
    if (actual_limit != 0 || ctx.active_persisting_l2_bytes != 0) {
        ++failures;
        std::cerr << "disable_persisting_l2_cache failed: limit=" << actual_limit
                  << ", active=" << ctx.active_persisting_l2_bytes << '\n';
    }
    infernix::enable_persisting_l2_cache(ctx);
    infernix::enable_persisting_l2_cache(ctx);
    CUDA_CHECK(cudaDeviceGetLimit(&actual_limit, cudaLimitPersistingL2CacheSize));
    if (actual_limit != ctx.persisting_l2_cache_bytes() ||
        ctx.active_persisting_l2_bytes != ctx.persisting_l2_cache_bytes()) {
        ++failures;
        std::cerr << "enable_persisting_l2_cache failed: expected " << ctx.persisting_l2_cache_bytes()
                  << ", limit=" << actual_limit << ", active=" << ctx.active_persisting_l2_bytes << '\n';
    }

    CUDA_CHECK(cudaFree(buf));
    return failures;
}

int check_mapped_pinned_zerocopy(const infernix::DeviceContext& ctx) {
    int failures = 0;
    const bool expected_uma =
        ctx.props.integrated != 0 && ctx.props.canMapHostMemory != 0 && ctx.props.unifiedAddressing != 0;
    if (ctx.integrated_uma() != expected_uma) {
        ++failures;
        std::cerr << "integrated_uma() did not match device properties\n";
    }
    if (!expected_uma) { return failures; }

    constexpr std::size_t kBytes = 4096;
    infernix::PinnedHostBuffer in_buf(kBytes);
    infernix::PinnedHostBuffer out_buf(kBytes);
    if (in_buf.device_data() == nullptr || out_buf.device_data() == nullptr) {
        ++failures;
        std::cerr << "PinnedHostBuffer::device_data() returned null on integrated UMA\n";
        return failures;
    }

    // Verify move construction / assignment preserve device_data() and clear moved-from state.
    {
        void* const orig_host = in_buf.data();
        void* const orig_dev  = in_buf.device_data();
        infernix::PinnedHostBuffer moved_ctor(std::move(in_buf));
        if (in_buf.data() != nullptr || in_buf.device_data() != nullptr ||
            moved_ctor.data() != orig_host || moved_ctor.device_data() != orig_dev) {
            ++failures;
            std::cerr << "PinnedHostBuffer move constructor did not preserve/clear device_data()\n";
        }
        in_buf = std::move(moved_ctor);
        if (moved_ctor.data() != nullptr || moved_ctor.device_data() != nullptr ||
            in_buf.data() != orig_host || in_buf.device_data() != orig_dev) {
            ++failures;
            std::cerr << "PinnedHostBuffer move assignment did not preserve/clear device_data()\n";
        }
    }

    // Repeated CUDA Graph reads/writes directly to mapped pinned host memory with release barrier.
    cudaGraph_t graph    = nullptr;
    cudaGraphExec_t exec = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(ctx.stream, cudaStreamCaptureModeThreadLocal));
    infernix::upload_pinned(out_buf.device_data(), in_buf.device_data(), kBytes, ctx.stream);
    CUDA_CHECK(cudaStreamEndCapture(ctx.stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));

    auto* const in_bytes  = static_cast<std::byte*>(in_buf.data());
    auto* const out_bytes = static_cast<std::byte*>(out_buf.data());
    for (int iter = 0; iter < 256; ++iter) {
        for (std::size_t i = 0; i < kBytes; ++i) {
            in_bytes[i] = static_cast<std::byte>((i * 73U + static_cast<std::size_t>(iter) * 19U + 5U) & 0xffU);
        }
        std::memset(out_bytes, 0, kBytes);
        infernix::publish_pinned_memory();
        CUDA_CHECK(cudaGraphLaunch(exec, ctx.stream));
        ctx.synchronize();
        if (std::memcmp(out_bytes, in_bytes, kBytes) != 0) {
            ++failures;
            std::cerr << "mapped zero-copy CUDA Graph round " << iter << " observed stale or torn data\n";
            break;
        }
    }
    CUDA_CHECK(cudaGraphExecDestroy(exec));
    CUDA_CHECK(cudaGraphDestroy(graph));

    // In-place upload_pinned_when (device_dst == pinned_src) degrades cleanly to gate wait.
    auto* const ready      = static_cast<std::uint32_t*>(out_buf.data());
    std::uint32_t* exp_dev = nullptr;
    CUDA_CHECK(cudaMalloc(&exp_dev, sizeof(std::uint32_t)));
    constexpr std::uint32_t kGateWord = 42U;
    CUDA_CHECK(cudaMemcpyAsync(exp_dev, &kGateWord, sizeof(kGateWord), cudaMemcpyHostToDevice, ctx.stream));
    infernix::publish_pinned_word(ready, kGateWord);
    infernix::upload_pinned_when(in_buf.device_data(), in_buf.device_data(), kBytes, ready, exp_dev, nullptr, nullptr,
                                 ctx.stream);
    ctx.synchronize();
    CUDA_CHECK(cudaFree(exp_dev));
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--invalid-sync") {
        try {
            infernix::DeviceContext ctx(0);
        } catch (const std::invalid_argument& error) {
            return std::string_view(error.what()).find("INFERNIX_CUDA_SYNC") != std::string_view::npos
                       ? 0
                       : fail("invalid sync setting has no configuration diagnostic");
        }
        return fail("invalid sync setting did not fail before CUDA initialization");
    }
    const unsigned int expected_flags =
        argc == 2 ? static_cast<unsigned int>(std::stoul(argv[1])) : cudaDeviceScheduleSpin;
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }
    if (count == 0) {
        std::cout << "SKIP: no CUDA devices\n";
        return 77;
    }

    int failures = 0;

    infernix::DeviceContext ctx(0);
    unsigned int actual_flags = 0;
    CUDA_CHECK(cudaGetDeviceFlags(&actual_flags));
    if ((actual_flags & cudaDeviceScheduleMask) != expected_flags) {
        return fail("CUDA did not apply the requested synchronization schedule");
    }
    if (ctx.device != 0) {
        ++failures;
        std::cerr << "ctx.device expected 0, got " << ctx.device << '\n';
    }
    failures += check_context(ctx, "ctx");
    failures += check_pinned_gate(ctx);
    failures += check_l2_persisting_policy(ctx);
    failures += check_mapped_pinned_zerocopy(ctx);
    int* device_value = nullptr;
    int* host_value   = nullptr;
    CUDA_CHECK(cudaMalloc(&device_value, sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&host_value, sizeof(int)));
    *host_value = 0;
    CUDA_CHECK(cudaMemsetAsync(device_value, 0x5a, sizeof(int), ctx.stream));
    CUDA_CHECK(
        cudaMemcpyAsync(host_value, device_value, sizeof(int), cudaMemcpyDeviceToHost, ctx.stream));
    ctx.synchronize();
    const bool transfer_complete = *host_value == 0x5a5a5a5a;
    CUDA_CHECK(cudaFreeHost(host_value));
    CUDA_CHECK(cudaFree(device_value));
    if (!transfer_complete) {
        return fail("stream synchronization returned before transfer completed");
    }

    const cudaStream_t original_stream = ctx.stream;
    infernix::DeviceContext moved(std::move(ctx));
    if (ctx.stream != nullptr || ctx.transfer_stream != nullptr) {
        ++failures;
        std::cerr << "move construction did not null source streams\n";
    }
    if (moved.stream != original_stream) {
        ++failures;
        std::cerr << "move construction did not transfer compute stream\n";
    }
    failures += check_context(moved, "moved");

    failures += expect_throws_device(count);

    infernix::CudaEventTimer timer(moved);
    timer.start();
    moved.synchronize();
    const float elapsed_ms = timer.stop_ms();
    if (elapsed_ms < 0.0f) {
        ++failures;
        std::cerr << "timer elapsed time was negative\n";
    }

    // try_elapsed_ms never waits: empty while the stop is queued behind host work, a value after.
    infernix::CudaEventTimer pending(moved);
    pending.start();
    CUDA_CHECK(cudaLaunchHostFunc(
        moved.stream,
        [](void*) { std::this_thread::sleep_for(std::chrono::milliseconds(200)); }, nullptr));
    pending.record_stop();
    if (pending.try_elapsed_ms()) {
        ++failures;
        std::cerr << "try_elapsed_ms returned a time before the stop completed\n";
    }
    moved.synchronize();
    const auto settled = pending.try_elapsed_ms();
    if (!settled || *settled < 100.0f) {
        ++failures;
        std::cerr << "try_elapsed_ms did not report the completed stop\n";
    }

    return failures == 0 ? 0 : fail("device test failed");
}
