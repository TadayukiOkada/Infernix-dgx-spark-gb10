#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <exception>
#include <iostream>

namespace {

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

int expect_value(void* device, std::uint32_t expected, const char* label) {
    std::uint32_t actual  = 0;
    const cudaError_t err = cudaMemcpy(&actual, device, sizeof(actual), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << label << " copy failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }
    if (actual == expected) { return 0; }
    std::cerr << label << " expected 0x" << std::hex << expected << ", got 0x" << actual << std::dec
              << '\n';
    return 1;
}

int check_external_event(infernix::DeviceContext& device) {
    infernix::DeviceArena storage(sizeof(std::uint32_t));
    infernix::PinnedHostBuffer host(sizeof(std::uint32_t));
    infernix::CudaCompletionEvent ready(device);
    infernix::DecodeGraphDefinition definitions[2];
    for (int i = 0; i < 2; ++i) {
        definitions[i].capture(device.stream, [&] {
            CUDA_CHECK(
                cudaMemsetAsync(storage.base(), i + 1, sizeof(std::uint32_t), device.stream));
            CUDA_CHECK(cudaMemcpyAsync(host.data(), storage.base(), sizeof(std::uint32_t),
                                       cudaMemcpyDeviceToHost, device.stream));
            ready.record_external(device.stream);
            // The handoff exposes the prefix, even though later graph nodes overwrite the source.
            CUDA_CHECK(cudaMemsetAsync(storage.base(), 0xff, sizeof(std::uint32_t), device.stream));
        });
    }
    infernix::DecodeGraphExecutable executable;
    executable.instantiate(definitions[0]);
    int failures = 0;
    for (int iteration = 0; iteration < 32; ++iteration) {
        const auto value = static_cast<std::uint32_t>(iteration % 2 + 1) * 0x01010101U;
        executable.update(definitions[iteration % 2]);
        *static_cast<std::uint32_t*>(host.data()) = 0;
        executable.launch(device.stream);
        ready.synchronize();
        if (*static_cast<std::uint32_t*>(host.data()) != value) {
            std::cerr << "external event exposed stale data after replay/update\n";
            ++failures;
        }
        device.synchronize();
        failures += expect_value(storage.base(), 0xffffffffU, "graph suffix");
    }
    return failures;
}

int check_l2_policy_graph_capture(infernix::DeviceContext& device) {
    const std::size_t max_persist = device.persisting_l2_cache_max_bytes();
    const std::size_t max_window  = device.access_policy_max_window_bytes();
    if (max_persist == 0 || max_window == 0) { return 0; }

    int failures                     = 0;
    constexpr std::size_t kBytes     = 4096;
    infernix::DeviceBuffer dst(kBytes);
    infernix::PinnedHostBuffer src(kBytes);
    auto* words = static_cast<std::uint32_t*>(src.data());
    for (std::size_t i = 0; i < kBytes / sizeof(std::uint32_t); ++i) { words[i] = 0x11111111U; }

    // 1. Direct CUDA Graph node inspection: kernel nodes captured inside ScopedL2AccessPolicyWindow
    // inherit cudaKernelNodeAttributeAccessPolicyWindow.
    cudaGraph_t raw_graph = nullptr;
    {
        infernix::ScopedL2AccessPolicyWindow l2_window(device.stream, dst.p, kBytes, max_window, 1.0F,
                                                       cudaAccessPropertyPersisting, cudaAccessPropertyNormal);
        CUDA_CHECK(cudaStreamBeginCapture(device.stream, cudaStreamCaptureModeThreadLocal));
        infernix::upload_pinned(dst.p, src.data(), kBytes, device.stream);
        CUDA_CHECK(cudaStreamEndCapture(device.stream, &raw_graph));
    }
    std::size_t node_count  = 0;
    cudaGraphNode_t node[4] = {};
    CUDA_CHECK(cudaGraphGetNodes(raw_graph, nullptr, &node_count));
    if (node_count != 1) {
        ++failures;
        std::cerr << "expected 1 captured kernel node for L2 policy test, got " << node_count << '\n';
    } else {
        CUDA_CHECK(cudaGraphGetNodes(raw_graph, node, &node_count));
        cudaKernelNodeAttrValue node_attr{};
        CUDA_CHECK(cudaGraphKernelNodeGetAttribute(node[0], cudaKernelNodeAttributeAccessPolicyWindow, &node_attr));
        if (node_attr.accessPolicyWindow.base_ptr != dst.p ||
            node_attr.accessPolicyWindow.num_bytes != kBytes ||
            node_attr.accessPolicyWindow.hitProp != cudaAccessPropertyPersisting ||
            node_attr.accessPolicyWindow.missProp != cudaAccessPropertyNormal) {
            ++failures;
            std::cerr << "captured kernel node did not inherit persisting L2 access policy window\n";
        }
    }
    CUDA_CHECK(cudaGraphDestroy(raw_graph));

    // 2. DecodeGraphDefinition + DecodeGraphExecutable update/launch compatibility.
    infernix::DecodeGraphDefinition def_a;
    infernix::DecodeGraphDefinition def_b;
    {
        infernix::ScopedL2AccessPolicyWindow l2_window(device.stream, dst.p, kBytes, max_window, 1.0F,
                                                       cudaAccessPropertyPersisting, cudaAccessPropertyNormal);
        def_a.capture(device.stream, [&] { infernix::upload_pinned(dst.p, src.data(), kBytes, device.stream); });
        def_b.capture(device.stream, [&] { infernix::upload_pinned(dst.p, src.data(), kBytes, device.stream); });
    }
    infernix::DecodeGraphExecutable exec;
    exec.instantiate(def_a);
    exec.launch(device.stream);
    device.synchronize();
    failures += expect_value(dst.p, 0x11111111U, "L2 policy graph launch");

    for (std::size_t i = 0; i < kBytes / sizeof(std::uint32_t); ++i) { words[i] = 0x22222222U; }
    exec.update(def_b);
    exec.launch(device.stream);
    device.synchronize();
    failures += expect_value(dst.p, 0x22222222U, "L2 policy graph update+launch");

    infernix::reset_persisting_l2_cache(device);
    return failures;
}

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err) || (count_err == cudaSuccess && count == 0)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }

    try {
        infernix::DeviceContext device(0);
        infernix::DeviceArena storage(sizeof(std::uint32_t));
        CUDA_CHECK(cudaMemsetAsync(storage.base(), 0x33, sizeof(std::uint32_t), device.stream));
        device.synchronize();

        infernix::DecodeGraphDefinition first;
        first.capture(device.stream, [&] {
            CUDA_CHECK(cudaMemsetAsync(storage.base(), 0x11, sizeof(std::uint32_t), device.stream));
        });
        infernix::DecodeGraphDefinition second;
        second.capture(device.stream, [&] {
            CUDA_CHECK(cudaMemsetAsync(storage.base(), 0x22, sizeof(std::uint32_t), device.stream));
        });

        int failures = 0;
        infernix::DecodeGraphExecutable executable;
        executable.instantiate(first);
        executable.upload(device.stream);
        device.synchronize();
        failures += expect_value(storage.base(), 0x33333333U, "initial graph upload");

        executable.launch(device.stream);
        device.synchronize();
        failures += expect_value(storage.base(), 0x11111111U, "first graph launch");

        executable.update(second);
        executable.upload(device.stream);
        device.synchronize();
        failures += expect_value(storage.base(), 0x11111111U, "updated graph upload");

        executable.launch(device.stream);
        device.synchronize();
        failures += expect_value(storage.base(), 0x22222222U, "updated graph launch");
        failures += check_external_event(device);
        failures += check_l2_policy_graph_capture(device);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "decode graph test failed: " << error.what() << '\n';
        return 1;
    }
}
