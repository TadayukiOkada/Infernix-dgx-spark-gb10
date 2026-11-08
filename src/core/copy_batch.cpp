#include "core/copy_batch.h"

#include "core/device.h"

#include <iterator>

namespace infernix {

bool CopyBatch::pending(const void* destination, std::size_t bytes) const {
    if (bytes == 0) { return false; }
    const auto begin = reinterpret_cast<std::uintptr_t>(destination), end = begin + bytes;
    for (const auto& r : ranges_) {
        if (begin < r.end && end > r.begin) { return true; }
    }
    return false;
}

void CopyBatch::add(void* destination, const void* source, std::size_t bytes) {
    if (bytes == 0) { return; }
    const auto begin = reinterpret_cast<std::uintptr_t>(destination), end = begin + bytes;
    if (pending(destination, bytes)) { flush(); }
    destinations_.push_back(destination);
    sources_.push_back(source);
    sizes_.push_back(bytes);
    ranges_.push_back({begin, end});
}

void CopyBatch::flush() {
    if (sizes_.empty()) { return; }
    cudaMemcpyAttributes attributes{};
    attributes.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    std::size_t first         = 0;
    CUDA_CHECK(cudaMemcpyBatchAsync(destinations_.data(), sources_.data(), sizes_.data(), sizes_.size(), &attributes,
                                    &first, 1, stream_));
    destinations_.clear();
    sources_.clear();
    sizes_.clear();
    ranges_.clear();
}

} // namespace infernix
