#pragma once

#include "infernix/ops/context_kv_materialize.h"

namespace infernix::ops::detail {

enum class ContextKVMaterializeRoute : std::uint8_t {
    KSplit16,
    KSplit24,
    Mma32,
    Mma80,
    Mma96,
    Fused64,
    Fused64Mid,
    Fused64Wide,
};

// Columns count the envelope's live prefix capacity across requests, not physical padding.
// The narrow bands defer the Key RMSNorm + RoPE to context_kv_key_post_kernel through the FP32
// key scratch: their split-K / half-height schedules win more CTAs than a Rows = kHeadDim CTA
// can hold. From 65 columns up, every MMA route runs Rows = kHeadDim and fuses the key store.
// The three Fused64 routes share one tile shape and differ only in the two measured staging
// groupings: cp.async codes with tail-grouped scales at two column blocks (-1.5..-4.4%),
// head-grouped scales from three through four (+3..5% over tail there), and LDG register
// staging from five up (-8..-9% versus cp.async at >=512 columns).
constexpr ContextKVMaterializeRoute context_kv_materialize_route(std::int32_t columns) {
    using Route = ContextKVMaterializeRoute;
    if (columns <= 16) return Route::KSplit16;
    if (columns <= 24) return Route::KSplit24;
    if (columns <= 32) return Route::KSplit16;
    if (columns <= 64) return Route::Mma32;
    if (columns <= 80) return Route::Mma80;
    if (columns <= 96) return Route::Mma96;
    if (columns <= 128) return Route::Fused64;
    if (columns <= 256) return Route::Fused64Mid;
    return Route::Fused64Wide;
}

constexpr bool context_kv_materialize_uses_scratch(ContextKVMaterializeRoute route) {
    return route == ContextKVMaterializeRoute::KSplit16 ||
           route == ContextKVMaterializeRoute::KSplit24 ||
           route == ContextKVMaterializeRoute::Mma32;
}

void context_kv_materialize_launch(
    const Tensor& context, const Tensor& positions, const Tensor& counts, const Tensor& state_slots,
    std::span<const ContextKVMaterializeLayerView> layers,
    const std::array<double, 64>& inv_frequency, ContextKVMaterializeExecutionEnvelope envelope,
    ContextKVMaterializeRoute route,
    const Tensor& key_scratch, cudaStream_t stream);

} // namespace infernix::ops::detail
