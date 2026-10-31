#pragma once

#include "ops/linear/common/output.cuh"

#include <cstdint>
#include <type_traits>

namespace infernix::ops::detail {

// Thread-local operations over a fully reduced accumulator. The contraction
// owns predicates and synchronization; epilogues only receive valid coordinates.
struct LinearIdentityEpilogue {
    __device__ __forceinline__ float apply(int, int, float value) const { return value; }
};

struct LinearResidualAddEpilogue {
    LinearBf16InputView residual;

    __device__ __forceinline__ float apply(int row, int token, float value) const {
        return value + residual.load(row, token);
    }
};

// The per-row scratch line an output claims from a contraction that supports prefetching: `words`
// 32-bit words per row, filled in `units` independent units so a contraction can spread them over
// its threads, or zero for an output that reads nothing but its own accumulator - a plain linear
// route therefore neither declares nor touches a line. See linear_finish_row.
template <class Output, class = void>
struct LinearRowPrefetch {
    static constexpr int words = 0;
    static constexpr int units = 0;
};

template <class Output>
struct LinearRowPrefetch<Output, std::void_t<decltype(Output::kPrefetchWords),
                                             decltype(Output::kPrefetchUnits)>> {
    static constexpr int words = Output::kPrefetchWords;
    static constexpr int units = Output::kPrefetchUnits;
};

// Row consumers run in the single thread owning the fully reduced row. They
// may publish fused Op state, but do not introduce warp/CTA synchronization.
// `row_prefetch` is the row's preloaded scratch line (see the NVFP4 GEMV's row prefetch): a line of
// 32-bit words the contraction filled from global memory before its mainloop, whose layout the
// output owns. It is null where the contraction prefetched nothing, and an output that consumes it
// reads the same values from global memory when it is.
template <class Output, class Epilogue, int Tokens>
__device__ __forceinline__ void
linear_finish_row(const Output& output, const Epilogue& epilogue, int row, int token_begin,
                  const float (&values)[Tokens], int active_tokens,
                  const std::uint32_t* row_prefetch = nullptr) {
    if constexpr (requires {
                      epilogue.apply_row(output, row, token_begin, values, active_tokens,
                                         row_prefetch);
                  }) {
        epilogue.apply_row(output, row, token_begin, values, active_tokens, row_prefetch);
    } else if constexpr (requires {
                             epilogue.apply_row(output, row, token_begin, values, active_tokens);
                         }) {
        epilogue.apply_row(output, row, token_begin, values, active_tokens);
    } else {
#pragma unroll
        for (int token = 0; token < Tokens; ++token) {
            if (token < active_tokens) {
                const int global_token = token_begin + token;
                output.store(row, global_token, epilogue.apply(row, global_token, values[token]));
            }
        }
    }
}

} // namespace infernix::ops::detail
