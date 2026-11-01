#pragma once

#include "ops/common/math.cuh"
#include "ops/gdn_input_proj/gdn_conv_output.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace infernix::ops::detail {

// Output map of the single-parent [query,key,value,z] GDN projection with the width-four causal
// convolution and SiLU in its epilogue, for a call whose contraction columns are B independent
// sequences of one position each (gdn_input_proj_conv_snapshot at W = 1). Contraction column b is
// sequence b. Rows [0,10240) are convolution channels: each consumes the FP32 projected value p
// as its current tap, with the three history taps read from the BF16 window in
// initial_state_slots[b], writes BF16 SiLU(conv) to query, key or value column b, and publishes
// the window (history1, history2, bf16(p)) to snapshot_base_slots[b]. Rows [10240,16384) are z,
// stored as bf16(p). The convolution's FMA order is the snapshot kernels' (oldest tap first, from
// zero), so the convolution differs from the composed projection + snapshot route only in that
// the current tap is p rather than bf16(p); the published window and z are those routes' bits.
struct GdnConvSnapshotColumnsOutput {
    const __nv_bfloat16* conv_weight;        // [C,4]: tap j of channel c at j*C+c, oldest first
    __nv_bfloat16* conv_states;              // [C,3,Slots]: history oldest first
    const std::int32_t* initial_state_slots; // [B]
    const std::int32_t* snapshot_base_slots; // [B]
    __nv_bfloat16* query;                    // [2048,B]
    __nv_bfloat16* key;                      // [2048,B]
    __nv_bfloat16* value;                    // [6144,B]
    __nv_bfloat16* z;                        // [6144,B]

    // The row scratch a contraction may preload: the four convolution taps, then per contraction
    // column c its three history taps (each a BF16 pattern in the low half of its word) and that
    // sequence's snapshot slot, at 4 + 4*c. A convolution row reads its own taps and its live
    // columns' windows; a z row reads none. The line is sized for the widest contraction the route
    // serves (one column per sequence, B <= 8) and is filled unit by unit, so a caller may spread a
    // line over threads or fill it from one.
    static constexpr int kPrefetchColumns = 8;
    static constexpr int kPrefetchUnits   = 1 + kPrefetchColumns;
    static constexpr int kPrefetchWords   = 4 * kPrefetchUnits;

    // Fills one unit of a row's line: unit 0 the four convolution taps, unit 1 + c the window of
    // contraction column c. A unit whose column is at or beyond `columns` (the exclusive upper bound
    // on the absolute column indices the call covers) or beyond the line writes nothing. See
    // prefetch_row, which a contraction calls for each row of its tile before the contraction.
    __device__ __forceinline__ void prefetch_unit(std::uint32_t* words, int row, int unit,
                                                  int columns) const {
        if (row >= kGdnChannels) { return; }
        constexpr std::int64_t kChannels = kGdnChannels;
        if (unit == 0) {
#pragma unroll
            for (int tap = 0; tap < 4; ++tap) {
                words[tap] =
                    bf16_word(conv_weight[static_cast<std::int64_t>(tap) * kChannels + row]);
            }
            return;
        }
        const int column = unit - 1;
        if (column >= columns || column >= kPrefetchColumns) { return; }
        const __nv_bfloat16* initial = conv_states +
            static_cast<std::int64_t>(initial_state_slots[column]) * (3 * kChannels);
        std::uint32_t* window        = words + 4 + 4 * column;
#pragma unroll
        for (int tap = 0; tap < 3; ++tap) {
            window[tap] = bf16_word(initial[static_cast<std::int64_t>(tap) * kChannels + row]);
        }
        window[3] = static_cast<std::uint32_t>(snapshot_base_slots[column]);
    }

    // Fills a whole row line. A contraction calls this once per row of its tile before the
    // contraction, so the memory round trip overlaps it instead of running in the row consumer's
    // tail; store_row below reads the same values from global memory where no line was preloaded.
    __device__ __forceinline__ void prefetch_row(std::uint32_t* words, int row,
                                                 int columns) const {
#pragma unroll
        for (int unit = 0; unit < kPrefetchUnits; ++unit) {
            prefetch_unit(words, row, unit, columns);
        }
    }

    // The line of a contraction that covers one column: the taps and column zero's window, and no
    // slot for the columns a single-column call cannot name. A contraction whose rows each have one
    // owner keeps this in that thread's registers, so its loads go out before the contraction and
    // are read back by the same thread without ever touching shared memory.
    static constexpr int kPrefetchLiveWords = 4 * 2;
    __device__ __forceinline__ void prefetch_single_row(std::uint32_t* words, int row) const {
        prefetch_unit(words, row, 0, 1);
        prefetch_unit(words, row, 1, 1);
    }

    // One fully reduced row over contraction columns [column_begin, column_begin + active). Every
    // history the row needs is loaded before the row publishes any window, so a sequence's
    // snapshot slot may be its own initial slot (the decode configuration). Distinct sequences
    // name distinct slots, and distinct rows touch distinct channels, so no other thread reads a
    // location this one writes.
    template <int Columns>
    __device__ __forceinline__ void store_row(int row, int column_begin,
                                              const float (&projected)[Columns],
                                              int active) const {
        store_row<Columns, false>(row, column_begin, projected, active, nullptr);
    }

    // The same row over a line the contraction preloaded with prefetch_row.
    template <int Columns>
    __device__ __forceinline__ void store_row(int row, int column_begin,
                                              const float (&projected)[Columns], int active,
                                              const std::uint32_t* words) const {
        static_assert(Columns <= kPrefetchColumns,
                      "a preloaded row line covers the route's widest contraction");
        store_row<Columns, true>(row, column_begin, projected, active, words);
    }

    __device__ __forceinline__ void store(int row, int column, float projected) const {
        const float values[1]{projected};
        store_row(row, column, values, 1);
    }

private:
    static __device__ __forceinline__ std::uint32_t bf16_word(__nv_bfloat16 value) {
        return static_cast<std::uint32_t>(__bfloat16_as_ushort(value));
    }

    static __device__ __forceinline__ __nv_bfloat16 bf16_pattern(std::uint32_t word) {
        return __ushort_as_bfloat16(static_cast<unsigned short>(word));
    }

    template <int Columns, bool Prefetched>
    __device__ __forceinline__ void store_row(int row, int column_begin,
                                              const float (&projected)[Columns], int active,
                                              const std::uint32_t* words) const {
        if (row >= kGdnChannels) {
#pragma unroll
            for (int column = 0; column < Columns; ++column) {
                if (column < active) {
                    z[static_cast<std::int64_t>(column_begin + column) * kGdnZRows + row -
                      kGdnChannels] = __float2bfloat16_rn(projected[column]);
                }
            }
            return;
        }

        constexpr std::int64_t kChannels  = kGdnChannels;
        constexpr std::int64_t kSlotWidth = 3 * kChannels;
        float w[4];
#pragma unroll
        for (int tap = 0; tap < 4; ++tap) {
            w[tap] = Prefetched ? __bfloat162float(bf16_pattern(words[tap]))
                                : __bfloat162float(
                                      conv_weight[static_cast<std::int64_t>(tap) * kChannels + row]);
        }

        __nv_bfloat16 history[Columns][3];
        std::int32_t snapshot_slot[Columns];
#pragma unroll
        for (int column = 0; column < Columns; ++column) {
            if (column < active) {
                const int sequence = column_begin + column;
                if constexpr (Prefetched) {
                    const std::uint32_t* window = words + 4 + 4 * sequence;
#pragma unroll
                    for (int tap = 0; tap < 3; ++tap) {
                        history[column][tap] = bf16_pattern(window[tap]);
                    }
                    snapshot_slot[column] = static_cast<std::int32_t>(window[3]);
                } else {
                    const __nv_bfloat16* initial = conv_states +
                        static_cast<std::int64_t>(initial_state_slots[sequence]) * kSlotWidth;
                    history[column][0]    = initial[row];
                    history[column][1]    = initial[kChannels + row];
                    history[column][2]    = initial[2 * kChannels + row];
                    snapshot_slot[column] = snapshot_base_slots[sequence];
                }
            }
        }

        __nv_bfloat16* destination;
        std::int32_t destination_rows;
        std::int32_t local_row;
        if (row < kGdnQueryRows) {
            destination      = query;
            destination_rows = kGdnQueryRows;
            local_row        = row;
        } else if (row < kGdnQueryRows + kGdnKeyRows) {
            destination      = key;
            destination_rows = kGdnKeyRows;
            local_row        = row - kGdnQueryRows;
        } else {
            destination      = value;
            destination_rows = kGdnValueRows;
            local_row        = row - kGdnQueryRows - kGdnKeyRows;
        }

#pragma unroll
        for (int column = 0; column < Columns; ++column) {
            if (column < active) {
                const int sequence = column_begin + column;
                const float p      = projected[column];
                float conv         = fmaf(w[0], __bfloat162float(history[column][0]), 0.0F);
                conv               = fmaf(w[1], __bfloat162float(history[column][1]), conv);
                conv               = fmaf(w[2], __bfloat162float(history[column][2]), conv);
                conv               = fmaf(w[3], p, conv);
                destination[static_cast<std::int64_t>(sequence) * destination_rows + local_row] =
                    __float2bfloat16_rn(silu(conv));

                __nv_bfloat16* snapshot =
                    conv_states + static_cast<std::int64_t>(snapshot_slot[column]) * kSlotWidth;
                snapshot[row]                 = history[column][1];
                snapshot[kChannels + row]     = history[column][2];
                snapshot[2 * kChannels + row] = __float2bfloat16_rn(p);
            }
        }
    }
};

// Hands each fully reduced row to its output map's row store (every Q8 decode contraction finishes
// a row in one thread), so the convolution issues all of a row's history loads before its first
// store instead of one dependent round trip per column.
struct GdnConvSnapshotRowEpilogue {
    __device__ __forceinline__ float apply(int, int, float value) const { return value; }

    template <class Output, int Columns>
    __device__ __forceinline__ void apply_row(const Output& output, int row, int column_begin,
                                              const float (&projected)[Columns], int active) const {
        output.store_row(row, column_begin, projected, active);
    }
};

// The hand-off for a contraction that preloaded each row's scratch line before its mainloop: the
// line travels with the row to the prefetched store. It is a separate type rather than another
// overload of the one above because a contraction either fills lines or does not, and a contraction
// that does not must not select the prefetched store with a null line.
struct GdnConvSnapshotPrefetchedRowEpilogue : GdnConvSnapshotRowEpilogue {
    using GdnConvSnapshotRowEpilogue::apply_row;

    template <class Output, int Columns>
    __device__ __forceinline__ void apply_row(const Output& output, int row, int column_begin,
                                              const float (&projected)[Columns], int active,
                                              const std::uint32_t* words) const {
        output.store_row(row, column_begin, projected, active, words);
    }
};

} // namespace infernix::ops::detail
