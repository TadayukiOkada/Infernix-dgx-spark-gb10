#pragma once

#include "models/qwen3_5/ngram.h"
#include "models/qwen3_5/program/suffix_automaton.h"

#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace infernix::models::qwen3_5::detail {

// Bounded, position-indexed proposal corpus backed solely by SuffixAutomaton.
// Never owns target model state.
class NgramProposer {
public:
    using Token = std::int32_t;
    using Match = NgramMatch;

    [[nodiscard]] static bool maximal(const Match& match, std::size_t history_size,
                                      std::uint32_t maximum) {
        return SuffixAutomaton::maximal(match, history_size, maximum);
    }

    explicit NgramProposer(std::size_t token_capacity = 1U << 20,
                           std::size_t bucket_count   = 1U << 19)
        : sa_(valid_dimensions(token_capacity, bucket_count)) {}

    [[nodiscard]] std::size_t capacity() const noexcept { return sa_.capacity(); }

    void boundary() { sa_.boundary(); }

    void set_boundaries(std::vector<Token> tokens) { sa_.set_boundaries(std::move(tokens)); }

    void append(Token token) { sa_.append(token); }

    void ingest(std::span<const Token> tokens) { sa_.ingest(tokens); }

    [[nodiscard]] Match propose_for_round(std::span<const Token> history, std::uint32_t maximum,
                                          std::uint32_t neural_drafts,
                                          std::uint32_t minimum_match = 12) const {
        return sa_.propose_for_round(history, maximum, neural_drafts, minimum_match);
    }

    [[nodiscard]] static Match finish_round(Match match, std::uint32_t maximum,
                                            std::uint32_t neural_drafts) {
        return SuffixAutomaton::finish_round(std::move(match), maximum, neural_drafts);
    }

    [[nodiscard]] Match propose(std::span<const Token> history, std::uint32_t maximum,
                                std::uint32_t minimum_match = 12) const {
        return sa_.propose(history, maximum, minimum_match);
    }

private:
    SuffixAutomaton sa_;

    [[nodiscard]] static bool power_of_two(std::size_t value) noexcept {
        return value != 0 && (value & (value - 1)) == 0;
    }

    [[nodiscard]] static std::size_t valid_dimensions(std::size_t token_capacity,
                                                      std::size_t bucket_count) {
        if (!power_of_two(token_capacity) || token_capacity < 64 || !power_of_two(bucket_count)) {
            throw std::invalid_argument("invalid ngram corpus capacity");
        }
        return token_capacity;
    }
};

} // namespace infernix::models::qwen3_5::detail