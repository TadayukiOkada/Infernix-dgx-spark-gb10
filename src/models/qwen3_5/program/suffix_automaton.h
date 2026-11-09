#pragma once

#include "models/qwen3_5/ngram.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace infernix::models::qwen3_5::detail {

// Suffix Automaton (SAM) speculative draft proposer.
// Represents all substrings of the prompt and generated history in linear space O(N) with at most
// 2N states and 3N transitions. Provides exact longest proper suffix lookup in O(1) amortized
// time by traversing suffix link pointers, replacing fixed-window multi-hash tables with zero
// hash collisions and arbitrary match length discovery.
class SuffixAutomaton {
public:
    using Token = TokenId;
    using Match = NgramMatch;

    struct LookupResult {
        int32_t pos = 0;
        int32_t len = 0;
    };

    explicit SuffixAutomaton(std::size_t token_capacity = 1U << 20)
        : token_capacity_(power_of_two(token_capacity) && token_capacity >= 64 ? token_capacity : 0) {
        if (token_capacity_ == 0) {
            throw std::invalid_argument("invalid suffix automaton capacity");
        }
        nodes_.push_back(Node{});
    }

    [[nodiscard]] std::size_t capacity() const noexcept { return token_capacity_; }
    [[nodiscard]] std::size_t size() const noexcept { return tokens_.size(); }
    [[nodiscard]] std::size_t node_count() const noexcept { return nodes_.size(); }

    void clear() {
        nodes_.clear();
        edges_.clear();
        tokens_.clear();
        nodes_.push_back(Node{});
        last_ = 0;
    }

    void boundary() {
        tokens_.push_back(-1);
        last_ = 0;
    }

    void set_boundaries(std::vector<Token> tokens) {
        boundaries_ = std::move(tokens);
        std::sort(boundaries_.begin(), boundaries_.end());
    }

    [[nodiscard]] bool is_boundary(Token t) const noexcept {
        if (t < 0) { return true; }
        return std::binary_search(boundaries_.begin(), boundaries_.end(), t);
    }

    void append(Token token) {
        if (is_boundary(token)) {
            boundary();
            return;
        }
        tokens_.push_back(token);
        const auto cur = static_cast<int32_t>(nodes_.size());
        nodes_.push_back(Node{
            .len        = nodes_[last_].len + 1,
            .link       = 0,
            .pos        = static_cast<int32_t>(tokens_.size() - 1),
            .first_key  = -1,
            .first_to   = -1,
            .next_edge  = -1
        });

        int32_t p = last_;
        while (p != -1 && get_transition(p, token) == -1) {
            set_transition(p, token, cur);
            p = nodes_[p].link;
        }

        if (p == -1) {
            nodes_[cur].link = 0;
        } else {
            const int32_t q = get_transition(p, token);
            if (nodes_[p].len + 1 == nodes_[q].len) {
                nodes_[cur].link = q;
            } else {
                const auto clone = static_cast<int32_t>(nodes_.size());
                nodes_.push_back(Node{
                    .len        = nodes_[p].len + 1,
                    .link       = nodes_[q].link,
                    .pos        = nodes_[q].pos,
                    .first_key  = -1,
                    .first_to   = -1,
                    .next_edge  = -1
                });
                copy_transitions(clone, q);

                int32_t p_curr = p;
                while (p_curr != -1 && get_transition(p_curr, token) == q) {
                    set_transition(p_curr, token, clone);
                    p_curr = nodes_[p_curr].link;
                }
                nodes_[q].link   = clone;
                nodes_[cur].link = clone;
            }
        }
        last_ = cur;
    }

    void ingest(std::span<const Token> tokens) {
        boundary();
        for (const auto t : tokens) {
            append(t);
        }
        boundary();
    }

    // O(1) online lookup from the generation tip (last_): walks suffix links to find the longest
    // suffix that appeared earlier in the corpus.
    [[nodiscard]] std::optional<LookupResult> lookup() const noexcept {
        if (nodes_.empty() || last_ == 0) { return std::nullopt; }

        int32_t state = last_;
        int32_t best_state = 0;

        while (state != 0) {
            const auto& nd = nodes_[state];
            const bool is_last = (nd.pos >= 0 && nd.pos + 1 >= static_cast<int32_t>(tokens_.size()));
            if (!is_last) {
                best_state = state;
                break;
            }
            if (nd.link == -1) { break; }
            state = nd.link;
        }

        if (best_state == 0) { return std::nullopt; }
        const auto& best_node = nodes_[best_state];
        if (best_node.pos < 0) { return std::nullopt; }

        return LookupResult{
            .pos = best_node.pos + 1,
            .len = best_node.len
        };
    }

    [[nodiscard]] Match propose(std::span<const Token> history, std::uint32_t maximum,
                                std::uint32_t minimum_match = 12) const {
        Match match;
        if (maximum == 0 || history.empty()) { return match; }

        // Fast path: if history matches the active tail ending at last_, use O(1) suffix link traversal.
        int32_t state = 0;
        int32_t matched = 0;

        for (const auto t : history) {
            if (is_boundary(t)) {
                state   = 0;
                matched = 0;
                continue;
            }
            while (state != -1 && get_transition(state, t) == -1) {
                state   = nodes_[state].link;
                matched = state != -1 ? nodes_[state].len : 0;
            }
            if (state == -1) {
                state   = 0;
                matched = 0;
            } else {
                state = get_transition(state, t);
                matched++;
            }
        }

        if (state == 0 || matched < static_cast<int32_t>(minimum_match)) {
            return match;
        }

        const auto live = static_cast<int32_t>(live_begin());

        while (state != 0 && (nodes_[state].pos + 1 >= static_cast<int32_t>(tokens_.size()) ||
                              nodes_[state].pos - nodes_[state].len + 1 < live)) {
            state = nodes_[state].link;
            if (state != 0) { matched = std::min(matched, nodes_[state].len); }
        }

        if (state == 0 || matched < static_cast<int32_t>(minimum_match) || nodes_[state].pos < 0) {
            return match;
        }

        const int32_t match_end = nodes_[state].pos + 1;
        uint32_t avail = 0;
        while (avail < maximum && match_end + avail < static_cast<int32_t>(tokens_.size()) &&
               !is_boundary(tokens_[match_end + avail])) {
            avail++;
        }

        if (avail == 0) { return match; }

        match.matched = static_cast<uint32_t>(matched);
        match.tokens.resize(avail);
        for (uint32_t i = 0; i < avail; ++i) {
            match.tokens[i] = tokens_[match_end + i];
        }
        return match;
    }

    [[nodiscard]] Match propose_for_round(std::span<const Token> history, std::uint32_t maximum,
                                          std::uint32_t neural_drafts,
                                          std::uint32_t minimum_match = 12) const {
        if (maximum == 0) { return {}; }
        if (maximum > std::numeric_limits<std::uint32_t>::max() - 2U) {
            throw std::invalid_argument("ngram proposal lookahead overflows");
        }
        return finish_round(propose(history, maximum + 2U, minimum_match), maximum, neural_drafts);
    }

    [[nodiscard]] static Match finish_round(Match match, std::uint32_t maximum,
                                            std::uint32_t neural_drafts) {
        if (maximum == 0) { return {}; }
        if (maximum > std::numeric_limits<std::uint32_t>::max() - 2U) {
            throw std::invalid_argument("ngram proposal lookahead overflows");
        }
        const auto available = static_cast<std::uint32_t>(match.tokens.size());
        const auto drafts = std::min(maximum, available > 2U ? available - 2U : 0U);
        if (available < maximum + 2U && drafts <= neural_drafts) { return {}; }
        match.tokens.resize(drafts);
        return match;
    }

    [[nodiscard]] static bool maximal(const Match& match, std::size_t history_size,
                                      std::uint32_t maximum) {
        return match.matched == history_size && match.tokens.size() == maximum;
    }

private:
    struct Edge {
        Token key;
        int32_t to;
        int32_t next;
    };

    struct Node {
        int32_t len = 0;
        int32_t link = -1;
        int32_t pos = -1;
        Token first_key = -1;
        int32_t first_to = -1;
        int32_t next_edge = -1;
    };

    std::size_t token_capacity_ = 0;
    std::vector<Node> nodes_;
    std::vector<Edge> edges_;
    std::vector<Token> tokens_;
    std::vector<Token> boundaries_;
    int32_t last_ = 0;

    [[nodiscard]] static bool power_of_two(std::size_t value) noexcept {
        return value != 0 && (value & (value - 1)) == 0;
    }

    [[nodiscard]] std::uint64_t live_begin() const noexcept {
        return tokens_.size() > token_capacity_ ? tokens_.size() - token_capacity_ : 0;
    }

    [[nodiscard]] int32_t get_transition(int32_t u, Token token) const noexcept {
        if (nodes_[u].first_key == token) { return nodes_[u].first_to; }
        for (int32_t e = nodes_[u].next_edge; e != -1; e = edges_[e].next) {
            if (edges_[e].key == token) { return edges_[e].to; }
        }
        return -1;
    }

    void set_transition(int32_t u, Token token, int32_t to) {
        if (nodes_[u].first_key == token) {
            nodes_[u].first_to = to;
            return;
        }
        for (int32_t e = nodes_[u].next_edge; e != -1; e = edges_[e].next) {
            if (edges_[e].key == token) {
                edges_[e].to = to;
                return;
            }
        }
        if (nodes_[u].first_key == -1) {
            nodes_[u].first_key = token;
            nodes_[u].first_to  = to;
        } else {
            const auto new_edge = static_cast<int32_t>(edges_.size());
            edges_.push_back(Edge{token, to, nodes_[u].next_edge});
            nodes_[u].next_edge = new_edge;
        }
    }

    void copy_transitions(int32_t dst, int32_t src) {
        nodes_[dst].first_key = nodes_[src].first_key;
        nodes_[dst].first_to  = nodes_[src].first_to;
        nodes_[dst].next_edge = -1;
        for (int32_t e = nodes_[src].next_edge; e != -1; e = edges_[e].next) {
            const auto new_edge = static_cast<int32_t>(edges_.size());
            edges_.push_back(Edge{edges_[e].key, edges_[e].to, nodes_[dst].next_edge});
            nodes_[dst].next_edge = new_edge;
        }
    }
};

} // namespace infernix::models::qwen3_5::detail
