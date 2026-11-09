#include "models/qwen3_5/program/suffix_automaton.h"

#include <iostream>
#include <numeric>
#include <random>
#include <vector>

using infernix::models::qwen3_5::detail::SuffixAutomaton;

static void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

int main() {
    try {
        // 1. Empty and single-token
        SuffixAutomaton empty_sa;
        require(!empty_sa.lookup().has_value(), "empty automaton has no lookup match");
        empty_sa.append(1);
        require(!empty_sa.lookup().has_value(), "single token has no lookup match");

        // 2. Simple repetition (test from sa_spec: test_suffix_automaton.cc)
        SuffixAutomaton rep_sa;
        rep_sa.append(1);
        rep_sa.append(1);
        rep_sa.append(1);
        auto res = rep_sa.lookup();
        require(res.has_value(), "repetition should find lookup match");
        require(res->len == 2, "repetition match len should be 2");
        require(res->pos == 2, "repetition match pos should be 2");

        rep_sa.append(1);
        res = rep_sa.lookup();
        require(res.has_value(), "repetition 4 tokens should find match");
        require(res->len == 3, "repetition match len should be 3");
        require(res->pos == 3, "repetition match pos should be 3");

        // 3. Alternating repetition (test from sa_spec)
        SuffixAutomaton alt_sa;
        alt_sa.append(1);
        alt_sa.append(2);
        require(!alt_sa.lookup().has_value(), "distinct initial tokens have no match");
        alt_sa.append(1);
        res = alt_sa.lookup();
        require(res.has_value() && res->len == 1 && res->pos == 1, "first repeat of token 1");
        alt_sa.append(2);
        res = alt_sa.lookup();
        require(res.has_value() && res->len == 2 && res->pos == 2, "repeat of sequence [1, 2]");

        // 4. Overlapping repeats ("ababa" -> [1, 2, 1, 2, 1])
        SuffixAutomaton ababa;
        for (int t : {1, 2, 1, 2, 1}) { ababa.append(t); }
        res = ababa.lookup();
        require(res.has_value() && res->len == 3 && res->pos == 3, "ababa has suffix 'aba' of len 3");

        // 5. Baseten test_add_request pattern from sa_spec test_api.py
        SuffixAutomaton api_sa;
        for (int t : {1, 2, 3, 4, 1, 2, 3, 1, 2}) { api_sa.append(t); }
        api_sa.append(3);
        api_sa.append(4);
        res = api_sa.lookup();
        require(res.has_value(), "api pattern should match");
        require(res->len == 4, "api pattern match length should be 4");
        require(res->pos == 4, "api pattern continuation pos should be 4");

        // Propose from this match
        auto match = api_sa.propose(std::vector<int>{1, 2, 3, 4}, 2, 4);
        require(match.matched == 4, "matched 4 tokens");
        require(match.tokens == std::vector<int>({1, 2}), "draft tokens should be [1, 2]");

        // 6. Test CUDA graph pattern from sa_spec (prompt + new_tokens)
        SuffixAutomaton graph_sa;
        for (int t : {1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1}) { graph_sa.append(t); }
        graph_sa.append(2);
        graph_sa.append(3);
        res = graph_sa.lookup();
        // SAM correctly discovers the maximal 8-token matching suffix [1, 2, 3, 4, 5, 1, 2, 3]
        require(res.has_value() && res->len == 8, "graph pattern should match maximal suffix (8 tokens)");
        auto graph_match = graph_sa.propose(std::vector<int>{1, 2, 3}, 2, 3);
        require(graph_match.tokens == std::vector<int>({4, 5}), "drafts should be [4, 5]");

        // 7. Boundary test: ensure proposals never cross boundary markers
        SuffixAutomaton b_sa;
        b_sa.set_boundaries({99});
        b_sa.ingest(std::vector<int>{10, 20, 30, 40, 99, 50, 60});
        auto b_match = b_sa.propose(std::vector<int>{10, 20}, 5, 2);
        require(b_match.matched == 2, "b_match length 2");
        require(b_match.tokens == std::vector<int>({30, 40}), "proposal stops before boundary 99");

        // 8. Round admission and lookahead checks
        SuffixAutomaton round_sa(256);
        std::vector<int> src(128);
        std::iota(src.begin(), src.end(), 100);
        round_sa.ingest(src);
        for (unsigned n : {4, 8, 12, 16, 32, 64}) {
            const auto m = round_sa.propose(std::span<const int>(src).first(n), 15, 4);
            require(m.matched == n, "exact match length");
            require(m.tokens == std::vector<int>(src.begin() + n, src.begin() + n + 15), "exact span continuation");
        }

        std::cout << "All Suffix Automaton unit tests passed successfully!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << '\n';
        return 1;
    }
}
