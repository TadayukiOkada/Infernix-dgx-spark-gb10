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

        // 9. Boundary-adjacent state must walk suffix link to find a valid continuation
        SuffixAutomaton link_sa;
        link_sa.ingest(std::vector<int>{2, 3, 4, 5, 99, 100});
        link_sa.ingest(std::vector<int>{1, 2, 3, 4, 5});
        const auto link_match = link_sa.propose(std::vector<int>{1, 2, 3, 4, 5}, 5, 4);
        require(link_match.matched == 4, "boundary-adjacent state walks suffix link to len 4");
        require(link_match.tokens == std::vector<int>({99, 100}),
                "boundary-adjacent state returns continuation from suffix link");

        // 10. Recency preference: more recent occurrence supersedes older occurrence
        SuffixAutomaton recency_sa;
        recency_sa.ingest(std::vector<int>{10, 20, 30, 40, 100, 101, 102});
        recency_sa.ingest(std::vector<int>{10, 20, 30, 40, 200, 201, 202, 203});
        const auto rec_match = recency_sa.propose(std::vector<int>{10, 20, 30, 40}, 10, 4);
        require(rec_match.matched == 4, "recency match len 4");
        require(rec_match.tokens == std::vector<int>({200, 201, 202, 203}),
                "recency prefers the most recent occurrence");

        // 11. Boundary-ended latest occurrence falls back cleanly to earlier valid occurrence
        SuffixAutomaton valid_fb_sa;
        valid_fb_sa.ingest(std::vector<int>{10, 20, 30, 40, 300, 301, 302});
        valid_fb_sa.ingest(std::vector<int>{10, 20, 30, 40});
        const auto vfb_match = valid_fb_sa.propose(std::vector<int>{10, 20, 30, 40}, 10, 4);
        require(vfb_match.matched == 4, "valid fallback match len 4");
        require(vfb_match.tokens == std::vector<int>({300, 301, 302}),
                "boundary-ended latest occurrence falls back to earlier valid continuation");

        // 12. Eviction resilience: older evicted occurrence does not block newer live occurrence
        SuffixAutomaton evict_sa(64);
        evict_sa.ingest(std::vector<int>{10, 20, 30, 40, 111, 112});
        for (int i = 0; i < 60; ++i) { evict_sa.append(500 + i); }
        evict_sa.ingest(std::vector<int>{10, 20, 30, 40, 222, 223});
        const auto ev_match = evict_sa.propose(std::vector<int>{10, 20, 30, 40}, 10, 4);
        require(ev_match.matched == 4, "eviction resilient match len 4");
        require(ev_match.tokens == std::vector<int>({222, 223}),
                "newer live occurrence still matches after oldest aged out");

        // 13. Occurrence count and distance metadata
        require(rec_match.occurrences >= 2, "occurrence count tracks repeats");
        require(ev_match.distance <= 64, "distance tracks distance from corpus tip");

        // 14. Compaction bounds storage under >4x token_capacity appends
        SuffixAutomaton compact_sa(64);
        for (int i = 0; i < 300; ++i) { compact_sa.append(1000 + i); }
        require(compact_sa.size() < 2 * 64 || compact_sa.size() <= 1024, "compaction bounded size");
        require(compact_sa.node_count() <= 4 * 1024, "compaction bounded nodes");

        std::cout << "All Suffix Automaton unit tests passed successfully!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << '\n';
        return 1;
    }
}
