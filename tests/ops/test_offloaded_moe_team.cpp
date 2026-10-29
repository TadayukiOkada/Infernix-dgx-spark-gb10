// The CPU expert worker team of offloaded_sparse_moe (docs/maintainer/qwen3_8-flash-next-design.md
// §10.3): any worker count, job mix and ISA gives the single-thread expert_forward bits, and the
// team survives many rounds with workers spinning and parking. Host-only: needs no GPU.

#include "ops/op_family.h"
#include "ops/offloaded_sparse_moe/cpu/expert_team.h"
#include "ops/offloaded_moe_fixtures.h"

#include <chrono>
#include <cstdio>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <pthread.h>

namespace moe      = infernix::ops::offloaded_moe;
namespace fixtures = infernix::test::offloaded_moe;

namespace {

int g_failures = 0;
bool g_quick   = false; // --quick: fewer workers and rounds, for sanitizer builds

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

struct Round {
    std::vector<fixtures::Expert> experts;
    std::vector<std::vector<std::uint16_t>> x, y, ref;
    std::vector<int> ncols;
    std::vector<moe::CpuExpertJob> jobs;
};

Round make_round(std::mt19937& rng, const std::vector<fixtures::Expert>& pool, int n_jobs,
                 moe::ExpertActivation activation = moe::ExpertActivation::kA4,
                 const moe::ExpertShape& shape = moe::kFlashNextShape) {
    const bool a16 = activation == moe::ExpertActivation::kA16;
    const auto hidden = static_cast<std::size_t>(shape.hidden);
    Round r;
    for (int j = 0; j < n_jobs; ++j) {
        const int n = 1 + static_cast<int>(rng() % moe::kMaxColumns);
        r.experts.push_back(pool[rng() % pool.size()]);
        r.x.push_back(a16 ? fixtures::wide_range_activations(rng, n, shape.hidden)
                          : fixtures::random_activations(rng, n, shape.hidden));
        r.y.emplace_back(static_cast<std::size_t>(n) * hidden, 0xFFFF);
        r.ref.emplace_back(static_cast<std::size_t>(n) * hidden);
        r.ncols.push_back(n);
    }
    for (int j = 0; j < n_jobs; ++j) {
        moe::CpuExpertJob job;
        job.record = r.experts[j].record.data();
        job.scales = r.experts[j].scales;
        job.ncols  = r.ncols[j];
        const std::uint16_t* xr[moe::kMaxColumns];
        std::uint16_t* yr[moe::kMaxColumns];
        for (int c = 0; c < job.ncols; ++c) {
            job.x[c] = xr[c] = &r.x[j][static_cast<std::size_t>(c) * hidden];
            job.y[c] = &r.y[j][static_cast<std::size_t>(c) * hidden];
            yr[c]    = &r.ref[j][static_cast<std::size_t>(c) * hidden];
        }
        const moe::CpuIsa isa = g_quick ? moe::best_cpu_isa() : moe::CpuIsa::kScalar;
        if (a16) {
            moe::expert_forward_a16(isa, shape, job.record, job.scales, job.ncols, xr, yr);
        } else {
            moe::expert_forward(isa, job.record, job.scales, job.ncols, xr, yr);
        }
        r.jobs.push_back(job);
    }
    return r;
}

bool equal(const Round& r) {
    for (std::size_t j = 0; j < r.y.size(); ++j) {
        if (r.y[j] != r.ref[j]) { return false; }
    }
    return true;
}

void test_worker_counts() {
    std::mt19937 rng(5);
    std::vector<fixtures::Expert> pool;
    for (int i = 0; i < 4; ++i) { pool.push_back(fixtures::random_expert(rng, i == 1)); }
    const std::vector<int> counts = g_quick ? std::vector<int>{1, 3, 4} : std::vector<int>{1, 2, 3, 4, 7, 13, 40};
    for (int workers : counts) {
        moe::CpuExpertTeam team({.workers = workers, .spin_iterations = 256, .cpus = {}});
        for (int n_jobs : {1, 2, 5}) {
            Round r = make_round(rng, pool, n_jobs);
            team.run(r.jobs);
            if (!equal(r)) {
                std::fprintf(stderr, "workers %d jobs %d differ\n", workers, n_jobs);
                check(false, "team output equals expert_forward");
            }
        }
    }
}

// A16 teams (design §16.2.1): any worker count gives expert_forward_a16's bits, at Flash-Next's
// geometry and at Qwen3.5-122B-A10B's (up to I / 16 = 64 workers).
void test_worker_counts_a16(const moe::ExpertShape& shape) {
    std::mt19937 rng(6);
    std::vector<fixtures::Expert> pool;
    for (int i = 0; i < 4; ++i) { pool.push_back(fixtures::random_a16_expert(rng, shape)); }
    const int most = shape.h_blocks();
    const std::vector<int> counts = g_quick ? std::vector<int>{1, 4} : std::vector<int>{1, 2, 3, 5, 13, most};
    for (int workers : counts) {
        moe::CpuExpertTeam team({.workers = workers, .spin_iterations = 256,
                                 .activation = moe::ExpertActivation::kA16, .shape = shape, .cpus = {}});
        for (int n_jobs : {1, 3, 6}) {
            Round r = make_round(rng, pool, n_jobs, moe::ExpertActivation::kA16, shape);
            team.run(r.jobs);
            if (!equal(r)) {
                std::fprintf(stderr, "A16 workers %d jobs %d differ\n", workers, n_jobs);
                check(false, "A16 team output equals expert_forward_a16");
            }
        }
    }
}

// The core-placement policy: on a big.LITTLE host it returns the performance tier only when the
// whole team fits on it, and no preference otherwise or on a symmetric host. The gate is the
// measured one: a team larger than the fast tier must not be pinned to it (it oversubscribes the
// fast cores and stalls the spin barriers).
void test_placement_policy() {
    const int big = static_cast<int>(moe::CpuExpertTeam::placement_cpus(1).size());
    // workers=1 always fits any non-empty tier, so it reveals the tier size on this host.
    if (big > 0) {
        // A team at or below the tier size gets the full tier; one above it gets nothing.
        const auto at = moe::CpuExpertTeam::placement_cpus(big);
        const auto over = moe::CpuExpertTeam::placement_cpus(big + 1);
        check(static_cast<int>(at.size()) == big, "a team that fits the tier is pinned to it");
        check(over.empty(), "a team larger than the tier is left unpinned");
    }
    // Placement never changes the bits: pinned and unpinned teams give expert_forward's output.
    std::mt19937 rng(21);
    std::vector<fixtures::Expert> pool;
    for (int i = 0; i < 3; ++i) { pool.push_back(fixtures::random_expert(rng, false)); }
    moe::CpuExpertTeam team({.workers = 4, .spin_iterations = 256,
                             .cpus = moe::CpuExpertTeam::placement_cpus(4)});
    // The one-to-one assignment pins worker 0 (the caller) only for a round and gives its prior
    // affinity back: a caller left on one core would strand the engine's main thread.
    cpu_set_t before;
    CPU_ZERO(&before);
    pthread_getaffinity_np(pthread_self(), sizeof(before), &before);
    const int before_count = CPU_COUNT(&before);
    for (int n_jobs : {1, 3, 5}) {
        Round r = make_round(rng, pool, n_jobs);
        team.run(r.jobs);
        if (!equal(r)) { check(false, "pinned team output equals expert_forward"); }
    }
    cpu_set_t after;
    CPU_ZERO(&after);
    pthread_getaffinity_np(pthread_self(), sizeof(after), &after);
    check(CPU_COUNT(&after) == before_count, "the caller's affinity is restored after a round");
}

void test_many_rounds_with_parking() {
    std::mt19937 rng(9);
    std::vector<fixtures::Expert> pool;
    for (int i = 0; i < 3; ++i) { pool.push_back(fixtures::random_expert(rng, false)); }
    moe::CpuExpertTeam team({.workers = 4, .spin_iterations = 64, .cpus = {}});
    int bad = 0;
    for (int round = 0; round < (g_quick ? 12 : 60); ++round) {
        Round r = make_round(rng, pool, 1 + round % 3);
        if (round % 10 == 0) { std::this_thread::sleep_for(std::chrono::milliseconds(2)); } // workers park
        team.run(r.jobs);
        bad += equal(r) ? 0 : 1;
    }
    check(bad == 0, "every round equals expert_forward, across spinning and parked workers");
}

} // namespace

INFERNIX_OP_TEST(offloaded_moe_team) {
    g_quick = argc > 1 && std::string(argv[1]) == "--quick";
    test_worker_counts();
    test_worker_counts_a16(moe::kFlashNextShape);
    test_worker_counts_a16(moe::kQwen3_5_122BShape);
    test_placement_policy();
    test_many_rounds_with_parking();
    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all offloaded_moe team checks passed\n");
    return 0;
}
