// Throughput of the CPU expert worker team under a core-placement policy
// (core/cpu_topology.h; docs/maintainer/qwen3_8-flash-next-design.md §10.3). Builds a pool of
// experts and one job set, runs timed rounds, and reports the median round time so
// `ckt ab --cmd-a "... --pin none" --cmd-b "... --pin 1to1"` compares policies under identical work.
//
// Policies: "--pin none" leaves placement to the scheduler; "--pin 1to1" hands the team the
// performance tier through CpuExpertTeam::placement_cpus (worker w to cpus[w], one core each);
// "--pin mask" confines the whole process to the performance tier with a single inherited affinity
// mask (the pre-1:1 policy, where CFS may land two workers on one core); "--pin all" confines the
// process to every core (a scheduler no-op that only fixes the mask).
//
// "--cuda on" additionally creates a CUDA context and keeps a background thread issuing H2D
// copies, imitating a live engine (driver thread, staging traffic and a busy unified bus) so a
// placement win measured on an idle host does not go unchallenged.

#include "core/cpu_topology.h"
#include "ops/offloaded_moe_fixtures.h"
#include "ops/offloaded_sparse_moe/cpu/expert_team.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <sched.h>

namespace moe      = infernix::ops::offloaded_moe;
namespace fixtures = infernix::test::offloaded_moe;

namespace {

struct Round {
    std::vector<fixtures::Expert> experts;
    std::vector<std::vector<std::uint16_t>> x, y;
    std::vector<moe::CpuExpertJob> jobs;
};

Round make_round(std::mt19937& rng, const std::vector<fixtures::Expert>& pool, int n_jobs, int columns) {
    Round r;
    r.experts.reserve(static_cast<std::size_t>(n_jobs));
    for (int j = 0; j < n_jobs; ++j) {
        r.experts.push_back(pool[rng() % pool.size()]);
        r.x.push_back(fixtures::random_activations(rng, columns));
        r.y.emplace_back(static_cast<std::size_t>(columns) * moe::kHidden, 0xFFFF);
    }
    for (int j = 0; j < n_jobs; ++j) {
        moe::CpuExpertJob job;
        job.record = r.experts[static_cast<std::size_t>(j)].record.data();
        job.scales = r.experts[static_cast<std::size_t>(j)].scales;
        job.ncols  = columns;
        for (int c = 0; c < columns; ++c) {
            job.x[c] = &r.x[static_cast<std::size_t>(j)][static_cast<std::size_t>(c) * moe::kHidden];
            job.y[c] = &r.y[static_cast<std::size_t>(j)][static_cast<std::size_t>(c) * moe::kHidden];
        }
        r.jobs.push_back(job);
    }
    return r;
}

std::vector<int> online_all() {
    std::vector<int> ids;
    const int n = static_cast<int>(std::thread::hardware_concurrency());
    for (int i = 0; i < n; ++i) { ids.push_back(i); }
    return ids;
}

void set_process_affinity(const std::vector<int>& cpus) {
    cpu_set_t set;
    CPU_ZERO(&set);
    for (const int cpu : cpus) { CPU_SET(cpu, &set); }
    if (sched_setaffinity(0, sizeof(set), &set) != 0) { std::perror("sched_setaffinity"); }
}

// A stand-in for a live engine's host side: an own CUDA context plus a background thread that keeps
// the unified bus busy with H2D staging copies. Without it a placement win is measured only against
// an idle CPU, with no driver thread competing for the performance cores and no memory contention.
struct CudaLoad {
    std::atomic<bool> stop{false};
    std::thread thread;
    void start() {
        cudaFree(nullptr); // create the context
        void* host   = nullptr;
        void* device = nullptr;
        const std::size_t bytes = 4u << 20;
        if (cudaMallocHost(&host, bytes) != cudaSuccess || cudaMalloc(&device, bytes) != cudaSuccess) {
            std::fprintf(stderr, "cuda load setup failed\n");
            return;
        }
        thread = std::thread([this, host, device, bytes] {
            cudaStream_t stream = nullptr;
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
            while (!stop.load(std::memory_order_relaxed)) {
                cudaMemcpyAsync(device, host, bytes, cudaMemcpyHostToDevice, stream);
                cudaStreamSynchronize(stream);
            }
            cudaStreamDestroy(stream);
        });
    }
    ~CudaLoad() {
        stop.store(true);
        if (thread.joinable()) { thread.join(); }
    }
};

} // namespace

int main(int argc, char** argv) {
    int workers = 6, rounds = 60, warmup = 8, jobs = 4, columns = moe::kMaxColumns;
    std::string pin       = "none";
    std::string cuda_load = "off";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto value = [&](const char* flag) {
            if (a.rfind(flag, 0) == 0 && a.size() > std::string(flag).size() + 1) { return a.substr(std::string(flag).size() + 1); }
            return std::string{};
        };
        if (const auto v = value("--workers"); !v.empty()) { workers = std::atoi(v.c_str()); }
        else if (const auto v = value("--rounds"); !v.empty()) { rounds = std::atoi(v.c_str()); }
        else if (const auto v = value("--warmup"); !v.empty()) { warmup = std::atoi(v.c_str()); }
        else if (const auto v = value("--jobs"); !v.empty()) { jobs = std::atoi(v.c_str()); }
        else if (const auto v = value("--columns"); !v.empty()) { columns = std::atoi(v.c_str()); }
        else if (const auto v = value("--pin"); !v.empty()) { pin = v; }
        else if (const auto v = value("--cuda"); !v.empty()) { cuda_load = v; }
        else { std::fprintf(stderr, "usage: --pin=none|1to1|mask|shared|all --cuda=on|off --workers N --jobs J --columns C --rounds R --warmup W\n"); return 2; }
    }
    if (pin != "none" && pin != "1to1" && pin != "mask" && pin != "shared" && pin != "all") { std::fprintf(stderr, "bad --pin\n"); return 2; }
    if (cuda_load != "on" && cuda_load != "off") { std::fprintf(stderr, "bad --cuda\n"); return 2; }

    // "mask" confines the whole process (and every thread it spawns, worker 0 included) to the
    // performance tier with one inherited affinity mask while the team pins nothing: the scheduler
    // may then stack two workers on one core, which the 1:1 assignment forbids.
    std::vector<std::uint64_t> cpus;
    if (pin == "1to1") { cpus = moe::CpuExpertTeam::placement_cpus(workers); }
    else if (pin == "mask") { set_process_affinity(infernix::core::performance_cpus()); }
    else if (pin == "all") { set_process_affinity(online_all()); }
    else if (pin == "shared") {
        // The pre-fix shared-mask mechanism: CpuExpertTeam takes its shared-mask fallback (all
        // spawned workers pinned to one common mask, so CFS may stack two workers on one core)
        // while worker 0 (the caller) and the CUDA load thread roam freely. The fallback selects
        // when the cpus list has no more entries than workers, so the tier list is truncated to
        // exactly that width -- passing the full tier (10 entries for 6 workers) would select the
        // 1:1 path and silently measure 1to1 against itself.
        const auto perf = infernix::core::performance_cpus();
        const std::size_t n = std::min<std::size_t>(perf.size(), static_cast<std::size_t>(workers));
        for (std::size_t i = 0; i < n; ++i) {
            const int c = perf[i];
            cpus.push_back(c >= 0 && c < 64 ? (std::uint64_t{1} << c) : 0);
        }
    }

    CudaLoad load;
    if (cuda_load == "on") { load.start(); }

    std::mt19937 rng(12345);
    std::vector<fixtures::Expert> pool;
    for (int i = 0; i < 6; ++i) { pool.push_back(fixtures::random_expert(rng, false)); }

    moe::CpuExpertTeam::Options options;
    options.workers         = workers;
    options.max_jobs        = jobs;
    options.spin_iterations = 1 << 16;
    options.activation      = moe::ExpertActivation::kA4;
    options.cpus            = cpus;
    moe::CpuExpertTeam team(options);

    // Fixed job set: every round recomputes the same jobs (outputs overwritten), so the measured
    // work is identical across policies and repetitions.
    Round r = make_round(rng, pool, jobs, columns);

    for (int i = 0; i < warmup; ++i) { team.run(r.jobs); }
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(rounds));
    for (int i = 0; i < rounds; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        team.run(r.jobs);
        const auto t1 = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    std::sort(samples.begin(), samples.end());
    const double median = samples[samples.size() / 2];
    std::printf("workers=%d jobs=%d cols=%d pin=%s cuda=%s cpus=%zu median=%.3f us\n", workers, jobs, columns,
                pin.c_str(), cuda_load.c_str(), cpus.size(), median);
    return 0;
}
