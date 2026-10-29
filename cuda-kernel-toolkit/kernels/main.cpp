// main.cpp — driver for the toolkit's sample kernel suite.
//
// This is the *target executable* the toolkit analyses.  It provides a stable
// machine-readable contract so the toolkit can:
//
//   --list                     enumerate workloads (name, default config)
//   --workload NAME            select one workload
//   --n N --blocks B --threads T --smem S --iters I --warmup W
//                              run with an explicit configuration
//   --sweep                    emit a built-in parameter sweep
//   --verify                   run the correctness check for the last config
//   --query-grid               report the grid/block actually launched
//
// Every run prints line-oriented `key<TAB>value` records on stdout.  The
// toolkit's profiler wraps this executable with `ncu` for metric collection and
// parses the same records for timing.
#include "harness.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

void usage() {
    std::fprintf(stderr,
                 "usage: ckt_kernels --list | --workload NAME [options]\n"
                 "options:\n"
                 "  --n N          problem size (workload-specific meaning)\n"
                 "  --blocks B     grid x dimension (or total blocks)\n"
                 "  --threads T    threads per block\n"
                 "  --smem S       dynamic shared memory bytes\n"
                 "  --iters I      timed iterations (default 30)\n"
                 "  --warmup W     warm-up iterations (default 5)\n"
                 "  --verify       run the correctness check\n"
                 "  --repeat R     repeat the whole timed run R times (default 1)\n");
}

Workload* find_workload(const std::string& name) {
    for (auto& w : workload_registry()) {
        if (name == w.name) return &w;
    }
    return nullptr;
}

// The harness needs a zero-argument launcher for `time_launches`, so the active
// configuration is held in file-scope state.
Workload* g_active = nullptr;
int g_blocks = 1, g_threads = 256, g_smem = 0, g_iters = 1;

int launch_active() {
    cudaError_t e = g_active->run(g_blocks, g_threads, g_smem, g_iters);
    if (e != cudaSuccess) {
        std::fprintf(stderr, "launch failed: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        usage();
        return 2;
    }

    std::string workload;
    int n = -1, blocks = -1, threads = -1, smem = -1, iters = 30, warmup = 5, repeat = 1;
    bool do_verify = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", flag);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--list") {
            for (auto& w : workload_registry()) {
                int pn = 0, b = 0, t = 0, s = 0;
                w.defaults(pn, b, t, s);
                std::printf("workload\t%s\tn\t%d\tblocks\t%d\tthreads\t%d\tsmem\t%d\n", w.name, pn, b, t, s);
            }
            return 0;
        } else if (a == "--workload") {
            workload = next("--workload");
        } else if (a == "--n") {
            n = std::atoi(next("--n"));
        } else if (a == "--blocks") {
            blocks = std::atoi(next("--blocks"));
        } else if (a == "--threads") {
            threads = std::atoi(next("--threads"));
        } else if (a == "--smem") {
            smem = std::atoi(next("--smem"));
        } else if (a == "--iters") {
            iters = std::atoi(next("--iters"));
        } else if (a == "--warmup") {
            warmup = std::atoi(next("--warmup"));
        } else if (a == "--repeat") {
            repeat = std::atoi(next("--repeat"));
        } else if (a == "--verify") {
            do_verify = true;
        } else if (a == "-h" || a == "--help") {
            usage();
            return 0;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
            usage();
            return 2;
        }
    }

    if (workload.empty()) {
        std::fprintf(stderr, "--workload is required (or --list)\n");
        return 2;
    }
    Workload* w = find_workload(workload);
    if (!w) {
        std::fprintf(stderr, "unknown workload: %s\n", workload.c_str());
        return 2;
    }

    int pn = 0, db = 0, dt = 0, ds = 0;
    w->defaults(pn, db, dt, ds);
    if (n < 0) n = pn;
    if (blocks < 0) blocks = db;
    if (threads < 0) threads = dt;
    if (smem < 0) smem = ds;

    // Device snapshot for provenance in the toolkit's report.
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    emit_kv("device.name", std::string(prop.name));
    emit_kv("device.sms", (long long)prop.multiProcessorCount);
    emit_kv("workload", w->name);
    emit_kv("config.n", (long long)n);
    emit_kv("config.blocks", (long long)blocks);
    emit_kv("config.threads", (long long)threads);
    emit_kv("config.smem", (long long)smem);

    w->setup(n);
    g_active = w;
    g_blocks = blocks;
    g_threads = threads;
    g_smem = smem;
    g_iters = iters;

    if (do_verify) {
        // Correctness is checked with a *single* fresh launch, and the timing
        // loop is skipped entirely so the toolkit's correctness gate stays
        // cheap.  Verifying after the timed loop would be wrong for in-place
        // kernels (y = a*x + y), which would have accumulated N applications.
        cudaError_t le = w->run(blocks, threads, smem, 1);
        cudaError_t se = cudaDeviceSynchronize();
        if (le != cudaSuccess || se != cudaSuccess) {
            emit_kv("verify.status", "fail");
            emit_kv("verify.detail",
                    std::string("launch error: ") +
                        cudaGetErrorString(le != cudaSuccess ? le : se));
            w->teardown();
            return 3;
        }
        std::string detail;
        int rc = w->verify(n, detail);
        emit_kv("verify.status", rc == 0 ? "pass" : "fail");
        emit_kv("verify.detail", detail);
        w->teardown();
        return rc == 0 ? 0 : 3;
    }

    TimingStats st = time_launches(&launch_active, warmup, iters);
    emit_kv("timing.median_us", st.median_us, 4);
    emit_kv("timing.p10_us", st.p10_us, 4);
    emit_kv("timing.p90_us", st.p90_us, 4);
    emit_kv("timing.min_us", st.min_us, 4);
    emit_kv("timing.max_us", st.max_us, 4);
    emit_kv("timing.cv", st.cv, 6);
    emit_kv("timing.samples", (long long)st.samples);

    // Derived throughput, using the workload's own accounting.
    const double bytes = w->bytes_per_launch ? w->bytes_per_launch(n) : -1.0;
    const double flops = w->flops_per_launch ? w->flops_per_launch(n) : -1.0;
    // Raw counts are emitted alongside the rates: the toolkit's roofline model
    // needs the absolute traffic and operation counts to compute arithmetic
    // intensity, and re-deriving them from a rounded rate would lose precision.
    if (bytes > 0.0) {
        emit_kv("derived.bytes_moved", (long long)bytes);
    }
    if (flops > 0.0) {
        // For per-thread FLOP kernels the launch width matters.
        double total = flops;
        if (flops < 1000.0) total = flops * (double)blocks * (double)threads;
        emit_kv("derived.flops", (long long)total);
    }
    if (bytes > 0.0 && st.median_us > 0.0) {
        emit_kv("derived.effective_bandwidth_gbps", bytes / (st.median_us * 1e-6) / 1e9, 2);
    }
    if (flops > 0.0 && st.median_us > 0.0) {
        // For per-thread FLOP kernels the launch width matters.
        double total = flops;
        if (flops < 1000.0) total = flops * (double)blocks * (double)threads;
        emit_kv("derived.tflops", total / (st.median_us * 1e-6) / 1e12, 4);
    }

    w->teardown();
    return 0;
}
