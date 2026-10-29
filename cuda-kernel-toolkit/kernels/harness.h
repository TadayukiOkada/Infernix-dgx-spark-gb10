// harness.h — shared contract for the toolkit's sample kernel suite.
//
// The toolkit needs a target executable whose kernels can be
//   (a) discovered in source,
//   (b) launched with externally supplied launch parameters,
//   (c) validated against an independent reference,
//   (d) timed with a stable, low-overhead protocol.
//
// This header defines that contract.  Every sample kernel is registered with a
// fixed signature so the harness can iterate over them without per-kernel
// special-casing, and every kernel is defined in its own .cu file so the
// toolkit's source discovery has a realistic multi-file target.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Kernel descriptor
// ---------------------------------------------------------------------------

// One tunable workload. `run` allocates nothing: buffers are created once by
// the harness so a launch-parameter sweep measures only the kernel.
struct Workload {
    const char* name;
    // Launch the kernel with the given config. Returns cudaError_t.
    cudaError_t (*run)(int blocks, int threads, int shared_mem, int iterations);
    // Bytes moved per launch (read + write), for effective-bandwidth reporting.
    // -1 means "not applicable / not memory bound".
    double (*bytes_per_launch)(int problem_n);
    // Floating-point (and equivalent) operations per launch, for TFLOP/s.
    double (*flops_per_launch)(int problem_n);
    // Configure device buffers for problem size n. Called once per size.
    void (*setup)(int problem_n);
    // Free buffers.
    void (*teardown)();
    // Validate the last launch against a CPU reference. Returns 0 on success,
    // non-zero on mismatch, and fills `detail` with a human-readable reason.
    int (*verify)(int problem_n, std::string& detail);
    // Default problem size for this workload.
    void (*defaults)(int& problem_n, int& blocks, int& threads, int& shared_mem);
};

// Registry populated by each kernels/*.cu translation unit.
std::vector<Workload>& workload_registry();

// Registration helper used at namespace scope via REGISTER_WORKLOAD.
struct WorkloadRegistrar {
    explicit WorkloadRegistrar(Workload w) { workload_registry().push_back(w); }
};
#define REGISTER_WORKLOAD(...) \
    static WorkloadRegistrar _ckt_reg_##__LINE__(__VA_ARGS__)

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

struct TimingStats {
    double median_us = 0.0;
    double p10_us = 0.0;
    double p90_us = 0.0;
    double min_us = 0.0;
    double max_us = 0.0;
    double cv = 0.0;              // relative dispersion (stdev / mean)
    int samples = 0;
};

// Times `iters` sequential launches with CUDA events, discarding `warmup`.
// The timing boundary excludes host-side allocation.
TimingStats time_launches(int (*launch)(), int warmup, int iters);

// ---------------------------------------------------------------------------
// Reporting helpers (line-oriented key<TAB>value + optional JSON)
// ---------------------------------------------------------------------------

void emit_kv(const std::string& key, const std::string& value);
void emit_kv(const std::string& key, long long value);
void emit_kv(const std::string& key, double value, int precision = 6);

// A convenience CUDA error check that reports and exits.
#define CKT_CUDA_CHECK(call)                                                    \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
                         cudaGetErrorString(_e));                               \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)
