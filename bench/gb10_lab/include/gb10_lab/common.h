#pragma once
//
// common.h — shared harness for the GB10 kernel laboratory.
//
// Design rules (see docs/performance/gb10-lab-project.md):
//   * Timing uses CUDA events on the device timeline. Host wall-clock is only
//     used for launch-overhead measurements where the host IS the subject.
//   * Every measurement reports median / p10 / p90 over repeated iterations;
//     a single sample is never trusted.
//   * Derived rates (GB/s, TFLOP/s) always come from documented byte/FLOP
//     counts stored next to the number in the JSONL record.
//   * Correctness is validated against an independent reference before any
//     timing is accepted; failures are recorded, not hidden.
//   * The lab is deliberately independent of the infernix_ops library: its own
//     kernels, its own allocations, no product allocators or dispatch logic.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#define LAB_CHECK(x)                                                        \
    do {                                                                    \
        cudaError_t e_ = (x);                                               \
        if (e_ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__,        \
                         __LINE__, cudaGetErrorString(e_));                 \
            std::exit(1);                                                   \
        }                                                                   \
    } while (0)

// NOTE: deliberately kept in the global namespace. Every op translation unit
// includes this header first and then pulls in its own system headers; wrapping
// the declarations in a namespace would force those subsequent #includes into
// the namespace as well. All lab symbols are used unqualified everywhere.

// ---------------------------------------------------------------------------
// Real model shape presets. Dimensions are taken from the Qwen3-family models
// this repository serves (src/models/qwen3_5, src/ops/* constants), so every
// sweep exercises shapes the production kernels actually see.
// ---------------------------------------------------------------------------
struct ModelPreset {
    const char* name;
    int hidden      = 0;  // H
    int n_heads     = 0;  // attention query heads
    int n_kv_heads  = 0;  // grouped-query KV heads
    int head_dim    = 0;  // d
    int inter       = 0;  // MLP intermediate size
    int vocab       = 0;
    int layers      = 0;
};

inline constexpr ModelPreset kPresets[] = {
    // Dense 32B-class: matches kHidden=5120 across src/ops/{embedding,attn_input_proj,...}.
    {"qwen3_dense_h5120", 5120, 64, 8, 128, 25600, 151936, 64},
    // MoE A3B-class: matches kHidden=2048 in src/ops/{sparse_moe,resident_moe,...}.
    // `inter` is the per-expert intermediate size used for GEMV sweeps.
    {"qwen3_moe_h2048", 2048, 32, 4, 128, 768, 151936, 48},
};

inline const ModelPreset& preset_by_name(const char* name) {
    for (const auto& p : kPresets)
        if (std::strcmp(p.name, name) == 0) return p;
    std::fprintf(stderr, "unknown preset '%s' (available: qwen3_dense_h5120, qwen3_moe_h2048)\n",
                 name);
    std::exit(1);
}

struct LabArgs {
    const ModelPreset& preset;
    int warmup       = 3;
    int iters        = 20;
    bool quick       = false;   // shrink sizes/iterations for smoke runs
    std::string out_path;       // JSONL output; empty = stdout tables only
};

// When >0, timed_iterations switches to a saturating measurement: it enqueues
// this many back-to-back launches asynchronously (single trailing sync) instead
// of syncing after every launch. This keeps the GPU continuously fed so power
// and utilization rise toward a real workload's steady state. Median/p10/p90
// collapse to the continuous-load mean in this mode. Set once from --sustained.
inline int g_sustained = 0;

// ---------------------------------------------------------------------------
// Device fingerprint (queried once).
// ---------------------------------------------------------------------------
struct DeviceInfo {
    std::string name;
    int cc_major = 0, cc_minor = 0;
    int sm_count = 0;
    std::size_t l2_bytes = 0;
    std::size_t shared_per_sm = 0;   // opt-in maximum per block
    std::size_t global_bytes = 0;
    std::size_t free_bytes = 0;
    std::string driver_version;
    std::string runtime_version;

    static const DeviceInfo& get() {
        static DeviceInfo info = [] {
            DeviceInfo d;
            cudaDeviceProp p{};
            LAB_CHECK(cudaGetDeviceProperties(&p, 0));
            d.name          = p.name;
            d.cc_major      = p.major;
            d.cc_minor      = p.minor;
            d.sm_count      = p.multiProcessorCount;
            d.l2_bytes      = static_cast<std::size_t>(p.l2CacheSize);
            d.shared_per_sm = p.sharedMemPerMultiprocessor;
            d.global_bytes  = p.totalGlobalMem;
            size_t free_ = 0, total_ = 0;
            cudaMemGetInfo(&free_, &total_);
            d.free_bytes = free_;
            int rv = 0, dv = 0;
            cudaRuntimeGetVersion(&rv);
            cudaDriverGetVersion(&dv);
            char buf[32];
            std::snprintf(buf, sizeof buf, "%d", rv);
            d.runtime_version = buf;
            std::snprintf(buf, sizeof buf, "%d", dv);
            d.driver_version = buf;
            return d;
        }();
        return info;
    }
};

// ---------------------------------------------------------------------------
// Timing.
// ---------------------------------------------------------------------------
struct TimingResult {
    double median_us = 0.0;
    double p10_us    = 0.0;
    double p90_us    = 0.0;
    double min_us    = 0.0;
    double max_us    = 0.0;
    int reps         = 0;
};

inline double percentile(std::vector<double> v, double pct) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const double rank = pct / 100.0 * (static_cast<double>(v.size()) - 1.0);
    const long long lo_i = static_cast<long long>(std::floor(rank));
    const long long hi_i = static_cast<long long>(std::ceil(rank));
    if (lo_i == hi_i) return v[lo_i];
    const double frac = rank - std::floor(rank);
    return v[lo_i] * (1.0 - frac) + v[hi_i] * frac;
}

// Time one launch of `body` on `stream`, repeated `iters` times with `warmup`
// untimed repetitions first. Returns per-iteration microseconds.
template <class Body>
std::vector<double> timed_iterations(Body&& body, int warmup, int iters,
                                     cudaStream_t stream = nullptr) {
    cudaEvent_t s, e;
    LAB_CHECK(cudaEventCreate(&s));
    LAB_CHECK(cudaEventCreate(&e));
    for (int i = 0; i < warmup; ++i) body(stream);
    LAB_CHECK(cudaStreamSynchronize(stream));

    if (g_sustained > 0) {
        // Saturating mode: fire a long async burst, single trailing sync, so the
        // device stays continuously busy (higher power/util). Per-launch cost is
        // the burst total divided by the launch count.
        const int burst = g_sustained * iters;
        LAB_CHECK(cudaEventRecord(s, stream));
        for (int i = 0; i < burst; ++i) body(stream);
        LAB_CHECK(cudaEventRecord(e, stream));
        LAB_CHECK(cudaEventSynchronize(e));
        float ms = 0.f;
        LAB_CHECK(cudaEventElapsedTime(&ms, s, e));
        const double per_us = static_cast<double>(ms) * 1000.0 / burst;
        LAB_CHECK(cudaEventDestroy(s));
        LAB_CHECK(cudaEventDestroy(e));
        return std::vector<double>(iters, per_us);
    }

    std::vector<double> us(iters);
    for (int i = 0; i < iters; ++i) {
        LAB_CHECK(cudaEventRecord(s, stream));
        body(stream);
        LAB_CHECK(cudaEventRecord(e, stream));
        LAB_CHECK(cudaEventSynchronize(e));
        float ms = 0.f;
        LAB_CHECK(cudaEventElapsedTime(&ms, s, e));
        us[i] = static_cast<double>(ms) * 1000.0;
    }
    LAB_CHECK(cudaEventDestroy(s));
    LAB_CHECK(cudaEventDestroy(e));
    return us;
}

inline TimingResult summarize(const std::vector<double>& us) {
    TimingResult r;
    r.median_us = percentile(us, 50.0);
    r.p10_us    = percentile(us, 10.0);
    r.p90_us    = percentile(us, 90.0);
    r.min_us    = *std::min_element(us.begin(), us.end());
    r.max_us    = *std::max_element(us.begin(), us.end());
    r.reps      = static_cast<int>(us.size());
    return r;
}

// CUDA Graph replay timing: captures `body` once, then replays the graph
// `reps` times. For short kernels the host launch path can dominate measured
// latency, so graph numbers are reported alongside eager ones.
template <class Body>
TimingResult graph_replay_timing(Body&& body, int warmup, int reps) {
    cudaStream_t stream;
    LAB_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    LAB_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    body(stream);
    LAB_CHECK(cudaStreamEndCapture(stream, &graph));
    LAB_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    for (int i = 0; i < warmup; ++i) LAB_CHECK(cudaGraphLaunch(exec, stream));
    LAB_CHECK(cudaStreamSynchronize(stream));
    cudaEvent_t s, e;
    LAB_CHECK(cudaEventCreate(&s));
    LAB_CHECK(cudaEventCreate(&e));
    std::vector<double> us(reps);
    for (int i = 0; i < reps; ++i) {
        LAB_CHECK(cudaEventRecord(s, stream));
        LAB_CHECK(cudaGraphLaunch(exec, stream));
        LAB_CHECK(cudaEventRecord(e, stream));
        LAB_CHECK(cudaEventSynchronize(e));
        float ms = 0.f;
        LAB_CHECK(cudaEventElapsedTime(&ms, s, e));
        us[i] = static_cast<double>(ms) * 1000.0;
    }
    LAB_CHECK(cudaEventDestroy(s));
    LAB_CHECK(cudaEventDestroy(e));
    LAB_CHECK(cudaGraphExecDestroy(exec));
    LAB_CHECK(cudaGraphDestroy(graph));
    LAB_CHECK(cudaStreamDestroy(stream));
    return summarize(us);
}

// ---------------------------------------------------------------------------
// bf16 vector helpers. uint4 <-> 8x bf16 with explicit memcpy so the compiler
// keeps the wide load/store.
// ---------------------------------------------------------------------------
struct alignas(16) BF16x8 { __nv_bfloat16 x[8]; };

// Marked for both host and device so kernels can use them directly. A plain
// definition may carry both qualifiers, and memcpy of a trivially-copyable
// struct lowers to an identity copy in each translation unit.
__host__ __device__ inline BF16x8 u4_to_bf16(uint4 v) {
    BF16x8 out;
    std::memcpy(&out, &v, sizeof(v));
    return out;
}
__host__ __device__ inline uint4 bf16_to_u4(const BF16x8& v) {
    uint4 out;
    std::memcpy(&out, &v, sizeof(v));
    return out;
}
__host__ __device__ inline float bf16_f(__nv_bfloat16 v) { return __bfloat162float(v); }
__host__ __device__ inline __nv_bfloat16 f_bf16(float v) { return __float2bfloat16_rn(v); }

// Deterministic small-range fill so correctness references match bit-for-bit
// in scope and runs are reproducible across processes.
inline void fill_bf16(std::vector<__nv_bfloat16>& h, std::uint64_t seed,
                      float scale = 0.5f) {
    std::uint64_t state = seed ? seed : 0x9E3779B97F4A7C15ull;
    for (auto& v : h) {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        const int bits = static_cast<int>((state >> 33) & 0xFFFFFF);
        v = f_bf16(static_cast<float>(bits) / static_cast<float>(0xFFFFFF) * 2.0f * scale - scale);
    }
}
inline void fill_f32(std::vector<float>& h, std::uint64_t seed, float scale = 0.5f) {
    std::uint64_t state = seed ? seed : 0x9E3779B97F4A7C15ull;
    for (auto& v : h) {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        const int bits = static_cast<int>((state >> 33) & 0xFFFFFF);
        v = static_cast<float>(bits) / static_cast<float>(0xFFFFFF) * 2.0f * scale - scale;
    }
}

// ---------------------------------------------------------------------------
// Correctness: elementwise max absolute error against a reference, with an
// allclose-style relative tolerance where magnitudes matter.
// ---------------------------------------------------------------------------
struct ErrStats {
    double max_abs_err = 0.0;
    double max_rel_err = 0.0;
    bool pass         = true;
};

inline ErrStats compare_f32(const float* got, const float* ref, std::size_t n,
                            double abs_tol = 1e-5, double rel_tol = 1e-4) {
    ErrStats s;
    for (std::size_t i = 0; i < n; ++i) {
        const double d = std::fabs(static_cast<double>(got[i]) - static_cast<double>(ref[i]));
        const double r = d / std::max(1e-30, std::fabs(static_cast<double>(ref[i])));
        s.max_abs_err = std::max(s.max_abs_err, d);
        s.max_rel_err = std::max(s.max_rel_err, r);
        if (!(d <= abs_tol || r <= rel_tol)) s.pass = false;
    }
    return s;
}

// ---------------------------------------------------------------------------
// JSONL output. One record per measurement with enough context to reproduce
// it (device fingerprint, op, shape, dtype, kernel parameters, derived rates).
// ---------------------------------------------------------------------------
inline std::string fmt_num(double v) {
    char buf[40];
    std::snprintf(buf, sizeof buf, "%.9g", v);
    return buf;
}

class JsonLine {
public:
    JsonLine() { s_ = "{"; }
    // Every key/value clause is prefixed with ',' unless this is the first one,
    // so chained .kv()/append_fragment() calls always yield valid JSON.
    JsonLine& kv(const char* k, const char* v) {
        if (s_.size() > 1) s_ += ",";
        s_ += std::string("\"") + k + "\":\"" + json_escape(v ? v : "") + "\"";
        return *this;
    }
    JsonLine& kv(const char* k, long long v) {
        char buf[32];
        std::snprintf(buf, sizeof buf, "%lld", v);
        return raw(k, buf);
    }
    JsonLine& kv(const char* k, int v) {   // disambiguate from the integral overloads
        char buf[32];
        std::snprintf(buf, sizeof buf, "%d", v);
        return raw(k, buf);
    }
    JsonLine& kv(const char* k, unsigned long long v) {
        char buf[32];
        std::snprintf(buf, sizeof buf, "%llu", v);
        return raw(k, buf);
    }
    JsonLine& kv(const char* k, double v) {
        char buf[40];
        std::snprintf(buf, sizeof buf, "%.9g", v);
        return raw(k, buf);
    }
    JsonLine& kv(const char* k, bool v) { return raw(k, v ? "true" : "false"); }
    // Pre-formed JSON fragment (numbers, arrays, nested objects). `fragment`
    // is a full ",\"k\":value" clause list appended verbatim.
    JsonLine& raw(const char* k, const std::string& v) {
        s_ += std::string(",\"") + k + "\":" + v;
        return *this;
    }
    JsonLine& append_fragment(const std::string& fragment) {
        s_ += fragment;
        return *this;
    }
    JsonLine& shapes(const std::vector<int>& sh) {
        std::string a = "[";
        for (std::size_t i = 0; i < sh.size(); ++i) {
            if (i) a += ",";
            char buf[16];
            std::snprintf(buf, sizeof buf, "%d", sh[i]);
            a += buf;
        }
        a += "]";
        return raw("shape", a);
    }
    std::string finish() { return s_ + "}"; }

private:
    static std::string json_escape(const std::string& in) {
        std::string out;
        for (char c : in) {
            switch (c) {
                case '"': out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                default: out += c;
            }
        }
        return out;
    }
    std::string s_;
};

// Global JSONL destination, set once in main(). Empty means stderr.
inline std::string& json_out_path() {
    static std::string p;
    return p;
}

// Write one finished record to the configured output file (or stderr).
inline void write_json_line(const std::string& rec) {
    if (!json_out_path().empty()) {
        std::ofstream f(json_out_path(), std::ios::app);
        f << rec << "\n";
    } else {
        std::fprintf(stderr, "%s\n", rec.c_str());
    }
}

// Append device/preset context to a caller-built JsonLine and emit it. This is
// the single place where fingerprint fields are attached, so every op record
// carries identical provenance without each op repeating the boilerplate.
inline void emit_record(const LabArgs& args, const JsonLine& line) {
    const DeviceInfo& d = DeviceInfo::get();
    JsonLine r = line;
    r.kv("device_name", d.name.c_str())
     .append_fragment(",\"cc\":\"" + std::to_string(d.cc_major) + "." + std::to_string(d.cc_minor) + "\"")
     .append_fragment(",\"sm_count\":" + fmt_num(static_cast<double>(d.sm_count)))
     .append_fragment(",\"l2_bytes\":" + fmt_num(static_cast<double>(d.l2_bytes)))
     .kv("preset", args.preset.name)
     .kv("warmup", args.warmup)
     .kv("iters", args.iters)
     .kv("quick", args.quick);
    write_json_line(r.finish());
}

// Emit one timing record. `params` is an optional pre-formed fragment of the
// form "\",key\":value,..." carrying kernel-specific launch parameters.
inline void emit_timing(const LabArgs& args, const char* op, const TimingResult& t,
                        const ErrStats& err, double bytes_moved, double flops,
                        const std::string& params = "") {
    JsonLine l;
    l.kv("op", op)
     .append_fragment(",\"median_us\":" + fmt_num(t.median_us))
     .append_fragment(",\"p10_us\":" + fmt_num(t.p10_us))
     .append_fragment(",\"p90_us\":" + fmt_num(t.p90_us))
     .append_fragment(",\"min_us\":" + fmt_num(t.min_us))
     .append_fragment(",\"max_us\":" + fmt_num(t.max_us))
     .append_fragment(",\"reps\":" + fmt_num(static_cast<double>(t.reps)))
     .append_fragment(std::string(",\"correct\":") + (err.pass ? "true" : "false"))
     .append_fragment(",\"max_abs_err\":" + fmt_num(err.max_abs_err))
     .append_fragment(",\"max_rel_err\":" + fmt_num(err.max_rel_err));
    if (bytes_moved > 0.0)
        l.append_fragment(",\"gb_per_s\":" + fmt_num(bytes_moved / (t.median_us * 1e-6) / 1e9));
    if (flops > 0.0)
        l.append_fragment(",\"tflop_per_s\":" + fmt_num(flops / (t.median_us * 1e-6) / 1e12));
    if (!params.empty()) l.append_fragment(params);
    emit_record(args, l);
}
