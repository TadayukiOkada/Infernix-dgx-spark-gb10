// harness.cpp — registry, timing and reporting implementation.
#include "harness.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

std::vector<Workload>& workload_registry() {
    static std::vector<Workload> reg;
    return reg;
}

static double pct(std::vector<double>& v, double p) {
    if (v.empty()) return 0.0;
    if (v.size() == 1) return v[0];
    const double rank = p / 100.0 * (v.size() - 1);
    const size_t lo = static_cast<size_t>(rank);
    const size_t hi = std::min(lo + 1, v.size() - 1);
    const double f = rank - lo;
    return v[lo] * (1.0 - f) + v[hi] * f;
}

TimingStats time_launches(int (*launch)(), int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) launch();
    CKT_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> us;
    us.reserve(iters);
    cudaEvent_t s, e;
    CKT_CUDA_CHECK(cudaEventCreate(&s));
    CKT_CUDA_CHECK(cudaEventCreate(&e));
    for (int i = 0; i < iters; ++i) {
        CKT_CUDA_CHECK(cudaEventRecord(s));
        launch();
        CKT_CUDA_CHECK(cudaEventRecord(e));
        CKT_CUDA_CHECK(cudaEventSynchronize(e));
        float ms = 0.f;
        CKT_CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
        us.push_back(static_cast<double>(ms) * 1000.0);
    }
    CKT_CUDA_CHECK(cudaEventDestroy(s));
    CKT_CUDA_CHECK(cudaEventDestroy(e));

    std::sort(us.begin(), us.end());
    TimingStats st;
    st.samples = static_cast<int>(us.size());
    st.median_us = pct(us, 50);
    st.p10_us = pct(us, 10);
    st.p90_us = pct(us, 90);
    st.min_us = us.front();
    st.max_us = us.back();
    double sum = 0.0;
    for (double v : us) sum += v;
    const double mean = sum / us.size();
    double var = 0.0;
    for (double v : us) var += (v - mean) * (v - mean);
    var = us.size() > 1 ? var / (us.size() - 1) : 0.0;
    st.cv = mean > 0.0 ? std::sqrt(var) / mean : 0.0;
    return st;
}

void emit_kv(const std::string& key, const std::string& value) {
    std::printf("%s\t%s\n", key.c_str(), value.c_str());
}
void emit_kv(const std::string& key, long long value) {
    std::printf("%s\t%lld\n", key.c_str(), value);
}
void emit_kv(const std::string& key, double value, int precision) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*f", precision, value);
    std::printf("%s\t%s\n", key.c_str(), buf);
}
