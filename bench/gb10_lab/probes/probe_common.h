// probe_common.h — tiny standalone timing helper for feature probes.
// Deliberately independent of gb10_lab/common.h so each probe stays an isolated
// translation unit (a compile failure in one probe cannot poison another). It
// gives every probe a REAL performance number (median us + sustained ops/s), so
// the feature matrix shows not just "supported" but "how fast this instruction
// family runs on this device."
#pragma once
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>

inline double probe_percentile(std::vector<double> v, double pct) {
    std::sort(v.begin(), v.end());
    const double rank = pct/100.0*(v.size()-1.0);
    long long lo=(long long)std::floor(rank), hi=(long long)std::ceil(rank);
    if (lo==hi) return v[lo];
    double f=rank-std::floor(rank);
    return v[lo]*(1.0-f)+v[hi]*f;
}

// Time `body` (one launch) over `warmup`+`reps` launches with CUDA events; print
// a machine-readable line carrying median_us and sustained ops_per_s given a
// fixed operation count per launch. Ops counted by the documented 2K convention
// where applicable; the caller supplies the exact per-launch op count.
template <class Body>
void probe_time_and_report(const char* name, Body&& body, double ops_per_launch,
                           int warmup=20, int reps=100) {
    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
    for (int i=0;i<warmup;++i) body();
    cudaDeviceSynchronize();
    std::vector<double> us(reps);
    for (int i=0;i<reps;++i) {
        cudaEventRecord(s); body(); cudaEventRecord(e); cudaEventSynchronize(e);
        float ms=0.f; cudaEventElapsedTime(&ms,s,e); us[i]=(double)ms*1000.0;
    }
    cudaEventDestroy(s); cudaEventDestroy(e);
    const double med = probe_percentile(us,50.0);
    const double p10 = probe_percentile(us,10.0);
    const double p90 = probe_percentile(us,90.0);
    const double ops_s = ops_per_launch/(med*1e-6);
    std::printf("%s median_us=%.4f p10_us=%.4f p90_us=%.4f ops_per_s=%.0f TOPS=%.4f\n",
                name, med, p10, p90, ops_s, ops_s/1e12);
}

// Same as above but reports effective bandwidth (GB/s) from documented bytes moved
// per launch instead of an operation count. Used by memory-bound probes.
template <class Body>
void probe_time_bw(const char* name, Body&& body, double bytes_per_launch,
                   int warmup=20, int reps=100) {
    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
    for (int i=0;i<warmup;++i) body();
    cudaDeviceSynchronize();
    std::vector<double> us(reps);
    for (int i=0;i<reps;++i) {
        cudaEventRecord(s); body(); cudaEventRecord(e); cudaEventSynchronize(e);
        float ms=0.f; cudaEventElapsedTime(&ms,s,e); us[i]=(double)ms*1000.0;
    }
    cudaEventDestroy(s); cudaEventDestroy(e);
    const double med = probe_percentile(us,50.0);
    const double p10 = probe_percentile(us,10.0);
    const double p90 = probe_percentile(us,90.0);
    const double gb_s = bytes_per_launch/(med*1e-6)/1e9;
    std::printf("%s median_us=%.4f p10_us=%.4f p90_us=%.4f GB_per_s=%.1f\n",
                name, med, p10, p90, gb_s);
}
