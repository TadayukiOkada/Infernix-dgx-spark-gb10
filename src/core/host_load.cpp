#include "core/host_load.h"

#    include <sys/resource.h>
#    include <unistd.h>

#    include <cstdio>

#include <algorithm>

namespace infernix {
namespace {


} // namespace

HostCpuTimes host_cpu_times() noexcept {
    HostCpuTimes out;
    out.at = std::chrono::steady_clock::now();
    if (std::FILE* stat = std::fopen("/proc/stat", "r")) {
        unsigned long long user = 0, nice = 0, system = 0, idle = 0, iowait = 0, irq = 0, softirq = 0, steal = 0;
        if (std::fscanf(stat, "cpu %llu %llu %llu %llu %llu %llu %llu %llu", &user, &nice, &system, &idle, &iowait,
                        &irq, &softirq, &steal) >= 4) {
            const long hz = sysconf(_SC_CLK_TCK);
            const std::uint64_t ticks = user + nice + system + irq + softirq + steal;
            out.busy_ns = hz > 0 ? ticks * (1000000000ULL / static_cast<std::uint64_t>(hz)) : 0;
        }
        std::fclose(stat);
    }
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        const auto ns = [](const timeval& t) {
            return static_cast<std::uint64_t>(t.tv_sec) * 1000000000ULL + static_cast<std::uint64_t>(t.tv_usec) * 1000ULL;
        };
        out.own_ns = ns(usage.ru_utime) + ns(usage.ru_stime);
    }
    const long cpus = sysconf(_SC_NPROCESSORS_ONLN);
    out.cpus = cpus > 0 ? static_cast<std::uint32_t>(cpus) : 1U;
    return out;
}

double other_process_cpu_share(const HostCpuTimes& begin, const HostCpuTimes& end) noexcept {
    const double wall_ns = std::chrono::duration<double, std::nano>(end.at - begin.at).count();
    if (!(wall_ns > 0.0) || end.busy_ns < begin.busy_ns) { return 0.0; }
    const double busy = static_cast<double>(end.busy_ns - begin.busy_ns);
    const double own  = end.own_ns > begin.own_ns ? static_cast<double>(end.own_ns - begin.own_ns) : 0.0;
    const double share = (busy - own) / (wall_ns * static_cast<double>(std::max<std::uint32_t>(1, end.cpus)));
    return std::clamp(share, 0.0, 1.0);
}

} // namespace infernix
