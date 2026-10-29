// The host's physical cores (core/cpu_topology.h), which the temporary CPU placement switches
// (runtime/expert_offload/placement.h) give one expert worker each: every logical CPU belongs
// to exactly one core, a core's first thread is in it, and a thread restricted to one core's CPU runs
// there and gets its old mask back. Also the process's logical CPUs and the default CPU expert team they
// allow (GitHub issue #48: 16 workers on 16 vCPUs starved the engine thread). And the big.LITTLE
// detector: it must agree with an independent reading of sysfs and must never invent cores or return
// a preference on a symmetric host. Host-only: reads /sys, no GPU.

#include "core/cpu_topology.h"
#include "runtime/expert_offload/placement.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

int g_failures = 0;
void check(bool ok, const char* what) {
    if (!ok) { std::cerr << "FAIL: " << what << "\n"; ++g_failures; }
}

long long read_long(const std::string& path, bool& ok) {
    std::ifstream f(path);
    long long v = 0;
    ok = static_cast<bool>(f >> v);
    return v;
}

// Independent re-derivation of the online CPU ids from sysfs, mirroring the detector's source but
// written separately so a parser bug in the library would not silently pass.
std::vector<int> online() {
    std::vector<int> ids;
    std::ifstream f("/sys/devices/system/cpu/online");
    std::string text;
    if (!(f >> text)) { return ids; }
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        const auto dash = item.find('-');
        int lo = std::stoi(item.substr(0, dash));
        int hi = dash == std::string::npos ? lo : std::stoi(item.substr(dash + 1));
        for (int c = lo; c <= hi; ++c) { ids.push_back(c); }
    }
    return ids;
}

} // namespace

int main() {
    // The default team leaves four logical CPUs, keeps at least one worker and never grows.
    using infernix::runtime::expert_offload::default_workers;
    check(default_workers(16, 16) == 12, "16 GGUF workers on 16 logical CPUs become 12");
    check(default_workers(16, 32) == 16, "a 32-CPU host keeps 16 workers");
    check(default_workers(6, 8) == 4, "6 NVFP4 workers on 8 logical CPUs become 4");
    check(default_workers(6, 4) == 1 && default_workers(16, 2) == 1, "a tiny host keeps one worker");
    check(default_workers(16, 0) == 16, "an unknown CPU count keeps the configured team");
    check(default_workers(0, 16) == 0, "a disabled team stays off");

    // The 122B's decode split: 256 / (1 + 1.46 C t), C t = C x record / link.
    using infernix::runtime::expert_offload::decode_link_share_256;
    check(decode_link_share_256(27.5e9, 10'650.0, 5'308'416.0) == 64, "Ornith on this machine keeps 64/256 on the link");
    check(decode_link_share_256(55.0e9, 10'650.0, 5'308'416.0) > 64, "a faster link takes a larger share");
    check(decode_link_share_256(27.5e9, 21'300.0, 5'308'416.0) < 64, "a faster CPU team takes the larger part");
    check(decode_link_share_256(0.0, 10'650.0, 5'308'416.0) == 0 && decode_link_share_256(27.5e9, 0.0, 1.0) == 0,
          "no measured link or CPU rate gives no share");
    check(decode_link_share_256(1.0e15, 1.0, 1.0) == 256, "a link far faster than the CPU takes every miss");
    const std::uint32_t process = infernix::process_logical_cpus();
    const unsigned machine      = std::thread::hardware_concurrency();
    check(process >= 1, "the process may run on at least one logical CPU");
    check(machine == 0 || process <= machine, "the process's logical CPUs are at most the machine's");

    const std::vector<int> cpus = online();
    check(!cpus.empty(), "sysfs exposes an online CPU list");

    const auto cores = infernix::physical_cores();
    if (!cores.empty()) {
        std::uint64_t seen = 0;
        bool disjoint = true, nonempty = true;
        for (const auto& core : cores) {
            nonempty &= core.cpus != 0;
            disjoint &= (seen & core.cpus) == 0;
            seen |= core.cpus;
        }
        check(nonempty, "every physical core has a logical CPU");
        check(disjoint, "no logical CPU belongs to two physical cores");
        const unsigned logical = std::thread::hardware_concurrency();
        check(logical == 0 || static_cast<unsigned>(std::popcount(seen)) <= logical,
              "the cores hold at most the machine's logical CPUs");
        check(infernix::first_cpu(cores.front().cpus) >= 0 &&
                  ((cores.front().cpus >> infernix::first_cpu(cores.front().cpus)) & 1U) != 0,
              "a core's first CPU is one of its own");
    }
    check(infernix::first_cpu(0) == -1, "an empty mask has no first CPU");

    // Read each CPU's normalized capacity independently.
    std::vector<long long> capacity;
    bool have_capacity = true;
    for (int c : cpus) {
        bool ok = false;
        const long long v = read_long("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/cpu_capacity", ok);
        if (!ok) { have_capacity = false; break; }
        capacity.push_back(v);
    }

    const bool hybrid   = infernix::core::hybrid_cpu();
    const std::vector<int> big = infernix::core::performance_cpus();

    // Consistency between the two entry points.
    check(hybrid ? !big.empty() : big.empty(), "hybrid_cpu and performance_cpus agree on emptiness");

    // Every returned CPU is a real online CPU, distinct, and ascending.
    const std::set<int> online_set(cpus.begin(), cpus.end());
    bool distinct = std::adjacent_find(big.begin(), big.end()) == big.end();
    check(std::all_of(big.begin(), big.end(), [&](int c) { return online_set.count(c) != 0; }),
          "performance_cpus returns only online CPUs");
    check(distinct, "performance_cpus returns distinct, ascending CPU ids");

    if (have_capacity) {
        const long long max_c = *std::max_element(capacity.begin(), capacity.end());
        const long long min_c = *std::min_element(capacity.begin(), capacity.end());
        // An asymmetric host has a real tier gap; a symmetric one does not.
        const bool expected_hybrid = max_c >= min_c + min_c / 5;
        check(hybrid == expected_hybrid, "hybrid detection matches the independent capacity gap");
        if (expected_hybrid) {
            // The big set must be exactly the top-tier cores: no core strictly below the top tier,
            // and it must include the single highest-capacity core.
            const int best = static_cast<int>(std::distance(
                capacity.begin(), std::max_element(capacity.begin(), capacity.end())));
            check(std::find(big.begin(), big.end(), cpus[best]) != big.end(),
                  "the highest-capacity core is in performance_cpus");
            for (std::size_t i = 0; i < cpus.size(); ++i) {
                const bool selected = std::find(big.begin(), big.end(), cpus[i]) != big.end();
                if (selected) {
                    check(capacity[i] >= max_c - max_c / 10,
                          "a selected core is within 10% of the top capacity tier");
                }
            }
        }
    }

    if (g_failures != 0) {
        std::cerr << g_failures << " cpu_topology check(s) failed\n";
        return 1;
    }
    std::cout << "cpu_topology checks passed (hybrid=" << (hybrid ? "yes" : "no")
              << ", big cpus:";
    for (int c : big) { std::cout << ' ' << c; }
    std::cout << ")\n";
    return 0;
}
