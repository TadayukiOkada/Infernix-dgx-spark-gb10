#include "models/qwen4_exp/frontend/ngram_hash.h"

#include <algorithm>
#include <condition_variable>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace infernix::models::qwen4_exp {
namespace {

constexpr std::uint64_t kGamma            = 0x9E3779B97F4A7C15ULL;
constexpr std::uint64_t kPrime1           = 10007;
constexpr std::size_t kParallelRowTokens  = 512;
constexpr std::size_t kParallelHashThreads = 8;

std::uint64_t splitmix64(std::uint64_t x) {
    x += kGamma;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

bool is_prime(std::uint64_t v) {
    if (v < 2) { return false; }
    if (v % 2 == 0) { return v == 2; }
    for (std::uint64_t d = 3; d * d <= v; d += 2) {
        if (v % d == 0) { return false; }
    }
    return true;
}

} // namespace

struct NgramHash::HashPool {
    std::mutex mutex;
    std::vector<std::unique_ptr<std::condition_variable>> worker_cv;
    std::condition_variable done_cv;
    std::vector<std::thread> workers;
    std::vector<std::exception_ptr> errors;
    const std::function<void(std::size_t, std::size_t)>* task = nullptr;
    std::size_t active  = 0;
    std::size_t pending = 0;
    std::uint64_t epoch = 0;
    bool stop           = false;

    explicit HashPool(std::size_t total_threads) : errors(total_threads) {
        if (total_threads <= 1) { return; }
        workers.reserve(total_threads - 1);
        worker_cv.reserve(total_threads - 1);
        for (std::size_t t = 1; t < total_threads; ++t) {
            worker_cv.push_back(std::make_unique<std::condition_variable>());
        }
        for (std::size_t t = 1; t < total_threads; ++t) {
            workers.emplace_back([this, t] {
                std::uint64_t seen = 0;
                for (;;) {
                    const std::function<void(std::size_t, std::size_t)>* fn = nullptr;
                    std::size_t n = 0;
                    {
                        std::unique_lock<std::mutex> lock(mutex);
                        worker_cv[t - 1]->wait(lock, [&] { return stop || epoch != seen; });
                        if (stop) { return; }
                        seen = epoch;
                        if (t < active) {
                            fn = task;
                            n  = active;
                        }
                    }
                    if (fn != nullptr) {
                        try {
                            (*fn)(t, n);
                        } catch (...) { errors[t] = std::current_exception(); }
                        std::lock_guard<std::mutex> lock(mutex);
                        if (--pending == 0) { done_cv.notify_one(); }
                    }
                }
            });
        }
    }

    ~HashPool() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stop = true;
            for (auto& cv : worker_cv) { cv->notify_one(); }
        }
        for (auto& w : workers) {
            if (w.joinable()) { w.join(); }
        }
    }

    void run(std::size_t threads, const std::function<void(std::size_t, std::size_t)>& fn) {
        if (threads <= 1 || workers.empty()) {
            fn(0, 1);
            return;
        }
        for (std::size_t t = 0; t < threads; ++t) { errors[t] = nullptr; }
        {
            std::lock_guard<std::mutex> lock(mutex);
            task    = &fn;
            active  = threads;
            pending = threads - 1;
            ++epoch;
            for (std::size_t t = 1; t < threads; ++t) {
                worker_cv[t - 1]->notify_one();
            }
        }
        try {
            fn(0, threads);
        } catch (...) { errors[0] = std::current_exception(); }
        {
            std::unique_lock<std::mutex> lock(mutex);
            done_cv.wait(lock, [&] { return pending == 0; });
            task = nullptr;
        }
        for (std::size_t t = 0; t < threads; ++t) {
            if (errors[t]) { std::rethrow_exception(errors[t]); }
        }
    }
};

NgramHash::~NgramHash()                                = default;
NgramHash::NgramHash(NgramHash&&) noexcept            = default;
NgramHash& NgramHash::operator=(NgramHash&&) noexcept = default;

NgramHash::NgramHash(const NgramHash& other)
    : config_(other.config_),
      multipliers_(other.multipliers_),
      sizes_(other.sizes_),
      offsets_(other.offsets_),
      padded_rows_(other.padded_rows_),
      pool_(nullptr) {}

NgramHash& NgramHash::operator=(const NgramHash& other) {
    if (this != &other) {
        config_      = other.config_;
        multipliers_ = other.multipliers_;
        sizes_       = other.sizes_;
        offsets_     = other.offsets_;
        padded_rows_ = other.padded_rows_;
        pool_.reset();
    }
    return *this;
}

NgramHash::NgramHash(const NgramConfig& config, std::uint32_t layer) : config_(config) {
    if (config.ngram_size < 2 || config.heads_per_ngram == 0 || config.vocab_size == 0 ||
        config.ngram_vocab_size_base == 0 || config.make_ngram_vocab_size_divisible_by == 0) {
        throw std::invalid_argument("qwen4_exp n-gram config is incomplete");
    }
    const std::uint64_t max_long   = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
    const std::uint64_t half_bound = std::max<std::uint64_t>(1, max_long / config.vocab_size / 2);
    const std::uint64_t base       = config.seed + kPrime1 * layer; // wraps like the 64-bit reference
    for (std::uint32_t i = 0; i < config.ngram_size; ++i) {
        multipliers_.push_back(2 * (splitmix64(base + kGamma * (i + 1)) % half_bound) + 1);
    }
    const std::uint64_t heads = static_cast<std::uint64_t>(config.ngram_size - 1) * config.heads_per_ngram;
    std::uint64_t prime = config.ngram_vocab_size_base - 1, total = 0;
    for (std::uint64_t i = 0; i < (static_cast<std::uint64_t>(layer) + 1) * heads; ++i) {
        do { ++prime; } while (!is_prime(prime));
        if (i >= static_cast<std::uint64_t>(layer) * heads) {
            sizes_.push_back(prime);
            offsets_.push_back(total);
            total += prime;
        }
    }
    const std::uint64_t d = config.make_ngram_vocab_size_divisible_by;
    padded_rows_          = (total + d - 1) / d * d;
    if (padded_rows_ > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("qwen4_exp n-gram table exceeds u32 row ids");
    }
}

void NgramHash::row_ids(std::span<const std::int32_t> history, std::size_t count, std::uint32_t* out) const {
    const std::size_t n_ctx = config_.ngram_size;
    if (history.size() < count + n_ctx - 1) { throw std::invalid_argument("n-gram history too short"); }
    if (n_ctx > 32) { throw std::invalid_argument("ngram_size above 32"); }
    const std::size_t start_pos  = history.size() - count;
    const std::size_t head_count = sizes_.size();
    const auto compute_range     = [&](std::size_t lo, std::size_t hi) {
        std::uint64_t context[32];
        std::uint32_t* dst = out + lo * head_count;
        for (std::size_t idx = lo; idx < hi; ++idx) {
            const std::size_t p = start_pos + idx;
            bool closed         = false; // an EOS among history[p - s .. p - 1]
            for (std::size_t s = 0; s < n_ctx; ++s) {
                if (s > 0 && history[p - s] == config_.eos_token_id) { closed = true; }
                const std::int32_t token = closed ? config_.eos_token_id : history[p - s];
                if (token < 0 || static_cast<std::uint32_t>(token) >= config_.vocab_size) {
                    throw std::invalid_argument("n-gram token outside the vocabulary");
                }
                context[s] = static_cast<std::uint64_t>(token);
            }
            std::uint64_t mixed = context[0] * multipliers_[0];
            for (std::size_t n = 2; n <= n_ctx; ++n) {
                mixed ^= context[n - 1] * multipliers_[n - 1];
                const std::size_t base = (n - 2) * config_.heads_per_ngram;
                for (std::size_t j = base; j < base + config_.heads_per_ngram; ++j) {
                    *dst++ = static_cast<std::uint32_t>(mixed % sizes_[j] + offsets_[j]);
                }
            }
        }
    };
    if (count < kParallelRowTokens) {
        compute_range(0, count);
        return;
    }
    if (!pool_) {
        pool_ = std::make_unique<HashPool>(kParallelHashThreads);
    }
    pool_->run(kParallelHashThreads, [&](std::size_t t, std::size_t n_threads) {
        const std::size_t lo = count * t / n_threads;
        const std::size_t hi = count * (t + 1) / n_threads;
        compute_range(lo, hi);
    });
}

} // namespace infernix::models::qwen4_exp
