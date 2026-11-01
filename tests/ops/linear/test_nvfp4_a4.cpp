#include "ops/op_family.h"
#include "ops/linear/linear_test_common.h"

#include <vector>
#include <exception>
#include <iostream>

namespace {

using namespace infernix;
using namespace infernix::test::linear;

int run_nvfp4_a4() {
    std::vector<Invocation> invocations;
    for (int t :
         {1,   2,   3,   4,   5,   7,   8,   16,  17,  63,  64,   65,   95,   96,   97,   127, 128,
          129, 191, 192, 193, 383, 384, 385, 511, 512, 513, 1023, 1024, 1025, 1280, 1500, 2048}) {
        invocations.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA4});
    }
    for (int t : {63, 97, 193, 385, 1023, 1024})
        invocations.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA4, true});
    int failures = 0;
    failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight,
                          {14336, 5120, 719U, Comparison::Sampled, true, invocations});
    failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight,
                          {16384, 5120, 721U, Comparison::Sampled, true, invocations});
    // These calls regenerate A4 codes/scales on graph replay and cover complete and partial tiles.
    const std::vector<Invocation> n34816_full{
        {1, CallForm::Policy, ops::LinearPolicy::AllowA4, true},
        {4, CallForm::Policy, ops::LinearPolicy::AllowA4, true},
        {8, CallForm::Policy, ops::LinearPolicy::AllowA4, true},
        {32, CallForm::Policy, ops::LinearPolicy::AllowA4, true},
        {33, CallForm::Policy, ops::LinearPolicy::AllowA4, true},
        {64, CallForm::Policy, ops::LinearPolicy::AllowA4, true},
        {65, CallForm::Policy, ops::LinearPolicy::AllowA4, true},
        {128, CallForm::Policy, ops::LinearPolicy::AllowA4, true},
    };
    failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight,
                          {34816, 5120, 722U, Comparison::Full, true, n34816_full});
    auto n34816_invocations = invocations;
    for (int t : {31, 32, 33, 34, 255, 256, 257, 511, 512, 513, 767, 768, 769})
        n34816_invocations.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA4, true});
    failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight,
                          {34816, 5120, 722U, Comparison::Sampled, false, n34816_invocations});
    failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight,
                          {5120, 6144, 723U, Comparison::Sampled, true, invocations});
    failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight,
                          {5120, 17408, 725U, Comparison::Sampled, true, invocations});
    // Two-term row-scaled A4 TMA route (Nvfp4A4Pair) on Qwen3.8-Flash-Next W4Dense shapes. The
    // widths below 65 are the bands the 2026-11-01 retune admitted the pair to (gates at 33, 48
    // and 56), so they are covered against the oracle at the widths the shape table now selects
    // it at, not only where it ran before.
    const std::vector<Invocation> pair_invocations{
        {33, CallForm::Policy, ops::LinearPolicy::A16Only, false, ops::LinearRoutes::Nvfp4A4Pair},
        {34, CallForm::Policy, ops::LinearPolicy::A16Only, false, ops::LinearRoutes::Nvfp4A4Pair},
        {48, CallForm::Policy, ops::LinearPolicy::A16Only, false, ops::LinearRoutes::Nvfp4A4Pair},
        {56, CallForm::Policy, ops::LinearPolicy::A16Only, false, ops::LinearRoutes::Nvfp4A4Pair},
        {64, CallForm::Policy, ops::LinearPolicy::A16Only, false, ops::LinearRoutes::Nvfp4A4Pair},
        {65, CallForm::Policy, ops::LinearPolicy::A16Only, false, ops::LinearRoutes::Nvfp4A4Pair},
        {96, CallForm::Policy, ops::LinearPolicy::A16Only, false, ops::LinearRoutes::Nvfp4A4Pair},
        {127, CallForm::Policy, ops::LinearPolicy::A16Only, false, ops::LinearRoutes::Nvfp4A4Pair},
        {128, CallForm::Policy, ops::LinearPolicy::A16Only, false, ops::LinearRoutes::Nvfp4A4Pair},
        {129, CallForm::Policy, ops::LinearPolicy::A16Only, false, ops::LinearRoutes::Nvfp4A4Pair},
        {256, CallForm::Policy, ops::LinearPolicy::A16Only, false, ops::LinearRoutes::Nvfp4A4Pair},
        {512, CallForm::Policy, ops::LinearPolicy::A16Only, false, ops::LinearRoutes::Nvfp4A4Pair},
    };
    for (auto [n, k, seed] : {std::tuple{16384, 2560, 731U}, std::tuple{13952, 2560, 733U},
                              std::tuple{12800, 2560, 739U}, std::tuple{10240, 2560, 743U},
                              std::tuple{6144, 2560, 751U}, std::tuple{2560, 6144, 757U},
                              std::tuple{2560, 2560, 761U}, std::tuple{2048, 2560, 769U},
                              std::tuple{5120, 6144, 787U}, std::tuple{5120, 17408, 797U},
                              std::tuple{14336, 5120, 809U}, std::tuple{16384, 5120, 811U},
                              std::tuple{34816, 5120, 821U}}) {
        failures += run_shape("NVFP4_A4_PAIR", ActivationCompute::A4, make_nvfp4_weight,
                              {n, k, seed, Comparison::Sampled, true, pair_invocations});
    }
    // Single-term A4 on the dense shapes that register it (GB10, 2026-10-31). Their uses_a4 admits
    // the ladder from T = 33 (K = 2560) / T = 17 ([2560, 6144]), so the widths on both sides of
    // every route boundary are covered: below it the public route runs the A16 table.
    std::vector<Invocation> dense_a4_invocations;
    for (int t : {1,   16,  17,  32,  33,  34,  64,  128, 129, 192, 256, 257, 384, 512, 513, 1024,
                  2048}) {
        dense_a4_invocations.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA4});
    }
    for (int t : {33, 129, 257, 513, 1024}) {
        dense_a4_invocations.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA4, true});
    }
    for (auto [n, k, seed] :
         {std::tuple{16384, 2560, 823U}, std::tuple{13952, 2560, 827U},
          std::tuple{12800, 2560, 829U}, std::tuple{10240, 2560, 839U},
          std::tuple{6144, 2560, 853U}, std::tuple{2560, 6144, 857U},
          std::tuple{2560, 2560, 859U}, std::tuple{2048, 2560, 863U},
          std::tuple{512, 2560, 877U}, std::tuple{128, 2560, 881U}}) {
        failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight,
                              {n, k, seed, Comparison::Sampled, true, dense_a4_invocations});
    }
    return failures;
}

} // namespace

INFERNIX_OP_TEST(linear_nvfp4_a4) {
    if (!infernix::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        const int failures = run_nvfp4_a4();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " NVFP4_A4 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "NVFP4_A4 Linear: " << error.what() << '\n';
        return 1;
    }
}
