#include "ops/op_family.h"
#include "ops/linear/linear_test_common.h"

#include <utility>
#include <vector>
#include <exception>
#include <iostream>

namespace {

using namespace infernix;
using namespace infernix::test::linear;

int run_nvfp4_a16() {
    std::vector<Invocation> invocations;
    for (int t = 1; t <= 33; ++t) invocations.push_back({t});
    for (int t : {63, 64, 65, 95, 96, 97, 127, 128, 129, 512, 1024}) invocations.push_back({t});
    for (int t : {3, 7, 11, 15, 19, 23, 27, 31, 33})
        invocations.push_back({t, CallForm::Policy, ops::LinearPolicy::A16Only, true});
    invocations.push_back({1, CallForm::A16Convenience});
    invocations.push_back({33, CallForm::Policy, ops::LinearPolicy::AllowA8});
    int failures = 0;
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {14336, 5120, 701U, Comparison::Sampled, true, invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {16384, 5120, 703U, Comparison::Sampled, true, invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {34816, 5120, 704U, Comparison::Sampled, true, invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {5120, 6144, 705U, Comparison::Sampled, true, invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {5120, 17408, 707U, Comparison::Sampled, true, invocations});
    // Qwen3.8-Flash-Next dense Tier-1 shapes: exercise the decode ladder (GEMV t=1, SIMT t<=2,
    // sliced-K t<=8/16/32) that single-stream MTP actually hits, plus one wider MMA rung.
    std::vector<Invocation> decode_invocations;
    for (int t = 1; t <= 8; ++t) decode_invocations.push_back({t});
    for (int t : {16, 32}) decode_invocations.push_back({t});
    decode_invocations.push_back({1, CallForm::A16Convenience});
    decode_invocations.push_back({4, CallForm::Policy, ops::LinearPolicy::A16Only, true});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {16384, 2560, 711U, Comparison::Sampled, true, decode_invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {13952, 2560, 713U, Comparison::Sampled, true, decode_invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {12800, 2560, 719U, Comparison::Sampled, true, decode_invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {2560, 6144, 721U, Comparison::Sampled, true, decode_invocations});
    // Qwen3.8-Flash-Next W4-dense tier (recipe `qwen3_8_flash_next_nvfp4_w4dense`):
    // the dense packed-projection classes that move from q8_g32_fp16 to nvfp4.
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {2048, 2560, 723U, Comparison::Sampled, true, decode_invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {6144, 2560, 727U, Comparison::Sampled, true, decode_invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {512, 2560, 729U, Comparison::Sampled, true, decode_invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {128, 2560, 733U, Comparison::Sampled, true, decode_invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {10240, 2560, 739U, Comparison::Sampled, true, decode_invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {2560, 2560, 743U, Comparison::Sampled, true, decode_invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {1280, 2560, 751U, Comparison::Sampled, true, decode_invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {2560, 640, 757U, Comparison::Sampled, true, decode_invocations});
    const std::vector<Invocation> rht16_a16_invocations{
        {1, CallForm::Policy, ops::LinearPolicy::A16Only},
        {2, CallForm::Policy, ops::LinearPolicy::A16Only},
        {4, CallForm::Policy, ops::LinearPolicy::A16Only, true},
        {8, CallForm::Policy, ops::LinearPolicy::A16Only},
        {16, CallForm::Policy, ops::LinearPolicy::A16Only},
        {32, CallForm::Policy, ops::LinearPolicy::A16Only},
        {33, CallForm::Policy, ops::LinearPolicy::A16Only},
        {65, CallForm::Policy, ops::LinearPolicy::A16Only},
    };
    failures += run_shape("NVFP4_RHT16_A16", ActivationCompute::A16, make_nvfp4_rht16_weight,
                          {16384, 2560, 761U, Comparison::Sampled, true, rht16_a16_invocations});
    failures += run_shape("NVFP4_RHT16_A16", ActivationCompute::A16, make_nvfp4_rht16_weight,
                          {2560, 6144, 769U, Comparison::Sampled, true, rht16_a16_invocations});
    for (auto [n, k] : {std::pair{14336, 5120}, std::pair{16384, 5120}, std::pair{34816, 5120},
                        std::pair{5120, 6144}, std::pair{5120, 17408}, std::pair{16384, 2560},
                        std::pair{13952, 2560}, std::pair{12800, 2560}, std::pair{2560, 6144},
                        std::pair{2048, 2560}, std::pair{6144, 2560}, std::pair{512, 2560},
                        std::pair{128, 2560}, std::pair{10240, 2560}, std::pair{2560, 2560},
                        std::pair{1280, 2560}, std::pair{2560, 640}}) {
        failures += verify_workspace_envelopes(QType::NVFP4, n, k);
    }
    return failures;
}

} // namespace

INFERNIX_OP_TEST(linear_nvfp4_a16) {
    if (!infernix::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        const int failures = run_nvfp4_a16();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " NVFP4_A16 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "NVFP4_A16 Linear: " << error.what() << '\n';
        return 1;
    }
}
