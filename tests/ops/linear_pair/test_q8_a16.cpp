#include "ops/op_family.h"
#include "ops/linear_pair/linear_pair_test_common.h"

#include <array>
#include <exception>
#include <iostream>

namespace {

using infernix::test::linear_pair::ShapeCase;

int q8_a16_conformance() {
    int failures = 0;

    // Boundary lists track kK5120Routes/kK2048Routes in q8_pair_plan.cpp: every band edge
    // (start-1, start, start+1) plus representative interiors of each live band.
    constexpr std::array<std::int32_t, 2> kK5120RouteStarts{21, 257};
    constexpr std::array<std::int32_t, 3> kK5120RouteInteriors{12, 128, 1024};
    failures += infernix::test::linear_pair::run_q8_a16_shape(
        "Q8_A16 LinearPair", ShapeCase{5120, 431U, kK5120RouteStarts, kK5120RouteInteriors});

    constexpr std::array<std::int32_t, 23> kK2048RouteStarts{
        2,    33,   49,   65,   193,  257,  385,  481,  897,  961,  977,  1281,
        1317, 1345, 1346, 1441, 1467, 1681, 1709, 1921, 1923, 2017, 2019,
    };
    constexpr std::array<std::int32_t, 27> kK2048RouteInteriors{
        1,    16,   40,   56,   128,  224,  320,  432,  640,  784,  896,  968,  1120,
        1296, 1332, 1345, 1392, 1456, 1576, 1696, 1816, 1921, 1968, 2017, 2112, 2240, 4096,
    };
    failures += infernix::test::linear_pair::run_q8_a16_shape(
        "Q8_A16 LinearPair", ShapeCase{2048, 433U, kK2048RouteStarts, kK2048RouteInteriors});
    return failures;
}

} // namespace

INFERNIX_OP_TEST(linear_pair_q8_a16) {
    if (!infernix::test::linear_pair::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    try {
        const int failures = q8_a16_conformance();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Q8_A16 LinearPair\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Q8_A16 LinearPair: " << error.what() << '\n';
        return 1;
    }
}
