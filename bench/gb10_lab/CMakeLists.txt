# GB10 kernel laboratory — standalone CUDA/C++ benchmark suite.
#
# Deliberately independent of the infernix_ops library and its allocators so
# measurements are not contaminated by product dispatch logic. Build directly:
#
#   cmake -S bench/gb10_lab -B bench/gb10_lab/build -DCMAKE_BUILD_TYPE=Release
#   cmake --build bench/gb10_lab/build -j
#
# It may also be added as a subdirectory of the top-level project; the
# CUDA_ARCHITECTURES guard below keeps it building even when the parent sets a
# different default.
cmake_minimum_required(VERSION 3.28)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(gb10_lab LANGUAGES CXX CUDA)
endif()

# Target the installed GB10 toolchain rather than inheriting a generic host
# default. The top-level Infernix project mandates 121a (Blackwell arch-specific
# ISA); when built standalone the cache may still hold a stale generic value
# (e.g. 75), which silently down-levels every kernel to Turing PTX and forces
# the GPU to JIT old code. Force the GB10 target unless the parent already set
# a valid Blackwell arch, so a standalone build can never regress to sm_75.
if(NOT CMAKE_CUDA_ARCHITECTURES MATCHES "^121a?(;121a?)?$")
  set(CMAKE_CUDA_ARCHITECTURES 121a CACHE STRING "CUDA architectures to build (GB10)" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

add_library(gb10_lab_common INTERFACE)
target_include_directories(gb10_lab_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(gb10_lab_common INTERFACE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>)

# Single executable exposing every op via --op. Keeps the surface area small and
# lets Python scripts drive sweeps through one binary.
file(GLOB GB10_LAB_OPS_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/ops/*.cu)

add_executable(gb10_bench ${CMAKE_CURRENT_SOURCE_DIR}/main.cu ${GB10_LAB_OPS_SOURCES})
target_link_libraries(gb10_bench PRIVATE
  gb10_lab_common
  CUDA::cudart
  CUDA::cublas
  Threads::Threads)

# cuBLAS provides strong GEMM/GEMV baselines; custom kernels are judged against
# them, never in isolation.
target_link_options(gb10_bench PRIVATE $<$<PLATFORM_ID:Linux>:>)

enable_testing()
# A smoke test runs every op in quick mode so CI catches regressions cheaply.
add_test(NAME gb10_lab_smoke COMMAND gb10_bench --op all --quick --iters 3 --warmup 1)
set_tests_properties(gb10_lab_smoke PROPERTIES TIMEOUT 600)
