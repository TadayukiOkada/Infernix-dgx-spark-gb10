cmake_minimum_required(VERSION 3.28)

# Architecture configuration:
# DGX Spark (NVIDIA GB10, compute capability 12.1): target sm_121a.
# Warp-specialized kernels use setmaxnreg, so the family-specific 'a' suffix is required.
# Plain 121 / 121-real (recommended by the DGX Spark porting guide) is normalized to 121a so
# PTXAS enables setmaxnreg.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 121a CACHE STRING "CUDA architectures to build")
endif()

# Normalize porting-guide syntax to the architecture-accelerated 'a' variant
if(CMAKE_CUDA_ARCHITECTURES MATCHES "^(121|121-real)$")
  message(STATUS "Normalizing CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES} to 121a (GB10 / DGX Spark)")
  set(CMAKE_CUDA_ARCHITECTURES 121a CACHE STRING "CUDA architectures to build" FORCE)
endif()

# Validate that the requested architecture is DGX Spark (121a)
if(NOT CMAKE_CUDA_ARCHITECTURES MATCHES "^121a$")
  message(FATAL_ERROR
    "Infernix supports CMAKE_CUDA_ARCHITECTURES=121a (GB10 / DGX Spark); got "
    "'${CMAKE_CUDA_ARCHITECTURES}'")
endif()

project(infernix LANGUAGES C CXX CUDA)

# --- Global settings ---------------------------------------------------------
option(INFERNIX_BUILD_APPS "Build the Infernix CLI, server and perplexity evaluator" ON)
option(BUILD_TESTING "Build Infernix tests" OFF)
option(INFERNIX_BUILD_BENCHMARKS "Build Infernix benchmarks" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# No C++20 modules: no dependency scan (MSBuild names its output by source file name, so a
# target's same-named sources, such as the five causal-cache plan.cpp, overwrite each other's).
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)        # clangd / IDE tooling
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Host CPU architecture optimizations for ARM64 (DGX Spark Cortex-X925 / Cortex-A725)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-march=armv9-a+simd+sve+sve2+i8mm+bf16>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=armv9-a+simd+sve+sve2+i8mm+bf16>
  )
endif()

if(CMAKE_GENERATOR MATCHES "Ninja")
  # Linking dominates an incremental rebuild: one CUDA object change relinks every executable
  # that includes it (242 links for a 12-file change was measured on the 20-core GB10 host), and
  # a serialized link pool left the machine ~63% idle for 347 s. A link peaks at ~0.5 GB RSS, so
  # a small depth multiplies throughput at negligible memory cost. Set INFERNIX_LINK_JOBS=1 to
  # restore strict serialization if a link ever exhausts memory.
  set(INFERNIX_LINK_JOBS 8 CACHE STRING
      "Concurrent link jobs for Ninja (1 serializes every link)")
  set_property(GLOBAL PROPERTY JOB_POOLS infernix_link=${INFERNIX_LINK_JOBS})
  set(CMAKE_JOB_POOL_LINK infernix_link)
endif()

# Tests also exercise product input, logging and protocol adapters.
set(INFERNIX_BUILD_PRODUCT_SUPPORT OFF)
if(INFERNIX_BUILD_APPS OR BUILD_TESTING)
  set(INFERNIX_BUILD_PRODUCT_SUPPORT ON)
endif()

include(cmake/Dependencies.cmake)
include(cmake/InfernixTargets.cmake)

# --- Subdirectories ----------------------------------------------------------
add_subdirectory(src)
if(INFERNIX_BUILD_APPS)
  add_subdirectory(apps)
endif()
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
if(INFERNIX_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()
