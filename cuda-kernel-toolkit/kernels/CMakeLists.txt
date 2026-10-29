# cuda-kernel-toolkit/kernels — sample CUDA kernel suite (build target for CKT)
#
# The suite is deliberately independent of the Infernix engine so the toolkit
# can analyse, profile and tune it without touching product code.  Build:
#
#   cmake -S kernels -B kernels/build -DCMAKE_BUILD_TYPE=Release
#   cmake --build kernels/build -j
#
# The default arch is detected from the CUDA runtime at configure time; pass
# -DCKT_CUDA_ARCH=sm_XXX to override.
cmake_minimum_required(VERSION 3.24)
project(ckt_kernels LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)

# Detect the target arch unless explicitly requested.  Prefer the arch-specific
# form (sm_121a) when the toolkit supports it, since ISA extensions change which
# instructions are emitted and therefore what the disassembler will show.
if(NOT CKT_CUDA_ARCH)
  execute_process(
    COMMAND ${CMAKE_CUDA_COMPILER} --list-gpu-code
    OUTPUT_VARIABLE _codes
    ERROR_QUIET)
  # Ask the runtime for the actual device capability via a tiny probe.
  file(WRITE ${CMAKE_BINARY_DIR}/ckt_arch_probe.cu
    "#include <cstdio>\n#include <cuda_runtime.h>\n"
    "int main(){cudaDeviceProp p{}; if(cudaGetDeviceProperties(&p,0)!=cudaSuccess) return 1;"
    "std::printf(\"%d%d\", p.major, p.minor); return 0;}\n")
  try_run(_run_ok _compile_ok
    ${CMAKE_BINARY_DIR}/ckt_arch_probe_build ${CMAKE_BINARY_DIR}/ckt_arch_probe.cu
    CMAKE_FLAGS "-DCMAKE_CUDA_ARCHITECTURES=native"
    RUN_OUTPUT_VARIABLE _cc
    COMPILE_OUTPUT_VARIABLE _cc_log)
  if(_compile_ok AND _run_ok AND _cc MATCHES "^[0-9]+$")
    # CMake's CUDA_ARCHITECTURES wants the numeric form (e.g. 121a), not sm_121a.
    if(_codes MATCHES "sm_${_cc}a")
      set(CKT_CUDA_ARCH "${_cc}a")
    else()
      set(CKT_CUDA_ARCH "${_cc}")
    endif()
  else()
    # Fall back to the newest code the compiler knows.
    string(REGEX MATCHALL "sm_[0-9]+" _code_list "${_codes}")
    list(GET _code_list -1 _newest)
    string(REPLACE "sm_" "" _newest "${_newest}")
    set(CKT_CUDA_ARCH "${_newest}")
  endif()
endif()
message(STATUS "CKT sample kernels: building for CUDA arch ${CKT_CUDA_ARCH}")

file(GLOB CKT_KERNEL_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/k_*.cu)

add_executable(ckt_kernels ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/harness.cpp
                           ${CKT_KERNEL_SOURCES})
target_include_directories(ckt_kernels PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(ckt_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>)
set_target_properties(ckt_kernels PROPERTIES
  CUDA_ARCHITECTURES "${CKT_CUDA_ARCH}"
  CUDA_SEPARABLE_COMPILATION OFF)
target_link_libraries(ckt_kernels PRIVATE CUDA::cudart)
