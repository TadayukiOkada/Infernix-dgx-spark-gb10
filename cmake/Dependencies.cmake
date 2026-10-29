find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# CUDA runtime is exposed as a single interface target so every component links the
# same runtime flavour.
add_library(infernix_cuda_runtime INTERFACE)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
target_link_libraries(infernix_cuda_runtime INTERFACE CUDA::cudart)

# FFmpeg from the system, via pkg-config.
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
  libavformat>=60 libavcodec>=60 libavutil>=58 libswscale>=7)
set(INFERNIX_FFMPEG_TARGET PkgConfig::FFMPEG)

# libjpeg(-turbo) decodes JPEG images as Pillow does (media decoding). The codec is
# repository-pinned rather than host-discovered: the checked-in libjpeg-turbo 2.1.5 headers and
# the aarch64 static archive under third_party/libjpeg-turbo are the only accepted implementation,
# so a clean checkout needs no distribution development package and decoded pixels cannot drift
# with a host libjpeg. FFmpeg still needs the system libjpeg.so.8 at runtime, which stays separate.
set(INFERNIX_LIBJPEG_ROOT "${PROJECT_SOURCE_DIR}/third_party/libjpeg-turbo/usr")
set(INFERNIX_LIBJPEG_LIBDIR "${INFERNIX_LIBJPEG_ROOT}/lib/${CMAKE_LIBRARY_ARCHITECTURE}")
if(NOT EXISTS "${INFERNIX_LIBJPEG_LIBDIR}/libjpeg.a")
  message(FATAL_ERROR
    "Repository-pinned libjpeg-turbo is incomplete: expected "
    "'${INFERNIX_LIBJPEG_ROOT}/include/jpeglib.h' and "
    "'${INFERNIX_LIBJPEG_LIBDIR}/libjpeg.a' (host library architecture "
    "'${CMAKE_LIBRARY_ARCHITECTURE}'). Restore third_party/libjpeg-turbo from git; "
    "it is version-controlled, not a generated or installed dependency.")
endif()
# Seeded before find_package so FindJPEG adopts the pinned copy instead of searching the host,
# and forced so a stale or host-derived cache entry from an earlier configure cannot win.
set(JPEG_INCLUDE_DIR "${INFERNIX_LIBJPEG_ROOT}/include" CACHE PATH "" FORCE)
set(JPEG_LIBRARY_RELEASE "${INFERNIX_LIBJPEG_LIBDIR}/libjpeg.a" CACHE FILEPATH "" FORCE)
find_package(JPEG REQUIRED)

# Repository-pinned header dependencies. No configure-time downloads.
add_library(infernix::json INTERFACE IMPORTED GLOBAL)
target_include_directories(infernix::json INTERFACE
  ${PROJECT_SOURCE_DIR}/third_party)
add_library(infernix::toml INTERFACE IMPORTED GLOBAL)
target_include_directories(infernix::toml INTERFACE
  ${PROJECT_SOURCE_DIR}/third_party/tomlplusplus)

# Source base for the custom-template frontend; consumers will link it explicitly.
add_subdirectory(third_party/llama-jinja EXCLUDE_FROM_ALL)

# CPU grammar source base; consumers will link it explicitly.
add_subdirectory(third_party/xgrammar EXCLUDE_FROM_ALL)

if(INFERNIX_BUILD_PRODUCT_SUPPORT)
  # Media acquisition uses CURLOPT_PROTOCOLS_STR and CURLOPT_REDIR_PROTOCOLS_STR,
  # introduced in libcurl 7.85 (not merely the version of the maintainer environment).
  pkg_check_modules(LIBCURL REQUIRED IMPORTED_TARGET libcurl>=7.85)
  set(INFERNIX_CURL_TARGET PkgConfig::LIBCURL)
  add_library(infernix::httplib INTERFACE IMPORTED GLOBAL)
  target_include_directories(infernix::httplib INTERFACE
    ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
  add_subdirectory(third_party/spdlog)
endif()
