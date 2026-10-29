# syntax=docker/dockerfile:1

FROM nvidia/cuda:13.4.2-devel-ubuntu26.04 AS build

# Compile target kernel architecture. Matches the host GB10 (sm_121a) by
# default; docker-compose can override it. Must be one of 120a|121a.
ARG INFERNIX_CUDA_ARCH=121a
ENV INFERNIX_CUDA_ARCH=${INFERNIX_CUDA_ARCH}

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        cmake \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libcurl4-openssl-dev \
        libjpeg-turbo8-dev \
        libswscale-dev \
        ninja-build \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

RUN cmake -S . -B /build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=${INFERNIX_CUDA_ARCH} \
        -DINFERNIX_BUILD_APPS=ON \
        -DBUILD_TESTING=OFF \
        -DINFERNIX_BUILD_BENCHMARKS=ON \
    && cmake --build /build --parallel --target infernix infernix-serve infernix_bench

FROM nvidia/cuda:13.4.2-runtime-ubuntu26.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        libavcodec62 \
        libavformat62 \
        libavutil60 \
        libcurl4t64 \
        libjpeg-turbo8 \
        libswscale9 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /build/apps/infernix /usr/local/bin/infernix
COPY --from=build /build/apps/infernix-serve /usr/local/bin/infernix-serve
COPY --from=build /build/bench/infernix_bench /usr/local/bin/infernix_bench

WORKDIR /workspace
EXPOSE 8080
STOPSIGNAL SIGTERM

CMD ["infernix-serve", "--help"]
