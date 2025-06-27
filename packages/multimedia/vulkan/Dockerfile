#---
# name: vulkan
# group: multimedia
# depends: [build-essential, pip_cache, cmake, llvm]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG VULKAN_VERSION \
    FORCE_BUILD=off \
    TMP=/tmp/vulkan

COPY build.sh install.sh $TMP/

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        lsb-release && \
    UBUNTU_VERSION=$(lsb_release -rs) && \
    if [ "$UBUNTU_VERSION" = "22.04" ]; then \
        apt-get install -y gcc-12 g++-12; \
    elif [ "$UBUNTU_VERSION" = "24.04" ] || [ "$UBUNTU_VERSION" = "24.04" ]; then \
        apt-get install -y gcc-14 g++-14; \
    else \
        echo "Unsupported Ubuntu version: $UBUNTU_VERSION" && exit 1; \
    fi && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        bison \
        ocaml-core \
        xz-utils \
        pkg-config \
        libglm-dev \
        libxcb-dri3-0 \
        libxcb-present0 \
        libpciaccess0 \
        libpng-dev \
        libxcb-keysyms1-dev \
        libxcb-dri3-dev \
        libxcb-glx0-dev \
        libxcb-glx0 \
        libx11-dev \
        libwayland-dev \
        libxrandr-dev \
        libxcb-randr0-dev \
        libxcb-ewmh-dev \
        libx11-xcb-dev \
        liblz4-dev \
        libzstd-dev \
        libxml2-dev \
        wayland-protocols && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

RUN $TMP/install.sh || $TMP/build.sh || true

RUN if [ ! -f "$TMP/.vulkan" ]; then \
      echo "FAILED to install Vulkan SDK $VULKAN_VERSION"; \
      exit 1; \
    fi
