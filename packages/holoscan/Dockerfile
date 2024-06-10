#---
# name: holoscan
# group: ml
# requires: '>=36.3'
# depends: [cuda, cudnn, tensorrt, cmake, python]
# test: [test.py]
# notes: https://docs.nvidia.com/holoscan/sdk-user-guide/getting_started.html
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN python3 -m pip install holoscan
# --------------------------------------------------------------------------
#
# Holohub run setup 
#
WORKDIR /opt/nvidia
RUN git clone https://github.com/nvidia-holoscan/holohub.git
WORKDIR /opt/nvidia/holohub

RUN chmod +x run
RUN ./run setup
# - This variable is consumed by all dependencies below as an environment variable (CMake 3.22+)
# - We use ARG to only set it at docker build time, so it does not affect cmake builds
#   performed at docker run time in case users want to use a different BUILD_TYPE
ARG CMAKE_BUILD_TYPE=Release

# Qcap dependency
RUN apt update \
    && apt install --no-install-recommends -y \
        libgstreamer1.0-0 \
        libgstreamer-plugins-base1.0-0 \
        libgles2 \
        libopengl0

# For benchmarking
RUN apt update \
    && apt install --no-install-recommends -y \
    libcairo2-dev \
    libgirepository1.0-dev \
    gobject-introspection \
    libgtk-3-dev \
    libcanberra-gtk-module \
    graphviz
RUN if ! grep -q "VERSION_ID=\"22.04\"" /etc/os-release; then \
        pip install setuptools; \
    fi

RUN pip install -r benchmarks/holoscan_flow_benchmarking/requirements.txt
