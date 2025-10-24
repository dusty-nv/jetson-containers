#---
# name: holoscan
# group: cv
# requires: '>=36'
# depends: [pytorch, cudastack:standard, cupy, ffmpeg]
# test: [test.sh]
# notes: https://docs.nvidia.com/holoscan/sdk-user-guide/overview.html
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt/nvidia

# Clone Holoscan-sdk & HoloHub for the examples
RUN git clone https://github.com/nvidia-holoscan/holoscan-sdk.git \
    && git clone https://github.com/nvidia-holoscan/holohub.git

# Install the Holoscan Debian package
RUN wget $WGET_FLAGS https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && apt-get update \
    && apt-get -y install holoscan

WORKDIR /
