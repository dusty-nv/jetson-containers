#---
# name: holoscan-sdk
# group: ml
# requires: '>=36'
# depends: [pytorch, cupy, tensorrt]
# test: [test.sh]
# notes: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt/nvidia
# Clone Holoscan-sdk for the examples
RUN git clone https://github.com/nvidia-holoscan/holoscan-sdk.git
# Install the Holoscan python package
RUN python3 -m pip install holoscan --no-cache-dir

WORKDIR /