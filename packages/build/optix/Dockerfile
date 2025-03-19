#---
# name: optix
# group: build
# depends: [build-essential, python]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN curl https://developer.nvidia.com/downloads/designworks/optix/secure/9.0.0/nvidia-optix-sdk-9.0.0-linux64-aarch64.sh && \
    chmod a+x ./nvidia-optix-sdk-9.0.0-linux64-aarch64.sh && \
    sh nvidia-optix-sdk-9.0.0-linux64-aarch64.sh

