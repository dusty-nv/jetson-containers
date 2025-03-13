#---
# name: optix
# group: cuda
# depends: [build-essential, python, cuda]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN curl https://developer.nvidia.com/downloads/designworks/optix/secure/9.0.0/nvidia-optix-sdk-9.0.0-linux64-aarch64.sh && \
    chmod +x nvidia-optix-sdk-9.0.0-linux64-aarch64.sh && \
    sh vidia-optix-sdk-9.0.0-linux64-aarch64.sh
