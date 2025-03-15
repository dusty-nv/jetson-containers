#---
# name: optix
# group: cuda
# depends: cuda
# config: config.py
# test: test.sh
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG OPTIX_URL
ARG OPTIX_DEB
ARG OPTIX_PACKAGES

RUN ls /etc/apt/sources.list.d/ && \
    apt-get update && \
    apt-cache search optix

RUN echo "Downloading OPTIX" && \
    mkdir /tmp/optix && cd /tmp/optix && \
    wget --quiet --show-progress --progress=bar:force:noscroll ${OPTIX_URL} && \
    chmod +x nvidia-optix-sdk-9.0.0-linux64-aarch64.sh && \
    sh nvidia-optix-sdk-9.0.0-linux64-aarch64.sh && \
    rm -rf /tmp/optix
