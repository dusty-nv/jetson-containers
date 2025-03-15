#---
# name: optix
# group: cuda
# depends: cuda
# config: config.py
# test: test.sh
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG OPTIX_VERSION
ARG OPTIX_URL
ARG OPTIX_DEB
ARG OPTIX_PACKAGES
ENV OPTIX_INSTALL_DIR="/usr/local/"

###########################
# Download and install OPTIX
###########################
RUN echo "Downloading OPTIX" && \
    mkdir /tmp/optix && cd /tmp/optix && \
    wget --quiet --show-progress --progress=bar:force:noscroll ${OPTIX_URL} && \
    chmod +x nvidia-optix-sdk-${OPTIX_VERSION}-linux64-aarch64.sh && \
    sh nvidia-optix-sdk-${OPTIX_VERSION}-linux64-aarch64.sh --skip-license \
       --prefix="${OPTIX_INSTALL_DIR}" && \
    rm -rf /tmp/optix

###########################
# Download and install NVTT 3
###########################
ENV NVTT_URL="https://developer.nvidia.com/downloads/texture-tools-linux-aarch64"
ENV NVTT_ARCHIVE="nvtt3_aarch64.tar.xz"
ENV NVTT_INSTALL_DIR="/opt/nvtt3"

RUN echo "Downloading NVTT 3 from: ${NVTT_URL}" && \
    mkdir -p "${NVTT_INSTALL_DIR}" && \
    wget -O "${NVTT_ARCHIVE}" "${NVTT_URL}" && \
    echo "Extracting ${NVTT_ARCHIVE} into ${NVTT_INSTALL_DIR}" && \
    tar -xf "${NVTT_ARCHIVE}" -C "${NVTT_INSTALL_DIR}" --strip-components=1 && \
    # (Optional) Clean up the archive
    rm "${NVTT_ARCHIVE}"
