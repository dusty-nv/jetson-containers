#---
# name: video-codec-sdk
# alias: [nvcodec, nvenc, nvcuvid, cuvid]
# group: multimedia
# depends: [cuda]
# notes: NVIDIA Video Codec SDK with NVENC and NVCUVID
# test: test.sh
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG NV_CODEC_VERSION \
    BUILD_SAMPLES=off \
    SOURCE=/opt/nvidia/video-codec-sdk \
		TMP=/tmp/video-codec-sdk
# Download the zip from https://developer.nvidia.com/video-codec-sdk/download
# and place it in the same directory as this Dockerfile.
COPY build.sh install.sh *.zip $TMP/

RUN $TMP/build.sh || $TMP/install.sh
