#---
# name: video-codec-sdk
# alias: [nvcodec, nvenc, nvcuvid, cuvid]
# group: multimedia
# depends: [cuda]
# notes: NVIDIA Video Codec SDK headers from https://github.com/FFmpeg/nv-codec-headers
# test: test.sh
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG NV_CODEC_VERSION \
    NV_CODEC_HEADERS_TAG

COPY install.sh /tmp/video-codec-sdk/

RUN /tmp/video-codec-sdk/install.sh
