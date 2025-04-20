#---
# name: video-codec-sdk
# alias: [nvcodec, nvenc, nvcuvid, cuvid]
# group: multimedia
# depends: [cuda, opengl, vulkan, ffmpeg, cmake]
# notes: NVIDIA Video Codec SDK with NVENC and NVCUVID
# test: test.sh
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG NV_CODEC_VERSION \
    SOURCE=/opt/nvidia/video-codec-sdk \
		TMP=/tmp/video-codec-sdk

COPY install.sh *.zip $TMP/

RUN $TMP/install.sh
