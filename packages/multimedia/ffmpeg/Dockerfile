#---
# name: ffmpeg
# group: multimedia
# depends: [build-essential]
# notes: https://github.com/FFmpeg/FFmpeg
# test: test.sh
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG FFMPEG_INSTALL \
    FFMPEG_VERSION \
		TMP=/tmp/ffmpeg

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential yasm nasm pkg-config libx264-dev libx265-dev libvpx-dev libfdk-aac-dev \
      libopus-dev libvorbis-dev libass-dev libunistring-dev && \
    rm -rf /var/lib/apt/lists/*

COPY build.sh install.sh $TMP/

RUN $TMP/install.sh
