#---
# name: ffmpeg
# group: multimedia
# depends: [build-essential]
# notes: installs ffmpeg
# test: test.sh
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG FFMPEG_INSTALL \
    FFMPEG_VERSION \
		TMP=/tmp/ffmpeg

COPY build.sh install.sh $TMP/

RUN $TMP/install.sh
