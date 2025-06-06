#---
# name: memvid
# group: rag
# config: config.py
# depends: [cmake, ffmpeg, opencv]
# requires: '>=35'
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG MEMVID_VERSION \
    IS_SBSA \
    FORCE_BUILD=off

COPY build.sh install.sh /tmp/memvid/

RUN /tmp/memvid/install.sh || /tmp/memvid/build.sh
    
