#---
# name: espeak
# group: audio
# depends: [build-essential]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG ESPEAK_NG_VERSION=master 
ENV ESPEAK_NG_DIR="/opt/espeak-ng" \
    ESPEAK_NG_DATA_DIR="/opt/espeak-ng/espeak-ng-data"

# install espeak-ng prerequisites
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        autoconf automake libtool pkg-config \
        libsonic-dev ronn kramdown \
        libpcaudio-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Even though piper-phonemize will download espeak-ng version
# automatically, we cannot use it because it will not use
# the espeak_TextToPhonemesWithTerminator:
# - https://github.com/rhasspy/piper-phonemize/issues/30
# - https://github.com/espeak-ng/espeak-ng/pull/2127
RUN git clone --branch ${ESPEAK_NG_VERSION} --depth 1 https://github.com/espeak-ng/espeak-ng /opt/espeak-ng && \
    cd /opt/espeak-ng && \
    git fetch origin pull/2127/head:pr-branch && \
    git checkout pr-branch && \
    git status && \
    ./autogen.sh && \
    ./configure --prefix=/usr/local && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    ls -l $ESPEAK_NG_DATA_DIR
    