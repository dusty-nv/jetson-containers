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

# install espeak-ng
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential git ca-certificates \
        autoconf automake libtool pkg-config \
        libpulse-dev libsonic-dev \
        ronn kramdown && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean && \
    \
    git clone --branch ${ESPEAK_NG_VERSION} --depth 1 https://github.com/espeak-ng/espeak-ng /opt/espeak-ng && \
    cd /opt/espeak-ng && \
    ./autogen.sh && \
    ./configure --prefix=/usr/local && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    ls -l $ESPEAK_NG_DATA_DIR
