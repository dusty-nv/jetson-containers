#---
# name: torchcodec
# group: pytorch
# config: config.py
# depends: [pytorch, ffmpeg, cmake]
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG TORCHCODEC_VERSION \
    FORCE_BUILD=off

COPY build.sh install.sh /tmp/torchcodec/

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libjpeg-dev \
        libpng-dev \
        zlib1g-dev \
	&& rm -rf /var/lib/apt/lists/* \
	&& apt-get clean

RUN /tmp/torchcodec/install.sh || /tmp/torchcodec/build.sh
