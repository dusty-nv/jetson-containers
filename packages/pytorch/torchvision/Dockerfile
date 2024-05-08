#---
# name: torchvision
# group: pytorch
# config: config.py
# depends: [pytorch, cmake]
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG TORCHVISION_VERSION

COPY build.sh /tmp/build_torchvision.sh

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libjpeg-dev \
        libpng-dev \
        zlib1g-dev \
	&& rm -rf /var/lib/apt/lists/* \
	&& apt-get clean

RUN pip3 install --no-cache-dir torchvision==${TORCHVISION_VERSION} || /tmp/build_torchvision.sh
