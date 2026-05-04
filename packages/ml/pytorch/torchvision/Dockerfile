#---
# name: torchvision
# group: pytorch
# config: config.py
# depends: [pytorch, cmake]
# buildkit_device: nvidia.com/gpu=all
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG TORCHVISION_VERSION \
    FORCE_BUILD=off

COPY build.sh install.sh /tmp/torchvision/

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libjpeg-dev \
        libpng-dev \
        zlib1g-dev \
	&& rm -rf /var/lib/apt/lists/* \
	&& apt-get clean

RUN /tmp/torchvision/install.sh || /tmp/torchvision/build.sh
