#---
# name: gsplat
# group: nerf
# config: config.py
# depends: [pytorch, torchvision, torchaudio, tinycudann]
# requires: '>=34.1.0'
# notes: https://pypi.org/project/gsplat/
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG GSPLAT_VERSION \
    FORCE_BUILD=off

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        unzip \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
    
COPY build.sh install.sh /tmp/GSPLAT/

RUN /tmp/GSPLAT/install.sh || /tmp/GSPLAT/build.sh
