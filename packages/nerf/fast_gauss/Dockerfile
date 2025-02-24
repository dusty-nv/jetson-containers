#---
# name: fast_gauss
# group: nerf
# config: config.py
# depends: [pytorch, torchvision, torchaudio, tinycudann]
# requires: '>=34.1.0'
# test: test.py
# notes: https://pypi.org/project/gsplat/
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG FAST_GAUSS_VERSION \
    FORCE_BUILD=off

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        unzip \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
    
COPY build.sh install.sh /tmp/FAST_GAUSS/

RUN /tmp/FAST_GAUSS/install.sh || /tmp/FAST_GAUSS/build.sh
