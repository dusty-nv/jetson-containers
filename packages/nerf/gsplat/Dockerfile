#---
# name: gsplat
# group: nerf
# config: config.py
# depends: [pytorch, torchvision, torchaudio, tinycudann, nerfview]
# requires: '>=34.1.0'
# notes: https://github.com/nerfstudio-project/gsplat
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG GSPLAT_VERSION \
    FORCE_BUILD=off

COPY build.sh install.sh /tmp/GSPLAT/

RUN /tmp/GSPLAT/install.sh || /tmp/GSPLAT/build.sh
