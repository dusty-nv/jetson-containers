#---
# name: fast_gauss
# group: nerf
# config: config.py
# depends: [pytorch, torchvision, torchaudio, opencv, tinycudann, open3d, pytorch3d]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/dendenxu/fast-gaussian-rasterization
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG FAST_GAUSS_VERSION \
    FORCE_BUILD=off

COPY build.sh install.sh /tmp/FAST_GAUSS/

RUN /tmp/FAST_GAUSS/install.sh || /tmp/FAST_GAUSS/build.sh
