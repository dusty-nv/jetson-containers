#---
# name: 4k4d
# group: nerf
# config: config.py
# depends: [pytorch, torchvision, torchaudio, opencv, tinycudann, fast_gauss, easyvolcap, open3d, pytorch3d]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/zju3dv/4k4D
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG 4k4D_VERSION \
    FORCE_BUILD=off

COPY build.sh install.sh /tmp/4k4D/

RUN /tmp/4k4D/install.sh || /tmp/4k4D/build.sh
