#---
# name: easyvolcap
# group: nerf
# config: config.py
# depends: [pytorch, torchvision, torchaudio, opencv, tinycudann, fast_gauss, open3d, pytorch3d]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/zju3dv/EasyVolcap
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG EASYVOLCAP_VERSION \
    FORCE_BUILD=off

COPY build.sh install.sh /tmp/easyvolcap/

RUN /tmp/easyvolcap/install.sh || /tmp/easyvolcap/build.sh
