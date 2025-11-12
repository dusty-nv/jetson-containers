#---
# name: pytorch3d
# group: pytorch
# config: config.py
# depends: [cuda, cmake, pytorch, torchvision, torchaudio]
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG PYTORCH3D_VERSION \
    PYTORCH3D_VERSION_SPEC \
    FORCE_BUILD=off

COPY build.sh install.sh /tmp/pytorch3d/

RUN /tmp/pytorch3d/install.sh || /tmp/pytorch3d/build.sh
