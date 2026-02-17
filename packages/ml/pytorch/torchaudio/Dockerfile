#---
# name: torchaudio
# group: pytorch
# config: config.py
# depends: [pytorch, cmake, torchcodec]
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG TORCHAUDIO_VERSION \
    FORCE_BUILD=off

COPY install.sh build.sh /tmp/torchaudio/

RUN /tmp/torchaudio/install.sh || /tmp/torchaudio/build.sh
