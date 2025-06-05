#---
# name: isaacsim
# alias: isaacsim
# group: sim
# config: config.py
# depends: [ffmpeg, cmake, h5py, pytorch, torchvision, torchaudio, transformers]
# requires: '>=35'
# test: [test.sh, test.py]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG ISAACSIM_VERSION \
    FORCE_BUILD="off"

COPY build.sh install.sh /tmp/isaacsim/

RUN /tmp/isaacsim/install.sh || /tmp/isaacsim/build.sh || echo "BUILD FAILED (isaacsim ${ISAACSIM_VERSION})"

