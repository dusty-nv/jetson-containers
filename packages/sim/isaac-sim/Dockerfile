#---
# name: isaacsim
# alias: isaacsim
# group: sim
# config: config.py
# depends: [ffmpeg, cmake, h5py, pytorch, torchvision, torchaudio, transformers, physx, warp, newton]
# requires: '>=35'
# test: [test.sh, test.py]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG ISAACSIM_VERSION \
    ARCH \
    FORCE_BUILD="off"

COPY build.sh install.sh /tmp/isaacsim/

RUN /tmp/isaacsim/install.sh || /tmp/isaacsim/build.sh || echo "BUILD FAILED (isaacsim ${ISAACSIM_VERSION})"

WORKDIR /opt/isaacsim

ENV OMNI_KIT_ALLOW_ROOT=1
# Isaac Sim root directory
ENV ISAACSIM_PATH="/opt/isaacsim/_build/linux-${ARCH}/release"
# Isaac Sim python executable
ENV ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"
