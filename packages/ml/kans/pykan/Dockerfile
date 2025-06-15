#---
# name: pykan
# group: pykan
# config: config.py
# depends: [cuda, pytorch, torchvision, torchaudio, triton]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/state-spaces/pykan
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG PYKAN_VERSION \
    PYKAN_VERSION_SPEC

COPY build.sh install.sh /tmp/pykan/
RUN /tmp/pykan/install.sh || /tmp/pykan/build.sh
