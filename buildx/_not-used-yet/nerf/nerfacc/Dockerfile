#---
# name: nerfacc
# group: nerf
# config: config.py
# depends: [pytorch, torchvision, torchaudio]
# requires: '>=34.1.0'
# notes: https://github.com/nerfstudio-project/nerfacc
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG NERFACC_VERSION \
FORCE_BUILD=off

ARG CUDAARCHS

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    unzip

COPY build.sh install.sh /tmp/NERFACC/

RUN cd /opt/

RUN /tmp/NERFACC/install.sh || /tmp/NERFACC/build.sh
