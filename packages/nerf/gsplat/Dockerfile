#---
# name: gsplat
# group: nerf
# config: config.py
# depends: [nerfstudio]
# requires: '>=34.1.0'
# notes: https://pypi.org/project/gsplat/
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG GSPLAT_VERSION \
FORCE_BUILD=off

ARG CUDAARCHS

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    unzip

COPY build.sh install.sh /tmp/GSPLAT/

RUN cd /opt/

RUN /tmp/GSPLAT/install.sh || /tmp/GSPLAT/build.sh
