#---
# name: nerfview
# group: nerf
# config: config.py
# depends: [pytorch, torchvision, torchaudio, pycolmap, tinycudann, gsplat]
# requires: '>=34.1.0'
# notes: https://pypi.org/project/nerfview/
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG NERFVIEW_VERSION \
    FORCE_BUILD=off

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        unzip \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
    
COPY build.sh install.sh /tmp/nerfview/

RUN /tmp/nerfview/install.sh || /tmp/nerfview/build.sh
