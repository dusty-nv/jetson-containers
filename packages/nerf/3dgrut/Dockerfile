#---
# name: 3dgrut
# group: nerf
# config: config.py
# depends: [pytorch, torchvision, torchaudio, tinycudann, polyscope]
# requires: '>=34.1.0'
# notes: https://pypi.org/project/gsplat/
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG GSPLAT_VERSION \
    FORCE_BUILD=off

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        unzip \
        gcc-11 \
        g++-11 \
        libgl1-mesa-dev \
        libglu1-mesa-dev \
        mesa-common-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
    
COPY build.sh install.sh /tmp/3dgrut/

RUN /tmp/3dgrut/install.sh || /tmp/3dgrut/build.sh

CMD ["/bin/bash"]
