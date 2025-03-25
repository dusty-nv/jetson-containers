#---
# name: 3dgrut
# group: nerf
# config: config.py
# depends: [pytorch, torchvision, torchaudio, pytorch3d, tinycudann, polyscope, kaolin]
# requires: '>=34.1.0'
# notes: https://github.com/nv-tlabs/3dgrut
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG THREEGRUT_VERSION \
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
    
COPY build.sh /tmp/3dgrut/

RUN /tmp/3dgrut/build.sh

CMD ["/bin/bash"]
