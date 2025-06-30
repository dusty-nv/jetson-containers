#---
# name: 3dgrut
# group: nerf
# config: config.py
# depends: [cuda, cupy, cuda-python, pytorch, torchvision, torchaudio, pytorch3d, tinycudann, polyscope, kaolin, opencv]
# requires: '>=34.1.0'
# notes: https://github.com/nv-tlabs/3dgrut
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG THREEGRUT_VERSION \
    FORCE_BUILD=off

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc-11 \
        g++-11 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

COPY build.sh /tmp/3dgrut/

RUN /tmp/3dgrut/build.sh

WORKDIR /opt/3dgrut/
CMD ["/bin/bash"]
