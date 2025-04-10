#---
# name: kaolin
# group: nerf
# config: config.py
# depends: [pytorch, torchvision, torchaudio, tinycudann, polyscope]
# test: test.sh
# requires: '>=34.1.0'
# notes: https://github.com/NVIDIAGameWorks/kaolin
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG KAOLIN_VERSION \
    FORCE_BUILD=off

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        unzip \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
    
COPY build.sh install.sh /tmp/kaolin/
RUN /tmp/kaolin/install.sh || /tmp/kaolin/build.sh

CMD ["/bin/bash"]
