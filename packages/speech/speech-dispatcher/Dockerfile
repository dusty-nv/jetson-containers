#---
# name: speech-dispatcher
# group: audio
# depends: []
# requires: '>=34.1.0'
# docs: docs.md
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

RUN apt-get update && \
    apt install -y speech-dispatcher alsa-utils && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

CMD speech-dispatcher --spawn  && /bin/bash