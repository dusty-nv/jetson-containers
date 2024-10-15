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
    apt install -y speech-dispatcher pulseaudio-utils alsa-utils && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

RUN echo "pactl info" >> /root/.bash_history
RUN echo "pactl list short sinks" >> /root/.bash_history
RUN echo "pactl set-default-sink 0" >> /root/.bash_history

CMD speech-dispatcher --spawn  && /bin/bash