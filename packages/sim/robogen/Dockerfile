#--- 
# name: robogen
# group: sim
# depends: [ffmpeg, genesis]
# requires: '>=34.1.0'
# docs: docs.md
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --recursive https://github.com/Genesis-Embodied-AI/RoboGen && \
    cd RoboGen && \
    pip3 install einops attrs pynvml opencv-contrib-python protobuf && \
    pip3 install ompl

WORKDIR /opt/RoboGen/
CMD ["/bin/bash"]