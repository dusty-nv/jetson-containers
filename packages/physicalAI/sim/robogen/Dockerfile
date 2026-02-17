#---
# name: robogen
# group: sim
# depends: [ffmpeg, genesis-world]
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
    uv pip install einops attrs nvidia-ml-py opencv-contrib-python protobuf && \
    uv pip install ompl

WORKDIR /opt/RoboGen/
CMD ["/bin/bash"]
