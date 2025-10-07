#---
# name: robosuite
# group: sim
# depends: [opencv, numba, ffmpeg, mujoco]
# test: [test.sh, test.py]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ADD https://api.github.com/repos/dusty-nv/robosuite/git/refs/heads/master /tmp/robosuite_version.json

RUN git clone https://github.com/dusty-nv/robosuite /opt/robosuite && \
    apt-get update && \
    apt-get install -y --no-install-recommends libhidapi-dev && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean && \
    cd /opt/robosuite && \
    uv pip install -e . && \
    uv pip install imageio[ffmpeg] pyspacemouse 'mujoco==3.2.0'

    #\
    #git clone https://github.com/ARISE-Initiative/robosuite-task-zoo /opt/robosuite-task-zoo && \
    #cd /opt/robosuite-task-zoo && \
    #uv pip install -e .
