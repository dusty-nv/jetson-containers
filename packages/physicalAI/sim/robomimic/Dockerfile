#---
# name: robomimic
# group: sim
# depends: [pytorch, torchvision, robosuite, h5py]
# test: [test.sh]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ADD https://api.github.com/repos/ARISE-Initiative/robomimic/git/refs/heads/master /tmp/robomimic_version.json

RUN git clone --depth=1 --recursive https://github.com/ARISE-Initiative/robomimic /opt/robomimic && \
    cd /opt/robomimic && \
    uv pip install -e .
