#---
# name: protomotions
# group: sim
# config: config.py
# depends: [genesis-world]
# buildkit_device: nvidia.com/gpu=all
# requires: '>=34.1.0'
# test: test.sh
# notes: https://github.com/NVlabs/ProtoMotions
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG PROTOMOTIONS_VERSION \
    CUDAARCHS

COPY build.sh install.sh /tmp/PROTOMOTIONS/

RUN /tmp/PROTOMOTIONS/install.sh || /tmp/PROTOMOTIONS/build.sh

WORKDIR /opt/protomotions
CMD ["/bin/bash"]
