#---
# name: mimicgen
# group: sim
# depends: [robomimic]
# test: [test.sh, test.py]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ADD https://api.github.com/repos/dusty-nv/mimicgen/git/refs/heads/main /tmp/mimicgen_version.json

RUN git clone --depth=1 --recursive https://github.com/dusty-nv/mimicgen /opt/mimicgen && \
    cd /opt/mimicgen && \
    uv pip install -e .
