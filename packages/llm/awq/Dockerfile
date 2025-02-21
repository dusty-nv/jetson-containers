#---
# name: awq
# group: llm
# config: config.py
# depends: [vila, sudonim]
# requires: '>=34.1.0'
# test: test.sh
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG AWQ_REPO \
    AWQ_BRANCH \
    AWQ_VERSION \
    AWQ_KERNEL_VERSION \
    FORCE_BUILD="off"

ADD https://api.github.com/repos/${AWQ_REPO}/git/refs/heads/main /tmp/awq_version.json

COPY build.sh install.sh /tmp/awq/

RUN /tmp/awq/install.sh || /tmp/awq/build.sh
    
COPY quantize.py /usr/local/bin/awq_quantize
