#---
# name: apex
# group: attention
# config: config.py
# depends: [pytorch, triton, xformers]
# requires: '>=35'
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG APEX_VERSION \
    IS_SBSA \
    FORCE_BUILD=off

COPY build.sh install.sh /tmp/apex/

RUN /tmp/apex/install.sh || /tmp/apex/build.sh

