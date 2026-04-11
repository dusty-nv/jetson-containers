#---
# name: multi-turboquant
# group: attention
# config: config.py
# depends: [pytorch, triton]
# requires: '>=35'
# test: test.py
# notes: https://github.com/rookiemann/multi-turboquant.git
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG MULTI_TURBOQUANT_VERSION \
    FORCE_BUILD=off

COPY build.sh install.sh /tmp/multi-turboquant/

RUN /tmp/multi-turboquant/install.sh || /tmp/multi-turboquant/build.sh
