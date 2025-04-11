#---
# name: habitat-sim
# group: sim
# config: config.py
# depends: [pytorch, numba]
# requires: '>=35'
# test: [test.sh, test.py]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG HABITAT_SIM_VERSION \
    HABITAT_SIM_BRANCH \
    FORCE_BUILD="off"

COPY build.sh install.sh /tmp/habitat-sim/

RUN /tmp/habitat-sim/install.sh || /tmp/habitat-sim/build.sh || echo "BUILD FAILED (habitat-sim ${HABITAT_SIM_VERSION})"

