#---
# name: psutil-home-assistant
# group: smart-home
# config: config.py
# requires: '>=34.1.0'
# depends: [build-essential, python]
# test: test.py
# notes: The `psutil-home-assistant` wheel that's build is saved in `/opt/wheels`
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG FORCE_BUILD=off \
    PSUTIL_HA_VERSION \
    PSUTIL_HA_BRANCH

WORKDIR /

COPY *.diff *.sh /tmp/psutil-home-assistant/

RUN /tmp/psutil-home-assistant/install.sh || /tmp/psutil-home-assistant/build.sh
