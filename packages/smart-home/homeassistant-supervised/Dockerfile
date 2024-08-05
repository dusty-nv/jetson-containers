#---
# name: homeassistant-supervised
# group: smart-home
# config: config.py
# requires: '>=34.1.0'
# depends: [build-essential]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG OS_AGENT_VERSION \
    SUPERVISED_INSTALLER_VERSION \
    BYPASS_OS_CHECK=true 

ENV DATA_SHARE=/usr/share/hassio

# Set shell
SHELL ["/bin/bash", "-xo", "pipefail", "-c"]

WORKDIR /usr/src

COPY *.sh /tmp/

RUN /tmp/install.sh

ENTRYPOINT ["/init"]
