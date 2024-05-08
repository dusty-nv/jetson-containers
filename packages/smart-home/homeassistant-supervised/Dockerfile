#---
# name: homeassistant-supervised
# group: smart-home
# config: config.py
# test: [test_os-agent.sh]
# requires: '>=34.1.0'
# depends: [build-essential]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG OS_AGENT_VERSION \
    SUPERVISED_VERSION \
    BYPASS_OS_CHECK=true 

ENV DATA_SHARE=/usr/share/hassio

# Set shell
SHELL ["/bin/bash", "-xo", "pipefail", "-c"]

WORKDIR /usr/src

COPY build_*.sh install_*.sh /tmp/

RUN /tmp/install_dependencies.sh
RUN /tmp/build_homeassistant_supervised.sh

VOLUME /run/systemd/system/bus

# VOLUME /var/lib/docker
# VOLUME /usr/bin/docker
# VOLUME /var/run/docker.sock
