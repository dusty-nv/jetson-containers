#---
# name: homeassistant-os-agent
# group: smart-home
# config: config.py
# depends: [go-lang]
# test: test.sh
# requires: '>=34.1.0'
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG OS_AGENT_VERSION \
    CGO_ENABLED=0 \
    OS_AGENT_TMP_DIR="/tmp/homeassistant-os-agent"

SHELL ["/bin/bash", "-xo", "pipefail", "-c"]

COPY *.sh *.diff ${OS_AGENT_TMP_DIR}-install/

RUN "${OS_AGENT_TMP_DIR}-install/install.sh"
