#---
# name: homeassistant-base-image
# group: smart-home
# config: config.py
# requires: '>=34.1.0'
# depends: [build-essential]
# notes: The `homeassistant` base ubuntu image with pre-installed dependencies based on `https://github.com/home-assistant/docker-base/blob/master/ubuntu/Dockerfile`
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV \
    LANG="C.UTF-8" \
    S6_BEHAVIOUR_IF_STAGE2_FAILS=2 \
    S6_CMD_WAIT_FOR_SERVICES_MAXTIME=0 \
    S6_CMD_WAIT_FOR_SERVICES=1 \
    S6_SERVICES_READYTIME=50

# Set shell
SHELL ["/bin/bash", "-exo", "pipefail", "-c"]

# Version
ARG BUILD_ARCH="aarch64" \
    S6_ARCH="aarch64" \
    BASHIO_VERSION \
    TEMPIO_VERSION \
    S6_OVERLAY_VERSION

# Base system
WORKDIR /usr/src

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        jq \
        xz-utils \
    \
    && curl -L -f -s "https://github.com/just-containers/s6-overlay/releases/download/v${S6_OVERLAY_VERSION}/s6-overlay-${S6_ARCH}.tar.xz" \
        | tar Jxvf - -C / \
    && curl -L -f -s "https://github.com/just-containers/s6-overlay/releases/download/v${S6_OVERLAY_VERSION}/s6-overlay-noarch.tar.xz" \
        | tar Jxvf - -C / \
    && curl -L -f -s "https://github.com/just-containers/s6-overlay/releases/download/v${S6_OVERLAY_VERSION}/s6-overlay-symlinks-arch.tar.xz" \
        | tar Jxvf - -C / \
    && curl -L -f -s "https://github.com/just-containers/s6-overlay/releases/download/v${S6_OVERLAY_VERSION}/s6-overlay-symlinks-noarch.tar.xz" \
        | tar Jxvf - -C / \
    && mkdir -p /etc/fix-attrs.d \
    && mkdir -p /etc/services.d \
    \
    && curl -L -f -s -o /usr/bin/tempio \
        "https://github.com/home-assistant/tempio/releases/download/${TEMPIO_VERSION}/tempio_${BUILD_ARCH}" \
    && chmod a+x /usr/bin/tempio \
    \
    && mkdir -p /usr/src/bashio \
    && curl -L -f -s "https://github.com/hassio-addons/bashio/archive/v${BASHIO_VERSION}.tar.gz" \
        | tar -xzf - --strip 1 -C /usr/src/bashio \
    \
    # FIX: Remove HA Supervisor from bashio (not used) \
    && sed -i \
        -e 's|^# shellcheck source=lib/supervisor.sh||g' \
        -e 's|source "${__BASHIO_LIB_DIR}/supervisor.sh"||g' \
        /usr/src/bashio/lib/bashio.sh \
    \
    && mv /usr/src/bashio/lib /usr/lib/bashio \
    && ln -s /usr/lib/bashio/bashio /usr/bin/bashio \
    \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /usr/src/*

# S6-Overlay
WORKDIR /
