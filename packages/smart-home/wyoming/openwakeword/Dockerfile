#---
# name: wyoming-openwakeword
# group: wyoming
# requires: '>=34.1.0'
# docs: docs.md
# depends: [homeassistant-base-image, python:3.11]
# notes: The `openWakeWord` using the `wyoming` protocol for usage with Home Assistant. Based on `https://github.com/home-assistant/addons/blob/master/openwakeword/Dockerfile`
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG OPENWAKEWORD_VERSION

ENV PIP_BREAK_SYSTEM_PACKAGES=1 \
    OPENWAKEWORD_PORT=10400 \
    OPENWAKEWORD_THRESHOLD=0.5 \
    OPENWAKEWORD_TRIGGER_LEVEL=1 \
    OPENWAKEWORD_PRELOAD_MODEL="ok_nabu" \
    OPENWAKEWORD_DEBUG=true

WORKDIR /usr/src

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        netcat-traditional \
        libopenblas0 \
    \
    && pip3 install --no-cache-dir -U \
        setuptools \
        wheel \
    && echo "wyoming-openwakeword: ${OPENWAKEWORD_VERSION}" \
    && pip3 install --no-cache-dir \
        --extra-index-url https://www.piwheels.org/simple \
        "wyoming-openwakeword @ https://github.com/rhasspy/wyoming-openwakeword/archive/refs/tags/v${OPENWAKEWORD_VERSION}.tar.gz" \
    \
    && git clone --depth=1 https://github.com/home-assistant/addons /tmp/addons \
    && git -C /tmp/addons sparse-checkout set --no-cone openwakeword/ '!*/openwakeword' \
    # Fix: Adjust reading config data from config.yaml file when not using HA Supervisor \
    && sed -i \
        -e "s/bashio::config.true 'debug_logging'/[[ \"$OPENWAKEWORD_DEBUG\" == \"true\" ]]/g" \
        -e "s/--threshold.*/--threshold \"$OPENWAKEWORD_THRESHOLD\" \\\\/g" \
        -e "s/--trigger-level.*/--trigger-level \"$OPENWAKEWORD_TRIGGER_LEVEL\" \${flags[@]}/g" \
        /tmp/addons/openwakeword/rootfs/etc/s6-overlay/s6-rc.d/openwakeword/run \
    \
    # Fix: Disable native Discovery handled by HA Supervisor \
    && cat /tmp/addons/openwakeword/rootfs/etc/s6-overlay/s6-rc.d/discovery/run \
    && sed -i '$d' /tmp/addons/openwakeword/rootfs/etc/s6-overlay/s6-rc.d/discovery/run \
    && sed -i '$d' /tmp/addons/openwakeword/rootfs/etc/s6-overlay/s6-rc.d/discovery/run \
    && sed -i '$d' /tmp/addons/openwakeword/rootfs/etc/s6-overlay/s6-rc.d/discovery/run \
    && sed -i '$d' /tmp/addons/openwakeword/rootfs/etc/s6-overlay/s6-rc.d/discovery/run \
    && sed -i '$d' /tmp/addons/openwakeword/rootfs/etc/s6-overlay/s6-rc.d/discovery/run \
    && cat /tmp/addons/openwakeword/rootfs/etc/s6-overlay/s6-rc.d/discovery/run \
    \
    && cp -r /tmp/addons/openwakeword/rootfs/* / \
    && cp /tmp/addons/openwakeword/config.yaml /etc/s6-overlay/s6-rc.d/openwakeword/config.yaml \
    && rm -rf /tmp/addons \
    \
    && python3 -c 'import wyoming_openwakeword; print(wyoming_openwakeword.__version__);' \
    \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /

# FIXME: For some reason below HEALTHCHECK is not playing nicelly with my CI/CD env
# instead of just declaring itself it executes the CMD...
# HEALTHCHECK --start-period=10m CMD echo '{ "type": "describe" }' | nc -w 1 localhost ${OPENWAKEWORD_PORT} | grep -iq "openWakeWord" || exit 1

EXPOSE ${OPENWAKEWORD_PORT}/tcp

ENTRYPOINT ["/init"]
