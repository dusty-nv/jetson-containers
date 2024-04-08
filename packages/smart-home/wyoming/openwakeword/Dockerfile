#---
# name: wyoming-openwakeword
# group: wyoming
# requires: '>=34.1.0'
# depends: [homeassistant-base-image, python:3.11]
# test: test.py
# notes: The `openWakeWord` using the `wyoming` protocol for usage with Home Assistant base on `https://github.com/home-assistant/addons/blob/master/openwakeword/Dockerfile`
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG OPENWAKEWORD_VERSION

ENV PIP_BREAK_SYSTEM_PACKAGES=1 \
    OPENWAKEWORD_PORT=10400 \
    OPENWAKEWORD_THRESHOLD=0.5 \
    OPENWAKEWORD_TRIGGER_LEVEL=1 \
    OPENWAKEWORD_PRELOAD_MODEL="ok_nabu"

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
    && cp -r /tmp/addons/openwakeword/rootfs/* / \
    && rm -rf /tmp/addons \
    \
    && python3 -c 'import wyoming_openwakeword; print(wyoming_openwakeword.__version__);' \
    \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /

HEALTHCHECK --start-period=10m \
    CMD echo '{ "type": "describe" }' \
        | nc -w 1 localhost ${OPENWAKEWORD_PORT} \
        | grep -iq "openWakeWord" \
        || exit 1

EXPOSE 10400/tcp

ENTRYPOINT ["/init"]

# CMD python3 -m wyoming_openwakeword \
#     --uri "tcp://0.0.0.0:${OPENWAKEWORD_PORT}" \
#     --preload-model "${OPENWAKEWORD_PRELOAD_MODEL}" \
#     --custom-model-dir /share/openwakeword \
#     --threshold "${OPENWAKEWORD_THRESHOLD}" \
#     --trigger-level "${OPENWAKEWORD_TRIGGER_LEVEL}" \
#     --debug-logging "false" \
#     "${flags[@]}"
