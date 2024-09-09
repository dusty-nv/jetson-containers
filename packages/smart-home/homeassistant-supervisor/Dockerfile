#---
# name: homeassistant-supervisor
# group: smart-home
# config: config.py
# requires: '>=34.1.0'
# depends: [homeassistant-base, ciso8601, eudev]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG COSIGN_VERSION \
  YQ_VERSION \
  SUPERVISOR_VERSION

SHELL ["/bin/bash", "-xo", "pipefail", "-c"]

WORKDIR /usr/src

ENV DEFAULT_PYTHON="3.12" \
  CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1 \
  S6_SERVICES_GRACETIME=10000 \
  SUPERVISOR_API=http://localhost \
  LD_PRELOAD="${LD_PRELOAD:-}:/usr/local/lib/libjemalloc.so.2" \
  MALLOC_CONF="background_thread:true,metadata_thp:auto" \
  SUPERVISOR_NAME="hassio_supervisor" \
  SUPERVISOR_MACHINE="qemuarm-64" \
  SUPERVISOR_SHARE="/usr/share/hassio"

COPY *.sh /tmp/homeassistant-supervisor/

RUN /tmp/homeassistant-supervisor/install_deps.sh \
  && /tmp/homeassistant-supervisor/build.sh

WORKDIR /

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -sSL "http://localhost:80/supervisor/ping" | jq -r '.result' | grep -q "ok" || exit 1

EXPOSE 80

LABEL \
  io.hass.type="supervisor" \
  io.hass.version="${SUPERVISOR_VERSION}" \
  io.hass.arch="${BUILD_ARCH}" \
  io.hass.machine="${SUPERVISOR_MACHINE}" \
  maintainer="Mieszko Syty <https://github.com/ms1design>" \
  org.opencontainers.image.title="Home Assistant Supervisor" \
  org.opencontainers.image.description="Container-based system for managing Home Assistant Core installation" \
  org.opencontainers.image.source="https://github.com/home-assistant/supervisor" \
  org.opencontainers.image.authors="The Home Assistant Authors" \
  org.opencontainers.image.url="https://www.home-assistant.io/" \
  org.opencontainers.image.documentation="https://www.home-assistant.io/docs/" \
  org.opencontainers.image.licenses="Apache License 2.0"

CMD ["/init"]
