#---
# name: vpi
# group: cv
# depends: [cuda, opengl, numpy]
# requires: '>=36.4'
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG APT_BASE_URL="https://repo.download.nvidia.com/jetson" \
    L4T_VERSION_SHORT="36.4"

RUN apt-key adv --fetch-key ${APT_BASE_URL}/jetson-ota-public.asc && \
    add-apt-repository "deb ${APT_BASE_URL}/common r${L4T_VERSION_SHORT} main" && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      libegl1-mesa && \
    apt-get install pva-allow-2 || true && \
    rm /var/lib/dpkg/info/pva-allow-2.post* && \
    dpkg --configure pva-allow-2 || true && \
    apt-get install -y --no-install-recommends \
      libnvvpi3 vpi3-dev vpi3-samples \
    && add-apt-repository -r "${APT_BASE_URL}/common" \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
