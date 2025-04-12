#---
# name: vulkan
# group: multimedia
# depends: [build-essential]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      vulkan-tools \
      libvulkan1 \
      libvulkan-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
