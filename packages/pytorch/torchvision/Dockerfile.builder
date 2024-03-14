#---
# name: torchvision:builder
# group: pytorch
# config: config.py
# depends: [pytorch, cmake]
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG TORCHVISION_VERSION

RUN set -ex \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        libjpeg-dev \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    \
    && git clone --branch ${TORCHVISION_VERSION} --recursive --depth=1 https://github.com/pytorch/vision /opt/torchvision \
    && cd /opt/torchvision \
    && git checkout ${TORCHVISION_VERSION} \
    && python3 setup.py --verbose bdist_wheel --dist-dir /opt \
    && rm -rf /opt/torchvision \
    \
    && pip3 install --no-cache-dir --verbose /opt/torchvision*.whl \
    && python3 -c 'import torchvision; print(torchvision.__version__);'
