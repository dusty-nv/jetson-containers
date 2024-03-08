#---
# name: torchvision
# group: pytorch
# config: config.py
# depends: [pytorch, cmake]
# test: test.py
#---
ARG BASE_IMAGE

# Build Stage
FROM ${BASE_IMAGE} AS build

ARG TORCHVISION_VERSION \
    TORCH_CUDA_ARCH_LIST="5.3;6.2;7.2;8.7"

RUN set -ex \
    && echo "torchvision version = $TORCHVISION_VERSION" \
    && echo "TORCH_CUDA_ARCH_LIST = $TORCH_CUDA_ARCH_LIST" \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        libjpeg-dev \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && git clone --branch ${TORCHVISION_VERSION} --recursive --depth=1 https://github.com/pytorch/vision /opt/torchvision \
    && cd /opt/torchvision \
    && git checkout ${TORCHVISION_VERSION} \
    && python3 setup.py --verbose bdist_wheel --dist-dir /opt \
    && rm -rf /opt/torchvision

# Install Stage
FROM ${BASE_IMAGE} AS install

COPY --from=build /opt/torchvision*.whl /opt/

RUN set -ex \
    && pip3 install --no-cache-dir --verbose /opt/torchvision*.whl \
    && python3 -c 'import torchvision; print(torchvision.__version__);'
