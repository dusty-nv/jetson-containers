#---
# name: torchvision
# group: pytorch
# config: config.py
# depends: [pytorch, cmake]
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG TORCHVISION_VERSION

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            libjpeg-dev \
		  zlib1g-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean && \
    \
    git clone --branch ${TORCHVISION_VERSION} --recursive --depth=1 https://github.com/pytorch/vision /opt/torchvision && \
    cd /opt/torchvision && \
    git checkout ${TORCHVISION_VERSION} && \
    python3 setup.py bdist_wheel && \
    cp dist/torchvision*.whl /opt && \
    cd ../ && \
    rm -rf torchvision && \
    \
    pip3 install --no-cache-dir --verbose /opt/torchvision*.whl && \
    python3 -c 'import torchvision; print(torchvision.__version__);'

# note:  this was used on older torchvision versions (~0.4) to workaround a bug,
#        but has since started causing another bug as of torchvision 0.11.1
# ARG PILLOW_VERSION=pillow<7    
# pip3 install --no-cache-dir "${PILLOW_VERSION}"
