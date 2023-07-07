#
# torchvision - see config.py for build args
#
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG TORCHVISION_VERSION
ARG TORCH_CUDA_ARCH_LIST="5.3;6.2;7.2;8.7"

RUN printenv && echo "torchvision version = $TORCHVISION_VERSION" && echo "TORCH_CUDA_ARCH_LIST = $TORCH_CUDA_ARCH_LIST"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		  git \
		  build-essential \
		  ninja-build \
            libjpeg-dev \
		  zlib1g-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN git clone --branch ${TORCHVISION_VERSION} --recursive --depth=1 https://github.com/pytorch/vision torchvision && \
    cd torchvision && \
    git checkout ${TORCHVISION_VERSION} && \
    python3 setup.py install && \
    cd ../ && \
    rm -rf torchvision

# note:  this was used on older torchvision versions (~0.4) to workaround a bug,
#        but has since started causing another bug as of torchvision 0.11.1
# ARG PILLOW_VERSION=pillow<7    
# pip3 install --no-cache-dir "${PILLOW_VERSION}"
