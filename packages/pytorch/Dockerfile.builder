# 
# Dockerfile for building PyTorch from source
# see the other Dockerfile & config.py for package configuration/metadata
#
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Docker build arguments passed from `config.py`
ARG PYTORCH_BUILD_VERSION \
    PYTORCH_BUILD_NUMBER \
    PYTORCH_BUILD_EXTRA_ENV \
    TORCH_CUDA_ARCH_ARGS

ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_ARGS}

RUN set -ex \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        libopenblas-dev \
        libopenmpi-dev \
        openmpi-bin \
        openmpi-common \
        gfortran \
        libomp-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    \
    # Build PyTorch wheel with extra environmental variables for custom feature switch \
    && git clone --branch "v${PYTORCH_BUILD_VERSION}" --depth=1 --recursive https://github.com/pytorch/pytorch /tmp/pytorch \
    && cd /tmp/pytorch \
    && export USE_NCCL=0 \
        USE_QNNPACK=0 \
        USE_PYTORCH_QNNPACK=0 \
        USE_NATIVE_ARCH=1 \
        USE_DISTRIBUTED=1 \
        USE_TENSORRT=0 \
        ${PYTORCH_BUILD_EXTRA_ENV} \
    && pip3 install -r requirements.txt \
    && pip3 install --no-cache-dir scikit-build ninja \
    && python3 setup.py bdist_wheel --dist-dir /opt \
    && rm -rf /tmp/pytorch \
    \
    # install the compiled wheel \
    && pip3 install --verbose /opt/torch*.whl \
    && python3 -c 'import torch; print(f"PyTorch version: {torch.__version__}"); print(f"CUDA available:  {torch.cuda.is_available()}"); print(f"cuDNN version:   {torch.backends.cudnn.version()}"); print(f"torch.distributed:   {torch.distributed.is_available()}"); print(torch.__config__.show());' \
    \
    # patch for https://github.com/pytorch/pytorch/issues/45323 \
    && PYTHON_ROOT=`pip3 show torch | grep Location: | cut -d' ' -f2` \
    && TORCH_CMAKE_CONFIG="${PYTHON_ROOT}/torch/share/cmake/Torch/TorchConfig.cmake" \
    && echo "patching _GLIBCXX_USE_CXX11_ABI in ${TORCH_CMAKE_CONFIG}" \
    && sed -i 's/  set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=")/  set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")/g' ${TORCH_CMAKE_CONFIG}

# set the torch hub model cache directory to mounted /data volume
ENV TORCH_HOME=/data/models/torch
