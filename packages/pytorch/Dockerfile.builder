# 
# Dockerfile for building PyTorch from source
# see the other Dockerfile & config.py for package configuration/metadata
#
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Docker build arguments passed from `config.py`
ARG PYTORCH_BUILD_VERSION
ARG PYTORCH_BUILD_NUMBER
ARG PYTORCH_BUILD_EXTRA_ENV

ARG TORCH_CUDA_ARCH_ARGS
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_ARGS}
RUN echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"

# install prerequisites
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		  libopenblas-dev \
		  libopenmpi-dev \
            openmpi-bin \
            openmpi-common \
		  gfortran \
		  libomp-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Adding sm_87 (only needed for PyTorch 2.0)
#ARG CPP_EXTENSION_PY_FILE=/opt/pytorch/torch/utils/cpp_extension.py
#RUN sed -z "s|       ('Turing', '7.5+PTX'),\n        ('Ampere', '8.0;8.6+PTX'),|       ('Turing', '7.5+PTX'),\n        ('Ampere+Tegra', '8.7'),('Ampere', '8.0;8.6+PTX'),|" -i ${CPP_EXTENSION_PY_FILE} && \ 
#    sed "s|'8.6', '8.9'|'8.6', '8.7', '8.9'|" -i ${CPP_EXTENSION_PY_FILE} && \ 
#    sed -n 1729,1746p ${CPP_EXTENSION_PY_FILE}

# Build PyTorch wheel with extra environmental variables for custom feature switch
RUN git clone --branch v${PYTORCH_BUILD_VERSION} --depth=1 --recursive https://github.com/pytorch/pytorch /tmp/pytorch && \
    cd /tmp/pytorch && \
    export USE_NCCL=0 && \
    export USE_QNNPACK=0 && \
    export USE_PYTORCH_QNNPACK=0 && \
    export USE_NATIVE_ARCH=1 && \
    export USE_DISTRIBUTED=1 && \
    export USE_TENSORRT=0 && \
    export ${PYTORCH_BUILD_EXTRA_ENV} && \
    pip3 install -r requirements.txt && \
    pip3 install --no-cache-dir scikit-build ninja && \
    python3 setup.py bdist_wheel && \
    cp dist/*.whl /opt && \
    rm -rf /tmp/pytorch
    
# install the compiled wheel
RUN pip3 install --verbose /opt/torch*.whl

RUN python3 -c 'import torch; print(f"PyTorch version: {torch.__version__}"); print(f"CUDA available:  {torch.cuda.is_available()}"); print(f"cuDNN version:   {torch.backends.cudnn.version()}"); print(f"torch.distributed:   {torch.distributed.is_available()}"); print(torch.__config__.show());'

# patch for https://github.com/pytorch/pytorch/issues/45323
RUN PYTHON_ROOT=`pip3 show torch | grep Location: | cut -d' ' -f2` && \
    TORCH_CMAKE_CONFIG=$PYTHON_ROOT/torch/share/cmake/Torch/TorchConfig.cmake && \
    echo "patching _GLIBCXX_USE_CXX11_ABI in ${TORCH_CMAKE_CONFIG}" && \
    sed -i 's/  set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=")/  set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")/g' ${TORCH_CMAKE_CONFIG}

# set the torch hub model cache directory to mounted /data volume
ENV TORCH_HOME=/data/models/torch

WORKDIR /