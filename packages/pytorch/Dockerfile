#---
# name: pytorch
# alias: torch
# group: pytorch
# config: config.py
# depends: [cuda, cudastack:standard, numpy, onnx]
# test: [test.sh, test.py]
# docs: |
#  Containers for PyTorch with CUDA support.
#  Note that the [`l4t-pytorch`](/packages/l4t/l4t-pytorch) containers also include PyTorch, `torchvision`, and `torchaudio`.
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# set the CUDA architectures that PyTorch extensions get built for
# set the torch hub model cache directory to mounted /data volume
ARG TORCH_CUDA_ARCH_LIST \
    TORCH_VERSION \
    PYTORCH_BUILD_VERSION \
    PIP_EXTRA_INDEX_URL \
    USE_NCCL=1 \
    USE_GLOO=1 \
    USE_MPI=0 \
    USE_FBGEMM=0 \
    USE_NNPACK=1 \
    USE_XNNPACK=1 \
    USE_PYTORCH_QNNPACK=1 \
    USE_BLAS=1 \
    BLAS="OpenBLAS" \
    DISTRO="ubuntu2004" \
    FORCE_BUILD=${FORCE_BUILD}\
    PYTORCH_OFFICIAL_WHL=off \
    IS_SBSA=${IS_SBSA}

ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL} \
    TORCH_HOME=/data/models/torch

ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all -compress-mode=balance"

# Conditional installation based on FORCE_BUILD
RUN if [ "${FORCE_BUILD}" = "on" ]; then \
        echo "Installing PyTorch build dependencies..." && \
        uv pip install \
            scikit-build \
            ninja \
            'cmake<4' \
            && rm -rf ~/.cache/pip; \
    else \
        echo "Skipping build dependency installation (FORCE_BUILD=${FORCE_BUILD})"; \
    fi


# Pre-clone PyTorch repository to cache it
RUN if [ "${FORCE_BUILD}" = "on" ]; then \
        echo "Pre-cloning PyTorch repository..." && \
        rm -rf /opt/pytorch-cache && \
        git clone --depth=1 --recursive https://github.com/pytorch/pytorch /opt/pytorch-cache && \
        echo "PyTorch repository cached successfully"; \
    else \
        echo "Skipping PyTorch repository cloning (FORCE_BUILD=${FORCE_BUILD})"; \
    fi

COPY requirements.txt requirements-build.txt /tmp/

# Conditional requirements installation
RUN if [ "${FORCE_BUILD}" = "on" ]; then \
        echo "Installing requirements.txt..." && \
        uv pip install -r /tmp/requirements.txt \
            && rm -rf ~/.cache/pip \
            && rm /tmp/requirements.txt; \
    else \
        echo "Skipping requirements installation (FORCE_BUILD=${FORCE_BUILD})"; \
    fi

COPY install.sh build.sh /tmp/pytorch/

# attempt to install from pip, and fall back to building it
RUN /tmp/pytorch/install.sh || /tmp/pytorch/build.sh
