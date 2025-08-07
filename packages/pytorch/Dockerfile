#---
# name: pytorch
# alias: torch
# group: pytorch
# config: config.py
# depends: [cuda, cudnn, nccl, gdrcopy, nvpl, numpy, onnx]
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
    DISTRO="ubuntu2204" \
    FORCE_BUILD=off \
    PYTORCH_OFFICIAL_WHL=off

ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL} \
    TORCH_HOME=/data/models/torch

ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all -compress-mode=balance"

COPY install.sh build.sh /tmp/pytorch/

# attempt to install from pip, and fall back to building it
RUN /tmp/pytorch/install.sh || /tmp/pytorch/build.sh
