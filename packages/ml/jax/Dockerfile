#---
# name: jax
# group: jax
# config: config.py
# depends: [cuda, cudastack:standard, numpy, onnx, llvm:21]
# test: test.py
# docs: Containers for JAX with CUDA support.
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# set the CUDA architectures that JAX extensions get built for
# set the JAX cache directory to mounted /data volume
ARG JAX_CUDA_ARCH_ARGS \
    JAX_VERSION \
    JAX_BUILD_VERSION \
    ENABLE_NCCL \
    CUDA_VERSION \
    IS_SBSA \
    FORCE_BUILD=off

ENV JAX_CUDA_ARCH_LIST=${JAX_CUDA_ARCH_ARGS} \
    JAX_CACHE_DIR=/data/models/jax

# copy installation and build scripts for JAX
COPY build.sh install.sh link_cuda.sh /tmp/JAX/

# attempt to install JAX from pip, and fall back to building it if the installation fails
RUN /tmp/JAX/install.sh || /tmp/JAX/build.sh
