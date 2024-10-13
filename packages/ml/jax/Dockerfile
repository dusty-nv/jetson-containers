#---
# name: jax
# group: jax
# config: config.py
# depends: [cuda, cudnn, numpy]
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
    FORCE_BUILD

ENV JAX_CUDA_ARCH_LIST=${JAX_CUDA_ARCH_ARGS} \
    JAX_CACHE_DIR=/data/models/jax

# copy installation and build scripts for JAX
COPY build.sh install.sh link_cuda.sh /tmp/JAX/

# attempt to install JAX from pip, and fall back to building it if the installation fails
RUN /tmp/JAX/install.sh || /tmp/JAX/build.sh
    
