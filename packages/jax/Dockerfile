#---
# name: jax
# group: jax
# config: config.py
# depends: [cuda, cudnn, tensorrt, numpy]
# test: test.py
# docs: |
#  Containers for JAX with CUDA support.
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# download and install the JAX wheel
ARG JAX_URL \
    JAX_WHL \
    JAX_CUDA_ARCH_ARGS

# set the CUDA architectures that JAX extensions get built for
ENV DEBIAN_FRONTEND=noninteractive \
    JAX_CUDA_ARCH_LIST=${JAX_CUDA_ARCH_ARGS}

# install prerequisites
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
    # download and install the JAX wheel
    && wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate ${JAX_URL} -O /opt/${JAX_WHL} \
    && pip3 install --verbose /opt/${JAX_WHL} \
    && python3 -c 'import jax; print(f"JAX version: {jax.__version__}"); print(f"CUDA devices: {jax.devices()}");' \
    \
    # JAX C++ extensions frequently use ninja for parallel builds
    && pip3 install --no-cache-dir \
        scikit-build \
        ninja

# set the JAX cache directory to mounted /data volume
ENV XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda
ENV JAX_CACHE_DIR=/data/models/jax
