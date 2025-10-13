#---
# name: cuda
# group: cuda
# config: config.py
# depends: [build-essential]
# test: test.sh
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG CUDA_URL \
    CUDA_DEB \
    CUDA_PACKAGES \
    CUDA_ARCH_LIST \
    CUDA_ARCH \
    CUDA_INSTALLED_VERSION \
    IS_SBSA \
    DISTRO="ubuntu2004"

COPY install.sh /tmp/cuda/install.sh
RUN /tmp/cuda/install.sh

ENV CUDA_HOME="/usr/local/cuda"
ENV NVCC_PATH="$CUDA_HOME/bin/nvcc"

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all \
    CUDAARCHS=${CUDA_ARCH_LIST} \
    CUDA_ARCHITECTURES=${CUDA_ARCH_LIST} \
    CUDA_INSTALLED_VERSION=${CUDA_INSTALLED_VERSION} \
    CUDA_HOME="/usr/local/cuda" \
    CUDNN_LIB_PATH="/usr/lib/aarch64-linux-gnu" \
    CUDNN_LIB_INCLUDE_PATH="/usr/include" \
    CMAKE_CUDA_COMPILER=${NVCC_PATH} \
    CUDA_NVCC_EXECUTABLE=${NVCC_PATH} \
    CUDACXX=${NVCC_PATH} \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CUDA_BIN_PATH="${CUDA_HOME}/bin" \
    CUDA_TOOLKIT_ROOT_DIR="${CUDA_HOME}" \
    LD_LIBRARY_PATH="${CUDA_HOME}/compat:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}" \
    LDFLAGS="-L/usr/local/cuda/lib64 ${LDFLAGS}" \
    DEBIAN_FRONTEND=noninteractive

# === Fix -lcuda and CCCL includes ===
# Ensure that the linker searches stubs first and keeps previous flags
# CUDA 13 moved thrust/cub to include/cccl
# ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}:/usr/local/cuda/lib64/stubs
# LIBRARY_PATH and LDFLAGS only affect the linker at build time. In that case,
# it makes sense to put .../stubs at the end as a “fallback” if the actual driver is not there.
# LD_LIBRARY_PATH is runtime. Even if you put stubs at the end,
# if the app does dlopen(“libcuda.so”) (without .1), often only that name exists in the stubs,
# and the loader will end up going to stubs anyway → cuInit does not exist → crashes.
# In addition, LD_LIBRARY_PATH takes priority over the ldconfig cache.


ENV PATH="${PATH}:${CUDA_HOME}/bin" \
    LIBRARY_PATH=${LIBRARY_PATH:+${LIBRARY_PATH}:}/usr/local/cuda/lib64/stubs \
    LDFLAGS="${LDFLAGS:+${LDFLAGS} }-L/usr/local/cuda/lib64/stubs -Wl,-rpath,/usr/local/cuda/lib64" \
    CPLUS_INCLUDE_PATH=/usr/local/cuda/include/cccl${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}} \
    C_INCLUDE_PATH=/usr/local/cuda/include/cccl${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}}

# Create symbolic links so that libcuda.so and libcuda.so.1 point to the CUDA stub libraries.
# This allows linking against CUDA when the actual driver libraries are not available at build time.
# The '-f' flag forces overwrite if the link or file already exists, and '|| true' prevents the build from failing if linking fails.
#RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so || true \
# && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so.1 /usr/local/cuda/lib64/libcuda.so.1 || true

WORKDIR /
