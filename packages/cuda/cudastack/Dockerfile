#---
# name: cudastack
# group: cuda
# config: config.py
# depends: [build-essential, cuda, numpy]
# test: test.sh
# notes: Consolidated CUDA stack - installs all components in ONE RUN to avoid layer limits
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Component versions
ARG CUDNN_VERSION \
    CUDNN_URL \
    CUDNN_DEB \
    CUDNN_PACKAGES \
    TENSORRT_VERSION \
    TENSORRT_URL \
    TENSORRT_DEB \
    TENSORRT_PACKAGES \
    NCCL_VERSION \
    NCCL_URL \
    CUDSS_VERSION \
    CUSPARSELT_VERSION \
    CUTENSOR_VERSION \
    CUTLASS_VERSION \
    GDRCOPY_VERSION \
    NVPL_VERSION \
    NVSHMEM_VERSION \
    CUDA_ARCH \
    CUDA_VERSION_MAJOR \
    CUDA_INSTALLED_VERSION \
    DISTRO \
    IS_SBSA \
    IS_TEGRA \
    MULTIARCH_URL \
    TAR_INDEX_URL \
    WITH_CUDNN=1 \
    WITH_TENSORRT=0 \
    WITH_NCCL=1 \
    WITH_CUDSS=0 \
    WITH_CUSPARSELT=0 \
    WITH_CUTENSOR=0 \
    WITH_CUTLASS=0 \
    WITH_GDRCOPY=0 \
    WITH_NVPL=0 \
    WITH_NVSHMEM=0 \
    ENABLE_DISTRIBUTED_JETSON_NCCL=0 \
    FORCE_BUILD=off \
    NUM_JOBS=8 \
    TMP=/tmp

# Persist component toggles as environment variables for testing
ENV WITH_CUDNN=${WITH_CUDNN} \
    WITH_TENSORRT=${WITH_TENSORRT} \
    WITH_NCCL=${WITH_NCCL} \
    WITH_CUDSS=${WITH_CUDSS} \
    WITH_CUSPARSELT=${WITH_CUSPARSELT} \
    WITH_CUTENSOR=${WITH_CUTENSOR} \
    WITH_CUTLASS=${WITH_CUTLASS} \
    WITH_GDRCOPY=${WITH_GDRCOPY} \
    WITH_NVPL=${WITH_NVPL} \
    WITH_NVSHMEM=${WITH_NVSHMEM} \
    ENABLE_DISTRIBUTED_JETSON_NCCL=${ENABLE_DISTRIBUTED_JETSON_NCCL}

# Copy ALL install scripts at once
COPY install/* /tmp/cuda-stack/install/
COPY build/* /tmp/cuda-stack/build/
#COPY install/*.sh build/*.sh /tmp/cuda-stack/

# Install ALL components in ONE RUN command to minimize layers
# This is the key to avoiding Docker's max depth error
RUN set -ex && \
    cd /tmp/cuda-stack && \
    chmod +x install/*.sh build/*.sh && \
    \
    echo "===== Installing CUDA Stack Components =====" && \
    \
    if [ "$WITH_CUDNN" = "1" ]; then \
        echo "===== 1/10: Installing cuDNN ${CUDNN_VERSION} =====" && \
        ./install/install_cudnn.sh; \
    fi && \
    \
    if [ "$WITH_TENSORRT" = "1" ]; then \
        echo "===== 2/10: Installing TensorRT ${TENSORRT_VERSION} =====" && \
        ./install/install_tensorrt.sh; \
    fi && \
    # Check for distributed Jetson NCCL first then fall back to standard NCCL
    if [ "$ENABLE_DISTRIBUTED_JETSON_NCCL" = "1" ]; then \
        echo "===== 3/10: Installing distributed (experimental) NCCL ${NCCL_VERSION} =====" && \
        (./build/build_nccl.sh); \
    elif [ "$WITH_NCCL" = "1" ]; then \
        echo "===== 3/10: Installing NCCL ${NCCL_VERSION} =====" && \
        (./install/install_nccl.sh || ./build/build_nccl.sh); \
    fi && \
    \
    if [ "$WITH_CUDSS" = "1" ]; then \
        echo "===== 4/10: Installing cuDSS ${CUDSS_VERSION} =====" && \
        ./install/install_cudss.sh; \
    fi && \
    \
    if [ "$WITH_CUSPARSELT" = "1" ]; then \
        echo "===== 5/10: Installing cuSPARSELt ${CUSPARSELT_VERSION} =====" && \
        ./install/install_cusparselt.sh; \
    fi && \
    \
    if [ "$WITH_CUTENSOR" = "1" ]; then \
        echo "===== 6/10: Installing cuTENSOR ${CUTENSOR_VERSION} =====" && \
        ./install/install_cutensor.sh; \
    fi && \
    \
    if [ "$WITH_NVPL" = "1" ]; then \
        echo "===== 7/10: Installing NVPL ${NVPL_VERSION} =====" && \
        ./install/install_nvpl.sh; \
    fi && \
    \
    if [ "$WITH_NVSHMEM" = "1" ]; then \
        echo "===== 8/10: Installing NVSHMEM ${NVSHMEM_VERSION} =====" && \
        ./install/install_nvshmem.sh; \
    fi && \
    \
    if [ "$WITH_GDRCOPY" = "1" ]; then \
        echo "===== 9/10: Installing GDRCopy ${GDRCOPY_VERSION} =====" && \
        (./install/install_gdrcopy.sh || ./build/build_gdrcopy.sh); \
    fi && \
    \
    if [ "$WITH_CUTLASS" = "1" ]; then \
        echo "===== 10/10: Building CUTLASS ${CUTLASS_VERSION} =====" && \
        ./build/build_cutlass.sh; \
    fi && \
    \
    echo "===== Cleaning up =====" && \
    rm -rf /tmp/cuda-stack && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean && \
    \
    echo "===== CUDA Stack Installation Complete ====="
