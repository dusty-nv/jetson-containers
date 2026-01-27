#---
# name: tritonserver
# group: ml
# config: config.py
# requires: '>=32.6'
# depends: [cuda, cudastack:standard, python]
# test: test.py
# notes: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/jetson.html
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG TRITON_URL
ARG TRITON_TAR
ARG TRITON_VERSION
ARG TRITON_CLIENTS

RUN cd /opt && \
    wget $WGET_FLAGS ${TRITON_URL} -O ${TRITON_TAR} && \
    tar -xzvf ${TRITON_TAR} && \
    rm ${TRITON_TAR}

RUN uv pip install --upgrade /opt/${TRITON_CLIENTS}/python/tritonclient-${TRITON_VERSION}-py3-none-manylinux2014_aarch64.whl[all]

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libb64-0d \
    libre2-9 \
    libssl3 \
    rapidjson-dev \
    libopenblas-dev \
    libarchive-dev \
    zlib1g \
    curl \
    jq \
    python3-scipy \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Rebuild V5: Downgrade Strategy
# Use System Numpy (1.21.5) to guarantee Triton stability.
# Downgrade Numba to <0.57 to be compatible with Numpy 1.21.
# Install Librosa (which works with older numba).
# Using --no-build-isolation to try to use system numpy if possible, or just standard pip.
# Note: we do NOT install numpy here.

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools && \
    python3 -m pip install --no-cache-dir \
    "numba<0.57" \
    "librosa" \
    "grpcio-tools" \
    "attrdict" \
    "pillow"

ENV PATH="$PATH:/opt/tritonserver/bin"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/targets/aarch64-linux/lib"

RUN uv run python -c 'import tritonclient; print("Triton Client verified")' 

RUN uv pip install $(find /usr -name "tensorrt-*-cp310-*-linux_aarch64.whl" -print -quit)
