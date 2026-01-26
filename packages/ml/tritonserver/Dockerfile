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
	&& rm -rf /var/lib/apt/lists/* \
	&& apt-get clean

RUN uv pip install --upgrade wheel setuptools && \
	uv pip install --upgrade grpcio-tools attrdict pillow

ENV PATH="$PATH:/opt/tritonserver/bin"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib/llvm-8/lib:/opt/tritonserver/lib"

RUN python3 -c 'import tritonclient; print("Triton Client verified")' 