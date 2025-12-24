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

RUN python3 -c 'import tritonclient; print("Triton Client verified")' && \
	python3 -c 'import vllm; print(f"vLLM version: {vllm.__version__}")'

# Install vLLM backend (Option 3: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/vllm_backend/README.html)
# The vllm python package is installed via 'depends' in config.py
RUN git clone https://github.com/triton-inference-server/vllm_backend.git /tmp/vllm_backend && \
	mkdir -p /opt/tritonserver/backends/vllm && \
	cp /tmp/vllm_backend/src/model.py /opt/tritonserver/backends/vllm/model.py && \
	rm -rf /tmp/vllm_backend

# Verify vLLM backend installation
RUN if [ ! -f "/opt/tritonserver/backends/vllm/model.py" ]; then \
	echo "ERROR: vllm backend model.py not found in /opt/tritonserver/backends/vllm"; \
	exit 1; \
	else \
	echo "vLLM backend installed successfully at /opt/tritonserver/backends/vllm"; \
	fi
