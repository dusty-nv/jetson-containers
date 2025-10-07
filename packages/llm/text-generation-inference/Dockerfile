#---
# name: text-generation-inference
# group: llm
# depends: [pytorch, torchvision, bitsandbytes, transformers, rust]
# requires: '>=34.1.0'
# notes: https://github.com/huggingface/text-generation-inference
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

# install protoc (we don't need the full version from protobuf:cpp,
# and the version from protobuf:apt is too old - 21.12 is in the HF docs)
ARG PROTOC_URL=https://github.com/protocolbuffers/protobuf/releases/download/v21.12/protoc-21.12-linux-aarch_64.zip
ARG PROTOC_ZIP=protoc-21.12-linux-aarch_64.zip

RUN wget $WGET_FLAGS ${PROTOC_URL} -O ${PROTOC_ZIP} && \
    unzip -o ${PROTOC_ZIP} -d /usr/local bin/protoc && \
    unzip -o ${PROTOC_ZIP} -d /usr/local 'include/*' && \
    rm ${PROTOC_ZIP}

RUN which protoc && protoc --version

# parts of the makefiles refer to 'python' instead of 'python3'
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/uv pip 1

RUN git clone --depth=1 https://github.com/huggingface/text-generation-inference

WORKDIR /opt/text-generation-inference/server

RUN sed 's|^bitsandbytes==.*|bitsandbytes|g' -i requirements.txt

ARG BUILD_EXTENSIONS=True

RUN env

RUN make gen-server

RUN sed 's|python_version >= "3.9"|python_version >= "3.8"|g' -i requirements.txt && \
    sed 's|==.* ;|\ ;|g' -i requirements.txt && \
    #sed 's|^scipy.*|scipy|g' -i requirements.txt && \
    #sed 's|^opentelemetry-api.*|opentelemetry-api|g' -i requirements.txt && \
    cat requirements.txt

RUN uv pip install -r requirements.txt

RUN sed 's|^python = .*|python = "^3.8"|g' -i pyproject.toml && \
    sed 's|^protobuf = .*|protobuf = \"*\"|g' -i pyproject.toml && \
    sed 's|^grpcio = .*|grpcio = \"*\"|g' -i pyproject.toml && \
    sed 's|^grpcio-status = .*|grpcio-status = \"*\"|g' -i pyproject.toml && \
    sed 's|^grpcio-reflection = .*|grpcio-reflection = \"*\"|g' -i pyproject.toml && \
    sed 's|^opentelemetry-api = .*|opentelemetry-api = \"*\"|g' -i pyproject.toml && \
    sed 's|^opentelemetry-exporter-otlp = .*|opentelemetry-exporter-otlp = \"*\"|g' -i pyproject.toml && \
    sed 's|^opentelemetry-instrumentation-grpc = .*|opentelemetry-instrumentation-grpc = \"*\"|g' -i pyproject.toml && \
    sed 's|^scipy = .*|scipy = \"*\"|g' -i pyproject.toml && \
    cat pyproject.toml

RUN uv pip install --no-deps -e ".[bnb, accelerate]" --verbose

WORKDIR /opt/text-generation-inference/

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		  libssl-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN make install-router
RUN make install-launcher

ARG TORCH_CUDA_ARCH_LIST
RUN env

RUN sed 's|\"-arch=compute_80\",| |g' -i server/custom_kernels/setup.py && \
    cat server/custom_kernels/setup.py

RUN make install-custom-kernels

# install the text-generation client library
RUN uv pip install text-generation

# re-install bitsandbytes because it got overwritten
RUN uv pip install /opt/bitsandbytes*.whl

# make sure it loads
RUN cd /opt/text-generation-inference/server/text_generation_server && python3 cli.py --help
