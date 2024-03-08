#---
# name: auto_gptq
# group: llm
# config: config.py
# depends: [transformers]
# requires: '>=34.1.0'
# test: test.py
# docs: docs.md
#---
ARG BASE_IMAGE

# Build Stage
FROM ${BASE_IMAGE} AS build

ARG AUTOGPTQ_BRANCH
ARG TORCH_CUDA_ARCH_LIST
ARG BUILD_CUDA_EXT=1

ADD https://api.github.com/repos/PanQiWei/AutoGPTQ/git/refs/heads/${AUTOGPTQ_BRANCH} /tmp/autogptq_version.json

RUN set -ex \
    && pip3 install --no-cache-dir gekko \
    && git clone --branch=${AUTOGPTQ_BRANCH} --depth=1 https://github.com/PanQiWei/AutoGPTQ.git /opt/AutoGPTQ \
    && cd /opt/AutoGPTQ \
    && python3 setup.py --verbose bdist_wheel --dist-dir /opt \
    && cd ../ \
    && rm -rf /opt/AutoGPTQ

# Install Stage
FROM ${BASE_IMAGE} AS install

COPY --from=build /opt/auto_gptq*.whl /opt/

RUN set -ex \
    && pip3 install --no-cache-dir --verbose /opt/auto_gptq*.whl \
    # make sure it loads
    && pip3 show auto-gptq \
    && python3 -c 'import auto_gptq'
