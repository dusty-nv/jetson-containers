#---
# name: auto_gptq:builder
# group: llm
# config: config.py
# depends: [transformers]
# requires: '>=34.1.0'
# test: test.py
# docs: docs.md
#---
ARG BASE_IMAGE

FROM ${BASE_IMAGE}
ARG AUTOGPTQ_BRANCH \
    TORCH_CUDA_ARCH_LIST \
    BUILD_CUDA_EXT=1

ADD https://api.github.com/repos/PanQiWei/AutoGPTQ/git/refs/heads/${AUTOGPTQ_BRANCH} /tmp/autogptq_version.json

RUN set -ex \
    && pip3 install --no-cache-dir gekko \
    && git clone --branch=${AUTOGPTQ_BRANCH} --depth=1 https://github.com/PanQiWei/AutoGPTQ.git /opt/AutoGPTQ \
    && cd /opt/AutoGPTQ \
    && python3 setup.py --verbose bdist_wheel --dist-dir /opt \
    && rm -rf /opt/AutoGPTQ \
    \
    && pip3 install --no-cache-dir --verbose /opt/auto_gptq*.whl \
    \
    && pip3 show auto-gptq \
    && python3 -c 'import auto_gptq'
