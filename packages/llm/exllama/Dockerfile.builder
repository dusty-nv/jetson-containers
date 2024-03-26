#---
# name: exllama:v1:builder
# alias: exllama
# group: llm
# config: config.py
# depends: [pytorch, huggingface_hub]
# requires: '>=34.1.0'
# test: test.sh
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG TORCH_CUDA_ARCH_LIST \
    EXLLAMA_REPO=jllllll/exllama \
    EXLLAMA_BRANCH=master

ADD https://api.github.com/repos/${EXLLAMA_REPO}/git/refs/heads/${EXLLAMA_BRANCH} /tmp/exllama_version.json

RUN set -ex \
    && git clone --branch=${EXLLAMA_BRANCH} --depth=1 https://github.com/${EXLLAMA_REPO} /opt/exllama \
    && sed 's|^torch.*|torch|g' -i /opt/exllama/requirements.txt \
    && sed 's|\[\"cublas\"\] if platform.system\(\) == \"Windows\" else \[\]|\[\"cublas\"\]|g' -i /opt/exllama/setup.py \
    \
    && cd /opt/exllama \
    && python3 setup.py --verbose bdist_wheel --dist-dir /opt \
    \
    && pip3 install --no-cache-dir --verbose /opt/exllama*.whl \
    \
    && pip3 show exllama \
    && python3 -c 'import exllama'
