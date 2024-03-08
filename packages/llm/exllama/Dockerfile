#---
# name: exllama:v1
# alias: exllama
# group: llm
# config: config.py
# depends: [pytorch, huggingface_hub]
# requires: '>=34.1.0'
# test: test.sh
# docs: docs.md
#---
ARG BASE_IMAGE

# Build Stage
FROM ${BASE_IMAGE} AS build

ARG TORCH_CUDA_ARCH_LIST
ARG EXLLAMA_REPO=jllllll/exllama
ARG EXLLAMA_BRANCH=master

# force rebuild on new git commits - https://stackoverflow.com/a/56945508
ADD https://api.github.com/repos/${EXLLAMA_REPO}/git/refs/heads/${EXLLAMA_BRANCH} /tmp/exllama_version.json

RUN set -ex \
    && git clone --branch=${EXLLAMA_BRANCH} --depth=1 https://github.com/${EXLLAMA_REPO} /opt/exllama \
    && sed 's|^torch.*|torch|g' -i /opt/exllama/requirements.txt \
    && sed 's|\[\"cublas\"\] if platform.system\(\) == \"Windows\" else \[\]|\[\"cublas\"\]|g' -i /opt/exllama/setup.py \
    # build the wheel \
    && cd /opt/exllama \
    && python3 setup.py --verbose bdist_wheel --dist-dir /opt
    # cannot uninstall 'blinker': It is a distutils installed project \
    # && pip3 install --ignore-installed --no-cache-dir blinker \
    # && pip3 install --no-cache-dir --verbose /opt/exllama*.whl \
    # make sure it loads \
    # && pip3 show exllama \
    # && python3 -c 'import exllama'
    # missing 'safetensors', but it's in the requirements.txt? \
    # && pip3 install --no-cache-dir --verbose \
    #     -r /opt/exllama/requirements.txt \
    #     -r /opt/exllama/requirements-web.txt \
    # this will build cuda_ext.py to ~/.cache/torch_extensions/ \
    # && cd exllama && python3 test_benchmark_inference.py --help

# Install Stage
FROM ${BASE_IMAGE} AS install

COPY --from=build /opt/exllama*.whl /opt/

RUN set -ex \
    && pip3 install --no-cache-dir --verbose /opt/exllama*.whl \
    # make sure it loads \
    && pip3 show exllama \
    && python3 -c 'import exllama'
