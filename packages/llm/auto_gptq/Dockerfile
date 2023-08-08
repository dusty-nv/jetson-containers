#---
# name: auto_gptq
# group: llm
# config: config.py
# depends: [pytorch, bitsandbytes, transformers]
# requires: '>=34.1.0'
# test: test.py
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG TORCH_CUDA_ARCH_LIST

# force rebuild on new git commits - https://stackoverflow.com/a/56945508
ADD https://api.github.com/repos/PanQiWei/AutoGPTQ/git/refs/heads/main /tmp/autogptq_version.json

RUN cd /opt && \
    git clone --depth=1 https://github.com/PanQiWei/AutoGPTQ.git && \
    cd AutoGPTQ && \
    python3 setup.py --verbose bdist_wheel && \
    cp dist/auto_gptq*.whl /opt

RUN pip3 install --no-cache-dir --verbose /opt/auto_gptq*.whl

RUN pip3 show auto-gptq && python3 -c 'import auto_gptq'

# for benchmark timing
RUN apt-get update && \
    apt-get install -y --no-install-recommends time \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean