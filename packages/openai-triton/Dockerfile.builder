#---
# name: openai-triton:builder
# group: ml
# depends: [build-essential, cmake, python, pytorch]
# config: config.py
# requires: '>=35'
# test: test.py
# notes: The OpenAI `triton` (https://github.com/openai/triton) wheel that's built is saved in the container under `/opt`. Based on https://cloud.tencent.com/developer/article/2317398, https://zhuanlan.zhihu.com/p/681714973, https://zhuanlan.zhihu.com/p/673525339
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas" \
    TRITON_CUOBJDUMP_PATH="$CUDA_HOME/bin/cuobjdump" \
    TRITON_NVDISASM_PATH="$CUDA_HOME/bin/nvdisasm"

ADD https://api.github.com/repos/openai/triton/git/refs/heads/main /tmp/triton_version.json

RUN set -ex \
    && git clone --depth=1 https://github.com/openai/triton /opt/triton \
    && git -C /opt/triton/third_party submodule update --init nvidia \
    && sed -i \
        -e 's|LLVMAMDGPUCodeGen||g' \
        -e 's|LLVMAMDGPUAsmParser||g' \
        -e 's|-Werror|-Wno-error|g' \
        /opt/triton/CMakeLists.txt \
    && pip3 wheel --wheel-dir=/opt --no-deps --verbose /opt/triton/python \
    && rm -rf /opt/triton \
    \
    && pip3 install --no-cache-dir --verbose /opt/triton*.whl \
    \
    && pip3 show triton \
    && python3 -c 'import triton'
