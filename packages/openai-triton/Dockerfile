#---
# name: openai-triton
# group: openai
# depends: [build-essential, cmake, python, pytorch]
# requires: '>=35'
# test: test.py
# notes: The openai-triton wheel that's built is saved in the container under /opt. Based on https://cloud.tencent.com/developer/article/2317398, https://zhuanlan.zhihu.com/p/681714973, https://zhuanlan.zhihu.com/p/673525339
#---

ARG BASE_IMAGE

FROM ${BASE_IMAGE}

ARG TRITON_DIR="/opt/triton" \
    TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas" \
    TRITON_CUOBJDUMP_PATH="$CUDA_HOME/bin/cuobjdump" \
    TRITON_NVDISASM_PATH="$CUDA_HOME/bin/nvdisasm"

ADD https://api.github.com/repos/openai/AutoAWQ/git/refs/heads/main /tmp/triton_version.json

# Install triton
# https://github.com/openai/triton
RUN set -ex \
    && git clone --depth=1 https://github.com/openai/triton $TRITON_DIR \
    && git -C "$TRITON_DIR/third_party" submodule update --init nvidia \
    && sed 's|LLVMAMDGPUCodeGen||g' -i $TRITON_DIR/CMakeLists.txt \
    && sed 's|LLVMAMDGPUAsmParser||g' -i $TRITON_DIR/CMakeLists.txt \
    && sed 's|-Werror|-Wno-error|g' -i $TRITON_DIR/CMakeLists.txt \
    && pip3 wheel --wheel-dir=/opt --no-deps --verbose $TRITON_DIR/python \
    && pip3 install --no-cache-dir --verbose /opt/triton*.whl \
    && rm -rf $TRITON_DIR

# test triton
RUN set -ex \
    && pip3 show triton \
    && python3 -c 'import triton'
