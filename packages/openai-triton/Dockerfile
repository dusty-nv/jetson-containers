#---
# name: openai-triton
# group: ml
# depends: [build-essential, cmake, python, pytorch]
# config: config.py
# requires: '>=35'
# test: test.py
# notes: The OpenAI `triton` (https://github.com/openai/triton) wheel that's built is saved in the container under `/opt`. Based on https://cloud.tencent.com/developer/article/2317398, https://zhuanlan.zhihu.com/p/681714973, https://zhuanlan.zhihu.com/p/673525339
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG OPENAITRIRON_VERSION \
    FORCE_BUILD=off

COPY build.sh install.sh /tmp/openai-triton/

RUN /tmp/openai-triton/install.sh || /tmp/openai-triton/build.sh

# ARG BASE_IMAGE
# ARG BUILD_IMAGE

# FROM ${BUILD_IMAGE} as builder
# FROM ${BASE_IMAGE} as runtime

# COPY --from=builder /opt/triton*.whl /opt/

# ENV TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas" \
#     TRITON_CUOBJDUMP_PATH="$CUDA_HOME/bin/cuobjdump" \
#     TRITON_NVDISASM_PATH="$CUDA_HOME/bin/nvdisasm"

# RUN set -ex \
#     && pip3 install --no-cache-dir --verbose /opt/triton*.whl \
#     \
#     && pip3 show triton \
#     && python3 -c 'import triton'
