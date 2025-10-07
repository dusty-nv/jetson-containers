#---
# name: triton
# group: ml
# depends: [pytorch]
# config: config.py
# requires: '>=35'
# test: test.py
# notes: The OpenAI `triton` (https://github.com/triton-lang/triton) wheel that's built is saved in the container under `/opt`. Based on https://cloud.tencent.com/developer/article/2317398, https://zhuanlan.zhihu.com/p/681714973, https://zhuanlan.zhihu.com/p/673525339
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG TRITON_VERSION \
    TRITON_BRANCH \
    FORCE_BUILD=off

ENV TRITON_PTXAS_PATH=${CUDA_HOME}/bin/ptxas
COPY build.sh install.sh /tmp/triton/

RUN /tmp/triton/install.sh || /tmp/triton/build.sh
