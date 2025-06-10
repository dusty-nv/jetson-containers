#---
# name: onnxruntime_genai
# group: ml
# config: config.py
# depends: [torch, torchvision, torchaudio, transformers, onnx, onnxruntime]
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG ONNXRUNTIME_GENAI_VERSION \
    ONNXRUNTIME_GENAI_BRANCH \
    CUDA_VERSION \
    FORCE_BUILD=off

COPY install.sh build.sh /tmp/onnxruntime_genai/

RUN /tmp/onnxruntime_genai/install.sh || /tmp/onnxruntime_genai/build.sh
