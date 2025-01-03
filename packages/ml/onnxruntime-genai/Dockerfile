#---
# name: onnxruntime-genai
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

COPY install.sh build.sh /tmp/onnxruntime-genai/

RUN /tmp/onnxruntime-genai/install.sh || /tmp/onnxruntime-genai/build.sh
