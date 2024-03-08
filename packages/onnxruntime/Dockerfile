#---
# name: onnxruntime
# group: ml
# config: config.py
# depends: [cuda, cudnn, tensorrt, cmake, python, numpy, onnx, openai-triton]
# test: test.py
# notes: the onnxruntime-gpu wheel that's built is saved in the container under /opt
#---
ARG BASE_IMAGE

FROM ${BASE_IMAGE}

ARG ONNXRUNTIME_WHEEL \
    ONNXRUNTIME_WHEEL_FILENAME

RUN set -ex \
    # Download pip wheel from location above for your version of JetPack \
    && wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate \
        ${ONNXRUNTIME_WHEEL} -O /opt/${ONNXRUNTIME_WHEEL_FILENAME} \
    && pip3 install --no-cache-dir --verbose /opt/${ONNXRUNTIME_WHEEL_FILENAME} \
    # This test fails in build stage, we gonna perform it in the test stage \
    && python3 -c 'import onnxruntime; print(onnxruntime.__version__);'
