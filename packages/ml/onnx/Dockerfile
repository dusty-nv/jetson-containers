#---
# name: onnx
# group: ml
# config: config.py
# depends: [cmake, python, numpy]
# test: test.py
# notes: https://github.com/onnx/onnx
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG ONNX_VERSION

RUN uv pip install onnx || \
    uv pip install git+https://github.com/onnx/onnx@${ONNX_VERSION} && \
    uv pip show onnx && \
    python3 -c 'import onnx; print(onnx.__version__)'

# error introduced by https://github.com/onnx/onnx/pull/5777 results in python protobuf mismatch
# https://github.com/onnx/onnx/blob/6ff456c1179c34827ad910e5601cb1486822d800/CMakeLists.txt#L249
#RUN set -ex \
#    && if [ `lsb_release --codename --short` != 'bionic' ]; then \
#        uv pip install protobuf==4.22.3; \
#    fi \
#    \
#    && uv pip install git+https://github.com/onnx/onnx@${ONNX_VERSION} \
#    && uv pip show onnx \
#    && python3 -c 'import onnx; print(onnx.__version__)'
