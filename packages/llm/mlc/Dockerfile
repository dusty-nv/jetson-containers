#---
# name: mlc
# group: llm
# config: config.py
# depends: [transformers]
# requires: '>=34.1.0'
# test: [test.py, test.sh]
# docs: docs.md
#---
ARG BASE_IMAGE
ARG BUILD_IMAGE

FROM ${BUILD_IMAGE} as build
FROM ${BASE_IMAGE} as deploy

ARG BUILD_IMAGE
RUN env
RUN echo "1. BASE_IMAGE is ${BASE_IMAGE}"
RUN echo "2. BUILD_IMAGE is ${BUILD_IMAGE}"


RUN env
RUN echo "3. BUILD_IMAGE is ${BUILD_IMAGE}"

RUN echo "4. BUILD_IMAGE is ${BUILD_IMAGE}"

WORKDIR /opt

COPY --from=build /opt/tvm*.whl /opt/
COPY --from=build /opt/mlc*.whl /opt/

RUN pip3 install --no-cache-dir --verbose /opt/tvm*.whl
RUN pip3 install --no-cache-dir --verbose /opt/mlc*.whl

#ENV TVM_HOME=/opt/mlc-llm/3rdparty/tvm
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/tvm:${LD_LIBRARY_PATH}"

RUN pip3 show mlc_llm && \
    python3 -m mlc_llm.build --help && \
    python3 -c "from mlc_chat import ChatModule; print(ChatModule)"

COPY benchmark.py mlc-llm/

WORKDIR /
