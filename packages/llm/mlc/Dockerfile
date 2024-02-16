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

FROM ${BUILD_IMAGE} as builder
FROM ${BASE_IMAGE} as runtime

WORKDIR /opt

# install the wheels from the development image
COPY --from=builder /opt/tvm*.whl /opt/
COPY --from=builder /opt/mlc*.whl /opt/

RUN pip3 install --no-cache-dir --verbose /opt/tvm*.whl
RUN pip3 install --no-cache-dir --verbose /opt/mlc*.whl

# we need the source because the MLC model builder relies on it
ARG MLC_REPO
ARG MLC_VERSION
ARG MLC_PATCH

RUN git clone https://github.com/${MLC_REPO} && \
    cd mlc-llm && \
    git checkout ${MLC_VERSION} && \
    git submodule update --init --recursive

COPY ${MLC_PATCH} mlc-llm/patch.diff
    
RUN cd mlc-llm && \
    if [ -s patch.diff ]; then git apply patch.diff; fi && \
    git status && \
    git diff --submodule=diff
    
# workarounds to resolve some various dependencies
ARG PYTHON_VERSION
RUN ln -s /opt/mlc-llm/3rdparty/tvm/3rdparty /usr/local/lib/python${PYTHON_VERSION}/dist-packages/tvm/3rdparty
#COPY --from=build /opt/mlc-llm/3rdparty/tvm/3rdparty/cutlass /usr/local/lib/python${PYTHON_VERSION}/dist-packages/tvm/3rdparty/cutlass
ENV LD_LIBRARY_PATH="/usr/local/lib/python${PYTHON_VERSION}/dist-packages/tvm:${LD_LIBRARY_PATH}"
ENV TVM_HOME=/opt/mlc-llm/3rdparty/tvm

# https://github.com/casper-hansen/AutoAWQ/issues/334
RUN pip3 install --no-cache-dir --verbose 'transformers<4.36'

# make sure it loads
RUN pip3 show mlc_llm && \
    python3 -m mlc_llm.build --help && \
    python3 -c "from mlc_chat import ChatModule; print(ChatModule)"

COPY benchmark.py mlc-llm/

WORKDIR /
