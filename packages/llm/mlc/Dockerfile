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

# install the wheels from the development image
COPY --from=builder /opt/tvm*.whl /opt/
COPY --from=builder /opt/mlc*.whl /opt/

RUN pip3 install --no-cache-dir --verbose /opt/tvm*.whl && \
    pip3 install --no-cache-dir --verbose /opt/mlc*.whl && \
    pip3 install --no-cache-dir --verbose 'pydantic>2'

# we need the source because the MLC model builder relies on it
ARG MLC_REPO
ARG MLC_VERSION
ARG MLC_PATCH

RUN cd /opt && \
    git clone https://github.com/${MLC_REPO} && \
    cd mlc-llm && \
    git checkout ${MLC_VERSION} && \
    git submodule update --init --recursive

COPY ${MLC_PATCH} /opt/mlc-llm/patch.diff
    
RUN cd /opt/mlc-llm && \
    if [ -s patch.diff ]; then git apply patch.diff; fi && \
    git status && \
    git diff --submodule=diff
    
# workarounds to resolve some various dependencies
ARG PYTHON_VERSION
RUN ln -s /opt/mlc-llm/3rdparty/tvm/3rdparty /usr/local/lib/python${PYTHON_VERSION}/dist-packages/tvm/3rdparty
ENV LD_LIBRARY_PATH="/usr/local/lib/python${PYTHON_VERSION}/dist-packages/tvm:${LD_LIBRARY_PATH}" \
    TVM_HOME=/opt/mlc-llm/3rdparty/tvm

# make sure it loads
RUN pip3 show mlc_llm && \
    python3 -m mlc_llm.build --help && \
    python3 -c "from mlc_chat import ChatModule; print(ChatModule)"

COPY benchmark.py /opt/mlc-llm/
