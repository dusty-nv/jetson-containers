#---
# name: llama_cpp
# group: llm
# config: config.py
# depends: [cuda, cudnn, cmake, python, numpy, huggingface_hub]
# requires: '>=34.1.0'
# test: test_version.py
# docs: docs.md
#---
ARG BASE_IMAGE
ARG BUILD_IMAGE

FROM ${BUILD_IMAGE} as builder
FROM ${BASE_IMAGE} as runtime

COPY --from=builder /opt/llama_cpp_python*.whl /opt/

RUN set -ex \
    && pip3 install --no-cache-dir --verbose \
        /opt/llama_cpp_python*.whl \
        typing-extensions \
        uvicorn \
        anyio \
        starlette \
        sse-starlette \
        starlette-context \
        fastapi \
        pydantic-settings \
    \
    && pip3 show llama-cpp-python | grep llama \
    && python3 -c 'import llama_cpp' \
    && python3 -m llama_cpp.server --help
