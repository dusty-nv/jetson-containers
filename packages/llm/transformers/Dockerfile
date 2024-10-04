#---
# name: transformers
# config: config.py
# group: llm
# depends: [pytorch, torchvision, huggingface_hub, rust]
# test: [test_version.py, huggingface-benchmark.py]
# docs: docs.md
# notes: for quantization support in Transformers, use the bitsandbytes, AutoGPTQ, or AutoAWQ containers.
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG TRANSFORMERS_PACKAGE=transformers \
    TRANSFORMERS_VERSION=https://pypi.org/pypi/transformers/json

ADD ${TRANSFORMERS_VERSION} /tmp/transformers_version.json

# if you want optimum[exporters,onnxruntime] see the optimum package
RUN pip3 install --no-cache-dir --verbose accelerate && \
    pip3 install --no-cache-dir --verbose sentencepiece && \
    pip3 install --no-cache-dir --verbose optimum && \
    \
    # install from pypi, git, ect (sometimes other version got installed)
    pip3 uninstall -y transformers && \
    pip3 install --no-cache-dir --verbose ${TRANSFORMERS_PACKAGE} && \
    \
    # "/usr/local/lib/python3.8/dist-packages/transformers/modeling_utils.py", line 118
    # AttributeError: module 'torch.distributed' has no attribute 'is_initialized'
    PYTHON_ROOT=`pip3 show transformers | grep Location: | cut -d' ' -f2` && \
    sed -i 's|torch.distributed.is_initialized|torch.distributed.is_available|g' -i ${PYTHON_ROOT}/transformers/modeling_utils.py
    
# add benchmark script
COPY huggingface-benchmark.py /usr/local/bin
    
# make sure it loads
RUN pip3 show transformers && python3 -c 'import transformers; print(transformers.__version__)'
