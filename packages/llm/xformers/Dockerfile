#---
# name: xformers
# group: llm
# config: config.py
# depends: [pytorch]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/facebookresearch/xformers
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG TORCH_CUDA_ARCH_LIST

ADD https://pypi.org/pypi/xformers/json /tmp/xformers_version.json

RUN set -ex \
    && pip3 list \
    && python3 -c 'import setuptools' \
    # Fix issue with overriden `pytorch` when installing `xformers` \
    # We install with `--no-deps` flag, because we already have all of \
    # `xformers` dependencies installed (numpy, pytorch). \
    && XFORMERS_MORE_DETAILS=1 MAX_JOBS=8 pip3 install --force-reinstall --no-cache-dir --verbose --no-deps 'xformers!=0.0.21' \
    && pip3 show xformers \
    && python3 -c 'import xformers.info; xformers.info.print_info()'
