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

# https://github.com/facebookresearch/xformers/blob/main/docs/source/custom_parts/index.rst#triton-parts
# could not find a version that satisfies the requirement triton
#RUN pip3 install --no-cache-dir --verbose triton

RUN pip3 install --no-cache-dir --verbose xformers

RUN pip3 show xformers && python3 -c 'import xformers.info; xformers.info.print_info()'
