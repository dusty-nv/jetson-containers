#---
# name: stable_diffusion_cpp
# group: llm
# config: config.py
# depends: [cuda, cudastack:standard, cmake, python, numpy, huggingface_hub, sudonim]
# requires: '>=34.1.0'
# test: test.py
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG STABLE_DIFFUSION_VERSION \
    STABLE_DIFFUSION_VERSION_PY \
    STABLE_DIFFUSION_BRANCH \
    STABLE_DIFFUSION_BRANCH_PY \
    STABLE_DIFFUSION_FLAGS \
    CUDA_ARCHITECTURES \
    FORCE_BUILD=off \
    SOURCE_DIR=/opt/stable_diffusion_cpp_python \
    TMP=/tmp/stable_diffusion_cpp

COPY build.sh install.sh $TMP/

RUN $TMP/install.sh || $TMP/build.sh || true

RUN if [ ! -f "$TMP/.stable_diffusion_cpp" ]; then \
      echo "FAILED to install stable_diffusion.cpp $STABLE_DIFFUSION_VERSION"; \
      exit 1; \
    fi
