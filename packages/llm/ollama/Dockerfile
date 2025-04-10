#---
# name: ollama
# group: llm
# config: config.py
# depends: cuda
# requires: '>=34.1.0'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG OLLAMA_REPO \
    OLLAMA_BRANCH \
    GOLANG_VERSION \
    CMAKE_VERSION \
    JETPACK_VERSION \
    CUDA_VERSION_MAJOR \
    CMAKE_CUDA_ARCHITECTURES
    
ENV OLLAMA_HOST=0.0.0.0 \
    OLLAMA_MODELS=/data/models/ollama/models \
    PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/cuda/lib64:/usr/local/cuda/include:${LD_LIBRARY_PATH} \
    JETSON_JETPACK=${JETPACK_VERSION} \
    CUDA_VERSION_MAJOR=${CUDA_VERSION_MAJOR}

COPY nv_tegra_release /etc/nv_tegra_release

EXPOSE 11434

# @dusty-nv: this curl does not work to automatically rebuild on upstream changes,
# because the curl command gets cached in docker. Reverting to the ADD method.
#RUN curl -H "Authorization: token ${GITHUB_TOKEN}" \
#    -o /tmp/ollama_version.json \
#    https://api.github.com/repos/ollama/ollama/branches/${OLLAMA_BRANCH}

ADD https://api.github.com/repos/ollama/ollama/branches/${OLLAMA_BRANCH} /tmp/ollama_version.json

RUN curl -fsSL https://ollama.com/install.sh | sh

RUN ln -s /usr/bin/python3 /usr/bin/python

COPY start_ollama /

CMD /start_ollama && /bin/bash
