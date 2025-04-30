#---
# name: ragflow-deps
# alias: ragflow-deps
# group: rag
# depends: [python, torch]
# requires: '>=34.1.0'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN set -ex && \
    cd /opt && \
    git clone https://github.com/infiniflow/ragflow.git && \
    cd ragflow/ && \
    uv run download_deps.py --china-mirrors && \
    mkdir -p /deps && \
    cp libssl1.1_1.1.1f-1ubuntu2_arm64.deb libssl1.1_1.1.1f-1ubuntu2_amd64.deb cl100k_base.tiktoken  tika-server-standard-3.0.0.jar tika-server-standard-3.0.0.jar.md5  /deps/ && \
    cp chrome* libssl*.deb / && \
    cp nltk_data /deps/nltk_data -r && \
    cp huggingface.co /huggingface.co -r && \
    ls -l /deps && \
    ls -l / && \
    echo "Files copied successfully." 

