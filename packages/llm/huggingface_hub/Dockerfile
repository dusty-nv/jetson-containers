#---
# name: huggingface_hub
# group: llm
# depends: [python]
# test: test.py
# notes: provides `huggingface-cli` and `huggingface-downloader` tools
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# ModuleNotFoundError: No module named 'dataclasses' (on JetPack 4)
RUN pip3 install --no-cache-dir --verbose huggingface_hub[cli] && \
    pip3 install --no-cache-dir --verbose dataclasses

# set the model cache dir
ENV TRANSFORMERS_CACHE=/data/models/huggingface \
    HUGGINGFACE_HUB_CACHE=/data/models/huggingface \
    HF_HOME=/data/models/huggingface

# add downloader tool
COPY huggingface-downloader /usr/local/bin/
COPY huggingface-downloader.py /usr/local/bin/_huggingface-downloader.py

# make sure it loads
RUN huggingface-cli --help && \
    huggingface-downloader --help && \
    pip3 show huggingface_hub && \
    python3 -c 'import huggingface_hub; print(huggingface_hub.__version__)'
    