#---
# name: huggingface_hub
# group: llm
# depends: [python]
# test: test.py
# notes: provides `huggingface-cli` and `huggingface-downloader` tools
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# install huggingface_hub package (with CLI)
RUN pip3 install --no-cache-dir --verbose huggingface_hub[cli]

# ModuleNotFoundError: No module named 'dataclasses' (on JetPack 4)
RUN pip3 install --no-cache-dir --verbose dataclasses

# set the model cache dir
ENV HUGGINGFACE_HUB_CACHE=/data/models/huggingface
ENV HF_HOME=/data/models/huggingface

# add downloader tool
COPY huggingface-downloader /usr/local/bin/
COPY huggingface-downloader.py /usr/local/bin/_huggingface-downloader.py

# make sure it loads
RUN huggingface-cli --help && \
    huggingface-downloader --help && \
    pip3 show huggingface_hub && \
    python3 -c 'import huggingface_hub; print(huggingface_hub.__version__)'
    
# for benchmark timing
RUN apt-get update && \
    apt-get install -y --no-install-recommends time \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean