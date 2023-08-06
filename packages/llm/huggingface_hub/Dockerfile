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

# set the model cache dir
ENV TRANSFORMERS_CACHE=/data/models/huggingface

# add downloader tool
COPY huggingface-downloader /usr/local/bin/
COPY huggingface-downloader.py /usr/local/bin/_huggingface-downloader.py

# make sure it loads
RUN huggingface-cli --help && \
    huggingface-downloader --help && \
    pip3 show huggingface_hub && \
    python3 -c 'import huggingface_hub; print(huggingface_hub.__version__)'