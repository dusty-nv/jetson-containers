#---
# name: huggingface_hub
# group: llm
# depends: [python]
# test: test.py
# notes: https://github.com/huggingface/huggingface_hub
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# set the model cache dir
ENV TRANSFORMERS_CACHE=/data/models/huggingface \
    HUGGINGFACE_HUB_CACHE=/data/models/huggingface \
    HF_HOME=/data/models/huggingface

# add downloader tool
COPY huggingface-downloader /usr/local/bin/
COPY huggingface-downloader.py /usr/local/bin/_huggingface-downloader.py

# install huggingface_hub package (with CLI)
RUN set -ex \
    # Fix: ModuleNotFoundError: No module named 'dataclasses' (on JetPack 4) \
    && uv pip install huggingface_hub \
    && python3 -c "import sys; sys.version_info < (3, 7) and __import__('subprocess').run(['uv pip', 'install', 'dataclasses'])" \
    && chmod a+x /usr/local/bin/huggingface-downloader /usr/local/bin/_huggingface-downloader.py \
    # make sure it loads \
    && hf --help \
    && huggingface-downloader --help \
    && uv pip show huggingface_hub \
    && python3 -c 'import huggingface_hub; print(huggingface_hub.__version__)' \
    \
    # for benchmark timing \
    && apt-get update \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
