#---
# name: transformers
# config: config.py
# group: llm
# depends: [pytorch, torchvision, huggingface_hub]
# test: [test.py, huggingface-benchmark.py]
# docs: docs.md
# notes: bitsandbytes dependency added on JetPack5 for 4-bit/8-bit quantization
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# this is actually a simple install, however the tests are useful
RUN pip3 install --no-cache-dir --verbose transformers accelerate

# this now gets set in the huggingface-hub package
# ENV TRANSFORMERS_CACHE=/data/models/huggingface

# add benchmark script
COPY huggingface-benchmark.py /usr/local/bin

# for benchmark timing
RUN apt-get update && \
    apt-get install -y --no-install-recommends time \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean