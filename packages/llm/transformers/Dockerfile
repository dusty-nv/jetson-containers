#---
# name: transformers
# config: config.py
# category: llm
# depends: [pytorch, torchvision]
# test: test.py
# notes: bitsandbytes dependency added on JetPack5 for 4/8-bit
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# this is actually a simple install, however the tests are useful
RUN pip3 install --no-cache-dir --verbose transformers

# and it's a good way to make sure the model cache dir is mounted
ENV TRANSFORMERS_CACHE=/data/models/huggingface
