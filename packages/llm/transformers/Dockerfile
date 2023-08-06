#---
# name: transformers
# config: config.py
# group: llm
# depends: [pytorch, torchvision, huggingface_hub]
# test: test.py
# notes: bitsandbytes dependency added on JetPack5 for 4-bit/8-bit quantization
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# this is actually a simple install, however the tests are useful
RUN pip3 install --no-cache-dir --verbose transformers

# this now gets set in the huggingface-hub package
# ENV TRANSFORMERS_CACHE=/data/models/huggingface
