#---
# name: partpacker
# group: objects
# config: config.py
# depends: [cuda, pytorch, torchvision, triton, onnxruntime, opencv, transformers, diffusers, flash-attention, xformers, pymeshlab]
# requires: '>=34.1.0'
# notes: https://github.com/NVlabs/PartPacker
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG PARTPACKER_VERSION

COPY install.sh /tmp/partpacker/
RUN /tmp/partpacker/install.sh

WORKDIR /opt/partpacker
CMD ["python3", "-m", "app.py"]
