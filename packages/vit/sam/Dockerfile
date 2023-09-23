#---
# name: sam
# group: vit
# depends: [opencv, python, pytorch, torchvision, numpy, onnxruntime, jupyterlab]
# requires: '>=34.1.0'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

RUN pip3 install matplotlib pycocotools

# Clone the repository:
RUN git clone https://github.com/facebookresearch/segment-anything && \
    cd segment-anything && \
    sed 's|, "onnx.*"||' -i setup.py && \
    pip3 install .
