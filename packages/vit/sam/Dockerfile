#---
# name: sam
# group: vit
# depends: [pytorch, torchvision, onnxruntime, opencv, jupyterlab]
# requires: '>=34.1.0'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

RUN uv pip install matplotlib pycocotools tqdm

# Clone the repository:
RUN git clone https://github.com/facebookresearch/segment-anything && \
    cd segment-anything && \
    sed 's|, "onnx.*"||' -i setup.py && \
    uv pip install .

ENV PYTHONPATH=${PYTHONPATH}:/opt/segment-anything


RUN mkdir -p /opt/segment-anything/benchmark
COPY benchmark.py /opt/segment-anything/benchmark

WORKDIR /opt/segment-anything/
