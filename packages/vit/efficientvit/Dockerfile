#---
# name: efficientvit
# group: vit
# depends: [pytorch:2.0-distributed, torchvision, opencv, transformers, onnx, sam, jupyterlab]
# requires: '>=34.1.0'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

# Clone the repository:
RUN git clone --branch dev-python-3.8 https://github.com/tokk-nv/efficientvit.git && \
    cd efficientvit && \
    sed 's|^torch.*||g' -i requirements.txt && \
    sed 's|^opencv-python||g' -i requirements.txt && \
    sed 's|^transformers||g' -i requirements.txt && \
    sed 's|^onnx$||g' -i requirements.txt && \
    sed 's|^onnxruntime$||g' -i requirements.txt && \
    sed 's|^git+https://github.com/facebookresearch/segment-anything.git||g' -i requirements.txt && \
    cat requirements.txt && \
    pip3 install -r requirements.txt 

# RUN cd /opt/efficientvit && \
#     pip3 install .
    
COPY benchmark.py /opt/efficientvit

WORKDIR /opt/efficientvit
