#---
# name: efficientvit
# group: vit
# depends: [pytorch, torchvision, opencv, transformers, onnx, sam, jupyterlab]
# requires: '>=34.1.0'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

# Clone the repository:
RUN git clone https://github.com/mit-han-lab/efficientvit.git && \
    cd efficientvit && \
    sed 's|^torch.*||g' -i requirements.txt && \
    sed 's|^opencv-python||g' -i requirements.txt && \
    sed 's|^transformers||g' -i requirements.txt && \
    sed 's|^onnx.*||g' -i requirements.txt && \
    sed 's|^git+https://github.com/facebookresearch/segment-anything.git||g' -i requirements.txt && \
    cat requirements.txt && \
    pip3 install -r requirements.txt 

# RUN cd /opt/efficientvit && \
#     pip3 install .
    
WORKDIR /opt/efficientvit