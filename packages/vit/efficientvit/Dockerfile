#---
# name: efficientvit
# group: vit
# depends: [pytorch, torchvision, opencv, transformers, onnxruntime, sam, jupyterlab]
# requires: '>=34.1.0'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

# Clone the repository:
RUN git clone --branch https://github.com/mit-han-lab/efficientvit && \
    cd efficientvit && \
    sed -i 's/==/>=/g' requirements.txt && \
    sed -i 's/~=/>=/g' requirements.txt && \
    sed 's|^git+https://github.com/facebookresearch/segment-anything.git||g' -i requirements.txt && \
    cat requirements.txt && \
    pip3 install -r requirements.txt

# RUN cd /opt/efficientvit && \
#     pip3 install .

COPY benchmark.py /opt/efficientvit

WORKDIR /opt/efficientvit
