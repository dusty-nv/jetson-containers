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
RUN git clone https://github.com/mit-han-lab/efficientvit && \
    cd efficientvit && \
    sed -i 's/==/>=/g' requirements.txt && \
    sed -i 's/~=/>=/g' requirements.txt && \
    sed 's|^git+https://github.com/facebookresearch/segment-anything.git||g' -i requirements.txt && \
    cat requirements.txt && \
    uv pip install -r requirements.txt

# RUN cd /opt/efficientvit && \
#     uv pip install .

COPY benchmark.py /opt/efficientvit

WORKDIR /opt/efficientvit
