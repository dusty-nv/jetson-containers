#---
# name: nanoowl
# group: vit
# depends: [pytorch, torch2trt, transformers, opencv]
# requires: '>=34.1.0'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

# 1. Install the dependencies
RUN pip3 install git+https://github.com/openai/CLIP.git
RUN pip3 install matplotlib

# 2. Install the NanoOWL Python package
RUN git clone https://github.com/NVIDIA-AI-IOT/nanoowl && \
    cd nanoowl && \
    git checkout main && \
    python3 setup.py develop --user

# 3. Build the TensorRT engine for the OWL-ViT vision encode
RUN cd /opt/nanoowl && \
    mkdir data && \
    python3 -m nanoowl.build_image_encoder_engine \
        data/owl_image_encoder_patch32.engine \
        --onnx_opset=16 && \
    ls -l data/

# 4. Run an example prediction to ensure everything is working
RUN cd /opt/nanoowl/examples/ && \
    python3 owl_predict.py \
        --prompt="[an owl, a glove]" \
        --threshold=0.1 \
        --image_encoder_engine=../data/owl_image_encoder_patch32.engine

COPY benchmark.py /opt/nanoowl/

WORKDIR /opt/nanoowl