#---
# name: nanosam
# group: vit
# depends: [pytorch, torchvision, torch2trt, transformers]
# requires: '>=34.1.0'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

# 1. Install the dependencies
#
# PyTorch and torch2trt are specified in the header yaml part (under "depends:")
#
RUN git clone https://github.com/NVIDIA-AI-IOT/trt_pose && \
    cd trt_pose && \
    echo "[build-system]" > pyproject.toml && \
    echo "requires = [\"setuptools>=40.8.0\", \"wheel\", \"torch\", \"torchvision\"]" >> pyproject.toml && \
    echo "build-backend = \"setuptools.build_meta\"" >> pyproject.toml && \
    uv pip install -e .

# 2. Install the NanoSAM Python package
RUN git clone https://github.com/NVIDIA-AI-IOT/nanosam && \
    cd nanosam && \
    python3 setup.py develop --user

ENV PYTHONPATH=${PYTHONPATH}:/opt/nanosam

# 3. Build the TensorRT engine for the mask decoder
RUN uv pip install timm

#RUN cd /opt/nanosam && \
#    mkdir data && \
#    python3 -m nanosam.tools.export_sam_mask_decoder_onnx \
#        --model-type=vit_t \
#        --checkpoint=assets/mobile_sam.pt \
#        --output=data/mobile_sam_mask_decoder.onnx

RUN mkdir -p /opt/nanosam/data && \
    wget $WGET_FLAGS \
	 https://nvidia.box.com/shared/static/ho09o7ohgp7lsqe0tcxqu5gs2ddojbis.onnx \
	 -O /opt/nanosam/data/mobile_sam_mask_decoder.onnx

RUN cd /opt/nanosam && \
    /usr/src/tensorrt/bin/trtexec \
        --onnx=data/mobile_sam_mask_decoder.onnx \
        --saveEngine=data/mobile_sam_mask_decoder.engine \
        --minShapes=point_coords:1x1x2,point_labels:1x1 \
        --optShapes=point_coords:1x1x2,point_labels:1x1 \
        --maxShapes=point_coords:1x10x2,point_labels:1x10

# 4. Build the TensorRT engine for the NanoSAM image encoder
RUN uv pip install gdown && \
    cd /opt/nanosam/data/ && \
#     gdown https://drive.google.com/uc?id=14-SsvoaTl-esC3JOzomHDnI9OGgdO2OR && \
    wget https://raw.githubusercontent.com/johnnynunez/nanosam/main/data/resnet18_image_encoder.onnx && \
    ls -lh && \
    cd /opt/nanosam/ && \
    /usr/src/tensorrt/bin/trtexec \
        --onnx=data/resnet18_image_encoder.onnx \
        --saveEngine=data/resnet18_image_encoder.engine \
        --fp16

# 5. Run the basic usage example
RUN uv pip install matplotlib
RUN cd /opt/nanosam/ && \
    python3 examples/basic_usage.py \
        --image_encoder=data/resnet18_image_encoder.engine \
        --mask_decoder=data/mobile_sam_mask_decoder.engine

COPY benchmark.py /opt/nanosam/
