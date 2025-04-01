#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${FRUITNERF_VERSION} --depth=1 --recursive https://github.com/johnnynunez/FruitNeRF /opt/fruitnerf || \
git clone --depth=1 --recursive https://github.com/johnnynunez/FruitNeRF /opt/fruitnerf

# Navigate to the directory containing fruitnerf's setup.py
cd /opt/fruitnerf 
mkdir segmentation && cd segmentation
git clone --recursive https://github.com/IDEA-Research/Grounded-Segment-Anything.git grounded_sam
cd grounded_sam

wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth

pip3 install -e segment_anything 
pip3 install --no-build-isolation -e GroundingDINO
pip3 install --upgrade diffusers[torch]
pip3 install -U pycocotools matplotlib ipykernel #opencv-python onnx onnxruntime-gpu 
pip3 install -U segment-anything-hq

cd /opt/fruitnerf

pip3 install -e .

