#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${PARTPACKER_VERSION} --depth=1 --recursive https://github.com/NVlabs/PartPacker /opt/partpacker  || \
git clone --depth=1 --recursive https://github.com/NVlabs/PartPacker /opt/partpacker

# Navigate to the directory containing partpacker's setup.py
cd /opt/partpacker

pip3 install -r requirements.txt

cd /opt/partpacker
mkdir pretrained
cd pretrained
wget https://huggingface.co/nvidia/PartPacker/resolve/main/vae.pt
wget https://huggingface.co/nvidia/PartPacker/resolve/main/flow.pt

# vae reconstruction of meshes
PYTHONPATH=. python vae/scripts/infer.py --ckpt_path pretrained/vae.pt --input assets/meshes/ --output_dir output/

# flow 3D generation from images
PYTHONPATH=. python flow/scripts/infer.py --ckpt_path pretrained/flow.pt --input assets/images/ --output_dir output/

# open local gradio app
python app.py
