#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=${ZIGMA_VERSION} --depth=1 --recursive https://github.com/CompVis/zigma opt/zigma || \
git clone --depth=1 --recursive https://github.com/CompVis/zigma /opt/zigma

# Navigate to the directory containing mamba's setup.py
cd /opt/zigma 
pip3 install -U torchdiffeq matplotlib h5py timm diffusers accelerate loguru blobfile ml_collections wandb
pip3 install -U hydra-core torch-fidelity webdataset einops pytorch_lightning
pip3 install -U torchmetrics scikit-learn moviepy imageio numpy-hilbert-curve av

cd /opt/zigma
pip3 install 'numpy<2'