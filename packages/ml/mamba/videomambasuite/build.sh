#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${VIDEOMAMBASUITE_VERSION} --depth=1 --recursive https://github.com/OpenGVLab/video-mamba-suite /opt/videomambasuite || \
git clone --depth=1 --recursive https://github.com/OpenGVLab/video-mamba-suite /opt/videomambasuite

# Navigate to the directory containing mamba's setup.py
cd /opt/videomambasuite
uv pip install -r requirement.txt
