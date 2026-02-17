#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${sparc3d_VERSION} --depth=1 --recursive https://github.com/lizhihao6/Sparc3D /opt/sparc3d  || \
git clone --depth=1 --recursive https://github.com/lizhihao6/Sparc3D /opt/sparc3d

# Navigate to the directory containing sparc3d's setup.py
cd /opt/sparc3d

uv pip install -r requirements.txt

cd /opt/sparc3d

# open local gradio app
python app.py
