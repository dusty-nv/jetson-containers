#!/bin/bash

#!/bin/bash

wget_if_not_exists() {
    local url="$1"
    local filename="$2"
    
    # If filename is not provided, extract it from the URL
    if [ -z "$filename" ]; then
        filename=$(basename "$url")
    fi
    
    if [ ! -f "$filename" ]; then
        echo "File $filename does not exist. Downloading..."
        wget "$url" -O "$filename"
        
        # Check if download was successful
        if [ $? -eq 0 ]; then
            echo "Successfully downloaded $filename"
            return 0
        else
            echo "Failed to download $filename"
            return 1
        fi
    else
        echo "File $filename already exists. Skipping download."
        return 0
    fi
}

wget_if_not_exists https://developer.download.nvidia.com/compute/cusparselt/0.6.3/local_installers/cusparselt-local-tegra-repo-ubuntu2204-0.6.3_1.0-1_arm64.deb
dpkg -i cusparselt-local-tegra-repo-ubuntu2204-0.6.3_1.0-1_arm64.deb
cp /var/cusparselt-local-tegra-repo-ubuntu2204-0.6.3/cusparselt-*-keyring.gpg /usr/share/keyrings/
apt-get update
apt-get -y install libcusparselt0 libcusparselt-dev

#wget_if_not_exists https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
#uv pip install torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl 

echo " ========================================================"
echo " Installing from https://pypi.jetson-ai-lab.dev/jp6/cu126" 
echo " ========================================================"
echo ""

# Add grep pip list | ctranslate2
echo " Installing ctranslate2 from https://pypi.jetson-ai-lab.dev/jp6/cu126" 
wget_if_not_exists  https://pypi.jetson-ai-lab.dev/jp6/cu126/+f/12e/c0ac6a30a6a2f/ctranslate2-4.5.0-cp310-cp310-linux_aarch64.whl
#wget_if_not_exists  /+f/6d2/9d09ec4904d72/ctranslate2-4.4.0-cp310-cp310-linux_aarch64.whl
uv pip install ctranslate2-4.4.0-cp310-cp310-linux_aarch64.whl 
#uv pip install ctranslate2-4.5.0-cp310-cp310-linux_aarch64.whl 

# Add grep pip list | torchaudio
echo ""
echo " Installing torchaudio from https://pypi.jetson-ai-lab.dev/jp6/cu126" 
wget_if_not_exists https://pypi.jetson-ai-lab.dev/jp6/cu126/+f/812/4fbc4ba6df0a3/torchaudio-2.5.0-cp310-cp310-linux_aarch64.whl
wget_if_not_exists https://pypi.jetson-ai-lab.dev/jp6/cu126/+f/a86/1895294d90440/torch-2.6.0rc1-cp310-cp310-linux_aarch64.whl
wget_if_not_exists https://pypi.jetson-ai-lab.dev/jp6/cu128/+f/406/faef6ad009ac1/torch-2.6.0-cp310-cp310-linux_aarch64.whl
uv pip install torchaudio-2.5.0-cp310-cp310-linux_aarch64.whl


wget_if_not_exists https://pypi.jetson-ai-lab.dev/jp6/cu126/+f/5f9/67f920de3953f/torchvision-0.20.0-cp310-cp310-linux_aarch64.whl
wget_if_not_exists https://pypi.jetson-ai-lab.dev/jp6/cu128/+f/0d9/ebbd08083f971/torchvision-0.21.0-cp310-cp310-linux_aarch64.whl
#wget_if_not_exists https://pypi.jetson-ai-lab.dev/jp6/cu126/+f/5f9/67f920de3953f/torchvision-0.20.0-cp310-cp310-linux_aarch64.whl
#up pip install torchvision-0.20.0-cp310-cp310-linux_aarch64.whl
# wget_if_not_exists https://pypi.jetson-ai-lab.dev/jp6/cu126/+f/0c4/18beb3326027d/onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl#sha256=0c418beb3326027d83acc283372ae42ebe9df12f71c3a8c2e9743a4e323443a

wget_if_not_exists https://pypi.jetson-ai-lab.dev/jp6/cu128/+f/e1a/c927d4536ffd2/torchaudio-2.6.0-cp310-cp310-linux_aarch64.whl

# For getting pth models
#curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
#sudo apt-get install git-lfs

# Get project root directory
PROJECT_ROOT=$(pwd)

# Set environment variables
export USE_GPU=true
export USE_ONNX=false
export PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/api
export MODEL_DIR=src/models
export VOICES_DIR=src/voices/v1_0
export WEB_PLAYER_PATH=$PROJECT_ROOT/web

# Install PyTorch
#wget https://pypi.jetson-ai-lab.dev/jp6/cu128/+f/406/faef6ad009ac1/torch-2.6.0-cp310-cp310-linux_aarch64.whl#sha256=406faef6ad009ac1607383f1369a638301e41f351428fa4bb0e3ad03ec57f6d7 # Install PyTorch
#mv torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl torch-2.5.0a0+872d972e41.nv24.8-cp310-cp310-linux_aarch64.whl && \
#uv pip install ./torch-2.5.0a0+872d972e41.nv24.8-cp310-cp310-linux_aarch64.whl 
#uv pip install ./torch-2.6.0-cp310-cp310-linux_aarch64.whl 
#rm torch-2.6.0-cp310-cp310-linux_aarch64.whl

# Run FastAPI with GPU extras using uv run
uv pip install -e ".[jetson]"
#uv pip install  ./torch-2.6.0-cp310-cp310-linux_aarch64.whl 

#uv pip install ./torch-2.5.0a0+872d972e41.nv24.8-cp310-cp310-linux_aarch64.whl 
#uv pip install torchvision torchaudio

uv run uvicorn api.src.main:app --reload --host 0.0.0.0 --port 8880
