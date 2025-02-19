#---
# name: kokoro-tts:fastapi
# group: audio
# depends: [python, cuda, pytorch, transformers, torchaudio]
# test: [test.py]
# docs: docs.md
# notes: run with cuda12.8
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Install required dependencies
RUN apt-get update && \
    apt-get install -y libespeak-ng-dev portaudio19-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    libcgal-dev \
    libc-bin \
    ffmpeg

# Install astral-uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:$PATH"

# Install specific versions of Python packages
RUN pip3 install \
    scipy==1.15.1 \
    loguru==0.7.3 \
    num2words==0.5.14 \
    spacy==3.5.0 \
    phonemizer==3.3.0 \
    rfc3986==1.5.0 \
    IPython==8.32.0 \ 
    pydantic==1.10.21 \
    espeakng-loader


# Install Kokoro dependencies without dependencies
RUN pip3 install misaki --no-deps
RUN pip3 install kokoro --no-deps

WORKDIR /

# Clone repository
RUN git clone https://github.com/remsky/Kokoro-FastAPI 
RUN ls -ltr

RUN pwd

WORKDIR /Kokoro-FastAPI/docker/scripts

# Create models directory, download model, and restructure directories
RUN mkdir -p /Kokoro-FastAPI/api/src/models/v1_0 && \
    cd /Kokoro-FastAPI/docker/scripts && \
    python3 ./download_model.py --output /tmp/models && \
    mv /tmp/models/kokoro-v1_0.pth /Kokoro-FastAPI/api/src/models/v1_0/ && \
    mv /tmp/models/config.json /Kokoro-FastAPI/api/src/models/v1_0/ && \
    rm -rf /tmp/models

# Expose port
EXPOSE 8880

# Set environment variables
ENV PYTHONUNBUFFERED=1

RUN ls -ltr

# Set work directory to Kokoro-FastAPI
WORKDIR /Kokoro-FastAPI

RUN ls -ltr

RUN pwd

# Insert torch dependencies with online URLs before the last closing bracket
RUN sed -i '/phonemizer-fork/a\    "torch @ https:\/\/pypi.jetson-ai-lab.dev\/jp6\/cu128\/+f\/406\/faef6ad009ac1\/torch-2.6.0-cp310-cp310-linux_aarch64.whl",\n    "torchaudio @ https:\/\/pypi.jetson-ai-lab.dev\/jp6\/cu128\/+f\/e1a\/c927d4536ffd2\/torchaudio-2.6.0-cp310-cp310-linux_aarch64.whl",\n    "torchvision @ https:\/\/pypi.jetson-ai-lab.dev\/jp6\/cu128\/+f\/0d9\/ebbd08083f971\/torchvision-0.21.0-cp310-cp310-linux_aarch64.whl",' pyproject.toml

# Make gpu dependencies an empty array
RUN sed -i 's/gpu = \[.*\]/gpu = \[\]/' pyproject.toml
# Remove the torch CUDA dependency
RUN sed -i 's/"torch==2.6.0+cu124",//' pyproject.toml

# Comment out the torch sources section
RUN sed -i '/\[tool\.uv\.sources\]/,/\]/s/^/#/' pyproject.toml

# Replace the pytorch-cuda URL
RUN sed -i 's#url = ".*"#url = "https://pypi.jetson-ai-lab.dev/jp6/cu128"#' $(grep -l "name = \"pytorch-cuda\"" pyproject.toml)

# Make start script executable
RUN chmod +x ./start-gpu.sh

# Insert function and commands after shebang line
RUN sed -i '1a\\' ./start-gpu.sh
RUN sed -i '2a wget_if_not_exists() {' ./start-gpu.sh
RUN sed -i '3a     local url="$1"' ./start-gpu.sh  
RUN sed -i '4a     local filename="$2"' ./start-gpu.sh
RUN sed -i '5a\\' ./start-gpu.sh
RUN sed -i '6a     # If filename is not provided, extract it from the URL' ./start-gpu.sh
RUN sed -i '7a     if [ -z "$filename" ]; then' ./start-gpu.sh
RUN sed -i '8a         filename=$(basename "$url")' ./start-gpu.sh
RUN sed -i '9a     fi' ./start-gpu.sh
RUN sed -i '10a\\' ./start-gpu.sh
RUN sed -i '11a     if [ ! -f "$filename" ]; then' ./start-gpu.sh
RUN sed -i '12a         echo "File $filename does not exist. Downloading..."' ./start-gpu.sh
RUN sed -i '13a         wget "$url" -O "$filename"' ./start-gpu.sh
RUN sed -i '14a\\' ./start-gpu.sh
RUN sed -i '15a         # Check if download was successful' ./start-gpu.sh
RUN sed -i '16a         if [ $? -eq 0 ]; then' ./start-gpu.sh
RUN sed -i '17a             echo "Successfully downloaded $filename"' ./start-gpu.sh
RUN sed -i '18a             return 0' ./start-gpu.sh
RUN sed -i '19a         else' ./start-gpu.sh
RUN sed -i '20a             echo "Failed to download $filename"' ./start-gpu.sh
RUN sed -i '21a             return 1' ./start-gpu.sh
RUN sed -i '22a         fi' ./start-gpu.sh
RUN sed -i '23a     else' ./start-gpu.sh
RUN sed -i '24a         echo "File $filename already exists. Skipping download."' ./start-gpu.sh
RUN sed -i '25a         return 0' ./start-gpu.sh
RUN sed -i '26a     fi' ./start-gpu.sh
RUN sed -i '27a }' ./start-gpu.sh
RUN sed -i '28a wget_if_not_exists https://developer.download.nvidia.com/compute/cusparselt/0.6.3/local_installers/cusparselt-local-tegra-repo-ubuntu2204-0.6.3_1.0-1_arm64.deb' ./start-gpu.sh
RUN sed -i '29a dpkg -i cusparselt-local-tegra-repo-ubuntu2204-0.6.3_1.0-1_arm64.deb' ./start-gpu.sh
RUN sed -i '30a cp /var/cusparselt-local-tegra-repo-ubuntu2204-0.6.3/cusparselt-*-keyring.gpg /usr/share/keyrings/' ./start-gpu.sh
RUN sed -i '31a apt-get update' ./start-gpu.sh
RUN sed -i '32a apt-get -y install libcusparselt0 libcusparselt-dev' ./start-gpu.sh
RUN sed -i '33a #wget_if_not_exists https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl' ./start-gpu.sh
RUN sed -i '34a #uv pip install torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl' ./start-gpu.sh
RUN sed -i '35a echo " =========================================================="' ./start-gpu.sh
RUN sed -i '36a echo " Installing from https://pypi.jetson-ai-lab.dev/jp6/cu126"' ./start-gpu.sh
RUN sed -i '37a echo " =========================================================="' ./start-gpu.sh
RUN sed -i '38a echo ""' ./start-gpu.sh
RUN sed -i '39a # Add grep pip list | ctranslate2' ./start-gpu.sh
RUN sed -i '40a echo " Installing ctranslate2 from https://pypi.jetson-ai-lab.dev/jp6/cu128"' ./start-gpu.sh
RUN sed -i '41a #wget_if_not_exists  https://pypi.jetson-ai-lab.dev/jp6/cu126/+f/12e/c0ac6a30a6a2f/ctranslate2-4.5.0-cp310-cp310-linux_aarch64.whl' ./start-gpu.sh
RUN sed -i '42a wget_if_not_exists https://pypi.jetson-ai-lab.dev/jp6/cu128/+f/e87/07569ad725031/ctranslate2-4.5.0-cp310-cp310-linux_aarch64.whl' ./start-gpu.sh
RUN sed -i '43a #wget_if_not_exists  /+f/6d2/9d09ec4904d72/ctranslate2-4.4.0-cp310-cp310-linux_aarch64.whl' ./start-gpu.sh
RUN sed -i '44a #uv pip install ctranslate2-4.4.0-cp310-cp310-linux_aarch64.whl' ./start-gpu.sh
RUN sed -i '45a uv pip install ctranslate2-4.5.0-cp310-cp310-linux_aarch64.whl' ./start-gpu.sh

# Start the application
#CMD ["./start-gpu.sh"]
RUN ./start-gpu.sh