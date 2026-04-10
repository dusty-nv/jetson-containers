#!/usr/bin/env bash
set -ex

echo "Building fastvideo ${FASTVIDEO_VERSION}"

git clone --branch=v${FASTVIDEO_VERSION} --depth=1 --recursive https://github.com/hao-ai-lab/FastVideo /opt/fastvideo || \
git clone --depth=1 --recursive https://github.com/hao-ai-lab/FastVideo /opt/fastvideo

# Build and install fastvideo-kernel (CUDA/C++ extensions) from source
cd /opt/fastvideo/fastvideo-kernel

MAX_JOBS=$(nproc) \
uv build --wheel . -v --no-deps --out-dir /opt/fastvideo/wheels/

uv pip install /opt/fastvideo/wheels/fastvideo_kernel*.whl

# Install the main fastvideo package
cd /opt/fastvideo

uv pip install --no-deps --no-build-isolation -e .

uv pip install \
    scipy six h5py requests \
    sentencepiece timm peft \
    accelerate pillow imageio imageio-ffmpeg einops \
    wandb loguru test-tube \
    tqdm pytest PyYAML protobuf \
    gradio moviepy flask flask_restful \
    aiohttp aiofiles cloudpickle omegaconf \
    gpustat remote-pdb \
    torchdata pyarrow datasets av \
    torchcodec ray ftfy \
    fastapi uvicorn

twine upload --verbose /opt/fastvideo/wheels/fastvideo_kernel*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
