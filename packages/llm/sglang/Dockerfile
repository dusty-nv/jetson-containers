#---
# name: sglang
# alias: sgl
# group: llm
# depends: [torch, torchvision, torchaudio, transformers, triton]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="8.7" \
    MAX_JOBS=$(nproc) \
    CUDA_HOME=/usr/local/cuda-12.6/ \
    PATH="${CUDA_HOME}/bin:$PATH" \
    LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH"

# Install required system libraries
RUN apt-get update && \
    apt-get install -y libavformat58 libavfilter7 && \
    rm -rf /var/lib/apt/lists/*

# Build FlashInfer from Source
WORKDIR /workspace
RUN git clone -b v0.2.0 https://github.com/flashinfer-ai/flashinfer.git --recursive && \
    cd flashinfer && \
    python3 setup.py --verbose bdist_wheel --dist-dir ./ && \
    pip3 install flashinfer-0.2.0-py3-none-any.whl

# Build SGLang and SGL-kernel from Source
WORKDIR /workspace
RUN git clone --recurse-submodules -b v0.4.1.post7 https://github.com/sgl-project/sglang.git && \
    cd sglang && \
    sed -i '/sgl-kernel>=0.0.2.post14/d' python/pyproject.toml && \
    sed -i '/flashinfer==0.1.6/d' python/pyproject.toml && \
    sed -i '/xgrammar>=0.1.10/d' python/pyproject.toml && \
    cd sgl-kernel && \
    python3 setup.py bdist_wheel && pip3 install ./dist/*.whl && \
    cd .. && \
    sed -i '/return min(memory_values)/s/.*/        return None/' python/sglang/srt/utils.py  && \
    sed -i '/if not memory_values:/,+1d' python/sglang/srt/utils.py && \
    pip3 install -e "python[all]"

# Install additional Python packages
RUN pip3 install vllm decord --index-url https://pypi.jetson-ai-lab.dev/jp6/cu126

# Install compressed_tensors
RUN pip3 install compressed_tensors xgrammar

# Validate installations
RUN pip3 show sglang && \
    python3 -c 'import sglang'

# Set the working directory
WORKDIR /workspace/sglang

# Default command
CMD ["bash"]
