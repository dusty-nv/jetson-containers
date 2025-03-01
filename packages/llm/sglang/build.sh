#!/usr/bin/env bash
set -ex

pip3 install compressed-tensors decord
# Clone the repository if it doesn't exist
echo "FLASH INFER ${SGLANG_VERSION}"
git clone --branch=v${SGLANG_VERSION} --recursive --depth=1 https://github.com/flashinfer-ai/flashinfer /opt/flashinfer ||
git clone --recursive --depth=1 https://github.com/flashinfer-ai/flashinfer /opt/flashinfer
cd /opt/flashinfer

export MAX_JOBS=$(nproc)
export TORCH_CUDA_ARCH_LIST="8.7"
export FLASHINFER_ENABLE_AOT=1
python3 setup.py --verbose bdist_wheel --dist-dir /opt/flashinfer/wheels/ && \
pip3 install --verbose /opt/flashinfer/wheels/flashinfer_python-*.whl

echo "Building SGLang ${SGLANG_VERSION}"
cd /opt/
git clone --branch=v${SGLANG_VERSION} --recursive --depth=1 https://github.com/sgl-project/sglang /opt/sglang ||
git clone --recursive --depth=1 https://github.com/sgl-project/sglang /opt/sglang
cd /opt/sglang

# Remove dependencies
sed -i '/sgl-kernel/d' python/pyproject.toml
sed -i '/flashinfer/d' python/pyproject.toml
sed -i '/xgrammar/d' python/pyproject.toml

echo "Building SGL-KERNEL"
cd /opt/sglang/sgl-kernel/
export SGL_KERNEL_ENABLE_BF16=1
python3 setup.py --verbose bdist_wheel --dist-dir /opt/sglang/sgl-kernel/wheels/ && \
pip3 install --verbose /opt/sglang/sgl-kernel/wheels/sgl_*.whl

cd /opt/sglang/
if test -f "python/sglang/srt/utils.py"; then
    sed -i '/return min(memory_values)/s/.*/        return None/' python/sglang/srt/utils.py
    sed -i '/if not memory_values:/,+1d' python/sglang/srt/utils.py
fi

# Install SGLang
# pip3 install --no-cache-dir -e "python[all]"
python3 setup.py --verbose bdist_wheel --dist-dir /opt/sglang/wheels/ && \
pip3 install --verbose /opt/sglang/wheels/sglang*.whl

# Install Gemlite python packages
pip3 install gemlite

# Validate installations
pip3 show sglang
python3 -c 'import sglang'

# Optionally upload to a repository using Twine
twine upload --verbose /opt/flashinfer/wheels/flashinfer_python*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
twine upload --verbose /opt/sglang/wheels/sglang*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"