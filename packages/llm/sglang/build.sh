#!/usr/bin/env bash
set -ex

pip3 install compressed-tensors decord

REPO_URL="https://github.com/sgl-project/sglang"
REPO_DIR="/opt/sglang"

ARCH="$(uname -m)"
ARCH_SED="s|x86_64|$ARCH|g" 
PLATFORM="$ARCH-linux"

echo "Building SGLang ${SGLANG_VERSION} for ${PLATFORM}"

git clone --recursive --depth=1 --branch=v${SGLANG_VERSION} $REPO_URL $REPO_DIR ||
git clone --recursive --depth=1 $REPO_URL $REPO_DIR

cd $REPO_DIR

# Remove dependencies
sed -i '/sgl-kernel/d' python/pyproject.toml
sed -i '/flashinfer/d' python/pyproject.toml
sed -i '/xgrammar/d' python/pyproject.toml

sed -i $ARCH_SED sgl-kernel/src/sgl-kernel/__init__.py
sed -i $ARCH_SED sgl-kernel/setup.py

sed -i 's|options={.*| |g' sgl-kernel/setup.py
echo "Patched sgl-kernel/setup.py"
cat sgl-kernel/setup.py  

echo "Building SGL-KERNEL"
cd $REPO_DIR/sgl-kernel/
python3 setup.py --verbose bdist_wheel --dist-dir $PIP_WHEEL_DIR --plat $PLATFORM
pip3 install $PIP_WHEEL_DIR/sgl*.whl

cd $REPO_DIR

if test -f "python/sglang/srt/utils.py"; then
    sed -i '/return min(memory_values)/s/.*/        return None/' python/sglang/srt/utils.py
    sed -i '/if not memory_values:/,+1d' python/sglang/srt/utils.py
fi

# Install SGLang
cd $REPO_DIR/python

sed -i 's|"torchao.*"|"torchao"|g' pyproject.toml
sed -i 's|"sgl-kernel.*"|"sgl-kernel"|g' pyproject.toml
sed -i 's|"vllm.*"|"vllm"|g' pyproject.toml
sed -i 's|"torch=.*"|"torch"|g' pyproject.toml

echo "Patched $REPO_DIR/python/pyproject.toml"
cat pyproject.toml 

pip3 wheel '.[all]'
pip3 install $PIP_WHEEL_DIR/sglang*.whl

cd /

# Install Gemlite python packages
pip3 install gemlite

# Validate installations
pip3 show sglang
python3 -c 'import sglang'

# Optionally upload to a repository using Twine
twine upload --verbose $PIP_WHEEL_DIR/sgl*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"