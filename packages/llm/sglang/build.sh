#!/usr/bin/env bash
set -ex

pip3 install compressed-tensors decord
# Clone the repository if it doesn't exist
echo "FLASH INFER ${SGLANG_VERSION}"
git clone --branch=v${SGLANG_VERSION} --recursive --depth=1 https://github.com/flashinfer-ai/flashinfer /opt/flashinfer ||
git clone --recursive --depth=1 https://github.com/flashinfer-ai/flashinfer /opt/flashinfer
cd /opt/flashinfer

python3 setup.py --verbose bdist_wheel --dist-dir /opt/flashinfer/wheels/ && \
pip3 install --verbose /opt/flashinfer/wheels/flashinfer-*.whl


echo "SG LANG ${SGLANG_VERSION}"
cd /opt/
git clone --branch=v${SGLANG_VERSION} --recursive --depth=1 https://github.com/sgl-project/sglang /opt/sglang ||
git clone --recursive --depth=1 https://github.com/sgl-project/sglang /opt/sglang
cd /opt/sglang

sed -i '/sgl-kernel>=0.0.2.post14/d' python/pyproject.toml && \
sed -i '/flashinfer==0.1.6/d' python/pyproject.toml && \
sed -i '/xgrammar>=0.1.10/d' python/pyproject.toml && \
cd sgl-kernel && \
python3 setup.py bdist_wheel && pip3 install ./dist/*.whl && \
cd .. && \
sed -i '/return min(memory_values)/s/.*/        return None/' python/sglang_backup/srt/utils.py  && \
sed -i '/if not memory_values:/,+1d' python/sglang_backup/srt/utils.py && \
pip3 install -e "python[all]"

pip3 wheel -v --wheel-dir=/opt/sglang/wheels .
pip3 install --no-cache-dir --verbose /opt/sglang/wheels/sglang*.whl

cd /opt/sglang


# Optionally upload to a repository using Twine
twine upload --verbose /opt/flashinfer/wheels/flashinfer*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
twine upload --verbose /opt/sglang/wheels/sglang*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
