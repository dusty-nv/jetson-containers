#!/usr/bin/env bash
# triton
set -ex

echo "============ Building triton ${TRITON_VERSION} (branch=${TRITON_BRANCH}) ============"

# Check for missing dependencies
apt-get update && apt-get install -y --no-install-recommends zlib1g-dev

uv pip uninstall triton

git clone --branch ${TRITON_BRANCH} --depth=1 --recursive https://github.com/triton-lang/triton /opt/triton
cd /opt/triton

#git checkout ${TRITON_BRANCH}
#git -C third_party submodule update --init nvidia || git submodule update --init --recursive

sed -i \
    -e 's|LLVMAMDGPUCodeGen||g' \
    -e 's|LLVMAMDGPUAsmParser||g' \
    -e 's|-Werror|-Wno-error|g' \
    CMakeLists.txt

sed -i 's|^download_and_copy_ptxas|#&|' python/setup.py || :

mkdir -p third_party/cuda
ln -sf /usr/local/cuda/bin/ptxas $(pwd)/third_party/cuda/ptxas

uv build --wheel --out-dir /opt .

cd /
rm -rf /opt/triton

uv pip install /opt/triton*.whl

uv pip show triton
python3 -c 'import triton'

twine upload --verbose /opt/triton*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
