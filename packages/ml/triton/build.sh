#!/usr/bin/env bash
# triton
set -ex

echo "============ Building triton ${TRITON_VERSION} (branch=${TRITON_BRANCH}) ============"

pip3 uninstall -y triton

git clone --branch ${TRITON_BRANCH} --depth=1 --recursive https://github.com/triton-lang/triton /opt/triton || { rm -rf /opt/triton; git clone --depth=1 --recursive https://github.com/triton-lang/triton /opt/triton; }
cd /opt/triton

#git checkout ${TRITON_BRANCH}
#git -C third_party submodule update --init nvidia || git submodule update --init --recursive

sed -i \
    -e 's|LLVMAMDGPUCodeGen||g' \
    -e 's|LLVMAMDGPUAsmParser||g' \
    -e 's|-Werror|-Wno-error|g' \
    CMakeLists.txt

if [[ $CUDA_INSTALLED_VERSION -ge 130 ]]; then
  git apply -p1 /tmp/triton/cu130.diff
else
  sed -i 's|^download_and_copy_ptxas|#&|' python/setup.py || :
  mkdir -p third_party/cuda
  ln -sf /usr/local/cuda/bin/ptxas $(pwd)/third_party/cuda/ptxas
fi

cat /opt/triton/pyproject.toml

# Build the wheel - this will trigger CMake build
echo "=== BUILDING FIRST WHEEL (triggers CMake) ==="
pip3 wheel --wheel-dir=/opt --no-deps ./python || pip3 wheel --wheel-dir=/opt --no-deps .

# Show what was built in the first attempt
echo "=== FIRST BUILD RESULTS ==="
echo "Build directory contents:"
ls -la build/ 2>/dev/null || echo "No build directory found"
echo ""
echo "Python build directories:"
find build -name "*cpython*" -type d 2>/dev/null || echo "No cpython directories found"
echo ""
echo "Triton module files:"
find build -name "triton" -type d 2>/dev/null || echo "No triton module directory found"
echo ""
echo "Wheel files created:"
ls -la /opt/triton*.whl 2>/dev/null || echo "No wheel files found"

# Fix build directory naming convention AFTER wheel creation
# CMake creates: build/lib.linux-aarch64-cpython-3.12
# Setuptools expects: build/lib.linux-aarch64-cpython-312
echo ""
echo "=== FIXING DIRECTORY NAMING CONVENTION ==="
for dir in build/lib.linux-aarch64-cpython-3.*; do
    if [[ -d "$dir" ]]; then
        new_dir="${dir/3./3}"
        echo "Renaming: $dir -> $new_dir"
        mv "$dir" "$new_dir"
    fi
done

# Show the corrected directory structure
echo ""
echo "=== CORRECTED DIRECTORY STRUCTURE ==="
echo "Build directory contents after rename:"
ls -la build/ 2>/dev/null || echo "No build directory found"
echo ""
echo "Python build directories after rename:"
find build -name "*cpython*" -type d 2>/dev/null || echo "No cpython directories found"
echo ""
echo "Triton module files after rename:"
find build -name "triton" -type d 2>/dev/null || echo "No triton module directory found"

# Rebuild the wheel with the corrected directory structure
echo ""
echo "=== REBUILDING WHEEL WITH CORRECTED STRUCTURE ==="
pip3 wheel --wheel-dir=/opt --no-deps ./python || pip3 wheel --wheel-dir=/opt --no-deps .

# Show final results
echo ""
echo "=== FINAL BUILD RESULTS ==="
echo "Final wheel files:"
ls -la /opt/triton*.whl 2>/dev/null || echo "No wheel files found"
echo ""
echo "Wheel contents (first wheel):"
if ls /opt/triton*.whl 1>/dev/null 2>&1; then
    unzip -l /opt/triton*.whl | head -20
else
    echo "No wheel files to inspect"
fi

cd /
rm -rf /opt/triton

pip3 install /opt/triton*.whl

pip3 show triton
python3 -c 'import triton'

twine upload --verbose /opt/triton*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"