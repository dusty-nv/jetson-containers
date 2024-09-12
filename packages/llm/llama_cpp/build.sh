#!/usr/bin/env bash
set -ex

echo "Building llama-cpp-python ${LLAMA_CPP_VERSION}"
 
cd /opt
git clone --branch=v${LLAMA_CPP_BRANCH} --depth=1 --recursive https://github.com/abetlen/llama-cpp-python

CMAKE_ARGS="${LLAMA_CPP_FLAGS} -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}" \
FORCE_CMAKE=1 \
pip3 wheel --wheel-dir=/opt/wheels --verbose ./llama-cpp-python

pip3 install --no-cache-dir --verbose /opt/wheels/llama_cpp_python*.whl
pip3 show llama-cpp-python

python3 -c 'import llama_cpp'
python3 -m llama_cpp.server --help

twine upload --verbose /opt/wheels/llama_cpp_python*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"

# install c++ binaries
cd /opt/llama-cpp-python/vendor/llama.cpp

cmake -B build ${LLAMA_CPP_FLAGS} -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}
cmake --build build --config Release --parallel $(nproc)
cmake --install build

ln -s /opt/llama-cpp-python/vendor/llama.cpp /opt/llama.cpp