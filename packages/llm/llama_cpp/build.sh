#!/usr/bin/env bash
set -ex

SOURCE_CPP=${SOURCE_DIR}/vendor/llama.cpp

echo "Building llama-cpp-python ${LLAMA_CPP_VERSION_PY}"

git clone --recursive --branch=${LLAMA_CPP_BRANCH_PY} \
    https://github.com/abetlen/llama-cpp-python ${SOURCE_DIR}

if [ -n "${LLAMA_CPP_BRANCH}" ]; then
    cd ${SOURCE_CPP}
    git checkout ${LLAMA_CPP_BRANCH}
fi

cd ${SOURCE_DIR}

CMAKE_ARGS="${LLAMA_CPP_FLAGS} -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}" \
FORCE_CMAKE=1 \
pip3 wheel --wheel-dir=${PIP_WHEEL_DIR} --verbose .

pip3 install ${PIP_WHEEL_DIR}/llama_cpp_python*.whl
pip3 show llama-cpp-python

python3 -c 'import llama_cpp'
python3 -m llama_cpp.server --help

twine upload --verbose ${PIP_WHEEL_DIR}/llama_cpp_python*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"

# install c++ binaries
cd ${SOURCE_CPP}

cmake -B build ${LLAMA_CPP_FLAGS} -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}
cmake --build build --config Release --parallel $(nproc)
cmake --install build

ln -s ${SOURCE_DIR}/vendor/llama.cpp ${SOURCE_CPP}