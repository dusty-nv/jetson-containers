#!/usr/bin/env bash
echo "Building llama-cpp-python ${LLAMA_CPP_VERSION_PY}"

SOURCE_CPP=${SOURCE_DIR}/vendor/llama.cpp
INSTALL_CPP=${SOURCE_DIR}/build/dist

set -ex

git clone --recursive --branch=${LLAMA_CPP_BRANCH_PY} https://github.com/abetlen/llama-cpp-python ${SOURCE_DIR} || \
git clone --recursive https://github.com/abetlen/llama-cpp-python

if [ -n "${LLAMA_CPP_BRANCH}" ]; then
    cd ${SOURCE_CPP}
    git checkout ${LLAMA_CPP_BRANCH}
fi

cd ${SOURCE_DIR}

FORCE_CMAKE=1 \
CMAKE_ARGS="${LLAMA_CPP_FLAGS} -DLLAVA_BUILD=OFF -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}" \
uv build --wheel --out-dir ${PIP_WHEEL_DIR} --verbose .

uv pip install ${PIP_WHEEL_DIR}/llama_cpp_python*.whl
uv pip show llama-cpp-python

twine upload --verbose ${PIP_WHEEL_DIR}/llama_cpp_python*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"

# install c++ binaries
cd ${SOURCE_CPP}

cmake -B build ${LLAMA_CPP_FLAGS} \
    -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_CPP} \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_EXAMPLES=ON \
    -DLLAMA_BUILD_TESTS=OFF

cmake --build build --config Release --parallel $(nproc)
cmake --install build

# upload packages to apt server
tarpack upload llama-cpp-${LLAMA_CPP_VERSION} ${INSTALL_CPP} || echo "failed to upload tarball"
echo "installed" > "$TMP/.llama_cpp"
cp -r ${INSTALL_CPP}/* /usr/local/

# create link to /opt/llama.cpp
ln -s ${SOURCE_DIR}/vendor/llama.cpp /opt/llama.cpp
