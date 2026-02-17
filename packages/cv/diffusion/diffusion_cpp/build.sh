#!/usr/bin/env bash
echo "Building stable-diffusion-python ${LLAMA_CPP_VERSION_PY}"

SOURCE_CPP=${SOURCE_DIR}/vendor/stable-diffusion.cpp
INSTALL_CPP=${SOURCE_DIR}/build/dist

set -ex

git clone --recursive --branch=${LLAMA_CPP_BRANCH_PY} https://github.com/william-murray1204/stable-diffusion-cpp-python ${SOURCE_DIR} || \
git clone --recursive https://github.com/william-murray1204/stable-diffusion-cpp-python ${SOURCE_DIR}

if [ -n "${LLAMA_CPP_BRANCH}" ]; then
    cd ${SOURCE_CPP}
fi

cd ${SOURCE_DIR}

FORCE_CMAKE=1 \
CMAKE_ARGS="${STABLE_DIFFUSION_FLAGS} -DSD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}" \
uv build --wheel --out-dir ${PIP_WHEEL_DIR} --verbose .

uv pip install ${PIP_WHEEL_DIR}/stable_diffusion_cpp_python*.whl
uv pip show stable-diffusion-cpp-python

twine upload --verbose ${PIP_WHEEL_DIR}/stable_diffusion_cpp_python*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"

# install c++ binaries
cd ${SOURCE_CPP}
mkdir -p build
cmake -B build -DSD_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_CPP}

cmake --build build --config Release --parallel $(nproc)
cmake --install build

# upload packages to apt server
tarpack upload stable-diffusion-cpp-${STABLE_DIFFUSION_VERSION} ${INSTALL_CPP} || echo "failed to upload tarball"
echo "installed" > "$TMP/.stable_diffusion_cpp"
cp -r ${INSTALL_CPP}/* /usr/local/

# create link to /opt/stable-diffusion.cpp
ln -s ${SOURCE_DIR}/vendor/stable-diffusion.cpp /opt/stable-diffusion.cpp
