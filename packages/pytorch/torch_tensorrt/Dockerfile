#---
# name: torch_tensorrt
# group: pytorch
# config: config.py
# depends: [pytorch, torchvision, cudastack:standard, bazel]
# test: [test.py, test.sh]
# notes: https://pytorch.org/TensorRT/getting_started/installation.html#installation
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG JETPACK_MAJOR \
    JETPACK_MINOR \
    TORCH_TRT_VERSION=main

COPY WORKSPACE.jp46 /tmp

#
# https://github.com/pytorch/TensorRT/commit/9b0b8f1ccac81050c08091f74870092cbac91ac0
#   uv pip install pyyaml
#   python3 -m pip wheel --verbose --prerelease=allow . --config-setting="--jetpack-version=5.0" --config-settings="--use-cxx11-abi=1" -w dist && \
#
RUN uv pip install pyyaml && \
    git clone --branch=${TORCH_TRT_VERSION} --depth=1 https://github.com/pytorch/TensorRT torch_tensorrt || \
    git clone --depth=1 https://github.com/pytorch/TensorRT torch_tensorrt && \
    cd torch_tensorrt && \
    mkdir -p toolchains/jp_workspaces || echo "path already existed" && \
    cp /tmp/WORKSPACE.jp46 toolchains/jp_workspaces && \
    cp toolchains/jp_workspaces/WORKSPACE.jp${JETPACK_MAJOR}${JETPACK_MINOR} WORKSPACE || echo "skipping copy of WORKSPACE" && \
    cd py && head setup.py || cd ../ && \
    sed 's|^        \"torch >=.*|        \"torch\"|g' -i setup.py && \
    sed "s|^        'torch>=.*|        'torch',|g" -i setup.py && \
    sed 's|^    python_requires=.*|    python_requires=\">=3.6\",|g' -i setup.py && \
    cat setup.py && \
    #python3 setup.py install --jetpack-version ${JETPACK_MAJOR}.${JETPACK_MINOR} --use-cxx11-abi && \
    python3 setup.py bdist_wheel --jetpack-version ${JETPACK_MAJOR}.${JETPACK_MINOR} --use-cxx11-abi && \
    cp dist/torch_tensorrt*.whl /opt && \
    cd ../../ && \
    rm -rf torch_tensorrt

RUN uv pip install /opt/torch_tensorrt*.whl && \
    twine upload --verbose /opt/torch_tensorrt*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"

# ./torchtrtc: error while loading shared libraries: libtorch.so: cannot open shared object file
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib/python${PYTHON_VERSION}/dist-packages/torch/lib \
    PATH=${PATH}:/usr/local/lib/python${PYTHON_VERSION}/dist-packages/torch_tensorrt/bin
