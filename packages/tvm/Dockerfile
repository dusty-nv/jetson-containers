#---
# name: tvm
# group: ml
# depends: [pytorch, rust]
# requires: '>=34.1.0'
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

ARG CUDAARCHS
ARG TORCH_CUDA_ARCH_LIST

ARG TVM_REPO=apache/tvm
ARG TVM_BRANCH=main

# MLC/TVM recommends to use LLVM
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		  llvm \
		  llvm-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
    
#ADD https://api.github.com/repos/${TVM_REPO}/git/refs/heads/${TVM_BRANCH} /tmp/tvm_version.json

RUN git clone --branch=${TVM_BRANCH} --depth=1 --recursive https://github.com/${TVM_REPO}

# https://github.com/Dao-AILab/flash-attention/pull/343
RUN wget https://raw.githubusercontent.com/Dao-AILab/flash-attention/main/csrc/flash_attn/src/static_switch.h -O tvm/3rdparty/libflash_attn/src/static_switch.h && \
    sed 's|^set_target_properties(flash_attn PROPERTIES CUDA_ARCHITECTURES.*||' -i tvm/3rdparty/libflash_attn/src/CMakeLists.txt && \
    sed 's|^.*-gencode.*|\\|' -i tvm/3rdparty/libflash_attn/src/CMakeLists.txt && \
    cat tvm/3rdparty/libflash_attn/src/CMakeLists.txt
    
RUN mkdir tvm/build && \
    cd tvm/build && \
    cmake -G Ninja \
     -DCMAKE_CXX_STANDARD=17 \
	-DCMAKE_CUDA_STANDARD=17 \
	-DCMAKE_CUDA_ARCHITECTURES=${CUDAARCHS} \
	-DUSE_CUDA=ON \
	-DUSE_CUDNN=ON \
	-DUSE_CUBLAS=ON \
	-DUSE_CURAND=ON \
	-DUSE_CUTLASS=ON \
	-DUSE_THRUST=ON \
	-DUSE_TENSORRT_CODEGEN=ON \
	-DUSE_TENSORRT_RUNTIME=ON \
	-DUSE_GRAPH_EXECUTOR_CUDA_GRAPH=ON \
	-DUSE_TARGET_ONNX=ON \
	-DUSE_LLVM=ON \
	-DUSE_STACKVM_RUNTIME=ON \
	-DUSE_LIBTORCH=$(pip3 show torch | grep Location: | cut -d' ' -f2)/torch \
	-DCAFFE2_USE_CUDNN=1 \
	-DSUMMARIZE=ON \
	../ && \
    ninja && \
    rm -rf CMakeFiles
 
RUN cd tvm/python && \
    python3 setup.py --verbose bdist_wheel && \
    cp dist/tvm*.whl /opt && \
    rm -rf dist && \
    rm -rf build

RUN pip3 install --no-cache-dir --verbose tvm*.whl
    
WORKDIR /

RUN pip3 show tvm && python3 -c 'import tvm'

ENV TVM_HOME=/opt/tvm

RUN ln -s /opt/mlc-llm/3rdparty/tvm/3rdparty /usr/local/lib/python3.10/dist-packages/tvm/3rdparty
