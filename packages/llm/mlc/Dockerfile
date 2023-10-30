#---
# name: mlc
# group: llm
# config: config.py
# depends: [transformers]
# requires: '>=34.1.0'
# test: [test.py]
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

ARG MLC_REPO
ARG MLC_VERSION

ARG CUDAARCHS
ARG TORCH_CUDA_ARCH_LIST

# MLC/TVM recommends to use LLVM
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		  llvm \
		  llvm-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
    
#ADD https://api.github.com/repos/${MLC_REPO}/git/refs/heads/${MLC_VERSION} /tmp/mlc_version.json

RUN git clone --recursive https://github.com/${MLC_REPO} && \
    cd mlc-llm && \
    git checkout ${MLC_VERSION}

# https://github.com/Dao-AILab/flash-attention/pull/343
RUN cd mlc-llm/3rdparty/tvm/3rdparty/libflash_attn && \
    wget https://raw.githubusercontent.com/Dao-AILab/flash-attention/main/csrc/flash_attn/src/static_switch.h -O src/static_switch.h && \
    sed 's|^set_target_properties(flash_attn PROPERTIES CUDA_ARCHITECTURES.*||' -i src/CMakeLists.txt && \
    sed 's|^.*-gencode.*|\\|' -i src/CMakeLists.txt && \
    cat src/CMakeLists.txt

# mlc currently uses it's own fork of tvm-unity (relax)
RUN sed 's|set(BUILD_DUMMY_LIBTVM ON)||' -i mlc-llm/CMakeLists.txt && \
    sed 's|set(USE_GRAPH_EXECUTOR OFF)||' -i mlc-llm/CMakeLists.txt && \
    cat mlc-llm/CMakeLists.txt
    
# disable pytorch: https://github.com/apache/tvm/issues/9362
RUN mkdir mlc-llm/build && \
    cd mlc-llm/build && \
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
	-DUSE_GRAPH_EXECUTOR_CUDA_GRAPH=ON \
	-DUSE_STACKVM_RUNTIME=ON \
	-DUSE_LLVM="/usr/bin/llvm-config --link-static" \
	-DHIDE_PRIVATE_SYMBOLS=ON \
	#-DUSE_LIBTORCH=$(pip3 show torch | grep Location: | cut -d' ' -f2)/torch \
	-DSUMMARIZE=ON \
	../ && \
    ninja && \
    rm -rf CMakeFiles && \
    rm -rf tvm/CMakeFiles && \
    rm -rf tokenizers/CMakeFiles && \
    rm -rf tokenizers/release

RUN cd mlc-llm/3rdparty/tvm/python && \
    TVM_LIBRARY_PATH=/opt/mlc-llm/build/tvm \
    python3 setup.py --verbose bdist_wheel && \
    cp dist/tvm*.whl /opt && \
    rm -rf dist && \
    rm -rf build
  
RUN pip3 install --no-cache-dir --verbose tvm*.whl
RUN pip3 show tvm && python3 -c 'import tvm'

ARG MLC_PATCH
COPY ${MLC_PATCH} mlc-llm/patch.diff

RUN cd mlc-llm && \
    git apply patch.diff && \
    git status && \
    git diff

RUN cd mlc-llm && \
    python3 setup.py --verbose bdist_wheel && \
    cp dist/mlc*.whl /opt 
 
RUN cd mlc-llm/python && \
    python3 setup.py --verbose bdist_wheel && \
    cp dist/mlc*.whl /opt

RUN pip3 install --no-cache-dir --verbose mlc*.whl

RUN pip3 show mlc_llm && \
    python3 -m mlc_llm.build --help && \
    python3 -c "from mlc_chat import ChatModule; print(ChatModule)"
    
RUN ln -s /opt/mlc-llm/3rdparty/tvm/3rdparty $(pip3 show torch | grep Location: | cut -d' ' -f2)/tvm/3rdparty

ENV TVM_HOME=/opt/mlc-llm/3rdparty/tvm

COPY benchmark.py mlc-llm/

WORKDIR /
