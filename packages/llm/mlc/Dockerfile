#---
# name: mlc
# group: llm
# config: config.py
# depends: [transformers]
# requires: '>=34.1.0'
# test: [test.py, test.sh]
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

ARG MLC_REPO
ARG MLC_VERSION
ARG MLC_PATCH

ARG CUDAARCHS
ARG TORCH_CUDA_ARCH_LIST
ARG LLVM_VERSION

# install LLVM the upstream way instead of apt because of:
# https://discourse.llvm.org/t/llvm-15-0-7-missing-libpolly-a/67942 
RUN wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh ${LLVM_VERSION} all && \
    ln -s /usr/bin/llvm-config-* /usr/bin/llvm-config

# could NOT find zstd (missing: zstd_LIBRARY zstd_INCLUDE_DIR)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		  libzstd-dev \
     && rm -rf /var/lib/apt/lists/* \
     && apt-get clean

# clone the sources
RUN git clone https://github.com/${MLC_REPO} && \
    cd mlc-llm && \
    git checkout ${MLC_VERSION} && \
    git submodule update --init --recursive

# apply patches to the source
COPY ${MLC_PATCH} mlc-llm/patch.diff

RUN cd mlc-llm && \
    if [ -s patch.diff ]; then git apply patch.diff; fi && \
    git status && \
    git diff --submodule=diff

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

# build TVM python module
RUN cd mlc-llm/3rdparty/tvm/python && \
    TVM_LIBRARY_PATH=/opt/mlc-llm/build/tvm \
    python3 setup.py --verbose bdist_wheel && \
    cp dist/tvm*.whl /opt && \
    rm -rf dist && \
    rm -rf build
  
RUN pip3 install --no-cache-dir --verbose tvm*.whl
RUN pip3 show tvm && python3 -c 'import tvm'

# build mlc-llm python module
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
