#---
# name: pytorch
# alias: torch
# group: pytorch
# config: config.py
# depends: [python, numpy, onnx]
# test: test.py
# docs: |
#  Containers for PyTorch with CUDA support.
#  Note that the [`l4t-pytorch`](/packages/l4t/l4t-pytorch) containers also include PyTorch, torchvision, and torchaudio.
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

# install prerequisites
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		  libopenblas-dev \
		  libopenmpi-dev \
            openmpi-bin \
            openmpi-common \
		  gfortran \
		  libomp-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# download and install the PyTorch wheel
ARG PYTORCH_URL
ARG PYTORCH_WHL

RUN cd /opt && \
    wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate ${PYTORCH_URL} -O ${PYTORCH_WHL} && \
    pip3 install --verbose ${PYTORCH_WHL}

RUN python3 -c 'import torch; print(f"PyTorch version: {torch.__version__}"); print(f"CUDA available:  {torch.cuda.is_available()}"); print(f"cuDNN version:   {torch.backends.cudnn.version()}"); print(torch.__config__.show());'

# patch for https://github.com/pytorch/pytorch/issues/45323
RUN PYTHON_ROOT=`pip3 show torch | grep Location: | cut -d' ' -f2` && \
    TORCH_CMAKE_CONFIG=$PYTHON_ROOT/torch/share/cmake/Torch/TorchConfig.cmake && \
    echo "patching _GLIBCXX_USE_CXX11_ABI in ${TORCH_CMAKE_CONFIG}" && \
    sed -i 's/  set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=")/  set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")/g' ${TORCH_CMAKE_CONFIG}

# PyTorch C++ extensions frequently use ninja parallel builds
RUN pip3 install --no-cache-dir scikit-build && \
    pip3 install --no-cache-dir ninja
    
# set the torch hub model cache directory to mounted /data volume
ENV TORCH_HOME=/data/models/torch
