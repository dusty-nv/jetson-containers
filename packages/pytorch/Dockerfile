#
# PyTorch - see config.py for build args
#
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive


#
# install prerequisites
#
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

RUN pip3 install --no-cache-dir --verbose onnx


#
# download and install the PyTorch wheel
#
ARG PYTORCH_URL
ARG PYTORCH_WHL

RUN wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate ${PYTORCH_URL} -O ${PYTORCH_WHL} && \
    pip3 install --no-cache-dir --verbose ${PYTORCH_WHL} && \
    rm ${PYTORCH_WHL}

RUN python3 -c 'import torch; print(f"PyTorch version: {torch.__version__}"); print(f"CUDA available:  {torch.cuda.is_available()}"); print(f"cuDNN version:   {torch.backends.cudnn.version()}"); print(torch.__config__.show());'


# patch for https://github.com/pytorch/pytorch/issues/45323
RUN PYTHON_ROOT=`pip3 show torch | grep Location: | cut -d' ' -f2` && \
    TORCH_CMAKE_CONFIG=$PYTHON_ROOT/torch/share/cmake/Torch/TorchConfig.cmake && \
    echo "patching _GLIBCXX_USE_CXX11_ABI in ${TORCH_CMAKE_CONFIG}" && \
    sed -i 's/  set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=")/  set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")/g' ${TORCH_CMAKE_CONFIG}


#
# if needed, patch PyTorch version string to be compliant with PEP 440
#
#RUN if [ -d "/usr/local/lib/python3.8/dist-packages/torch-2.0.0.nv23.05.dist-info" ]; then \
#     echo "patching PyTorch version string to be PEP 440 compliant..."; \
#	sed -i 's/2.0.0.nv23.05/2.0.0/g' /usr/local/lib/python3.8/dist-packages/torch/version.py; \
#	sed -i 's/2.0.0.nv23.05/2.0.0/g' /usr/local/lib/python3.8/dist-packages/torch-2.0.0.nv23.05.dist-info/METADATA; \
#	head /usr/local/lib/python3.8/dist-packages/torch/version.py; \
#	head /usr/local/lib/python3.8/dist-packages/torch-2.0.0.nv23.05.dist-info/METADATA; \
#    fi




