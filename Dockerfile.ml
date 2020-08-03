# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

ARG BASE_IMAGE=nvcr.io/nvidia/l4t-base:r32.4.3
ARG PYTORCH_IMAGE
ARG TENSORFLOW_IMAGE

FROM ${PYTORCH_IMAGE} as pytorch
FROM ${TENSORFLOW_IMAGE} as tensorflow
FROM ${BASE_IMAGE}


# 
# setup environment
#
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME="/usr/local/cuda"
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV LLVM_CONFIG="/usr/bin/llvm-config-9"
ARG MAKEFLAGS=-j6

RUN printenv


#
# apt packages
#
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
          python3-pip \
		  python3-dev \
          python3-matplotlib \
		  build-essential \
		  gfortran \
		  git \
		  cmake \
		  libopenblas-dev \
		  liblapack-dev \
		  libblas-dev \
		  libhdf5-serial-dev \
		  hdf5-tools \
		  libhdf5-dev \
		  zlib1g-dev \
		  zip \
		  libjpeg8-dev \
		  libopenmpi2 \
          openmpi-bin \
          openmpi-common \
		  nodejs \
		  npm \
		  protobuf-compiler \
          libprotoc-dev \
		llvm-9 \
          llvm-9-dev \
    && rm -rf /var/lib/apt/lists/*


#
# python packages from TF/PyTorch containers
#
COPY --from=tensorflow /usr/local/lib/python2.7/dist-packages/ /usr/local/lib/python2.7/dist-packages/
COPY --from=tensorflow /usr/local/lib/python3.6/dist-packages/ /usr/local/lib/python3.6/dist-packages/

COPY --from=pytorch /usr/local/lib/python2.7/dist-packages/ /usr/local/lib/python2.7/dist-packages/
COPY --from=pytorch /usr/local/lib/python3.6/dist-packages/ /usr/local/lib/python3.6/dist-packages/


#
# python pip packages
#
RUN pip3 install pybind11 --ignore-installed 
RUN pip3 install onnx --verbose
RUN pip3 install scipy --verbose
RUN pip3 install scikit-learn --verbose
RUN pip3 install pandas --verbose
RUN pip3 install pycuda --verbose
RUN pip3 install numba --verbose


#
# restore missing cuDNN headers
#
#RUN ln -s /usr/include/aarch64-linux-gnu/cudnn_v8.h /usr/include/cudnn.h && \
#    ln -s /usr/include/aarch64-linux-gnu/cudnn_version_v8.h /usr/include/cudnn_version.h && \
#    ln -s /usr/include/aarch64-linux-gnu/cudnn_backend_v8.h /usr/include/cudnn_backend.h && \
#    ln -s /usr/include/aarch64-linux-gnu/cudnn_adv_infer_v8.h /usr/include/cudnn_adv_infer.h && \
#    ln -s /usr/include/aarch64-linux-gnu/cudnn_adv_train_v8.h /usr/include/cudnn_adv_train.h && \
#    ln -s /usr/include/aarch64-linux-gnu/cudnn_cnn_infer_v8.h /usr/include/cudnn_cnn_infer.h && \
#    ln -s /usr/include/aarch64-linux-gnu/cudnn_cnn_train_v8.h /usr/include/cudnn_cnn_train.h && \
#    ln -s /usr/include/aarch64-linux-gnu/cudnn_ops_infer_v8.h /usr/include/cudnn_ops_infer.h && \
#    ln -s /usr/include/aarch64-linux-gnu/cudnn_ops_train_v8.h /usr/include/cudnn_ops_train.h && \
#    ls -ll /usr/include/cudnn*

 
#
# CuPy
#
ARG CUPY_NVCC_GENERATE_CODE="arch=compute_53,code=sm_53;arch=compute_62,code=sm_62;arch=compute_72,code=sm_72"
ENV CUB_PATH="/opt/cub"
#ARG CFLAGS="-I/opt/cub"
#ARG LDFLAGS="-L/usr/lib/aarch64-linux-gnu"

RUN git clone https://github.com/NVlabs/cub opt/cub && \
    git clone -b v8.0.0b4 https://github.com/cupy/cupy cupy && \
    cd cupy && \
    pip3 install fastrlock && \
    python3 setup.py install --verbose && \
    cd ../ && \
    rm -rf cupy

#RUN pip3 install cupy --verbose


#
# JupyterLab
#
RUN pip3 install jupyter jupyterlab --verbose
#RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager@2

RUN jupyter lab --generate-config
RUN python3 -c "from notebook.auth.security import set_password; set_password('nvidia', '/root/.jupyter/jupyter_notebook_config.json')" 

CMD /bin/bash -c "jupyter lab --ip 0.0.0.0 --port 8888 --allow-root &> /var/log/jupyter.log" & \
	echo "allow 10 sec for JupyterLab to start @ http://localhost:8888 (password nvidia)" && \
	echo "JupterLab logging location:  /var/log/jupyter.log  (inside the container)" && \
	/bin/bash


