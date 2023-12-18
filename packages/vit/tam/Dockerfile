#---
# name: tam
# group: vit
# depends: [sam]
# requires: '>=34.1.0'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

# Clone the repository:
RUN git clone https://github.com/gaomingqi/Track-Anything.git && \
    cd Track-Anything && \
    sed 's|^numpy||g' -i requirements.txt && \
    sed 's|^git+https://github.com/facebookresearch/segment-anything.git||g' -i requirements.txt && \
    sed 's|^opencv-python||g' -i requirements.txt && \
    pip3 install -r requirements.txt 

RUN LD_LIBRARY_PATH=/usr/local/lib/python3.8/dist-packages/opencv_python.libs/:$LD_LIBRARY_PATH

RUN cd /opt/Track-Anything && \
    sed 's|cuda:3|cuda:0|g' -i app.py

# Dependencies and Installation
#RUN git clone https://github.com/MCG-NKU/E2FGVI.git

RUN pip3 install  opencv-python==4.5.4.60

ENV LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.8/dist-packages/opencv_python.libs/

RUN python3 -c 'import torch; print(f"PyTorch version: {torch.__version__}"); print(f"CUDA available:  {torch.cuda.is_available()}"); print(f"cuDNN version:   {torch.backends.cudnn.version()}"); print(f"torch.distributed:   {torch.distributed.is_available()}"); print(torch.__config__.show());'

ENV CPP_EXTENSION_PY_FILE=/usr/local/lib/python3.8/dist-packages/torch/utils/cpp_extension.py
RUN sed "s|('Ampere', '8.0;8.6+PTX'),|('Ampere', '8.0;8.6+PTX'),('Ampere+Tegra', '8.7'),|" -i ${CPP_EXTENSION_PY_FILE}
RUN sed "s|'8.6', '8.9'|'8.6', '8.7', '8.9'|" -i ${CPP_EXTENSION_PY_FILE}
RUN sed -n 1729,1746p ${CPP_EXTENSION_PY_FILE}

RUN mim install mmcv --verbose --no-cache-dir

RUN cd /opt && ./opencv_install.sh

# https://github.com/dusty-nv/jetson-containers/issues/326
RUN pip3 install --no-cache-dir --verbose gradio==3.38.0

# patch mmengine and mmcv to not use torch.distributed
RUN PYTHON_ROOT=`pip3 show mmengine | grep Location: | cut -d' ' -f2` && \
    sed 's|-> torch_dist\.ReduceOp||' -i $PYTHON_ROOT/mmengine/dist/dist.py && \
    sed 's|digit_version(TORCH_VERSION)|(0,0,0)|' -i $PYTHON_ROOT/mmengine/model/wrappers/__init__.py && \
    sed 's|digit_version(TORCH_VERSION)|(0,0,0)|' -i $PYTHON_ROOT/mmengine/model/__init__.py && \
    cat $PYTHON_ROOT/mmengine/model/wrappers/__init__.py

# cache the checkpoints under /data
RUN mkdir -p /data/models/tam && \
    ln -s /data/models/tam /opt/Track-Anything/checkpoints
    
WORKDIR /

RUN cd /opt/Track-Anything && python3 app.py --help
CMD cd /opt/Track-Anything && python3 app.py