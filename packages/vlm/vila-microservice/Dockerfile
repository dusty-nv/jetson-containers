# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
FROM nvcr.io/nvidia/l4t-jetpack:r36.3.0

#initial set up
ENV DEBIAN_FRONTEND=noninteractive \
    LANGUAGE=en_US:en \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

# #Setup User
# RUN groupadd -g 150 logusers && \
#     groupadd -g 2009 genai && \
#     useradd -m -u 2009 -g genai genai && \
#     usermod -aG logusers genai
# #Setup data folder
# RUN mkdir /data && \
#     chown genai /data && \
#     chgrp genai /data

RUN set -ex
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip git-all lsb-release libopenblas0

#Set to get precompiled jetson wheels
RUN export PIP_INDEX_URL=http://jetson.webredirect.org/jp6/cu122
RUN export PIP_TRUSTED_HOST=jetson.webredirect.org

#install extra gstreamer plugins
RUN apt-get update && apt-get install -y --no-install-recommends \
          gstreamer1.0-rtsp \
          libgstrtspserver-1.0-0 \
          gstreamer1.0-plugins-rtp \
		  libgstreamer1.0-dev \
		  libgstreamer-plugins-base1.0-dev \
		  libgstreamer-plugins-good1.0-dev \
		  libgstreamer-plugins-bad1.0-dev \
		  python3-gi \
		  python3-gst-1.0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

#install dependecies to build jetson-utils
RUN pip3 install --upgrade --force-reinstall --no-cache-dir --verbose cmake

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		    libglew-dev \
		    glew-utils \
		    libsoup2.4-dev \
		    libjson-glib-dev \
		    libgstrtspserver-1.0-dev \
		    avahi-utils \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

#jetson utils
WORKDIR /opt
RUN git clone  --depth=1 https://github.com/dusty-nv/jetson-utils
WORKDIR /opt/jetson-utils
RUN mkdir build
WORKDIR /opt/jetson-utils/build
RUN cmake ../
RUN make -j$(nproc)
RUN make install
RUN ldconfig
RUN python3 -c 'import jetson_utils'

#Install precompiled ml packages
RUN pip3 install numpy==1.26.4
RUN pip3 install http://jetson.webredirect.org/jp6/cu122/+f/6c5/c648d0d96a592/torch-2.3.1-cp310-cp310-linux_aarch64.whl#sha256=6c5c648d0d96a5924d237eacb95fee98de1fce7ab2ec26064f31649d5e381218
RUN pip3 install http://jetson.webredirect.org/jp6/cu122/+f/28a/62174f5604aa1/torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl#sha256=28a62174f5604aa111f06c7d61f488250a3c8940fce05bbf245500b90b1d1486
RUN pip3 install http://jetson.webredirect.org/jp6/cu122/+f/54d/5094104195427/onnxruntime_gpu-1.17.0-cp310-cp310-linux_aarch64.whl#sha256=54d509410419542764627aebf6542734a7c5b4d260ad0860b650e05f38b1b47a
RUN pip3 install http://jetson.webredirect.org/jp6/cu122/+f/9a3/9fc36dc7cf169/cuda_python-12.2.0+0.g2ae98f9.dirty-cp310-cp310-linux_aarch64.whl#sha256=9a39fc36dc7cf1693686774c16c78a07274f609037627fa69f5d8159793e96cd
RUN pip3 install transformers
RUN pip3 install --no-cache-dir --verbose onnx

#install torch2trt
RUN cd /opt && \
    git clone --depth=1 https://github.com/NVIDIA-AI-IOT/torch2trt && \
    cd torch2trt && \
    python3 setup.py install && \
    sed 's|^set(CUDA_ARCHITECTURES.*|#|g' -i CMakeLists.txt && \
    cmake -B build -DCUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} . && \
    cmake --build build --target install && \
    ldconfig

#Install TVM & MLC
WORKDIR /opt
RUN pip3 install sentencepiece
RUN pip3 install --index-url http://jetson.webredirect.org/jp6/cu122 --trusted-host jetson.webredirect.org  --no-cache-dir --verbose tvm==0.15.0 mlc-llm==0.1.0


#Clone Jetson Containers to get mlc patch
RUN git clone https://github.com/dusty-nv/jetson-containers && \
    cd jetson-containers && \
    git checkout d5bc44e && \
    cp packages/llm/mlc/patches/607dc5a.diff /tmp/patch.diff && \
    cd /opt && \
    rm -rf jetson-containers

#Setup MLC and TVM. Need the source to build the models
RUN git clone https://github.com/mlc-ai/mlc-llm /opt/mlc-llm
WORKDIR /opt/mlc-llm
RUN git checkout 607dc5a
RUN git submodule update --init --recursive
RUN git apply /tmp/patch.diff
ENV TVM_HOME=/opt/mlc-llm/3rdparty/tvm
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/tvm:$LD_LIBRARY_PATH
RUN ln -s /opt/mlc-llm/3rdparty/tvm/3rdparty /usr/local/lib/python3.10/dist-packages/tvm/3rdparty

#install deps for nanollm

#Install NanoLLM dependecies
RUN pip3 install termcolor tabulate accelerate cachetools decorator-args inflect flask tqdm getch natsort pyyaml

#instal clip TRT
WORKDIR /opt
RUN git clone https://github.com/dusty-nv/clip_trt
WORKDIR /opt/clip_trt
RUN pip3 install -e .

#Install NanoLLM
WORKDIR /opt
RUN git clone https://github.com/dusty-nv/NanoLLM
WORKDIR /opt/NanoLLM
RUN git checkout 24.7
ENV PYTHONPATH=${PYTHONPATH}:/opt/NanoLLM

#Delete unused plugins
#RUN rm -rf /opt/NanoLLM/nano_llm/plugins/audio
#RUN sed -i '/from \.audio import AutoASR, AutoTTS, AudioOutputDevice, AudioOutputFile/d' /opt/NanoLLM/nano_llm/plugins/__init__.py
#RUN sed -i '/from \.nanodb import NanoDB/d' /opt/NanoLLM/nano_llm/plugins/__init__.py
RUN sed -i '/from \.database import */d' /opt/NanoLLM/nano_llm/plugins/__init__.py
#RUN sed -i '/from nanodb.utils import */d' /opt/NanoLLM/nano_llm/utils/__init__.py
RUN sed -i '/from nanodb.utils import */c\from clip_trt.utils import *' /opt/NanoLLM/nano_llm/utils/__init__.py
RUN sed -i '/from \.audio import */d' /opt/NanoLLM/nano_llm/utils/__init__.py
RUN sed -i '/from \.speech import */d' /opt/NanoLLM/nano_llm/plugins/__init__.py
RUN sed -i '/from \.audio import */d' /opt/NanoLLM/nano_llm/plugins/__init__.py


#Move in vlm inference service code
COPY . /jetson-services/inference/vlm/
WORKDIR /jetson-services/inference/vlm/

ENV WORK_DIR /jetson-services/inference/vlm/

RUN cd ${WORK_DIR}/src/ && \
    git clone https://github.com/NVIDIA-AI-IOT/mmj_utils && \
    cp -r ${WORK_DIR}/src/mmj_utils_overlay/* ${WORK_DIR}/src/mmj_utils && \
    cat ${WORK_DIR}/src/mmj_utils/mmj_utils/api_server.py

# RUN chown -R genai .
RUN ln -s /dev/null /tmp/null.mp4
RUN pip install -r /jetson-services/inference/vlm/src/requirements.txt

#Install moj utils
RUN pip install -e /jetson-services/inference/vlm/src/mmj_utils/

#clean up
RUN apt-get -y purge python3-pip git-all lsb-release libglew-dev libsoup2.4-dev libjson-glib-dev libgstrtspserver-1.0-dev avahi-utils
RUN apt-get -y purge libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-good1.0-dev libgstreamer-plugins-bad1.0-dev
RUN apt-get -y purge netbase
RUN apt-get -y autoremove
RUN apt-get clean
# USER genai

RUN mv /jetson-services/inference/vlm/config /config
ENV MAIN_CONFIG_PATH        /config/main_config.json
ENV CHAT_SERVER_CONFIG_PATH /config/chat_server_config.json
ENV MAIN_CONFIG_PATH_UNDER_DATA_DIR        /data/configs/vila-microservice/config/main_config.json
ENV CHAT_SERVER_CONFIG_PATH_UNDER_DATA_DIR /data/configs/vila-microservice/config/chat_server_config.json
ENV TRANSFORMERS_CACHE /data/models/huggingface

CMD bash -c "/jetson-services/inference/vlm/docker_start.sh | tee -a /data/vlm.log"