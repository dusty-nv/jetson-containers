# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

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
RUN export PIP_INDEX_URL=https://pypi.jetson-ai-lab.io/jp6/cu126
RUN export PIP_TRUSTED_HOST=pypi.jetson-ai-lab.io

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
RUN uv pip install --upgrade --force-reinstall --no-cache-dir --verbose cmake

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
RUN uv pip install numpy>="1.26.4"
RUN uv pip install torch torchvision torchaudio cuda_python onnxruntime
RUN uv pip install transformers
RUN uv pip install --no-cache-dir --verbose onnx

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
RUN uv pip install sentencepiece
RUN uv pip install --index-url http://jetson-ai-lab.io/jp6/cu126 --trusted-host jetson-ai-lab.io  --no-cache-dir --verbose tvm==0.15.0 mlc-llm==0.1.0


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
RUN uv pip install termcolor tabulate accelerate cachetools decorator-args inflect flask tqdm getch natsort pyyaml

#instal clip TRT
WORKDIR /opt
RUN git clone https://github.com/dusty-nv/clip_trt
WORKDIR /opt/clip_trt
RUN uv pip install -e .

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
