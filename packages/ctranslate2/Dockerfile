#---
# name: ctranslate2
# group: transformer
# depends: [cmake]
# requires: '>=34.1.0'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

# RUN apt-get update && \
#     apt-get install -y --no-install-recommends \
# 		  ffmpeg \
#     && rm -rf /var/lib/apt/lists/* \
#     && apt-get clean

# RUN pip3 install scipy jiwer ipywebrtc

# Clone the repository:
RUN git clone --recursive https://github.com/OpenNMT/CTranslate2.git && \
    cd CTranslate2 && \
    mkdir build && cd build && \
    cmake .. -DWITH_CUDA=ON -DWITH_CUDNN=ON -DWITH_MKL=OFF -DOPENMP_RUNTIME=COMP && \
    make -j4 && \
    sudo make install && \
    sudo ldconfig

RUN cd /opt/CTranslate2/python && \
    pip3 install -r install_requirements.txt && \
    python3 setup.py bdist_wheel && \
    pip3 install dist/*.whl

WORKDIR /opt/ctranslate2