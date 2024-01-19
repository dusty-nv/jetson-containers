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

RUN git clone --recursive https://github.com/OpenNMT/CTranslate2.git && \
    cd CTranslate2 && \
    mkdir build && cd build && \
    cmake .. -DWITH_CUDA=ON -DWITH_CUDNN=ON -DWITH_MKL=OFF -DOPENMP_RUNTIME=COMP && \
    make -j4 && \
    make install && \
    ldconfig

RUN cd CTranslate2/python && \
    pip3 install --no-cache-dir --verbose -r install_requirements.txt && \
    python3 setup.py bdist_wheel
    
RUN cp CTranslate2/python/dist/ctranslate2*.whl /opt && \
    pip3 install --no-cache-dir --verbose /opt/ctranslate2*.whl

RUN pip3 show ctranslate2 && python3 -c 'import ctranslate2'