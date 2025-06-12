#---
# name: faiss
# group: vectordb
# config: config.py
# depends: [cuda, numpy, cmake]
# test: [test.py]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG FAISS_VERSION \
    FAISS_BRANCH \
    FORCE_BUILD=off
    
COPY install.sh build.sh /tmp/faiss/
    
RUN /tmp/faiss/install.sh || /tmp/faiss/build.sh