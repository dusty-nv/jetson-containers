#---
# name: faiss
# group: vectordb
# config: config.py
# depends: [cuda, numpy, cmake]
# test: [test.py]
#---
ARG BASE_IMAGE
ARG BUILD_IMAGE

FROM ${BUILD_IMAGE} as builder
FROM ${BASE_IMAGE} as runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		  libopenblas-dev \
		  swig \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
    
COPY --from=builder /opt/faiss*.whl /opt/
COPY --from=builder /usr/local/lib/libfaiss.a /usr/local/lib/libfaiss.a
COPY --from=builder /usr/local/include/faiss /usr/local/include/faiss
COPY --from=builder /usr/local/share/faiss /usr/local/share/faiss

RUN pip3 install --no-cache-dir --verbose /opt/faiss*.whl

RUN pip3 show faiss && python3 -c 'import faiss'