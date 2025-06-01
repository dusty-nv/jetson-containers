#---
# name: graphiti
# group: rag
# depends: [numpy]
# requires: '>=34.1.0'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN cd /opt && \
    pip3 install \
        neo4j && \
    git clone https://github.com/getzep/graphiti && \
    cd graphiti && \
    pip3 install -e .

CMD /bin/bash
