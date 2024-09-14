#---
# name: crossformer
# group: robots
# docs: docs.md
# depends: [transformers, opencv:4.10.0, jax]
# requires: '>=36'
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}


RUN git clone --branch=main --depth=1 https://github.com/rail-berkeley/crossformer /opt/crossformer && \
    cd /opt/crossformer && \
    sed -i 's/==/>=/g' requirements.txt && \
    pip3 install -U -r requirements.txt && \
    pip3 install mediapy && \
    pip3 install -e .

CMD ["/bin/bash"]