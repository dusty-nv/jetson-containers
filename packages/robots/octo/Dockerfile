#---
# name: octo
# group: robots
# docs: docs.md
# depends: [transformers, opencv:4.10.0, jax]
# requires: '>=36'
# test: [test.sh, test.py]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}


RUN git clone --branch=main --depth=1 https://github.com/octo-models/octo /opt/octo && \
    cd /opt/octo && \
    sed -i 's/==/>=/g' requirements.txt && \
    pip3 install -U -r requirements.txt && \
    pip3 install mediapy && \
    pip3 install -e .

CMD ["/bin/bash"]