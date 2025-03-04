#---
# name: robopoint
# group: robots
# depends: [torch, torchvision, transformers, bitsandbytes, numpy]
# test: []
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN git clone https://github.com/wentaoyuan/RoboPoint.git /opt/robopoint && \
    cd /opt/robopoint && \
    sed -i '/"torch[^"]*",/d' pyproject.toml && \
    sed -i '/"torchvision[^"]*",/d' pyproject.toml && \
    sed -i '/"transformers[^"]*",/d' pyproject.toml && \
    sed -i '/"bitsandbytes[^"]*",/d' pyproject.toml && \
    sed -i '/"numpy[^"]*",/d' pyproject.toml && \
    cat -n pyproject.toml

RUN cd /opt/robopoint && \
    pip3 install  -e . --verbose

RUN pip3 install gradio --upgrade --no-deps
