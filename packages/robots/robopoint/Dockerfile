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

RUN pip3 install gradio --upgrade

CMD python3 -m robopoint.serve.controller --host 0.0.0.0 --port 10000 & \
    python3 -m robopoint.serve.gradio_web_server --controller http://localhost:10000 --model-list-mode reload --share & \
    python3 -m robopoint.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 20000 --worker http://localhost:20000 --model-path /data/models/robopoint-v1-vicuna-v1.5-13b
