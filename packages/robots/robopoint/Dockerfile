#---
# name: robopoint
# group: robots
# docs: docs.md
# depends: [transformers, bitsandbytes, flash-attention]
# test: [test.py]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN git clone https://github.com/mschweig/RoboPoint.git /opt/robopoint

RUN cd /opt/robopoint && \
    pip3 install --verbose -e .

ENV ROBOPOINT_HOST="0.0.0.0" \
    ROBOPOINT_PORT=10000 \
    ROBOPOINT_MODEL="wentao-yuan/robopoint-v1-vicuna-v1.5-13b"
    
EXPOSE ${ROBOPOINT_PORT}
COPY client.py start-server.sh /opt/robopoint/

CMD /opt/robopoint/start-server.sh