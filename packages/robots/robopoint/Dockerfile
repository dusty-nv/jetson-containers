#---
# name: robopoint
# group: robots
# docs: docs.md
# depends: [transformers, bitsandbytes, flash-attention, opencv:4.11.0]
# test: [test.py]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV ROBOPOINT_MODEL="wentao-yuan/robopoint-v1-vicuna-v1.5-13b" \
    ROBOPOINT_QUANTIZATION="int4" \
    ROBOPOINT_HOST="0.0.0.0" \
    ROBOPOINT_PORT=10000
    
ARG ROBOPOINT_REPO="mschweig/RoboPoint" \
    ROBOPOINT_BRANCH=master

ADD https://api.github.com/repos/${ROBOPOINT_REPO}/branches/${ROBOPOINT_BRANCH} /tmp/robopoint_version.json
RUN git clone --branch=${ROBOPOINT_BRANCH} https://github.com/${ROBOPOINT_REPO}.git /opt/robopoint

RUN cd /opt/robopoint && \
    pip3 install --verbose -e .

EXPOSE ${ROBOPOINT_PORT}
COPY client.py start-server.sh /opt/robopoint/

CMD /opt/robopoint/start-server.sh