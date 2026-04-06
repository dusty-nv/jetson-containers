#---
# name: openclaw
# group: agents
# depends: [cuda-stack:standard, python, nodejs]
# config: config.py
# test: test.sh
# docs: https://github.com/openclaw/openclaw
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG OPENCLAW_VERSION

COPY build.sh /tmp/openclaw/
RUN /tmp/openclaw/build.sh

ENV OPENCLAW_HOME=/data/openclaw
RUN mkdir -p $OPENCLAW_HOME

EXPOSE 18789

WORKDIR /data/openclaw
CMD ["openclaw", "gateway", "--bind", "0.0.0.0", "--port", "18789"]
