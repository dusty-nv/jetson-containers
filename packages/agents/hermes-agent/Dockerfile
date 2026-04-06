#---
# name: hermes-agent
# group: agents
# depends: [cuda-stack:standard, python, nodejs]
# config: config.py
# test: test.sh
# docs: https://github.com/NousResearch/hermes-agent
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG HERMES_AGENT_VERSION

COPY build.sh /tmp/hermes-agent/
RUN /tmp/hermes-agent/build.sh

ENV HERMES_HOME=/data/hermes-agent
RUN mkdir -p $HERMES_HOME

EXPOSE 18789

WORKDIR /data/hermes-agent
CMD ["hermes"]
