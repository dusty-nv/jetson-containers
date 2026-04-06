#---
# name: zeroclaw
# group: agents
# depends: [cuda-stack:standard, python, rust]
# config: config.py
# test: test.sh
# docs: https://github.com/zeroclaw-labs/zeroclaw
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG ZEROCLAW_VERSION

COPY build.sh /tmp/zeroclaw/
RUN /tmp/zeroclaw/build.sh

ENV ZEROCLAW_HOME=/data/zeroclaw
RUN mkdir -p $ZEROCLAW_HOME

EXPOSE 42617

WORKDIR /data/zeroclaw
CMD ["zeroclaw", "daemon"]
