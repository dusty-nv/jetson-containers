#---
# name: openfang
# group: agents
# depends: [rust]
# config: config.py
# test: test.sh
# docs: https://github.com/RightNow-AI/openfang
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG OPENFANG_VERSION

COPY build.sh /tmp/openfang/
RUN /tmp/openfang/build.sh

ENV OPENFANG_HOME=/data/openfang
RUN mkdir -p $OPENFANG_HOME

EXPOSE 4200

WORKDIR /data/openfang
CMD ["openfang", "start"]
