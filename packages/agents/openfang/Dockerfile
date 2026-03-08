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

# OpenFang (Tauri/wry) needs GLib, GTK3, WebKit2GTK, and related dev libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libglib2.0-dev \
    libgtk-3-dev \
    libwebkit2gtk-4.1-dev \
    libayatana-appindicator3-dev \
    librsvg2-dev \
    libsoup-3.0-dev \
    libssl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY build.sh /tmp/openfang/
RUN /tmp/openfang/build.sh

ENV OPENFANG_HOME=/data/openfang
RUN mkdir -p $OPENFANG_HOME

EXPOSE 4200

WORKDIR /data/openfang
CMD ["openfang", "start"]
