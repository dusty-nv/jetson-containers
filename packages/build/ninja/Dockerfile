#---
# name: ninja
# group: core
# depends: [python]
# test: [test.sh]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    unzip wget curl jq

RUN LATEST_VERSION=$(curl -s https://api.github.com/repos/ninja-build/ninja/releases/latest | jq -r .tag_name) && \
    wget -q "https://github.com/ninja-build/ninja/releases/download/${LATEST_VERSION}/ninja-linux-aarch64.zip" && \
    unzip -o ninja-linux-aarch64.zip -d /usr/bin/ && \
    rm ninja-linux-aarch64.zip && \
    chmod +x /usr/bin/ninja && \
    apt-get -y remove wget unzip curl jq && \
    apt-get -y autoremove && \
    apt-get -y clean && \
    rm -rf /var/lib/apt/lists/*
