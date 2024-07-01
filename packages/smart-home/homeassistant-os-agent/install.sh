#!/usr/bin/env bash
# homeassistant-os-agent

set -xo pipefail

# Install dependencies
apt-get update
apt-get install -y --no-install-recommends \
	libglib2.0-bin \
	udisks2 \
	dbus
apt-get clean
rm -rf /var/lib/apt/lists/*

# Install OS-Agent
# echo "Installing Home Assistant OS-Agent ${OS_AGENT_VERSION}..."
# wget --quiet --show-progress --progress=bar:force:noscroll \
# 	https://github.com/home-assistant/os-agent/releases/download/${OS_AGENT_VERSION}/os-agent_${OS_AGENT_VERSION}_linux_${BUILD_ARCH}.deb \
# 	-O /tmp/os-agent_${OS_AGENT_VERSION}_linux_${BUILD_ARCH}.deb
# dpkg -i /tmp/os-agent_*.deb

go get github.com/goreleaser/goreleaser

git clone --branch ${OS_AGENT_VERSION} https://github.com/home-assistant/os-agent ${OS_AGENT_TMP_DIR}
cd ${OS_AGENT_TMP_DIR}
ls -l
go mod tidy
go generate ./...
go build -ldflags="-s -w -X 'main.version=${OS_AGENT_VERSION}' -X 'main.board=Supervised'"
ls -l
go install

# Copy the entrypoint script
COPY entrypoint.sh /usr/local/bin/entrypoint.sh

# Set the entrypoint
# FIXME: Fix tests
# ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
