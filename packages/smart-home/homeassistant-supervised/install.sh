#!/usr/bin/env bash
# homeassistant-supervised

# https://github.com/dcmartin/motion-ai/blob/master/docs/QUICKSTART.md

set -euxo pipefail

echo "Installing Home Assistant Supervised ${SUPERVISED_INSTALLER_VERSION}..."

# Install dependencies
apt-get update
apt-get install -y --no-install-recommends \
	apparmor \
	cifs-utils \
	curl \
	dbus \
	jq \
	libglib2.0-bin \
	lsb-release \
	network-manager \
	nfs-common \
	systemd \
	systemd-journal-remote \
	udisks2 \
	wget \
	debhelper \
	devscripts \
	dpkg-dev
apt-get clean
rm -rf /var/lib/apt/lists/*

# Install Docker
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -
add-apt-repository \
    "deb [arch=$(dpkg --print-architecture)] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
apt-get update
apt-get install -y --no-install-recommends \
	docker-ce \
	docker-ce-cli \
	containerd.io

# Install OS-Agent
echo "Installing Home Assistant OS-Agent ${OS_AGENT_VERSION}..."
wget --quiet --show-progress --progress=bar:force:noscroll \
	https://github.com/home-assistant/os-agent/releases/download/${OS_AGENT_VERSION}/os-agent_${OS_AGENT_VERSION}_linux_${BUILD_ARCH}.deb \
	-O /tmp/os-agent_${OS_AGENT_VERSION}_linux_${BUILD_ARCH}.deb
dpkg -i /tmp/os-agent_*.deb

# Install latest homeassistant-supervised
echo "Installing Home Assistant Supervised ${SUPERVISED_INSTALLER_VERSION}..."
wget --quiet --show-progress --progress=bar:force:noscroll \
	https://github.com/home-assistant/supervised-installer/releases/download/${SUPERVISED_INSTALLER_VERSION}/homeassistant-supervised.deb \
	- O /tmp/homeassistant-supervised.deb
sudo dpkg -i /tmp/homeassistant-supervised.deb

ha jobs options --ignore-conditions healthy

echo "BEFORE:"
cat /etc/docker/daemon.json
# cat > /etc/docker/daemon.json << EOF
# {
#   "runtimes": {
#     "nvidia": {
#       "path": "/usr/bin/nvidia-container-runtime",
#       "runtimeArgs": []
#     }
#   },
#   "log-driver": "journald",
#   "storage-driver": "overlay2",
#   "default-runtime": "nvidia",
#   "experimental": true
# }
# EOF
# echo "AFTER:"
# cat /etc/docker/daemon.json
