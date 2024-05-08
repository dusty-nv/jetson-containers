#!/usr/bin/env bash
# homeassistant-supervised

set -euxo pipefail

echo "Installing Home Assistant Supervised dependencies..."

# Install dependencies
# add-apt-repository universe
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
	\
	debhelper \
	devscripts \
	dpkg-dev
apt-get clean
rm -rf /var/lib/apt/lists/*
