#!/usr/bin/env bash
# homeassistant-supervisor

set -xo pipefail

# Install dependencies
apt-get update
apt-get install -y --no-install-recommends \
	findutils \
	git \
	libffi-dev \
	libpulse-dev \
	musl-tools \
	openssl \
	libyaml-dev \
	systemd-journal-remote \
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
	udisks2 \
	wget \
	debhelper \
	devscripts \
	dpkg-dev
apt-get clean
rm -rf /var/lib/apt/lists/*
