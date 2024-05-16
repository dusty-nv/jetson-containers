#!/usr/bin/env bash
set -e

current_user=$(whoami)

# Create hassio system on host
sudo mkdir -p /usr/share/hassio/audio/asound
sudo mkdir -p /usr/share/hassio/dns
sudo mkdir -p /usr/share/hassio/homeassistant
sudo mkdir -p /usr/share/hassio/ssl
sudo mkdir -p /usr/share/hassio/share
sudo mkdir -p /usr/share/hassio/media
sudo mkdir -p /usr/share/hassio/tmp

sudo chown -R ${current_user}:${current_user} /usr/share/hassio

# Create the symlink to `/etc/pulse/client.conf`
sudo ln -sf /etc/pulse/client.conf /usr/share/hassio/tmp/homeassistant_pulse
