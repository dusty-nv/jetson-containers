#!/usr/bin/env bash
# homeassistant-supervised

set -euxo pipefail

echo "Installing Home Assistant Supervised..."
# echo "Verify system bus: Ensure that the system bus (D-Bus) is running and accessible:"

# systemctl status systemd
# systemctl status dbus
# systemctl status network-manager
# # Make sure dbus is set to start at boot
# systemctl enable systemd
# systemctl enable dbus
# systemctl enable network-manager

# echo "Check if systemd is running: Verify whether your system is using systemd as its init system:"
# pidof systemd
# pidof dbus
# pidof network-manager

# check if we have apparmor on system
# apparmor profile: https://version.home-assistant.io/apparmor.txt
# apparmor_status

# IMPORTANT: Skip Docker-CE installation
# ----------
# Using the existing Docker instance: This involves mounting the Docker socket from the host into 
# the container. By doing this, the service running in the container can interact with the Docker 
# daemon on the host. This is a common approach and is typically preferred for most use cases because
# it avoids the overhead and complexity of running Docker within Docker (dind). Here’s how you might do it:
# ```
# docker run \
# 	-v /var/run/docker.sock:/var/run/docker.sock \
# 	-v /usr/bin/docker:/usr/bin/docker \							# https://github.com/home-assistant/supervised-installer/blob/main/homeassistant-supervised/DEBIAN/postinst#L9
#	-v /run/docker.sock:/run/docker.sock \ 							# https://github.com/home-assistant/supervised-installer/blob/main/homeassistant-supervised/etc/systemd/system/hassio-supervisor.service#L9
# 	-v /run/dbus/system_bus_socket:/run/dbus/system_bus_socket \ 	# https://github.com/home-assistant/supervised-installer/blob/main/homeassistant-supervised/etc/systemd/system/hassio-supervisor.service#L8
# 	--privileged \
# 	my-service-image
# ```
# curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -
# add-apt-repository \
#     "deb [arch=$(dpkg --print-architecture)] https://download.docker.com/linux/ubuntu \
#     $(lsb_release -cs) stable"
# apt-get update
# apt-get install -y --no-install-recommends \
# 	docker-ce \
# 	docker-ce-cli \
# 	containerd.io

# Install the Agent for Home Assistant OS - https://github.com/home-assistant/os-agent
echo "Installing Home Assistant OS-Agent..."
wget --quiet --show-progress --progress=bar:force:noscroll \
	https://github.com/home-assistant/os-agent/releases/download/${OS_AGENT_VERSION}/os-agent_${OS_AGENT_VERSION}_linux_aarch64.deb \
	-O /tmp/os-agent_${OS_AGENT_VERSION}_linux_aarch64.deb
dpkg -i /tmp/os-agent_*.deb
# test if the installation was successful
gdbus introspect --system --dest io.hass.os --object-path /io/hass/os
rm /tmp/os-agent_*.deb

# Build Home Assistant Supervised
git clone --branch=${SUPERVISED_VERSION} https://github.com/home-assistant/supervised-installer /tmp/supervised-installer
cd /tmp/supervised-installer/homeassistant-supervised
dpkg-buildpackage -us -uc
ls -l /tmp/supervised-installer
dpkg -i /tmp/supervised-installer/apparmor-deb_*.deb
# cp -r /tmp/supervised-installer/* /

# Reconfigure installation if user changed the default path for our $DATA_SHARE from /usr/share/hassio.
# This path is used to store all home assistant related things.
# if [ $DATA_SHARE != "/usr/share/hassio" ]; then
# 	dpkg --force-confdef --force-confold -i /opt/homeassistant-supervised.deb
# fi

# systemctl enable docker


# Copy Home Assistant Supervised system overlay
# cp -r /tmp/supervised-installer/etc/* /etc/
# cp -r /tmp/supervised-installer/usr/* /usr/