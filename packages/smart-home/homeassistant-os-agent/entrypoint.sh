#!/usr/bin/env bash
set -e

# Create the systemd service file
cat <<EOF >/etc/systemd/system/haos-agent.service
[Unit]
Description=Home Assistant OS Agent
DefaultDependencies=no
Requires=dbus.socket udisks2.service
After=dbus.socket

[Service]
BusName=io.hass.os
Type=notify
Restart=always
RestartSec=5s
Environment="DBUS_SYSTEM_BUS_ADDRESS=unix:path=/run/dbus/system_bus_socket"
ExecStart=/usr/bin/os-agent

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd manager configuration
systemctl daemon-reload

# Enable the service to start on boot
systemctl enable haos-agent.service

# Start the service
systemctl start haos-agent.service

# Keep the container running
exec "$@"
