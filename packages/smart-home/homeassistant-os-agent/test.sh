#!/usr/bin/env bash
# homeassistant-os-agent
set -exo pipefail

echo "testing homeassistant-os-agent..."

gdbus introspect --system --dest io.hass.os --object-path /io/hass/os
gdbus call --system --dest io.hass.os --object-path /io/hass/os/Boards/Yellow --method org.freedesktop.DBus.Properties.Set io.hass.os.Boards.Yellow PowerLED "<false>"

echo "homeassistant-os-agent OK"
