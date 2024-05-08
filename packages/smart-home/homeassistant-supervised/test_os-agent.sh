#!/usr/bin/env bash
# Agent for Home Assistant OS

echo "Testing Agent for Home Assistant OS..."

gdbus introspect --system --dest io.hass.os --object-path /io/hass/os

echo "Agent for Home Assistant OS - OK"
