#!/bin/bash
# System information script for GitHub Actions workflows
# Displays Jetson system information

echo "=== Jetson Orin System Info ==="
echo "Hello from $HOSTNAME running on Jetson Orin!"
uname -a
nvidia-smi || echo "No nvidia-smi (Jetson usually)"
echo "GPU Memory:"
cat /proc/device-tree/compatible || echo "Device tree info not available"
