#!/bin/bash
# Pre-checkout cleanup script for GitHub Actions workflows
# Removes permission-restricted files before checkout

echo "=== Pre-checkout Cleanup ==="
echo "Cleaning up permission-restricted files before checkout..."
sudo rm -rf /home/jetson/actions-runner/_work/jetson-containers/jetson-containers/data || echo "Data cleanup completed"
sudo rm -rf /home/jetson/actions-runner/_work/jetson-containers/jetson-containers/logs || echo "Logs cleanup completed"
echo "Pre-checkout cleanup completed"
