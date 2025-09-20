#!/bin/bash
# Pre-checkout cleanup script for GitHub Actions workflows
# Removes permission-restricted files before checkout

echo "=== Pre-checkout Cleanup ==="
echo "Cleaning up permission-restricted files before checkout..."
#sudo rm -rf /home/jetson/actions-runner/_work/jetson-containers/jetson-containers/data || echo "Data cleanup completed" # We don't want to delete the ./data dir because we want to keep the models and data for the jc-build test phase
sudo rm -rf /home/jetson/actions-runner/_work/jetson-containers/jetson-containers/logs || echo "Logs cleanup completed"
echo "Pre-checkout cleanup completed"
