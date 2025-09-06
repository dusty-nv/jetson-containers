#!/bin/bash
# Clean environment script for GitHub Actions workflows
# Cleans up Docker, cache, and permission-restricted directories

echo "=== Cleaning Environment ==="
echo "Removing any existing Docker containers and images..."
docker system prune -f || echo "Docker cleanup failed"
echo "Clearing any cached build artifacts..."
rm -rf ~/.cache/* || echo "Cache cleanup failed"

# Clean up any permission-restricted directories
echo "Cleaning up permission-restricted directories..."
rm -rf /home/jetson/actions-runner/_work/jetson-containers/jetson-containers/data || echo "Data cleanup failed"
rm -rf /home/jetson/actions-runner/_work/jetson-containers/jetson-containers/logs || echo "Logs cleanup failed"

# Reset permissions on work directory
chown -R jetson:jetson /home/jetson/actions-runner/_work/jetson-containers/ || echo "Permission reset failed"

echo "Environment cleaned for fresh build"
