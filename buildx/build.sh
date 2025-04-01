#!/bin/bash

# Docker Hub username
DOCKER_USERNAME=kairin

# Image name
IMAGE_NAME=001

# Get the current date and time formatted as YYYYMMDD-HHMMSS
CURRENT_DATE_TIME=$(date +"%Y%m%d-%H%M%S")

# Create the tag with the current date and time and append number 1
TAG="${DOCKER_USERNAME}/${IMAGE_NAME}:${CURRENT_DATE_TIME}-1"

# Determine the current platform
ARCH=$(uname -m)

if [ "$ARCH" = "x86_64" ]; then
    PLATFORM="linux/amd64"
elif [ "$ARCH" = "aarch64" ]; then
    PLATFORM="linux/arm64"
else
    echo "Unsupported architecture: $ARCH"
    exit 1
fi

# Check if the builder already exists
if ! docker buildx inspect jetson-builder &>/dev/null; then
  # Create the builder instance if it doesn't exist
  docker buildx create --name jetson-builder
fi

# Use the builder instance
docker buildx use jetson-builder

# Build the Docker image using buildx and push to Docker Hub
docker buildx build --platform $PLATFORM \
    -t $TAG \
    --push .

echo "Docker image tagged and pushed as $TAG"
