#!/bin/bash

# Load environment variables from the .env file
if [ -f .env ]; then
  set -a
  source .env
  set +a
else
  echo ".env file not found!"
  exit 1
fi

# Docker Hub username
DOCKER_USERNAME=${DOCKER_USERNAME}

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

# Ask if the user wants to build with or without cache
read -p "Do you want to build with cache? (y/n): " use_cache

# Build the Docker image using buildx and push to Docker Hub
if [ "$use_cache" = "y" ]; then
  docker buildx build --platform $PLATFORM --build-arg TARGETPLATFORM=$PLATFORM \
      -t $TAG \
      --push .
else
  docker buildx build --no-cache --platform $PLATFORM --build-arg TARGETPLATFORM=$PLATFORM \
      -t $TAG \
      --push .
fi

echo "Docker image tagged and pushed as $TAG"
