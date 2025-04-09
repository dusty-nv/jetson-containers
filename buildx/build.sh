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

# Base image
BASE_IMAGE="kairin/001:nvcr.io-nvidia-pytorch-25.02-py3-igpu"

# Get the current date and time formatted as YYYYMMDD-HHMMSS
CURRENT_DATE_TIME=$(date +"%Y%m%d-%H%M%S")

# Determine the current platform
ARCH=$(uname -m)

if [ "$ARCH" = "x86_64" ]; then
    echo "This script is intended for your current device only. x86_64 (amd64) builds are not supported."
    exit 1
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

# Function to build Docker images from the build folder
build_images_from_directory() {
  local build_folder="build"
  for dir in $build_folder/*; do
    if [ -d "$dir" ]; then
      # Extract the image name from the folder name
      local image_name=$(basename "$dir")
      local tag="${DOCKER_USERNAME}/001:${image_name}-${CURRENT_DATE_TIME}-1"

      echo "Building image: $image_name for platform: $PLATFORM"

      if [ "$use_cache" = "y" ]; then
        docker buildx build --platform $PLATFORM -t $tag --build-arg BASE_IMAGE=$BASE_IMAGE --push "$dir"
      else
        docker buildx build --no-cache --platform $PLATFORM -t $tag --build-arg BASE_IMAGE=$BASE_IMAGE --push "$dir"
      fi

      echo "Docker image tagged and pushed as $tag"
    fi
  done
}

# Build images from the build folder
build_images_from_directory
