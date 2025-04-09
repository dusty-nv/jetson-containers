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

if [ "$ARCH" != "aarch64" ]; then
    echo "This script is only intended to build for aarch64 devices."
    exit 1
fi

PLATFORM="linux/arm64"

# Check if the builder already exists
if ! docker buildx inspect jetson-builder &>/dev/null; then
  # Create the builder instance if it doesn't exist
  docker buildx create --name jetson-builder
fi

# Use the builder instance
docker buildx use jetson-builder

# Ask if the user wants to build with or without cache
read -p "Do you want to build with cache? (y/n): " use_cache

# List of specific folders to build
folders=("build/bazel" "build/build-essential")

# Function to build Docker images for specified folders
build_images_from_folders() {
  for folder in "${folders[@]}"; do
    if [ -d "$folder" ]; then
      # Extract the image name from the folder name
      local image_name=$(basename "$folder")
      local tag="${DOCKER_USERNAME}/001:${image_name}-${CURRENT_DATE_TIME}-1"

      echo "Building image: $image_name for platform: $PLATFORM"

      if [ "$use_cache" = "y" ]; then
        docker buildx build --platform $PLATFORM -t $tag --build-arg BASE_IMAGE=$BASE_IMAGE --push "$folder"
      else
        docker buildx build --no-cache --platform $PLATFORM -t $tag --build-arg BASE_IMAGE=$BASE_IMAGE --push "$folder"
      fi

      echo "Docker image tagged and pushed as $tag"
    else
      echo "Directory $folder not found! Skipping..."
    fi
  done
}

# Build images for the specified folders
build_images_from_folders
