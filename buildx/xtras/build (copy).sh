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
BASE_IMAGE=kairin/001:20250402-063709-1

# Image names
BUILD_ESSENTIAL_IMAGE=build-essential
STABLE_DIFFUSION_WEBUI_IMAGE=stable-diffusion-webui
COMFYUI_IMAGE=comfyui

# Get the current date and time formatted as YYYYMMDD-HHMMSS
CURRENT_DATE_TIME=$(date +"%Y%m%d-%H%M%S")

# Create the tags with the current date and time and append number 1
BUILD_ESSENTIAL_TAG="${DOCKER_USERNAME}/${BUILD_ESSENTIAL_IMAGE}:${CURRENT_DATE_TIME}-1"
STABLE_DIFFUSION_WEBUI_TAG="${DOCKER_USERNAME}/${STABLE_DIFFUSION_WEBUI_IMAGE}:${CURRENT_DATE_TIME}-1"
COMFYUI_TAG="${DOCKER_USERNAME}/${COMFYUI_IMAGE}:${CURRENT_DATE_TIME}-1"

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

# Build the Docker images using buildx and push to Docker Hub
build_image() {
  local directory=$1
  local tag=$2

  if [ "$use_cache" = "y" ]; then
    docker buildx build --platform $PLATFORM -t $tag --build-arg BASE_IMAGE=$BASE_IMAGE --push $directory
  else
    docker buildx build --no-cache --platform $PLATFORM -t $tag --build-arg BASE_IMAGE=$BASE_IMAGE --push $directory
  fi

  echo "Docker image tagged and pushed as $tag"
}

# Build the images
build_image "./l4t/build-essential" $BUILD_ESSENTIAL_TAG
build_image "./l4t/stable-diffusion-webui" $STABLE_DIFFUSION_WEBUI_TAG
build_image "./l4t/comfyui" $COMFYUI_TAG
