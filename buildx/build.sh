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

# Ensure DOCKER_USERNAME is set
if [ -z "$DOCKER_USERNAME" ]; then
  echo "Error: DOCKER_USERNAME is not set. Please define it in the .env file."
  exit 1
fi

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
while [[ "$use_cache" != "y" && "$use_cache" != "n" ]]; do
  echo "Invalid input. Please enter 'y' for yes or 'n' for no."
  read -p "Do you want to build with cache? (y/n): " use_cache
done

# Function to build a Docker image
build_image() {
  local folder=$1
  local base_image=$2
  local image_name=$(basename "$folder")
  local tag="${DOCKER_USERNAME}/001:${image_name}-${CURRENT_DATE_TIME}-1"

  echo "Building image: $image_name for platform: $PLATFORM"
  echo "Building folder: $folder"
  echo "Dockerfile path: $folder/Dockerfile"

  # Build the image
  if [ "$use_cache" = "y" ]; then
    docker buildx build --platform $PLATFORM -t $tag --push "$folder"
  else
    docker buildx build --no-cache --platform $PLATFORM -t $tag --push "$folder"
  fi

  # Check if the build succeeded
  if [ $? -ne 0 ]; then
    echo "Error: Failed to build image for $image_name. Exiting..."
    exit 1
  fi

  # Output only the tag
  echo "$tag"
}

# Build the images
echo "Starting build process..."
BUILD_ESSENTIAL_TAG=$(build_image "build/build-essential" "kairin/001:jetc-nvidia-pytorch-25.03-py3-igpu")
BAZEL_TAG=$(build_image "build/bazel" "$BUILD_ESSENTIAL_TAG")
echo "Build process complete!"

# Pull the most recently built image
echo "Pulling the most recently built image: $BAZEL_TAG"
docker pull "$BAZEL_TAG"
if [ $? -ne 0 ]; then
  echo "Error: Failed to pull the image $BAZEL_TAG. Exiting..."
  exit 1
fi

# Run the pulled image
echo "Running the pulled image: $BAZEL_TAG"
docker run -it --rm "$BAZEL_TAG" bash
if [ $? -ne 0 ]; then
  echo "Error: Failed to run the image $BAZEL_TAG. Exiting..."
  exit 1
fi

# Announce completion
echo "Build, push, pull, and run processes completed successfully!"
