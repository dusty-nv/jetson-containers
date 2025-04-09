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

# Define mapping of folders to their respective base images
declare -A folder_base_image_map=(
  ["build/bazel"]="kairin/001:nvcr.io-nvidia-pytorch-25.02-py3-igpu"
  ["build/build-essential"]="kairin/001:nvcr.io-nvidia-pytorch-25.02-py3-igpu"
)

# Function to build Docker images for specified folders
build_images_from_folders() {
  for folder in "${!folder_base_image_map[@]}"; do
    if [ -d "$folder" ]; then
      if [ -f "$folder/Dockerfile" ]; then
        local base_image="${folder_base_image_map[$folder]}"
        local image_name=$(basename "$folder")
        local tag="${DOCKER_USERNAME}/001:${image_name}-${CURRENT_DATE_TIME}-1"

        echo "Building image: $image_name for platform: $PLATFORM using base image: $base_image"
        echo "Building folder: $folder"
        echo "Dockerfile path: $folder/Dockerfile"

        if [ "$use_cache" = "y" ]; then
          if ! docker buildx build --platform $PLATFORM -t $tag --build-arg BASE_IMAGE=$base_image --push "$folder"; then
            echo "Error: Failed to build image for $image_name. Skipping..."
            continue
          fi
        else
          if ! docker buildx build --no-cache --platform $PLATFORM -t $tag --build-arg BASE_IMAGE=$base_image --push "$folder"; then
            echo "Error: Failed to build image for $image_name. Skipping..."
            continue
          fi
        fi

        echo "Docker image tagged and pushed as $tag"
      else
        echo "Dockerfile not found in $folder. Skipping..."
      fi
    else
      echo "Directory $folder not found! Skipping..."
    fi
  done
}

# Build images for the specified folders
build_images_from_folders
