#!/bin/bash

# Docker Hub username
DOCKER_USERNAME=kairin

# Base image name
IMAGE_NAME=my-custom-image

# Get the current date and time formatted as YYYYMMDD-HHMMSS
CURRENT_DATE_TIME=$(date +"%Y%m%d-%H%M%S")

# Create the tag with the current date and time and append number 1
TAG="${DOCKER_USERNAME}/001:${CURRENT_DATE_TIME}-1"

# Remove any existing builder instance
docker buildx rm mybuilder || true

# Set the builder instance
docker buildx create --name mybuilder --use

# Build the Docker image using buildx and push to Docker Hub
docker buildx build --platform linux/amd64,linux/arm64 \
    -t $TAG \
    --push .

# Remove the builder instance
docker buildx rm mybuilder

echo "Docker image tagged and pushed as $TAG"
