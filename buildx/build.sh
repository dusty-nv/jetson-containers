#!/bin/bash

# Docker Hub username
DOCKER_USERNAME=kairin

# Image name
IMAGE_NAME=my-custom-image

# Remove any existing builder instance
docker buildx rm mybuilder || true

# Set the builder instance
docker buildx create --name mybuilder --use

# Build the Docker image using buildx and push to Docker Hub
docker buildx build --platform linux/amd64,linux/arm64 \
    -t $DOCKER_USERNAME/$IMAGE_NAME:latest \
    --push .

# Remove the builder instance
docker buildx rm mybuilder
