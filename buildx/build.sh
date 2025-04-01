#!/bin/bash

# Remove any existing builder instance
docker buildx rm mybuilder || true

# Set the builder instance
docker buildx create --name mybuilder --use

# Build the Docker image using buildx
docker buildx build --platform linux/amd64,linux/arm64 \
    -t my-custom-image:latest .

# Remove the builder instance
docker buildx rm mybuilder
