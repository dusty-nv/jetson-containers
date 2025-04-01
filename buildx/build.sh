#!/bin/bash

# Remove any existing builder instance
docker buildx rm mybuilder || true

# Set the builder instance
docker buildx create --name mybuilder --use

# Change permissions of the installer file
chmod +x /media/kkk/jetc/buildx/qt-online-installer-linux-arm64-4.9.0.run

# Build the Docker image using buildx
docker buildx build --platform linux/amd64,linux/arm64 \
    -t my-custom-image:latest .

# Remove the builder instance
docker buildx rm mybuilder
