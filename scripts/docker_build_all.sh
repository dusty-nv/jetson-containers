#!/usr/bin/env bash

# dev packages
sh ./scripts/setup_packages.sh

# PyTorch
sh ./scripts/docker_build.sh l4t-pytorch:r32.4-pth1.2-py3 Dockerfile.pytorch 

# TensorFlow
sh ./scripts/docker_build.sh l4t-tensorflow:r32.4-tf1.15-py3 Dockerfile.tensorflow

# Machine Learning
#sh ./scripts/docker_build.sh l4t-ml:r32.4-py3

