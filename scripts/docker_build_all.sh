#!/usr/bin/env bash

# dev packages
#sh ./scripts/stage_dev.sh

# PyTorch
sh ./scripts/docker_build.sh l4t-pytorch:r32.4.2-pth1.2-py3 Dockerfile.pytorch 

# TensorFlow
sh ./scripts/docker_build.sh l4t-tensorflow:r32.4.2-tf1.15-py3 Dockerfile.tensorflow

# TensorRT
#sh ./scripts/docker_build.sh l4t-tensorrt:r32.4-py3 Dockerfile.tensorrt

# Machine Learning
sh ./scripts/docker_build.sh l4t-ml:r32.4.2-py3 Dockerfile.ml

