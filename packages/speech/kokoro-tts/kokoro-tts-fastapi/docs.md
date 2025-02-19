# KokoroTTS FastAPI

This Dockerfile sets up a text-to-speech (TTS) service based on the Kokoro TTS engine using FastAPI. Here's what it generally does:

1. It starts with a base image that supports CUDA 12.8 for GPU acceleration
2. Installs necessary system dependencies including audio libraries and ffmpeg
3. Sets up the Python environment with specific versions of scientific and NLP packages
4. Installs the Kokoro TTS engine and its dependencies
5. Downloads the pre-trained Kokoro voice model
6. Configures PyTorch with CUDA support specifically for aarch64 (ARM) architecture
7. Exposes port 8880 for the API service
8. Sets up the application to run as a web service

The key aspects of this setup are:
- It's designed to run on Jetson or other ARM devices with CUDA support
- It modifies package dependencies to use ARM-compatible PyTorch builds
- It provides a ready-to-use TTS service that can generate speech from text

When users run this container, they'll get a FastAPI-based web service that converts text to speech using the Kokoro TTS model with GPU acceleration.

## Build with
CUDA_VERSION=12.8 jetson-containers build --name=kokoro-tts:fastapi kokoro-tts:fastapi

## Run with
jetson-containers run -p 8880:8880 kokoro-tts:fastapi-r36.4.3-cu128


