# Use the specified base image
FROM kairin/001:20250402-063709-1 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages
RUN apt-get update && \
    apt-get install -y --allow-downgrades --allow-remove-essential --allow-change-held-packages \
    build-essential curl wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Stage for comfyui
FROM base as comfyui
COPY l4t/comfyui /opt/comfyui
WORKDIR /opt/comfyui
RUN docker build -t comfyui-image .

# Stage for cuda-python
FROM base as cuda-python
COPY l4t/cuda-python /opt/cuda-python
WORKDIR /opt/cuda-python
RUN docker build -t cuda-python-image .

# Copy the build.sh script to the final image and make it executable
FROM base
COPY buildx/build.sh /usr/local/bin/build.sh
RUN chmod +x /usr/local/bin/build.sh

# Define the entrypoint or command if needed
CMD ["bash"]
