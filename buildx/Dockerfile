# Use the specified base image
FROM kairin/001:nvcr.io-nvidia-pytorch-25.02-py3-igpu

# Install any dependencies or files needed
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy the necessary files into the container
COPY ../../qt-online-installer-linux-arm64-4.9.0.run /media/kkk/buildx/apps

# Set the working directory
WORKDIR /media/kkk/buildx/apps

# Execute any additional commands
RUN echo "Installation complete"
