# Dockerfile
FROM kairin/001:nvcr.io-nvidia-pytorch-25.02-py3-igpu

# Install any dependencies or files needed
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    libxkbcommon-x11-0 \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

# Copy the necessary files into the container
COPY qt-online-installer-linux-arm64-4.9.0.run /media/kkk/buildx/apps/

# Set the working directory
WORKDIR /media/kkk/buildx/apps

# Make the installer executable
RUN chmod +x qt-online-installer-linux-arm64-4.9.0.run

# Run the Qt installer with GUI
CMD ["./qt-online-installer-linux-arm64-4.9.0.run"]
