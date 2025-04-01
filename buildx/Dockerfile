# Use the specified base image with the correct platform
FROM --platform=linux/arm64 kairin/001:nvcr.io-nvidia-pytorch-25.02-py3-igpu

# Install any dependencies or files needed
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    libxkbcommon-x11-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the necessary files into the container
COPY qt-online-installer-linux-arm64-4.9.0.run /media/kkk/buildx/apps/
COPY qt-installer-noninteractive.qs /media/kkk/buildx/apps/

# Set the working directory
WORKDIR /media/kkk/buildx/apps

# Make the installer executable
RUN chmod +x qt-online-installer-linux-arm64-4.9.0.run

# Run the Qt installer
RUN ./qt-online-installer-linux-arm64-4.9.0.run --platform minimal --script qt-installer-noninteractive.qs

# Execute any additional commands
RUN echo "Installation complete"
