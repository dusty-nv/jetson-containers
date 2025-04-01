# Use the specified base image
FROM kairin/001:20250402-063709-1

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages (if any additional packages are needed)
RUN apt-get update && apt-get install -y \
    # List all additional necessary packages here
    ...

# Copy the files and directories from the build context into the image
COPY bazel /opt/bazel
COPY cmake /opt/cmake
COPY cosmos /opt/cosmos
COPY h5py /opt/h5py
COPY holoscan /opt/holoscan
COPY l4t /opt/l4t
COPY libcom /opt/libcom
COPY nodejs /opt/nodejs
COPY numba /opt/numba
COPY numpy /opt/numpy
COPY opencv /opt/opencv
COPY protobuf /opt/protobuf
COPY pyav /opt/pyav
COPY pycuda /opt/pycuda
COPY python /opt/python
COPY rust /opt/rust

# Optionally, run installation scripts if they exist
RUN if [ -f /opt/bazel/install.sh ]; then chmod +x /opt/bazel/install.sh && /opt/bazel/install.sh; fi
RUN if [ -f /opt/cmake/install.sh ]; then chmod +x /opt/cmake/install.sh && /opt/cmake/install.sh; fi
RUN if [ -f /opt/cosmos/install.sh ]; then chmod +x /opt/cosmos/install.sh && /opt/cosmos/install.sh; fi
# Repeat for other directories...

# Define the entrypoint or command if needed
CMD ["bash"]
