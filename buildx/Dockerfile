# Use the specified base image
ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Copy the build.sh script to the final image and make it executable
COPY build.sh /usr/local/bin/build.sh
RUN chmod +x /usr/local/bin/build.sh

# Default start-up command
CMD ["/bin/bash"]
