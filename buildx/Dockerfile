# Use the specified base image
FROM kairin/001:20250402-063709-1 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages (if any additional packages are needed)
RUN apt-get update && apt-get install -y --allow-downgrades --allow-remove-essential --allow-change-held-packages \
    # List all additional necessary packages here
    ...

# Stage for ninja
FROM base as ninja
COPY buildx/l4t/ninja /opt/ninja
WORKDIR /opt/ninja
RUN docker build -t ninja-image .

# Stage for small-stable-diffusion
FROM base as small-stable-diffusion
COPY buildx/l4t/small-stable-diffusion /opt/small-stable-diffusion
WORKDIR /opt/small-stable-diffusion
RUN docker build -t small-stable-diffusion-image .

# Stage for stable-diffusion-webui
FROM base as stable-diffusion-webui
COPY buildx/l4t/stable-diffusion-webui /opt/stable-diffusion-webui
WORKDIR /opt/stable-diffusion-webui
RUN docker build -t stable-diffusion-webui-image .

# Stage for stable-diffusion
FROM base as stable-diffusion
COPY buildx/l4t/stable-diffusion /opt/stable-diffusion
WORKDIR /opt/stable-diffusion
RUN docker build -t stable-diffusion-image .

# Repeat similar stages for other directories
FROM base as build-essential
COPY buildx/l4t/build-essential /opt/build-essential
WORKDIR /opt/build-essential
RUN docker build -t build-essential-image .

FROM base as comfyui
COPY buildx/l4t/comfyui /opt/comfyui
WORKDIR /opt/comfyui
RUN docker build -t comfyui-image .

FROM base as crossformer
COPY buildx/l4t/crossformer /opt/crossformer
WORKDIR /opt/crossformer
RUN docker build -t crossformer-image .

FROM base as cuda-python
COPY buildx/l4t/cuda-python /opt/cuda-python
WORKDIR /opt/cuda-python
RUN docker build -t cuda-python-image .

FROM base as cuda
COPY buildx/l4t/cuda /opt/cuda
WORKDIR /opt/cuda
RUN docker build -t cuda-image .

FROM base as cudnn
COPY buildx/l4t/cudnn /opt/cudnn
WORKDIR /opt/cudnn
RUN docker build -t cudnn-image .

FROM base as cupy
COPY buildx/l4t/cupy /opt/cupy
WORKDIR /opt/cupy
RUN docker build -t cupy-image .

FROM base as diffusers
COPY buildx/l4t/diffusers /opt/diffusers
WORKDIR /opt/diffusers
RUN docker build -t diffusers-image .

FROM base as docker
COPY buildx/l4t/docker /opt/docker
WORKDIR /opt/docker
RUN docker build -t docker-image .

FROM base as ffmpeg
COPY buildx/l4t/ffmpeg /opt/ffmpeg
WORKDIR /opt/ffmpeg
RUN docker build -t ffmpeg-image .

FROM base as gstreamer
COPY buildx/l4t/gstreamer /opt/gstreamer
WORKDIR /opt/gstreamer
RUN docker build -t gstreamer-image .

FROM base as l4t-diffusion
COPY buildx/l4t/l4t-diffusion /opt/l4t-diffusion
WORKDIR /opt/l4t-diffusion
RUN docker build -t l4t-diffusion-image .

# Final stage
FROM base

# Copy built artifacts from previous stages if necessary
COPY --from=ninja /path/to/artifacts /final/path
COPY --from=small-stable-diffusion /path/to/artifacts /final/path
COPY --from=stable-diffusion-webui /path/to/artifacts /final/path
COPY --from=stable-diffusion /path/to/artifacts /final/path
COPY --from=build-essential /path/to/artifacts /final/path
COPY --from=comfyui /path/to/artifacts /final/path
COPY --from=crossformer /path/to/artifacts /final/path
COPY --from=cuda-python /path/to/artifacts /final/path
COPY --from=cuda /path/to/artifacts /final/path
COPY --from=cudnn /path/to/artifacts /final/path
COPY --from=cupy /path/to/artifacts /final/path
COPY --from=diffusers /path/to/artifacts /final/path
COPY --from=docker /path/to/artifacts /final/path
COPY --from=ffmpeg /path/to/artifacts /final/path
COPY --from=gstreamer /path/to/artifacts /final/path
COPY --from=l4t-diffusion /path/to/artifacts /final/path

# Define the entrypoint or command if needed
CMD ["bash"]
