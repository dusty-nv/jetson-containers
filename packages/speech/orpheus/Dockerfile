#---
# name: orpheus
# group: audio
# depends: [torchaudio, numpy, tensorrt, onnxruntime, ffmpeg]
# requires: '>=34.1.0'
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Set non-interactive frontend
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    portaudio19-dev \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create app directory
RUN mkdir -p /app/outputs && \
    chmod 777 /app/outputs

# Clone and install Orpheus-FastAPI
RUN git clone https://github.com/Lex-au/Orpheus-FastAPI /opt/orpheus && \
    cd /opt/orpheus && \
    # Update requirements to use specific versions that work with Jetson
    sed -i 's/numpy==1.24.0/numpy>=1.24.0/g' requirements.txt && \
    echo "👁️######## Orpheus-FastAPI - requirements.txt ########👁️" && \
    cat requirements.txt && \
    echo "👁️^^^^^^^^ Orpheus-FastAPI - requirements.txt ^^^^^^^^👁️" && \
    pip3 install -r requirements.txt

# Set up environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/orpheus \
    USE_GPU=true

# Verify installations
RUN python3 -c "import torch; print('PyTorch version:', torch.__version__); \
    print('CUDA available:', torch.cuda.is_available()); \
    print('CUDA device count:', torch.cuda.device_count() if torch.cuda.is_available() else 'N/A')"

RUN python3 -c "from snac import SNAC; print('SNAC import successful')"


# Set default port
ARG PORT=5005
ENV PORT=${PORT}
EXPOSE ${PORT}

# Set working directory
WORKDIR /opt/orpheus

# Set the entrypoint
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5005", "--workers", "1"]
