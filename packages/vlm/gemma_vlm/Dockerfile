# Dockerfile for Gemma VLM (Gemma3) model
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG PYTORCH_IMAGE
ARG TENSORRT_IMAGE
ARG OPENCV_IMAGE

# Argument to specify the Hugging Face model ID for Gemma VLM
ARG GEMMA_VLM_MODEL_ID="google/gemma-3-4b-it"
ENV GEMMA_VLM_MODEL_ID=${GEMMA_VLM_MODEL_ID}

# Argument for Hugging Face Token
ARG HUGGINGFACE_TOKEN
ENV HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}

# Set the working directory
WORKDIR /opt/gemma_vlm

# Install dependencies
# Ensure transformers is recent enough for Gemma3
# Pillow for image loading, accelerate for model loading, requests for test script
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir --upgrade transformers pillow accelerate requests einops bitsandbytes huggingface_hub triton==3.1.0 

# Pre-download the model to the container image to save time on first run.
COPY download_model.py ./
RUN python3 download_model.py

# Copy test script into the container
COPY test_gemma_vlm.py ./

# Set an entrypoint or command (optional)
# For now, we assume the user will run the test script or their own script.
# CMD ["python3", "test_gemma_vlm.py"]

# Add a healthcheck (optional)
# HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
#  CMD python3 -c \"from transformers import Gemma3ForConditionalGeneration; Gemma3ForConditionalGeneration.from_pretrained('$GEMMA_VLM_MODEL_ID', token=os.environ.get('HUGGINGFACE_TOKEN') if (os.environ.get('HUGGINGFACE_TOKEN') and os.environ.get('HUGGINGFACE_TOKEN') != 'None' and os.environ.get('HUGGINGFACE_TOKEN') != '') else None); exit 0\" || exit 1

# Add labels for metadata (optional)
LABEL maintainer="GitHub Copilot"
LABEL description="Container for Gemma VLM (Gemma3) model ${GEMMA_VLM_MODEL_ID}"
