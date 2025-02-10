#---
# name: kokoro-tts
# group: audio
# depends: [onnxruntime, numba]
# test: [test.py]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Install required dependencies
RUN apt-get update &&  \
    apt-get install -y libespeak-ng-dev portaudio19-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /opt

# Clone the repository and modify pyproject.toml in one step
RUN git clone https://github.com/thewh1teagle/kokoro-onnx.git && \
    cd kokoro-onnx && \
    sed -i "/onnxruntime/d" pyproject.toml && \
    sed -i "s/numpy>=2.0.2/numpy/" pyproject.toml && \
    sed -i 's/providers = \["CPUExecutionProvider"\]/providers = \["CUDAExecutionProvider", "CPUExecutionProvider"\]/' src/kokoro_onnx/__init__.py && \
    pip install -e . 

# Install Python dependencies in one step
RUN pip install gradio soundfile sounddevice

# Download model and voices files
RUN wget -P /opt/kokoro-onnx/examples \
    https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx \
    https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin

# Reinstall correct numpy
RUN bash /tmp/numpy/install.sh

# This dockerfile may get used in others, so restore $WORKDIR
WORKDIR /

# Default command
CMD ["bash"]
