# Use the appropriate base image
FROM kairin/001:nvcr.io-nvidia-pytorch-25.02-py3-igpu

# IMPORTANT: This base image should not be changed without explicit instruction and a clear reason provided to the user.

# Set the working directory
WORKDIR /opt

# Install required dependencies and update locales
RUN set -ex && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    locales \
    locales-all \
    tzdata && \
    locale-gen en_US en_US.UTF-8

# Clone the ComfyUI repository and install requirements
RUN git clone https://github.com/comfyanonymous/ComfyUI.git && \
    cd ComfyUI && \
    pip3 install -r requirements.txt

# Clone custom nodes repositories
RUN cd /opt/ComfyUI/custom_nodes && \
    git clone --recursive https://github.com/ltdrdata/ComfyUI-Manager.git && \
    git clone --recursive https://github.com/discus0434/comfyui-flux-accelerator.git && \
    git clone --recursive https://github.com/your/repo/ComfyUI-to-Python-Extension.git

# Update pip and install requirements for custom nodes
RUN python3 -m venv venv && \
    source venv/bin/activate && \
    pip3 install --upgrade pip && \
    pip3 install -U -r /opt/ComfyUI/custom_nodes/ComfyUI-Manager/requirements.txt && \
    pip3 install -U -r /opt/ComfyUI/custom_nodes/comfyui-flux-accelerator/requirements.txt && \
    pip3 install -U -r /opt/ComfyUI/custom_nodes/ComfyUI-to-Python-Extension/requirements.txt

# Uninstall deprecated egg packages and reinstall using pip (excluding nvfuser)
RUN pip uninstall lightning_thunder looseversion dill sympy lightning_utilities opt_einsum -y && \
    pip install lightning_thunder looseversion dill sympy lightning_utilities opt_einsum

# Copy the qt-online-installer file to the image
COPY qt-online-installer-linux-arm64-4.9.0.run /opt/qt-online-installer-linux-arm64-4.9.0.run

# Make the installer executable
RUN chmod +x /opt/qt-online-installer-linux-arm64-4.9.0.run

# Copy the tarpack script to the bin directory
COPY tarpack /usr/local/bin/

# Set command to run the application
CMD ["python3", "-m", "jetson_containers.build"]
