#---
# name: transformers
# config: config.py
# group: llm
# depends: [pytorch, torchvision, huggingface_hub, rust]
# test: [test_version.py, huggingface-benchmark.py]
# docs: docs.md
# notes: for quantization support in Transformers, use the bitsandbytes, AutoGPTQ, or AutoAWQ containers.
#---
ARG BASE_IMAGE=kairin/001:2025-03-25-0404
FROM ${BASE_IMAGE}

# Add the missing GPG key for the Kitware repository
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 16FAAD7AF99A65E2 \
    && echo "deb https://apt.kitware.com/ubuntu/ noble main" > /etc/apt/sources.list.d/kitware.list

# Ensure apt-transport-https is installed
RUN apt-get update && apt-get install -y apt-transport-https

# Update package lists and install necessary packages
RUN apt-get update && apt-get install -y unzip wget curl jq --no-install-recommends

# Set the TRANSFORMERS_PACKAGE environment variable
ARG TRANSFORMERS_PACKAGE=transformers
ENV TRANSFORMERS_PACKAGE=${TRANSFORMERS_PACKAGE}

# Copy and run the install script
COPY install.sh /tmp/transformers/install.sh
RUN chmod +x /tmp/transformers/install.sh
RUN /tmp/transformers/install.sh

# Copy the benchmark script
COPY huggingface-benchmark.py /usr/local/bin

# Any additional steps


COPY install.sh /tmp/transformers/install.sh
RUN /tmp/transformers/install.sh
    
COPY huggingface-benchmark.py /usr/local/bin
    
RUN pip3 show transformers \
    && python3 -c 'import transformers; print(transformers.__version__)'
