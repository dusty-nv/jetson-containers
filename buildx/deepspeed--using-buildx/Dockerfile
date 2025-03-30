#---
# name: deepspeed
# group: llm
# config: config.py
# depends: [transformers, deepspeed-kernels, triton]
# requires: '>=35'
# test: test.sh
#---
ARG BASE_IMAGE=kairin/001:2025-03-25-0404
FROM ${BASE_IMAGE}

    
# Ensure apt-transport-https is installed
RUN apt-get update && apt-get install -y apt-transport-https

# Update package lists and install necessary packages
RUN apt-get update && apt-get install -y unzip wget curl jq --no-install-recommends

# Set the DEEPSPEED_BRANCH environment variable
ARG DEEPSPEED_BRANCH=main  # Specify the branch you want to use
ENV DEEPSPEED_BRANCH=${DEEPSPEED_BRANCH}

# Copy and run the install script
COPY install.sh /tmp/deepspeed/install.sh
RUN chmod +x /tmp/deepspeed/install.sh
RUN /tmp/deepspeed/install.sh

# Verify installation
RUN pip3 show transformers \
    && python3 -c 'import transformers; print(transformers.__version__)'

