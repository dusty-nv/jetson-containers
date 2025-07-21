#---
# name: n8n
# group: rag
# depends: [cuda, cudnn, cmake, python, nodejs]
# config: config.py
# test: test.sh
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG N8N_VERSION

RUN apt-get update -y && \
    npm install -g npm

RUN useradd -m -s /bin/bash node

# Create non-root user
RUN useradd -ms /bin/bash node
USER node
WORKDIR /home/node

# Install latest n8n globally
RUN npm install -g n8n@${N8N_VERSION}

# Default port
EXPOSE 5678

CMD ["n8n"]
