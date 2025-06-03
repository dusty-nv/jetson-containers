#---
# name: lobechat
# group: llm
# depends: [nodejs, pytorch, torchvision, torchaudio, opencv, bitsandbytes, transformers, xformers, torchao, flash-attention, sage-attention, sparge-attention, flexprefill, paraattention, diffusers, huggingface_hub]
# requires: '>=35.0.0'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

RUN npm install -g pnpm
# Clone the repository:
RUN git clone --recursive https://github.com/lobehub/lobe-chat /opt/lobe-chat && \
    cd /opt/lobe-chat && \
    pnpm install \
    pnpm dev

ENV NODE_ENV="production" \
    NODE_OPTIONS="--dns-result-order=ipv4first --use-openssl-ca" \
    NODE_EXTRA_CA_CERTS="" \
    NODE_TLS_REJECT_UNAUTHORIZED="" \
    SSL_CERT_DIR="/etc/ssl/certs/ca-certificates.crt"

ENV MIDDLEWARE_REWRITE_THROUGH_LOCAL="1"
ENV HOSTNAME="0.0.0.0" \
    PORT="3210"

ENV ACCESS_CODE="" \
    API_KEY_SELECT_MODE="" \
    DEFAULT_AGENT_CONFIG="" \
    SYSTEM_AGENT="" \
    FEATURE_FLAGS="" \
    PROXY_URL=""

EXPOSE 3210/tcp
ENTRYPOINT ["/bin/node"]
CMD ["/app/startServer.js"]
