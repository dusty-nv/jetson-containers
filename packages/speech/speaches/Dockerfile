#---
# name: faster-whisper:speaches
# group: audio
# depends: [ctranslate2:master-builder,faster-whisper, python:3.12, numpy]
# requires: '>=34.1.0'
# test: [test.py]
# docs: docs.md
# notes: run with cuda12.6
#---

    ARG BASE_IMAGE
    FROM ${BASE_IMAGE}
    
    ARG SPEACHES_ROOT="/opt/speaches"
    ARG WHEEL_DIR="/opt/wheels"

    WORKDIR /opt

    # Setup wheel directory
    RUN mkdir -p ${WHEEL_DIR}

    # Clean up to reduce image size
    RUN rm -rf /root/.cache/pip \
        && rm -rf /root/.cache/uv \
        && rm -rf /tmp/* \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*
    
    # Remove any existing .venv directories
    RUN find / -name ".venv" -type d -exec rm -rf {} +  2>/dev/null || true
    
    # Build faster-whisper wheel
    RUN git clone https://github.com/guillaumekln/faster-whisper && \
        cd faster-whisper && \
        sed -i \
            -e 's|^onnxruntime.*||g' \
            -e 's|^ctranslate2.*||g' \
            -e 's|^huggingface_hub.*||g' \
            -e 's|^tokenizers.*|tokenizers|g' \
            requirements.txt && \
        pip3 install --no-cache-dir --verbose -r requirements.txt && \
        python3 setup.py bdist_wheel && \
        cp dist/faster_whisper*.whl ${WHEEL_DIR}/


    # Install required dependencies
    RUN apt-get update && \
        apt-get install -y libsndfile1-dev && \
        apt-get clean && rm -rf /var/lib/apt/lists/*
    
    # Install astral-uv
    RUN curl -LsSf https://astral.sh/uv/install.sh | sh
    
    ENV PATH="/root/.local/bin:$PATH"
    
    # Print locations of globally installed packages (for debugging)
    RUN python3 -c "import sys, numpy, ctranslate2, faster_whisper; \
        print(f'Python path: {sys.path}'); \
        print(f'NumPy: {numpy.__file__}'); \
        print(f'ctranslate2: {ctranslate2.__file__}'); \
        print(f'faster_whisper: {faster_whisper.__file__}')" || \
        echo "Warning: Not all packages could be imported."
    
    # Clone speaches repository
    WORKDIR /opt
    RUN git clone https://github.com/speaches-ai/speaches
    
    WORKDIR /opt/speaches
    
    # Copy startup script and make it executable
    COPY start-speaches-server.sh ${SPEACHES_ROOT}/start-speaches-server.sh
    RUN chmod +x ${SPEACHES_ROOT}/start-speaches-server.sh
    
    # Expose port
    EXPOSE 8000
    
    ENTRYPOINT ["/opt/speaches/start-speaches-server.sh"]