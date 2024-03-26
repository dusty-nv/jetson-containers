#---
# name: text-generation-webui
# group: llm
# config: config.py
# depends: [transformers, auto_gptq, gptq-for-llama, exllama:v1, exllama:v2, llama_cpp:gguf]
# requires: '>=34.1.0'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG OOBABOOGA_REF \
    OOBABOOGA_SHA \
    OOBABOOGA_ROOT_DIR="/opt/text-generation-webui" \
    OOBABOOGA_MODEL_DIR="/data/models/text-generation-webui" \
    FLASH_ATTENTION_BRANCH="v2.5.6" \
    LD_PRELOAD_LIBS=""

# Check GitHub API for version information
ADD https://api.github.com/repos/oobabooga/text-generation-webui/git/${OOBABOOGA_REF} /tmp/oobabooga_version.json

# Copy all patch files
COPY *.diff /tmp/

# Install text-generation-webui
RUN set -ex \
    && git clone https://github.com/oobabooga/text-generation-webui "$OOBABOOGA_ROOT_DIR" \
    && git -C "$OOBABOOGA_ROOT_DIR" checkout "$OOBABOOGA_SHA" \
    \
    # Fix text-generation-webui requirements \
    && sed -i \
        -e 's|^bitsandbytes.*|#bitsandbytes|g' \
        -e 's|^llama-cpp-python.*|llama-cpp-python|g' \
        -e 's|^exllamav2.*|exllamav2|g' \
        -e 's|^autoawq.*||g' \
        -e 's|^transformers.*|transformers|g' \
        -e 's|^https://github.com/jllllll/ctransformers-cuBLAS-wheels.*|#https://github.com/jllllll/ctransformers-cuBLAS-wheels|g' \
        "$OOBABOOGA_ROOT_DIR/requirements.txt" \
    \
    # Fix https://github.com/oobabooga/text-generation-webui/issues/4644 \
    && sed 's|to(self\.projector_device)|to(self\.projector_device,dtype=self\.projector_dtype)|' -i "$OOBABOOGA_ROOT_DIR/extensions/multimodal/pipelines/llava/llava.py" \
    # Fix: cannot uninstall 'blinker': It is a distutils installed project \
    && pip3 install --no-cache-dir --ignore-installed blinker \
    # Add text-generation-webui required dependencies \
    && echo "git+https://github.com/oobabooga/torch-grammar.git@main" >> "$OOBABOOGA_ROOT_DIR/requirements.txt" \
    && echo "git+https://github.com/UKPLab/sentence-transformers.git@master" >> "$OOBABOOGA_ROOT_DIR/requirements.txt" \
    \
    # Create a symbolic link from /opt/GPTQ-for-LLaMa/*.py to oobabooga root dir \
    # TODO: Grab the /opt/GPTQ-for-LLaMa/*.py files from `builder` image \
    && ln -s /opt/GPTQ-for-LLaMa/*.py "$OOBABOOGA_ROOT_DIR" \
    \
    # Install text-generation-webui \
    && pip3 install --no-cache-dir --verbose -r "$OOBABOOGA_ROOT_DIR/requirements.txt" \
    # Install text-generation-webui extensions \
    && cd "$OOBABOOGA_ROOT_DIR" \
    && python -c "from one_click import install_extensions_requirements; install_extensions_requirements()" \
    \
    # Install flash-attention \
    && git clone --depth=1 --branch="$FLASH_ATTENTION_BRANCH" https://github.com/Dao-AILab/flash-attention /opt/flash-attention \
    && git -C /opt/flash-attention apply /tmp/flash-attn.diff \
    && export FLASH_ATTENTION_FORCE_BUILD=1 \
        FLASH_ATTENTION_FORCE_CXX11_ABI=0 \
        FLASH_ATTENTION_SKIP_CUDA_BUILD=0 \
    && cd /opt/flash-attention/ \
    && python3 setup.py install \
    \
    # Cleanup \
    && rm -rf \
        /opt/flash-attention \
        /var/lib/apt/lists/* \
        "$OOBABOOGA_ROOT_DIR/api-examples" \
        "$OOBABOOGA_ROOT_DIR/docker" \
        "$OOBABOOGA_ROOT_DIR/docs" \
    && apt-get clean

COPY settings.yaml ${OOBABOOGA_ROOT_DIR}

ENV LD_PRELOAD="${LD_PRELOAD}:${LD_PRELOAD_LIBS}"

CMD ["/bin/bash", "-c", "cd $OOBABOOGA_ROOT_DIR && python3", "server.py", "--model-dir=$OOBABOOGA_MODEL_DIR", "--listen", "--verbose", "--triton"]
