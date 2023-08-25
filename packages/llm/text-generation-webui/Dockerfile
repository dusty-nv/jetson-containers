#---
# name: text-generation-webui
# group: llm
# depends: [pytorch, bitsandbytes, transformers, auto_gptq, gptq-for-llama, exllama, llama_cpp]
# requires: '>=34.1.0'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

# force rebuild on new commits
ADD https://api.github.com/repos/oobabooga/text-generation-webui/git/refs/heads/main /tmp/oobabooga_version.json

RUN git clone --depth=1 https://github.com/oobabooga/text-generation-webui

RUN cd text-generation-webui && \
    sed 's|^numpy.*|numpy|g' -i requirements.txt && \
    sed 's|^accelerate.*|accelerate|g' -i requirements.txt && \
    sed 's|^bitsandbytes.*|bitsandbytes|g' -i requirements.txt && \
    sed 's|^fastapi.*|fastapi|g' -i requirements.txt && \
    sed 's|^safetensors.*|safetensors|g' -i requirements.txt && \
    sed 's|^llama-cpp-python.*|llama-cpp-python|g' -i requirements.txt && \
    sed 's|^git+https://github.com/huggingface/transformers.*|transformers|g' -i requirements.txt && \
    sed 's|^git+https://github.com/huggingface/peft|#git+https://github.com/huggingface/peft|g' -i requirements.txt && \
    sed 's|^https://github.com/PanQiWei/AutoGPTQ|#https://github.com/PanQiWei/AutoGPTQ|g' -i requirements.txt && \
    sed 's|^https://github.com/jllllll/ctransformers-cuBLAS-wheels.*|#https://github.com/jllllll/ctransformers-cuBLAS-wheels|g' -i requirements.txt && \
    cat requirements.txt
    
RUN cd text-generation-webui && \
    pip3 freeze > /tmp/constraints.txt && \
    pip3 install --no-cache-dir --verbose -r requirements.txt --constraint /tmp/constraints.txt && \
    rm /tmp/constraints.txt && \
    sed 's|@functools.cache|@functools.lru_cache\(maxsize=None\)|' -i modules/chat.py && \
    sed 's|@functools.cache|@functools.lru_cache\(maxsize=None\)|' -i modules/loaders.py && \
    sed 's|@functools.cache|@functools.lru_cache\(maxsize=None\)|' -i modules/presets.py 

RUN cp /opt/GPTQ-for-LLaMa/*.py /opt/text-generation-webui

# https://github.com/oobabooga/text-generation-webui/issues/3042#issuecomment-1626160643
RUN pip3 install --no-cache-dir --verbose 'gradio>=3.36.1'
RUN pip3 install --no-cache-dir --verbose -r text-generation-webui/extensions/api/requirements.txt

WORKDIR /

CMD /bin/bash -c "cd /opt/text-generation-webui && python3 server.py --model-dir=/data/models/text-generation-webui --listen --verbose"
