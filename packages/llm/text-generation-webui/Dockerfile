#---
# name: text-generation-webui
# group: llm
# depends: [pytorch, bitsandbytes, transformers, auto-gptq, gptq-for-llama]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN cd opt && \
    git clone --depth=1 https://github.com/oobabooga/text-generation-webui && \
    cd text-generation-webui && \
    sed 's|^accelerate==.*|accelerate|g' -i requirements.txt && \
    sed 's|^bitsandbytes==.*|bitsandbytes|g' -i requirements.txt && \
    sed 's|^git+https://github.com/huggingface/peft|#git+https://github.com/huggingface/peft|g' -i requirements.txt && \
    sed 's|^git+https://github.com/huggingface/peft|#git+https://github.com/huggingface/peft|g' -i requirements.txt && \
    #sed 's|^https://github.com/PanQiWei/AutoGPTQ|#https://github.com/PanQiWei/AutoGPTQ|g' -i requirements.txt && \
    #sed 's|^llama-cpp-python|#llama-cpp-python|g' -i requirements.txt && \
    cat requirements.txt && \
    pip3 freeze > /tmp/constraints.txt && \
    pip3 install --no-cache-dir --verbose -r requirements.txt --constraint /tmp/constraints.txt && \
    rm /tmp/constraints.txt && \
    sed 's|@functools.cache|@functools.lru_cache\(maxsize=None\)|' -i modules/chat.py && \
    sed 's|@functools.cache|@functools.lru_cache\(maxsize=None\)|' -i modules/loaders.py && \
    sed 's|@functools.cache|@functools.lru_cache\(maxsize=None\)|' -i modules/presets.py 

RUN cp /opt/GPTQ-for-LLaMa/*.py /opt/text-generation-webui
